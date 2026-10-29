"""Encoder–decoder 1D configurable para A0–A5, A5a y A5b.

Todos los anchos vienen explícitos de configs/experiment.yaml; no hay proyecciones ocultas ni recortes:
si una concatenación no coincide en longitud o canales, se lanza un error.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import nn

from .config import validate_architecture


class ChannelLayerNorm(nn.Module):
    """LayerNorm sobre los canales de cada instante: (B, C, T) → (B, T, C) → LN → (B, C, T)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "layer":
        return ChannelLayerNorm(channels)
    raise ValueError(f"normalización desconocida: {kind}")


class ConvNormAct(nn.Module):
    """Conv1d sin sesgo (longitud conservada) → normalización → GELU → Dropout."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, norm: str, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, bias=False)
        self.norm = make_norm(norm, c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


def _trace(trace: list | None, stage: str, op: str, x: torch.Tensor) -> None:
    if trace is not None:
        trace.append((stage, op, int(x.shape[1]), int(x.shape[-1])))


@lru_cache(maxsize=64)
def _linear_weights(length: int, size: int, dtype: torch.dtype, device: torch.device):
    # Índice fuente de upsample_linear1d de PyTorch (area_pixel_compute_source_index, align_corners=False).
    scale = torch.tensor(length, dtype=dtype) / size
    position = ((torch.arange(size, dtype=dtype) + 0.5) * scale - 0.5).clamp_min(0)
    left = position.floor().long()
    right = (left + 1).clamp_max(length - 1)
    weight = position - left
    return left.to(device), right.to(device), weight.to(device)


def linear_upsample(x: torch.Tensor, size: int) -> torch.Tensor:
    """``F.interpolate(x, size, mode="linear", align_corners=False)`` sobre el último eje.

    Misma fórmula, escrita como selección de índices y combinación ponderada. El kernel CUDA de
    PyTorch paraleliza solo sobre las posiciones de salida y recorre batch × canales en serie (~96 %
    del paso en el experimento anterior), y su retropropagación usa sumas atómicas no deterministas.
    """
    compute = torch.promote_types(x.dtype, torch.float32)
    left, right, weight = _linear_weights(x.shape[-1], size, compute, x.device)
    out = x[..., left].to(compute) * (1 - weight) + x[..., right].to(compute) * weight
    return out.to(x.dtype)


class DecoderLevel(nn.Module):
    """Un nivel del decoder: upsampling + Conv1d + concatenación, en el orden indicado."""

    def __init__(self, name: str, order: str, in_ch: int, skip_ch: int, out_ch: int, kernel: int,
                 norm: str, dropout: float, upsample: str):
        super().__init__()
        self.name, self.order = name, order
        # ConvTranspose1d depthwise: duplica la longitud con un filtro aprendible por canal.
        self.up = (
            nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, groups=in_ch, bias=False)
            if upsample == "convtranspose" else None
        )
        conv_in = in_ch if order == "upsample_conv_concat" else in_ch + skip_ch
        self.block = ConvNormAct(conv_in, out_ch, kernel, 1, norm, dropout)
        self.out_channels = out_ch + skip_ch if order == "upsample_conv_concat" else out_ch
        self.dims = {"in": in_ch, "skip": skip_ch, "conv_in": conv_in, "conv_out": out_ch, "out": self.out_channels}

    def _upsample(self, x: torch.Tensor, length: int) -> torch.Tensor:
        if self.up is None:
            return linear_upsample(x, length)
        return self.up(x)

    def _concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != skip.shape[-1]:
            raise ValueError(f"{self.name}: longitud {x.shape[-1]} ≠ skip {skip.shape[-1]}")
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, trace: list | None = None) -> torch.Tensor:
        x = self._upsample(x, skip.shape[-1])
        _trace(trace, self.name, "upsample", x)
        if self.order == "upsample_conv_concat":
            x = self.block(x)
            _trace(trace, self.name, "conv", x)
            x = self._concat(x, skip)
            _trace(trace, self.name, "concat", x)
        else:
            x = self._concat(x, skip)
            _trace(trace, self.name, "concat", x)
            x = self.block(x)
            _trace(trace, self.name, "conv", x)
        return x


class EncoderDecoder1D(nn.Module):
    """Entrada (B, C, 40, T) o (B, C·40, T); salida (B, 5, T) sin activación final."""

    def __init__(self, n_inputs: int, spec: dict[str, Any], n_outputs: int = 5, window: int = 256):
        super().__init__()
        validate_architecture("modelo", spec, window)
        self.spec, self.n_inputs, self.window = dict(spec), n_inputs, window
        norm, p = spec["normalization"], float(spec["dropout"])
        enc_ch = list(spec["encoder_channels"])

        self.stem = ConvNormAct(n_inputs, spec["stem_channels"], 3, 1, norm, p)
        c_in, blocks = spec["stem_channels"], []
        for c_out, k, d in zip(enc_ch, spec["kernel_sizes"], spec["dilations"]):
            blocks.append(ConvNormAct(c_in, c_out, k, d, norm, p))
            c_in = c_out
        self.encoder = nn.ModuleList(blocks)
        self.pool = nn.MaxPool1d(2)

        levels, in_ch = [], enc_ch[-1]
        for j, (skip_ch, out_ch, k) in enumerate(
            zip(reversed(enc_ch), spec["decoder_channels"], spec["decoder_kernel_sizes"])
        ):
            level = DecoderLevel(f"dec{j + 1}", spec["decoder_order"], in_ch, skip_ch, out_ch, k, norm, p, spec["upsample"])
            levels.append(level)
            in_ch = level.out_channels
        self.decoder = nn.ModuleList(levels)
        self.head = nn.Conv1d(in_ch, n_outputs, kernel_size=1)

        n = len(enc_ch)
        self.expected = {
            "skips": [{"channels": c, "length": window // 2 ** i} for i, c in enumerate(enc_ch)],
            "bottleneck": {"channels": enc_ch[-1], "length": window // 2 ** n},
            "decoder": [dict(level.dims, length=window // 2 ** (n - 1 - j)) for j, level in enumerate(levels)],
            "head_in": in_ch,
        }
        self.trace_enabled = False
        self.last_trace: list[tuple[str, str, int, int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1, 2)
        if x.shape[1] != self.n_inputs or x.shape[-1] != self.window:
            raise ValueError(f"entrada {tuple(x.shape)}; se esperaba (B, {self.n_inputs}, {self.window})")
        trace = [] if self.trace_enabled else None
        x = self.stem(x)
        _trace(trace, "stem", "conv", x)
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)  # skip antes del pooling
            _trace(trace, f"enc{i + 1}", "skip", x)
            x = self.pool(x)
            _trace(trace, f"enc{i + 1}", "pool", x)
        for j, level in enumerate(self.decoder):
            x = level(x, skips[-1 - j], trace)
        x = self.head(x)
        _trace(trace, "head", "conv", x)
        if trace is not None:
            self.last_trace = trace
        return x


def build_model(arch_name: str, cfg: dict[str, Any], n_inputs: int) -> EncoderDecoder1D:
    return EncoderDecoder1D(
        n_inputs, cfg["architectures"][arch_name], n_outputs=len(cfg["data"]["fingers"]),
        window=cfg["preprocessing"]["window"],
    )


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


@torch.no_grad()
def describe(model: EncoderDecoder1D) -> dict[str, Any]:
    """Parámetros entrenables por sección y dimensiones reales por etapa (forward de prueba)."""
    was_training, was_tracing = model.training, model.trace_enabled
    device = next(model.parameters()).device
    model.eval()
    model.trace_enabled = True
    model(torch.zeros(1, model.n_inputs, model.window, device=device))
    stages = [{"stage": s, "op": op, "channels": c, "length": n} for s, op, c, n in model.last_trace]
    model.trace_enabled = was_tracing
    model.train(was_training)
    return {
        "spec": model.spec,
        "n_inputs": model.n_inputs,
        "window": model.window,
        "parameters": {
            "stem": count_parameters(model.stem),
            "encoder": count_parameters(model.encoder),
            "decoder": count_parameters(model.decoder),
            "head": count_parameters(model.head),
            "total": count_parameters(model),
        },
        "expected": model.expected,
        "stages": stages,
    }
