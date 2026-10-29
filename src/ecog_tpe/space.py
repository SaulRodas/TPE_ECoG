"""Espacio de búsqueda conjunto (plan §3.1), warmups equilibrados (§6.1) y construcción de A5b/A5 por (k, L, f)."""

from __future__ import annotations

from typing import Any

from .config import SEARCH_PARAMS


def architecture_spec(cfg: dict[str, Any], architecture_id: str, kernel_size: int, encoder_levels: int,
                      base_filters: int) -> dict[str, Any]:
    """Spec de ``EncoderDecoder1D``: base A5b/A5 de la config con anchos, kernels y dilataciones del plan §3.

    Con k=5, L=5, f=64 reproduce exactamente A5b/A5 de las ablaciones. El decoder es el espejo del encoder.
    """
    f, k, levels = int(base_filters), int(kernel_size), int(encoder_levels)
    channels = [f, f, 2 * f, 4 * f, 8 * f, 8 * f][:levels + 1]  # stem + L bloques
    kernels = [7, 7] + [k] * (levels - 2)
    return {
        **cfg["architectures"][architecture_id],
        "stem_channels": channels[0],
        "encoder_channels": channels[1:],
        "kernel_sizes": kernels,
        "dilations": [1, 2, 3, 1, 2][:levels],
        "decoder_channels": channels[1:][::-1],
        "decoder_kernel_sizes": kernels[::-1],
    }


def params_spec(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return architecture_spec(cfg, params["architecture_id"], params["kernel_size"], params["encoder_levels"],
                             params["base_filters"])


def suggest(trial, space: dict[str, Any]) -> dict[str, Any]:
    """Los seis hiperparámetros, en el orden y con los dominios del plan §3.1."""
    params = {
        "architecture_id": trial.suggest_categorical("architecture_id", list(space["architecture_id"])),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        "kernel_size": trial.suggest_categorical("kernel_size", list(space["kernel_size"])),
        "encoder_levels": trial.suggest_categorical("encoder_levels", list(space["encoder_levels"])),
        "base_filters": trial.suggest_categorical("base_filters", list(space["base_filters"])),
    }
    assert tuple(params) == SEARCH_PARAMS
    return params


def warmup_trials(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Los diez warmups como (clave, parámetros), en orden par1-A5b, par1-A5, par2-A5b…; el par 1 es la referencia."""
    out = []
    for i, pair in enumerate(cfg["search"]["warmups"], start=1):
        for arch in cfg["search"]["space"]["architecture_id"]:
            params = {"architecture_id": arch, **pair}
            out.append((f"p{i}_{arch}", {k: params[k] for k in SEARCH_PARAMS}))
    return out


def label(params: dict[str, Any]) -> str:
    """Etiqueta corta, p. ej. ``A5 k5 L5 f64``."""
    return f"{params['architecture_id']} k{params['kernel_size']} L{params['encoder_levels']} f{params['base_filters']}"
