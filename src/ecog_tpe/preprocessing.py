"""Cadena común: operaciones P0–P5, FIR pasabanda + notch, potencia Morlet, antialias, escalado y ventanas.

Todo el filtrado se hace por bloque (ningún filtro cruza una frontera) en float64. Las convoluciones se
calculan por FFT en modo 'same' con relleno de ceros, igual que MNE; los bordes contaminados se descartan
con ``edge_columns``. Todas las estadísticas se ajustan solo con datos de entrenamiento del fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import mne
import numpy as np
import torch
from scipy import signal
from scipy.fft import next_fast_len
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .cleaning import apply_clip, fit_clip
from .data import fold_blocks, glove_to_feature_rate, split_blocks


def resolve_device(name: str | None = "auto") -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------- filtros


@dataclass
class FilterBank:
    bandpass: np.ndarray
    notch: np.ndarray
    antialias: np.ndarray
    wavelets: list[np.ndarray]
    freqs: np.ndarray
    decim: int
    edge_samples: int
    _cache: dict = field(default_factory=dict, repr=False)

    def fir_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = ("fir", str(device))
        if key not in self._cache:
            self._cache[key] = {
                name: torch.as_tensor(getattr(self, name), dtype=torch.float64, device=device)
                for name in ("bandpass", "notch", "antialias")
            }
        return self._cache[key]

    def wavelet_ffts(self, device: torch.device, n_fft: int) -> list[tuple[torch.Tensor, int]]:
        key = ("morlet", str(device), n_fft)
        if key not in self._cache:
            self._cache[key] = [
                (torch.fft.fft(torch.as_tensor(w, dtype=torch.complex128, device=device), n=n_fft), (w.size - 1) // 2)
                for w in self.wavelets
            ]
        return self._cache[key]


def build_filter_bank(cfg: dict[str, Any]) -> FilterBank:
    prep = cfg["preprocessing"]
    fs = cfg["data"]["fs_ecog"]
    decim = cfg["_derived"]["decim"]
    fir = dict(filter_length="auto", method="fir", phase="zero", fir_window="hamming", fir_design="firwin", verbose=False)

    bp = prep["bandpass"]
    bandpass = mne.filter.create_filter(
        None, fs, l_freq=bp["l_freq"], h_freq=bp["h_freq"],
        l_trans_bandwidth=bp["l_trans_bandwidth"], h_trans_bandwidth=bp["h_trans_bandwidth"], **fir,
    )

    # Band-stop con las convenciones de mne.filter.notch_filter: ancho f/200 y transición por lado.
    nt = prep["notch"]
    centers = np.asarray(nt["freqs"], dtype=float)
    tb = nt["trans_bandwidth_per_side"]
    half = centers / nt["width_divisor"] / 2 + tb
    notch = mne.filter.create_filter(
        None, fs, l_freq=centers + half, h_freq=centers - half, l_trans_bandwidth=tb, h_trans_bandwidth=tb, **fir,
    )

    aa = prep["antialias"]
    n_taps, beta = signal.kaiserord(aa["attenuation_db"], (aa["stop_hz"] - aa["pass_hz"]) / (fs / 2))
    n_taps += 1 - n_taps % 2  # tipo I (longitud impar) para fase cero exacta
    antialias = signal.firwin(n_taps, (aa["pass_hz"] + aa["stop_hz"]) / 2, window=("kaiser", beta), fs=fs)

    mo = prep["morlet"]
    freqs = np.logspace(np.log10(mo["f_min"]), np.log10(mo["f_max"]), mo["n_freqs"])
    wavelets = mne.time_frequency.morlet(fs, freqs, n_cycles=mo["n_cycles"], zero_mean=mo["zero_mean"])

    for name, kernel, expected in (
        ("pasabanda", bandpass, bp["n_taps"]), ("notch", notch, nt["n_taps"]), ("antialias", antialias, aa["n_taps"]),
    ):
        if kernel.size != expected:
            raise ValueError(f"FIR {name}: {kernel.size} coeficientes, se esperaban {expected}")
    edge_samples = sum((k.size - 1) // 2 for k in (notch, bandpass, antialias)) + max((w.size - 1) // 2 for w in wavelets)
    edge_columns = -(-edge_samples // decim)
    if edge_columns != prep["edge_columns"]:
        raise ValueError(f"el contexto de filtrado es {edge_columns} columnas, la config dice {prep['edge_columns']}")
    return FilterBank(bandpass, notch, antialias, list(wavelets), freqs, decim, edge_samples)


def fft_conv_same(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Convolución lineal centrada ('same') por FFT sobre el último eje; ``h`` real de longitud impar."""
    length, k = x.shape[-1], h.shape[-1]
    n = next_fast_len(length + k - 1, real=True)
    y = torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(h, n=n), n=n)
    start = (k - 1) // 2
    return y[..., start:start + length]


def _morlet_power(x: torch.Tensor, bank: FilterBank) -> Iterator[torch.Tensor]:
    """|x * w_f|² por frecuencia, (canales, muestras); misma convolución que tfr_array_morlet de MNE."""
    length = x.shape[-1]
    n = next_fast_len(length + max(w.size for w in bank.wavelets) - 1)
    xf = torch.fft.fft(x, n=n)
    for wf, start in bank.wavelet_ffts(x.device, n):
        y = torch.fft.ifft(xf * wf)[..., start:start + length]
        yield y.real.square() + y.imag.square()


def _bandpass_notch(x_block: np.ndarray, bank: FilterBank, device: torch.device) -> torch.Tensor:
    fir = bank.fir_tensors(device)
    x = torch.as_tensor(np.ascontiguousarray(x_block.T), dtype=torch.float64, device=device)
    return fft_conv_same(fft_conv_same(x, fir["bandpass"]), fir["notch"])


def morlet_power_full(x_block: np.ndarray, bank: FilterBank, device: torch.device) -> np.ndarray:
    """Potencia Morlet a 1000 Hz de una señal (muestras, canales) ya filtrada; (canales, frecuencias, muestras)."""
    x = torch.as_tensor(np.ascontiguousarray(x_block.T), dtype=torch.float64, device=device)
    return torch.stack(list(_morlet_power(x, bank)), dim=1).cpu().numpy()


def block_features(x_block: np.ndarray, bank: FilterBank, device: torch.device) -> np.ndarray:
    """Bloque (muestras, canales) → potencia Morlet filtrada y diezmada (canales, frecuencias, columnas) float32."""
    x = _bandpass_notch(x_block, bank, device)
    aa = bank.fir_tensors(device)["antialias"]
    out = torch.empty((x.shape[0], len(bank.wavelets), x.shape[1] // bank.decim), dtype=torch.float32, device=device)
    for i, power in enumerate(_morlet_power(x, bank)):
        out[:, i] = fft_conv_same(power, aa)[..., ::bank.decim].to(torch.float32)
    return out.cpu().numpy()


# ----------------------------------------------------------------------------- P0–P5


def _spatial_median(x: np.ndarray) -> np.ndarray:
    """CMR: mediana entre canales en cada instante."""
    return np.median(x, axis=1, keepdims=True)


def fit_input_transform(name: str, train_blocks: list[np.ndarray], std_floor: float) -> dict[str, np.ndarray]:
    """Estadísticos temporales por canal, ajustados solo con los bloques de entrenamiento (ECoG limpia)."""
    if name in ("P0", "P1"):
        return {}
    x = np.concatenate(train_blocks, axis=0)
    if name == "P2":
        return {"median": np.median(x, axis=0)}
    mean, std = x.mean(axis=0), x.std(axis=0)
    floored = std < std_floor
    params = {"mean": mean, "std": np.where(floored, std_floor, std), "std_floored": floored}
    if name in ("P4", "P5"):
        params["z_median"] = np.median((x - mean) / params["std"], axis=0)
    return params


def apply_input_transform(name: str, block: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    if name == "P0":
        return block - _spatial_median(block)
    if name == "P1":
        return block
    if name == "P2":
        return block - params["median"]
    z = (block - params["mean"]) / params["std"]
    if name == "P3":
        return z - _spatial_median(z)
    u = z - params["z_median"]
    if name == "P4":
        return u
    if name == "P5":
        return u - _spatial_median(u)
    raise ValueError(f"preprocesamiento desconocido: {name}")


# ----------------------------------------------------------------------------- fold


@dataclass
class FoldContext:
    """Señal limpia y guante de un sujeto/fold; común a todas las operaciones P."""

    subject: str
    fold: str
    val_block: int
    train_blocks: list[int]
    blocks: list[np.ndarray]          # ECoG limpia (muestras, canales) por bloque, 1000 Hz
    glove: np.ndarray                 # (bloques, dedos, columnas) en unidades originales
    clip: dict[str, np.ndarray]
    clip_low: np.ndarray              # (bloques, canales) muestras recortadas por debajo
    clip_high: np.ndarray
    raw_min: np.ndarray               # (bloques, canales) antes del clip
    raw_max: np.ndarray
    kept_channels: np.ndarray
    train_ranges: dict[int, tuple[int, int]]  # columnas X [lo, hi) usadas en train por bloque

    @property
    def n_channels(self) -> int:
        return self.blocks[0].shape[1]

    def clipped_fraction(self, which: str) -> float:
        rows = self.train_blocks if which == "train" else [self.val_block]
        n = sum(self.blocks[b].size for b in rows)
        return float((self.clip_low[rows].sum() + self.clip_high[rows].sum()) / n)


def train_column_ranges(val_block: int, train_blocks: list[int], cfg: dict[str, Any]) -> dict[int, tuple[int, int]]:
    """Región válida de cada bloque de train, con purga junto al bloque validado."""
    d, purge = cfg["_derived"], cfg["preprocessing"]["purge_columns"]
    ranges = {}
    for b in train_blocks:
        lo = d["valid_lo"] + (purge if b == val_block + 1 else 0)
        hi = d["valid_hi"] - (purge if b == val_block - 1 else 0)
        ranges[b] = (lo, hi)
    return ranges


def prepare_fold_context(
    ecog: np.ndarray, dg: np.ndarray, kept_channels: np.ndarray, subject: str, fold: str, cfg: dict[str, Any]
) -> FoldContext:
    """Divide en bloques, ajusta el clip robusto con train y lo aplica a todos los bloques."""
    n_blocks = cfg["_derived"]["n_blocks"]
    val, train = fold_blocks(fold, cfg)
    raw = split_blocks(ecog, n_blocks)
    cl = cfg["cleaning"]
    clip = fit_clip([raw[b] for b in train], cl["clip_k"], cl["mad_scale"], cl["mad_floor"])
    blocks, low, high = [], [], []
    for blk in raw:
        clean, n_low, n_high = apply_clip(blk, clip)
        blocks.append(clean)
        low.append(n_low)
        high.append(n_high)
    glove = np.stack([glove_to_feature_rate(g, cfg) for g in split_blocks(dg, n_blocks)])
    return FoldContext(
        subject=subject, fold=fold, val_block=val, train_blocks=train, blocks=blocks, glove=glove, clip=clip,
        clip_low=np.stack(low), clip_high=np.stack(high),
        raw_min=np.stack([b.min(axis=0) for b in raw]), raw_max=np.stack([b.max(axis=0) for b in raw]),
        kept_channels=np.asarray(kept_channels), train_ranges=train_column_ranges(val, train, cfg),
    )


def train_windows(train_ranges: dict[int, tuple[int, int]], window: int, stride: int) -> np.ndarray:
    """Inicios (bloque, columna) de las ventanas de train; no cruzan bloques ni regiones inválidas."""
    rows = []
    for b, (lo, hi) in sorted(train_ranges.items()):
        starts = np.arange(lo, hi - window + 1, stride)
        rows.append(np.stack([np.full_like(starts, b), starts], axis=1))
    return np.concatenate(rows).astype(np.int64)


def val_window_starts(lo: int, hi: int, window: int, stride: int) -> np.ndarray:
    """Inicios de validación con stride fijo; se añade la ventana final si hace falta."""
    starts = list(range(lo, hi - window + 1, stride))
    if starts[-1] != hi - window:
        starts.append(hi - window)
    return np.asarray(starts, dtype=np.int64)


@dataclass
class FoldFeatures:
    preprocessing: str
    X: np.ndarray                     # (bloques, canales·frecuencias, columnas) float32 escalado
    Y: np.ndarray                     # (bloques, dedos, columnas) float32 escalado MinMax
    y_min: np.ndarray                 # MinMaxScaler.min_
    y_scale: np.ndarray               # MinMaxScaler.scale_
    train_index: np.ndarray           # (N, 2): bloque, inicio
    val_block: int
    val_starts: np.ndarray
    valid_lo: int
    valid_hi: int
    delay: int
    window: int
    n_channels: int
    n_freqs: int
    params: dict[str, np.ndarray]
    stats: dict[str, Any]

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return (y_scaled - self.y_min[:, None]) / self.y_scale[:, None]

    def val_target_columns(self) -> slice:
        """Columnas del objetivo reconstruido en el bloque validado (incluye el retardo)."""
        return slice(self.valid_lo + self.delay, self.valid_hi + self.delay)


def compute_fold_features(
    ctx: FoldContext, name: str, bank: FilterBank, cfg: dict[str, Any], device: torch.device
) -> FoldFeatures:
    prep, d = cfg["preprocessing"], cfg["_derived"]
    delay, window = d["delay_columns"], prep["window"]
    p_params = fit_input_transform(name, [ctx.blocks[b] for b in ctx.train_blocks], prep["std_floor"])

    n_freqs = len(bank.wavelets)
    n_feat = ctx.n_channels * n_freqs
    X = np.empty((len(ctx.blocks), n_feat, d["block_columns"]), dtype=np.float32)
    for b, blk in enumerate(ctx.blocks):
        X[b] = block_features(apply_input_transform(name, blk, p_params), bank, device).reshape(n_feat, -1)

    # Escaladores ajustados con las columnas que cubren las ventanas de entrenamiento.
    x_train = np.concatenate([X[b][:, lo:hi] for b, (lo, hi) in ctx.train_ranges.items()], axis=1).T
    robust = RobustScaler(quantile_range=tuple(prep["robust_quantile_range"]), unit_variance=True).fit(x_train)
    del x_train
    center = robust.center_.astype(np.float32)[:, None]
    scale = robust.scale_.astype(np.float32)[:, None]
    for b in range(len(X)):
        X[b] -= center
        X[b] /= scale

    y_train = np.concatenate(
        [ctx.glove[b][:, lo + delay:hi + delay] for b, (lo, hi) in ctx.train_ranges.items()], axis=1
    ).T
    minmax = MinMaxScaler().fit(y_train)
    Y = (ctx.glove * minmax.scale_[None, :, None] + minmax.min_[None, :, None]).astype(np.float32)

    params = {f"p_{k}": v for k, v in p_params.items()}
    params.update(x_center=robust.center_, x_scale=robust.scale_, y_min=minmax.min_, y_scale=minmax.scale_)
    tr_index = train_windows(ctx.train_ranges, window, prep["train_stride"])
    val_starts = val_window_starts(d["valid_lo"], d["valid_hi"], window, prep["val_stride"])
    feats = FoldFeatures(
        preprocessing=name, X=X, Y=Y, y_min=minmax.min_, y_scale=minmax.scale_, train_index=tr_index,
        val_block=ctx.val_block, val_starts=val_starts, valid_lo=d["valid_lo"], valid_hi=d["valid_hi"],
        delay=delay, window=window, n_channels=ctx.n_channels, n_freqs=n_freqs, params=params, stats={},
    )
    feats.stats = feature_stats(feats)
    return feats


def feature_stats(feats: FoldFeatures) -> dict[str, Any]:
    valid = feats.X[:, :, feats.valid_lo:feats.valid_hi]
    sample = valid[:, :, ::7]
    q = np.percentile(sample, [0.1, 50, 99.9])
    return {
        "finite": bool(np.isfinite(valid).all() and np.isfinite(feats.Y).all()),
        "x_min": float(valid.min()),
        "x_max": float(valid.max()),
        "x_p001": float(q[0]),
        "x_median": float(q[1]),
        "x_p999": float(q[2]),
        "n_train_windows": int(len(feats.train_index)),
        "n_val_windows": int(len(feats.val_starts)),
        "n_val_columns": int(feats.valid_hi - feats.valid_lo),
    }


def feature_equivalence(a: FoldFeatures, b: FoldFeatures) -> dict[str, float]:
    """Diferencias entre dos tensores escalados en la región válida de todos los bloques."""
    lo, hi = a.valid_lo, a.valid_hi
    max_abs = ref_max = abs_sum = 0.0
    n = 0
    sa = sb = saa = sbb = sab = 0.0
    for blk in range(a.X.shape[0]):
        xa = a.X[blk, :, lo:hi].astype(np.float64)
        xb = b.X[blk, :, lo:hi].astype(np.float64)
        diff = np.abs(xa - xb)
        max_abs = max(max_abs, float(diff.max()))
        ref_max = max(ref_max, float(np.abs(xb).max()))
        abs_sum += float(diff.sum())
        n += diff.size
        sa, sb = sa + xa.sum(), sb + xb.sum()
        saa, sbb, sab = saa + (xa * xa).sum(), sbb + (xb * xb).sum(), sab + (xa * xb).sum()
    cov = sab / n - (sa / n) * (sb / n)
    corr = cov / np.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2))
    return {
        "max_abs_diff": max_abs,
        "rel_max_diff": max_abs / ref_max if ref_max > 0 else float("nan"),
        "mean_abs_diff": abs_sum / n,
        "corr": float(corr),
    }
