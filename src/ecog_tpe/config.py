"""Carga y validación de configs/tpe.yaml (sin dependencias de torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tpe.yaml"

# Valores que implementa EncoderDecoder1D (copiado de ecog_ablations); A5b y A5 usan una parte.
NORMALIZATIONS = ("batch", "layer")
UPSAMPLES = ("interp", "convtranspose")
DECODER_ORDERS = ("upsample_conv_concat", "upsample_concat_conv")
CONVTRANSPOSE_MODES = ("depthwise",)
SKIP_POSITIONS = ("pre_pool",)
ARCH_KEYS = (
    "stem_channels", "encoder_channels", "kernel_sizes", "dilations", "decoder_channels",
    "decoder_kernel_sizes", "normalization", "dropout", "upsample", "decoder_order",
    "convtranspose_mode", "skip_position",
)
# Plan §3.1: orden de sugerencia de los seis hiperparámetros.
SEARCH_PARAMS = ("architecture_id", "learning_rate", "weight_decay", "kernel_size", "encoder_levels", "base_filters")
MAX_LEVELS = 5  # las dilataciones [1, 2, 3, 1, 2] se truncan a L niveles


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    cfg["_derived"] = derive(cfg)
    cfg["_path"] = str(path)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Rutas relativas se interpretan desde la raíz del repositorio."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def derive(cfg: dict[str, Any]) -> dict[str, int]:
    data, prep = cfg["data"], cfg["preprocessing"]
    fs, fs_feat = data["fs_ecog"], prep["fs_features"]
    if fs % fs_feat:
        raise ValueError("fs_ecog debe ser múltiplo de fs_features")
    decim = fs // fs_feat
    n_blocks = len(cfg["folds"]["names"])
    block_samples = int(cfg["folds"]["block_seconds"] * fs)
    if n_blocks * block_samples != data["n_samples"]:
        raise ValueError("los bloques no cubren exactamente n_samples")
    block_columns = block_samples // decim
    delay = prep["delay_ms"] * fs_feat / 1000
    if delay != int(delay):
        raise ValueError(f"delay_ms={prep['delay_ms']} no es un número entero de columnas a {fs_feat} Hz")
    edge = prep["edge_columns"]
    valid_lo, valid_hi = edge, block_columns - edge
    window = prep["window"]
    if valid_hi - valid_lo - prep["purge_columns"] < window:
        raise ValueError("la región válida no admite ventanas tras la purga")
    if valid_hi + int(delay) > block_columns:
        raise ValueError("el retardo saca el objetivo fuera del bloque")
    return {
        "decim": decim,
        "n_blocks": n_blocks,
        "block_samples": block_samples,
        "block_columns": block_columns,
        "delay_columns": int(delay),
        "valid_lo": valid_lo,
        "valid_hi": valid_hi,
    }


def validate_architecture(name: str, spec: dict[str, Any], window: int | None = None) -> None:
    missing = [k for k in ARCH_KEYS if k not in spec]
    if missing:
        raise ValueError(f"{name}: faltan parámetros {missing}")
    n_levels = len(spec["encoder_channels"])
    for key in ("kernel_sizes", "dilations", "decoder_channels", "decoder_kernel_sizes"):
        if len(spec[key]) != n_levels:
            raise ValueError(f"{name}: {key} debe tener {n_levels} elementos")
    for k in list(spec["kernel_sizes"]) + list(spec["decoder_kernel_sizes"]):
        if k % 2 == 0:
            raise ValueError(f"{name}: los kernels deben ser impares (padding d·(k−1)//2)")
    choices = {
        "normalization": NORMALIZATIONS,
        "upsample": UPSAMPLES,
        "decoder_order": DECODER_ORDERS,
        "skip_position": SKIP_POSITIONS,
    }
    for key, allowed in choices.items():
        if spec[key] not in allowed:
            raise ValueError(f"{name}: {key}={spec[key]!r} no implementado; opciones: {allowed}")
    if spec["upsample"] == "convtranspose":
        if spec["convtranspose_mode"] not in CONVTRANSPOSE_MODES:
            raise ValueError(f"{name}: convtranspose_mode={spec['convtranspose_mode']!r} no implementado")
    elif spec["convtranspose_mode"] is not None:
        raise ValueError(f"{name}: convtranspose_mode solo aplica con upsample=convtranspose")
    if not 0.0 <= float(spec["dropout"]) < 1.0:
        raise ValueError(f"{name}: dropout fuera de rango")
    if window is not None and window % (2 ** n_levels):
        raise ValueError(f"{name}: la ventana {window} no es divisible entre 2^{n_levels}")


def _validate_search(cfg: dict[str, Any]) -> None:
    search, window = cfg["search"], cfg["preprocessing"]["window"]
    space = search["space"]
    if tuple(space) != SEARCH_PARAMS:
        raise ValueError(f"search.space debe definir, en este orden, {SEARCH_PARAMS}")
    unknown = [a for a in space["architecture_id"] if a not in cfg["architectures"]]
    if unknown:
        raise ValueError(f"search.space.architecture_id: arquitecturas sin base en 'architectures': {unknown}")
    for key in ("learning_rate", "weight_decay"):
        low, high = space[key]
        if not 0 < low < high:
            raise ValueError(f"search.space.{key}: se espera 0 < mínimo < máximo (escala logarítmica)")
    if any(k % 2 == 0 for k in space["kernel_size"]):
        raise ValueError("search.space.kernel_size: los kernels deben ser impares")
    for levels in space["encoder_levels"]:
        if not 2 <= levels <= MAX_LEVELS or window % (2 ** levels):
            raise ValueError(f"search.space.encoder_levels: L={levels} fuera de 2–{MAX_LEVELS} o no divide la ventana")
    for i, pair in enumerate(search["warmups"], start=1):
        if set(pair) != set(SEARCH_PARAMS) - {"architecture_id"}:
            raise ValueError(f"search.warmups[{i}]: debe fijar {SEARCH_PARAMS[1:]}")
        for key in ("kernel_size", "encoder_levels", "base_filters"):
            if pair[key] not in space[key]:
                raise ValueError(f"search.warmups[{i}]: {key}={pair[key]} fuera del espacio")
        for key in ("learning_rate", "weight_decay"):
            low, high = space[key]
            if not low <= pair[key] <= high:
                raise ValueError(f"search.warmups[{i}]: {key}={pair[key]} fuera de [{low}, {high}]")
    missing = [s for s in cfg["data"]["subjects"] if s not in search["sampler"]["seed_by_subject"]]
    if missing:
        raise ValueError(f"search.sampler.seed_by_subject: faltan sujetos {missing}")
    es = cfg["training"]["early_stopping"]
    if not (1 <= es["patience"] and 0 <= es["min_epochs"] < es["max_epochs"] and es["min_delta"] >= 0):
        raise ValueError("training.early_stopping: se espera patience ≥ 1, 0 ≤ min_epochs < max_epochs y min_delta ≥ 0")
    bases = [cfg["training"]["seed_base"], *cfg["verification"]["seed_bases"]]
    if len(set(bases)) != len(bases):
        raise ValueError("las semillas base de búsqueda y verificación deben ser distintas")


def validate_config(cfg: dict[str, Any]) -> None:
    subjects = cfg["data"]["subjects"]
    for name, spec in subjects.items():
        excluded = spec["excluded"]
        if len(set(excluded)) != len(excluded) or not all(0 <= c < spec["n_channels_raw"] for c in excluded):
            raise ValueError(f"{name}: canales excluidos inválidos")
        if spec["n_channels_raw"] - len(excluded) != spec["n_channels_used"]:
            raise ValueError(f"{name}: n_channels_used no coincide con las exclusiones")
    window = cfg["preprocessing"]["window"]
    for name, spec in cfg["architectures"].items():
        validate_architecture(name, spec, window)
    _validate_search(cfg)


def seed_for(cfg: dict[str, Any], subject: str, fold: str, base: int | None = None) -> int:
    """Semilla de entrenamiento: base + 5·índice_sujeto + índice_fold (base 42 en búsqueda; 1042/2042 en verificación)."""
    subject_index = list(cfg["data"]["subjects"]).index(subject)
    fold_index = cfg["folds"]["names"].index(fold)
    base = cfg["training"]["seed_base"] if base is None else base
    return int(base + 5 * subject_index + fold_index)
