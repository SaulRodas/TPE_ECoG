"""Lectura de BCI Competition IV — Dataset 4 y división en bloques cronológicos.

Solo se leen ``train_data`` y ``train_dg``; ``test_data`` y las etiquetas oficiales los lee únicamente ``final_test.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat


def subject_path(data_dir: str | Path, subject: str, cfg: dict[str, Any]) -> Path:
    n = cfg["data"]["subjects"][subject]["file_index"]
    return Path(data_dir) / cfg["data"]["file_pattern"].format(n=n)


def load_subject(data_dir: str | Path, subject: str, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Devuelve ``ecog`` (n_muestras, canales_originales) y ``dg`` (n_muestras, 5) en float64."""
    path = subject_path(data_dir, subject, cfg)
    if not path.exists():
        raise FileNotFoundError(f"no se encontró {path}")
    mat = loadmat(path, variable_names=["train_data", "train_dg"])
    ecog = np.asarray(mat["train_data"], dtype=np.float64)
    dg = np.asarray(mat["train_dg"], dtype=np.float64)
    spec = cfg["data"]["subjects"][subject]
    n = cfg["data"]["n_samples"]
    if ecog.shape != (n, spec["n_channels_raw"]):
        raise ValueError(f"{subject}: train_data tiene forma {ecog.shape}, se esperaba {(n, spec['n_channels_raw'])}")
    if dg.shape != (n, len(cfg["data"]["fingers"])):
        raise ValueError(f"{subject}: train_dg tiene forma {dg.shape}")
    if not (np.isfinite(ecog).all() and np.isfinite(dg).all()):
        raise ValueError(f"{subject}: valores no finitos en los datos crudos")
    return {"ecog": ecog, "dg": dg}


def glove_hold_check(dg: np.ndarray, hold_rows: int) -> dict[str, float]:
    """Comprueba que los cambios de la retención de orden cero caen en filas múltiplo de ``hold_rows``."""
    changed = np.any(np.diff(dg, axis=0) != 0, axis=1)
    first_rows = np.flatnonzero(changed) + 1
    aligned = float(np.mean(first_rows % hold_rows == 0)) if first_rows.size else 1.0
    return {"n_changes": int(first_rows.size), "aligned_fraction": aligned}


def split_blocks(x: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    if x.shape[0] % n_blocks:
        raise ValueError("la longitud no es divisible entre el número de bloques")
    size = x.shape[0] // n_blocks
    return [x[i * size:(i + 1) * size] for i in range(n_blocks)]


def fold_blocks(fold: str, cfg: dict[str, Any]) -> tuple[int, list[int]]:
    """Índice del bloque validado y lista de bloques de entrenamiento."""
    names = cfg["folds"]["names"]
    val = names.index(fold)
    return val, [b for b in range(len(names)) if b != val]


def glove_to_feature_rate(dg_block: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Recupera las muestras efectivas de 25 Hz y las interpola con spline cúbico a 100 Hz.

    Devuelve (n_dedos, columnas_del_bloque). La interpolación no cruza fronteras de bloque.
    """
    hold = cfg["data"]["glove_hold_rows"]
    fs = cfg["data"]["fs_ecog"]
    n_cols = cfg["_derived"]["block_columns"]
    samples = dg_block[::hold]
    t_knots = np.arange(samples.shape[0]) * hold / fs
    t_cols = np.arange(n_cols) / cfg["preprocessing"]["fs_features"]
    return CubicSpline(t_knots, samples, axis=0)(t_cols).T
