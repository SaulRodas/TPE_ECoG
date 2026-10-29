"""Limpieza de la ECoG cruda: exclusión fija de canales y limitación robusta de amplitud.

Se aplica por igual a todas las configuraciones y siempre antes de la operación P.
"""

from __future__ import annotations

import numpy as np


def exclude_channels(ecog: np.ndarray, excluded: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Elimina (no anula) los canales excluidos; devuelve la señal y los índices originales conservados."""
    kept = np.setdiff1d(np.arange(ecog.shape[1]), np.asarray(excluded, dtype=int))
    return np.ascontiguousarray(ecog[:, kept]), kept


def fit_clip(train_blocks: list[np.ndarray], k: float, mad_scale: float, mad_floor: float) -> dict[str, np.ndarray]:
    """Mediana y 1.4826·MAD por canal, estimadas solo con los bloques de entrenamiento."""
    x = np.concatenate(train_blocks, axis=0)
    median = np.median(x, axis=0)
    mad = mad_scale * np.median(np.abs(x - median), axis=0)
    floored = mad < mad_floor
    scale = np.where(floored, mad_floor, mad)
    return {
        "median": median,
        "mad_scaled": mad,
        "floored": floored,
        "lower": median - k * scale,
        "upper": median + k * scale,
    }


def apply_clip(block: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorta con los límites de train; devuelve la señal y los conteos por canal (inferior, superior)."""
    lower, upper = params["lower"], params["upper"]
    n_low = (block < lower).sum(axis=0)
    n_high = (block > upper).sum(axis=0)
    return np.clip(block, lower, upper), n_low, n_high
