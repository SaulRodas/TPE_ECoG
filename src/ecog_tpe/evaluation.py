"""Reconstrucción por overlap-add y métricas sobre trayectorias completas (nunca por minibatch)."""

from __future__ import annotations

from typing import Any

import numpy as np


def overlap_add(window_preds: np.ndarray, starts: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Promedia predicciones solapadas (N, canales, W) sobre las columnas [lo, hi)."""
    n_out, width = window_preds.shape[1], window_preds.shape[2]
    acc = np.zeros((n_out, hi - lo), dtype=np.float64)
    count = np.zeros(hi - lo, dtype=np.float64)
    for pred, start in zip(window_preds, starts):
        a = int(start) - lo
        acc[:, a:a + width] += pred
        count[a:a + width] += 1
    if np.any(count == 0):
        raise ValueError("hay instantes de validación sin ventana que los cubra")
    return acc / count


def _invalid_reason(y_true: np.ndarray, y_pred: np.ndarray) -> str | None:
    if not np.all(np.isfinite(y_pred)):
        return "predicción no finita"
    if np.std(y_pred) <= 1e-10 * max(1.0, float(np.max(np.abs(y_pred)))):
        return "predicción constante"
    if np.std(y_true) <= 1e-10 * max(1.0, float(np.max(np.abs(y_true)))):
        return "objetivo constante"
    return None


def pearson_per_channel(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
    """Pearson por fila (canal/dedo); NaN y motivo cuando la evaluación es inválida."""
    r = np.full(y_true.shape[0], np.nan)
    reasons: list[str | None] = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        reason = _invalid_reason(t, p)
        reasons.append(reason)
        if reason is None:
            r[i] = float(np.corrcoef(t, p)[0, 1])
    return r, reasons


def concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de concordancia de Lin (varianzas poblacionales)."""
    mt, mp = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    return float(2 * cov / (y_true.var() + y_pred.var() + (mt - mp) ** 2))


def _none_if_nan(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray, fingers: list[str]) -> dict[str, Any]:
    """Métricas por dedo y macro; y_true / y_pred en unidades originales del guante, (dedos, tiempo)."""
    r, reasons = pearson_per_channel(y_true, y_pred)
    valid = all(reason is None for reason in reasons)
    out: dict[str, Any] = {
        "valid": valid,
        "invalid_reasons": {f: reason for f, reason in zip(fingers, reasons) if reason is not None},
        "pearson": {f: _none_if_nan(v) for f, v in zip(fingers, r)},
        "pearson_macro": float(np.mean(r)) if valid else None,
    }
    for name in ("mse", "rmse", "ccc"):
        out[name] = {}
    for f, t, p, reason in zip(fingers, y_true, y_pred, reasons):
        finite = bool(np.all(np.isfinite(p)))
        mse = float(np.mean((t - p) ** 2)) if finite else np.nan
        out["mse"][f] = _none_if_nan(mse)
        out["rmse"][f] = _none_if_nan(np.sqrt(mse))
        out["ccc"][f] = _none_if_nan(concordance(t, p)) if reason is None else None
    for name in ("mse", "rmse", "ccc"):
        values = list(out[name].values())
        out[f"{name}_macro"] = float(np.mean(values)) if valid and all(v is not None for v in values) else None
    return out
