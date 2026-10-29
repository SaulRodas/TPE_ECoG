"""Corridas de un fold: características P0 por sujeto (calculadas una vez), entrenamiento con early stopping,
artefactos por fold y utilidades comunes (JSON, entorno, lock del runner).

Las utilidades y el lock son los de ``ecog_ablations.runner`` (commit 043c44f).
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from . import __version__
from .cleaning import exclude_channels
from .config import REPO_ROOT
from .data import load_subject
from .evaluation import trajectory_metrics
from .models import EncoderDecoder1D, count_parameters
from .preprocessing import FoldFeatures, build_filter_bank, compute_fold_features, prepare_fold_context
from .training import EarlyStopping, set_seed, train_fold

log = logging.getLogger("ecog_tpe")


# ----------------------------------------------------------------------------- utilidades


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"no serializable: {type(obj)}")


def to_builtin(obj: Any) -> Any:
    """Convierte tipos de NumPy a tipos de Python (los user_attrs de Optuna deben ser JSON)."""
    return json.loads(json.dumps(obj, default=_json_default))


def write_json(path: Path, obj: Any) -> None:
    """Escritura atómica (el monitor nunca lee un JSON a medias)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """CSV completo y atómico (se reescribe entero: sin filas duplicadas si Drive reintenta un append)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def best_effort(action: Callable[[], Any], what: str, attempts: int = 3) -> bool:
    """Escritura en Drive que no debe tumbar un entrenamiento: reintenta y, si sigue fallando, avisa y continúa."""
    for attempt in range(1, attempts + 1):
        try:
            action()
            return True
        except OSError as exc:
            if attempt == attempts:
                log.warning("no se pudo escribir %s: %s", what, exc)
                return False
            time.sleep(2 * attempt)
    return False


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_info() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "package_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": None if status is None else bool(status),
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    import mne
    import optuna
    import scipy
    import sklearn

    cuda = device.type == "cuda"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "mne": mne.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "optuna": optuna.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def prepare_dir(path: Path, archive_root: Path) -> Path:
    """Carpeta vacía para un intento nuevo; los restos de un intento no terminado se mueven a ``archive_root``."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        target = Path(archive_root) / f"{path.name}_{datetime.now():%Y%m%d-%H%M%S}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        log.warning("intento previo de %s archivado en %s", path.name, target)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pid_alive(pid: int) -> bool:
    if os.name != "posix":
        return False  # en Windows no se comprueba (uso local, manual)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def raise_interrupt(signum, frame):  # SIGTERM → mismo tratamiento que Ctrl+C
    raise KeyboardInterrupt


class RunnerLock:
    """Un runner por equipo: ``results/runners/<host>-<pid>.json`` con latido; la celda «detener» lo usa.

    Varias sesiones de Colab pueden trabajar sobre el mismo ``results/`` con sujetos disjuntos (un runtime por sujeto).
    """

    def __init__(self, results_dir: str | Path, device: torch.device):
        self.dir = Path(results_dir) / "runners"
        self.host = socket.gethostname()
        self.path = self.dir / f"{self.host}-{os.getpid()}.json"
        self.device = device

    def acquire(self) -> "RunnerLock":
        self.dir.mkdir(parents=True, exist_ok=True)
        for path in self.dir.glob(f"{self.host}-*.json"):
            pid = (read_json(path) or {}).get("pid")
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise RuntimeError(f"ya hay un runner activo en este equipo (pid {pid}); deténlo antes de lanzar otro")
            path.unlink(missing_ok=True)
        write_json(self.path, {"pid": os.getpid(), "host": self.host, "started_at": now(), "heartbeat": now(),
                               "current": None, "argv": sys.argv, "device": str(self.device)})
        return self

    def touch(self, current: str | None) -> None:
        def update():
            info = read_json(self.path) or {}
            info.update(heartbeat=now(), current=current)
            write_json(self.path, info)
        best_effort(update, "el latido del runner", attempts=1)

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- características por sujeto


@dataclass
class FoldData:
    feats: FoldFeatures
    y_true: np.ndarray              # (dedos, columnas validadas) en unidades originales del guante
    t_seconds: np.ndarray           # instante de cada columna validada dentro del registro
    clip_fraction_train: float
    clip_fraction_val: float


class SubjectData:
    """Características P0 de los cinco folds de un sujeto: se calculan una vez y las reutilizan todos los trials.

    Cada fold ajusta limpieza y escaladores solo con su entrenamiento (plan §2); la caché vive en RAM durante el
    proceso (≈ 0.4 GB por fold en S3).
    """

    def __init__(self, cfg: dict[str, Any], data_dir: str | Path, subject: str, device: torch.device):
        self.cfg, self.data_dir, self.subject, self.device = cfg, Path(data_dir), subject, device
        self._raw: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._bank = None
        self._folds: dict[str, FoldData] = {}

    def fold(self, fold: str) -> FoldData:
        if fold not in self._folds:
            t0 = time.perf_counter()
            if self._raw is None:
                raw = load_subject(self.data_dir, self.subject, self.cfg)
                ecog, kept = exclude_channels(raw["ecog"], self.cfg["data"]["subjects"][self.subject]["excluded"])
                self._raw, self._bank = (ecog, raw["dg"], kept), build_filter_bank(self.cfg)
            ecog, dg, kept = self._raw
            ctx = prepare_fold_context(ecog, dg, kept, self.subject, fold, self.cfg)
            feats = compute_fold_features(ctx, "P0", self._bank, self.cfg, self.device)
            if not feats.stats["finite"]:
                raise ValueError(f"[{self.subject} {fold}] características P0 no finitas")
            cols = feats.val_target_columns()
            columns = feats.val_block * self.cfg["_derived"]["block_columns"] + np.arange(cols.start, cols.stop)
            self._folds[fold] = FoldData(
                feats=feats, y_true=ctx.glove[feats.val_block][:, cols].copy(),
                t_seconds=columns / self.cfg["preprocessing"]["fs_features"],
                clip_fraction_train=ctx.clipped_fraction("train"), clip_fraction_val=ctx.clipped_fraction("val"),
            )
            del ctx
            free_memory(self.device)
            log.info("[%s %s] características P0: %.1f s, %d ventanas de train, %d canales", self.subject, fold,
                     time.perf_counter() - t0, feats.stats["n_train_windows"], feats.n_channels)
        return self._folds[fold]

    def release(self) -> None:
        self._folds.clear()
        self._raw = self._bank = None
        free_memory(self.device)


# ----------------------------------------------------------------------------- un fold


def fold_record(metrics: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen compacto del fold para los user_attrs del trial (el estudio basta para regenerar los reportes)."""
    return to_builtin({
        "seed": metrics["seed"],
        "best_epoch": metrics["best_epoch"],
        "epochs_trained": metrics["epochs_trained"],
        "stop_reason": metrics["stop_reason"],
        "best_pearson_macro": metrics["best_val_pearson_macro"],
        "restored_pearson_macro": metrics.get("pearson_macro"),
        "pearson": metrics.get("pearson"),
        "mse_macro": metrics.get("mse_macro"),
        "ccc_macro": metrics.get("ccc_macro"),
        "n_parameters": metrics["n_parameters"],
        "train_seconds": metrics["train_seconds"],
        "curve": [round(float(row["val_pearson_macro"]), 6) for row in history],
    })


def run_fold(
    data: FoldData,
    spec: dict[str, Any],
    params: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    stopper: EarlyStopping,
    tag: str = "",
    max_batches: int | None = None,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Modelo y optimizador nuevos en un fold; devuelve las métricas del mejor checkpoint y la historia por época.

    Escribe en ``out_dir``: status.json, history.csv, metrics.json y predictions.npz. No guarda pesos.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingers = cfg["data"]["fingers"]
    history: list[dict[str, Any]] = []
    status = {"status": "running", "seed": seed, "started_at": now(), "heartbeat": now(), "epoch": 0,
              "max_epochs": stopper.max_epochs}
    best_effort(lambda: write_json(out_dir / "status.json", status), "status.json")
    model = None
    try:
        set_seed(seed)
        model = EncoderDecoder1D(data.feats.X.shape[1], spec, n_outputs=len(fingers),
                                 window=cfg["preprocessing"]["window"]).to(device)
        n_parameters = count_parameters(model)
        tcfg = {**cfg["training"], "lr": params["learning_rate"], "weight_decay": params["weight_decay"]}

        def _on_epoch(row: dict[str, Any]) -> None:
            history.append(row)
            best_effort(lambda: write_rows(out_dir / "history.csv", history), "history.csv", attempts=1)
            status.update(epoch=row["epoch"], heartbeat=now(), last_val_pearson=row["val_pearson_macro"],
                          best_val_pearson=row["best_val_pearson_macro"], best_epoch=row["best_epoch"],
                          wait=row["wait"], last_epoch_seconds=row["epoch_seconds"])
            best_effort(lambda: write_json(out_dir / "status.json", status), "status.json", attempts=1)
            log.info("%s época %d/%d  train %.4f  val %.4f  r_val %.4f  mejor %.4f (ép. %s)  espera %d  (%.1f s)",
                     tag, row["epoch"], stopper.max_epochs, row["train_loss"], row["val_loss"],
                     row["val_pearson_macro"], row["best_val_pearson_macro"], row["best_epoch"], row["wait"],
                     row["epoch_seconds"])
            if on_epoch is not None:
                on_epoch(row)

        result = train_fold(model, data.feats, tcfg, stopper, device, seed, fingers, _on_epoch, max_batches)
        y_pred = data.feats.inverse_y(result.y_pred_scaled)
        metrics = trajectory_metrics(data.y_true, y_pred, fingers)
        if abs(result.restored_score - result.best_score) > 1e-4:
            log.warning("%s el checkpoint restaurado da r=%.6f y el mejor observado r=%.6f", tag,
                        result.restored_score, result.best_score)
        metrics.update({
            "seed": seed, "best_epoch": result.best_epoch, "epochs_trained": result.epochs_trained,
            "stop_reason": result.stop_reason, "best_val_pearson_macro": result.best_score,
            "restored_val_pearson_macro": result.restored_score, "n_parameters": n_parameters,
            "train_seconds": result.train_seconds, "n_train_windows": int(len(data.feats.train_index)),
            "n_val_columns": int(data.y_true.shape[1]), "max_batches": max_batches,
            "clip_fraction_train": data.clip_fraction_train, "clip_fraction_val": data.clip_fraction_val,
        })
        best_effort(lambda: write_json(out_dir / "metrics.json", metrics), "metrics.json")
        best_effort(lambda: np.savez_compressed(
            out_dir / "predictions.npz", y_true=data.y_true.astype(np.float32), y_pred=y_pred.astype(np.float32),
            t_seconds=data.t_seconds, fingers=np.array(fingers), best_epoch=result.best_epoch), "predictions.npz")
        status.update(status="completed", finished_at=now(), heartbeat=now(), best_epoch=result.best_epoch,
                      epochs_trained=result.epochs_trained, stop_reason=result.stop_reason,
                      best_val_pearson=result.best_score)
        best_effort(lambda: write_json(out_dir / "status.json", status), "status.json")
        log.info("%s fold terminado: mejor r=%.4f en la época %d de %d (%s)", tag, result.best_score,
                 result.best_epoch, result.epochs_trained, result.stop_reason)
        return metrics, history
    except Exception as exc:
        status.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        best_effort(lambda: write_json(out_dir / "status.json", status), "status.json")
        raise
    finally:
        del model
        free_memory(device)
