"""Verificación de finalistas (plan §7): el mejor trial de A5b y el de A5 de cada sujeto se reentrenan con dos
semillas nuevas en los mismos cinco folds y con la misma política de early stopping.

Es una comprobación de estabilidad de entrenamiento, no una evaluación independiente. Reanudable: las corridas
completadas se saltan; un intento interrumpido se archiva en ``_attempts/`` y se repite.
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path
from typing import Any, Callable

import torch
from optuna.trial import FrozenTrial, TrialState

from .config import seed_for
from .preprocessing import resolve_device
from .runs import (
    RunnerLock,
    SubjectData,
    code_info,
    environment_info,
    now,
    prepare_dir,
    raise_interrupt,
    read_json,
    run_fold,
    write_json,
)
from .search import load_trials
from .space import label, params_spec
from .training import EarlyStopping, configure_torch

log = logging.getLogger("ecog_tpe")


def select_finalists(trials: list[FrozenTrial], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Mejor trial de cada arquitectura entre los primeros ``n_trials_per_subject`` COMPLETE (empate: número menor)."""
    n = cfg["search"]["n_trials_per_subject"]
    complete = sorted((t for t in trials if t.state == TrialState.COMPLETE), key=lambda t: t.number)[:n]
    if len(complete) < n:
        raise RuntimeError(f"el estudio tiene {len(complete)} trials COMPLETE; la verificación requiere {n}")
    finalists = {}
    for arch in cfg["search"]["space"]["architecture_id"]:
        mine = [t for t in complete if t.params["architecture_id"] == arch]
        if not mine:
            raise RuntimeError(f"ningún trial COMPLETE de {arch}")
        best = max(mine, key=lambda t: (t.value, -t.number))
        finalists[arch] = {"trial": best.number, "value": best.value, "params": dict(best.params),
                           "label": label(best.params)}
    return finalists


def run_verification(
    cfg: dict[str, Any],
    subject: str,
    data_dir: str | Path,
    results_dir: str | Path,
    device: str | torch.device | None = None,
    folds: list[str] | None = None,
    data=None,
    runner: Callable = run_fold,
) -> dict[str, Any]:
    """Entrena los dos finalistas del sujeto con cada semilla de verificación y cada fold."""
    device = device if isinstance(device, torch.device) else resolve_device(device or cfg["preprocessing"]["device"])
    configure_torch(cfg["runtime"])
    results_dir = Path(results_dir)
    out = results_dir / "verification" / subject
    folds = list(folds or cfg["folds"]["names"])
    lock = RunnerLock(results_dir, device).acquire()
    previous = signal.signal(signal.SIGTERM, raise_interrupt)
    owns_data = data is None
    try:
        selection = read_json(out / "finalists.json")
        if selection is None:  # se decide una sola vez; reanudar no vuelve a elegir
            trials = load_trials(results_dir, subject)
            if trials is None:
                raise RuntimeError(f"no existe el estudio publicado de {subject} en {results_dir / 'studies'}")
            selection = {"subject": subject, "selected_at": now(), "finalists": select_finalists(trials, cfg)}
            write_json(out / "finalists.json", selection)
        env, code = environment_info(device), code_info()
        if data is None:
            data = SubjectData(cfg, data_dir, subject, device)
        for arch, info in selection["finalists"].items():
            spec = params_spec(cfg, info["params"])
            log.info("[%s verif] %s: trial %d (%s), objetivo de búsqueda %.4f", subject, arch, info["trial"],
                     info["label"], info["value"])
            for base in cfg["verification"]["seed_bases"]:
                for fold in folds:
                    run_dir = out / arch / f"seed_{base}" / fold
                    status = read_json(run_dir / "status.json") or {}
                    if status.get("status") == "completed" and (run_dir / "metrics.json").exists():
                        continue
                    prepare_dir(run_dir, run_dir.parent / "_attempts")
                    seed = seed_for(cfg, subject, fold, base)
                    tag = f"[{subject} verif {arch} s{base} {fold}]"
                    write_json(run_dir / "config.json", {
                        "subject": subject, "architecture_id": arch, "source_trial": info["trial"],
                        "params": info["params"], "spec": spec, "seed_base": base, "seed": seed, "fold": fold,
                        "early_stopping": cfg["training"]["early_stopping"], "environment": env, "code": code,
                        "started_at": now()})
                    lock.touch(tag)
                    runner(data.fold(fold), spec, info["params"], seed, cfg, device, run_dir,
                           EarlyStopping.from_config(cfg["training"]["early_stopping"]), tag=tag,
                           on_epoch=lambda row, t=tag: lock.touch(t))
        log.info("[%s] verificación completa", subject)
        return selection
    except KeyboardInterrupt:
        log.warning("[%s] verificación interrumpida: la corrida en curso se repetirá al reanudar", subject)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if owns_data and data is not None:
            data.release()
        lock.release()
