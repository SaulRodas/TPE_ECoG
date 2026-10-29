"""Estudio Optuna/TPE por sujeto (plan §5–§6): estudio local publicado en Drive, warmups, objetivo y bucle.

Cada trial entrena desde cero un modelo por fold (cinco) con early stopping y devuelve la media de los mejores
Pearson de validación. El bucle se repite hasta acumular ``n_trials_per_subject`` trials COMPLETE; los FAIL quedan
registrados con su causa y no cuentan.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import signal
import sqlite3
import tempfile
import time
import traceback
import warnings
import zipfile
from contextlib import closing
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import FrozenTrial, TrialState

from .config import seed_for
from .data import fold_blocks
from .models import EncoderDecoder1D, count_parameters
from .preprocessing import resolve_device, train_column_ranges, train_windows
from .runs import (
    RunnerLock,
    SubjectData,
    best_effort,
    code_info,
    environment_info,
    fold_record,
    free_memory,
    now,
    prepare_dir,
    raise_interrupt,
    run_fold,
    to_builtin,
    write_json,
)
from .space import architecture_spec, label, params_spec, suggest, warmup_trials
from .training import EarlyStopping, configure_torch, make_optimizer, mse_cosine_loss, set_seed

log = logging.getLogger("ecog_tpe")


def study_name(subject: str) -> str:
    return f"joint_tpe_{subject}"


def make_sampler(cfg: dict[str, Any], subject: str) -> optuna.samplers.TPESampler:
    s = cfg["search"]["sampler"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(seed=s["seed_by_subject"][subject], n_startup_trials=s["n_startup_trials"],
                                          multivariate=s["multivariate"], constant_liar=s["constant_liar"])


def n_complete(trials: list[FrozenTrial]) -> int:
    return sum(t.state == TrialState.COMPLETE for t in trials)


def _dispose(storage) -> None:
    engine = getattr(storage, "engine", None)
    if engine is not None:
        engine.dispose()  # cierra la conexión SQLite (en Windows, si no, el archivo queda bloqueado)


# ----------------------------------------------------------------------------- almacenamiento


class StudyArchive:
    """SQLite de trabajo en disco local, publicado tras cada trial como un zip atómico ``results/studies/<S>_study.zip``.

    El zip lleva la base (``study.sqlite``), el sampler serializado (``sampler.pkl``, para que el RNG continúe
    exactamente al reanudar) y ``meta.json``. SQLite nunca se escribe en vivo sobre Drive y la base y el sampler nunca
    quedan desincronizados. La carpeta local es nueva en cada proceso: nunca se hereda una base de otro estudio.
    """

    def __init__(self, results_dir: str | Path, subject: str, work_dir: str | Path | None = None):
        self.subject = subject
        self.path = Path(results_dir) / "studies" / f"{subject}_study.zip"
        work = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        work.mkdir(parents=True, exist_ok=True)
        self.local = Path(tempfile.mkdtemp(prefix=f"tpe_{subject}_", dir=work))
        self.database = self.local / "study.sqlite"
        self.storage = None

    def restore(self) -> tuple[Any | None, dict[str, Any] | None]:
        """Copia la base publicada al disco local; devuelve (sampler o None, meta o None si no hay estudio)."""
        if not self.path.exists():
            return None, None
        with zipfile.ZipFile(self.path) as archive:
            self.database.write_bytes(archive.read("study.sqlite"))
            meta = json.loads(archive.read("meta.json"))
            if meta.get("optuna") == optuna.__version__:
                return pickle.loads(archive.read("sampler.pkl")), meta
        log.warning("[%s] el estudio se publicó con Optuna %s y ahora hay %s: sampler nuevo con la misma semilla "
                    "(las propuestas siguientes no continúan el RNG anterior)", self.subject, meta.get("optuna"),
                    optuna.__version__)
        return None, meta

    def open_storage(self) -> optuna.storages.RDBStorage:
        self.storage = optuna.storages.RDBStorage(f"sqlite:///{self.database.as_posix()}")
        return self.storage

    def save(self, study: optuna.Study, sampler) -> None:
        snapshot = self.local / "snapshot.sqlite"
        snapshot.unlink(missing_ok=True)
        with closing(sqlite3.connect(self.database)) as source, closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
        trials = study.get_trials(deepcopy=False)
        meta = {"subject": self.subject, "study_name": study.study_name, "optuna": optuna.__version__,
                "n_trials": len(trials), "n_complete": n_complete(trials), "saved_at": now()}
        bundle = self.local / "study.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot, "study.sqlite")
            archive.writestr("sampler.pkl", pickle.dumps(sampler))
            archive.writestr("meta.json", json.dumps(meta, indent=2))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        for attempt in range(1, 4):
            try:
                shutil.copyfile(bundle, partial)
                partial.replace(self.path)
                return
            except OSError as exc:
                if attempt == 3:
                    raise
                log.warning("[%s] no se pudo publicar el estudio (%s); reintento", self.subject, exc)
                time.sleep(5 * attempt)

    def close(self) -> None:
        if self.storage is not None:
            _dispose(self.storage)
        shutil.rmtree(self.local, ignore_errors=True)


def load_trials(results_dir: str | Path, subject: str) -> list[FrozenTrial] | None:
    """Trials del estudio publicado, leídos desde una copia temporal (nunca se abre SQLite sobre Drive)."""
    path = Path(results_dir) / "studies" / f"{subject}_study.zip"
    if not path.exists():
        return None
    tmp = Path(tempfile.mkdtemp(prefix=f"tpe_read_{subject}_"))
    storage = None
    try:
        with zipfile.ZipFile(path) as archive:
            (tmp / "study.sqlite").write_bytes(archive.read("study.sqlite"))
        storage = optuna.storages.RDBStorage(f"sqlite:///{(tmp / 'study.sqlite').as_posix()}")
        return optuna.load_study(study_name=study_name(subject), storage=storage).get_trials(deepcopy=True)
    finally:
        if storage is not None:
            _dispose(storage)
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_warmups(study: optuna.Study, cfg: dict[str, Any]) -> int:
    """Encola los warmups (§6.1) sin trial COMPLETE ni en cola y con menos de ``max_warmup_attempts`` FAIL.

    Idempotente: se llama al crear el estudio y después de cada FAIL, de modo que un warmup que falla por una causa
    transitoria se repite con los mismos parámetros (plan §6, «reintentar fallas transitorias»).
    """
    trials = study.get_trials(deepcopy=False)
    added = 0
    for key, params in warmup_trials(cfg):
        mine = [t for t in trials if t.user_attrs.get("warmup") == key]
        if any(t.state in (TrialState.COMPLETE, TrialState.WAITING, TrialState.RUNNING) for t in mine):
            continue
        failures = sum(t.state == TrialState.FAIL for t in mine)
        if failures >= cfg["search"]["max_warmup_attempts"]:
            continue
        study.enqueue_trial(params, user_attrs={"source": "warmup", "warmup": key, "attempt": failures + 1})
        added += 1
    return added


def open_study(cfg: dict[str, Any], subject: str, archive: StudyArchive,
               results_dir: Path) -> tuple[optuna.Study, Any, bool]:
    """Restaura el estudio publicado o crea uno nuevo con sus diez warmups; devuelve (estudio, sampler, restaurado)."""
    sampler, meta = archive.restore()
    if meta is None:
        orphans = sorted((results_dir / "trials" / subject).glob("trial_*"))
        if orphans:
            raise RuntimeError(f"hay {len(orphans)} carpetas de trials en {results_dir / 'trials' / subject} pero no "
                               f"existe {archive.path}: ¿RESULTS_DIR equivocado o Drive sin sincronizar? No se crea un "
                               f"estudio nuevo encima.")
    sampler_restored = sampler is not None
    if sampler is None:
        sampler = make_sampler(cfg, subject)
    study = optuna.create_study(study_name=study_name(subject), storage=archive.open_storage(), sampler=sampler,
                                direction="maximize", load_if_exists=True, pruner=optuna.pruners.NopPruner())
    if meta is None:
        study.set_user_attr("subject", subject)
        study.set_user_attr("created_at", now())
        study.set_user_attr("config", to_builtin({k: cfg[k] for k in ("search", "training", "preprocessing")}))
        ensure_warmups(study, cfg)
        archive.save(study, sampler)
    return study, sampler, sampler_restored


# ----------------------------------------------------------------------------- objetivo


class Objective:
    """J_s(θ) = media sobre los cinco folds del mejor Pearson de validación (plan §5)."""

    def __init__(self, cfg: dict[str, Any], subject: str, data, results_dir: Path, device: torch.device,
                 folds: list[str], session: dict[str, Any], lock: RunnerLock, es_overrides: dict[str, Any] | None,
                 max_batches: int | None, runner: Callable = run_fold):
        self.cfg, self.subject, self.data, self.device = cfg, subject, data, device
        self.trials_dir = results_dir / "trials" / subject
        self.folds, self.session, self.lock = folds, session, lock
        self.es_overrides, self.max_batches, self.runner = es_overrides or {}, max_batches, runner
        self.environment = environment_info(device)

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest(trial, self.cfg["search"]["space"])
        source = trial.user_attrs.get("source", "tpe")
        trial.set_user_attr("source", source)
        for key, value in self.session.items():
            trial.set_user_attr(key, value)
        duplicate = next((t.number for t in trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                          if t.params == params), None)
        if duplicate is not None:  # con LR/L2 continuos no debería ocurrir; solo se anota
            trial.set_user_attr("duplicate_of", duplicate)
            log.warning("[%s t%03d] repite exactamente los parámetros del trial %d", self.subject, trial.number,
                        duplicate)

        tag = f"[{self.subject} t{trial.number:03d}]"
        trial_dir = prepare_dir(self.trials_dir / f"trial_{trial.number:03d}", self.trials_dir / "_interrupted")
        spec = params_spec(self.cfg, params)
        seeds = {fold: seed_for(self.cfg, self.subject, fold) for fold in self.folds}
        summary = {"subject": self.subject, "trial": trial.number, "source": source,
                   "warmup": trial.user_attrs.get("warmup"), "params": params, "label": label(params),
                   "state": "RUNNING", "started_at": now()}
        best_effort(lambda: write_json(trial_dir / "config.json", {
            **summary, "spec": spec, "folds": self.folds, "seeds": seeds,
            "early_stopping": {**self.cfg["training"]["early_stopping"], **self.es_overrides},
            "max_batches": self.max_batches, "session": self.session, "environment": self.environment,
            "code": code_info()}), "config.json")
        best_effort(lambda: write_json(trial_dir / "trial.json", summary), "trial.json")
        log.info("%s %s  lr=%.3g  wd=%.3g  (%s)", tag, label(params), params["learning_rate"],
                 params["weight_decay"], trial.user_attrs.get("warmup") or source)

        records: dict[str, dict[str, Any]] = {}
        try:
            for fold in self.folds:
                fold_tag = f"{tag} {fold}"
                self.lock.touch(fold_tag)
                stopper = EarlyStopping.from_config(self.cfg["training"]["early_stopping"], **self.es_overrides)
                metrics, history = self.runner(
                    self.data.fold(fold), spec, params, seeds[fold], self.cfg, self.device, trial_dir / fold,
                    stopper, tag=fold_tag, max_batches=self.max_batches,
                    on_epoch=lambda row, t=fold_tag: self.lock.touch(t))
                records[fold] = fold_record(metrics, history)
                trial.set_user_attr(f"fold_{fold}", records[fold])
            scores = [records[f]["best_pearson_macro"] for f in self.folds]
            value = float(np.mean(scores))
            attrs = {
                "fold_best_pearson": scores,
                "fold_best_epochs": [records[f]["best_epoch"] for f in self.folds],
                "fold_epochs_trained": [records[f]["epochs_trained"] for f in self.folds],
                "n_parameters": records[self.folds[0]]["n_parameters"],
                "train_seconds": float(sum(records[f]["train_seconds"] for f in self.folds)),
            }
            for key, value_ in attrs.items():
                trial.set_user_attr(key, value_)
            best_effort(lambda: write_json(trial_dir / "trial.json", {
                **summary, **attrs, "state": "COMPLETE", "value": value, "finished_at": now()}), "trial.json")
            log.info("%s objetivo %.4f  (folds: %s)", tag, value, " ".join(f"{s:.4f}" for s in scores))
            return value
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            trial.set_user_attr("error", message[:1000])
            best_effort(lambda: (trial_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8"),
                        "error.txt")
            best_effort(lambda: write_json(trial_dir / "trial.json", {
                **summary, "state": "FAIL", "error": message, "finished_at": now()}), "trial.json")
            raise


# ----------------------------------------------------------------------------- bucle


def run_search(
    cfg: dict[str, Any],
    subject: str,
    data_dir: str | Path,
    results_dir: str | Path,
    device: str | torch.device | None = None,
    work_dir: str | Path | None = None,
    n_trials: int | None = None,
    folds: list[str] | None = None,
    es_overrides: dict[str, Any] | None = None,
    max_batches: int | None = None,
    data=None,
    runner: Callable = run_fold,
) -> dict[str, Any]:
    """Ejecuta (o reanuda) el estudio de un sujeto hasta tener ``n_trials`` trials COMPLETE."""
    device = device if isinstance(device, torch.device) else resolve_device(device or cfg["preprocessing"]["device"])
    configure_torch(cfg["runtime"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results_dir = Path(results_dir)
    n_trials = n_trials or cfg["search"]["n_trials_per_subject"]
    folds = list(folds or cfg["folds"]["names"])
    lock = RunnerLock(results_dir, device).acquire()
    archive = StudyArchive(results_dir, subject, work_dir)
    previous = signal.signal(signal.SIGTERM, raise_interrupt)
    owns_data = data is None
    try:
        study, sampler, sampler_restored = open_study(cfg, subject, archive, results_dir)
        env = environment_info(device)
        session = {"session": f"{now()} {env['host']}", "sampler_restored": sampler_restored, "gpu": env["gpu"],
                   "torch": env["torch"], "cuda": env["cuda"], "optuna": env["optuna"]}
        if data is None:
            data = SubjectData(cfg, data_dir, subject, device)
        objective = Objective(cfg, subject, data, results_dir, device, folds, session, lock, es_overrides,
                              max_batches, runner)
        trials = study.get_trials(deepcopy=False)
        complete = n_complete(trials)
        log.info("[%s] estudio %s: %d/%d trials COMPLETE, %d en cola; sampler %s; dispositivo %s", subject,
                 study.study_name, complete, n_trials, sum(t.state == TrialState.WAITING for t in trials),
                 "restaurado" if sampler_restored else "nuevo", device)
        last: dict[str, FrozenTrial] = {}
        consecutive = 0
        while complete < n_trials:
            study.optimize(objective, n_trials=1, catch=(Exception,), gc_after_trial=True,
                           callbacks=[lambda _study, frozen: last.__setitem__("trial", frozen)])
            trial = last["trial"]
            if trial.state == TrialState.FAIL:
                consecutive += 1
                log.error("[%s t%03d] FAIL: %s", subject, trial.number, trial.user_attrs.get("error"))
                ensure_warmups(study, cfg)
            else:
                consecutive = 0
            archive.save(study, sampler)
            free_memory(device)
            complete = n_complete(study.get_trials(deepcopy=False))
            if consecutive >= cfg["search"]["max_consecutive_failures"]:
                raise RuntimeError(f"[{subject}] {consecutive} trials FAIL seguidos: revisa error.txt antes de relanzar")
        best = study.best_trial
        log.info("[%s] estudio completo: %d trials COMPLETE; mejor objetivo %.4f (trial %d, %s)", subject, complete,
                 best.value, best.number, label(best.params))
        return {"subject": subject, "n_complete": complete, "best_value": best.value, "best_trial": best.number}
    except KeyboardInterrupt:
        log.warning("[%s] interrumpido: el trial en curso no se publica y se repetirá al reanudar", subject)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if owns_data and data is not None:
            data.release()
        archive.close()
        lock.release()


def run_smoke(cfg: dict[str, Any], data_dir: str | Path, results_dir: str | Path,
              device: str | torch.device | None = None, work_dir: str | Path | None = None,
              subject: str | None = None) -> dict[str, Any]:
    """Cadena completa con datos reales en ``<results>/_smoke``: los dos warmups del par 1 en el primer fold,
    2 épocas × 2 batches. La carpeta del smoke anterior se borra (salida desechable)."""
    smoke_dir = Path(results_dir) / "_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    return run_search(cfg, subject or list(cfg["data"]["subjects"])[0], data_dir, smoke_dir, device, work_dir,
                      n_trials=2, folds=cfg["folds"]["names"][:1],
                      es_overrides={"max_epochs": 2, "min_epochs": 1, "patience": 1}, max_batches=2)


# ----------------------------------------------------------------------------- capacidad


def train_batches_per_epoch(cfg: dict[str, Any]) -> float:
    """Batches de entrenamiento por época, media de los cinco folds (≈ 231 con la configuración del plan)."""
    prep, batch = cfg["preprocessing"], cfg["training"]["batch_size"]
    counts = []
    for fold in cfg["folds"]["names"]:
        val, train = fold_blocks(fold, cfg)
        n = len(train_windows(train_column_ranges(val, train, cfg), prep["window"], prep["train_stride"]))
        counts.append(n // batch if cfg["training"]["drop_last"] else -(-n // batch))
    return float(np.mean(counts))


def run_capacity(cfg: dict[str, Any], results_dir: str | Path, device: str | torch.device | None = None,
                 subject: str | None = None, steps: int = 3) -> pd.DataFrame:
    """Plan §3.2: forward/backward + paso de Adam con el batch real en las 16 combinaciones discretas.

    Usa el sujeto con más canales (S3 por defecto) y reserva en la GPU un tensor del tamaño de X de un fold, como en
    el entrenamiento. Si alguna combinación no cabe, hay que ajustar el espacio ANTES del primer estudio.
    """
    device = device if isinstance(device, torch.device) else resolve_device(device or cfg["preprocessing"]["device"])
    configure_torch(cfg["runtime"])
    subjects = cfg["data"]["subjects"]
    subject = subject or max(subjects, key=lambda s: subjects[s]["n_channels_used"])
    prep, tcfg, d = cfg["preprocessing"], cfg["training"], cfg["_derived"]
    n_inputs = subjects[subject]["n_channels_used"] * prep["morlet"]["n_freqs"]
    cuda = device.type == "cuda"
    batches = train_batches_per_epoch(cfg)
    resident = torch.empty((d["n_blocks"], n_inputs, d["block_columns"]), dtype=torch.float32, device=device)
    space, rows = cfg["search"]["space"], []
    for arch, k, levels, f in product(space["architecture_id"], space["kernel_size"], space["encoder_levels"],
                                      space["base_filters"]):
        params = {"architecture_id": arch, "kernel_size": k, "encoder_levels": levels, "base_filters": f}
        row = {"subject": subject, **params, "label": label(params), "batch_size": tcfg["batch_size"]}
        model = optimizer = None
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            set_seed(0)
            model = EncoderDecoder1D(n_inputs, architecture_spec(cfg, arch, k, levels, f),
                                     n_outputs=len(cfg["data"]["fingers"]), window=prep["window"]).to(device)
            optimizer = make_optimizer(model, {**tcfg, "lr": 1e-4, "weight_decay": 1e-6})
            x = torch.randn(tcfg["batch_size"], n_inputs, prep["window"], device=device)
            y = torch.rand(tcfg["batch_size"], len(cfg["data"]["fingers"]), prep["window"], device=device)
            times = []
            for _ in range(steps + 1):  # el primero calienta
                if cuda:
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                loss = mse_cosine_loss(model(x), y, tcfg["loss"])[0]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if cuda:
                    torch.cuda.synchronize(device)
                times.append(time.perf_counter() - t0)
            seconds = float(np.median(times[1:]))
            row.update(status="ok", n_parameters=count_parameters(model), seconds_per_step=seconds,
                       est_epoch_seconds=seconds * batches,
                       peak_memory_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else None)
        except torch.cuda.OutOfMemoryError as exc:
            row.update(status="OOM", error=str(exc).splitlines()[0][:200])
        finally:
            del model, optimizer
            free_memory(device)
        rows.append(row)
        log.info("capacidad %s: %s", row["label"], row["status"])
    del resident
    free_memory(device)
    table = pd.DataFrame(rows)
    gpu = torch.cuda.get_device_name(device) if cuda else "cpu"
    path = Path(results_dir) / "capacity" / f"capacity_{''.join(c if c.isalnum() else '_' for c in gpu)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.assign(gpu=gpu, measured_at=now()).to_csv(path, index=False)
    return table
