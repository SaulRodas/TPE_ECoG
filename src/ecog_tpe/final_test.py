"""Evaluación final en el test oficial: los dos finalistas de cada sujeto (mejor A5b y mejor A5) × tres semillas.

Cada modelo se entrena con B1–B4 y early stopping sobre B5 (máximo 70 épocas), igual que el fold B5 de la búsqueda;
el test se predice una sola vez, con el mejor checkpoint restaurado. Es el único módulo que lee ``test_data`` y
``test_dg``: clip, escaladores, finalistas y elegida salen del desarrollo y quedan fijados en ``plan.json`` antes de
abrir el test. A diferencia de la búsqueda y la verificación, estos modelos sí guardan sus pesos (``model.pt``).
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.io import loadmat

from .cleaning import apply_clip, exclude_channels
from .config import seed_for
from .data import load_subject, subject_path
from .evaluation import overlap_add, trajectory_metrics
from .models import EncoderDecoder1D, count_parameters
from .preprocessing import (
    apply_input_transform,
    block_features,
    build_filter_bank,
    compute_fold_features,
    prepare_fold_context,
    resolve_device,
    val_window_starts,
)
from .reporting import _completed, verification_table
from .runs import (
    FoldData,
    RunnerLock,
    best_effort,
    code_info,
    environment_info,
    free_memory,
    now,
    prepare_dir,
    raise_interrupt,
    read_json,
    write_json,
    write_rows,
)
from .space import params_spec
from .training import EarlyStopping, configure_torch, set_seed, train_fold

log = logging.getLogger("ecog_tpe")

DAMAGED_RATIO = 20.0  # aviso si un canal usado supera este cociente pico/típico en test (los excluidos: 252–1817×)


# ----------------------------------------------------------------------------- datos de test


def labels_path(labels_dir: str | Path, subject: str, cfg: dict[str, Any]) -> Path:
    n = cfg["data"]["subjects"][subject]["file_index"]
    return Path(labels_dir) / cfg["final_test"]["test_labels_pattern"].format(n=n)


def channel_peak_ratio(ecog: np.ndarray, fs: int) -> np.ndarray:
    """Criterio del EDA: máximo / mediana de la desviación estándar local en ventanas de 1 s, por canal."""
    n = ecog.shape[0] // fs * fs
    local = ecog[:n].reshape(-1, fs, ecog.shape[1]).std(axis=1)
    return local.max(axis=0) / np.median(local, axis=0)


def load_test(data_dir: str | Path, labels_dir: str | Path, subject: str, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """``ecog`` (muestras, canales originales) y ``glove`` (dedos, columnas a 100 Hz) en float64.

    ``test_dg`` ya viene interpolado de forma suave a 1000 Hz (no es una retención de orden cero como ``train_dg``):
    se toma en el instante de cada columna, t = c/100 s, la misma convención que el guante de desarrollo.
    """
    comp, labels = subject_path(data_dir, subject, cfg), labels_path(labels_dir, subject, cfg)
    for path in (comp, labels):
        if not path.exists():
            raise FileNotFoundError(f"no se encontró {path}")
    ecog = np.asarray(loadmat(comp, variable_names=["test_data"])["test_data"], dtype=np.float64)
    dg = np.asarray(loadmat(labels, variable_names=["test_dg"])["test_dg"], dtype=np.float64)
    n, spec = cfg["final_test"]["n_samples"], cfg["data"]["subjects"][subject]
    if ecog.shape != (n, spec["n_channels_raw"]):
        raise ValueError(f"{subject}: test_data tiene forma {ecog.shape}, se esperaba {(n, spec['n_channels_raw'])}")
    if dg.shape != (n, len(cfg["data"]["fingers"])):
        raise ValueError(f"{subject}: test_dg tiene forma {dg.shape}")
    if not (np.isfinite(ecog).all() and np.isfinite(dg).all()):
        raise ValueError(f"{subject}: valores no finitos en el test")
    return {"ecog": ecog, "glove": np.ascontiguousarray(dg[::cfg["_derived"]["decim"]].T)}


@dataclass
class EvalSegment:
    X: np.ndarray                     # (canales·frecuencias, columnas) float32 escalado con el fold
    y_true: np.ndarray                # (dedos, hi − lo) en unidades originales, con el retardo
    t_seconds: np.ndarray             # instante de cada columna evaluada dentro del test
    starts: np.ndarray                # inicios de ventana (stride de validación + ventana final)
    lo: int
    hi: int


def eval_segment(ecog: np.ndarray, glove: np.ndarray, clip: dict[str, np.ndarray], feats, bank,
                 cfg: dict[str, Any], device: torch.device) -> tuple[EvalSegment, float]:
    """El test como un segmento continuo: clip, P0, Morlet y escalado ajustados con el train del fold (solo se
    transforma). Devuelve también la fracción de muestras recortadas."""
    prep, d = cfg["preprocessing"], cfg["_derived"]
    clean, n_low, n_high = apply_clip(ecog, clip)
    p_params = {k[2:]: v for k, v in feats.params.items() if k.startswith("p_")}
    X = block_features(apply_input_transform(feats.preprocessing, clean, p_params), bank, device)
    X = X.reshape(-1, X.shape[-1])
    X -= feats.params["x_center"].astype(np.float32)[:, None]
    X /= feats.params["x_scale"].astype(np.float32)[:, None]
    lo, hi = prep["edge_columns"], X.shape[1] - prep["edge_columns"]
    if hi + d["delay_columns"] > glove.shape[1]:
        raise ValueError("el retardo saca el objetivo fuera del test")
    cols = np.arange(lo + d["delay_columns"], hi + d["delay_columns"])
    segment = EvalSegment(X=X, y_true=glove[:, cols].copy(), t_seconds=cols / prep["fs_features"],
                          starts=val_window_starts(lo, hi, prep["window"], prep["val_stride"]), lo=lo, hi=hi)
    return segment, float((n_low.sum() + n_high.sum()) / clean.size)


class SubjectTestData:
    """Fold de validación (B5) y test de un sujeto: se calculan una vez y los usan las seis corridas."""

    def __init__(self, cfg: dict[str, Any], data_dir: str | Path, labels_dir: str | Path, subject: str,
                 device: torch.device):
        fold, d = cfg["final_test"]["validation_fold"], cfg["_derived"]
        excluded = cfg["data"]["subjects"][subject]["excluded"]
        raw = load_subject(data_dir, subject, cfg)
        ecog, kept = exclude_channels(raw["ecog"], excluded)
        bank = build_filter_bank(cfg)
        ctx = prepare_fold_context(ecog, raw["dg"], kept, subject, fold, cfg)
        del raw, ecog
        feats = compute_fold_features(ctx, "P0", bank, cfg, device)
        if not feats.stats["finite"]:
            raise ValueError(f"[{subject} {fold}] características P0 no finitas")
        cols = feats.val_target_columns()
        columns = feats.val_block * d["block_columns"] + np.arange(cols.start, cols.stop)
        self.fold = FoldData(
            feats=feats, y_true=ctx.glove[feats.val_block][:, cols].copy(),
            t_seconds=columns / cfg["preprocessing"]["fs_features"],
            clip_fraction_train=ctx.clipped_fraction("train"), clip_fraction_val=ctx.clipped_fraction("val"),
        )

        test = load_test(data_dir, labels_dir, subject, cfg)
        ratio = channel_peak_ratio(test["ecog"], cfg["data"]["fs_ecog"])
        x_test, _ = exclude_channels(test["ecog"], excluded)
        self.test, clip_fraction_test = eval_segment(x_test, test["glove"], ctx.clip, feats, bank, cfg, device)
        if not np.isfinite(self.test.X).all():
            raise ValueError(f"[{subject}] características de test no finitas")

        used = ratio[kept]
        self.info = {
            "subject": subject, "validation_fold": fold, "excluded_channels": list(excluded),
            "n_channels_used": int(len(kept)), "test_samples": int(x_test.shape[0]),
            "n_test_columns": int(self.test.hi - self.test.lo),
            "test_eval_seconds": [float(self.test.t_seconds[0]), float(self.test.t_seconds[-1])],
            "peak_ratio_excluded": {int(c): float(ratio[c]) for c in excluded},
            "peak_ratio_used_max": float(used.max()), "peak_ratio_used_argmax": int(kept[np.argmax(used)]),
            "peak_ratio_all": [float(r) for r in ratio],
            "clip_fraction_train": self.fold.clip_fraction_train, "clip_fraction_val": self.fold.clip_fraction_val,
            "clip_fraction_test": clip_fraction_test,
        }
        if used.max() > DAMAGED_RATIO:
            log.warning("[%s] el canal usado %d tiene un cociente pico/típico de %.0f× en test (posible daño; la "
                        "lista de exclusión es fija)", subject, kept[np.argmax(used)], used.max())
        # Lo necesario para aplicar los pesos guardados a datos nuevos.
        self.preprocessing = {"kept_channels": kept, **{f"clip_{k}": v for k, v in ctx.clip.items()}, **feats.params}
        del ctx, x_test, test
        free_memory(device)
        log.info("[%s] fold %s y test listos: %d columnas de test evaluadas, recorte en test %.2e, máximo pico/típico "
                 "en canales usados %.1f×", subject, fold, self.info["n_test_columns"], clip_fraction_test,
                 used.max())

    def release(self) -> None:
        self.fold = self.test = None


# ----------------------------------------------------------------------------- plan congelado


def freeze_plan(results_dir: str | Path, subject: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Finalistas del sujeto y elegida (mayor media de las dos semillas de verificación), fijados antes de leer el
    test. Se escribe una vez; después solo se lee."""
    results_dir = Path(results_dir)
    path = results_dir / "final_test" / subject / "plan.json"
    plan = read_json(path)
    if plan is not None:
        return plan
    selection = read_json(results_dir / "verification" / subject / "finalists.json")
    if selection is None:
        raise RuntimeError(f"{subject}: no hay verificación (verification/{subject}/finalists.json)")
    ver = verification_table(results_dir, cfg)
    done = _completed(ver[ver["subject"] == subject]) if not ver.empty else ver
    expected = len(cfg["verification"]["seed_bases"]) * len(cfg["folds"]["names"])
    finalists = {}
    for arch, info in selection["finalists"].items():
        runs = done[done["architecture_id"] == arch] if not done.empty else done
        if len(runs) < expected:
            raise RuntimeError(f"{subject} {arch}: verificación incompleta ({len(runs)}/{expected} corridas)")
        seed_means = runs.groupby("seed_base")["best_pearson_macro"].mean()
        finalists[arch] = {**info, "verification_cv_best": float(seed_means.mean()),
                           "verification_seed_sd": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else None}
    chosen = max(finalists, key=lambda a: finalists[a]["verification_cv_best"])  # empate: el primero, como el reporte
    ft = cfg["final_test"]
    plan = {"subject": subject, "frozen_at": now(), "chosen": chosen, "validation_fold": ft["validation_fold"],
            "max_epochs": ft["max_epochs"], "seed_bases": list(ft["seed_bases"]), "finalists": finalists}
    write_json(path, plan)
    log.info("[%s] plan de test fijado: elegida %s (%s)", subject, chosen,
             ", ".join(f"{a} {f['verification_cv_best']:.4f}" for a, f in finalists.items()))
    return plan


# ----------------------------------------------------------------------------- una corrida


@torch.no_grad()
def predict_segment(model: torch.nn.Module, segment: EvalSegment, window: int, batch_size: int,
                    device: torch.device) -> np.ndarray:
    """Predicción escalada (dedos, hi − lo) por ventanas y overlap-add, en modo eval."""
    model.eval()
    windows = torch.from_numpy(segment.X).to(device).unfold(1, window, 1).permute(1, 0, 2)  # (T−W+1, F, W)
    preds = []
    for i in range(0, len(segment.starts), batch_size):
        idx = torch.as_tensor(segment.starts[i:i + batch_size], device=device)
        preds.append(model(windows[idx]).float().cpu().numpy())
    return overlap_add(np.concatenate(preds), segment.starts, segment.lo, segment.hi)


def save_model(path: Path, model: EncoderDecoder1D, extra: dict[str, Any]) -> None:
    """Pesos del mejor checkpoint y lo necesario para reconstruir el modelo; escritura atómica."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "n_inputs": model.n_inputs, "spec": model.spec, "window": model.window, **extra}, tmp)
    os.replace(tmp, path)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[EncoderDecoder1D, dict[str, Any]]:
    """Reconstruye un modelo guardado por ``save_model`` (en modo eval)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EncoderDecoder1D(ckpt["n_inputs"], ckpt["spec"], n_outputs=ckpt["n_outputs"], window=ckpt["window"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def run_test_model(
    data: SubjectTestData,
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
) -> dict[str, Any]:
    """Entrena con early stopping sobre el fold de validación, restaura el mejor checkpoint y predice el test una vez.

    Escribe en ``out_dir``: status.json, history.csv, metrics.json, predictions.npz (test) y model.pt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingers = cfg["data"]["fingers"]
    feats = data.fold.feats
    history: list[dict[str, Any]] = []
    status = {"status": "running", "seed": seed, "started_at": now(), "heartbeat": now(), "epoch": 0,
              "max_epochs": stopper.max_epochs}
    best_effort(lambda: write_json(out_dir / "status.json", status), "status.json")
    model = None
    try:
        set_seed(seed)
        model = EncoderDecoder1D(feats.X.shape[1], spec, n_outputs=len(fingers),
                                 window=cfg["preprocessing"]["window"]).to(device)
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

        result = train_fold(model, feats, tcfg, stopper, device, seed, fingers, _on_epoch, max_batches)
        val = trajectory_metrics(data.fold.y_true, feats.inverse_y(result.y_pred_scaled), fingers)
        # Una sola predicción del test, con el mejor checkpoint ya restaurado por train_fold.
        y_pred = feats.inverse_y(predict_segment(model, data.test, cfg["preprocessing"]["window"],
                                                 cfg["training"]["batch_size"], device))
        test = trajectory_metrics(data.test.y_true, y_pred, fingers)
        metrics = {
            "seed": seed, "best_epoch": result.best_epoch, "epochs_trained": result.epochs_trained,
            "stop_reason": result.stop_reason, "best_val_pearson_macro": result.best_score,
            "restored_val_pearson_macro": result.restored_score, "val": val, "test": test,
            "n_parameters": count_parameters(model), "train_seconds": result.train_seconds,
            "n_train_windows": int(len(feats.train_index)), "n_test_columns": int(data.test.y_true.shape[1]),
            "max_batches": max_batches,
        }
        if not best_effort(lambda: save_model(out_dir / "model.pt", model, {
                "n_outputs": len(fingers), "params": params, "seed": seed, "best_epoch": result.best_epoch}),
                "model.pt"):
            raise OSError("no se pudieron guardar los pesos (model.pt)")
        best_effort(lambda: write_json(out_dir / "metrics.json", metrics), "metrics.json")
        best_effort(lambda: np.savez_compressed(
            out_dir / "predictions.npz", y_true=data.test.y_true.astype(np.float32), y_pred=y_pred.astype(np.float32),
            t_seconds=data.test.t_seconds, fingers=np.array(fingers), best_epoch=result.best_epoch), "predictions.npz")
        status.update(status="completed", finished_at=now(), heartbeat=now(), best_epoch=result.best_epoch,
                      epochs_trained=result.epochs_trained, stop_reason=result.stop_reason,
                      best_val_pearson=result.best_score, test_pearson=test["pearson_macro"])
        best_effort(lambda: write_json(out_dir / "status.json", status), "status.json")
        log.info("%s terminado: r_val %.4f en la época %d de %d (%s) · r_test %s", tag, result.best_score,
                 result.best_epoch, result.epochs_trained, result.stop_reason,
                 f"{test['pearson_macro']:.4f}" if test["pearson_macro"] is not None else "inválido")
        return metrics
    except Exception as exc:
        status.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        best_effort(lambda: write_json(out_dir / "status.json", status), "status.json")
        raise
    finally:
        del model
        free_memory(device)


# ----------------------------------------------------------------------------- un sujeto


def _completed_run(run_dir: Path) -> bool:
    status = read_json(run_dir / "status.json") or {}
    return status.get("status") == "completed" and (run_dir / "metrics.json").exists()


def run_final_test(
    cfg: dict[str, Any],
    subject: str,
    data_dir: str | Path,
    labels_dir: str | Path,
    results_dir: str | Path,
    device: str | torch.device | None = None,
    data=None,
    runner: Callable = run_test_model,
    max_batches: int | None = None,
    es_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Los dos finalistas del sujeto × las semillas de ``final_test``. Reanudable: las corridas completadas se
    saltan; un intento interrumpido se archiva en ``_attempts/`` y se repite."""
    device = device if isinstance(device, torch.device) else resolve_device(device or cfg["preprocessing"]["device"])
    configure_torch(cfg["runtime"])
    results_dir = Path(results_dir)
    out = results_dir / "final_test" / subject
    ft = cfg["final_test"]
    lock = RunnerLock(results_dir, device).acquire()
    previous = signal.signal(signal.SIGTERM, raise_interrupt)
    owns_data = data is None
    try:
        plan = freeze_plan(results_dir, subject, cfg)  # antes de abrir el test
        pending = [(arch, base) for arch in plan["finalists"] for base in plan["seed_bases"]
                   if not _completed_run(out / arch / f"seed_{base}")]
        if pending and data is None:
            lock.touch(f"[{subject} test] características")
            data = SubjectTestData(cfg, data_dir, labels_dir, subject, device)
            write_json(out / "test_data.json", data.info)
            np.savez(out / "preprocessing.npz", **data.preprocessing)
        env, code = environment_info(device), code_info()
        overrides = {"max_epochs": plan["max_epochs"], **(es_overrides or {})}
        for arch, base in pending:
            info = plan["finalists"][arch]
            spec = params_spec(cfg, info["params"])
            run_dir = out / arch / f"seed_{base}"
            prepare_dir(run_dir, run_dir.parent / "_attempts")
            seed = seed_for(cfg, subject, plan["validation_fold"], base)
            tag = f"[{subject} test {arch} s{base}]"
            write_json(run_dir / "config.json", {
                "subject": subject, "architecture_id": arch, "chosen": arch == plan["chosen"],
                "source_trial": info["trial"], "params": info["params"], "spec": spec, "seed_base": base,
                "seed": seed, "validation_fold": plan["validation_fold"],
                "early_stopping": {**cfg["training"]["early_stopping"], **overrides}, "environment": env,
                "code": code, "started_at": now()})
            lock.touch(tag)
            runner(data, spec, info["params"], seed, cfg, device, run_dir,
                   EarlyStopping.from_config(cfg["training"]["early_stopping"], **overrides), tag=tag,
                   max_batches=max_batches, on_epoch=lambda row, t=tag: lock.touch(t))
        log.info("[%s] test completo", subject)
        return plan
    except KeyboardInterrupt:
        log.warning("[%s] test interrumpido: la corrida en curso se repetirá al reanudar", subject)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if owns_data and data is not None:
            data.release()
        lock.release()
