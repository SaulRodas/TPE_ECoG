"""Tablas (plan §9), simulación retrospectiva (§1.1), comparación con las ablaciones (§10), figuras y monitor.

Los trials se leen del estudio publicado (``results/studies/<S>_study.zip``), no de las carpetas: cada trial guarda en
sus user_attrs el resumen de sus folds. La verificación y el test se leen de los ``metrics.json`` de sus corridas.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optuna.trial import FrozenTrial, TrialState

from .config import SEARCH_PARAMS
from .runs import read_json
from .search import load_trials
from .space import label
from .training import EarlyStopping, simulate

# --- paleta de las ablaciones (modo claro para PNG)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ARCH_COLORS = {"A5b": SLOTS[0], "A5": SLOTS[1]}
SOURCE_LABELS = {
    "ablation_epoch50": "ablación, época 50",
    "ablation_es_sim": "ablación, parada simulada",
    "warmup_reference": "warmup de referencia (par 1)",
    "tpe_best": "mejor trial TPE",
    "verification": "verificación (2 semillas)",
}

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "font.family": "sans-serif", "lines.linewidth": 2.0,
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def round_half_up(x: float) -> int:
    """Entero más cercano con los medios hacia arriba (``round`` de Python redondea medios al par)."""
    return int(np.floor(x + 0.5))


# ============================================================================ trials y folds (estudio publicado)


def load_all_trials(results_dir: str | Path, cfg: dict[str, Any]) -> dict[str, list[FrozenTrial]]:
    out = {}
    for subject in cfg["data"]["subjects"]:
        trials = load_trials(results_dir, subject)
        if trials is not None:
            out[subject] = trials
    return out


def _fold_records(trial: FrozenTrial, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f: trial.user_attrs[f"fold_{f}"] for f in cfg["folds"]["names"] if f"fold_{f}" in trial.user_attrs}


def trials_table(trials: dict[str, list[FrozenTrial]], cfg: dict[str, Any]) -> pd.DataFrame:
    """``tpe_trials.csv``: una fila por trial ejecutado (COMPLETE o FAIL; los que siguen en cola no aparecen)."""
    rows = []
    for subject, subject_trials in trials.items():
        for t in subject_trials:
            if t.state == TrialState.WAITING:
                continue
            records = list(_fold_records(t, cfg).values())
            complete = t.state == TrialState.COMPLETE
            duration = (t.datetime_complete - t.datetime_start).total_seconds() \
                if t.datetime_complete and t.datetime_start else None
            rows.append({
                "subject": subject, "trial": t.number, "state": t.state.name,
                "source": t.user_attrs.get("source"), "warmup": t.user_attrs.get("warmup"),
                **{p: t.params.get(p) for p in SEARCH_PARAMS},
                "label": label(t.params) if len(t.params) == len(SEARCH_PARAMS) else None,
                "objective_cv_best": t.value if complete else None,
                "n_parameters": records[0]["n_parameters"] if records else None,
                "mean_best_epoch": _mean(r["best_epoch"] for r in records),
                "mean_epochs_trained": _mean(r["epochs_trained"] for r in records),
                "total_train_seconds": float(sum(r["train_seconds"] for r in records)) if records else None,
                "n_valid_folds": len(records),
                "error": t.user_attrs.get("error"), "duplicate_of": t.user_attrs.get("duplicate_of"),
                "session": t.user_attrs.get("session"), "sampler_restored": t.user_attrs.get("sampler_restored"),
                "gpu": t.user_attrs.get("gpu"), "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete, "duration_seconds": duration,
            })
    return pd.DataFrame(rows)


def folds_table(trials: dict[str, list[FrozenTrial]], cfg: dict[str, Any]) -> pd.DataFrame:
    """``tpe_folds.csv``: una fila por fold terminado (también los folds previos al fallo de un trial FAIL)."""
    fingers = cfg["data"]["fingers"]
    rows = []
    for subject, subject_trials in trials.items():
        for t in subject_trials:
            for fold, r in _fold_records(t, cfg).items():
                rows.append({
                    "subject": subject, "trial": t.number, "trial_state": t.state.name,
                    "architecture_id": t.params.get("architecture_id"), "fold": fold, "seed": r["seed"],
                    "best_epoch": r["best_epoch"], "epochs_trained": r["epochs_trained"],
                    "stop_reason": r["stop_reason"], "best_pearson_macro": r["best_pearson_macro"],
                    **{f"pearson_{f}": (r.get("pearson") or {}).get(f) for f in fingers},
                    "restored_pearson_macro": r.get("restored_pearson_macro"),
                    "mse_macro_best": r.get("mse_macro"), "ccc_macro_best": r.get("ccc_macro"),
                    "n_parameters": r["n_parameters"], "train_seconds": r["train_seconds"],
                })
    return pd.DataFrame(rows)


def search_complete(trials: list[FrozenTrial], cfg: dict[str, Any]) -> list[FrozenTrial]:
    """Los primeros ``n_trials_per_subject`` trials COMPLETE (presupuesto cerrado del plan)."""
    complete = sorted((t for t in trials if t.state == TrialState.COMPLETE), key=lambda t: t.number)
    return complete[:cfg["search"]["n_trials_per_subject"]]


# ============================================================================ verificación


def verification_table(results_dir: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """``tpe_verification.csv``: una fila por finalista × semilla × fold (también las pendientes)."""
    fingers, rows = cfg["data"]["fingers"], []
    for subject in cfg["data"]["subjects"]:
        base_dir = Path(results_dir) / "verification" / subject
        selection = read_json(base_dir / "finalists.json")
        if not selection:
            continue
        for arch, info in selection["finalists"].items():
            for base in cfg["verification"]["seed_bases"]:
                for fold in cfg["folds"]["names"]:
                    run_dir = base_dir / arch / f"seed_{base}" / fold
                    status = read_json(run_dir / "status.json") or {}
                    row = {"subject": subject, "architecture_id": arch, "source_trial": info["trial"],
                           "seed_base": base, "seed": status.get("seed"), "fold": fold,
                           "state": status.get("status", "pending")}
                    metrics = read_json(run_dir / "metrics.json") if row["state"] == "completed" else None
                    if metrics:
                        row.update({
                            "best_epoch": metrics["best_epoch"], "epochs_trained": metrics["epochs_trained"],
                            "stop_reason": metrics["stop_reason"],
                            "best_pearson_macro": metrics["best_val_pearson_macro"],
                            **{f"pearson_{f}": metrics["pearson"].get(f) for f in fingers},
                            "mse_macro_best": metrics.get("mse_macro"), "ccc_macro_best": metrics.get("ccc_macro"),
                            "train_seconds": metrics["train_seconds"],
                        })
                    rows.append(row)
    return pd.DataFrame(rows)


def _completed(ver: pd.DataFrame) -> pd.DataFrame:
    return ver[ver["state"] == "completed"] if not ver.empty else ver


def verification_paired(ver: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A5 − A5b apareado por sujeto, semilla y fold (plan §7.5); resumen por sujeto y global (sujetos con igual peso)."""
    done = _completed(ver)
    if done.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = done.pivot_table(index=["subject", "seed_base", "fold"], columns="architecture_id",
                            values="best_pearson_macro").reset_index()
    if not {"A5", "A5b"} <= set(wide.columns):
        return pd.DataFrame(), pd.DataFrame()
    wide = wide.dropna(subset=["A5", "A5b"])
    wide["delta_A5_minus_A5b"] = wide["A5"] - wide["A5b"]
    summary = (wide.groupby("subject")["delta_A5_minus_A5b"]
               .agg(n_pairs="size", delta_mean="mean", delta_sd="std",
                    n_positive=lambda d: int((d > 0).sum())).reset_index())
    if len(summary) > 1:
        glob = {"subject": "global", "n_pairs": int(summary["n_pairs"].sum()),
                "delta_mean": summary["delta_mean"].mean(), "delta_sd": summary["delta_mean"].std(ddof=1),
                "n_positive": int(summary["n_positive"].sum())}
        summary = pd.concat([summary, pd.DataFrame([glob])], ignore_index=True)
    return wide, summary


def selected_models(trials_df: pd.DataFrame, ver: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """``tpe_selected_models.csv``: por sujeto, el finalista con mayor media de las dos semillas de verificación."""
    done = _completed(ver)
    if done.empty:
        return pd.DataFrame()
    expected = len(cfg["verification"]["seed_bases"]) * len(cfg["folds"]["names"])
    rows = []
    for subject, g in done.groupby("subject", sort=False):
        candidates = []
        for arch, ga in g.groupby("architecture_id", sort=False):
            seed_means = ga.groupby("seed_base")["best_pearson_macro"].mean()
            candidates.append({
                "architecture": arch, "trial": int(ga["source_trial"].iloc[0]),
                "verification_cv_best": float(seed_means.mean()),
                "verification_seed_sd": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else np.nan,
                "median_best_epoch": round_half_up(float(ga["best_epoch"].median())),
                "complete": len(ga) == expected,
            })
        chosen = max(candidates, key=lambda c: c["verification_cv_best"])
        others = [c for c in candidates if c is not chosen]
        trial = trials_df[(trials_df["subject"] == subject) & (trials_df["trial"] == chosen["trial"])]
        params = trial.iloc[0] if not trial.empty else {}
        rows.append({
            "subject": subject, "chosen_architecture": chosen["architecture"], "chosen_trial": chosen["trial"],
            **{p: params.get(p) for p in SEARCH_PARAMS if p != "architecture_id"},
            "search_cv_best": params.get("objective_cv_best"),
            "verification_cv_best": chosen["verification_cv_best"],
            "verification_seed_sd": chosen["verification_seed_sd"],
            "median_best_epoch": chosen["median_best_epoch"],
            "verification_complete": all(c["complete"] for c in candidates) and len(candidates) == 2,
            "other_architecture": others[0]["architecture"] if others else None,
            "other_verification_cv_best": others[0]["verification_cv_best"] if others else None,
        })
    return pd.DataFrame(rows)


# ============================================================================ evaluación final en test


TEST_PROTOCOLS = {"b1b4_es_b5": "final_test", "b1b5_fixed": "final_test_full"}  # protocolo → carpeta de results/
PROTOCOL_LABELS = {"b1b4_es_b5": "B1–B4, parada en B5", "b1b5_fixed": "B1–B5, épocas fijas"}


def final_test_table(results_dir: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """``tpe_test.csv``: una fila por protocolo × sujeto × finalista × semilla (también las pendientes)."""
    fingers, rows = cfg["data"]["fingers"], []
    for protocol, folder in TEST_PROTOCOLS.items():
        for subject in cfg["data"]["subjects"]:
            base_dir = Path(results_dir) / folder / subject
            plan = read_json(base_dir / "plan.json")
            if not plan:
                continue
            for arch, info in plan["finalists"].items():
                for base in plan["seed_bases"]:
                    run_dir = base_dir / arch / f"seed_{base}"
                    status = read_json(run_dir / "status.json") or {}
                    row = {"protocol": protocol, "subject": subject, "architecture_id": arch,
                           "chosen": arch == plan["chosen"], "source_trial": info["trial"], "label": info["label"],
                           "seed_base": base, "seed": status.get("seed"), "state": status.get("status", "pending"),
                           "verification_cv_best": info["verification_cv_best"], "epochs_fixed": info.get("epochs")}
                    metrics = read_json(run_dir / "metrics.json") if row["state"] == "completed" else None
                    if metrics:
                        test = metrics["test"]
                        row.update({
                            "best_epoch": metrics["best_epoch"], "epochs_trained": metrics["epochs_trained"],
                            "stop_reason": metrics["stop_reason"],
                            "val_pearson_macro": metrics["best_val_pearson_macro"],
                            "test_pearson_macro": test["pearson_macro"],
                            **{f"test_pearson_{f}": test["pearson"].get(f) for f in fingers},
                            "test_mse_macro": test.get("mse_macro"), "test_ccc_macro": test.get("ccc_macro"),
                            "train_seconds": metrics["train_seconds"],
                        })
                    rows.append(row)
    return pd.DataFrame(rows)


def _global_groups(summary: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Grupos de las filas «global»: cada arquitectura y las elegidas (sujetos con igual peso)."""
    groups = [(arch, summary[summary["architecture_id"] == arch]) for arch in summary["architecture_id"].unique()]
    groups.append(("elegida", summary[summary["chosen"]]))
    return [(name, g) for name, g in groups if not g.empty]


def final_test_summary(test: pd.DataFrame) -> pd.DataFrame:
    """``tpe_test_summary.csv``: media y desviación de las semillas por protocolo × sujeto × finalista; filas globales
    por protocolo, arquitectura y para las elegidas (sujetos con igual peso; ahí la desviación es entre sujetos)."""
    done = _completed(test)
    if done.empty:
        return pd.DataFrame()
    out = []
    for protocol, p in done.groupby("protocol", sort=False):
        rows = []
        for (subject, arch), g in p.groupby(["subject", "architecture_id"], sort=False):
            rows.append({
                "protocol": protocol, "subject": subject, "architecture_id": arch, "chosen": bool(g["chosen"].iloc[0]),
                "label": g["label"].iloc[0], "source_trial": int(g["source_trial"].iloc[0]), "n_seeds": len(g),
                "test_pearson_mean": g["test_pearson_macro"].mean(),
                "test_pearson_sd": g["test_pearson_macro"].std(ddof=1),
                "val_pearson_mean": g["val_pearson_macro"].mean(),
                "verification_cv_best": float(g["verification_cv_best"].iloc[0]),
                "mean_best_epoch": g["best_epoch"].mean(), "mean_epochs_trained": g["epochs_trained"].mean(),
            })
        summary = pd.DataFrame(rows)
        glob = [{"protocol": protocol, "subject": "global", "architecture_id": name, "chosen": name == "elegida",
                 "n_seeds": int(g["n_seeds"].sum()), "test_pearson_mean": g["test_pearson_mean"].mean(),
                 "test_pearson_sd": g["test_pearson_mean"].std(ddof=1) if len(g) > 1 else np.nan,
                 "val_pearson_mean": g["val_pearson_mean"].mean(),
                 "verification_cv_best": g["verification_cv_best"].mean(),
                 "mean_best_epoch": g["mean_best_epoch"].mean(), "mean_epochs_trained": g["mean_epochs_trained"].mean()}
                for name, g in _global_groups(summary)]
        out += [summary, pd.DataFrame(glob)]
    return pd.concat(out, ignore_index=True)


def final_test_protocols(test: pd.DataFrame) -> pd.DataFrame:
    """``tpe_test_protocols.csv``: test con B1–B5 frente a B1–B4 + parada en B5, apareado por sujeto, finalista y
    semilla (la misma semilla da la misma inicialización); filas globales por arquitectura y para las elegidas."""
    done = _completed(test)
    if done.empty or done["protocol"].nunique() < len(TEST_PROTOCOLS):
        return pd.DataFrame()
    wide = (done.pivot_table(index=["subject", "architecture_id", "chosen", "seed_base"], columns="protocol",
                             values="test_pearson_macro", sort=False)
            .reset_index().dropna(subset=list(TEST_PROTOCOLS)))
    if wide.empty:
        return pd.DataFrame()
    es, full = TEST_PROTOCOLS
    wide["delta"] = wide[full] - wide[es]
    rows = [{"subject": subject, "architecture_id": arch, "chosen": bool(g["chosen"].iloc[0]), "n_pairs": len(g),
             f"test_{es}": g[es].mean(), f"test_{full}": g[full].mean(), "delta_mean": g["delta"].mean(),
             "delta_sd": g["delta"].std(ddof=1) if len(g) > 1 else np.nan, "n_positive": int((g["delta"] > 0).sum())}
            for (subject, arch), g in wide.groupby(["subject", "architecture_id"], sort=False)]
    summary = pd.DataFrame(rows)
    glob = [{"subject": "global", "architecture_id": name, "chosen": name == "elegida", "n_pairs": int(g["n_pairs"].sum()),
             f"test_{es}": g[f"test_{es}"].mean(), f"test_{full}": g[f"test_{full}"].mean(),
             "delta_mean": g["delta_mean"].mean(), "delta_sd": g["delta_mean"].std(ddof=1) if len(g) > 1 else np.nan,
             "n_positive": int(g["n_positive"].sum())}
            for name, g in _global_groups(summary)]
    return pd.concat([summary, pd.DataFrame(glob)], ignore_index=True)


# ============================================================================ ablaciones: retrospectiva (§1.1)


def ablation_retrospective(reference_dir: str | Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La política de parada aplicada a las curvas de validación de A5b y A5 de las ablaciones (50 épocas).

    Son métricas de selección sobre validación reconstruidas de los históricos: no son resultados nuevos ni
    implican pesos guardados (plan §1.1).
    """
    fingers, rows = cfg["data"]["fingers"], []
    for path in sorted(Path(reference_dir).glob("history_*.csv")):
        history = pd.read_csv(path)
        for (arch, subject, fold), g in history.groupby(["variant", "subject", "fold"], sort=False):
            g = g.sort_values("epoch").reset_index(drop=True)
            scores = g["val_pearson_macro"].to_numpy(dtype=float)
            sim = simulate(scores, EarlyStopping.from_config(cfg["training"]["early_stopping"]))
            at_best, last = g.iloc[sim["best_epoch"] - 1], g.iloc[-1]
            rows.append({
                "architecture_id": arch, "subject": subject, "fold": fold, "n_epochs": len(g),
                "pearson_epoch50": float(last["val_pearson_macro"]),
                "pearson_es": sim["best_score"], "best_epoch_es": sim["best_epoch"],
                "epochs_trained_es": sim["epochs_trained"], "stop_reason_es": sim["stop_reason"],
                "pearson_best50": float(scores.max()), "best_epoch50": int(np.argmax(scores)) + 1,
                **{f"pearson_{f}_epoch50": float(last[f"val_pearson_{f}"]) for f in fingers},
                **{f"pearson_{f}_es": float(at_best[f"val_pearson_{f}"]) for f in fingers},
            })
    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, pd.DataFrame()
    cols = ["pearson_epoch50", "pearson_es", "pearson_best50", "epochs_trained_es", "best_epoch_es"]
    summary = []
    for keys, g in [*folds.groupby(["architecture_id", "subject"], sort=False),
                    *[((arch, "todos"), g) for arch, g in folds.groupby("architecture_id", sort=False)],
                    (("ambas", "todos"), folds)]:
        row = {"architecture_id": keys[0], "subject": keys[1], "n_curves": len(g), **g[cols].mean().to_dict()}
        row["diff_es_minus_best50"] = row["pearson_es"] - row["pearson_best50"]
        summary.append(row)
    return folds, pd.DataFrame(summary)


# ============================================================================ comparación (§10)


def comparison_tables(retro: pd.DataFrame, trials: dict[str, list[FrozenTrial]], ver: pd.DataFrame,
                      cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson macro y por dedo por sujeto, arquitectura, fuente y fold; resumen por sujeto y global.

    Fuentes: ablación en la época 50 (histórico), ablación con la parada simulada, warmup de referencia (par 1, con
    la política real), mejor trial TPE y verificación (media de las dos semillas). Las tres primeras usan la misma
    arquitectura k5 L5 f64; solo las dos últimas cambian hiperparámetros.
    """
    fingers, rows = cfg["data"]["fingers"], []
    for _, r in retro.iterrows():
        for source, suffix, macro in (("ablation_epoch50", "epoch50", "pearson_epoch50"),
                                      ("ablation_es_sim", "es", "pearson_es")):
            rows.append({"subject": r["subject"], "architecture_id": r["architecture_id"], "source": source,
                         "trial": None, "fold": r["fold"], "pearson_macro": r[macro],
                         **{f"pearson_{f}": r[f"pearson_{f}_{suffix}"] for f in fingers}})
    for subject, subject_trials in trials.items():
        complete = search_complete(subject_trials, cfg)
        for arch in cfg["search"]["space"]["architecture_id"]:
            reference = [t for t in complete if t.user_attrs.get("warmup") == f"p1_{arch}"]
            mine = [t for t in complete if t.params["architecture_id"] == arch]
            picks = [("warmup_reference", reference[0] if reference else None),
                     ("tpe_best", max(mine, key=lambda t: (t.value, -t.number)) if mine else None)]
            for source, t in picks:
                if t is None:
                    continue
                for fold, rec in _fold_records(t, cfg).items():
                    rows.append({"subject": subject, "architecture_id": arch, "source": source, "trial": t.number,
                                 "fold": fold, "pearson_macro": rec["best_pearson_macro"],
                                 **{f"pearson_{f}": (rec.get("pearson") or {}).get(f) for f in fingers}})
    done = _completed(ver)
    if not done.empty:
        cols = ["best_pearson_macro", *[f"pearson_{f}" for f in fingers]]
        means = done.groupby(["subject", "architecture_id", "source_trial", "fold"], sort=False)[cols].mean()
        for (subject, arch, trial, fold), r in means.iterrows():
            rows.append({"subject": subject, "architecture_id": arch, "source": "verification", "trial": trial,
                         "fold": fold, "pearson_macro": r["best_pearson_macro"],
                         **{f"pearson_{f}": r[f"pearson_{f}"] for f in fingers}})
    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, pd.DataFrame()
    metrics = ["pearson_macro", *[f"pearson_{f}" for f in fingers]]
    per_subject = (folds.groupby(["subject", "architecture_id", "source"], sort=False)
                   .agg(n_folds=("fold", "size"), **{m: (m, "mean") for m in metrics}).reset_index())
    glob = (per_subject.groupby(["architecture_id", "source"], sort=False)
            .agg(n_subjects=("subject", "size"), **{m: (m, "mean") for m in metrics}).reset_index()
            .assign(subject="global"))
    summary = pd.concat([per_subject, glob], ignore_index=True)
    order = {s: i for i, s in enumerate(SOURCE_LABELS)}
    summary = summary.sort_values(["subject", "architecture_id", "source"],
                                  key=lambda s: s.map(order) if s.name == "source" else s, kind="stable")
    return folds, summary.reset_index(drop=True)


# ============================================================================ figuras


def plot_optimization_history(trials_df: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    if trials_df.empty:
        return None
    subjects = [s for s in cfg["data"]["subjects"] if s in set(trials_df["subject"])]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(subjects), figsize=(4.6 * len(subjects), 3.6), squeeze=False,
                                 constrained_layout=True)
        for ax, subject in zip(axes[0], subjects):
            g = trials_df[trials_df["subject"] == subject].sort_values("trial")
            done = g[g["state"] == "COMPLETE"]
            for arch, color in ARCH_COLORS.items():
                d = done[done["architecture_id"] == arch]
                warm = d["source"] == "warmup"
                ax.scatter(d.loc[~warm, "trial"], d.loc[~warm, "objective_cv_best"], s=22, color=color, zorder=3,
                           edgecolors=SURFACE, linewidths=0.8, label=f"{arch} TPE")
                ax.scatter(d.loc[warm, "trial"], d.loc[warm, "objective_cv_best"], s=26, facecolors=SURFACE,
                           edgecolors=color, linewidths=1.4, zorder=3, label=f"{arch} warmup")
                if not d.empty:
                    best = d.set_index("trial")["objective_cv_best"].cummax()
                    ax.step(best.index, best.values, where="post", color=color, linewidth=1.2, alpha=0.8)
            n_fail = int((g["state"] == "FAIL").sum())
            ax.set_title(f"{subject} · {len(done)} COMPLETE" + (f" · {n_fail} FAIL" if n_fail else ""))
            ax.set_xlabel("trial")
        axes[0, 0].set_ylabel("objetivo: media de los mejores Pearson\n(5 folds)")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="línea = mejor acumulado")
        return _save(fig, path)


def plot_params_vs_objective(trials_df: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    done = trials_df[trials_df["state"] == "COMPLETE"] if not trials_df.empty else trials_df
    if done.empty:
        return None
    subjects = [s for s in cfg["data"]["subjects"] if s in set(done["subject"])]
    params = ["learning_rate", "weight_decay", "kernel_size", "encoder_levels", "base_filters"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(subjects), len(params), figsize=(3.0 * len(params), 2.5 * len(subjects)),
                                 squeeze=False, sharey="row", constrained_layout=True)
        for i, subject in enumerate(subjects):
            g = done[done["subject"] == subject]
            for j, param in enumerate(params):
                ax = axes[i, j]
                for k, (arch, color) in enumerate(ARCH_COLORS.items()):
                    d = g[g["architecture_id"] == arch]
                    x = d[param].astype(float)
                    if param not in ("learning_rate", "weight_decay"):
                        x = x + (k - 0.5) * 0.25 * (1 if param != "base_filters" else 16)
                    ax.scatter(x, d["objective_cv_best"], s=16, color=color, alpha=0.8, edgecolors=SURFACE,
                               linewidths=0.6, label=arch)
                if param in ("learning_rate", "weight_decay"):
                    ax.set_xscale("log")
                else:
                    ax.set_xticks(sorted(cfg["search"]["space"][param]))
                if i == len(subjects) - 1:
                    ax.set_xlabel(param)
            axes[i, 0].set_ylabel(f"{subject}\nobjetivo")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return _save(fig, path)


def plot_verification(folds_df: pd.DataFrame, ver: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    done = _completed(ver)
    if done.empty:
        return None
    subjects = [s for s in cfg["data"]["subjects"] if s in set(done["subject"])]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(subjects), figsize=(4.2 * len(subjects), 3.6), squeeze=False,
                                 sharey=True, constrained_layout=True)
        for ax, subject in zip(axes[0], subjects):
            ticks, labels = [], []
            for k, (arch, color) in enumerate(ARCH_COLORS.items()):
                v = done[(done["subject"] == subject) & (done["architecture_id"] == arch)]
                if v.empty:
                    continue
                trial = int(v["source_trial"].iloc[0])
                s = folds_df[(folds_df["subject"] == subject) & (folds_df["trial"] == trial)]
                for x, values, fill in ((2 * k, s["best_pearson_macro"], True), (2 * k + 1, v["best_pearson_macro"], False)):
                    ax.scatter(np.full(len(values), x) + np.linspace(-0.12, 0.12, len(values)), values, s=18,
                               color=color if fill else SURFACE, edgecolors=color, linewidths=1.2, zorder=3)
                    if len(values):
                        ax.hlines(values.mean(), x - 0.25, x + 0.25, color=color, linewidth=2.0, zorder=4)
                ticks += [2 * k, 2 * k + 1]
                labels += [f"{arch}\nbúsqueda (t{trial})", f"{arch}\nverificación"]
            ax.set_xticks(ticks, labels)
            ax.grid(axis="x", visible=False)
            ax.set_title(subject)
        axes[0, 0].set_ylabel("mejor Pearson de validación por fold\n(línea = media)")
        return _save(fig, path)


def plot_reference_curves(reference_dir: str | Path, trials: dict[str, list[FrozenTrial]], cfg: dict[str, Any],
                          path: Path) -> Path | None:
    """Warmup de referencia (par 1) frente a las curvas de las ablaciones: misma arquitectura, LR, L2, semilla e
    inicialización, así que deberían coincidir salvo por el no determinismo de la GPU (mismo tipo de GPU)."""
    histories = [pd.read_csv(p) for p in sorted(Path(reference_dir).glob("history_*.csv"))]
    if not histories or not trials:
        return None
    history = pd.concat(histories, ignore_index=True)
    archs = list(cfg["search"]["space"]["architecture_id"])
    subjects = [s for s in cfg["data"]["subjects"] if s in trials]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(archs), len(subjects), figsize=(4.2 * len(subjects), 3.0 * len(archs)),
                                 squeeze=False, sharex=True, sharey="col", constrained_layout=True)
        for i, arch in enumerate(archs):
            for j, subject in enumerate(subjects):
                ax = axes[i, j]
                old = history[(history["variant"] == arch) & (history["subject"] == subject)]
                for k, (fold, g) in enumerate(old.groupby("fold")):
                    ax.plot(g["epoch"], g["val_pearson_macro"], color=MUTED, linewidth=1.0,
                            label="ablación (50 épocas)" if k == 0 else None)
                ref = [t for t in trials[subject] if t.user_attrs.get("warmup") == f"p1_{arch}"
                       and t.state == TrialState.COMPLETE]
                if ref:
                    for k, rec in enumerate(_fold_records(ref[0], cfg).values()):
                        epochs = np.arange(1, len(rec["curve"]) + 1)
                        ax.plot(epochs, rec["curve"], color=ARCH_COLORS.get(arch, INK), linewidth=1.4,
                                label=f"warmup p1 (t{ref[0].number})" if k == 0 else None)
                        ax.scatter([rec["best_epoch"]], [rec["best_pearson_macro"]], s=14,
                                   color=ARCH_COLORS.get(arch, INK), zorder=3)
                ax.set_title(f"{arch} · {subject}")
                if i == len(archs) - 1:
                    ax.set_xlabel("época")
            axes[i, 0].set_ylabel("Pearson macro de validación")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="punto = mejor checkpoint")
        return _save(fig, path)


def plot_test_scores(test: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    """Pearson de test por semilla (puntos) y media (línea) de A5b y A5 en cada sujeto y protocolo (círculo = B1–B4
    con parada en B5, rombo = B1–B5 con épocas fijas); × = media de verificación."""
    done = _completed(test)
    if done.empty:
        return None
    subjects = [s for s in cfg["data"]["subjects"] if s in set(done["subject"])]
    protocols = [p for p in TEST_PROTOCOLS if p in set(done["protocol"])]
    offsets = dict(zip(protocols, np.linspace(-0.17, 0.17, len(protocols)) if len(protocols) > 1 else [0.0]))
    markers = {"b1b4_es_b5": "o", "b1b5_fixed": "D"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(subjects), figsize=(4.4 * len(subjects), 3.6), squeeze=False,
                                 sharey=True, constrained_layout=True)
        for ax, subject in zip(axes[0], subjects):
            ticks, labels, chosen = [], [], None
            for k, (arch, color) in enumerate(ARCH_COLORS.items()):
                v = done[(done["subject"] == subject) & (done["architecture_id"] == arch)]
                if v.empty:
                    continue
                for protocol in protocols:  # mismo color por arquitectura; el marcador distingue el protocolo
                    values = v.loc[v["protocol"] == protocol, "test_pearson_macro"]
                    if values.empty:
                        continue
                    x = k + offsets[protocol]
                    ax.scatter(np.full(len(values), x) + np.linspace(-0.05, 0.05, len(values)), values, s=20,
                               marker=markers.get(protocol, "o"), color=color, edgecolors=SURFACE, linewidths=0.8,
                               zorder=3, label=f"test, {PROTOCOL_LABELS[protocol]}" if k == 0 else None)
                    ax.hlines(values.mean(), x - 0.1, x + 0.1, color=color, linewidth=2.0, zorder=4)
                ax.scatter([k + 0.38], [v["verification_cv_best"].iloc[0]], marker="x", s=30, color=MUTED, zorder=3,
                           label="verificación (CV)" if k == 0 else None)
                if v["chosen"].iloc[0]:
                    chosen = arch
                ticks.append(k)
                labels.append(f"{arch}{' ★' if v['chosen'].iloc[0] else ''}\n{v['label'].iloc[0].split(' ', 1)[1]}")
            ax.set_xticks(ticks, labels, fontsize=7)
            ax.set_xlim(-0.6, len(ARCH_COLORS) - 0.4)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{subject} · elegida {chosen}" if chosen else subject)
        axes[0, 0].set_ylabel("Pearson macro de test\n(línea = media de semillas)")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="★ = elegida por verificación")
        return _save(fig, path)


def plot_test_trajectories(results_dir: str | Path, test: pd.DataFrame, cfg: dict[str, Any], path: Path,
                           seconds: float = 30.0) -> Path | None:
    """Guante real frente a predicho en test: modelo elegido de cada sujeto, primera semilla, primeros segundos; la
    predicción B1–B5 va en verde discontinuo cuando existe."""
    done = _completed(test)
    done = done[done["chosen"]] if not done.empty else done
    if done.empty:
        return None
    fingers = cfg["data"]["fingers"]
    runs = []  # (sujeto, arquitectura, etiqueta, semilla, {protocolo: fila})
    for subject, g in done.groupby("subject", sort=False):
        base = int(g["seed_base"].min())
        rows = {r["protocol"]: r for _, r in g[g["seed_base"] == base].iterrows()}
        first = next(iter(rows.values()))
        runs.append((subject, first["architecture_id"], first["label"], base, rows))
    styles = {"b1b4_es_b5": "-", "b1b5_fixed": "--"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(fingers), len(runs), figsize=(5.0 * len(runs), 1.5 * len(fingers)),
                                 squeeze=False, sharex="col", constrained_layout=True)
        for j, (subject, arch, label_, base, rows) in enumerate(runs):
            truth_drawn = False
            for protocol in TEST_PROTOCOLS:
                if protocol not in rows:
                    continue
                pred = np.load(Path(results_dir) / TEST_PROTOCOLS[protocol] / subject / arch / f"seed_{base}"
                               / "predictions.npz")
                t = pred["t_seconds"]
                keep = t < t[0] + seconds
                for i in range(len(fingers)):
                    if not truth_drawn:
                        axes[i, j].plot(t[keep], pred["y_true"][i, keep], color=INK, linewidth=1.2, label="real")
                    axes[i, j].plot(t[keep], pred["y_pred"][i, keep], linewidth=1.2, linestyle=styles[protocol],
                                    color=ARCH_COLORS.get(arch, SLOTS[0]) if protocol == "b1b4_es_b5" else SLOTS[2],
                                    label=f"predicho, {PROTOCOL_LABELS[protocol]}")
                truth_drawn = True
            for i, finger in enumerate(fingers):
                r = [rows[p].get(f"test_pearson_{finger}") for p in TEST_PROTOCOLS if p in rows]
                axes[i, j].set_ylabel(f"{finger}\nr = " + " / ".join(f"{v:.2f}" if pd.notna(v) else "—" for v in r),
                                      fontsize=8)
            axes[0, j].set_title(f"{subject} · {label_} · semilla {base}")
            axes[-1, j].set_xlabel("tiempo en test (s)")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return _save(fig, path)


# ============================================================================ resumen


def write_summary(results_dir: str | Path, cfg: dict[str, Any], reference_dir: str | Path) -> dict[str, Any]:
    """Genera ``results/summary/`` (CSV y figuras) con lo que haya: se puede ejecutar a mitad de campaña."""
    results_dir = Path(results_dir)
    out = results_dir / "summary"
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    trials = load_all_trials(results_dir, cfg)
    trials_df, folds_df = trials_table(trials, cfg), folds_table(trials, cfg)
    ver = verification_table(results_dir, cfg)
    paired, paired_summary = verification_paired(ver)
    selected = selected_models(trials_df, ver, cfg)
    retro_folds, retro_summary = ablation_retrospective(reference_dir, cfg)
    comparison_folds, comparison = comparison_tables(retro_folds, trials, ver, cfg)
    test = final_test_table(results_dir, cfg)
    tables = {
        "tpe_trials.csv": trials_df, "tpe_folds.csv": folds_df, "tpe_verification.csv": ver,
        "tpe_verification_paired.csv": paired, "tpe_verification_paired_summary.csv": paired_summary,
        "tpe_selected_models.csv": selected, "ablation_retrospective.csv": retro_summary,
        "ablation_retrospective_folds.csv": retro_folds, "tpe_comparison.csv": comparison,
        "tpe_comparison_folds.csv": comparison_folds,
        "tpe_test.csv": test, "tpe_test_summary.csv": final_test_summary(test),
        "tpe_test_protocols.csv": final_test_protocols(test),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    figs = {
        "optimization_history": plot_optimization_history(trials_df, cfg, figures / "optimization_history.png"),
        "params_vs_objective": plot_params_vs_objective(trials_df, cfg, figures / "params_vs_objective.png"),
        "verification": plot_verification(folds_df, ver, cfg, figures / "verification.png"),
        "reference_curves": plot_reference_curves(reference_dir, trials, cfg, figures / "reference_curves.png"),
        "test_scores": plot_test_scores(test, cfg, figures / "test_scores.png"),
        "test_trajectories": plot_test_trajectories(results_dir, test, cfg, figures / "test_trajectories.png"),
    }
    states = trials_df.groupby(["subject", "state"]).size().unstack(fill_value=0).to_dict("index") \
        if not trials_df.empty else {}
    return {"trials": states, "verification_completed": int(len(_completed(ver))),
            "test_completed": int(len(_completed(test))), "figures": {k: str(v) for k, v in figs.items() if v}}


# ============================================================================ monitoreo (notebook)


def _age_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    h, rem = divmod(int(seconds), 3600)
    return f"{h} h {rem // 60:02d} min"


def search_progress(results_dir: str | Path, cfg: dict[str, Any], subjects: list[str]) -> list[dict[str, Any]]:
    """Estado de cada sujeto a partir de los trial.json/status.json (no abre el estudio)."""
    n_target, out = cfg["search"]["n_trials_per_subject"], []
    for subject in subjects:
        trials = [(p.parent, read_json(p) or {}) for p in
                  sorted((Path(results_dir) / "trials" / subject).glob("trial_*/trial.json"))]
        done = [t for _, t in trials if t.get("state") == "COMPLETE"]
        durations = [(datetime.fromisoformat(t["finished_at"]) - datetime.fromisoformat(t["started_at"]))
                     .total_seconds() for t in done if t.get("finished_at")]
        best = {}
        for t in done:
            arch = t["params"]["architecture_id"]
            if arch not in best or t["value"] > best[arch]["value"]:
                best[arch] = t
        running = None
        for trial_dir, t in trials:
            if t.get("state") == "RUNNING":
                status_files = sorted(trial_dir.glob("B*/status.json"), key=lambda p: p.stat().st_mtime)
                fold_status = read_json(status_files[-1]) if status_files else None
                running = {"trial": t, "fold": status_files[-1].parent.name if status_files else None,
                           "status": fold_status or {}, "dir": trial_dir}
        remaining = max(n_target - len(done), 0)
        eta = np.mean(durations) * remaining if durations else None
        out.append({"subject": subject, "complete": len(done), "fail": sum(t.get("state") == "FAIL" for _, t in trials),
                    "target": n_target, "best": best, "running": running, "eta_seconds": eta})
    return out


def monitor(results_dir: str | Path, cfg: dict[str, Any], subjects: list[str] | None = None, refresh: int = 60,
            log_path: str | Path | None = None, max_updates: int | None = None, verification: bool = False,
            final_test: bool = False, protocols: list[str] | None = None) -> None:
    """Panel reejecutable: progreso, ETA, trial y fold en curso, mejores valores y cola del log.

    Con ``final_test``, ``protocols`` limita el panel a esos protocolos de test (por defecto, todos). Detenerlo no
    detiene el entrenamiento.
    """
    from IPython.display import clear_output

    results_dir = Path(results_dir)
    subjects = subjects or list(cfg["data"]["subjects"])
    updates = 0
    try:
        while True:
            clear_output(wait=True)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
            for runner in sorted((results_dir / "runners").glob("*.json")):
                info = read_json(runner) or {}
                age = _age_seconds(info.get("heartbeat"))
                print(f"runner {runner.stem}: último latido hace {age:.0f} s · {info.get('current')}"
                      if age is not None else f"runner {runner.stem}")
            if final_test:
                _print_final_test(results_dir, cfg, subjects, protocols)
            elif verification:
                _print_verification(results_dir, cfg, subjects)
            else:
                for p in search_progress(results_dir, cfg, subjects):
                    best = " · ".join(f"{arch} {t['value']:.4f} (t{t['trial']})" for arch, t in sorted(p["best"].items()))
                    print(f"{p['subject']}: COMPLETE {p['complete']}/{p['target']} · FAIL {p['fail']} · "
                          f"mejor {best or '—'} · ETA {_fmt_duration(p['eta_seconds'])}")
                    if p["running"]:
                        t, s = p["running"]["trial"], p["running"]["status"]
                        print(f"  ▶ t{t['trial']:03d} {t['label']} lr={t['params']['learning_rate']:.3g} "
                              f"wd={t['params']['weight_decay']:.3g} · {p['running']['fold']} época "
                              f"{s.get('epoch')}/{s.get('max_epochs')} · r {s.get('last_val_pearson', float('nan')):.4f}"
                              f" · mejor {s.get('best_val_pearson', float('nan')):.4f} (ép. {s.get('best_epoch')})"
                              f" · espera {s.get('wait')}")
                        hist = p["running"]["dir"] / p["running"]["fold"] / "history.csv" if p["running"]["fold"] else None
                        if hist is not None and hist.exists():
                            _plot_live_history(pd.read_csv(hist), f"{p['subject']} t{t['trial']:03d} {p['running']['fold']}")
            log_file = Path(log_path) if log_path else _latest_log(results_dir)
            if log_file and log_file.exists():
                print(f"— últimas líneas de {log_file.name} —")
                print("\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]))
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
            time.sleep(refresh)
    except KeyboardInterrupt:
        print("monitor detenido (el entrenamiento en segundo plano continúa)")


def _print_verification(results_dir: Path, cfg: dict[str, Any], subjects: list[str]) -> None:
    ver = verification_table(results_dir, cfg)
    expected = 2 * len(cfg["verification"]["seed_bases"]) * len(cfg["folds"]["names"])
    for subject in subjects:
        v = ver[ver["subject"] == subject] if not ver.empty else ver
        if v.empty:
            print(f"{subject}: verificación sin empezar")
            continue
        done = _completed(v)
        means = done.groupby("architecture_id")["best_pearson_macro"].mean().to_dict() if not done.empty else {}
        print(f"{subject}: {len(done)}/{expected} corridas · media " +
              " · ".join(f"{a} {m:.4f}" for a, m in means.items()) +
              " · en curso: " + ", ".join(f"{r.architecture_id} s{r.seed_base} {r.fold}"
                                          for r in v[v["state"] == "running"].itertuples()))


def _print_final_test(results_dir: Path, cfg: dict[str, Any], subjects: list[str],
                      protocols: list[str] | None = None) -> None:
    test = final_test_table(results_dir, cfg)
    for protocol, folder in TEST_PROTOCOLS.items():
        if protocols is not None and protocol not in protocols:
            continue
        print(f"— test {PROTOCOL_LABELS[protocol]} —")
        for subject in subjects:
            t = test[(test["protocol"] == protocol) & (test["subject"] == subject)] if not test.empty else test
            if t.empty:
                print(f"{subject}: sin empezar")
                continue
            done = _completed(t)
            means = done.groupby("architecture_id")["test_pearson_macro"].mean().to_dict() if not done.empty else {}
            chosen = t.loc[t["chosen"], "architecture_id"].iloc[0]
            print(f"{subject}: {len(done)}/{len(t)} corridas · elegida {chosen} · test " +
                  (" · ".join(f"{a} {m:.4f}" for a, m in means.items()) or "—"))
            for r in t[t["state"] == "running"].itertuples():
                run_dir = results_dir / folder / subject / r.architecture_id / f"seed_{r.seed_base}"
                s = read_json(run_dir / "status.json") or {}
                line = f"  ▶ {r.architecture_id} s{r.seed_base} época {s.get('epoch')}/{s.get('max_epochs')}"
                if "best_val_pearson" in s:
                    line += (f" · r_val {s.get('last_val_pearson', float('nan')):.4f} · mejor "
                             f"{s['best_val_pearson']:.4f} (ép. {s.get('best_epoch')}) · espera {s.get('wait')}")
                print(line)
                if (run_dir / "history.csv").exists():
                    _plot_live_history(pd.read_csv(run_dir / "history.csv"),
                                       f"{subject} test {PROTOCOL_LABELS[protocol]} {r.architecture_id} s{r.seed_base}")


def _latest_log(results_dir: Path) -> Path | None:
    logs = sorted((results_dir / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _plot_live_history(history: pd.DataFrame, title: str) -> None:
    if "val_pearson_macro" not in history:  # épocas fijas (test con B1–B5): solo hay pérdida de train
        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(4.5, 2.6), constrained_layout=True)
            ax.plot(history["epoch"], history["train_loss"], color=SLOTS[0])
            ax.set_title("pérdida de train")
            ax.set_xlabel("época")
            fig.suptitle(title, fontsize=10)
            plt.show()
        return
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), constrained_layout=True)
        axes[0].plot(history["epoch"], history["train_loss"], color=SLOTS[0], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], color=SLOTS[1], label="validación")
        axes[0].set_title("pérdida")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["val_pearson_macro"], color=SLOTS[0], label="época")
        axes[1].plot(history["epoch"], history["best_val_pearson_macro"], color=INK, linewidth=1.2, label="mejor")
        axes[1].set_title("Pearson macro de validación")
        axes[1].legend()
        for ax in axes:
            ax.set_xlabel("época")
        fig.suptitle(title, fontsize=10)
        plt.show()


def show_images(paths: list[str | Path]) -> None:
    from IPython.display import Image, display

    for path in paths:
        if path and Path(path).exists():
            display(Image(filename=str(path)))
