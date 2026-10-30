#!/usr/bin/env python
"""Genera results/summary/ (CSV y figuras) a partir de los estudios publicados, la verificación y el test.

Se puede ejecutar en cualquier momento: con la campaña a medias reporta lo que haya. La simulación retrospectiva
sobre las ablaciones (reference/ablaciones) no necesita ningún resultado de TPE.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # logs en UTF-8 también en Windows

import pandas as pd  # noqa: E402

from ecog_tpe.config import load_config, resolve_path  # noqa: E402
from ecog_tpe.reporting import write_summary  # noqa: E402


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reference-dir", default="reference/ablaciones",
                        help="históricos por época de A5b/A5 de las ablaciones (history_*.csv)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_dir = resolve_path(args.results_dir)
    info = write_summary(results_dir, cfg, resolve_path(args.reference_dir))
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    summary = results_dir / "summary"

    pd.set_option("display.width", 160)
    retro = _read(summary / "ablation_retrospective.csv")
    if not retro.empty:
        print("\nAblaciones con la política de parada simulada (plan §1.1; media de las curvas sujeto-fold):")
        cols = ["pearson_epoch50", "pearson_es", "pearson_best50", "diff_es_minus_best50", "epochs_trained_es"]
        print(retro[retro["subject"] == "todos"].set_index("architecture_id")[cols].round(6).to_string())
    comparison = _read(summary / "tpe_comparison.csv")
    glob = comparison[comparison["subject"] == "global"] if not comparison.empty else comparison
    if not glob.empty:
        print("\nPearson macro global (media de sujetos con igual peso):")
        print(glob.pivot_table(index="source", columns="architecture_id", values="pearson_macro",
                               sort=False).round(4).to_string())
    selected = _read(summary / "tpe_selected_models.csv")
    if not selected.empty:
        print("\nModelo elegido por sujeto (media de las dos semillas de verificación):")
        print(selected.round(6).to_string(index=False))
    test = _read(summary / "tpe_test_summary.csv")
    if not test.empty:
        print("\nTest oficial (media ± sd de las semillas; en 'global', media y sd entre sujetos):")
        cols = ["subject", "architecture_id", "chosen", "label", "n_seeds", "test_pearson_mean", "test_pearson_sd",
                "val_pearson_mean", "verification_cv_best", "mean_best_epoch"]
        print(test[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
