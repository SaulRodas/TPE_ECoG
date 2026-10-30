#!/usr/bin/env python
"""Búsqueda TPE por sujeto, verificación de finalistas, evaluación final en test y comprobaciones previas
(capacidad y smoke).

Ejemplos:
    python scripts/run_tpe.py --capacity                  # 16 combinaciones en la GPU (plan §3.2)
    python scripts/run_tpe.py --smoke                     # S1/B1, warmups del par 1, 2 épocas × 2 batches → results/_smoke
    python scripts/run_tpe.py --subjects S1               # estudio de S1 hasta 32 trials COMPLETE (reanudable)
    python scripts/run_tpe.py --subjects S1 S2 S3         # varios sujetos, uno detrás de otro
    python scripts/run_tpe.py --subjects S1 --verify      # finalistas A5b/A5 × 2 semillas × 5 folds (reanudable)
    python scripts/run_tpe.py --subjects S1 --test --test-labels-dir RUTA   # finalistas × 3 semillas en el test oficial
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # antes de inicializar CUDA
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # logs en UTF-8 también en Windows

from ecog_tpe.config import load_config, resolve_path  # noqa: E402

EPOCHS_PER_FOLD = 32.3  # media de la simulación retrospectiva (plan §1.1); solo para estimar tiempos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="ruta del YAML (por defecto configs/tpe.yaml)")
    parser.add_argument("--data-dir", default=None, help="carpeta con sub{1,2,3}_comp.mat (por defecto data/raw)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--work-dir", default=None,
                        help="disco local para el SQLite de trabajo (por defecto, la carpeta temporal del sistema)")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--trials", type=int, default=None, help="trials COMPLETE por sujeto (por defecto, la config)")
    parser.add_argument("--verify", action="store_true", help="verificación de finalistas en lugar de búsqueda")
    parser.add_argument("--test", action="store_true",
                        help="evaluación final en el test oficial (requiere la verificación completa del sujeto)")
    parser.add_argument("--test-labels-dir", default=None,
                        help="carpeta con sub{1,2,3}_testlabels.mat (por defecto, la de --data-dir)")
    parser.add_argument("--capacity", action="store_true", help="solo la comprobación de memoria/tiempo en la GPU")
    parser.add_argument("--smoke", action="store_true", help="cadena completa mínima en <results>/_smoke")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("ecog_tpe")
    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else resolve_path(cfg["data"]["raw_dir"])
    results_dir = resolve_path(args.results_dir)
    subjects = args.subjects or []
    unknown = [s for s in subjects if s not in cfg["data"]["subjects"]]
    if unknown:
        parser.error(f"sujeto desconocido: {unknown}")

    try:
        if args.capacity:
            from ecog_tpe.search import run_capacity

            table = run_capacity(cfg, results_dir, args.device)
            print(table.drop(columns=["subject", "batch_size"]).round(4).to_string(index=False))
            ok = table[table["status"] == "ok"]
            if len(ok) < len(table):
                print("⚠ hay combinaciones que no caben: ajustar el espacio ANTES del primer estudio (plan §3.2)")
            if not ok.empty:
                per_trial = ok["est_epoch_seconds"].mean() * EPOCHS_PER_FOLD * len(cfg["folds"]["names"])
                hours = per_trial * cfg["search"]["n_trials_per_subject"] / 3600
                print(f"estimación orientativa: ~{per_trial / 60:.0f} min por trial, ~{hours:.1f} h de GPU por sujeto "
                      f"(con {EPOCHS_PER_FOLD} épocas por fold de media; los nuevos LR pueden cambiarlo)")
            return 0
        if args.smoke:
            from ecog_tpe.search import run_smoke

            info = run_smoke(cfg, data_dir, results_dir, args.device, args.work_dir, subjects[0] if subjects else None)
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return 0
        if not subjects:
            parser.error("indica --subjects (o usa --capacity / --smoke)")
        if args.test:
            from ecog_tpe.final_test import run_final_test

            labels_dir = Path(args.test_labels_dir) if args.test_labels_dir else data_dir
            for subject in subjects:
                run_final_test(cfg, subject, data_dir, labels_dir, results_dir, args.device)
        elif args.verify:
            from ecog_tpe.verification import run_verification

            for subject in subjects:
                run_verification(cfg, subject, data_dir, results_dir, args.device)
        else:
            from ecog_tpe.search import run_search

            for subject in subjects:
                run_search(cfg, subject, data_dir, results_dir, args.device, args.work_dir, n_trials=args.trials)
    except KeyboardInterrupt:
        log.warning("detenido; se reanuda al volver a lanzarlo")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
