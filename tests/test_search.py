"""Estudio Optuna con un runner de folds falso: warmups, reanudación, fallos, verificación y reportes."""

from __future__ import annotations

import copy
import os
import socket

import numpy as np
import pytest
from optuna.trial import TrialState

from ecog_tpe.config import REPO_ROOT, load_config
from ecog_tpe.reporting import write_summary
from ecog_tpe.runs import RunnerLock, now, write_json
from ecog_tpe.search import load_trials, run_search
from ecog_tpe.space import warmup_trials
from ecog_tpe.verification import run_verification

FOLDS = ["B1", "B2"]


class FakeData:
    def fold(self, fold):
        return fold  # el runner falso no necesita características

    def release(self):
        pass


class FakeRunner:
    """Sustituye a run_fold: puntuación determinista según los hiperparámetros; escribe status/metrics como él."""

    def __init__(self, fail=None, interrupt_at=None):
        self.calls, self.fail, self.interrupt_at = [], fail, interrupt_at

    def __call__(self, data, spec, params, seed, cfg, device, out_dir, stopper, tag="", max_batches=None,
                 on_epoch=None):
        self.calls.append((dict(params), seed, data))
        if self.interrupt_at == len(self.calls):
            raise KeyboardInterrupt
        if self.fail is not None and self.fail(params, len(self.calls)):
            raise RuntimeError("fallo simulado")
        lr, wd = np.log10(params["learning_rate"]), np.log10(params["weight_decay"])
        score = (0.6 + 0.05 * (params["architecture_id"] == "A5") - 0.02 * (lr + 3.7) ** 2 - 0.002 * (wd + 6) ** 2
                 + 0.001 * params["kernel_size"] + 0.0001 * (seed % 7))
        history = [{"epoch": e, "val_pearson_macro": score - 0.01 * abs(e - 3)} for e in range(1, 6)]
        metrics = {"seed": seed, "best_epoch": 3, "epochs_trained": 5, "stop_reason": "patience_exhausted",
                   "best_val_pearson_macro": score, "pearson_macro": score, "valid": True,
                   "pearson": {f: score for f in cfg["data"]["fingers"]}, "mse_macro": 0.01, "ccc_macro": 0.5,
                   "n_parameters": 1000, "train_seconds": 0.01}
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "status.json", {"status": "completed", "seed": seed})
        return metrics, history


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def search(cfg, results, runner, n_trials=None):
    return run_search(cfg, "S1", "sin_datos", results, device="cpu", work_dir=results.parent / "work",
                      n_trials=n_trials, folds=FOLDS, data=FakeData(), runner=runner)


def complete(results):
    return sorted((t for t in load_trials(results, "S1") if t.state == TrialState.COMPLETE), key=lambda t: t.number)


def test_new_study_enqueues_the_ten_warmups_once(cfg, tmp_path):
    results = tmp_path / "results"
    search(cfg, results, FakeRunner(), n_trials=3)
    trials = load_trials(results, "S1")
    warm = sorted((t for t in trials if t.user_attrs.get("source") == "warmup"), key=lambda t: t.number)
    assert [t.user_attrs["warmup"] for t in warm] == [key for key, _ in warmup_trials(cfg)]
    assert sum(t.state == TrialState.COMPLETE for t in trials) == 3
    assert sum(t.state == TrialState.WAITING for t in trials) == 7

    search(cfg, results, FakeRunner(), n_trials=12)  # reanuda desde el zip: no vuelve a encolarlos
    trials = load_trials(results, "S1")
    assert sum(t.user_attrs.get("source") == "warmup" for t in trials) == 10
    done = complete(results)
    assert len(done) == 12
    assert [t.params for t in done[:10]] == [params for _, params in warmup_trials(cfg)]
    assert all(t.user_attrs["source"] == "tpe" for t in done[10:])
    first = done[0].user_attrs
    assert first["fold_best_pearson"] == [first["fold_B1"]["best_pearson_macro"], first["fold_B2"]["best_pearson_macro"]]
    assert first["fold_B1"]["curve"] and first["fold_B1"]["seed"] == 42 and first["fold_B2"]["seed"] == 43


def test_resume_reproduces_an_uninterrupted_study(cfg, tmp_path):
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    search(cfg, straight, FakeRunner(), n_trials=14)
    search(cfg, resumed, FakeRunner(), n_trials=12)
    search(cfg, resumed, FakeRunner(), n_trials=14)  # el sampler serializado continúa el mismo RNG
    assert [t.params for t in complete(straight)] == [t.params for t in complete(resumed)]
    assert [t.user_attrs["sampler_restored"] for t in complete(resumed)][-2:] == [True, True]


def test_failed_warmup_is_retried_and_failures_do_not_count(cfg, tmp_path):
    results = tmp_path / "results"
    search(cfg, results, FakeRunner(fail=lambda params, call: call == 3), n_trials=11)  # 1.er fold del trial 1
    trials = load_trials(results, "S1")
    failed = [t for t in trials if t.state == TrialState.FAIL]
    assert len(failed) == 1 and failed[0].user_attrs["warmup"] == "p1_A5"
    assert "fallo simulado" in failed[0].user_attrs["error"]
    retry = [t for t in trials if t.user_attrs.get("warmup") == "p1_A5" and t.state == TrialState.COMPLETE]
    assert len(retry) == 1 and retry[0].user_attrs["attempt"] == 2
    assert len(complete(results)) == 11
    trial_dir = results / "trials" / "S1" / f"trial_{failed[0].number:03d}"
    assert (trial_dir / "error.txt").exists()


def test_stops_after_consecutive_failures(cfg, tmp_path):
    results = tmp_path / "results"
    with pytest.raises(RuntimeError, match="3 trials FAIL seguidos"):
        search(cfg, results, FakeRunner(fail=lambda params, call: True), n_trials=5)
    assert sum(t.state == TrialState.FAIL for t in load_trials(results, "S1")) == 3  # publicados para diagnóstico


def test_refuses_trial_folders_without_published_study(cfg, tmp_path):
    results = tmp_path / "results"
    (results / "trials" / "S1" / "trial_000").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="estudio nuevo"):
        search(cfg, results, FakeRunner(), n_trials=2)


def test_interrupted_trial_is_repeated_with_the_same_parameters(cfg, tmp_path):
    results = tmp_path / "results"
    with pytest.raises(KeyboardInterrupt):
        search(cfg, results, FakeRunner(interrupt_at=5), n_trials=6)  # 1.er fold del trial 2
    assert len(complete(results)) == 2 and len(load_trials(results, "S1")) == 10  # el intento no se publica
    assert (results / "trials" / "S1" / "trial_002").exists()

    search(cfg, results, FakeRunner(), n_trials=6)
    trials = load_trials(results, "S1")
    assert not any(t.state == TrialState.FAIL for t in trials)
    assert complete(results)[2].params == warmup_trials(cfg)[2][1]
    assert len(list((results / "trials" / "S1" / "_interrupted").glob("trial_002_*"))) == 1


def test_verification_and_report(cfg, tmp_path):
    small = copy.deepcopy(cfg)
    small["search"]["n_trials_per_subject"] = 12
    results = tmp_path / "results"
    search(small, results, FakeRunner())
    runner = FakeRunner()
    selection = run_verification(small, "S1", "sin_datos", results, device="cpu", folds=FOLDS, data=FakeData(),
                                 runner=runner)
    done = complete(results)
    for arch, info in selection["finalists"].items():
        best = max((t for t in done if t.params["architecture_id"] == arch), key=lambda t: (t.value, -t.number))
        assert info["trial"] == best.number
    assert len(runner.calls) == 2 * 2 * len(FOLDS)
    assert sorted({seed for _, seed, _ in runner.calls}) == [1042, 1043, 2042, 2043]

    again = FakeRunner()  # reanudar no repite corridas completadas
    run_verification(small, "S1", "sin_datos", results, device="cpu", folds=FOLDS, data=FakeData(), runner=again)
    assert again.calls == []

    info = write_summary(results, small, REPO_ROOT / "reference" / "ablaciones")
    summary = results / "summary"
    for name in ("tpe_trials.csv", "tpe_folds.csv", "tpe_verification.csv", "tpe_selected_models.csv",
                 "tpe_comparison.csv", "ablation_retrospective.csv"):
        assert (summary / name).stat().st_size > 1, name
    assert set(info["figures"]) == {"optimization_history", "params_vs_objective", "verification",
                                    "reference_curves"}
    import pandas as pd

    selected = pd.read_csv(summary / "tpe_selected_models.csv").iloc[0]
    assert selected["chosen_architecture"] == "A5"  # la superficie falsa favorece A5
    assert selected["median_best_epoch"] == 3 and not selected["verification_complete"]  # solo 2 de 5 folds
    trials = pd.read_csv(summary / "tpe_trials.csv")
    assert (trials["state"] == "COMPLETE").sum() == 12 and len(trials) == 12


@pytest.mark.skipif(os.name != "posix", reason="la comprobación de PID solo se hace en POSIX (Colab)")
def test_second_runner_on_the_same_machine_is_refused(tmp_path):
    runners = tmp_path / "runners"
    write_json(runners / f"{socket.gethostname()}-{os.getppid()}.json", {"pid": os.getppid(), "heartbeat": now()})
    import torch

    with pytest.raises(RuntimeError, match="runner activo"):
        RunnerLock(tmp_path, torch.device("cpu")).acquire()
