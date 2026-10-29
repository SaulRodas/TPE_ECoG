"""Política de early stopping del plan §4.1 y su simulación retrospectiva sobre las ablaciones (§1.1)."""

from __future__ import annotations

import numpy as np
import pytest

from ecog_tpe.config import REPO_ROOT, load_config
from ecog_tpe.reporting import ablation_retrospective
from ecog_tpe.training import EarlyStopping, simulate


def run(scores, **kwargs):
    return simulate(scores, EarlyStopping(**kwargs))


def test_monotone_improvement_trains_until_epoch_50():
    result = run(np.linspace(0.1, 0.9, 60))
    assert (result["best_epoch"], result["epochs_trained"], result["stop_reason"]) == (50, 50, "max_epochs")


def test_peak_before_minimum_first_stop_is_epoch_25():
    result = run([0.1, 0.2, 0.3, 0.4, 0.9] + [0.5] * 45)
    assert (result["best_epoch"], result["epochs_trained"], result["stop_reason"]) == (5, 25, "patience_exhausted")
    # Mejoras durante las épocas 1–15 no permiten parar antes de la 25.
    assert run([0.1, 0.9] + [0.2] * 48)["epochs_trained"] == 25


def test_improvement_after_minimum_restarts_patience():
    scores = list(np.linspace(0.1, 0.5, 15)) + [0.4] * 4 + [0.6] + [0.5] * 30  # nuevo máximo en la época 20
    result = run(scores)
    assert (result["best_epoch"], result["epochs_trained"], result["stop_reason"]) == (20, 30, "patience_exhausted")


def test_tie_is_not_an_improvement():
    result = run([0.1] * 15 + [0.7] * 35)  # máximo en la 16; repetirlo no reinicia la paciencia
    assert (result["best_epoch"], result["epochs_trained"]) == (16, 26)
    assert run([0.5] * 50)["best_epoch"] == 1


def test_patience_exhausted_exactly_at_epoch_50():
    result = run(list(np.linspace(0.1, 0.8, 40)) + [0.7] * 10)
    assert (result["best_epoch"], result["epochs_trained"], result["stop_reason"]) == (40, 50, "patience_exhausted")


def test_non_finite_score_raises():
    stopper = EarlyStopping()
    stopper.update(1, 0.3)
    with pytest.raises(ValueError, match="época 2"):
        stopper.update(2, float("nan"))


def test_retrospective_reproduces_plan_table():
    folds, summary = ablation_retrospective(REPO_ROOT / "reference" / "ablaciones", load_config())
    assert len(folds) == 30 and (folds["n_epochs"] == 50).all()
    s = summary.set_index(["architecture_id", "subject"])
    expected = {"A5b": (0.677915, 0.711371, 34.0), "A5": (0.683715, 0.733021, 30.6)}  # plan §1.1
    for arch, (epoch50, es, epochs) in expected.items():
        row = s.loc[(arch, "todos")]
        assert row["pearson_epoch50"] == pytest.approx(epoch50, abs=5e-7)
        assert row["pearson_es"] == pytest.approx(es, abs=5e-7)
        assert row["epochs_trained_es"] == pytest.approx(epochs, abs=1e-9)
    both = s.loc[("ambas", "todos")]
    assert both["epochs_trained_es"] == pytest.approx(32.3, abs=1e-9)
    assert both["pearson_es"] == pytest.approx(0.7222, abs=5e-5)
    assert both["pearson_best50"] == pytest.approx(0.7254, abs=5e-5)
    assert both["diff_es_minus_best50"] == pytest.approx(-0.0032, abs=5e-5)
