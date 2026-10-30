"""Evaluación final en test (B1–B4 + parada en B5, y B1–B5 con épocas fijas): lectura del test, ausencia de fugas,
geometría, entrenamiento sin validación, planes congelados, flujo, comparación de protocolos y pesos."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import savemat

from ecog_tpe import final_test, training
from ecog_tpe.config import REPO_ROOT, load_config
from ecog_tpe.evaluation import overlap_add
from ecog_tpe.final_test import (
    SubjectTestData,
    freeze_full_plan,
    freeze_plan,
    load_model,
    load_test,
    predict_segment,
    run_final_test,
    save_model,
)
from ecog_tpe.models import EncoderDecoder1D
from ecog_tpe.preprocessing import val_window_starts
from ecog_tpe.reporting import write_summary
from ecog_tpe.runs import write_json
from ecog_tpe.verification import run_verification
from test_pipeline import synthetic_recording
from test_search import FakeData, FakeRunner, search
from test_training import TCFG, TINY, tiny_features

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def small_cfg(cfg):
    """S1 con 3 canales crudos y el 1 excluido, para datos sintéticos pequeños."""
    small = copy.deepcopy(cfg)
    small["data"]["subjects"]["S1"].update(n_channels_raw=3, excluded=[1], n_channels_used=2)
    return small


def synthetic_test(cfg, seed=1):
    """200 s de test con la misma estructura que el desarrollo sintético; el guante, suave a 1000 Hz."""
    ecog, dg = synthetic_recording(cfg, n_channels=3, seed=seed)
    n = cfg["final_test"]["n_samples"]
    t = np.arange(n) / cfg["data"]["fs_ecog"]
    smooth = np.stack([np.sin(2 * np.pi * (0.2 + 0.05 * k) * t) for k in range(5)], axis=1)
    return ecog[:n], smooth


def test_load_test_reads_both_files(small_cfg, tmp_path):
    ecog, dg = synthetic_test(small_cfg)
    savemat(tmp_path / "sub1_comp.mat", {"test_data": ecog})
    savemat(tmp_path / "sub1_testlabels.mat", {"test_dg": dg})
    out = load_test(tmp_path, tmp_path, "S1", small_cfg)
    np.testing.assert_array_equal(out["ecog"], ecog)
    assert out["glove"].shape == (5, 20000)
    np.testing.assert_array_equal(out["glove"], dg[::10].T)  # instante de cada columna, t = c/100 s

    savemat(tmp_path / "sub1_testlabels.mat", {"test_dg": dg[:-40]})
    with pytest.raises(ValueError, match="test_dg"):
        load_test(tmp_path, tmp_path, "S1", small_cfg)
    with pytest.raises(FileNotFoundError):
        load_test(tmp_path, tmp_path / "no_existe", "S1", small_cfg)


def test_test_data_never_changes_what_is_fitted(small_cfg, monkeypatch):
    dev_ecog, dev_dg = synthetic_recording(small_cfg, n_channels=3)
    test_ecog, test_dg = synthetic_test(small_cfg)
    monkeypatch.setattr(final_test, "load_subject", lambda *a: {"ecog": dev_ecog, "dg": dev_dg})

    def build(ecog):
        monkeypatch.setattr(final_test, "load_test",
                            lambda *a: {"ecog": ecog, "glove": np.ascontiguousarray(test_dg[::10].T)})
        return SubjectTestData(small_cfg, "dev", "labels", "S1", CPU)

    data = build(test_ecog)
    altered = test_ecog * 100 + 1e4
    altered[100000, :] = 1e8
    data2 = build(altered)

    # Todo lo ajustado sale de B1–B4: idéntico bit a bit aunque el test cambie por completo.
    for key, value in data.preprocessing.items():
        np.testing.assert_array_equal(value, data2.preprocessing[key])
    np.testing.assert_array_equal(data.fold.feats.X, data2.fold.feats.X)
    np.testing.assert_array_equal(data.fold.feats.train_index, data2.fold.feats.train_index)
    assert not np.array_equal(data.test.X, data2.test.X)
    assert data.fold.feats.val_block == 4 and 4 not in set(data.fold.feats.train_index[:, 0])  # B5 solo valida
    np.testing.assert_array_equal(data.preprocessing["kept_channels"], [0, 2])
    assert data.info["excluded_channels"] == [1] and data2.info["clip_fraction_test"] > data.info["clip_fraction_test"]

    # Geometría del segmento de test: 125 columnas descartadas por extremo y retardo de 2 columnas.
    seg, w = data.test, small_cfg["preprocessing"]["window"]
    assert (seg.lo, seg.hi) == (125, 19875) and seg.X.shape == (2 * 40, 20000)
    assert seg.starts[0] == 125 and seg.starts[-1] == 19875 - w and np.all(np.diff(seg.starts) <= 64)
    np.testing.assert_array_equal(seg.y_true, test_dg[::10].T[:, 127:19877])
    assert seg.t_seconds[0] == pytest.approx(1.27) and len(seg.t_seconds) == 19750


def test_full_development_trains_on_every_block(small_cfg, monkeypatch):
    dev_ecog, dev_dg = synthetic_recording(small_cfg, n_channels=3)
    test_ecog, test_dg = synthetic_test(small_cfg)
    monkeypatch.setattr(final_test, "load_subject", lambda *a: {"ecog": dev_ecog, "dg": dev_dg})

    def build(ecog, full=True):
        monkeypatch.setattr(final_test, "load_test",
                            lambda *a: {"ecog": ecog, "glove": np.ascontiguousarray(test_dg[::10].T)})
        return SubjectTestData(small_cfg, "dev", "labels", "S1", CPU, full=full)

    data = build(test_ecog)
    altered = test_ecog * 100 + 1e4
    altered[100000, :] = 1e8
    data2 = build(altered)

    # Los cinco bloques entrenan con la región válida completa (sin purga) y no hay validación.
    assert data.fold is None and data.feats.val_block is None
    index, w = data.feats.train_index, small_cfg["preprocessing"]["window"]
    assert set(index[:, 0]) == {0, 1, 2, 3, 4} and len(index) == 5 * (7750 - w + 1)
    for b in range(5):
        starts = index[index[:, 0] == b, 1]
        assert starts.min() == 125 and starts.max() + w == 7875
    assert data.info["train_blocks"] == small_cfg["folds"]["names"] and data.info["validation_fold"] is None

    # El test no cambia nada de lo ajustado con B1–B5.
    for key, value in data.preprocessing.items():
        np.testing.assert_array_equal(value, data2.preprocessing[key])
    np.testing.assert_array_equal(data.feats.X, data2.feats.X)
    assert not np.array_equal(data.test.X, data2.test.X)

    # Clip y escaladores incluyen B5: difieren de los del primer protocolo (solo B1–B4).
    es = build(test_ecog, full=False)
    assert not np.array_equal(data.preprocessing["clip_median"], es.preprocessing["clip_median"])
    assert not np.array_equal(data.preprocessing["x_center"], es.preprocessing["x_center"])


def test_train_fixed_epochs_runs_exactly_the_given_epochs(monkeypatch):
    def train(seed):
        torch.manual_seed(0)
        model = EncoderDecoder1D(6, TINY, n_outputs=5, window=256)
        rows, _ = training.train_fixed_epochs(model, tiny_features(), TCFG, 3, CPU, seed)
        return model, rows

    model, rows = train(5)
    assert [r["epoch"] for r in rows] == [1, 2, 3] and "val_pearson_macro" not in rows[0]
    again, _ = train(5)
    for a, b in zip(model.state_dict().values(), again.state_dict().values()):
        torch.testing.assert_close(a, b)  # misma semilla, mismo orden de lotes y mismos pesos

    monkeypatch.setattr(training, "mse_cosine_loss", lambda pred, target, lcfg: (pred.sum() * float("nan"),) * 3)
    seen = []
    with pytest.raises(FloatingPointError):
        training.train_fixed_epochs(model, tiny_features(), TCFG, 3, CPU, 5, on_epoch=seen.append)
    assert len(seen) == 1  # la época se registra antes del error


def test_predict_segment_matches_window_by_window():
    torch.manual_seed(0)
    model = EncoderDecoder1D(6, TINY, n_outputs=5, window=256)
    X = np.random.default_rng(0).standard_normal((6, 900)).astype(np.float32)
    lo, hi = 30, 870
    seg = final_test.EvalSegment(X=X, y_true=np.zeros((5, hi - lo)), t_seconds=np.zeros(hi - lo),
                                 starts=val_window_starts(lo, hi, 256, 64), lo=lo, hi=hi)
    got = predict_segment(model, seg, 256, 4, CPU)
    with torch.no_grad():
        model.eval()
        windows = np.stack([model(torch.from_numpy(X[None, :, s:s + 256]))[0].numpy() for s in seg.starts])
    np.testing.assert_allclose(got, overlap_add(windows, seg.starts, lo, hi), atol=1e-5)  # float32: lote vs 1


def test_saved_weights_rebuild_the_same_model(tmp_path):
    torch.manual_seed(0)
    model = EncoderDecoder1D(6, TINY, n_outputs=5, window=256).eval()
    save_model(tmp_path / "model.pt", model, {"n_outputs": 5, "seed": 7})
    rebuilt, ckpt = load_model(tmp_path / "model.pt")
    x = torch.randn(2, 6, 256)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x), model(x))
    assert ckpt["seed"] == 7 and not (tmp_path / "model.pt.tmp").exists()


class FakeTestRunner:
    """Sustituye a run_test_model: escribe status, metrics y predictions como él."""

    def __init__(self):
        self.calls = []

    def __call__(self, data, spec, params, seed, cfg, device, out_dir, stopper, tag="", max_batches=None,
                 on_epoch=None, epochs=None):
        self.calls.append((params["architecture_id"], seed, stopper.max_epochs if stopper else None, epochs))
        fingers = cfg["data"]["fingers"]
        score = (0.5 + 0.05 * (params["architecture_id"] == "A5") + 0.001 * (seed % 7)
                 + 0.01 * (stopper is None))  # B1–B5 (épocas fijas) da +0.01 en todas las semillas
        t = np.arange(4000) / 100
        y = np.stack([np.sin(t + k) for k in range(5)])
        np.savez_compressed(out_dir / "predictions.npz", y_true=y, y_pred=0.9 * y, t_seconds=t,
                            fingers=np.array(fingers), best_epoch=7)
        metrics = {"seed": seed, "best_epoch": 7, "epochs_trained": 17, "stop_reason": "patience_exhausted",
                   "best_val_pearson_macro": score + 0.1,
                   "test": {"pearson_macro": score, "pearson": {f: score for f in fingers}, "mse_macro": 0.01,
                            "ccc_macro": 0.4},
                   "train_seconds": 0.01}
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "status.json", {"status": "completed", "seed": seed})
        return metrics


def test_plan_runs_and_report(cfg, tmp_path):
    small = copy.deepcopy(cfg)
    small["search"]["n_trials_per_subject"] = 12
    results = tmp_path / "results"
    search(small, results, FakeRunner())
    with pytest.raises(RuntimeError, match="verificación"):
        freeze_plan(results, "S1", small)  # sin verificación no hay plan

    run_verification(small, "S1", "sin_datos", results, device="cpu", folds=small["folds"]["names"][:2],
                     data=FakeData(), runner=FakeRunner())
    with pytest.raises(RuntimeError, match="incompleta"):
        freeze_plan(results, "S1", small)  # 2 de 5 folds
    run_verification(small, "S1", "sin_datos", results, device="cpu", data=FakeData(), runner=FakeRunner())

    runner = FakeTestRunner()
    plan = run_final_test(small, "S1", "sin_datos", "sin_etiquetas", results, device="cpu", data=FakeData(),
                          runner=runner)
    assert plan["chosen"] == "A5" and set(plan["finalists"]) == {"A5b", "A5"}  # la superficie falsa favorece A5
    assert len(runner.calls) == 2 * 3 and {(m, e) for _, _, m, e in runner.calls} == {(70, None)}
    assert sorted({s for _, s, _, _ in runner.calls}) == [3046, 4046, 5046]  # base + 5·0 + índice de B5
    assert freeze_plan(results, "S1", small)["frozen_at"] == plan["frozen_at"]  # congelado

    again = FakeTestRunner()  # reanudar no repite corridas completadas
    run_final_test(small, "S1", "sin_datos", "sin_etiquetas", results, device="cpu", data=FakeData(), runner=again)
    assert again.calls == []

    # Segundo protocolo: B1–B5, épocas fijas = mediana de las 10 best_epoch de verificación (3 en el runner falso).
    full_runner = FakeTestRunner()
    full_plan = run_final_test(small, "S1", "sin_datos", "sin_etiquetas", results, device="cpu", data=FakeData(),
                               runner=full_runner, full=True)
    assert full_plan["chosen"] == plan["chosen"] and full_plan["base_plan_frozen_at"] == plan["frozen_at"]
    assert all(f["epochs"] == 3 and f["verification_best_epochs"] == [3] * 10 for f in full_plan["finalists"].values())
    assert {(m, e) for _, _, m, e in full_runner.calls} == {(None, 3)} and len(full_runner.calls) == 6
    assert sorted({s for _, s, _, _ in full_runner.calls}) == [3046, 4046, 5046]  # mismas semillas que B1–B4
    assert (results / "final_test_full" / "S1" / "A5" / "seed_3042" / "metrics.json").exists()
    assert freeze_full_plan(results, "S1", small)["frozen_at"] == full_plan["frozen_at"]  # congelado
    again = FakeTestRunner()
    run_final_test(small, "S1", "sin_datos", "sin_etiquetas", results, device="cpu", data=FakeData(), runner=again,
                   full=True)
    assert again.calls == []

    info = write_summary(results, small, REPO_ROOT / "reference" / "ablaciones")
    assert info["test_completed"] == 12 and {"test_scores", "test_trajectories"} <= set(info["figures"])
    summary = pd.read_csv(results / "summary" / "tpe_test_summary.csv")
    for protocol in ("b1b4_es_b5", "b1b5_fixed"):
        p = summary[summary["protocol"] == protocol]
        s1 = p[p["subject"] == "S1"].set_index("architecture_id")
        assert s1.loc["A5", "chosen"] and not s1.loc["A5b", "chosen"] and (s1["n_seeds"] == 3).all()
        elegida = p[(p["subject"] == "global") & (p["architecture_id"] == "elegida")].iloc[0]
        assert elegida["test_pearson_mean"] == pytest.approx(s1.loc["A5", "test_pearson_mean"])
    assert len(pd.read_csv(results / "summary" / "tpe_test.csv")) == 12
    protocols = pd.read_csv(results / "summary" / "tpe_test_protocols.csv")
    s1 = protocols[protocols["subject"] == "S1"].set_index("architecture_id")
    assert (s1["n_pairs"] == 3).all() and np.allclose(s1["delta_mean"], 0.01) and (s1["n_positive"] == 3).all()
