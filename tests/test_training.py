"""train_fold: early stopping sobre el Pearson de validación, restauración del mejor checkpoint y fallos numéricos."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecog_tpe import training
from ecog_tpe.models import EncoderDecoder1D
from ecog_tpe.preprocessing import FoldFeatures, val_window_starts
from ecog_tpe.training import EarlyStopping, train_fold

CPU = torch.device("cpu")
FINGERS = ["thumb", "index", "middle", "ring", "little"]
TINY = {
    "stem_channels": 4, "encoder_channels": [4, 8, 8, 8], "kernel_sizes": [3, 3, 3, 3], "dilations": [1, 1, 1, 1],
    "decoder_channels": [8, 8, 8, 4], "decoder_kernel_sizes": [3, 3, 3, 3], "normalization": "layer",
    "dropout": 0.0, "upsample": "interp", "decoder_order": "upsample_concat_conv", "convtranspose_mode": None,
    "skip_position": "pre_pool",
}
TCFG = {"lr": 3e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8, "batch_size": 16, "drop_last": True,
        "loss": {"mse_weight": 0.5, "cosine_weight": 0.5, "cosine_eps": 1e-8}}


def tiny_features(seed: int = 0) -> FoldFeatures:
    """Dos bloques (0 = train, 1 = validación) de 600 columnas con 6 características correlacionadas con el guante."""
    rng = np.random.default_rng(seed)
    t = np.arange(600) / 100
    Y = np.stack([np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (0.3 + 0.07 * k) * t + b) for k in range(5)])
                  for b in range(2)]).astype(np.float32)
    X = (np.repeat(Y[:, :3], 2, axis=1) + 0.3 * rng.standard_normal((2, 6, 600))).astype(np.float32)
    lo, hi = 20, 580
    starts = np.arange(lo, hi - 256 + 1, 4)
    return FoldFeatures(
        preprocessing="P0", X=X, Y=Y, y_min=np.zeros(5), y_scale=np.ones(5),
        train_index=np.stack([np.zeros_like(starts), starts], axis=1).astype(np.int64), val_block=1,
        val_starts=val_window_starts(lo, hi, 256, 64), valid_lo=lo, valid_hi=hi, delay=2, window=256,
        n_channels=6, n_freqs=1, params={}, stats={},
    )


def test_restores_the_best_checkpoint():
    feats = tiny_features()
    torch.manual_seed(0)
    model = EncoderDecoder1D(6, TINY)
    states = {}

    def on_epoch(row):
        states[row["epoch"]] = {k: v.clone() for k, v in model.state_dict().items()}

    stopper = EarlyStopping(patience=2, min_epochs=1, max_epochs=8)
    result = train_fold(model, feats, TCFG, stopper, CPU, seed=0, fingers=FINGERS, on_epoch=on_epoch)
    scores = [row["val_pearson_macro"] for row in result.history]
    assert result.epochs_trained == len(result.history) == len(states)
    assert result.best_epoch == int(np.argmax(scores)) + 1 and result.best_score == max(scores)
    for key, value in model.state_dict().items():  # pesos de best_epoch, no de la última época
        torch.testing.assert_close(value, states[result.best_epoch][key])
    assert result.restored_score == pytest.approx(result.best_score, abs=1e-6)
    assert result.y_pred_scaled.shape == (5, 560)
    assert result.history[-1]["best_epoch"] == result.best_epoch


def test_scripted_scores_follow_the_policy(monkeypatch):
    values = iter([0.2, 0.5, 0.4, 0.3, 0.5])  # el último corresponde a la revalidación tras restaurar
    monkeypatch.setattr(training, "pearson_per_channel", lambda y, p: (np.full(5, next(values)), [None] * 5))
    torch.manual_seed(0)
    stopper = EarlyStopping(patience=2, min_epochs=1, max_epochs=10)
    result = train_fold(EncoderDecoder1D(6, TINY), tiny_features(), TCFG, stopper, CPU, seed=0, fingers=FINGERS)
    assert (result.best_epoch, result.epochs_trained, result.stop_reason) == (2, 4, "patience_exhausted")
    assert [row["wait"] for row in result.history] == [0, 0, 1, 2]
    assert result.best_score == 0.5 and result.restored_score == 0.5


def test_invalid_pearson_fails_after_recording_the_epoch(monkeypatch):
    values = iter([(0.3, None), (np.nan, "predicción constante")])

    def fake(y, p):
        value, reason = next(values)
        return np.array([0.3, 0.3, value, 0.3, 0.3]), [None, None, reason, None, None]

    monkeypatch.setattr(training, "pearson_per_channel", fake)
    rows = []
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="época 2.*middle: predicción constante"):
        train_fold(EncoderDecoder1D(6, TINY), tiny_features(), TCFG, EarlyStopping(min_epochs=1), CPU, seed=0,
                   fingers=FINGERS, on_epoch=rows.append)
    assert [row["epoch"] for row in rows] == [1, 2] and np.isnan(rows[1]["val_pearson_macro"])
