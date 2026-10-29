"""Espacio de búsqueda (plan §3.1), warmups (§6.1) y las 16 combinaciones discretas de A5b/A5 (§3.2)."""

from __future__ import annotations

import itertools

import optuna
import pytest
import torch

from ecog_tpe.config import SEARCH_PARAMS, load_config
from ecog_tpe.models import EncoderDecoder1D, count_parameters
from ecog_tpe.space import architecture_spec, suggest, warmup_trials

# Especificaciones de A5b y A5 en configs/experiment.yaml de las ablaciones (commit 043c44f).
_COMMON = {
    "stem_channels": 64, "encoder_channels": [64, 128, 256, 512, 512], "kernel_sizes": [7, 7, 5, 5, 5],
    "dilations": [1, 2, 3, 1, 2], "decoder_channels": [512, 512, 256, 128, 64],
    "decoder_kernel_sizes": [5, 5, 5, 7, 7], "normalization": "layer", "dropout": 0.1,
    "decoder_order": "upsample_concat_conv", "skip_position": "pre_pool",
}
ABLATION_SPECS = {
    "A5b": {**_COMMON, "upsample": "interp", "convtranspose_mode": None},
    "A5": {**_COMMON, "upsample": "convtranspose", "convtranspose_mode": "depthwise"},
}
# Parámetros entrenables en S3 (63 canales × 40 frecuencias), como en las ablaciones (≈ 9.36 M y 9.37 M).
PARAMS_S3 = {"A5b": 9_362_117, "A5": 9_369_797}
COMBOS = list(itertools.product(["A5b", "A5"], [5, 7], [4, 5], [64, 96]))


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("arch", ["A5b", "A5"])
def test_reference_combination_reproduces_ablation_specs(cfg, arch):
    assert architecture_spec(cfg, arch, 5, 5, 64) == ABLATION_SPECS[arch]
    assert cfg["architectures"][arch] == ABLATION_SPECS[arch]


@pytest.mark.parametrize("arch", ["A5b", "A5"])
def test_reference_models_for_each_subject(cfg, arch):
    for n_channels in (61, 46, 63):  # S1, S2, S3: solo cambia el stem
        torch.manual_seed(0)
        model = EncoderDecoder1D(n_channels * 40, architecture_spec(cfg, arch, 5, 5, 64))
        assert model(torch.randn(2, n_channels, 40, 256)).shape == (2, 5, 256)
        if n_channels == 63:
            assert count_parameters(model) == PARAMS_S3[arch]


@pytest.mark.parametrize("arch,kernel,levels,filters", COMBOS)
def test_all_combinations_forward_backward(cfg, arch, kernel, levels, filters):
    spec = architecture_spec(cfg, arch, kernel, levels, filters)
    assert spec["encoder_channels"] == [filters, 2 * filters, 4 * filters, 8 * filters, 8 * filters][:levels]
    assert spec["kernel_sizes"] == [7, 7] + [kernel] * (levels - 2)
    assert spec["dilations"] == [1, 2, 3, 1, 2][:levels]
    torch.manual_seed(0)
    model = EncoderDecoder1D(63 * 40, spec)  # S3, el sujeto con más canales
    model.trace_enabled = True
    out = model(torch.randn(2, 63 * 40, 256))
    assert out.shape == (2, 5, 256)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    decoder_ops = [op for stage, op, *_ in model.last_trace if stage.startswith("dec")]
    assert decoder_ops == ["upsample", "concat", "conv"] * levels
    assert [n for _, op, _, n in model.last_trace if op == "upsample"] == [256 >> (levels - 1 - j) for j in range(levels)]
    assert model.expected["head_in"] == filters


def test_warmups(cfg):
    warmups = warmup_trials(cfg)
    assert len(warmups) == 10
    assert [key for key, _ in warmups] == [f"p{i}_{a}" for i in range(1, 6) for a in ("A5b", "A5")]
    assert warmups[0][1] == {"architecture_id": "A5b", "learning_rate": 8.42e-5, "weight_decay": 1.0e-6,
                             "kernel_size": 5, "encoder_levels": 5, "base_filters": 64}
    for (_, a5b), (_, a5) in zip(warmups[::2], warmups[1::2]):
        assert {**a5b, "architecture_id": "A5"} == a5  # mismos hiperparámetros para ambas arquitecturas
    for _, params in warmups:
        assert tuple(params) == SEARCH_PARAMS
        assert suggest(optuna.trial.FixedTrial(params), cfg["search"]["space"]) == params  # dentro del espacio
