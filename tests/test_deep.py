"""PyTorch building blocks: loss, networks, devices, batch-size finder, training loop."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.evaluation.metrics import pinball
from dcforecast.models import create_model
from dcforecast.models.deep.batch_size import find_batch_size
from dcforecast.models.deep.device import num_workers, resolve_device
from dcforecast.models.deep.networks import (
    LSTMEncoderDecoder,
    TCNForecaster,
    count_parameters,
    pinball_loss,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_pinball_loss_matches_numpy_reference() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(4, 24)).astype(np.float32)
    pred = rng.normal(size=(4, 24, 3)).astype(np.float32)
    ref = np.mean(
        [pinball(y.ravel(), pred[..., i].ravel(), q) for i, q in enumerate((0.1, 0.5, 0.9))]
    )
    got = float(pinball_loss(torch.from_numpy(pred), torch.from_numpy(y)))
    assert got == pytest.approx(ref, rel=1e-5)
    mask = torch.zeros(4, 24)
    mask[0] = 1
    masked = float(pinball_loss(torch.from_numpy(pred), torch.from_numpy(y), mask))
    ref0 = np.mean([pinball(y[0], pred[0, :, i], q) for i, q in enumerate((0.1, 0.5, 0.9))])
    assert masked == pytest.approx(ref0, rel=1e-5)


@pytest.mark.parametrize("arch", ["lstm", "tcn"])
def test_networks_output_three_quantiles_per_horizon(arch: str) -> None:
    b, lookback, horizon, enc_dim, dec_dim = 5, 168, 24, 9, 12
    if arch == "lstm":
        net: torch.nn.Module = LSTMEncoderDecoder(
            3, enc_dim, dec_dim, hidden=16, layers=1, emb_dim=4
        )
    else:
        net = TCNForecaster(3, enc_dim, dec_dim, horizon, channels=8, levels=6, emb_dim=4)
    enc = torch.randn(b, lookback, enc_dim)
    dec = torch.randn(b, horizon, dec_dim)
    site = torch.tensor([0, 1, 2, 0, 1])
    out = net(enc, dec, site)
    assert out.shape == (b, horizon, 3)
    assert count_parameters(net) > 0
    out.sum().backward()  # differentiable end to end


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda").type == "cpu"  # graceful fallback, no crash
    assert resolve_device("auto").type in {"cpu", "mps"}
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_num_workers_policy(tiny_cfg: Config) -> None:
    assert num_workers(dataclasses.replace(tiny_cfg.deep, num_workers=3)) == 3
    auto = num_workers(dataclasses.replace(tiny_cfg.deep, num_workers="auto"))
    assert auto in (0, 2)


def test_batch_size_finder_on_cpu_is_fixed(tiny_cfg: Config) -> None:
    res = find_batch_size(
        torch.nn.Linear(2, 2), lambda b: (), torch.device("cpu"), tiny_cfg.deep, False
    )
    assert tiny_cfg.deep.min_batch_size <= res.batch_size <= tiny_cfg.deep.max_batch_size
    assert res.budget_mb is None


def test_training_checkpoints_and_warm_start(
    tiny_cfg: Config, tiny_data: PanelData, tmp_path: Path
) -> None:
    model = create_model("lstm", tiny_cfg)
    model.checkpoint_path = tmp_path / "best.pt"  # type: ignore[attr-defined]
    model.fit(tiny_data.history, device="cpu")
    info = model.fit_info_
    assert (tmp_path / "best.pt").exists()
    assert 1 <= info["epochs_run"] <= tiny_cfg.deep.epochs
    assert len(info["history"]) == info["epochs_run"]
    assert np.isfinite(info["best_val_loss"])
    assert info["device"] == "cpu" and info["amp"] is False
    warm = create_model("lstm", tiny_cfg).fit(
        tiny_data.history, device="cpu", init_from=model, epochs=1
    )  # type: ignore[call-arg]
    assert warm.fit_info_["warm_start"] is True
    assert warm.fit_info_["epochs_run"] == 1


@requires_cuda
def test_gpu_batch_size_finder_respects_vram_budget(tiny_cfg: Config) -> None:
    cfg = dataclasses.replace(
        tiny_cfg.deep, vram_budget_mb=48, min_batch_size=16, max_batch_size=4096
    )
    dev = torch.device("cuda")
    net = LSTMEncoderDecoder(2, 9, 12, hidden=64, layers=2, emb_dim=4).to(dev)

    def make_batch(b: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(b, 168, 9),
            torch.randn(b, 24, 12),
            torch.zeros(b, dtype=torch.long),
            torch.randn(b, 24),
            torch.ones(b, 24),
        )

    res = find_batch_size(net, make_batch, dev, cfg, amp=True)
    assert res.budget_mb is not None and res.budget_mb <= 48
    assert res.batch_size < 4096
    fitting = [p for p in res.probes if p["batch_size"] == res.batch_size]
    assert not fitting or fitting[0]["peak_mb"] <= res.budget_mb


@requires_cuda
def test_gpu_training_uses_amp(tiny_cfg: Config, tiny_data: PanelData) -> None:
    cfg = dataclasses.replace(
        tiny_cfg, deep=dataclasses.replace(tiny_cfg.deep, device="cuda", amp=True, epochs=1)
    )
    model = create_model("tcn", cfg).fit(tiny_data.history, device="cuda")
    assert model.fit_info_["amp"] is True
    assert model.fit_info_["peak_gpu_mem_mb"] > 0
