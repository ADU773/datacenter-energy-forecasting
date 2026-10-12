"""Forecaster wrappers around the PyTorch networks (common fit / predict / save / load)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from torch import nn

from dcforecast.data.sites import SITES
from dcforecast.models.base import Forecaster, finalize_quantiles
from dcforecast.models.deep.batch_size import find_batch_size
from dcforecast.models.deep.data import (
    NormStats,
    WindowDataset,
    build_arrays,
    origin_rows,
)
from dcforecast.models.deep.device import amp_enabled, resolve_device
from dcforecast.models.deep.networks import LSTMEncoderDecoder, TCNForecaster, count_parameters
from dcforecast.models.deep.trainer import TrainResult, make_loader, train_model
from dcforecast.utils import get_logger, seed_everything

log = get_logger(__name__)


class DeepForecaster(Forecaster):
    """Shared logic for the global sequence models."""

    uses_device: ClassVar[bool] = True

    def __init__(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        self.net: nn.Module | None = None
        self.stats: NormStats | None = None
        self.device = torch.device("cpu")
        self.dims: dict[str, int] = {}
        self.checkpoint_path: Path | None = None
        self.train_result: TrainResult | None = None

    # ------------------------------------------------------------------ architecture
    def build_net(self, enc_dim: int, dec_dim: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ data
    def _dataset(self, panel: pd.DataFrame, rows_kw: dict[str, Any]) -> WindowDataset:
        assert self.stats is not None
        d = self.cfg.deep
        arrays = build_arrays(panel, self.stats, self.target)
        rows = origin_rows(arrays, d.lookback, self.horizon, **rows_kw)
        return WindowDataset(arrays, rows, d.lookback, self.horizon, d.decoder_lags)

    def train_val_datasets(self, history: pd.DataFrame) -> tuple[WindowDataset, WindowDataset]:
        """Chronological split: the last ``deep.val_days`` of history validate early stopping."""
        assert self.stats is not None
        d = self.cfg.deep
        end = history["timestamp"].max()
        val_start = end - pd.Timedelta(days=d.val_days) + pd.Timedelta(hours=1)
        arrays = build_arrays(history, self.stats, self.target)
        train_rows = origin_rows(
            arrays, d.lookback, self.horizon, last=val_start - pd.Timedelta(hours=self.horizon + 1)
        )
        val_rows = origin_rows(
            arrays, d.lookback, self.horizon, first=val_start - pd.Timedelta(hours=1)
        )
        mk = lambda rows: WindowDataset(arrays, rows, d.lookback, self.horizon, d.decoder_lags)  # noqa: E731
        return mk(train_rows), mk(val_rows)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        history: pd.DataFrame,
        device: str | None = None,
        init_from: DeepForecaster | None = None,
        epochs: int | None = None,
    ) -> DeepForecaster:
        """Train on ``history``; optionally warm-start from another fitted model."""
        d = self.cfg.deep
        dev = resolve_device(device or d.device)
        self.device = dev
        seed_everything(self.cfg.seed, d.deterministic)
        self.sites_ = list(dict.fromkeys(history["site_id"]))
        self.stats = init_from.stats if init_from is not None else NormStats.fit(history)
        train_ds, val_ds = self.train_val_datasets(history)
        if len(train_ds) == 0:
            raise ValueError("Not enough history to build training windows")
        self.dims = {"enc_dim": train_ds.enc_dim, "dec_dim": train_ds.dec_dim}
        net = self.build_net(train_ds.enc_dim, train_ds.dec_dim)
        if init_from is not None and init_from.net is not None:
            net.load_state_dict(init_from.net.state_dict())
        net.to(dev)
        amp = amp_enabled(dev, d)
        rng = np.random.default_rng(self.cfg.seed)

        def make_batch(b: int) -> tuple[torch.Tensor, ...]:
            idx = rng.choice(len(train_ds), size=b, replace=len(train_ds) < b).tolist()
            return train_ds.__getitems__(idx)[:5]

        if d.batch_size == "auto":
            bs_result = find_batch_size(net, make_batch, dev, d, amp)
            batch_size = bs_result.batch_size
            bs_info: dict[str, Any] = {
                "budget_mb": bs_result.budget_mb,
                "probes": bs_result.probes,
                "reason": bs_result.reason,
            }
        else:
            batch_size, bs_info = int(d.batch_size), {"reason": "fixed in config"}
        n_epochs = epochs or d.epochs
        log.info(
            "%s: %d params, %d train / %d val windows, batch %d, device %s, AMP %s",
            self.name,
            count_parameters(net),
            len(train_ds),
            len(val_ds),
            batch_size,
            dev,
            amp,
        )
        result = train_model(
            net,
            train_ds,
            val_ds if len(val_ds) else None,
            d,
            dev,
            batch_size,
            n_epochs,
            seed=self.cfg.seed,
            checkpoint_path=self.checkpoint_path,
            log_prefix=f"[{self.name}] ",
        )
        self.net = net
        self.train_result = result
        self.fit_info_ = {
            **result.to_dict(),
            "params": count_parameters(net),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "batch_size_search": bs_info,
            "warm_start": init_from is not None,
        }
        return self

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        if self.net is None or self.stats is None:
            raise RuntimeError("model is not fitted")
        d = self.cfg.deep
        origins_idx = pd.DatetimeIndex(origins)
        lo = origins_idx.min() - pd.Timedelta(hours=d.lookback + 24)
        hi = origins_idx.max() + pd.Timedelta(hours=self.horizon)
        window = panel[(panel["timestamp"] >= lo) & (panel["timestamp"] <= hi)]
        window = window[window["site_id"].isin(self.sites_)]
        arrays = build_arrays(window, self.stats, self.target)
        rows, keys = [], []
        origin_values = origins_idx.values
        for site, (start, end) in arrays.bounds.items():
            ts = arrays.timestamps[start:end]
            pos = np.searchsorted(ts, origin_values)
            for o, p in zip(origin_values, pos, strict=True):
                r = start + int(p)
                if (
                    p < len(ts)
                    and ts[p] == o
                    and r - d.lookback + 1 >= start
                    and r + self.horizon < end
                ):
                    rows.append(r)
                    keys.append((site, o))
        if not rows:
            return finalize_quantiles(
                pd.DataFrame(
                    columns=["site_id", "origin", "timestamp", "horizon", "p10", "p50", "p90"]
                )
            )
        ds = WindowDataset(arrays, np.array(rows), d.lookback, self.horizon, d.decoder_lags)
        loader = make_loader(ds, 512, self.device, d, shuffle=False)
        amp = amp_enabled(self.device, d)
        self.net.eval()
        outs = []
        for enc, dec, site, _tgt, _msk, scale in loader:
            enc, dec, site = (t.to(self.device, non_blocking=True) for t in (enc, dec, site))
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
                pred = self.net(enc, dec, site)
            outs.append((pred.float().cpu() * scale[:, None, None]).numpy())
        q = np.concatenate(outs, axis=0)  # (N, H, 3)
        n, h = q.shape[0], q.shape[1]
        hs = np.arange(1, h + 1)
        sites = np.repeat([k[0] for k in keys], h)
        origin_rep = np.repeat(np.array([k[1] for k in keys]), h)
        frame = pd.DataFrame(
            {
                "site_id": sites,
                "origin": origin_rep,
                "horizon": np.tile(hs, n),
                "timestamp": origin_rep + np.tile(hs, n) * np.timedelta64(1, "h"),
                "p10": q[:, :, 0].ravel(),
                "p50": q[:, :, 1].ravel(),
                "p90": q[:, :, 2].ravel(),
            }
        )
        return finalize_quantiles(frame)

    # ------------------------------------------------------------------ persistence
    def to(self, device: str | torch.device) -> DeepForecaster:
        """Move the network to another device (e.g. CPU for serving)."""
        self.device = resolve_device(str(device))
        if self.net is not None:
            self.net.to(self.device)
        return self

    def _save_payload(self, path: Path) -> None:
        assert self.net is not None and self.stats is not None
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                "stats": self.stats.to_dict(),
                "dims": self.dims,
            },
            path / "model.pt",
        )

    def _load_payload(self, path: Path, device: str | None = None) -> None:
        payload = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        self.stats = NormStats.from_dict(payload["stats"])
        self.dims = {k: int(v) for k, v in payload["dims"].items()}
        net = self.build_net(self.dims["enc_dim"], self.dims["dec_dim"])
        net.load_state_dict(payload["state_dict"])
        self.net = net
        self.to(device or "cpu")
        self.net.eval()


class LSTMForecaster(DeepForecaster):
    """LSTM encoder-decoder, global over sites."""

    name = "lstm"

    def build_net(self, enc_dim: int, dec_dim: int) -> nn.Module:
        d = self.cfg.deep
        return LSTMEncoderDecoder(
            n_sites=len(SITES),
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            hidden=d.hidden_size,
            layers=d.num_layers,
            emb_dim=d.embedding_dim,
            dropout=d.dropout,
        )


class TCNModelForecaster(DeepForecaster):
    """Temporal Convolutional Network, global over sites."""

    name = "tcn"

    def build_net(self, enc_dim: int, dec_dim: int) -> nn.Module:
        d = self.cfg.deep
        return TCNForecaster(
            n_sites=len(SITES),
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            horizon=self.horizon,
            channels=d.tcn_channels,
            levels=d.tcn_levels,
            kernel_size=d.tcn_kernel_size,
            emb_dim=d.embedding_dim,
            dropout=d.dropout,
        )
