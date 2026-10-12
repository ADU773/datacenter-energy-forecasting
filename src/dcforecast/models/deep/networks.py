"""PyTorch architectures: LSTM encoder-decoder and Temporal Convolutional Network (TCN).

Both are *global* models (one network for all sites, with a learned site embedding) that
produce direct multi-horizon forecasts: all H = 24 steps in one forward pass, each with three
quantiles. The quantile head is monotone by construction (P10 <= P50 <= P90).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

QUANTILES = (0.1, 0.5, 0.9)


class QuantileHead(nn.Module):
    """Linear head emitting ``(P10, P50, P90)`` without quantile crossing.

    The middle output is the median; the outer two are positive offsets (softplus) below and
    above it.
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, 3)
        with torch.no_grad():
            self.proj.bias.copy_(torch.tensor([-2.0, 1.0, -2.0]))  # median ~ 1 (scaled target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        mid = out[..., 1]
        lo = mid - F.softplus(out[..., 0])
        hi = mid + F.softplus(out[..., 2])
        return torch.stack([lo, mid, hi], dim=-1)


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    quantiles: Sequence[float] = QUANTILES,
) -> torch.Tensor:
    """Mean pinball (quantile) loss over quantiles, ignoring masked-out targets.

    Args:
        pred: ``(B, H, Q)`` predictions.
        target: ``(B, H)`` targets.
        mask: optional ``(B, H)`` 0/1 weights.
    """
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1.0) * diff).mean(dim=-1)
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class StackedLSTM(nn.Module):
    """Multi-layer LSTM built from single-layer cuDNN LSTMs with dropout in between.

    Functionally the same as ``nn.LSTM(num_layers=k, dropout=p)``. It exists because the fused
    multi-layer cuDNN kernel *with* inter-layer dropout crashes the interpreter at exit
    (0xC0000409) with torch 2.11 on Windows; single-layer kernels plus ``nn.Dropout`` avoid it.
    """

    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.LSTM(input_size if i == 0 else hidden, hidden, 1, batch_first=True)
            for i in range(layers)
        )
        self.dropout = nn.Dropout(dropout if layers > 1 else 0.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hs, cs = [], []
        for i, lstm in enumerate(self.layers):
            if i > 0:
                x = self.dropout(x)
            init = None if state is None else (state[0][i : i + 1], state[1][i : i + 1])
            x, (h, c) = lstm(x, init)
            hs.append(h)
            cs.append(c)
        return x, (torch.cat(hs, dim=0), torch.cat(cs, dim=0))


class LSTMEncoderDecoder(nn.Module):
    """Sequence-to-sequence LSTM.

    The encoder reads the lookback window; its final (h, c) initialises a decoder LSTM that runs
    over the known future covariates (calendar, weather forecast, seasonal lags) of the 24
    target hours -- no autoregressive feedback, so training and inference behave identically.
    """

    def __init__(
        self,
        n_sites: int,
        enc_dim: int,
        dec_dim: int,
        hidden: int = 64,
        layers: int = 2,
        emb_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.site_emb = nn.Embedding(n_sites, emb_dim)
        self.encoder = StackedLSTM(enc_dim + emb_dim, hidden, layers, dropout)
        self.decoder = StackedLSTM(dec_dim + emb_dim, hidden, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + dec_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            QuantileHead(hidden),
        )

    def forward(self, enc: torch.Tensor, dec: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        e = self.site_emb(site)
        enc_in = torch.cat([enc, e[:, None, :].expand(-1, enc.size(1), -1)], dim=-1)
        _, state = self.encoder(enc_in)
        dec_in = torch.cat([dec, e[:, None, :].expand(-1, dec.size(1), -1)], dim=-1)
        out, _ = self.decoder(dec_in, state)
        return self.head(torch.cat([out, dec], dim=-1))


class CausalConv1d(nn.Conv1d):
    """1-D convolution that only looks at the past (left padding)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__(c_in, c_out, kernel_size, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection (Bai et al., 2018).

    No normalisation layer: inputs are already instance-normalised, and a GroupNorm variant
    cost about a quarter of the training throughput in a micro-benchmark on the laptop GPU.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.conv1(x)))
        h = self.drop(F.gelu(self.conv2(h)))
        return x + h


class TCNForecaster(nn.Module):
    """Dilated causal TCN encoder + per-horizon MLP decoder.

    With kernel 3 and dilations 1..32 (6 levels) the receptive field is 253 h, covering the
    168 h lookback. The context vector (last time step) is combined with each target hour's
    known covariates, the site embedding and a horizon embedding.
    """

    def __init__(
        self,
        n_sites: int,
        enc_dim: int,
        dec_dim: int,
        horizon: int,
        channels: int = 48,
        levels: int = 6,
        kernel_size: int = 3,
        emb_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.site_emb = nn.Embedding(n_sites, emb_dim)
        self.horizon_emb = nn.Embedding(horizon, emb_dim)
        self.input_proj = nn.Conv1d(enc_dim + emb_dim, channels, 1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(channels, kernel_size, 2**i, dropout) for i in range(levels)]
        )
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2**levels - 1)
        d_in = channels * 2 + dec_dim + 2 * emb_dim
        self.decoder = nn.Sequential(
            nn.Linear(d_in, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
            nn.GELU(),
            QuantileHead(channels),
        )

    def forward(self, enc: torch.Tensor, dec: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        b, length, _ = enc.shape
        h_steps = dec.size(1)
        e = self.site_emb(site)
        x = torch.cat([enc, e[:, None, :].expand(-1, length, -1)], dim=-1).transpose(1, 2)
        h = self.blocks(self.input_proj(x))  # (B, C, L)
        ctx = torch.cat([h[:, :, -1], h[:, :, -24:].mean(dim=-1)], dim=-1)  # last + last-day mean
        hz = self.horizon_emb(torch.arange(h_steps, device=enc.device))[None].expand(b, -1, -1)
        z = torch.cat(
            [
                ctx[:, None, :].expand(-1, h_steps, -1),
                dec,
                e[:, None, :].expand(-1, h_steps, -1),
                hz,
            ],
            dim=-1,
        )
        return self.decoder(z)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
