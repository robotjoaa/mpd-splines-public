import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class SmallOverlapEncoder(nn.Module):
    """
    Lightweight encoder for short overlap windows (len_ovlap < ~8).

    Pipeline:
    - Per-timestep MLP to lift inputs to a hidden size.
    - Optional shallow Conv1d to capture local correlations across the short horizon.
    - Global pooling (mean or max) over time.
    - Final MLP projection to the desired embedding dimension.

    Input:  x of shape [B, L, D] where L=len_ovlap, D=state_dim
    Output: embedding of shape [B, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        conv_dim: int = 64,
        out_dim: int = 128,
        conv_kernel: int = 3,
        pool: Literal["mean", "max"] = "mean",
        use_time_emb: bool = False,
        time_emb_dim: int = 32,
    ) -> None:
        super().__init__()

        self.use_time_emb = use_time_emb
        if use_time_emb:
            # Simple sinusoidal positional encoding; no learnable params.
            self.register_buffer("pos_indices", None, persistent=False)
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.Mish(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None

        self.per_step = nn.Sequential(
            nn.Linear(in_dim + (time_emb_dim if use_time_emb else 0), hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        padding = (conv_kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, conv_dim, kernel_size=conv_kernel, padding=padding),
            nn.Mish(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=conv_kernel, padding=padding),
            nn.Mish(),
        )

        self.pool = pool
        self.proj = nn.Sequential(
            nn.Linear(conv_dim, max(out_dim, conv_dim)),
            nn.Mish(),
            nn.Linear(max(out_dim, conv_dim), out_dim),
        )

    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, D]
        time: optional [B] or [B, L] timesteps for positional encoding (only used if use_time_emb=True)
        """
        b, l, _ = x.shape

        if self.use_time_emb:
            if time is None:
                # default: positions 0..L-1
                if self.pos_indices is None or self.pos_indices.numel() != l:
                    self.pos_indices = torch.arange(l, device=x.device, dtype=torch.float32)
                t_inp = self.pos_indices[None, :].expand(b, l)
            else:
                # broadcast if [B]
                t_inp = time[:, None] if time.dim() == 1 else time
            t_feat = self.time_mlp(t_inp)  # [B, L, time_emb_dim]
            x = torch.cat([x, t_feat], dim=-1)

        h = self.per_step(x)  # [B, L, hidden_dim]
        h = h.transpose(1, 2)  # [B, hidden_dim, L]
        h = self.conv(h)       # [B, conv_dim, L]

        if self.pool == "mean":
            h = h.mean(dim=-1)
        elif self.pool == "max":
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        return self.proj(h)


class SinusoidalPosEmb(nn.Module):
    """
    Minimal sinusoidal position embedding (copied locally to avoid extra deps).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = x[..., None] * emb
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            # zero pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


__all__ = ["SmallOverlapEncoder"]
