import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import register
from .scan import linear_scan


class ShortConv(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, size, groups=dim, padding=size - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))


class FFTNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        channels: int = 1,
        mode: str = "ridge",
        expand: float = 2.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_gate: bool = True,
        per_freq_gates: bool = False,
        ridge_lambda: float = 1.0,
        checkpoint_scan: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
    ):
        super().__init__()
        if mode not in ("ridge", "delta"):
            raise ValueError(f"mode must be 'ridge' or 'delta', got {mode!r}")
        inner = int(d_model * expand)
        head_dim = inner // num_heads
        if head_dim * num_heads != inner or head_dim % channels != 0:
            raise ValueError("d_model*expand must be divisible by num_heads, and head_dim by channels")

        self.H, self.C, self.mode = num_heads, channels, mode
        self.head_dim = head_dim
        self.seg = head_dim // channels
        self.n_freq = self.seg // 2 + 1
        self.checkpoint_scan = checkpoint_scan

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.conv = nn.ModuleList(ShortConv(inner, conv_size) for _ in range(3)) if use_short_conv else None

        n_gate = num_heads * (self.n_freq if per_freq_gates else 1)
        self.a_proj = nn.Linear(d_model, n_gate, bias=False)
        self.A_log = nn.Parameter(torch.empty(n_gate).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(n_gate) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        if mode == "delta":
            self.b_proj = nn.Linear(d_model, n_gate, bias=True)
        else:
            self.lam = nn.Parameter(torch.full((num_heads,), math.log(math.expm1(ridge_lambda))))

        self.g_proj = nn.Linear(d_model, inner, bias=False) if use_gate else None
        self.norm = nn.RMSNorm(head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.conv is not None:
            q, k, v = (c(t) for c, t in zip(self.conv, (q, k, v)))

        with torch.autocast(x.device.type, enabled=False):
            xf = x.float()
            q, k, v = (self._spectrum(t) for t in (q, k, v))
            k = k / (k.abs().square().sum(-1, keepdim=True).sqrt() + 1e-6)
            alpha = torch.exp(-F.softplus(self.a_proj(xf) + self.dt_bias) * self.A_log.exp())
            alpha = alpha.view(B, T, self.H, -1)
            if self.mode == "ridge":
                y = self._ridge(q, k, v, alpha)
            else:
                beta = torch.sigmoid(self.b_proj(xf)).view(B, T, self.H, -1)
                y = self._delta(q, k, v, alpha, beta)
            y = torch.fft.irfft(y.transpose(-1, -2), n=self.seg, dim=-1)
            y = y.reshape(B, T, self.H, self.head_dim)

        y = self.norm(y)
        if self.g_proj is not None:
            y = y * F.silu(self.g_proj(x)).view(B, T, self.H, self.head_dim)
        return self.o_proj(y.reshape(B, T, -1))

    def _spectrum(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        t = t.float().view(B, T, self.H, self.C, self.seg)
        return torch.fft.rfft(t, dim=-1).transpose(-1, -2)

    def _scan(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_scan and self.training:
            return checkpoint(linear_scan, a, b, use_reentrant=False)
        return linear_scan(a, b)

    def _ridge(self, q, k, v, alpha):
        lam = F.softplus(self.lam).view(1, 1, -1, 1)
        if self.C == 1:
            q, k, v = (t.squeeze(-1) for t in (q, k, v))
            A = self._scan(alpha, k.abs().square())
            Bm = self._scan(alpha, v * k.conj())
            return (Bm * q / (A + lam)).unsqueeze(-1)
        kc = k.conj().unsqueeze(-2)
        a = alpha[..., None, None]
        A = self._scan(a, k.unsqueeze(-1) * kc)
        Bm = self._scan(a, v.unsqueeze(-1) * kc)
        eye = torch.eye(self.C, dtype=A.dtype, device=A.device)
        z = torch.linalg.solve(A + lam[..., None, None] * eye, q.unsqueeze(-1))
        return (Bm @ z).squeeze(-1)

    def _delta(self, q, k, v, alpha, beta):
        if self.C == 1:
            q, k, v = (t.squeeze(-1) for t in (q, k, v))
            h = self._scan(alpha * (1 - beta), beta * v * k.conj())
            return (h * q).unsqueeze(-1)
        B, T, H, Fq, C = q.shape
        state = q.new_zeros(B, H, Fq, C, C)
        kc = k.conj().unsqueeze(-2)
        a, b = alpha[..., None, None], beta[..., None, None]
        ys = []
        for t in range(T):
            sk = state @ k[:, t].unsqueeze(-1)
            state = a[:, t] * (state - b[:, t] * (sk @ kc[:, t])) + b[:, t] * (v[:, t].unsqueeze(-1) @ kc[:, t])
            ys.append((state @ q[:, t].unsqueeze(-1)).squeeze(-1))
        return torch.stack(ys, dim=1)


@register("fftnet")
def build_fftnet(d_model: int, layer_idx: int, n_layer: int, **kwargs):
    return FFTNet(d_model, layer_idx=layer_idx, **kwargs)
