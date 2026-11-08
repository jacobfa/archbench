"""N-step gated delta rule (idea.tex).

Per head: B = alpha S, r = v - B k, S = B + r u^T, y = S q, where the write direction u is N gradient steps on
1/2 (1 - k.u)^2 + lam/2 (p.u)^2 with p = k_{t-1}; u = beta k for N = 1 (plain GDN).  u is computed in PyTorch;
the Triton kernel runs the generalised recurrence S = alpha S (I - k u^T) + v u^T with fp32 state in registers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

CHUNK = 32  # tokens between stored states; the backward recomputes inside a chunk
ROWS = 16  # rows of S per program
NUM_WARPS = 4


def nsgdn_available(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda


def write_direction(k, p, beta, lam, n_steps: int) -> torch.Tensor:
    b = beta[..., None]
    l = lam[None, None, :, None]
    u = torch.zeros_like(k)
    for _ in range(n_steps):
        ku = (k * u).sum(-1, keepdim=True)
        pu = (p * u).sum(-1, keepdim=True)
        u = u + b * ((1.0 - ku) * k - l * pu * p)
    return u


if triton is not None:

    @triton.jit
    def _nsgdn_fwd_kernel(
        q_ptr, k_ptr, u_ptr, v_ptr, la_ptr, y_ptr, r_ptr, st_ptr, T, H, NB, NCH,
        DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, CT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh, nb = pid // NB, pid % NB
        b, h = bh // H, bh % H
        rl = tl.arange(0, BV)[:, None]
        ri = nb * BV + rl
        jj = tl.arange(0, DK)[None, :]
        S = tl.zeros([BV, DK], dtype=tl.float32)
        tok = ((b * T) * H + h).to(tl.int64)
        for c in range(NCH):
            tl.store(st_ptr + ((bh * NCH + c).to(tl.int64) * DV + ri) * DK + jj, S)
            for i in range(CT):
                k_r = tl.load(k_ptr + tok * DK + jj)
                u_r = tl.load(u_ptr + tok * DK + jj)
                q_r = tl.load(q_ptr + tok * DK + jj)
                v_c = tl.load(v_ptr + tok * DV + ri)
                la = tl.load(la_ptr + tok)
                S = tl.exp(la) * S
                r = v_c - tl.sum(S * k_r, 1, keep_dims=True)
                S = S + r * u_r
                tl.store(r_ptr + tok * DV + ri, r)
                tl.store(y_ptr + tok * DV + ri, tl.sum(S * q_r, 1, keep_dims=True))
                tok += H

    @triton.jit
    def _nsgdn_bwd_kernel(
        q_ptr, k_ptr, u_ptr, la_ptr, r_ptr, st_ptr, gy_ptr, sc_ptr, gq_ptr, gk_ptr, gu_ptr, gv_ptr, gla_ptr, T, H, NB, NCH,
        DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, CT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh, nb = pid // NB, pid % NB
        b, h = bh // H, bh % H
        rl = tl.arange(0, BV)[:, None]
        ri = nb * BV + rl
        jj = tl.arange(0, DK)[None, :]
        sc = sc_ptr + (pid.to(tl.int64) * CT * BV + rl) * DK + jj
        G = tl.zeros([BV, DK], dtype=tl.float32)
        for ci in range(NCH):
            c = NCH - 1 - ci
            S = tl.load(st_ptr + ((bh * NCH + c).to(tl.int64) * DV + ri) * DK + jj)
            tok0 = ((b * T + c * CT) * H + h).to(tl.int64)
            tok = tok0
            for i in range(CT):
                u_r = tl.load(u_ptr + tok * DK + jj)
                la = tl.load(la_ptr + tok)
                r_c = tl.load(r_ptr + tok * DV + ri)
                gy_c = tl.load(gy_ptr + tok * DV + ri)
                S = tl.exp(la) * S
                tl.store(sc + i * BV * DK, S)
                S = S + r_c * u_r
                tl.atomic_add(gq_ptr + tok * DK + jj, tl.sum(S * gy_c, 0, keep_dims=True))
                tok += H
            tok = tok0 + (CT - 1) * H
            for i2 in range(CT):
                i = CT - 1 - i2
                k_r = tl.load(k_ptr + tok * DK + jj)
                u_r = tl.load(u_ptr + tok * DK + jj)
                q_r = tl.load(q_ptr + tok * DK + jj)
                la = tl.load(la_ptr + tok)
                r_c = tl.load(r_ptr + tok * DV + ri)
                gy_c = tl.load(gy_ptr + tok * DV + ri)
                Sp = tl.load(sc + i * BV * DK)
                G = G + gy_c * q_r
                gr = tl.sum(G * u_r, 1, keep_dims=True)
                tl.store(gv_ptr + tok * DV + ri, gr)
                tl.atomic_add(gu_ptr + tok * DK + jj, tl.sum(G * r_c, 0, keep_dims=True))
                G = G - gr * k_r
                tl.atomic_add(gk_ptr + tok * DK + jj, -tl.sum(Sp * gr, 0, keep_dims=True))
                tl.atomic_add(gla_ptr + tok, tl.sum(G * Sp))
                G = tl.exp(la) * G
                tok -= H

    def _chunk(T: int) -> int:
        for c in (CHUNK, 16, 8, 4, 2, 1):
            if c <= CHUNK and T % c == 0:
                return c
        return 1

    @torch.library.custom_op("archbench::nsgdn_fwd", mutates_args=())
    def _nsgdn_fwd(
        q: torch.Tensor, k: torch.Tensor, u: torch.Tensor, v: torch.Tensor, la: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, DK = q.shape
        DV = v.shape[-1]
        CT = _chunk(T)
        NCH = T // CT
        BV = min(ROWS, DV)
        NB = DV // BV
        y, r = torch.empty_like(v), torch.empty_like(v)
        st = torch.empty(B, H, NCH, DV, DK, dtype=torch.float32, device=q.device)
        _nsgdn_fwd_kernel[(B * H * NB,)](q, k, u, v, la, y, r, st, T, H, NB, NCH, DK=DK, DV=DV, BV=BV, CT=CT, num_warps=NUM_WARPS)
        return y, r, st

    @_nsgdn_fwd.register_fake
    def _(q, k, u, v, la):
        B, T, H, DK = q.shape
        DV = v.shape[-1]
        NCH = T // _chunk(T)
        return torch.empty_like(v), torch.empty_like(v), q.new_empty(B, H, NCH, DV, DK)

    @torch.library.custom_op("archbench::nsgdn_bwd", mutates_args=())
    def _nsgdn_bwd(
        q: torch.Tensor, k: torch.Tensor, u: torch.Tensor, la: torch.Tensor, r: torch.Tensor, st: torch.Tensor, gy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, DK = q.shape
        DV = r.shape[-1]
        NCH = st.shape[2]
        CT = T // NCH
        BV = min(ROWS, DV)
        NB = DV // BV
        gq, gk, gu = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
        gv = torch.empty_like(r)
        gla = torch.zeros_like(la)
        sc = torch.empty(B * H * NB, CT, BV, DK, dtype=torch.float32, device=q.device)
        _nsgdn_bwd_kernel[(B * H * NB,)](
            q, k, u, la, r, st, gy, sc, gq, gk, gu, gv, gla, T, H, NB, NCH, DK=DK, DV=DV, BV=BV, CT=CT, num_warps=NUM_WARPS
        )
        return gq, gk, gu, gv, gla

    @_nsgdn_bwd.register_fake
    def _(q, k, u, la, r, st, gy):
        return torch.empty_like(q), torch.empty_like(q), torch.empty_like(q), torch.empty_like(r), torch.empty_like(la)

    def _setup(ctx, inputs, output):
        q, k, u, v, la = inputs
        _, r, st = output
        ctx.save_for_backward(q, k, u, la, r, st)

    def _backward(ctx, gy, gr, gst):
        q, k, u, la, r, st = ctx.saved_tensors
        return _nsgdn_bwd(q, k, u, la, r, st, gy.contiguous())

    _nsgdn_fwd.register_autograd(_backward, setup_context=_setup)


def nsgdn_scan(q, k, u, v, la) -> torch.Tensor:
    """y (B, T, H, DV) fp32 for q, k, u (B, T, H, DK), v (B, T, H, DV), log-retention la (B, T, H). DK, DV powers of 2."""
    return _nsgdn_fwd(q.contiguous(), k.contiguous(), u.contiguous(), v.contiguous(), la.contiguous())[0]


def nsgdn_ref(q, k, u, v, la) -> torch.Tensor:
    B, T, H, DK = q.shape
    S = q.new_zeros(B, H, v.shape[-1], DK)
    ys = []
    for t in range(T):
        S = la[:, t].exp()[..., None, None] * S
        r = v[:, t] - (S @ k[:, t][..., None]).squeeze(-1)
        S = S + r[..., :, None] * u[:, t][..., None, :]
        ys.append((S @ q[:, t][..., None]).squeeze(-1))
    return torch.stack(ys, 1)


class ShortConv(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.empty(dim, 1, size).uniform_(-1 / math.sqrt(size), 1 / math.sqrt(size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if causal_conv1d_fn is not None and x.is_cuda:
            return causal_conv1d_fn(x.transpose(1, 2), self.weight.squeeze(1).to(x.dtype), activation="silu").transpose(1, 2)
        T = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), self.weight.to(x.dtype), padding=self.size - 1, groups=x.shape[2])
        return F.silu(y[..., :T].transpose(1, 2))


class NStepGDN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: int = 128,
        expand_v: float = 2.0,
        n_steps: int = 2,
        lam: float = 1.0,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
    ):
        super().__init__()
        DK, DV = head_dim, int(head_dim * expand_v)
        if DK & (DK - 1) or DV & (DV - 1):
            raise ValueError("head_dim and head_dim * expand_v must be powers of 2")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.H, self.DK, self.DV, self.n_steps = num_heads, DK, DV, n_steps
        self.qk_dim, self.v_dim = num_heads * DK, num_heads * DV
        self.use_gate = use_gate
        self.n_gate = self.v_dim if use_gate else 0
        width = 2 * self.qk_dim + self.v_dim + self.n_gate + 2 * num_heads
        self.in_proj = nn.Linear(d_model, (width + 7) // 8 * 8, bias=False)
        self.conv = ShortConv(2 * self.qk_dim + self.v_dim, conv_size) if use_short_conv else None
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.b_bias = nn.Parameter(torch.zeros(num_heads))
        self.lam = nn.Parameter(torch.full((num_heads,), math.log(math.expm1(lam))))
        self.norm = nn.RMSNorm(DV, eps=norm_eps)
        self.o_proj = nn.Linear(self.v_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, DK, DV = self.H, self.DK, self.DV
        n_qkv = 2 * self.qk_dim + self.v_dim
        proj = self.in_proj(x)
        qkv, g, ab = proj[..., :n_qkv], proj[..., n_qkv : n_qkv + self.n_gate], proj[..., n_qkv + self.n_gate : n_qkv + self.n_gate + 2 * H]
        if self.conv is not None:
            qkv = self.conv(qkv)
        with torch.autocast(x.device.type, enabled=False):
            qkv = qkv.float()
            q = F.normalize(qkv[..., : self.qk_dim].view(B, T, H, DK), dim=-1) * DK**-0.5
            k = F.normalize(qkv[..., self.qk_dim : 2 * self.qk_dim].view(B, T, H, DK), dim=-1)
            v = qkv[..., 2 * self.qk_dim :].view(B, T, H, DV)
            ab = ab.float()
            la = -F.softplus(ab[..., :H] + self.dt_bias) * self.A_log.exp()
            beta = torch.sigmoid(ab[..., H:] + self.b_bias)
            p = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))
            u = write_direction(k, p, beta, F.softplus(self.lam.float()), self.n_steps)
            scan = nsgdn_scan if nsgdn_available(x) else nsgdn_ref
            y = scan(q, k, u, v, la)
        y = self.norm(y.to(x.dtype))
        if self.use_gate:
            y = y * F.silu(g).view(B, T, H, DV)
        return self.o_proj(y.reshape(B, T, -1))


@register("nsgdn")
def build_nsgdn(d_model: int, layer_idx: int, n_layer: int, **kwargs):
    return NStepGDN(d_model, layer_idx=layer_idx, **kwargs)
