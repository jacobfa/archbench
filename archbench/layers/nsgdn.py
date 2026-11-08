"""N-step gated delta rule (new.tex).

Per head: B = alpha S, r = v - B k, S = B + r u^T, y = S q.  The write direction u starts as the ordinary gated
delta step u_1 = beta k; the remaining N-1 steps descend lam/2 |(S - B) p|^2 with p = k_{t-1}, projected onto
updates with Delta S k = 0, i.e. u <- u - lam beta (p.u) p_perp, p_perp = p - (p.k) k.  u = beta k for N = 1
(plain GDN).  u is computed in PyTorch.

The recurrence S = alpha S (I - k u^T) + v u^T is a DPLR transition (diagonal alpha plus rank-1 -alpha u k^T on the
transposed state), so by default it runs on fla's chunked DPLR kernel (tensor cores, bf16 inputs, fp32 state) with
k_fla = b = u, a = -alpha k, gk = log alpha
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function, causal_conv1d_fwd_function
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
    from fla.ops.utils import chunk_local_cumsum, solve_tril
    from fla.ops.utils.constant import RCP_LN2
except ImportError:  # pragma: no cover
    chunk_dplr_delta_rule = None

CHUNK = 32  # tokens between stored states; the backward recomputes inside a chunk
ROWS = 16  # rows of S per program
NUM_WARPS = 4


def nsgdn_available(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda


def pick_scan(x: torch.Tensor, impl: str = "chunk"):
    """impl: 'chunk' (scalar-gate chunked kernel), 'dplr' (fla DPLR), 'triton' (fp32 token scan), 'ref'."""
    if impl == "ref" or not x.is_cuda:
        return nsgdn_ref
    if impl == "triton" or chunk_dplr_delta_rule is None:
        return nsgdn_scan if triton is not None else nsgdn_ref
    return nsgdn_scan_dplr if impl == "dplr" else nsgdn_scan_chunk


def write_direction(k, p, beta, lam, n_steps: int) -> torch.Tensor:
    """u_1 = beta k; then N-1 steps u <- u - lam beta (p.u) p_perp with p_perp = p - (p.k) k (new.tex).

    The projection keeps k.u = beta (current-key gain fixed) while each extra step shrinks the disturbance p.u
    to the previous read by (1 - lam beta |p_perp|^2).
    """
    b = beta[..., None]
    l = lam[None, None, :, None]
    u = b * k
    if n_steps > 1:
        p_perp = p - (p * k).sum(-1, keepdim=True) * k
        for _ in range(n_steps - 1):
            pu = (p * u).sum(-1, keepdim=True)
            u = u - l * b * pu * p_perp
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


def nsgdn_scan_dplr(q, k, u, v, la, la_min: float | None = None) -> torch.Tensor:
    """Same recurrence on fla's chunked DPLR kernel; y (B, T, H, DV) in bf16.

    With S' = S^T (DK x DV) the step is S' = (alpha I - alpha u k^T) S' + u v^T, y = S'^T q, i.e. fla's
    S' = (diag(exp gk) + b a^T) S' + k_fla v^T with k_fla = b = u, a = -alpha k, gk = la broadcast over DK, scale 1
    (q is already scaled by DK^-1/2 in the module).  If la >= la_min is guaranteed (la_min >= -2.6), fla can use
    64-token chunks and the tensor-core A stage; otherwise it falls back to 16-token chunks.
    """
    dt = torch.bfloat16
    a = -(la.exp()[..., None] * k)
    gk = la[..., None].expand_as(k)
    u = u.to(dt)
    y, _ = chunk_dplr_delta_rule(q.to(dt), u, v.to(dt), a.to(dt), u, gk, scale=1.0, lower_bound=la_min)
    return y


# --- chunked scan specialised to a scalar gate and separate read (k) / write (u) vectors -------------------------
# fla's GatedDeltaNet pipeline is S' = alpha S' + k (v - alpha S'^T k)^T beta with ONE vector k.  Ours is
# S' = alpha S' + u (v - alpha S'^T k)^T.  Within a 64-token chunk (gamma_t = prod alpha, T = (I + L)^-1):
#     L_ij = (k_i . u_j) gamma_i / gamma_j   (i > j),   w = T (k * gamma),   u_wy = T v,
#     r = u_wy - w S'_0,   S'_C = gamma_C S'_0 + sum_j (gamma_C / gamma_j) u_j r_j^T,
#     y_t = gamma_t S'_0^T q_t + sum_{j <= t} (gamma_t / gamma_j) (u_j . q_t) r_j,
# which is exactly fla's GDN pipeline with beta = 1 and the write vector u in place of k everywhere except the
# WY prep.  So the WY prep (L, T, and its backward) is ours; inter-chunk state, output and their backward reuse
# fla's tensor-core kernels.  fla's g is the chunk-local cumsum of la in log2 units, and dg is accumulated in
# natural units (coefficient * gamma) and reverse-cumsummed, as in fla.

BT_CHUNK = 64

if triton is not None:

    @triton.jit
    def _wy_L_kernel(k_ptr, u_ptr, g_ptr, L_ptr, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
        """L = tril(k u^T, -1) * exp2(g_i - g_j) for one (chunk, b, h); fp32 (B, T, H, BT)."""
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        o_j = tl.arange(0, BT)
        row = (i_b * T + o_t) * H + i_h  # token row index into (B, T, H, *)
        b_g = tl.load(g_ptr + row)
        b_M = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            b_k = tl.load(k_ptr + row[:, None] * K + o_k[None, :])
            b_u = tl.load(u_ptr + row[:, None] * K + o_k[None, :])
            b_M = tl.dot(b_k, tl.trans(b_u), b_M)
        m_A = o_j[:, None] > o_j[None, :]
        b_E = tl.where(m_A, tl.exp2(b_g[:, None] - b_g[None, :]), 0.0)
        tl.store(L_ptr + row[:, None] * BT + o_j[None, :], b_M * b_E)

    @triton.jit
    def _wy_bwd_kernel(
        k_ptr, u_ptr, v_ptr, g_ptr, A_ptr, dw_ptr, du_ptr, dk_ptr, duo_ptr, dv_ptr, dg_ptr, T,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        """Backward of w = A (k gamma), u_wy = A v with A = (I + L)^-1, L = tril(k u^T, -1) * E.  Mirrors fla's
        prepare_wy_repr_bwd_kernel with the write vector u in place of the second k.  dg in natural units."""
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        o_j = tl.arange(0, BT)
        row = (i_b * T + o_t) * H + i_h
        b_g = tl.load(g_ptr + row)
        b_ge = tl.exp2(b_g)
        b_At = tl.load(A_ptr + row[None, :] * BT + o_j[:, None])  # A^T (bf16)
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_dg = tl.zeros([BT], dtype=tl.float32)
        # pass 1 over K: dA += dw (k gamma)^T, dk_partial = A^T dw gamma, dg += rowsum(A^T dw * k gamma)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            b_k = tl.load(k_ptr + row[:, None] * K + o_k[None, :])
            b_dw = tl.load(dw_ptr + row[:, None] * K + o_k[None, :])
            b_kg = b_k * b_ge[:, None]
            b_dA = tl.dot(b_dw, tl.trans(b_kg).to(b_dw.dtype), b_dA)
            b_dkg = tl.dot(b_At, b_dw)
            b_dg += tl.sum(b_dkg * b_kg, 1)
            tl.store(dk_ptr + row[:, None] * K + o_k[None, :], (b_dkg * b_ge[:, None]).to(dk_ptr.dtype.element_ty))
        # pass over V: dA += du v^T, dv = A^T du
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            b_v = tl.load(v_ptr + row[:, None] * V + o_v[None, :])
            b_du = tl.load(du_ptr + row[:, None] * V + o_v[None, :])
            b_dA = tl.dot(b_du, tl.trans(b_v), b_dA)
            tl.store(dv_ptr + row[:, None] * V + o_v[None, :], tl.dot(b_At, b_du).to(dv_ptr.dtype.element_ty))
        # dL = -(A^T dA A^T) masked;  dM = dL * E
        m_A = o_j[:, None] > o_j[None, :]
        b_dA = tl.where(m_A, b_dA, 0.0)
        b_dA = tl.dot(b_dA.to(b_At.dtype), b_At)
        b_dA = tl.dot(b_At, b_dA.to(b_At.dtype))
        b_E = tl.where(m_A, tl.exp2(b_g[:, None] - b_g[None, :]), 0.0)
        b_dM = tl.where(m_A, -b_dA * b_E, 0.0)
        b_dMb = b_dM.to(b_At.dtype)
        # pass 2 over K: dk += dM u, du = dM^T k, and recompute M = k u^T for dg
        b_M = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            b_k = tl.load(k_ptr + row[:, None] * K + o_k[None, :])
            b_u = tl.load(u_ptr + row[:, None] * K + o_k[None, :])
            b_M = tl.dot(b_k, tl.trans(b_u), b_M)
            b_dk = tl.dot(b_dMb, b_u) + tl.load(dk_ptr + row[:, None] * K + o_k[None, :])
            tl.store(dk_ptr + row[:, None] * K + o_k[None, :], b_dk.to(dk_ptr.dtype.element_ty))
            b_du = tl.dot(tl.trans(b_dMb), b_k)
            tl.store(duo_ptr + row[:, None] * K + o_k[None, :], b_du.to(duo_ptr.dtype.element_ty))
        # dg += rowsum(dL * L) - colsum(dL * L), with dL * L = dM * M
        b_P = b_dM * b_M
        b_dg += tl.sum(b_P, 1) - tl.sum(b_P, 0)
        tl.store(dg_ptr + row, b_dg)


def _wy_fwd(k, u, g, BT):
    B, T, H, K = k.shape
    L = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    _wy_L_kernel[(T // BT, B * H)](k, u, g, L, T, H=H, K=K, BT=BT, BK=min(K, 64), num_warps=4)
    return solve_tril(L, output_dtype=k.dtype)  # (I + L)^-1 as (B, T, H, BT)


def _wy_bwd(k, u, v, g, A, dw, du_wy, BT):
    B, T, H, K = k.shape
    V = v.shape[-1]
    dk, du, dv, dg = torch.empty_like(k), torch.empty_like(u), torch.empty_like(v), torch.empty_like(g)
    _wy_bwd_kernel[(T // BT, B * H)](
        k, u, v, g, A, dw, du_wy, dk, du, dv, dg, T, H=H, K=K, V=V, BT=BT, BK=min(K, 64), BV=min(V, 64), num_warps=4
    )
    return dk, du, dv, dg


if chunk_dplr_delta_rule is not None:

    @torch.library.custom_op("archbench::nsgdn_chunk_fwd", mutates_args=())
    def _nsgdn_chunk_fwd(
        q: torch.Tensor, k: torch.Tensor, u: torch.Tensor, v: torch.Tensor, la: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        BT = BT_CHUNK
        g = chunk_local_cumsum(la, chunk_size=BT, scale=RCP_LN2)
        A = _wy_fwd(k, u, g, BT)
        ones = torch.ones_like(la, dtype=k.dtype)
        w, u_wy = recompute_w_u_fwd(k=k, v=v, beta=ones, A=A, g=g)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=u, w=w, u=u_wy, g=g, chunk_size=BT)
        y = chunk_fwd_o(q=q, k=u, v=v_new, h=h, g=g, scale=1.0, chunk_size=BT)
        return y, g, A

    @_nsgdn_chunk_fwd.register_fake
    def _(q, k, u, v, la):
        return torch.empty_like(v), torch.empty_like(la), k.new_empty(*k.shape[:3], BT_CHUNK)

    @torch.library.custom_op("archbench::nsgdn_chunk_bwd", mutates_args=())
    def _nsgdn_chunk_bwd(
        q: torch.Tensor, k: torch.Tensor, u: torch.Tensor, v: torch.Tensor, g: torch.Tensor, A: torch.Tensor, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        BT = BT_CHUNK
        ones = torch.ones_like(g, dtype=k.dtype)
        w, u_wy = recompute_w_u_fwd(k=k, v=v, beta=ones, A=A, g=g)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=u, w=w, u=u_wy, g=g, chunk_size=BT)
        dv_new = chunk_bwd_dv_local(q=q, k=u, g=g, do=dy, scale=1.0, chunk_size=BT)
        dh, _, dv_new = chunk_gated_delta_rule_bwd_dhu(q=q, k=u, w=w, g=g, do=dy, dv=dv_new, scale=1.0, chunk_size=BT)
        dq, du, dw, dg = chunk_bwd_dqkwg(q=q, k=u, v=v_new, w=w, g=g, h=h, dv=dv_new, do=dy, dh=dh, scale=1.0, chunk_size=BT)
        dk, du2, dv, dg2 = _wy_bwd(k, u, v, g, A, dw, dv_new, BT)
        dla = chunk_local_cumsum(dg + dg2, chunk_size=BT, reverse=True)
        return dq, dk, du + du2, dv, dla

    @_nsgdn_chunk_bwd.register_fake
    def _(q, k, u, v, g, A, dy):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(u), torch.empty_like(v), torch.empty_like(g)

    def _chunk_setup(ctx, inputs, output):
        q, k, u, v, la = inputs
        _, g, A = output
        ctx.save_for_backward(q, k, u, v, g, A)

    def _chunk_backward(ctx, dy, dg, dA):
        q, k, u, v, g, A = ctx.saved_tensors
        return _nsgdn_chunk_bwd(q, k, u, v, g, A, dy.contiguous())

    _nsgdn_chunk_fwd.register_autograd(_chunk_backward, setup_context=_chunk_setup)


def nsgdn_scan_chunk(q, k, u, v, la) -> torch.Tensor:
    """Chunked scalar-gate kernel (see above); bf16 inputs/outputs, fp32 state and gate."""
    if q.shape[1] % BT_CHUNK:
        return nsgdn_scan_dplr(q, k, u, v, la)
    dt = torch.bfloat16
    return _nsgdn_chunk_fwd(q.to(dt).contiguous(), k.to(dt).contiguous(), u.to(dt).contiguous(), v.to(dt).contiguous(), la.contiguous().float())[0]


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


if causal_conv1d_fn is not None:
    # causal_conv1d's own autograd.Function calls its C++ op with `out=`, which breaks torch.compile graphs; these
    # custom ops wrap the same kernels as a single opaque op (x, y, dy are (B, D, T) channel-last views).

    def _channel_last(x: torch.Tensor) -> torch.Tensor:
        return x if x.stride(1) == 1 else x.transpose(1, 2).contiguous().transpose(1, 2)

    @torch.library.custom_op("archbench::causal_conv_silu", mutates_args=())
    def _causal_conv_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return causal_conv1d_fwd_function(_channel_last(x), weight, None, None, None, None, True)

    @_causal_conv_silu.register_fake
    def _(x, weight):
        B, D, T = x.shape
        return torch.empty(B, T, D, dtype=x.dtype, device=x.device).transpose(1, 2)

    @torch.library.custom_op("archbench::causal_conv_silu_bwd", mutates_args=())
    def _causal_conv_silu_bwd(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _channel_last(x)
        dx, dw, _, _ = causal_conv1d_bwd_function(x, weight, None, _channel_last(dy), None, None, None, None, False, True)
        return dx, dw

    @_causal_conv_silu_bwd.register_fake
    def _(x, weight, dy):
        B, D, T = x.shape
        return torch.empty(B, T, D, dtype=x.dtype, device=x.device).transpose(1, 2), torch.empty_like(weight)

    def _conv_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def _conv_backward(ctx, dy):
        x, weight = ctx.saved_tensors
        return _causal_conv_silu_bwd(x, weight, dy)

    _causal_conv_silu.register_autograd(_conv_backward, setup_context=_conv_setup)


class ShortConv(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.empty(dim, 1, size).uniform_(-1 / math.sqrt(size), 1 / math.sqrt(size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if causal_conv1d_fn is not None and x.is_cuda:
            # (B, T, D) -> channel-last (B, D, T) view, as causal_conv1d expects; the kernels keep that layout
            y = _causal_conv_silu(x.transpose(1, 2), self.weight.squeeze(1).to(x.dtype))
            return y.transpose(1, 2)
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
        scan_impl: str = "chunk",
        la_min: float | None = None,
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
        if scan_impl not in ("chunk", "dplr", "triton", "ref"):
            raise ValueError(f"unknown scan_impl {scan_impl!r}")
        self.scan_impl = scan_impl
        self.la_min = la_min  # optional floor on the log-retention (only needed for the dplr path's 64-token chunks)
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
            if self.la_min is not None:
                la = la.clamp_min(self.la_min)
            beta = torch.sigmoid(ab[..., H:] + self.b_bias)
            p = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))
            u = write_direction(k, p, beta, F.softplus(self.lam.float()), self.n_steps)
            scan = pick_scan(x, self.scan_impl)
            y = scan(q, k, u, v, la, la_min=self.la_min) if scan is nsgdn_scan_dplr else scan(q, k, u, v, la)
        y = self.norm(y.to(x.dtype))
        if self.use_gate:
            y = y * F.silu(g).view(B, T, H, DV)
        return self.o_proj(y.reshape(B, T, -1))


@register("nsgdn")
def build_nsgdn(d_model: int, layer_idx: int, n_layer: int, **kwargs):
    return NStepGDN(d_model, layer_idx=layer_idx, **kwargs)
