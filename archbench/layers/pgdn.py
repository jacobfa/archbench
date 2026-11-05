"""Gated DeltaNet whose write is n steps of test-time training: one gradient step (the GDN delta rule) followed by
n - 1 greedy coordinate steps (matching pursuit) on the discounted ridge loss of the same memory.

Per head the memory is M (D x D fast weights, value x key). Alongside it we keep the ridge statistics
    A = sum (decayed) beta k k^T + lam I,   C = sum (decayed) beta v k^T,   P = M A   (P is kept up to date by rank-1 updates).
Token t with q, k, v in R^D, retention alpha and write strength beta:
    p = alpha M k;  e = v - p;  M <- alpha M + beta e k^T                                  (GDN: one gradient step)
    A <- alpha A + beta k k^T + (1 - alpha) lam I;  C <- alpha C + beta v k^T
    repeat n_steps - 1 times:  R = C - P;  j = argmax_j |R[:, j]|^2 / A_jj;  d = R[:, j] / A_jj;  M[:, j] += d;  P += d A[j, :]
    y = M q
Each pursuit step is the exact minimiser of the ridge loss 1/2 tr(M A M^T) - tr(M C^T) along the coordinate column j
with the largest decrease (Gauss-Southwell), so n_steps = 1 is plain GDN and n_steps -> inf converges to the ridge solution C A^{-1}.

Kernels (Triton, fp32 states in registers, sequential over time):
  forward      - one program per (batch, head).  Stores y, the error e, (A k), and per pursuit step (j, d, C[:, j],
                 A[j, :], A_jj); the (M, A) state at every CHUNK tokens.
  backward     - the argmax is treated as constant, which makes the adjoint linear.  Given A and the pursuit choices the
                 rows of M, P, C (and their adjoints) are independent, so `_bwd_rows_kernel` runs one single-warp program
                 per slab of ROWS rows (every reduction is a warp shuffle).  Each chunk is recomputed forward from its
                 boundary into a small scratch with the reduction-free recurrence (e and d are stored), then swept in
                 reverse.  Sums over all rows (gq, gk, d loss / d log alpha, d beta, and the injections into the adjoint
                 of A) are written as per-slab partials and summed on the host.  `_gram_bwd_kernel` then sweeps the
                 adjoint of A, the only state whose adjoint couples all rows.  <gP, P> and <gC, C> are tracked by scalar
                 recurrences, so P and C are never materialised in the backward.
Gradients are taken w.r.t. log alpha (no division by alpha anywhere).
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

CHUNK = 8  # tokens between stored (M, A) states; the backward recomputes inside a chunk


def pgdn_available(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda


if triton is not None:

    @triton.jit
    def _pick(row, jj, i: tl.constexpr):
        """Entry i of a packed [1, D] row as a [1, 1] tensor."""
        return tl.sum(tl.where(jj == i, row, 0.0), 1, keep_dims=True)

    @triton.jit
    def _pgdn_fwd_kernel(
        q_ptr, k_ptr, v_ptr, la_ptr, b_ptr, y_ptr, e_ptr, j_ptr, d_ptr, c_ptr, s_ptr, a_ptr, ar_ptr, dj_ptr, lam_ptr, T, H, NCH,
        D: tl.constexpr, NP: tl.constexpr, CT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b, h = pid // H, pid % H
        lam = tl.load(lam_ptr + h)
        ii = tl.arange(0, D)[:, None]
        jj = tl.arange(0, D)[None, :]
        eye = ii == jj
        # the initial state (0 and lam I, written by the host) is loaded rather than built in registers: the tile-shaped
        # load pins the tile layout (rows split across warps), which makes every row reduction a warp-local shuffle
        M = tl.load(s_ptr + pid * NCH * 2 * D * D + ii * D + jj)
        A = tl.load(s_ptr + (pid * NCH * 2 + 1) * D * D + ii * D + jj)
        P = M * 0.0
        C = M * 0.0
        Dg = tl.sum(tl.where(eye, A, 0.0), 0, keep_dims=True)  # diag(A) as a [1, D] row, kept by its own recurrence
        tok = (b * T) * H + h
        # inputs of the next token are loaded one iteration ahead (the chain is latency bound)
        k_r = tl.load(k_ptr + tok * D + jj)
        k_c = tl.load(k_ptr + tok * D + ii)
        q_r = tl.load(q_ptr + tok * D + jj)
        v_c = tl.load(v_ptr + tok * D + ii)
        la = tl.load(la_ptr + tok)
        be = tl.load(b_ptr + tok)
        for c in range(NCH):
            if c > 0:
                sb = s_ptr + (pid * NCH + c) * 2 * D * D + ii * D + jj
                tl.store(sb, M)
                if NP > 0:
                    tl.store(sb + D * D, A)
            for i in range(CT):
                t = c * CT + i
                base = tok * D
                nxt = tok + H
                more = t + 1 < T
                k_rn = tl.load(k_ptr + nxt * D + jj, mask=more & (jj >= 0), other=0.0)
                k_cn = tl.load(k_ptr + nxt * D + ii, mask=more & (ii >= 0), other=0.0)
                q_rn = tl.load(q_ptr + nxt * D + jj, mask=more & (jj >= 0), other=0.0)
                v_cn = tl.load(v_ptr + nxt * D + ii, mask=more & (ii >= 0), other=0.0)
                la_n = tl.load(la_ptr + nxt, mask=more, other=0.0)
                be_n = tl.load(b_ptr + nxt, mask=more, other=0.0)
                al = tl.exp(la)
                Mp = al * M
                p = tl.sum(Mp * k_r, 1, keep_dims=True)  # [D, 1] = alpha M k
                e = v_c - p
                M = Mp + be * e * k_r
                tl.store(e_ptr + base + ii, e)
                if NP > 0:
                    A = al * A + be * k_c * k_r + tl.where(eye, (1.0 - al) * lam, 0.0)
                    a_r = tl.sum(A * k_c, 0, keep_dims=True)  # [1, D] = (A k)^T
                    tl.store(a_ptr + base + jj, a_r)
                    Dg = al * Dg + be * k_r * k_r + (1.0 - al) * lam
                    P = (al * al) * P + be * p * k_r + ((1.0 - al) * lam) * Mp + be * e * a_r
                    C = al * C + be * v_c * k_r
                    for s in tl.static_range(NP):
                        R = C - P
                        score = tl.sum(R * R, 0, keep_dims=True) / Dg
                        j = tl.argmax(score, 1, keep_dims=True)  # [1, 1]
                        selj = jj == j
                        Dj = tl.sum(tl.where(selj, Dg, 0.0), 1, keep_dims=True)
                        d = tl.sum(tl.where(selj, R, 0.0), 1, keep_dims=True) / Dj  # [D, 1]
                        arow = tl.sum(tl.where(ii == j, A, 0.0), 0, keep_dims=True)  # [1, D]
                        M = M + tl.where(selj, d, 0.0)
                        P = P + d * arow
                        tl.store(j_ptr + tok * NP + s + tl.zeros([1, 1], dtype=tl.int32), j)
                        tl.store(dj_ptr + tok * NP + s + tl.zeros([1, 1], dtype=tl.int32), Dj)
                        tl.store(ar_ptr + (tok * NP + s) * D + jj, arow)
                        tl.store(d_ptr + (tok * NP + s) * D + ii, d)
                        tl.store(c_ptr + (tok * NP + s) * D + ii, tl.sum(tl.where(selj, C, 0.0), 1, keep_dims=True))
                tl.store(y_ptr + base + ii, tl.sum(M * q_r, 1, keep_dims=True))
                tok = nxt
                k_r, k_c, q_r, v_c, la, be = k_rn, k_cn, q_rn, v_cn, la_n, be_n

    @triton.jit
    def _bwd_rows_kernel(
        q_ptr, k_ptr, v_ptr, la_ptr, b_ptr, e_ptr, j_ptr, d_ptr, c_ptr, s_ptr, a_ptr, ar_ptr, dj_ptr, gy_ptr, sc_ptr,
        gv_ptr, gq_ptr, gk_ptr, ga_ptr, ru_ptr, dg_ptr, gl_ptr, gb_ptr, glam_ptr, lam_ptr, T, H, NCH, R,
        D: tl.constexpr, BV: tl.constexpr, NP: tl.constexpr, CT: tl.constexpr,
    ):
        """Reverse sweep over a slab of BV rows of M / P / C and their adjoints (rows are independent given A and the
        pursuit choices).  Each chunk is first recomputed forward into a per-program scratch, then swept in reverse.
        Row sums that other slabs also contribute to are written as per-slab partials."""
        pid = tl.program_id(0)
        bh, r = pid // R, pid % R
        b, h = bh // H, bh % H
        lam = tl.load(lam_ptr + h)
        rl = tl.arange(0, BV)[:, None]
        ri = r * BV + rl
        jj = tl.arange(0, D)[None, :]
        scb = sc_ptr + pid * CT * BV * D + rl * D + jj
        M = tl.load(s_ptr + (bh * NCH) * 2 * D * D + ri * D + jj)
        gM = M * 0.0
        gP = M * 0.0
        gC = M * 0.0
        tok = (b * T + T - 1) * H + h
        q_r = tl.load(q_ptr + tok * D + jj + rl * 0)
        k_r = tl.load(k_ptr + tok * D + jj + rl * 0)
        a_r = tl.load(a_ptr + tok * D + jj + rl * 0, mask=(jj >= 0) & (NP > 0), other=0.0)
        v_c = tl.load(v_ptr + tok * D + ri)
        e_c = tl.load(e_ptr + tok * D + ri)
        gy_c = tl.load(gy_ptr + tok * D + ri)
        la = tl.load(la_ptr + tok)
        be = tl.load(b_ptr + tok)
        # <gP, P> and <gC, C> restricted to each row: every reduction in this kernel is over a row, so the per-row
        # pieces of d loss / d log alpha, d beta and the Gram injection are written as columns and summed by the host
        psP = gy_c * 0.0
        psC = gy_c * 0.0
        glam = gy_c * 0.0  # d loss / d lam through the (1 - alpha) lam M' term of P, per row
        for ci in range(NCH):
            c = NCH - 1 - ci
            # pass 1: recompute the slab of M after every token of the chunk (rank-1 updates only)
            M = tl.load(s_ptr + (bh * NCH + c) * 2 * D * D + ri * D + jj)
            toks = (b * T + c * CT) * H + h
            k_rs = tl.load(k_ptr + toks * D + jj)
            e_cs = tl.load(e_ptr + toks * D + ri)
            la_s = tl.load(la_ptr + toks)
            be_s = tl.load(b_ptr + toks)
            for i in range(CT):
                ntoks = toks + H
                more_s = i + 1 < CT
                k_rsn = tl.load(k_ptr + ntoks * D + jj, mask=more_s & (jj >= 0), other=0.0)
                e_csn = tl.load(e_ptr + ntoks * D + ri, mask=more_s & (ri >= 0), other=0.0)
                la_sn = tl.load(la_ptr + ntoks, mask=more_s, other=0.0)
                be_sn = tl.load(b_ptr + ntoks, mask=more_s, other=0.0)
                M = tl.exp(la_s) * M + be_s * e_cs * k_rs
                for sp in tl.static_range(NP):
                    jx = tl.load(j_ptr + toks * NP + sp)
                    M = M + tl.where(jj == jx, tl.load(d_ptr + (toks * NP + sp) * D + ri), 0.0)
                if i + 1 < CT:
                    tl.store(scb + i * BV * D, M)
                toks = ntoks
                k_rs, e_cs, la_s, be_s = k_rsn, e_csn, la_sn, be_sn
            tl.debug_barrier()
            # pass 2: adjoints, newest token first (M of the last token is still in registers)
            for ir in range(CT):
                i = CT - 1 - ir
                t = c * CT + i
                nxt = tok - H
                more = t > 0
                q_rn = tl.load(q_ptr + nxt * D + jj + rl * 0, mask=more & (jj >= 0), other=0.0)
                k_rn = tl.load(k_ptr + nxt * D + jj + rl * 0, mask=more & (jj >= 0), other=0.0)
                a_rn = tl.load(a_ptr + nxt * D + jj + rl * 0, mask=more & (jj >= 0) & (NP > 0), other=0.0)
                v_cn = tl.load(v_ptr + nxt * D + ri, mask=more & (ri >= 0), other=0.0)
                e_cn = tl.load(e_ptr + nxt * D + ri, mask=more & (ri >= 0), other=0.0)
                gy_cn = tl.load(gy_ptr + nxt * D + ri, mask=more & (ri >= 0), other=0.0)
                la_n = tl.load(la_ptr + nxt, mask=more, other=0.0)
                be_n = tl.load(b_ptr + nxt, mask=more, other=0.0)
                Mn = tl.load(scb + (i - 1) * BV * D, mask=(i > 0) & (rl >= 0), other=0.0)
                al = tl.exp(la)
                gM = gM + gy_c * q_r
                tl.atomic_add(gq_ptr + tok * D + jj, tl.sum(M * gy_c, 0, keep_dims=True), sem="relaxed")
                for sp in tl.static_range(NP):
                    s = NP - 1 - sp
                    jx = tl.load(j_ptr + tok * NP + s)
                    Dj = tl.load(dj_ptr + tok * NP + s)
                    d_c = tl.load(d_ptr + (tok * NP + s) * D + ri)
                    cc = tl.load(c_ptr + (tok * NP + s) * D + ri)
                    arow = tl.load(ar_ptr + (tok * NP + s) * D + jj + rl * 0)
                    selj = jj == jx
                    gPa = tl.sum(gP * arow, 1, keep_dims=True)
                    gd = gPa + tl.sum(tl.where(selj, gM, 0.0), 1, keep_dims=True)
                    tl.atomic_add(ru_ptr + (tok * NP + s) * D + jj, tl.sum(gP * d_c, 0, keep_dims=True), sem="relaxed")
                    dgd = d_c * gd
                    tl.store(dg_ptr + (tok * NP + s) * D + ri, dgd)
                    gR = gd / Dj
                    gC = gC + tl.where(selj, gR, 0.0)
                    gP = gP - tl.where(selj, gR, 0.0)
                    cgd = cc * gd
                    psC = psC + cgd / Dj
                    psP = psP - d_c * gPa - cgd / Dj + dgd
                    M = M - tl.where(selj, d_c, 0.0)
                p_c = v_c - e_c
                Mp = M - be * e_c * k_r
                gMk = tl.sum(gM * k_r, 1, keep_dims=True)
                ge = be * gMk
                gb = e_c * gMk
                if NP > 0:
                    gCk = tl.sum(gC * k_r, 1, keep_dims=True)
                    gPk = tl.sum(gP * k_r, 1, keep_dims=True)
                    gPa2 = tl.sum(gP * a_r, 1, keep_dims=True)
                    gPMp = tl.sum(gP * Mp, 1, keep_dims=True)
                    tl.atomic_add(ga_ptr + tok * D + jj, be * tl.sum(gP * e_c, 0, keep_dims=True), sem="relaxed")
                    ge = ge + be * gPa2
                    gp = be * gPk - ge
                    gMp = ((1.0 - al) * lam) * gP + gM + gp * k_r
                    s0 = v_c * gCk
                    pgPk = p_c * gPk
                    egPa = e_c * gPa2
                    s1 = be * pgPk + ((1.0 - al) * lam) * gPMp + be * egPa
                    gla = psC - be * s0 + 2.0 * (psP - s1) + tl.sum(gMp * Mp, 1, keep_dims=True) - (al * lam) * gPMp
                    gb = gb + s0 + pgPk + egPa
                    glam = glam + (1.0 - al) * gPMp
                    psC = psC - be * s0
                    psP = psP - s1
                    gv = be * gCk + ge
                    gk = tl.sum(be * gC * v_c + be * gP * p_c + be * gM * e_c + Mp * gp, 0, keep_dims=True)
                    gC = al * gC
                    gP = (al * al) * gP
                    gM = al * gMp
                else:
                    gp = -ge
                    gMp = gM + gp * k_r
                    gla = tl.sum(gMp * Mp, 1, keep_dims=True)
                    gv = ge
                    gk = tl.sum(be * gM * e_c + Mp * gp, 0, keep_dims=True)
                    gM = al * gMp
                tl.store(gv_ptr + tok * D + ri, gv)
                tl.atomic_add(gk_ptr + tok * D + jj, gk, sem="relaxed")
                tl.store(gl_ptr + tok * D + ri, gla)
                tl.store(gb_ptr + tok * D + ri, gb)
                tok = nxt
                M = Mn
                q_r, k_r, a_r, v_c, e_c, gy_c, la, be = q_rn, k_rn, a_rn, v_cn, e_cn, gy_cn, la_n, be_n
        if NP > 0:
            tl.atomic_add(glam_ptr + h, tl.sum(glam), sem="relaxed")

    @triton.jit
    def _gram_bwd_kernel(
        k_ptr, la_ptr, b_ptr, j_ptr, s_ptr, dj_ptr, ga_ptr, ru_ptr, dg_ptr, sc_ptr, gk_ptr, gl_ptr, gb_ptr, glam_ptr, lam_ptr, T, H, NCH, R,
        D: tl.constexpr, BV: tl.constexpr, NP: tl.constexpr, CT: tl.constexpr,
    ):
        """Reverse sweep of the adjoint of A over a slab of BV rows (its injections are already summed over all rows).
        Its contributions to gk, d log alpha and d beta are accumulated onto the row kernel's results with atomics."""
        pid = tl.program_id(0)
        bh, r = pid // R, pid % R
        b, h = bh // H, bh % H
        lam = tl.load(lam_ptr + h)
        rl = tl.arange(0, BV)[:, None]
        ri = r * BV + rl
        jj = tl.arange(0, D)[None, :]
        eye = ri == jj
        scb = sc_ptr + pid * CT * BV * D + rl * D + jj
        A = tl.load(s_ptr + (bh * NCH * 2 + 1) * D * D + ri * D + jj)
        gA = A * 0.0
        glam = 0.0  # d loss / d lam through the (1 - alpha) lam I term of A and the initial state lam I
        tok = (b * T + T - 1) * H + h
        k_r = tl.load(k_ptr + tok * D + jj)
        k_c = tl.load(k_ptr + tok * D + ri)
        ga_r = tl.load(ga_ptr + tok * D + jj)
        ga_c = tl.load(ga_ptr + tok * D + ri)
        la = tl.load(la_ptr + tok)
        be = tl.load(b_ptr + tok)
        for ci in range(NCH):
            c = NCH - 1 - ci
            A = tl.load(s_ptr + (bh * NCH + c) * 2 * D * D + D * D + ri * D + jj)
            toks = (b * T + c * CT) * H + h
            k_rs = tl.load(k_ptr + toks * D + jj)
            k_cs = tl.load(k_ptr + toks * D + ri)
            la_s = tl.load(la_ptr + toks)
            be_s = tl.load(b_ptr + toks)
            for i in range(CT):
                ntoks = toks + H
                more_s = i + 1 < CT
                k_rsn = tl.load(k_ptr + ntoks * D + jj, mask=more_s & (jj >= 0), other=0.0)
                k_csn = tl.load(k_ptr + ntoks * D + ri, mask=more_s & (ri >= 0), other=0.0)
                la_sn = tl.load(la_ptr + ntoks, mask=more_s, other=0.0)
                be_sn = tl.load(b_ptr + ntoks, mask=more_s, other=0.0)
                al_s = tl.exp(la_s)
                A = al_s * A + be_s * k_cs * k_rs + tl.where(eye, (1.0 - al_s) * lam, 0.0)
                if i + 1 < CT:
                    tl.store(scb + i * BV * D, A)
                toks = ntoks
                k_rs, k_cs, la_s, be_s = k_rsn, k_csn, la_sn, be_sn
            tl.debug_barrier()
            for ir in range(CT):
                i = CT - 1 - ir
                t = c * CT + i
                nxt = tok - H
                more = t > 0
                k_rn = tl.load(k_ptr + nxt * D + jj, mask=more & (jj >= 0), other=0.0)
                k_cn = tl.load(k_ptr + nxt * D + ri, mask=more & (ri >= 0), other=0.0)
                ga_rn = tl.load(ga_ptr + nxt * D + jj, mask=more & (jj >= 0), other=0.0)
                ga_cn = tl.load(ga_ptr + nxt * D + ri, mask=more & (ri >= 0), other=0.0)
                la_n = tl.load(la_ptr + nxt, mask=more, other=0.0)
                be_n = tl.load(b_ptr + nxt, mask=more, other=0.0)
                An = tl.load(scb + (i - 1) * BV * D, mask=(i > 0) & (rl >= 0), other=0.0)
                al = tl.exp(la)
                for sp in tl.static_range(NP):
                    s = NP - 1 - sp
                    jx = tl.load(j_ptr + tok * NP + s)
                    Dj = tl.load(dj_ptr + tok * NP + s)
                    ru = tl.load(ru_ptr + (tok * NP + s) * D + jj)
                    dgd = tl.load(dg_ptr + tok * NP + s)
                    seli = ri == jx
                    gA = gA + tl.where(seli, ru, 0.0) + tl.where(eye & seli, -dgd / Dj, 0.0)
                gA = gA + ga_c * k_r
                gAk_c = tl.sum(gA * k_r, 1, keep_dims=True)  # own rows of gA k
                Aga_c = tl.sum(A * ga_r, 1, keep_dims=True)  # own rows of A ga
                tl.atomic_add(gk_ptr + tok * D + ri, be * gAk_c + Aga_c, sem="relaxed")
                tl.atomic_add(gk_ptr + tok * D + jj, be * tl.sum(gA * k_c, 0, keep_dims=True), sem="relaxed")  # slab part of gA^T k
                kgAk = tl.sum(k_c * gAk_c)
                trgA = tl.sum(tl.where(eye, gA, 0.0))
                tl.atomic_add(gl_ptr + tok, tl.sum(gA * A) - lam * trgA - be * kgAk, sem="relaxed")
                tl.atomic_add(gb_ptr + tok, kgAk, sem="relaxed")
                glam = glam + (1.0 - al) * trgA
                gA = al * gA
                tok = nxt
                A = An
                k_r, k_c, ga_r, ga_c, la, be = k_rn, k_cn, ga_rn, ga_cn, la_n, be_n
        glam = glam + tl.sum(tl.where(eye, gA, 0.0))  # gA is now the adjoint of the initial state lam I
        tl.atomic_add(glam_ptr + h, glam, sem="relaxed")

    FWD_WARPS = 4

    def _chunk(T: int) -> int:
        for c in (CHUNK, 16, 8, 4, 2, 1):
            if c <= CHUNK and T % c == 0:
                return c
        return 1

    @torch.library.custom_op("archbench::pgdn_fwd", mutates_args=())
    def _pgdn_fwd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, beta: torch.Tensor, lam: torch.Tensor, n_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, D = q.shape
        NP = n_steps - 1
        CT = _chunk(T)
        NCH = T // CT
        y, e, a = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
        j = torch.empty(B, T, H, NP, dtype=torch.int32, device=q.device)
        d = torch.empty(B, T, H, NP, D, dtype=torch.float32, device=q.device)
        cc, ar = torch.empty_like(d), torch.empty_like(d)
        dj = torch.empty(B, T, H, NP, dtype=torch.float32, device=q.device)
        st = torch.empty(B, H, NCH, 2, D, D, dtype=torch.float32, device=q.device)
        st[:, :, 0, 0].zero_()
        st[:, :, 0, 1] = lam[:, None, None] * torch.eye(D, device=q.device)
        _pgdn_fwd_kernel[(B * H,)](q, k, v, la, beta, y, e, j, d, cc, st, a, ar, dj, lam, T, H, NCH, D=D, NP=NP, CT=CT, num_warps=FWD_WARPS)
        return y, e, j, d, cc, st, a, ar, dj

    @_pgdn_fwd.register_fake
    def _(q, k, v, la, beta, lam, n_steps):
        B, T, H, D = q.shape
        NP = n_steps - 1
        NCH = T // _chunk(T)
        return (
            torch.empty_like(q),
            torch.empty_like(q),
            q.new_empty(B, T, H, NP, dtype=torch.int32),
            q.new_empty(B, T, H, NP, D),
            q.new_empty(B, T, H, NP, D),
            q.new_empty(B, H, NCH, 2, D, D),
            torch.empty_like(q),
            q.new_empty(B, T, H, NP, D),
            q.new_empty(B, T, H, NP),
        )

    ROWS = 8  # rows of M per backward program (one warp each)
    GRAM_ROWS = 16  # rows of A per Gram-adjoint program

    @torch.library.custom_op("archbench::pgdn_bwd", mutates_args=())
    def _pgdn_bwd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, beta: torch.Tensor, e: torch.Tensor, j: torch.Tensor,
        d: torch.Tensor, cc: torch.Tensor, st: torch.Tensor, a: torch.Tensor, ar: torch.Tensor, dj: torch.Tensor, gy: torch.Tensor, lam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, D = q.shape
        NP = j.shape[-1]
        NCH = st.shape[2]
        CT = T // NCH
        BV = min(ROWS, D)
        R = D // BV
        gy = gy.contiguous()
        dev = q.device
        gv = torch.empty_like(q)
        gq = torch.zeros_like(q)  # row sums over all slabs are accumulated with atomics
        gk, ga = torch.zeros_like(q), torch.zeros_like(q)
        ru = torch.zeros(B, T, H, NP, D, device=dev)
        dg_p = torch.empty(B, T, H, NP, D, device=dev)
        gl_p = torch.empty(B, T, H, D, device=dev)
        gb_p = torch.empty_like(gl_p)
        glam = torch.zeros_like(lam)
        sc = torch.empty(B * H * R, CT, BV, D, device=dev)
        _bwd_rows_kernel[(B * H * R,)](
            q, k, v, la, beta, e, j, d, cc, st, a, ar, dj, gy, sc, gv, gq, gk, ga, ru, dg_p, gl_p, gb_p, glam, lam, T, H, NCH, R,
            D=D, BV=BV, NP=NP, CT=CT, num_warps=1,
        )
        gla, gb = gl_p.sum(3), gb_p.sum(3)  # per-row pieces
        if NP > 0:
            dg = dg_p.sum(4)
            BG = min(GRAM_ROWS, D)
            RG = D // BG
            scg = torch.empty(B * H * RG, CT, BG, D, device=dev)
            _gram_bwd_kernel[(B * H * RG,)](
                k, la, beta, j, st, dj, ga, ru, dg, scg, gk, gla, gb, glam, lam, T, H, NCH, RG, D=D, BV=BG, NP=NP, CT=CT, num_warps=1
            )
        return gq, gk, gv, gla, gb, glam

    @_pgdn_bwd.register_fake
    def _(q, k, v, la, beta, e, j, d, cc, st, a, ar, dj, gy, lam):
        return torch.empty_like(q), torch.empty_like(q), torch.empty_like(q), torch.empty_like(la), torch.empty_like(beta), torch.empty_like(lam)

    def _setup(ctx, inputs, output):
        q, k, v, la, beta, lam, n_steps = inputs
        _, e, j, d, cc, st, a, ar, dj = output
        ctx.save_for_backward(q, k, v, la, beta, lam, e, j, d, cc, st, a, ar, dj)

    def _backward(ctx, gy, ge, gj, gd, gcc, gst, ga, gar, gdj):
        q, k, v, la, beta, lam, e, j, d, cc, st, a, ar, dj = ctx.saved_tensors
        gq, gk, gv, gla, gb, glam = _pgdn_bwd(q, k, v, la, beta, e, j, d, cc, st, a, ar, dj, gy, lam)
        return gq, gk, gv, gla, gb, glam, None

    _pgdn_fwd.register_autograd(_backward, setup_context=_setup)


def pgdn_scan(q, k, v, la, beta, lam, n_steps: int) -> torch.Tensor:
    """y (B, T, H, D) fp32 for q, k, v (B, T, H, D), log-retention la and write strength beta (B, T, H), ridge lam (H,).
    D must be a power of 2."""
    return _pgdn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), la.contiguous(), beta.contiguous(), lam.float().contiguous(), int(n_steps))[0]


def pgdn_ref(q, k, v, la, beta, lam, n_steps: int) -> torch.Tensor:
    """Sequential reference (same math as the kernel; autograd treats the argmax as constant)."""
    B, T, H, D = q.shape
    eye = torch.eye(D, dtype=q.dtype, device=q.device)
    lam = lam.to(q.dtype)[:, None, None]  # (H, 1, 1)
    M = q.new_zeros(B, H, D, D)
    A = eye * lam
    C = torch.zeros_like(M)
    ys = []
    for t in range(T):
        al = la[:, t].exp()[..., None, None]
        be = beta[:, t][..., None, None]
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        Mp = al * M
        p = (Mp @ kt[..., None]).squeeze(-1)
        M = Mp + be * (vt - p)[..., :, None] * kt[..., None, :]
        A = al * A + be * kt[..., :, None] * kt[..., None, :] + (1 - al) * lam * eye
        C = al * C + be * vt[..., :, None] * kt[..., None, :]
        for _ in range(n_steps - 1):
            R = C - M @ A
            Dg = A.diagonal(0, -2, -1)
            jx = ((R * R).sum(-2) / Dg).argmax(-1)  # (B, H)
            col = R.gather(-1, jx[..., None, None].expand(B, H, D, 1)).squeeze(-1)
            d = col / Dg.gather(-1, jx[..., None])
            M = M + d[..., :, None] * eye[jx][..., None, :]
        ys.append((M @ qt[..., None]).squeeze(-1))
    return torch.stack(ys, 1)


class ShortConv(nn.Module):
    """Depthwise causal conv + SiLU over (B, T, dim); the fused CUDA kernel reads the channels-last layout directly."""

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



class PursuitGDN(nn.Module):
    """Gated DeltaNet block (fused q|k|v|gate|a|b projection, short conv, gated RMSNorm) with the n-step pursuit write."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: int = 64,
        n_steps: int = 2,
        ridge: float = 1.0,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
    ):
        super().__init__()
        if head_dim & (head_dim - 1):
            raise ValueError("head_dim must be a power of 2")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.H, self.D, self.n_steps = num_heads, head_dim, n_steps
        self.inner = num_heads * head_dim
        self.use_gate = use_gate
        self.n_gate = self.inner if use_gate else 0
        width = 3 * self.inner + self.n_gate + 2 * num_heads
        self.in_proj = nn.Linear(d_model, (width + 7) // 8 * 8, bias=False)
        self.conv = ShortConv(3 * self.inner, conv_size) if use_short_conv else None
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.b_bias = nn.Parameter(torch.zeros(num_heads))
        self.lam = nn.Parameter(torch.full((num_heads,), math.log(math.expm1(ridge))))  # per-head ridge, softplus(lam) = ridge at init
        self.norm = nn.RMSNorm(head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, I = self.H, self.D, self.inner
        proj = self.in_proj(x)
        qkv, g, ab = proj[..., : 3 * I], proj[..., 3 * I : 3 * I + self.n_gate], proj[..., 3 * I + self.n_gate : 3 * I + self.n_gate + 2 * H]
        if self.conv is not None:
            qkv = self.conv(qkv)
        with torch.autocast(x.device.type, enabled=False):
            q, k, v = qkv.float().view(B, T, 3, H, D).unbind(2)
            q = F.normalize(q, dim=-1) * D**-0.5
            k = F.normalize(k, dim=-1)
            ab = ab.float()
            la = -F.softplus(ab[..., :H] + self.dt_bias) * self.A_log.exp()
            beta = torch.sigmoid(ab[..., H:] + self.b_bias)
            scan = pgdn_scan if pgdn_available(x) else pgdn_ref
            y = scan(q, k, v, la, beta, F.softplus(self.lam.float()), self.n_steps)
        y = self.norm(y.to(x.dtype))
        if self.use_gate:
            y = y * F.silu(g).view(B, T, H, D)
        return self.o_proj(y.reshape(B, T, -1))


@register("pgdn")
def build_pgdn(d_model: int, layer_idx: int, n_layer: int, **kwargs):
    return PursuitGDN(d_model, layer_idx=layer_idx, **kwargs)
