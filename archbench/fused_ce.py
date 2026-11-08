"""Chunked linear + cross-entropy that never materialises the full (N, V) logits.

Like fla's FusedLinearCrossEntropyLoss, but the weight gradient is accumulated with bf16 GEMMs writing fp32
(torch.addmm out_dtype) instead of upcasting the logit gradients and running the GEMM in fp32/TF32.
"""

import torch
import torch.nn.functional as F

try:
    from fla.modules.fused_cross_entropy.ops import cross_entropy_bwd, cross_entropy_fwd
except ImportError:  # pragma: no cover
    cross_entropy_fwd = None


def fused_ce_available() -> bool:
    return cross_entropy_fwd is not None


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight, num_chunks: int, ignore_index: int):
        N, H = x.shape
        C = -(-N // num_chunks)
        total = target.ne(ignore_index).sum().clamp_min(1)
        dloss = total.float().reciprocal()
        loss = torch.zeros((), device=x.device, dtype=torch.float)
        dx = torch.empty_like(x)
        dw = torch.zeros(weight.shape, device=x.device, dtype=torch.float)
        for s in range(0, N, C):
            e = min(s + C, N)
            c_x, c_t = x[s:e], target[s:e]
            logits = F.linear(c_x, weight)
            c_loss, _, lse, total_classes, class_start = cross_entropy_fwd(
                logits=logits, target=c_t, label_smoothing=0.0, logit_scale=1.0, logit_softcapping=None,
                ignore_index=ignore_index, process_group=None,
            )
            loss += c_loss.sum()
            cross_entropy_bwd(
                logits=logits, target=c_t, lse=lse, dloss=dloss.expand(e - s), label_smoothing=0.0, logit_scale=1.0,
                logit_softcapping=None, ignore_index=ignore_index, total_classes=total_classes,
                class_start_idx=class_start, inplace_backward=True,
            )
            dx[s:e] = logits @ weight
            dw = torch.addmm(dw, logits.t(), c_x, out_dtype=torch.float32)
        ctx.save_for_backward(dx, dw.to(weight.dtype))
        return loss / total

    @staticmethod
    def backward(ctx, dout):
        dx, dw = ctx.saved_tensors
        return dx * dout, None, dw * dout, None, None


def fused_linear_cross_entropy(x, target, weight, num_chunks: int = 8, ignore_index: int = -100):
    """x (B, T, H), target (B, T), weight (V, H) -> mean token cross-entropy."""
    return _FusedLinearCE.apply(x.reshape(-1, x.shape[-1]), target.reshape(-1), weight, num_chunks, ignore_index)
