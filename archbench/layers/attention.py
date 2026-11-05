import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, :, None], sin[None, :, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, rope_base: float = 10000.0, layer_idx: int | None = None):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.H, self.hd, self.base = num_heads, d_model // num_heads, rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _freqs(self, T: int, device: torch.device):
        if self._cache is None or self._cache[0].shape[0] < T or self._cache[0].device != device:
            inv = 1.0 / (self.base ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
            f = torch.outer(torch.arange(T, device=device).float(), inv)
            self._cache = (f.cos(), f.sin())
        cos, sin = self._cache
        return cos[:T], sin[:T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.hd).unbind(2)
        cos, sin = self._freqs(T, x.device)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))


@register("attention")
def build_attention(d_model: int, layer_idx: int, n_layer: int, **kwargs):
    return Attention(d_model, layer_idx=layer_idx, **kwargs)
