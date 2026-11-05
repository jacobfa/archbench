import inspect
import math

import torch.nn as nn

from . import register


class FLAMixer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


def build_fla(cls_name: str, d_model: int, layer_idx: int, n_layer: int, **kwargs) -> FLAMixer:
    try:
        import fla.layers as L
    except ImportError as e:
        raise ImportError("install flash-linear-attention: pip install 'archbench[fla]'") from e
    cls = getattr(L, cls_name)
    accepted = inspect.signature(cls).parameters
    auto = {"hidden_size": d_model, "layer_idx": layer_idx, "num_hidden_layers": n_layer}
    kwargs = {**{k: v for k, v in auto.items() if k in accepted}, **kwargs}
    return FLAMixer(cls(**kwargs))


@register("gdn")
def build_gdn(d_model, layer_idx, n_layer, **kwargs):
    return build_fla("GatedDeltaNet", d_model, layer_idx, n_layer, **kwargs)


@register("mesanet")
def build_mesanet(d_model, layer_idx, n_layer, **kwargs):
    return build_fla("MesaNet", d_model, layer_idx, n_layer, **kwargs)


@register("mamba")
def build_mamba(d_model, layer_idx, n_layer, **kwargs):
    expand = kwargs.pop("expand", 2)
    kwargs.setdefault("intermediate_size", int(expand * d_model))
    kwargs.setdefault("time_step_rank", math.ceil(d_model / 16))
    kwargs.setdefault("use_bias", False)
    return build_fla("Mamba", d_model, layer_idx, n_layer, **kwargs)


@register("mamba2")
def build_mamba2(d_model, layer_idx, n_layer, **kwargs):
    kwargs.setdefault("expand", 2)
    kwargs.setdefault("head_dim", 64)
    kwargs.setdefault("num_heads", kwargs["expand"] * d_model // kwargs["head_dim"])
    kwargs.setdefault("use_bias", False)
    return build_fla("Mamba2", d_model, layer_idx, n_layer, **kwargs)
