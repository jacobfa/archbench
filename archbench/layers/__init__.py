_REGISTRY: dict[str, callable] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_mixer(name: str, d_model: int, layer_idx: int, n_layer: int, **kwargs):
    if name.startswith("fla:"):
        from .fla import build_fla

        return build_fla(name[4:], d_model, layer_idx, n_layer, **kwargs)
    if name not in _REGISTRY:
        raise KeyError(f"unknown arch {name!r}; available: {available()} or 'fla:<ClassName>'")
    return _REGISTRY[name](d_model=d_model, layer_idx=layer_idx, n_layer=n_layer, **kwargs)


from . import attention, fla, pgdn  # noqa: E402,F401
