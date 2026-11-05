import torch


def _shift(x: torch.Tensor, k: int, fill: float) -> torch.Tensor:
    pad = x.new_full((x.shape[0], k, *x.shape[2:]), fill)
    return torch.cat([pad, x[:, :-k]], dim=1)


def linear_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t along dim 1, h_0 = 0. `a` broadcasts against `b`."""
    T = b.shape[1]
    h, g, k = b, a, 1
    while k < T:
        h = h + g * _shift(h, k, 0.0)
        g = g * _shift(g, k, 1.0)
        k *= 2
    return h


def linear_scan_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)
