import math

import torch
from torch.optim import Optimizer

from .config import Config
from .model import LM


def lr_mult(step: int, cfg: Config) -> float:
    o, total = cfg.optim, cfg.train.steps
    if step < o.warmup_steps:
        return (step + 1) / o.warmup_steps
    if o.schedule == "wsd":
        progress = min(1.0, (total - step) / max(1, round(o.warmdown_ratio * total)))
        return o.min_lr_ratio + (1 - o.min_lr_ratio) * progress
    if o.schedule != "cosine":
        raise ValueError(f"unknown schedule {o.schedule!r}; choose cosine or wsd")
    progress = min(1.0, (step - o.warmup_steps) / max(1, total - o.warmup_steps))
    return o.min_lr_ratio + 0.5 * (1 - o.min_lr_ratio) * (1 + math.cos(math.pi * progress))


def muon_momentum(step: int, cfg: Config) -> float:
    o = cfg.optim
    frac = min(1.0, step / o.muon_momentum_warmup) if o.muon_momentum_warmup > 0 else 1.0
    return 0.85 + frac * (o.muon_momentum - 0.85)


# Polar Express coefficients (https://arxiv.org/abs/2505.16932)
POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def muon_update(g, p, buf, v, momentum, lr, wd, beta2, ns_steps: int):
    """Nesterov momentum -> Polar Express orthogonalization -> NorMuon variance reduction -> cautious weight decay, on (K, m, n) stacks."""
    buf.lerp_(g, 1 - momentum)
    g = g.lerp_(buf, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_EXPRESS[:ns_steps]:
        if X.size(-2) > X.size(-1):
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
        else:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X.to(p.dtype)
    red = -1 if g.size(-2) >= g.size(-1) else -2
    n = g.size(red)
    v_mean = g.float().square().mean(dim=red, keepdim=True)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * n).sqrt()
    v.lerp_(v_mean.to(v.dtype), 1 - beta2)
    scale = v.clamp_min(1e-10).rsqrt()
    v_norm_new = (v_mean * n * scale.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (scale * v_norm / v_norm_new.clamp_min(1e-10)).to(g.dtype)
    p.sub_(lr * g + lr * wd * p * ((g * p) >= 0))


class MuonAdamW(Optimizer):
    def __init__(self, param_groups, lr=0.02, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8, compile=False):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay, betas=betas, eps=eps, use_muon=True)
        super().__init__(param_groups, defaults)
        for g in self.param_groups:
            if g["use_muon"] and len({p.shape for p in g["params"]}) != 1:
                raise ValueError("every muon param group must hold params of a single 2D shape")
        self._update = torch.compile(muon_update, fullgraph=True) if compile else muon_update

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon(group)
            else:
                self._adamw(group)
        return loss

    def _muon(self, group):
        params = group["params"]
        (rows, cols), p0 = params[0].shape, params[0]
        st = self.state[p0]
        if not st:
            st["momentum_buffer"] = torch.zeros(len(params), rows, cols, device=p0.device, dtype=p0.dtype)
            st["second_momentum_buffer"] = torch.zeros(len(params), *((rows, 1) if rows >= cols else (1, cols)), device=p0.device, dtype=p0.dtype)
        g = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        w = torch.stack(params)
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=p0.device)  # noqa: E731
        lr = group["lr"] * max(1.0, rows / cols) ** 0.5
        self._update(g, w, st["momentum_buffer"], st["second_momentum_buffer"], t(group["momentum"]), t(lr), t(group["weight_decay"]), t(group["beta2"]), group["ns_steps"])
        torch._foreach_copy_(params, list(w.unbind(0)))

    def _adamw(self, group):
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if not st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.lerp_(p.grad, 1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
            p.mul_(1 - lr * wd)
            p.addcdiv_(m, (v / bc2).sqrt().add_(eps), value=-lr / bc1)


class SOAP(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=None, eps=1e-8, weight_decay=0.01, precondition_frequency=10, max_precond_dim=8192):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps, weight_decay=weight_decay, precondition_frequency=precondition_frequency, max_precond_dim=max_precond_dim)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            bs = group["shampoo_beta"] if group["shampoo_beta"] is not None else b2
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                precond = p.ndim >= 2
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(g)
                    st["exp_avg_sq"] = torch.zeros_like(g)
                    if precond:
                        st["GG"] = [torch.zeros(d, d, device=g.device, dtype=g.dtype) if d <= group["max_precond_dim"] else None for d in g.shape]
                        st["Q"] = None
                st["step"] += 1
                if precond:
                    self._update_gg(st["GG"], g, bs)
                    if st["Q"] is None:
                        st["Q"] = [self._eigenbasis(gg) if gg is not None else None for gg in st["GG"]]
                        continue
                    gp = self._project(g, st["Q"])
                else:
                    gp = g

                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(gp, gp, value=1 - b2)
                bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
                num = self._project(m, st["Q"]) if precond else m
                upd = num / (v / bc2).sqrt().add_(eps)
                if precond:
                    upd = self._project_back(upd, st["Q"])
                p.mul_(1 - lr * wd)
                p.add_(upd, alpha=-lr / bc1)

                if precond and st["step"] % group["precondition_frequency"] == 0:
                    self._update_q(st)
        return loss

    @staticmethod
    def _update_gg(GG, g, beta):
        dims = list(range(g.ndim))
        for i, gg in enumerate(GG):
            if gg is None:
                continue
            others = [d for d in dims if d != i]
            gg.lerp_(torch.tensordot(g, g, dims=[others, others]), 1 - beta)

    @staticmethod
    def _eigenbasis(gg):
        eye = torch.eye(gg.shape[0], device=gg.device, dtype=gg.dtype)
        _, Q = torch.linalg.eigh(gg + 1e-30 * eye)
        return Q.flip(-1)

    @staticmethod
    def _project(x, Q):
        for q in Q:
            x = torch.tensordot(x, q, dims=[[0], [0]]) if q is not None else x.movedim(0, -1)
        return x

    @staticmethod
    def _project_back(x, Q):
        for q in Q:
            x = torch.tensordot(x, q, dims=[[0], [1]]) if q is not None else x.movedim(0, -1)
        return x

    @staticmethod
    def _update_q(st):
        for i, (gg, q) in enumerate(zip(st["GG"], st["Q"])):
            if q is None:
                continue
            power = gg @ q
            idx = torch.argsort((q * power).sum(0), descending=True)
            st["exp_avg_sq"] = st["exp_avg_sq"].index_select(i, idx)
            st["Q"][i], _ = torch.linalg.qr(power[:, idx])


def build_optimizer(model: LM, cfg: Config) -> Optimizer:
    o = cfg.optim
    scalars = [{"params": ps, "use_muon": False, "lr": o.scalar_lr * mult, "betas": (0.8, 0.95), "weight_decay": 0.0} for ps, mult in model.scalar_groups()]
    scalar_ids = {id(p) for g in scalars for p in g["params"]}
    params = [p for p in model.parameters() if p.requires_grad and id(p) not in scalar_ids]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    betas = (o.beta1, o.beta2)

    if o.name == "adamw":
        groups = [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}, *scalars]
        return torch.optim.AdamW(groups, lr=o.lr, betas=betas, eps=o.eps, fused=torch.cuda.is_available())
    if o.name == "soap":
        groups = [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}, *scalars]
        return SOAP(groups, lr=o.lr, betas=betas, eps=o.eps, weight_decay=o.weight_decay, precondition_frequency=o.soap_precondition_frequency, max_precond_dim=o.soap_max_precond_dim)
    if o.name == "muon_adamw":
        embed = {id(model.embed.weight), id(model.lm_head.weight)}
        matrices = [p for p in decay if p.ndim == 2 and id(p) not in embed]
        matrix_ids = {id(m) for m in matrices}
        groups = [{"params": [p for p in matrices if p.shape == s], "use_muon": True, "lr": o.muon_lr, "weight_decay": o.muon_weight_decay} for s in sorted({p.shape for p in matrices})]
        groups += [
            {"params": [p for p in decay if id(p) not in matrix_ids], "use_muon": False, "lr": o.lr, "weight_decay": o.weight_decay},
            {"params": no_decay, "use_muon": False, "lr": o.lr, "weight_decay": 0.0},
            *scalars,
        ]
        return MuonAdamW(
            [g for g in groups if g["params"]],
            momentum=o.muon_momentum,
            ns_steps=o.muon_ns_steps,
            beta2=o.muon_beta2,
            betas=betas,
            eps=o.eps,
            compile=cfg.train.compile and torch.cuda.is_available(),
        )
    raise ValueError(f"unknown optimizer {o.name!r}; choose muon_adamw, adamw or soap")
