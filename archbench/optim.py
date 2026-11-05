import math

import torch
from torch.optim import Optimizer

from .config import Config
from .model import LM


def lr_mult(step: int, cfg: Config) -> float:
    o, total = cfg.optim, cfg.train.steps
    if step < o.warmup_steps:
        return (step + 1) / o.warmup_steps
    progress = min(1.0, (step - o.warmup_steps) / max(1, total - o.warmup_steps))
    return o.min_lr_ratio + 0.5 * (1 - o.min_lr_ratio) * (1 + math.cos(math.pi * progress))


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.mT if transposed else X).to(G.dtype)


class MuonAdamW(Optimizer):
    def __init__(self, param_groups, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, betas=betas, eps=eps, use_muon=True)
        super().__init__(param_groups, defaults)

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
        lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p)
            buf = st["momentum_buffer"]
            buf.lerp_(p.grad, 1 - mom)
            g = p.grad.lerp(buf, mom) if group["nesterov"] else buf
            u = newton_schulz(g.reshape(g.shape[0], -1), group["ns_steps"]).view_as(p)
            p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr * max(1.0, p.shape[0] / p[0].numel()) ** 0.5)

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
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    betas = (o.beta1, o.beta2)

    if o.name == "adamw":
        groups = [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=o.lr, betas=betas, eps=o.eps, fused=torch.cuda.is_available())
    if o.name == "soap":
        groups = [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return SOAP(groups, lr=o.lr, betas=betas, eps=o.eps, weight_decay=o.weight_decay, precondition_frequency=o.soap_precondition_frequency, max_precond_dim=o.soap_max_precond_dim)
    if o.name == "muon_adamw":
        embed = {id(model.embed.weight), id(model.lm_head.weight)}
        matrices = [p for p in decay if p.ndim == 2 and id(p) not in embed]
        rest = [p for p in decay if id(p) not in {id(m) for m in matrices}]
        groups = [
            {"params": matrices, "use_muon": True, "lr": o.muon_lr, "weight_decay": o.muon_weight_decay},
            {"params": rest, "use_muon": False, "lr": o.lr, "weight_decay": o.weight_decay},
            {"params": no_decay, "use_muon": False, "lr": o.lr, "weight_decay": 0.0},
        ]
        return MuonAdamW([g for g in groups if g["params"]], momentum=o.muon_momentum, ns_steps=o.muon_ns_steps, betas=betas, eps=o.eps)
    raise ValueError(f"unknown optimizer {o.name!r}; choose muon_adamw, adamw or soap")
