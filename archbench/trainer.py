import gc
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim.lr_scheduler import LambdaLR

from .config import Config, resolve_steps
from .core_eval import evaluate_core, sample
from .data import TokenData, token_bytes
from .logger import Logger, Meter, PhaseTimer, fmt_count, fmt_duration, peak_flops
from .model import LM
from .optim import build_optimizer, lr_mult, muon_momentum

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def build_ds_config(cfg: Config, micro_bs: int) -> dict:
    d = cfg.deepspeed
    conf = {
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": cfg.train.grad_accum,
        "gradient_clipping": cfg.optim.grad_clip,
        "zero_optimization": {
            "stage": d.zero_stage,
            "overlap_comm": d.overlap_comm,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
    }
    if cfg.train.dtype != "float32":
        # DeepSpeed disables an outer torch.autocast inside engine.forward unless this section is present.
        conf["torch_autocast"] = {"enabled": True, "dtype": cfg.train.dtype, "lower_precision_safe_modules": []}
    conf.update(d.extra)
    return conf


class Trainer:
    def __init__(self, cfg: Config):
        import deepspeed

        self.cfg = cfg
        t_start = time.perf_counter()
        for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"}.items():
            os.environ.setdefault(k, v)
        self.cuda = torch.cuda.is_available()
        deepspeed.init_distributed(dist_backend="nccl" if self.cuda else "gloo")
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.main = self.rank == 0
        self.device = torch.device(f"cuda:{self.local_rank}" if self.cuda else "cpu")
        if self.cuda:
            torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.train.seed + self.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        dtype = DTYPES[cfg.train.dtype]
        self.autocast = torch.autocast("cuda", dtype=dtype) if self.cuda and dtype != torch.float32 else nullcontext()

        self.data = TokenData(cfg.data.path, cfg.data.seq_len)
        self.tok = self.data.tokenizer()
        self.token_bytes = token_bytes(self.tok).to(self.device)
        if cfg.model.vocab_size is None:
            cfg.model.vocab_size = (self.data.vocab_size + 63) // 64 * 64

        t = cfg.train
        if t.batch_size % (self.world * t.grad_accum):
            raise ValueError("batch_size must be divisible by world_size * grad_accum")
        if cfg.optim.name != "adamw" and cfg.deepspeed.zero_stage > 0:
            raise ValueError(f"{cfg.optim.name} needs full parameter shapes; set deepspeed.zero_stage=0")
        self.micro_bs = t.batch_size // (self.world * t.grad_accum)
        self.tokens_per_step = t.batch_size * cfg.data.seq_len

        model = LM(cfg.model).to(self.device)
        self.n_params = model.num_params()
        self.param_groups = model.param_breakdown()
        self.layer_params = model.layer_params()
        self.flops_per_token = model.flops_per_token(cfg.data.seq_len)
        self.peak_flops = t.peak_tflops * 1e12 if t.peak_tflops else peak_flops(torch.cuda.get_device_name(self.device)) if self.cuda else 0.0
        t.steps = resolve_steps(t, self.tokens_per_step, self.n_params)
        opt = build_optimizer(model, cfg)
        sched = LambdaLR(opt, lambda s: lr_mult(s, cfg))
        self.engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, lr_scheduler=sched, config=build_ds_config(cfg, self.micro_bs))
        raw_forward = model.forward
        self.logits = lambda x: raw_forward(x)[0]
        if t.compile:
            self.engine.module.forward = torch.compile(self.engine.module.forward, dynamic=False)

        self.run_name = t.resume or cfg.wandb.name or f"{cfg.model.arch}-{cfg.optim.name}-{time.strftime('%m%d-%H%M%S')}"
        self.log_dir = os.path.join(os.path.expanduser(t.log_dir), self.run_name)
        self.ckpt_dir = os.path.join(os.path.expanduser(t.ckpt_dir), self.run_name)

        self.step, self.best_val, self.best_core, wandb_id = 0, float("inf"), float("-inf"), None
        if t.resume:
            _, client = self.engine.load_checkpoint(self.ckpt_dir, tag="last")
            self.step, self.best_val, wandb_id = client["step"], client["best_val"], client.get("wandb_id")
            self.best_core = client.get("best_core", self.best_core)

        self.log = Logger(cfg, self.log_dir, self.main, wandb_id)
        if self.main:
            with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        self.timer = PhaseTimer(self.cuda)
        self.meter = Meter(t.steps, {"eval": t.eval_every, "ckpt": t.ckpt_every, "core": t.core_every})
        self.log.summary(
            params=self.n_params,
            params_non_embedding=self.n_params - self.param_groups["embed"],
            **{f"params_{k}": v for k, v in self.param_groups.items()},
            flops_per_token=self.flops_per_token,
            total_flops=self.flops_per_token * self.tokens_per_step * t.steps,
            tokens_per_step=self.tokens_per_step,
            total_tokens=self.tokens_per_step * t.steps,
            tokens_per_param=self.tokens_per_step * t.steps / self.n_params,
            total_steps=t.steps,
            world_size=self.world,
            device=torch.cuda.get_device_name(self.device) if self.cuda else "cpu",
            startup_s=time.perf_counter() - t_start,
        )

    def _param_lines(self) -> str:
        c, g, n = self.cfg.model, self.param_groups, self.n_params
        lines = [f"params   {n:,} total ({fmt_count(n)}), {n - g['embed']:,} non-embedding ({fmt_count(n - g['embed'])})"]
        for name, count in g.items():
            if not count:
                continue
            note = "tied with lm_head" if name == "embed" and c.tie_embeddings else f"{c.n_layer} x {fmt_count(count / c.n_layer)}" if name in ("mixer", "mlp") else ""
            lines.append(f"         {name:8s} {fmt_count(count):>8s} {100 * count / n:5.1f}%  {note}")
        for part, params in self.layer_params.items():
            lines.append(f"         block.{part}: " + ", ".join(f"{k} {list(s)} {fmt_count(v)}" for k, (s, v) in params.items()))
        return "\n".join(lines)

    def banner(self):
        c, m = self.cfg, self.cfg.model
        tweaks = [k for k in ("embed_norm", "resid_scalars", "smear", "backout", "zero_init") if getattr(m, k)]
        self.log.print(
            f"run      {self.run_name}  logs={self.log_dir} ckpts={self.ckpt_dir}\n"
            f"arch     {m.arch}  d_model={m.d_model} n_layer={m.n_layer} mlp={m.mlp_act if m.mlp else False} layer={m.layer}\n"
            f"tweaks   {' '.join(tweaks) or '-'}" + (f" softcap={m.logit_softcap:g}" if m.logit_softcap else "") + "\n"
            f"{self._param_lines()}\n"
            f"flops    {fmt_count(self.flops_per_token)}/tok, {self.flops_per_token * self.tokens_per_step * c.train.steps:.2e} total, "
            f"peak {self.peak_flops / 1e12:g} TFLOP/s per device\n"
            f"optim    {c.optim.name}  lr={c.optim.lr:g} {c.optim.schedule} wd={c.optim.weight_decay:g} warmup={c.optim.warmup_steps} clip={c.optim.grad_clip:g}"
            + (f" muon_lr={c.optim.muon_lr:g}" if c.optim.name == "muon_adamw" else "")
            + (f" scalar_lr={c.optim.scalar_lr:g}" if tweaks else "") + "\n"
            f"eval     val every {c.train.eval_every} ({c.train.eval_steps} batches), "
            + (f"core every {c.train.core_every} ({c.train.core_max_per_task}/task)" if c.train.core_every > 0 else "core off") + "\n"
            f"data     {c.data.path}  train={fmt_count(self.data.num_tokens('train'))} val={fmt_count(self.data.num_tokens('val'))} seq_len={c.data.seq_len}\n"
            f"batch    {c.train.batch_size} x {c.data.seq_len} = {fmt_count(self.tokens_per_step)} tok/step, "
            f"{fmt_count(self.tokens_per_step * c.train.steps)} total over {c.train.steps} steps "
            f"({self.tokens_per_step * c.train.steps / self.n_params:.1f} tok/param, "
            f"budget={'fixed train.tokens' if c.train.tokens is not None else f'{c.train.tokens_per_param:g} tok/param'}, accum={c.train.grad_accum})\n"
            f"system   {self.device} x{self.world}  dtype={c.train.dtype} zero={c.deepspeed.zero_stage} compile={c.train.compile}"
        )

    def _rng(self, *keys: int) -> np.random.Generator:
        return np.random.default_rng([self.cfg.train.seed, *keys])

    def _reduce_mean(self, x: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(x, op=dist.ReduceOp.AVG)
        return x

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def _lrs(self) -> dict:
        lrs = {}
        for g in self.engine.optimizer.param_groups:
            lrs.setdefault("train/lr_muon" if g.get("use_muon") else "train/lr", g["lr"])
        return lrs

    @torch.no_grad()
    def _param_norm(self) -> float:
        sq = torch.zeros((), device=self.device)
        for p in self.engine.module.parameters():
            sq += p.detach().float().square().sum()
        return sq.sqrt().item()

    def _memory(self) -> dict:
        if not self.cuda:
            return {}
        return {
            "perf/mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "perf/mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "perf/mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.engine.eval()
        self._sync()
        t0 = time.perf_counter()
        total = torch.zeros((), device=self.device)
        n_bytes = torch.zeros((), device=self.device, dtype=torch.int64)
        for i in range(self.cfg.train.eval_steps):
            x, y = self.data.batch("val", self.micro_bs, self._rng(10**6, i, self.rank), self.device)
            with self.autocast:
                _, loss = self.engine(x, y)
            total += loss
            n_bytes += self.token_bytes[y].sum()
        self.engine.train()
        loss = self._reduce_mean(total / self.cfg.train.eval_steps).item()
        if self.world > 1:
            dist.all_reduce(n_bytes)
        self._sync()
        took = time.perf_counter() - t0
        tokens = self.cfg.train.eval_steps * self.micro_bs * self.world * self.cfg.data.seq_len
        return {
            "val/loss": loss,
            "val/ppl": math.exp(min(loss, 20)),
            "val/bpb": loss * tokens / (math.log(2) * max(1, n_bytes.item())),
            "perf/eval_s": took,
            "perf/eval_tokens_per_sec": tokens / took,
        }

    @torch.no_grad()
    def core(self):
        t = self.cfg.train
        self.engine.eval()
        t0 = time.perf_counter()
        with self.autocast:
            metric, _, centered = evaluate_core(self.logits, self.tok, self.device, t.core_dir, t.core_max_per_task, self.cfg.data.seq_len)
            samples = sample(self.logits, self.tok, self.device) if self.main else []
        self.engine.train()
        took = time.perf_counter() - t0
        self.meter.update_event("core", took)
        self.best_core = max(self.best_core, metric)
        self.log.log({"core/metric": metric, **{f"core/{k}": v for k, v in centered.items()}, "perf/core_s": took}, self.step)
        self.log.summary(core_metric=metric, best_core_metric=self.best_core)
        self.log.print(f"core {self.step:>6}/{t.steps} | metric {metric:.4f} | best {self.best_core:.4f} | took {took:.1f}s")
        for s in samples:
            self.log.print(f"         {s!r}")

    def save(self, tag: str) -> float:
        t0 = time.perf_counter()
        client = {
            "step": self.step,
            "best_val": self.best_val,
            "best_core": self.best_core,
            "wandb_id": self.log.wandb_id,
            "config": self.cfg.to_dict(),
            "tokenizer": self.data.meta["tokenizer"],
        }
        self.engine.save_checkpoint(self.ckpt_dir, tag=tag, client_state=client)
        self._sync()
        took = time.perf_counter() - t0
        self.meter.update_event("ckpt", took)
        return took

    def train_step(self):
        t = self.cfg.train
        if self.cfg.optim.name == "muon_adamw":
            mom = muon_momentum(self.step, self.cfg)
            for g in self.engine.optimizer.param_groups:
                if g["use_muon"]:
                    g["momentum"] = mom
        loss_sum = torch.zeros((), device=self.device)
        gnorm = None
        for micro in range(t.grad_accum):
            with self.timer("data", gpu=False):
                x, y = self.data.batch("train", self.micro_bs, self._rng(self.step, micro, self.rank), self.device)
            with self.timer("forward"), self.autocast:
                _, loss = self.engine(x, y)
            with self.timer("backward"):
                self.engine.backward(loss)
            loss_sum += loss.detach()
            with self.timer("optimizer"):
                if self.cfg.deepspeed.zero_stage == 0 and self.engine.is_gradient_accumulation_boundary():
                    gnorm = torch.nn.utils.clip_grad_norm_(self.engine.module.parameters(), float("inf"))
                self.engine.step()
        ds_norm = self.engine.get_global_grad_norm()
        return loss_sum / t.grad_accum, ds_norm if ds_norm is not None else gnorm

    def _log_train(self, loss_acc: torch.Tensor, gnorm, n_loss: int, n_time: int, wall: float):
        t = self.cfg.train
        dt = wall / max(1, n_time)
        if n_time:
            self.meter.update(wall, n_time)
        loss_avg = self._reduce_mean(loss_acc / n_loss).item()
        gn = float(gnorm) if gnorm is not None else float("nan")
        phases = {k: v / max(1, n_time) for k, v in self.timer.collect().items()}
        tok_s = self.tokens_per_step / dt
        eta = self.meter.eta(self.step)
        metrics = {
            "train/loss": loss_avg,
            "train/ppl": math.exp(min(loss_avg, 20)),
            "train/grad_norm": gn,
            "train/param_norm": self._param_norm(),
            **self._lrs(),
            "perf/tokens_per_sec": tok_s,
            "perf/tokens_per_sec_per_gpu": tok_s / self.world,
            "perf/samples_per_sec": t.batch_size / dt,
            "perf/step_ms": dt * 1e3,
            **{f"perf/{k}_ms": v for k, v in phases.items()},
            "perf/other_ms": max(0.0, dt * 1e3 - sum(phases.values())),
            **self._memory(),
            "perf/elapsed_min": self.meter.elapsed() / 60,
            "perf/eta_min": eta / 60,
            "perf/epoch": self.step * self.tokens_per_step / self.data.num_tokens("train"),
            "tokens": self.step * self.tokens_per_step,
            "flops": self.step * self.tokens_per_step * self.flops_per_token,
        }
        if self.peak_flops:
            metrics["perf/mfu"] = self.flops_per_token * tok_s / (self.peak_flops * self.world)
        self.log.log(metrics, self.step)
        mem = metrics.get("perf/mem_peak_gb", 0.0)
        mfu = f" | mfu {100 * metrics['perf/mfu']:.1f}%" if self.peak_flops else ""
        self.log.print(
            f"step {self.step:>6}/{t.steps} | loss {loss_avg:.4f} | lr {metrics['train/lr']:.2e} | gnorm {gn:.2f} | "
            f"{fmt_count(tok_s)} tok/s{mfu} | {dt * 1e3:.0f}ms/step "
            f"(data {phases.get('data', 0):.0f} fwd {phases.get('forward', 0):.0f} bwd {phases.get('backward', 0):.0f} opt {phases.get('optimizer', 0):.0f}) | "
            f"mem {mem:.1f}G | elapsed {fmt_duration(self.meter.elapsed())} | eta {fmt_duration(eta)} (~{self.meter.finish_at(self.step)})"
        )

    def run(self):
        t = self.cfg.train
        self.banner()
        self.engine.train()
        loss_acc = torch.zeros((), device=self.device)
        n_loss = n_time = 0
        first = True
        t0 = time.perf_counter()

        while self.step < t.steps:
            loss, gnorm = self.train_step()
            loss_acc += loss
            n_loss += 1
            n_time += 1
            self.step += 1
            last = self.step == t.steps

            if first:
                self._sync()
                first_s = time.perf_counter() - t0
                self.log.summary(first_step_s=first_s)
                self.log.print(f"first step took {first_s:.1f}s (excluded from throughput and eta)")
                self.timer.collect()
                # python's cyclic gc otherwise stalls steps for ~0.5s scanning long-lived setup objects
                gc.collect()
                gc.freeze()
                gc.disable()
                first, n_time, t0 = False, 0, time.perf_counter()
            elif self.step % 5000 == 0:
                gc.collect()

            if self.step % t.log_every == 0 or last:
                self._sync()
                self._log_train(loss_acc, gnorm, n_loss, n_time, time.perf_counter() - t0)
                loss_acc.zero_()
                n_loss = n_time = 0
                t0 = time.perf_counter()

            if self.step % t.eval_every == 0 or last:
                res = self.evaluate()
                self.meter.update_event("eval", res["perf/eval_s"])
                improved = res["val/loss"] < self.best_val
                self.best_val = min(res["val/loss"], self.best_val)
                res["val/best_loss"] = self.best_val
                if improved:
                    res["perf/ckpt_s"] = self.save("best")
                self.log.log(res, self.step)
                if improved:
                    self.log.summary(best_val_loss=self.best_val, best_val_ppl=res["val/ppl"], best_step=self.step)
                self.log.print(
                    f"eval {self.step:>6}/{t.steps} | val_loss {res['val/loss']:.4f} | val_ppl {res['val/ppl']:.2f} | val_bpb {res['val/bpb']:.4f} | "
                    f"best {self.best_val:.4f}{' *' if improved else ''} | {fmt_count(res['perf/eval_tokens_per_sec'])} tok/s | took {res['perf/eval_s']:.1f}s"
                )
                t0 = time.perf_counter()

            if t.core_every > 0 and (self.step % t.core_every == 0 or last):
                self.core()
                t0 = time.perf_counter()

            if self.step % t.ckpt_every == 0 or last:
                took = self.save("last")
                self.log.log({"perf/ckpt_s": took}, self.step)
                self.log.print(f"ckpt {self.step:>6}/{t.steps} | saved {self.ckpt_dir}/last in {took:.1f}s")
                t0 = time.perf_counter()

        total = self.meter.elapsed()
        self.log.summary(train_time_min=total / 60, avg_tokens_per_sec=self.tokens_per_step * t.steps / total)
        self.log.print(f"done in {fmt_duration(total)} | best val_loss {self.best_val:.4f}" + (f" | best core {self.best_core:.4f}" if t.core_every > 0 else ""))
        self.log.finish()
        dist.barrier()
        dist.destroy_process_group()


def train(cfg: Config):
    Trainer(cfg).run()
