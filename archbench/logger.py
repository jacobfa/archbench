import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from .config import Config


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# dense bf16 peak FLOP/s, most specific pattern first
PEAK_FLOPS = (
    (("gb200",), 2.5e15),
    (("b200",), 2.25e15),
    (("h200",), 989e12),
    (("h100", "nvl"), 835e12),
    (("h100", "pcie"), 756e12),
    (("h100",), 989e12),
    (("a100",), 312e12),
    (("l40s",), 362e12),
    (("a40",), 149.7e12),
    (("l4",), 121e12),
    (("mi300x",), 1.3074e15),
    (("5090",), 209.5e12),
    (("4090",), 165.2e12),
    (("3090",), 71e12),
)


def peak_flops(device_name: str) -> float:
    name = device_name.lower()
    return next((f for pats, f in PEAK_FLOPS if all(p in name for p in pats)), 0.0)


def fmt_count(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


class PhaseTimer:
    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.events = defaultdict(list)
        self.cpu = defaultdict(float)

    @contextmanager
    def __call__(self, name: str, gpu: bool = True):
        if self.cuda and gpu:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events[name].append((start, end))
        else:
            t = time.perf_counter()
            yield
            self.cpu[name] += time.perf_counter() - t

    def collect(self) -> dict[str, float]:
        out = {name: sum(s.elapsed_time(e) for s, e in evs) for name, evs in self.events.items()}
        out.update({name: t * 1e3 for name, t in self.cpu.items()})
        self.events.clear()
        self.cpu.clear()
        return out


class Meter:
    def __init__(self, total_steps: int, every: dict[str, int], momentum: float = 0.9):
        self.total, self.every, self.m = total_steps, every, momentum
        self.step_time, self.event_time = None, {}
        self.steady_time, self.steady_steps = 0.0, 0
        self.start = time.perf_counter()

    def _ema(self, old, new):
        return new if old is None else self.m * old + (1 - self.m) * new

    def update(self, wall: float, n_steps: int):
        self.steady_time += wall
        self.steady_steps += n_steps
        self.step_time = self._ema(self.step_time, wall / n_steps)

    def update_event(self, name: str, seconds: float):
        self.event_time[name] = self._ema(self.event_time.get(name), seconds)

    def _remaining_events(self, step: int, every: int) -> int:
        if every <= 0:
            return 0
        n = self.total // every - step // every
        return n + (1 if self.total % every else 0)

    def eta(self, step: int) -> float:
        if self.step_time is None:
            return float("nan")
        avg = self.steady_time / max(1, self.steady_steps)
        per_step = 0.5 * (self.step_time + avg)
        remaining = self.total - step
        return remaining * per_step + sum(self._remaining_events(step, every) * self.event_time.get(name, 0.0) for name, every in self.every.items())

    def finish_at(self, step: int) -> str:
        eta = self.eta(step)
        if eta != eta:
            return "?"
        t = time.localtime(time.time() + eta)
        return time.strftime("%H:%M" if eta < 86400 else "%a %H:%M", t)

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


class Logger:
    def __init__(self, cfg: Config, run_dir: str, enabled: bool, wandb_id: str | None = None):
        self.enabled = enabled
        self.run = None
        self.wandb_id = wandb_id
        self.file = None
        if not enabled:
            return
        os.makedirs(run_dir, exist_ok=True)
        self.file = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        if cfg.wandb.mode != "disabled":
            import wandb

            self.run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=os.path.basename(run_dir),
                mode=cfg.wandb.mode,
                tags=cfg.wandb.tags,
                config=cfg.to_dict(),
                dir=run_dir,
                id=wandb_id,
                resume="allow",
            )
            self.wandb_id = self.run.id
            self.run.define_metric("tokens")
            for prefix in ("train/*", "val/*", "core/*", "perf/*"):
                self.run.define_metric(prefix, step_metric="step")

    def log(self, metrics: dict, step: int):
        if not self.enabled:
            return
        self.file.write(json.dumps({"step": step, **metrics}) + "\n")
        self.file.flush()
        if self.run is not None:
            self.run.log({"step": step, **metrics}, step=step)

    def summary(self, **kwargs):
        if self.run is not None:
            self.run.summary.update(kwargs)

    def print(self, *args):
        if self.enabled:
            print(*args, flush=True)

    def finish(self):
        if self.file is not None:
            self.file.close()
        if self.run is not None:
            self.run.finish()
