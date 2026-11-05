import json
import os

import numpy as np
import torch


class TokenData:
    def __init__(self, path: str, seq_len: int):
        if not os.path.isfile(os.path.join(path, "meta.json")):
            raise FileNotFoundError(f"no dataset at {path!r}; run prepare_data.py first")
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        dtype = np.dtype(self.meta["dtype"])
        self.splits = {s: np.memmap(os.path.join(path, f"{s}.bin"), dtype=dtype, mode="r") for s in ("train", "val")}
        self.seq_len = seq_len

    @property
    def vocab_size(self) -> int:
        return self.meta["vocab_size"]

    def num_tokens(self, split: str) -> int:
        return len(self.splits[split])

    def batch(self, split: str, batch_size: int, rng: np.random.Generator, device: torch.device):
        data = self.splits[split]
        ix = rng.integers(0, len(data) - self.seq_len - 1, size=batch_size)
        buf = torch.from_numpy(np.stack([data[i : i + self.seq_len + 1] for i in ix]).astype(np.int64))
        if device.type == "cuda":
            buf = buf.pin_memory()
        buf = buf.to(device, non_blocking=True)
        return buf[:, :-1], buf[:, 1:]
