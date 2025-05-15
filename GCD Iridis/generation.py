import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Union, List
import model
from model import MathTokenizer


class GCDDataset(Dataset):
    def __init__(
        self,
        max_num: int = 100,
        num_samples: int = 100_000,
        seed: int = 42,
        base: int = 10,
        distribution: str = "uniform",
        distribution_params: dict = None
    ):
        self.rng = np.random.RandomState(seed)
        self.tokenizer = MathTokenizer(base)
        self.data: List[tuple[List[int], List[int]]] = []
        params = distribution_params or {}
        if distribution == "uniform":
            sampler = lambda size: self.rng.randint(1, max_num, size=size)
        elif distribution == "normal":
            loc = params.get("loc", (max_num - 1) / 2)
            scale = params.get("scale", (max_num - 1) / 4)
            def sampler(size):
                s = self.rng.normal(loc=loc, scale=scale, size=size)
                return np.rint(s).astype(int).clip(1, max_num - 1)
        elif distribution == "loguniform":
            low, high = np.log(1), np.log(max_num)
            def sampler(size):
                s = self.rng.uniform(low, high, size=size)
                return np.exp(s).astype(int).clip(1, max_num - 1)
        elif distribution == "poisson":
            lam = params.get("lam", max_num / 10)
            def sampler(size):
                return self.rng.poisson(lam, size=size).clip(1, max_num - 1)
        elif distribution == "geometric":
            p = params.get("p", 0.5)
            def sampler(size):
                return self.rng.geometric(p, size=size).clip(1, max_num - 1)
        elif distribution == "exponential":
            lam = params.get("lam", 1.0)
            def sampler(size):
                s = self.rng.exponential(scale=1/lam, size=size)
                return np.rint(s).astype(int).clip(1, max_num - 1)
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        for _ in range(num_samples):
            a, b = sampler(size=2)
            gcd_val = math.gcd(a, b)
            sign_a = '+' if a >= 0 else '-'
            sign_b = '+' if b >= 0 else '-'
            digits_a = self.tokenizer._int_to_base(abs(a))
            digits_b = self.tokenizer._int_to_base(abs(b))
            digits_gcd = self.tokenizer._int_to_base(gcd_val)
            src_tokens = [sign_a] + digits_a + [sign_b] + digits_b
            tgt_tokens = ['+'] + digits_gcd
            src_ids = self.tokenizer.encode(src_tokens)
            tgt_ids = self.tokenizer.encode(tgt_tokens)
            self.data.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)