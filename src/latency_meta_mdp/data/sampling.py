"""Query-balanced sampling shared by current and historical policy datasets."""

from math import gcd

import numpy as np


class UniformDelaySampler:
    """Balance valid D20 labels in shuffled blocks, covering each delay's index pool.

    Each epoch visits every valid pair at least once. Shorter delay pools are
    repeated only as needed to equalize counts. No episode-PMF weighting is used.
    """

    def __init__(self, indices_by_delay, *, seed: int, batch_size: int = 20, start_batch: int = 0):
        if type(seed) is not int or seed < 0:
            raise ValueError("sampler seed must be a nonnegative integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("sampler batch_size must be a positive integer")
        if type(start_batch) is not int or start_batch < 0:
            raise ValueError("sampler start_batch must be a nonnegative integer")
        self.indices = tuple(np.asarray(x, dtype=np.int64) for x in indices_by_delay)
        if len(self.indices) != 20 or any(x.ndim != 1 or not len(x) for x in self.indices):
            raise ValueError("balanced sampling requires real action labels at every D20 delay")
        multiple = batch_size // gcd(batch_size, 20)
        self.per_delay_count = ((max(map(len, self.indices)) + multiple - 1) // multiple) * multiple
        self.seed = seed
        self.epoch, self.offset = divmod(start_batch * batch_size, 20 * self.per_delay_count)

    def __len__(self):
        return 20 * self.per_delay_count - self.offset

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        self.epoch += 1
        offset, self.offset = self.offset, 0
        columns = []
        for indices in self.indices:
            parts = []
            remaining = self.per_delay_count
            while remaining:
                part = rng.permutation(indices)[:remaining]
                parts.append(part)
                remaining -= len(part)
            columns.append(np.concatenate(parts))
        # Every consecutive block has one sample of each delay, in random order.
        ordered = rng.permuted(np.column_stack(columns), axis=1)
        yield from map(int, ordered.ravel()[offset:])
