"""Bounded CPU prefetch and lossless raw-frame transport for actor learners."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from latency_meta_mdp.meta.information.memory import collate_contexts
from latency_meta_mdp.meta.information.memory_schema import MemoryBatch


def transfer_batch(batch, device):
    tensors = {k: v.to(device, non_blocking=True) for k, v in batch.tensors.items()}
    if "visual_pool" in tensors:
        pool = tensors.pop("visual_pool")
        for name in ("visual", "forecast"):
            index = tensors.pop(name + "_index")
            value = pool[index.clamp_min(0)]
            value.masked_fill_((index < 0)[..., None, None, None], 0)
            tensors[name] = value
    return MemoryBatch(tensors, batch.refs)


def _slice(batch, start, end):
    return MemoryBatch(
        {k: v if k == "visual_pool" else v[start:end] for k, v in batch.tensors.items()},
        batch.refs[start:end],
    )


@dataclass
class InputBatch:
    indices: np.ndarray
    rows: list
    future: list
    combined: MemoryBatch

    @property
    def context(self):
        return _slice(self.combined, 0, len(self.rows))

    @property
    def following(self):
        return _slice(self.combined, len(self.rows), len(self.combined.refs))

    def to(self, device):
        return replace(self, combined=transfer_batch(self.combined, device))


class InputPipeline:
    """One producer per rank, at most one future microbatch; no policy state in workers."""

    def __init__(self, replay, config, *, microbatch, prefetch=True, pin_memory=False):
        self.replay, self.config, self.microbatch = replay, config, microbatch
        self.prefetch, self.pin_memory = prefetch, pin_memory
        self.pool = None

    def __enter__(self):
        if self.prefetch:
            self.pool = ThreadPoolExecutor(max_workers=1)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def prepare(self, indices, successors):
        rows = [self.replay[int(i)] for i in indices]
        if any(not r["legal"][r["action"]] for r in rows):
            raise ValueError("illegal training action")
        future = [i for i, r in enumerate(rows) if successors and not r["terminated"]]
        if any(rows[i]["next_context"] is None for i in future):
            raise ValueError("nonterminal transition missing successor")
        contexts = [r["context"] for r in rows] + [rows[i]["next_context"] for i in future]
        batch = collate_contexts(contexts, config=self.config, compact_visual=self.prefetch)
        if self.pin_memory:
            batch = MemoryBatch({k: v.pin_memory() for k, v in batch.tensors.items()}, batch.refs)
        return InputBatch(indices, rows, future, batch)

    def batches(self, order, *, successors):
        chunks = iter(order[i : i + self.microbatch] for i in range(0, len(order), self.microbatch))
        if self.pool is None:
            for indices in chunks:
                yield self.prepare(indices, successors)
            return
        first = next(chunks, None)
        if first is None:
            return
        pending = self.pool.submit(self.prepare, first, successors)
        while pending is not None:
            batch = pending.result()
            indices = next(chunks, None)
            pending = (
                None if indices is None else self.pool.submit(self.prepare, indices, successors)
            )
            yield batch
