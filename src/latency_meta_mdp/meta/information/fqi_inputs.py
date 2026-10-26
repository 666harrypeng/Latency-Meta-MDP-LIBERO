"""Bounded, lossless CPU and CUDA prefetch for fixed-target attention FQI."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import torch

from latency_meta_mdp.meta.information.actor_inputs import transfer_batch
from latency_meta_mdp.meta.information.attention_replay import AttentionEpisodes
from latency_meta_mdp.meta.information.memory import collate_contexts


class FQIInputs:
    """Two bounded producers; raw transfers overlap compute without caching features."""

    def __init__(self, replay, config, *, microbatch, device):
        self.replay, self.config = replay, config
        self.microbatch, self.device = microbatch, device
        self.pool = None
        self.read_lock = Lock()
        self.stream = None

    def __enter__(self):
        if isinstance(self.replay.episodes, AttentionEpisodes):
            # The campaign owns immutable replay for this entire target/regression
            # pass. Admission already checks content; recheck metadata once here.
            for entry in self.replay.episodes.entries:
                entry.verify_unchanged()
        if str(self.device).startswith("cuda"):
            self.stream = torch.cuda.Stream(device=self.device)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fqi-input")
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.stream is not None:
            self.stream.synchronize()

    def prepare(self, indices, following):
        # Serialize only the mutable episode LRU, not the large NumPy collation.
        with self.read_lock, self.replay.batch(indices, verify=False) as rows:
            contexts = [
                r["next_context"] if following and r["next_context"] is not None else r["context"]
                for r in rows
            ]
        batch = collate_contexts(
            contexts,
            config=self.config,
            compact_visual=True,
            pin_memory=str(self.device).startswith("cuda"),
        )
        return indices, rows, batch

    def _prepare_transfer(self, indices, following):
        indices, rows, batch = self.prepare(indices, following)
        if self.stream is None:
            return indices, rows, transfer_batch(batch, self.device), None
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            batch = transfer_batch(batch, self.device)
            ready = self.stream.record_event()
        return indices, rows, batch, ready

    def batches(self, order, *, following):
        chunks = iter(order[i : i + self.microbatch] for i in range(0, len(order), self.microbatch))
        pending = deque()

        def enqueue():
            ids = next(chunks, None)
            if ids is not None:
                pending.append(self.pool.submit(self._prepare_transfer, ids, following))

        for _ in range(2):
            enqueue()
        while pending:
            indices, rows, batch, ready = pending.popleft().result()
            enqueue()
            if ready is not None:
                consumer = torch.cuda.current_stream(self.device)
                consumer.wait_event(ready)
                for tensor in batch.tensors.values():
                    tensor.record_stream(consumer)
            yield indices, rows, batch
