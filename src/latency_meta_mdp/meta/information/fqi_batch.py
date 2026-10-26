"""Resolve learner memory use without changing global batches or optimizer steps."""

import gc

import numpy as np
import torch

from latency_meta_mdp.meta.information.fqi_inputs import FQIInputs
from latency_meta_mdp.meta.information.memory_schema import validate_batch


def batch_candidates(requested, global_batch, world_size):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("positive world size required")
    if type(global_batch) is not int or global_batch < 1 or global_batch % world_size:
        raise ValueError("global batch must divide across ranks")
    local = global_batch // world_size
    if requested == "auto":
        return [n for n in range(local, 0, -1) if local % n == 0]
    if requested == "full":
        return [local]
    if isinstance(requested, str) and requested.isdigit():
        requested = int(requested)
    if type(requested) is not int or requested < 1:
        raise ValueError("microbatch must be full, auto or a positive integer")
    validate_batch(
        global_batch=global_batch,
        world_size=world_size,
        microbatch=requested,
        accumulation=local // requested,
    )
    return [requested]


def select_microbatch(requested, *, model, replay, global_batch, world_size, indices, bf16):
    candidates = batch_candidates(requested, global_batch, world_size)
    if requested != "auto":
        return candidates[0], []
    device = next(model.parameters()).device
    if device.type != "cuda":
        return candidates[0], [dict(microbatch=candidates[0], status="cpu_full")]
    free, _ = torch.cuda.mem_get_info(device)
    initial = torch.cuda.memory_allocated(device)
    reserve = max(512 * 2**20, int(0.05 * free))
    report = []

    def probe(size):
        # Two prefetched batches exercise accumulation and the real input path.
        order = np.resize(np.asarray(indices), size * 3)
        with FQIInputs(replay, model.config["memory"], microbatch=size, device=device) as inputs:
            for number, (_, _, batch) in enumerate(inputs.batches(order, following=False)):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    model(batch).float().square().mean().backward()
                if number == 1:
                    break
        torch.cuda.synchronize(device)

    # Probing does not call Adam, sample new decisions, or mutate model weights.
    # Fork the RNG as well, so future stochastic layers cannot consume training RNG.
    with torch.random.fork_rng(devices=[device.index or 0]):
        for size in candidates:
            torch.cuda.reset_peak_memory_stats(device)
            fits = False
            try:
                probe(size)
                peak = torch.cuda.max_memory_allocated(device)
                fits = peak - initial + reserve <= free
                report.append(
                    dict(
                        microbatch=size,
                        peak_bytes=peak,
                        reserve_bytes=reserve,
                        status="fits" if fits else "headroom",
                    )
                )
            except torch.cuda.OutOfMemoryError:
                report.append(dict(microbatch=size, status="out_of_memory"))
            finally:
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            if fits:
                return size, report
    raise RuntimeError("no FQI microbatch fits available CUDA memory with reserve")
