"""One shared disk cache for random minibatches of immutable visual replay shards."""

import hashlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch


def cache_visual_shard(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    # Hash a small identity descriptor, not the large dataset contents.
    identity = json.dumps([str(source.resolve()), stat.st_size, stat.st_mtime_ns])
    destination = directory / (hashlib.sha256(identity.encode()).hexdigest() + '.npy')
    if not destination.exists():
        with np.load(source, allow_pickle=False) as d:
            array = d["visual"]
        if array.ndim != 5 or array.shape[1:] != (2, 2, 196, 384):
            raise ValueError("invalid visual replay shape")
        if array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError("invalid visual replay values")
        temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as f:
            np.save(f, array, allow_pickle=False)
        os.replace(temporary, destination)
    return destination


class VisualReplay:
    """Advanced integer indexing returns only the requested Torch minibatch on CPU."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.open_shards = OrderedDict()
        lengths, self.nbytes = [], 0
        for path in paths:
            shard = np.load(path, mmap_mode='r', allow_pickle=False)
            lengths.append(len(shard))
            self.nbytes += shard.nbytes
            shard._mmap.close()
        self.offsets = np.concatenate(([0], np.cumsum(lengths)))

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, indices):
        ids = (indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor)
               else np.asarray(indices))
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError("visual replay indices outside committed states")
        result = np.empty((len(ids), 2, 2, 196, 384), np.float16)
        shard_ids = np.searchsorted(self.offsets[1:], ids, side='right')
        for shard in np.unique(shard_ids):
            if shard not in self.open_shards:
                if len(self.open_shards) >= 64:
                    _, old = self.open_shards.popitem(last=False)
                    old._mmap.close()
                self.open_shards[shard] = np.load(
                    self.paths[shard], mmap_mode='r', allow_pickle=False
                )
            self.open_shards.move_to_end(shard)
            mask = shard_ids == shard
            result[mask] = self.open_shards[shard][ids[mask] - self.offsets[shard]]
        return torch.from_numpy(result)
