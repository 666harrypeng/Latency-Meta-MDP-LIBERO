import hashlib
import io

import pytest

from latency_meta_mdp.artifacts import sha256_file


@pytest.mark.parametrize("size", [0, 13, 3 * 1024 * 1024 + 17])
def test_file_hash_matches_sha256_without_unbounded_reads(size):
    payload = (b"abcdefg" * (size // 7 + 1))[:size]
    reads = []

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 4 * 1024 * 1024, "file hashing must use bounded reads"
            reads.append(size)
            return super().read(size)

    class File:
        def read_bytes(self):
            raise MemoryError("large artifact cannot be loaded into RAM")

        def open(self, mode):
            assert mode == "rb"
            return BoundedReader(payload)

    assert sha256_file(File()) == hashlib.sha256(payload).hexdigest()
    assert reads
