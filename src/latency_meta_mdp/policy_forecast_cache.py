"""One immutable, lossless RGB/proprio cache; CPU workers never run the world model."""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from latency_meta_mdp.artifacts import sha256_file

_FORMAT = "rtc_forecast_rgb_cache_v2"
_TAIL = "missing_forecast_preserve_action_supervision_v1"
_HASH_KEYS = {
    "predictor_sha256",
    "decoder_sha256",
    "jepa_normalization_sha256",
    "source_manifest_sha256",
    "split_manifest_sha256",
    "vision_cache_manifest_sha256",
}


def _check_bindings(bindings):
    if not isinstance(bindings, dict) or set(bindings) != _HASH_KEYS | {"predictor_architecture"}:
        raise ValueError("forecast cache bindings are incomplete")
    if bindings["predictor_architecture"] != "jepa_direct_q20_history_stride4_w3_v1" or any(
        not isinstance(bindings[k], str) or re.fullmatch(r"[a-f0-9]{64}", bindings[k]) is None
        for k in _HASH_KEYS
    ):
        raise ValueError(
            "forecast cache bindings require the Direct architecture and SHA256 values"
        )


def write_forecast_cache(
    *,
    records,
    normalization,
    engine,
    output_dir: Path,
    bindings: dict,
    batch_size: int = 32,
    resume: bool = False,
):
    """Batch only real endpoints. Missing history/tails have no rows or invented labels."""
    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import ForecastQuery
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
        materialize_direct_query,
    )

    _check_bindings(bindings)
    if (
        not records
        or len({r.episode_id for r in records}) != len(records)
        or len({r.level for r in records}) != 1
    ):
        raise ValueError("cache records require unique train episodes of one level")
    if any(r.split != "train" or r.level != normalization.level for r in records):
        raise ValueError("forecast SFT cache requires matching train records and normalization")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("cache batch size must be positive")
    root = Path(output_dir).resolve()
    contract = {
        "bindings": bindings,
        "episodes": [(r.episode_id, r.terminal_tick) for r in records],
    }
    contract = json.loads(json.dumps(contract))
    if resume:
        if json.loads((root / "generation.json").read_text()) != contract:
            raise ValueError("Forecast resume identity mismatch")
        if (root / "manifest.json").exists():
            ForecastCache(root, expected_bindings=bindings)
            return root / "manifest.json"
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "generation.json").write_text(json.dumps(contract, indent=2))
    database = root / "forecasts.sqlite"
    conn = sqlite3.connect(database)
    png_pool = ThreadPoolExecutor(max_workers=4)
    episodes, count, png_bytes = [], 0, 0
    phase_coverage = {}
    started = time.monotonic()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions (episode TEXT, source INTEGER, q INTEGER, "
            "rgb BLOB, proprio BLOB, PRIMARY KEY(episode,source,q)) WITHOUT ROWID"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes "
            "(episode TEXT PRIMARY KEY, proprio BLOB, controls BLOB)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS completed "
            "(episode TEXT PRIMARY KEY, rows INTEGER, png_bytes INTEGER)"
        )
        done = {
            r[0]: (r[1], r[2]) for r in conn.execute("SELECT episode,rows,png_bytes FROM completed")
        }
        count = sum(r[0] for r in done.values())
        png_bytes = sum(r[1] for r in done.values())
        pending = []

        def flush():
            nonlocal count, png_bytes
            if not pending:
                return
            query = ForecastQuery(
                **{
                    name: torch.cat([getattr(item[3], name) for item in pending])
                    for name in vars(pending[0][3])
                }
            )
            rgb, proprio = engine.predict(query)
            if rgb.shape != (len(pending), 2, 224, 224, 3) or rgb.dtype != np.uint8:
                raise ValueError("cache engine returned invalid RGB")
            if proprio.shape != (len(pending), 16) or not np.isfinite(proprio).all():
                raise ValueError("cache engine returned invalid physical proprio")

            def encode(item):
                i, (episode, h, q, _) = item
                stream = io.BytesIO()
                Image.fromarray(np.concatenate(rgb[i], axis=1)).save(
                    stream, format="PNG", compress_level=1
                )
                image = stream.getvalue()
                return episode, h, q, image, np.asarray(proprio[i], dtype="<f4").tobytes()

            encoded = list(png_pool.map(encode, enumerate(pending)))
            png_bytes += sum(len(row[3]) for row in encoded)
            with conn:
                conn.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)", encoded)
            count += len(pending)
            pending.clear()
            if count % (50 * batch_size) == 0:
                print(
                    json.dumps(
                        {
                            "prediction_rows": count,
                            "png_bytes": png_bytes,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )

        for record in records:
            n = record.terminal_tick
            for h in range(n):
                phase = record.phases[h] or "unlabeled"
                coverage = phase_coverage.setdefault(
                    phase, {"real_action_sources": 0, "forecast_available_by_query": [0] * 20}
                )
                coverage["real_action_sources"] += 1
                if h >= 10:
                    for q in range(1, min(20, n - h) + 1):
                        coverage["forecast_available_by_query"][q - 1] += 1
            episodes.append(
                {
                    "episode_id": record.episode_id,
                    "frame_count": n,
                    "level": record.level,
                    "logical_master_task_index": record.logical_master_task_index,
                }
            )
            if record.episode_id in done:
                continue
            # Incomplete episode writes are rolled back on resume; complete episodes remain.
            with conn:
                conn.execute("DELETE FROM predictions WHERE episode=?", (record.episode_id,))
                conn.execute("DELETE FROM episodes WHERE episode=?", (record.episode_id,))
            before_count, before_bytes = count, png_bytes
            with conn:
                conn.execute(
                    "INSERT INTO episodes VALUES (?,?,?)",
                    (
                        record.episode_id,
                        np.asarray(record.proprio_physical, dtype="<f4").tobytes(),
                        np.asarray(record.controls[:n], dtype="<f4").tobytes(),
                    ),
                )
            for h in range(10, n):
                for q in range(1, min(20, n - h) + 1):
                    query = materialize_direct_query(
                        record, source_tick=h, query_ticks=q, normalization=normalization
                    )
                    pending.append((record.episode_id, h, q, query))
                    if len(pending) == batch_size:
                        flush()
            flush()
            with conn:
                conn.execute(
                    "INSERT INTO completed VALUES (?,?,?)",
                    (record.episode_id, count - before_count, png_bytes - before_bytes),
                )
            print(
                json.dumps(
                    {
                        "cache_episode": record.episode_id,
                        "episodes_done": len(episodes),
                        "prediction_rows": count,
                        "png_bytes": png_bytes,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        expected = sum(max(0, r.terminal_tick - q - 9) for r in records for q in range(1, 21))
        if (
            count != expected
            or conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] != expected
        ):
            raise ValueError("forecast cache has incomplete endpoint coverage")
        conn.close()
        manifest = {
            "format_id": _FORMAT,
            "complete": True,
            "source_tail_policy": _TAIL,
            "bindings": dict(bindings),
            "episodes": episodes,
            "prediction_count": count,
            "png_bytes": png_bytes,
            "database_bytes": database.stat().st_size,
            "verification_policy": "metadata_and_read_validation",
            "phase_coverage": phase_coverage,
            "implementation_sha256": {
                Path(__file__).name: sha256_file(Path(__file__)),
                "forecast_provider.py": sha256_file(
                    Path(__file__).parent / "belief/action_conditioned_jepa/forecast_provider.py"
                ),
            },
        }
        with (root / "manifest.json").open("x") as f:
            json.dump(manifest, f, indent=2)
    finally:
        png_pool.shutdown(wait=True)
        conn.close()
    return root / "manifest.json"


class ForecastCache:
    def __init__(self, root: Path, *, expected_bindings: dict):
        _check_bindings(expected_bindings)
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (
            m.get("format_id") != _FORMAT
            or m.get("complete") is not True
            or m.get("source_tail_policy") != _TAIL
        ):
            raise ValueError("forecast cache is incomplete or incompatible")
        if m.get("bindings") != expected_bindings:
            raise ValueError("forecast cache binding mismatch")
        if m.get("verification_policy") != "metadata_and_read_validation":
            raise ValueError("forecast cache verification policy is incompatible")
        if (self.root / "forecasts.sqlite").stat().st_size != m["database_bytes"]:
            raise ValueError("forecast cache database size mismatch")
        self.episodes = {e["episode_id"]: e for e in m["episodes"]}
        if len(self.episodes) != len(m["episodes"]):
            raise ValueError("forecast cache episode identities are not unique")
        self._db = None
        self._pid = None

    def __getstate__(self):
        return {**self.__dict__, "_db": None, "_pid": None}

    def _connect(self):
        if self._db is None or self._pid != os.getpid():
            if self._db is not None:
                self._db.close()
            self._db = sqlite3.connect(
                (self.root / "forecasts.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
            )
            self._pid = os.getpid()
        return self._db

    def episode_arrays(self, episode_id):
        n = self.episodes[episode_id]["frame_count"]
        row = (
            self._connect()
            .execute("SELECT proprio,controls FROM episodes WHERE episode=?", (episode_id,))
            .fetchone()
        )
        if row is None:
            raise ValueError("forecast cache is missing episode arrays")
        return (
            np.frombuffer(row[0], dtype="<f4").reshape(n + 1, 16),
            np.frombuffer(row[1], dtype="<f4").reshape(n, 7),
        )

    def read(self, episode_id, source_tick, query_ticks):
        n = self.episodes[episode_id]["frame_count"]
        if (
            type(source_tick) is not int
            or not 0 <= source_tick < n
            or type(query_ticks) is not int
            or not 1 <= query_ticks <= 20
        ):
            raise ValueError("forecast cache query identity is invalid")
        if source_tick < 10 or source_tick + query_ticks > n:
            return None
        row = (
            self._connect()
            .execute(
                "SELECT rgb,proprio FROM predictions WHERE episode=? AND source=? AND q=?",
                (episode_id, source_tick, query_ticks),
            )
            .fetchone()
        )
        if row is None:
            raise ValueError("forecast cache is missing a required prediction")
        with Image.open(io.BytesIO(row[0])) as im:
            if im.mode != "RGB" or im.size != (448, 224):
                raise ValueError("forecast PNG shape mismatch")
            rgb = np.asarray(im, dtype=np.uint8)
        proprio = np.frombuffer(row[1], dtype="<f4").copy()
        if proprio.shape != (16,) or not np.isfinite(proprio).all():
            raise ValueError("invalid cached proprio")
        return np.stack([rgb[:, :224], rgb[:, 224:]]), proprio
