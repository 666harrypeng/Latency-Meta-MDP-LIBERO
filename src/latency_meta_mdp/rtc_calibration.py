"""Read causal initial delay history from immutable completed-request evidence."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RtcDelayCalibration:
    delay_ticks: tuple[int, ...]
    source_sha256: str
    calibration_sha256: str


def load_rtc_calibration(path: Path, *, project_root: Path) -> RtcDelayCalibration:
    raw = path.read_bytes()
    config = json.loads(raw)
    if set(config) != {"format_id", "clock", "source_result", "source_sha256", "request_ids"}:
        raise ValueError("invalid calibration fields")
    if config["format_id"] != "rtc_delay_calibration_v1" or config["clock"] != "controlled_logical":
        raise ValueError("this evaluation requires controlled-logical calibration")
    relative = Path(config["source_result"])
    root = project_root.resolve()
    source = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(root):
        raise ValueError("calibration source must be a project-relative path")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != config["source_sha256"]:
        raise ValueError("calibration source hash mismatch")
    evidence = json.loads(data)
    if evidence.get("latency_mode") != "controlled_logical_policy_delay":
        raise ValueError("calibration source uses a different clock")
    ids = config["request_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or i < 0 for i in ids)
        or ids != sorted(set(ids))
    ):
        raise ValueError("calibration request selection must be nonempty, ordered and unique")
    requests = evidence["request_ledger"]
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("duplicate source request identity")
    requests = {r["request_id"]: r for r in requests}
    delays = []
    previous_arrival = -1
    for request_id in ids:
        row = requests.get(request_id)
        if row is None or row.get("arrival_formal_tick") is None:
            raise ValueError("calibration can include only completed requests")
        origin, launch, arrival, delay = (
            row.get("source_formal_tick"),
            row.get("launch_formal_tick"),
            row["arrival_formal_tick"],
            row.get("realized_delay_ticks"),
        )
        if (
            any(type(x) is not int or x < 0 for x in (origin, launch, arrival, delay))
            or not origin <= launch <= arrival
            or origin < previous_arrival
            or arrival - launch != delay
            or not 0 <= arrival - origin <= 20
        ):
            raise ValueError("calibration request duration/order is inconsistent")
        delays.append(arrival - origin)
        previous_arrival = arrival
    return RtcDelayCalibration(
        tuple(delays), config["source_sha256"], hashlib.sha256(raw).hexdigest()
    )
