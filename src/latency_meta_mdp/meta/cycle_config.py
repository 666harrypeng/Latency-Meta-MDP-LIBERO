"""Inputs for a local offline-to-online Meta cycle, separate from optimizer recipes."""

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

from latency_meta_mdp.meta.cost import load_cost_profile

EVALUATION_PATHS = (
    "checkpoint",
    "checkpoint_verification",
    "preparation_root",
    "forecast_assets",
    "bootstrap_checkpoint",
    "bootstrap_verification",
    "rtc_calibration",
)


def validate_cohorts(cohorts):
    if set(cohorts) != {"train", "validation", "feedback"}:
        raise ValueError("cohorts require train, validation and feedback")
    masters, levels = {}, set()
    regimes = set(cohorts["train"])
    if not regimes or not regimes <= {"nominal", "family"}:
        raise ValueError("cycle regimes must be nominal and/or family")
    for group, files in cohorts.items():
        if set(files) != regimes:
            raise ValueError("all cohort groups must cover the same regimes")
        masters[group] = set()
        for path in files.values():
            data = json.loads(Path(path).read_text())
            levels.add(data["level"])
            if data["partition"] != "train_pool_development" or not data["cases"]:
                raise ValueError("Meta cycle requires nonempty train-pool cohorts")
            keys = [(c["master_index"], c["policy_seed"]) for c in data["cases"]]
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate cohort cases")
            part = "validation" if group == "validation" else "train"
            if any(c["meta_partition"] != part for c in data["cases"]):
                raise ValueError("cohort partition differs from its role")
            masters[group].update(c["master_index"] for c in data["cases"])
    if len(levels) != 1 or masters["train"] & masters["validation"]:
        raise ValueError("cohorts need one level and disjoint train/validation masters")
    if not masters["feedback"] <= masters["train"]:
        raise ValueError("feedback must use declared training masters")
    return masters


def load_cycle_config(path):
    """Asset paths are repository-relative, like the other experiment job files."""
    root = Path.cwd()
    cfg = yaml.safe_load(Path(path).read_text())
    required = {
        "schema_version",
        "training_config",
        "cost_profile",
        "budget",
        "phases",
        "evaluation_python",
        "gpu_ids",
        "evaluation",
        "cohorts",
    }
    optional = {
        "training_python",
        "initial_replay",
        "initial_checkpoint",
        "initial_multiplier",
        "multiplier_step",
        "exploration_epsilon",
        "replica_start",
        "record_video",
        "use_future",
    }
    if not isinstance(cfg, dict) or required - cfg.keys() or cfg.keys() - required - optional:
        raise ValueError("missing or unknown Meta cycle configuration fields")
    if cfg["schema_version"] != 1:
        raise ValueError("unsupported Meta cycle schema")
    cfg = {
        "training_python": sys.executable,
        "initial_replay": None,
        "initial_checkpoint": None,
        "initial_multiplier": 0.1,
        "multiplier_step": 0.1,
        "exploration_epsilon": 0.2,
        "replica_start": 0,
        "record_video": True,
        "use_future": True,
        **cfg,
    }
    for key in ("budget", "initial_multiplier", "multiplier_step", "exploration_epsilon"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"invalid cycle value: {key}")
    if cfg["budget"] == 0 or cfg["exploration_epsilon"] > 1:
        raise ValueError("budget must be positive and epsilon at most one")
    if type(cfg["phases"]) is not int or cfg["phases"] < 1:
        raise ValueError("phases must be a positive integer")
    if type(cfg["replica_start"]) is not int or cfg["replica_start"] < 0:
        raise ValueError("replica_start must be nonnegative")
    for key in ("record_video", "use_future"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    gpu_ids = cfg["gpu_ids"]
    if (
        not gpu_ids
        or any(type(i) is not int or i < 0 for i in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("gpu_ids must contain distinct nonnegative device indices")

    def resolve(value, *, executable=False):
        p = root / value
        p = p.absolute() if executable else p.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)

    for key in (
        "training_config",
        "cost_profile",
        "evaluation_python",
        "training_python",
        "initial_replay",
        "initial_checkpoint",
    ):
        if cfg[key] is not None:
            cfg[key] = resolve(cfg[key], executable=key.endswith("_python"))
    evaluation = cfg["evaluation"]
    if set(EVALUATION_PATHS) - evaluation.keys() or evaluation.keys() - set(EVALUATION_PATHS) - {
        "decision_interval_ticks",
        "rtc_max_guidance_weight",
    }:
        raise ValueError("missing or unknown evaluation fields")
    cfg["evaluation"] = {
        "decision_interval_ticks": 4,
        "rtc_max_guidance_weight": 5.0,
        **evaluation,
        **{k: resolve(evaluation[k]) for k in EVALUATION_PATHS},
    }
    interval = cfg["evaluation"]["decision_interval_ticks"]
    guidance = cfg["evaluation"]["rtc_max_guidance_weight"]
    if type(interval) is not int or interval < 1 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("invalid decision interval or RTC guidance")
    cfg["cohorts"] = {
        group: {r: resolve(p) for r, p in files.items()} for group, files in cfg["cohorts"].items()
    }
    masters = validate_cohorts(cfg["cohorts"])
    if cfg["initial_replay"]:
        replay = json.loads(Path(cfg["initial_replay"]).read_text())
        if (
            replay["status"] != "completed"
            or replay["schema"] != 2
            or any(set(replay[k + "_masters"]) != masters[k] for k in ("train", "validation"))
        ):
            raise ValueError("initial replay and collection cohorts have different masters")
    recipe = yaml.safe_load(Path(cfg["training_config"]).read_text())
    if recipe.get("objective") != "rtc_budget_success_v1" or not recipe.get("recoverable_training"):
        raise ValueError("cycle requires recoverable budget training")
    # Bind small configuration/identity files, not multi-GB datasets or weight trees.
    inputs = [
        cfg["training_config"],
        cfg["cost_profile"],
        *[p for files in cfg["cohorts"].values() for p in files.values()],
    ]
    inputs += [
        cfg["evaluation"][k] for k in EVALUATION_PATHS if Path(cfg["evaluation"][k]).is_file()
    ]
    if cfg["initial_replay"]:
        inputs.append(cfg["initial_replay"])
    if cfg["initial_checkpoint"]:
        inputs.append(str(Path(cfg["initial_checkpoint"]) / "config.json"))
    identity = {
        "config": cfg,
        "files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
    }
    cfg["identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cfg["cost_profile_path"] = cfg["cost_profile"]
    cfg["cost_profile"] = load_cost_profile(cfg["cost_profile_path"])
    return cfg
