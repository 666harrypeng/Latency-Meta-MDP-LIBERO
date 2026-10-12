"""Paired scene-level evaluation; balls and delay variants remain clustered."""

import numpy as np


def paired_summary(rows, *, reference, candidate="fqi", repeats=10000, seed=0):
    groups = {candidate: {}, reference: {}}
    for row in rows:
        ep, report = row["episode"], row["report"]
        name = ep["behavior"]
        if name not in groups:
            continue
        key = (ep["scene"], ep["regime"])
        if key in groups[name]:
            raise ValueError("duplicate paired evaluation episode")
        reward = float(report["task_reward"])
        if not np.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("invalid paired task return")
        groups[name][key] = reward
    a, b = groups[candidate], groups[reference]
    if not a or set(a) != set(b):
        raise ValueError("incomplete paired evaluation")
    scenes = sorted({s for s, r in a})
    if set(a) != {(s, r) for s in scenes for r in ("nominal", "family")}:
        raise ValueError("paired scenes require both delay regimes")
    differences = np.array([[a[s, r] - b[s, r] for r in ("nominal", "family")] for s in scenes])
    scene_delta = differences.mean(axis=1)
    draws = (
        np.random.default_rng(seed)
        .choice(scene_delta, size=(repeats, len(scenes)), replace=True)
        .mean(axis=1)
    )
    return dict(
        reference=reference,
        candidate=candidate,
        paired_scenes=len(scenes),
        mean_difference=float(scene_delta.mean()),
        interval_95=np.quantile(draws, [0.025, 0.975]).tolist(),
        wins=int((differences > 0).sum()),
        ties=int((differences == 0).sum()),
        losses=int((differences < 0).sum()),
        regime_differences=dict(
            zip(("nominal", "family"), differences.mean(axis=0).tolist(), strict=True)
        ),
        scope="paired development; bootstrap unit is scene, not ball",
    )
