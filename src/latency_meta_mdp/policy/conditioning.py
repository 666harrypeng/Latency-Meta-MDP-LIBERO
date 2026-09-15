"""Conditioning modes bind inputs, supervision and runtime indexing together."""

CURRENT_AND_FORECAST = "current_and_forecast"
FORECAST_ONLY = "forecast_only"


def conditioning_contract(mode):
    if mode == CURRENT_AND_FORECAST:
        return {
            "conditioning": "native_rtc_forecast_rgb_v1",
            "target_alignment": "observation_time",
            "plan_construction": None,
        }
    if mode == FORECAST_ONLY:
        return {
            "conditioning": "native_rtc_forecast_only_rgb_v1",
            "target_alignment": "forecast_time",
            "plan_construction": "rtc_planned_handoff_h50_v1",
        }
    raise ValueError(f"Unsupported conditioning input mode: {mode}")


def conditioning_patches(root, mode):
    conditioning_contract(mode)
    patches = tuple(sorted((root / "patches/openpi").glob("000[1-9]-*.patch")))
    if mode == FORECAST_ONLY:
        patches += (root / "patches/openpi/0010-native-forecast-only-data.patch",)
    return patches
