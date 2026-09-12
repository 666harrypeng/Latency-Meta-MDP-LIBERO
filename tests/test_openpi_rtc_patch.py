import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture(scope="module")
def patched_rtc():
    from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy

    patches = tuple(sorted(Path("patches/openpi").glob("000[1-7]-*.patch")))
    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=patches,
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        try:
            yield
        finally:
            sys.path.remove(str(root / "src"))


def test_real_pi0_sampler_zero_guidance_matches_native_and_preserves_padding(
    monkeypatch, patched_rtc
):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    from test_openpi_belief_prefix import _tiny_pi05

    from latency_meta_mdp.openpi_rtc import build_rtc_weights

    _, model, obs = _tiny_pi05(monkeypatch, prefix=False)
    sample = module_jit(model.sample_actions)
    noise = jax.random.normal(jax.random.key(3), (1, 50, 32))
    target = jnp.zeros_like(noise)
    weights = jnp.asarray(
        build_rtc_weights(
            valid_mask=np.arange(50) < 25,
            estimated_delay_ticks=4,
            active_action_dim=7,
            model_action_dim=32,
        )
    )[None]
    common = dict(num_steps=2, noise=noise)
    native = sample(jax.random.key(5), obs, **common)
    zero = sample(
        jax.random.key(5),
        obs,
        **common,
        rtc_previous_actions=target,
        rtc_weights=weights,
        rtc_max_guidance_weight=0.0,
    )
    np.testing.assert_allclose(zero, native, atol=1e-6, rtol=1e-6)
    guided = sample(
        jax.random.key(5),
        obs,
        **common,
        rtc_previous_actions=target,
        rtc_weights=weights,
        rtc_max_guidance_weight=2.0,
    )
    assert np.isfinite(guided).all()
    assert np.max(np.abs(guided[..., :7] - native[..., :7])) > 1e-5
    np.testing.assert_array_equal(guided[..., 7:], noise[..., 7:])


def test_policy_bridge_normalizes_previous_actions_exactly_once(patched_rtc):
    import flax.nnx as nnx
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.libero_policy import LiberoOutputs
    from openpi.policies.policy import Policy
    from openpi.shared.normalize import NormStats

    from latency_meta_mdp.openpi_policy_data import StructuredPolicyInputs
    from latency_meta_mdp.policy_execution import InProcessRtcOpenpiPolicy, PolicyObservation
    from latency_meta_mdp.rtc_protocol import RtcInferenceContext

    class NormalizedActionFixture(nnx.Module):
        # The model boundary adds one normalized unit. Actual normalization and
        # output transforms, including padding, belong to the production Policy.
        active_action_dim = 7
        action_horizon = 50
        pi05 = True

        def sample_actions(self, rng, observation, *, rtc_previous_actions, rtc_weights, noise):
            assert rtc_previous_actions.shape == (1, 50, 32)
            assert rtc_weights.shape == (1, 50, 32)
            return rtc_previous_actions + jnp.ones_like(rtc_previous_actions)

    tokenized_states = []

    class TokenizerFixture:
        def tokenize(self, prompt, state):
            tokenized_states.append(np.array(state))
            return np.ones(4, dtype=np.int32), np.ones(4, dtype=bool)

    stats = {
        "state": NormStats(mean=np.zeros(16), std=np.ones(16)),
        "actions": NormStats(mean=np.full(7, 0.25), std=np.full(7, 0.5)),
    }
    policy = Policy(
        NormalizedActionFixture(),
        transforms=[
            StructuredPolicyInputs(model_type=ModelType.PI05),
            transforms.Normalize(stats),
            transforms.PadStatesAndActions(32),
            transforms.TokenizePrompt(TokenizerFixture(), discrete_state_input=True),
        ],
        output_transforms=[transforms.Unnormalize(stats), LiberoOutputs()],
    )
    previous = np.full((50, 7), -0.2, dtype=np.float32)
    inputs = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(16, np.float32),
        "actions": previous,
        "actions_is_pad": np.arange(50) >= 25,
    }
    obs = PolicyObservation(
        formal_tick=0,
        image=inputs["observation/image"],
        wrist_image=inputs["observation/wrist_image"],
        state=inputs["observation/state"],
    )
    context = RtcInferenceContext(
        request_id=0,
        origin_tick=0,
        observation=obs,
        buffer_version=0,
        previous_actions=previous,
        previous_action_mask=np.arange(50) < 25,
        estimated_delay_ticks=4,
    )
    bridge = InProcessRtcOpenpiPolicy(policy, noise_rng=np.random.default_rng(1))
    result = bridge(obs, context)
    np.testing.assert_allclose(result["actions"][:25, :7], 0.3, atol=2e-6)
    np.testing.assert_array_equal(inputs["actions"], previous)
    from dataclasses import replace

    from latency_meta_mdp.planned_handoff_policy import PlannedHandoffPolicy
    from latency_meta_mdp.policy_forecast import DecodedForecast

    future_state = np.linspace(-.8, .8, 16, dtype=np.float32)
    context = replace(context, forecast=DecodedForecast(
        0, 4, 4, 0, np.full((2, 224, 224, 3), 128, np.uint8), future_state,
    ))
    result = PlannedHandoffPolicy(bridge, mode="forecast")(obs, context)
    np.testing.assert_array_equal(result["actions"][:4], previous[:4])
    np.testing.assert_allclose(result["actions"][4:25, :7], .3, atol=2e-6)
    np.testing.assert_allclose(tokenized_states[-1][:16], future_state, atol=2e-6)
    assert result["handoff"]["policy_input_tick"] == 4


def test_rtc_cli_preflight_preserves_training_preparation_and_records_runtime_patch(
    tmp_path, capsys, monkeypatch, patched_rtc
):
    import json

    from latency_meta_mdp.artifacts import sha256_file
    from latency_meta_mdp.cli.evaluate_policy import main

    checkpoint = tmp_path / "6000"
    checkpoint.mkdir()
    prep = tmp_path / "preparation"
    prep.mkdir()
    cohort = tmp_path / "cohort.json"
    case = {
        "master_index": 62,
        "policy_seed": 0,
        "task_instance_id": {
            "level": 3,
            "task_instance_seed": 62,
            "motion_profile_sha256": "a" * 64,
            "initial_state_sha256": "b" * 64,
        },
    }
    cohort.write_text(
        json.dumps({"partition": "train_pool_development", "level": 3, "cases": [case]})
    )
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "all_downloaded_hashes_match": True,
                "checkpoint_root": str(checkpoint),
                "repo_id": "yypeng666/metamdp-pi05-l3-clean-state16-h50-full-sft-v1",
                "repo_sha": "test-preflight-no-weights-loaded",
                "checkpoint_step": 6000,
            }
        )
    )
    preparation = {
        "profile_sha256": sha256_file(Path("configs/policy/pi05_structured_state16_h50_v1.yaml")),
        "level": 3,
        "patches": {
            p.name: sha256_file(p) for p in Path("patches/openpi").glob("000[1-3]-*.patch")
        },
    }
    prep_file = prep / "preparation.json"
    prep_file.write_text(json.dumps(preparation))
    before = prep_file.read_bytes()
    args = [
        "--cohort",
        str(cohort),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-verification",
        str(verification),
        "--preparation-root",
        str(prep),
        "--output-root",
        str(tmp_path / "eval"),
        "--protocol",
        "rtc",
        "--preflight-only",
    ]
    main(args)
    result = json.loads(capsys.readouterr().out)
    assert result["identity"]["protocol_id"] == "rtc_observation_time_h50_v1"
    assert "0007-inference-time-rtc.patch" in result["identity"]["runtime_patch_sha256"]
    assert prep_file.read_bytes() == before
    from latency_meta_mdp.rtc_calibration import RtcDelayCalibration

    monkeypatch.setattr(
        "latency_meta_mdp.cli.evaluate_policy.load_rtc_calibration",
        lambda path, project_root: RtcDelayCalibration((4, 6), "source-sha", "calibration-sha"),
    )
    main(args + ["--rtc-calibration", str(tmp_path / "calibration.json")])
    calibrated = json.loads(capsys.readouterr().out)
    assert calibrated["initial_delay_ticks"] == [4, 6]
    assert calibrated["identity"]["rtc_calibration"]["calibration_sha256"] == "calibration-sha"
    main(args + ["--fixed-delay-ticks", "0", "1", "20"])
    grid = json.loads(capsys.readouterr().out)
    assert grid["case_count"] == 3
    assert grid["fixed_delay_ticks"] == [0, 1, 20]
    from openpi.policies import policy_config

    # No model or simulator is needed when the identical case result already exists.
    monkeypatch.setattr(policy_config, "create_trained_policy", lambda *a, **kw: None)
    cached = tmp_path / "eval/master-062-seed-0-zero.json"
    cached.write_text(json.dumps({"identity": calibrated["identity"], "case": case}))
    cached_before = cached.read_bytes()
    resume_args = [a for a in args if a != "--preflight-only"]
    main(resume_args + ["--rtc-calibration", str(tmp_path / "calibration.json")])
    assert cached.read_bytes() == cached_before
    main(args + ["--planned-handoff", "current"])
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["identity"]["plan_construction"] == "rtc_planned_handoff_h50_v1"
    assert handoff["identity"]["policy_observation_mode"] == "current"
    assets = tmp_path / "forecast-assets.json"
    assets.write_text(json.dumps({"level": 3, "forecast_identity": {"predictor_sha256": "a"*64}}))
    main(args + ["--planned-handoff", "forecast", "--forecast-assets", str(assets),
                 "--bootstrap-checkpoint", str(checkpoint),
                 "--bootstrap-verification", str(verification)])
    replacement = json.loads(capsys.readouterr().out)
    assert replacement["identity"]["policy_observation_mode"] == "forecast"
    assert replacement["identity"]["conditioning"] == "native_forecast_replacement_v1"
    assert "0008-native-rtc-forecast-inputs.patch" not in (
        replacement["identity"]["runtime_patch_sha256"]
    )
    assert replacement["state_tokens"] and replacement["action_horizon"] == 50
    data = json.loads(verification.read_text())
    forecast_identity = {
        "predictor_architecture": "jepa_direct_q20_history_stride4_w3_v1",
        "predictor_sha256": "a" * 64,
        "decoder_sha256": "b" * 64,
        "jepa_normalization_sha256": "c" * 64,
    }
    assets.write_text(json.dumps({"level": 3, "forecast_identity": forecast_identity}))
    data.update(repo_id="local/p0-l3-rtc-forecast-directq20-state16-h50-v1",
                conditioning="native_rtc_forecast_rgb_v1",
                forecast_identity=forecast_identity)
    bootstrap_verification = tmp_path / "bootstrap.json"
    bootstrap_verification.write_bytes(verification.read_bytes())
    verification.write_text(json.dumps(data))
    import subprocess

    # A separate interpreter must import the four-image patch set, not this
    # fixture's already imported native seven-patch OpenPI modules.
    shared = json.loads(subprocess.check_output([
        sys.executable, "-m", "latency_meta_mdp.cli.evaluate_policy", *args,
        "--forecast-assets", str(assets), "--bootstrap-checkpoint", str(checkpoint),
        "--bootstrap-verification", str(bootstrap_verification),
        "--prepare-forecast-before-decision", "--decision-interval-ticks", "4",
    ], text=True))
    assert shared["identity"]["prepare_forecast_before_decision"] is True
    assert shared["identity"]["decision_interval_ticks"] == 4
    data = json.loads(verification.read_text())
    data["repo_id"] = "yypeng666/metamdp-pi05-l3-predicted-mixture-state16-h50-prefix-q4-2epochs-v1"
    verification.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="clean checkpoint"):
        main(args)
    with pytest.raises(ValueError, match="clean checkpoint"):
        main(args + ["--planned-handoff", "current"])
