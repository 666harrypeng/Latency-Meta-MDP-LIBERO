import numpy as np
import pytest


def test_soft_mask_matches_rtc_overlap_rule_and_ignores_model_padding():
    from latency_meta_mdp.policy.openpi.rtc import build_rtc_weights

    weights = build_rtc_weights(
        valid_mask=np.arange(50) < 2,
        estimated_delay_ticks=1,
        active_action_dim=7,
        model_action_dim=32,
    )
    np.testing.assert_allclose(weights[0, :7], 1)
    np.testing.assert_allclose(weights[1, :7], 0.1887703344, rtol=1e-6)
    np.testing.assert_array_equal(weights[2:], 0)
    np.testing.assert_array_equal(weights[:, 7:], 0)


def test_soft_mask_handles_full_prefix_and_rejects_unavailable_controls():
    from latency_meta_mdp.policy.openpi.rtc import build_rtc_weights

    weights = build_rtc_weights(
        valid_mask=np.arange(50) < 20,
        estimated_delay_ticks=20,
        active_action_dim=7,
        model_action_dim=32,
    )
    np.testing.assert_array_equal(weights[:20, :7], 1)
    np.testing.assert_array_equal(weights[20:], 0)
    with pytest.raises(ValueError):
        build_rtc_weights(
            valid_mask=np.arange(50) < 2,
            estimated_delay_ticks=3,
            active_action_dim=7,
            model_action_dim=32,
        )
    with pytest.raises(ValueError):
        build_rtc_weights(
            valid_mask=np.arange(50) != 2,
            estimated_delay_ticks=1,
            active_action_dim=7,
            model_action_dim=32,
        )


def test_clean_estimate_uses_openpi_reverse_time_convention():
    from latency_meta_mdp.policy.openpi.rtc import estimate_clean_actions

    action = np.array([0.3, -0.5])
    noise = np.array([-0.2, 0.7])
    for t in [0, 0.2, 0.8, 1]:
        mixed = t * noise + (1 - t) * action
        np.testing.assert_allclose(estimate_clean_actions(mixed, t, noise - action), action)


def test_guidance_vjp_sign_corrects_toward_known_actions_with_negative_dt():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from latency_meta_mdp.policy.openpi.rtc import rtc_guided_velocity

    x = jnp.zeros((1, 3, 2))
    target = jnp.ones_like(x)
    weights = jnp.ones_like(x)
    velocity = jax.jit(
        lambda x: rtc_guided_velocity(
            lambda value: value * 0,
            x,
            0.5,
            target,
            weights,
            max_guidance_weight=2,
        )
    )(x)
    np.testing.assert_allclose(velocity, -2)
    # Native OpenPI integrates from t=1 down to0.
    updated = x - 0.1 * velocity
    assert np.linalg.norm(updated - target) < np.linalg.norm(x - target)


def test_guidance_is_finite_at_endpoints_and_zero_weight_is_exact_native():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from latency_meta_mdp.policy.openpi.rtc import rtc_guided_velocity

    x = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2) / 10

    def field(value):
        return 0.2 * value + 0.1

    for t in [0, 1e-8, 0.5, 1 - 1e-8, 1]:
        v = rtc_guided_velocity(
            field, x, t, jnp.ones_like(x), jnp.ones_like(x), max_guidance_weight=5
        )
        assert np.isfinite(v).all()
        native = rtc_guided_velocity(
            field, x, t, jnp.ones_like(x), jnp.ones_like(x), max_guidance_weight=0
        )
        # Compare equally compiled operations; eager float32 multiply/add may
        # differ by one ULP from the fused multiply-add inside a compiled branch.
        np.testing.assert_array_equal(native, jax.jit(field)(x))


def test_unweighted_target_values_cannot_change_guidance():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from latency_meta_mdp.policy.openpi.rtc import rtc_guided_velocity

    x = jnp.zeros((1, 3, 2))
    weights = jnp.zeros_like(x).at[:, 0, 0].set(1)
    target = jnp.ones_like(x)
    poisoned = target.at[:, 1:, :].set(999).at[:, :, 1].set(-999)

    def field(value):
        return 0.2 * value

    a = rtc_guided_velocity(field, x, 0.4, target, weights, max_guidance_weight=5)
    b = rtc_guided_velocity(field, x, 0.4, poisoned, weights, max_guidance_weight=5)
    np.testing.assert_array_equal(a, b)
