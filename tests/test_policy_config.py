from pathlib import Path


def test_experiment_recipe_preserves_current_clean_schedule_and_contract():
    from latency_meta_mdp.policy.config import load_training_job, resolve_policy_profile
    from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule

    job = load_training_job(Path("configs/experiments/moving_ball/l2/clean.yaml"))
    profile = resolve_policy_profile(job)
    schedule = resolve_sft_schedule(
        profile=profile, request=SFTLaunchRequest(2, "test", "formal", False, 8, 256)
    )
    assert profile.state_dim == 16 and profile.masked_action_tails and profile.discrete_state_input
    assert schedule.num_train_steps == 3000
    assert schedule.warmup_steps == 150
    assert schedule.expected_checkpoint_steps == (1000, 2000, 3000)
    assert profile.peak_learning_rate == 5e-5 and profile.decay_learning_rate == 5e-6
    assert job["training"].is_file()
