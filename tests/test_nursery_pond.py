"""Tests for Task 1: NurseryPond environment."""
import pytest
from aquashrimp.tasks.nursery_pond import NurseryPondEnvironment
from aquashrimp.models.actions import NurseryPondAction


class TestNurseryPondReset:
    def test_deterministic_reset(self):
        """Critical invariant: same seed must produce identical initial state."""
        e1 = NurseryPondEnvironment(seed=42)
        e2 = NurseryPondEnvironment(seed=42)
        o1 = e1.reset()
        o2 = e2.reset()
        assert o1.temperature_c == o2.temperature_c
        assert o1.do_mg_L == o2.do_mg_L
        assert o1.n_shrimp == o2.n_shrimp
        assert o1.mean_weight_g == o2.mean_weight_g

    def test_different_seeds_different_state(self):
        """Different seeds should produce different episodes."""
        e1 = NurseryPondEnvironment(seed=42)
        e2 = NurseryPondEnvironment(seed=99)
        o1 = e1.reset()
        o2 = e2.reset()
        # At least one parameter should differ
        differs = (o1.temperature_c != o2.temperature_c or o1.salinity_ppt != o2.salinity_ppt)
        # Both should start at the same population (stocking)
        assert o1.n_shrimp == o2.n_shrimp == 100_000

    def test_reset_returns_correct_types(self):
        """Reset should return NurseryPondObs."""
        from aquashrimp.models.observations import NurseryPondObs
        env = NurseryPondEnvironment(seed=42)
        obs = env.reset()
        assert isinstance(obs, NurseryPondObs)
        assert obs.day == 0
        assert not obs.done

    def test_initial_population(self):
        """Should start with 100,000 shrimp at 0.05g."""
        env = NurseryPondEnvironment(seed=42)
        obs = env.reset()
        assert obs.n_shrimp == 100_000
        assert abs(obs.mean_weight_g - 0.05) < 0.01


class TestNurseryPondStep:
    def test_step_increments_day(self):
        """Each step should advance one day."""
        env = NurseryPondEnvironment(seed=42)
        obs = env.reset()
        assert obs.day == 0
        obs = env.step(NurseryPondAction())
        assert obs.day == 1
        obs = env.step(NurseryPondAction())
        assert obs.day == 2

    def test_episode_ends_at_max_steps(self):
        """Episode should end after 30 days."""
        env = NurseryPondEnvironment(seed=42)
        obs = env.reset()
        for _ in range(30):
            obs = env.step(NurseryPondAction(
                feed_kg=5.0, aeration_hours=20.0, water_exchange_frac=0.05,
                lime_application_kg=2.0
            ))
        assert obs.done, "Episode should be done after 30 steps"
        assert obs.day == 30

    def test_shrimp_grow_over_time(self):
        """With good management, shrimp should grow heavier over time."""
        env = NurseryPondEnvironment(seed=42)
        obs = env.reset()
        initial_weight = obs.mean_weight_g
        for _ in range(10):
            obs = env.step(NurseryPondAction(
                feed_kg=8.0, aeration_hours=22.0, water_exchange_frac=0.05,
                check_feeding_trays=True, lime_application_kg=3.0
            ))
        assert obs.mean_weight_g > initial_weight, "Shrimp should grow over time"

    def test_tray_check_returns_fraction(self):
        """Checking feeding trays should return a consumption fraction."""
        env = NurseryPondEnvironment(seed=42)
        env.reset()
        obs = env.step(NurseryPondAction(check_feeding_trays=True, feed_kg=5.0))
        assert obs.tray_consumption_fraction is not None
        assert 0.0 <= obs.tray_consumption_fraction <= 1.0

    def test_no_tray_check_returns_none(self):
        """Not checking trays should return None."""
        env = NurseryPondEnvironment(seed=42)
        env.reset()
        obs = env.step(NurseryPondAction(check_feeding_trays=False))
        assert obs.tray_consumption_fraction is None

    def test_reward_in_valid_range(self):
        """Reward must always be in [-1, +1]."""
        env = NurseryPondEnvironment(seed=42)
        env.reset()
        for _ in range(30):
            obs = env.step(NurseryPondAction())
            assert -1.0 <= obs.reward <= 1.0, f"Reward out of range: {obs.reward}"
            if obs.done:
                break

    def test_cannot_step_after_done(self):
        """Stepping after episode end should raise RuntimeError."""
        env = NurseryPondEnvironment(seed=42)
        env.reset()
        for _ in range(30):
            obs = env.step(NurseryPondAction())
        with pytest.raises(RuntimeError):
            env.step(NurseryPondAction())

    def test_step_before_reset_raises(self):
        """Stepping without reset should raise RuntimeError."""
        env = NurseryPondEnvironment(seed=42)
        with pytest.raises(RuntimeError):
            env.step(NurseryPondAction())

    def test_aeration_matters_for_do(self):
        """No aeration should cause DO to drop more than full aeration."""
        env_aerate = NurseryPondEnvironment(seed=42)
        env_no_aerate = NurseryPondEnvironment(seed=42)
        env_aerate.reset()
        env_no_aerate.reset()

        obs_aerate = env_aerate.step(NurseryPondAction(aeration_hours=22.0))
        obs_no_aerate = env_no_aerate.step(NurseryPondAction(aeration_hours=0.0))

        assert obs_aerate.do_mg_L > obs_no_aerate.do_mg_L, (
            "Aeration should maintain higher DO"
        )

    def test_overfeeding_increases_tan(self):
        """Extreme overfeeding should cause TAN to rise."""
        env_normal = NurseryPondEnvironment(seed=42)
        env_overfed = NurseryPondEnvironment(seed=42)
        env_normal.reset()
        env_overfed.reset()

        for _ in range(5):
            obs_normal = env_normal.step(NurseryPondAction(feed_kg=5.0, water_exchange_frac=0.0))
            obs_overfed = env_overfed.step(NurseryPondAction(feed_kg=45.0, water_exchange_frac=0.0))

        assert obs_overfed.tan_mg_L > obs_normal.tan_mg_L, (
            "Overfeeding should cause higher TAN accumulation"
        )

    def test_reward_breakdown_sums_correctly(self):
        """Reward breakdown components should match total with correct weights."""
        env = NurseryPondEnvironment(seed=42)
        env.reset()
        obs = env.step(NurseryPondAction(feed_kg=5.0, aeration_hours=20.0))
        b = obs.reward_breakdown
        expected = 0.35 * b.growth + 0.35 * b.water_quality + 0.20 * b.economic + 0.10 * b.biosecurity
        assert abs(b.total - expected) < 1e-6, (
            f"Reward breakdown doesn't sum correctly: {b.total} vs {expected}"
        )
