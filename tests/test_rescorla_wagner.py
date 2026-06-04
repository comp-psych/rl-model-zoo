"""Tests for the Rescorla-Wagner model."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.rescorla_wagner import RescorlaWagner


class TestRescorlaWagnerSimulation:
    """Test simulation functionality."""

    def test_simulate_runs_without_errors(self):
        """Simulation should complete without raising exceptions."""
        model = RescorlaWagner(alpha=0.3)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert "rewards" in data
        assert "values" in data
        assert "prediction_errors" in data
        assert "choices" in data

    def test_simulate_output_shapes(self):
        """Output arrays should have correct shapes."""
        model = RescorlaWagner(alpha=0.5)
        rng = np.random.RandomState(42)
        n_trials = 300
        data = model.simulate(n_trials=n_trials, rng=rng)
        assert len(data["rewards"]) == n_trials
        assert len(data["values"]) == n_trials
        assert len(data["prediction_errors"]) == n_trials
        assert len(data["choices"]) == n_trials

    def test_simulate_rewards_binary(self):
        """Rewards should be binary (0 or 1)."""
        model = RescorlaWagner(alpha=0.5)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert np.all((data["rewards"] == 0) | (data["rewards"] == 1))

    def test_simulate_values_bounded(self):
        """Values should stay bounded between 0 and 1."""
        model = RescorlaWagner(alpha=0.8)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=500, rng=rng)
        assert np.all(data["values"] >= -0.01)
        assert np.all(data["values"] <= 1.01)

    def test_simulate_deterministic_with_seed(self):
        """Same seed should produce same results."""
        model = RescorlaWagner(alpha=0.3)
        data1 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        data2 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        np.testing.assert_array_equal(data1["choices"], data2["choices"])
        np.testing.assert_array_equal(data1["rewards"], data2["rewards"])


class TestRescorlaWagnerFitting:
    """Test parameter recovery via fitting."""

    def test_parameter_recovery_alpha(self):
        """Fitted alpha should be within 20% of true alpha."""
        true_alpha = 0.4
        model = RescorlaWagner(alpha=true_alpha)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=500, reward_prob=0.7, rng=rng)

        fit_model = RescorlaWagner(alpha=0.5)
        result = fit_model.fit(
            data["choices"], data["rewards"],
            n_starts=15,
            rng=np.random.RandomState(99),
        )

        recovered_alpha = result["alpha"]
        relative_error = abs(recovered_alpha - true_alpha) / true_alpha
        assert relative_error < 0.20, (
            f"Alpha recovery failed: true={true_alpha}, "
            f"recovered={recovered_alpha:.4f}, error={relative_error:.2%}"
        )

    def test_fit_returns_required_keys(self):
        """Fit result should contain expected keys."""
        model = RescorlaWagner(alpha=0.3)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        result = model.fit(
            data["choices"], data["rewards"],
            rng=np.random.RandomState(99),
        )
        assert "alpha" in result
        assert "nll" in result
        assert "bic" in result

    def test_fit_alpha_in_bounds(self):
        """Fitted alpha should be in (0, 1)."""
        model = RescorlaWagner(alpha=0.5)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        result = model.fit(
            data["choices"], data["rewards"],
            rng=np.random.RandomState(99),
        )
        assert 0 < result["alpha"] < 1


class TestRescorlaWagnerEdgeCases:
    """Test edge cases."""

    def test_invalid_alpha_raises(self):
        """Alpha out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            RescorlaWagner(alpha=0.0)
        with pytest.raises(ValueError):
            RescorlaWagner(alpha=1.0)
        with pytest.raises(ValueError):
            RescorlaWagner(alpha=-0.1)
        with pytest.raises(ValueError):
            RescorlaWagner(alpha=1.5)

    def test_high_learning_rate(self):
        """High alpha should still produce valid outputs."""
        model = RescorlaWagner(alpha=0.99)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert not np.any(np.isnan(data["values"]))
        assert not np.any(np.isinf(data["values"]))

    def test_low_learning_rate(self):
        """Low alpha should still produce valid outputs."""
        model = RescorlaWagner(alpha=0.01)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert not np.any(np.isnan(data["values"]))
        # With very low alpha, values should change slowly
        diffs = np.abs(np.diff(data["values"]))
        assert np.mean(diffs) < 0.05
