"""Tests for the Dual Learning Rate model."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.dual_lr import DualLearningRate


class TestDualLRSimulation:
    """Test simulation functionality."""

    def test_simulate_runs_without_errors(self):
        """Simulation should complete without raising exceptions."""
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        assert "choices" in data
        assert "rewards" in data
        assert "values" in data
        assert "prediction_errors" in data

    def test_simulate_output_shapes(self):
        """Output arrays should have correct shapes."""
        n_trials = 300
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=n_trials, rng=rng)
        assert len(data["choices"]) == n_trials
        assert len(data["rewards"]) == n_trials
        assert data["values"].shape == (n_trials, 2)
        assert len(data["prediction_errors"]) == n_trials

    def test_simulate_choices_binary(self):
        """Choices should be binary."""
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert np.all((data["choices"] == 0) | (data["choices"] == 1))

    def test_simulate_rewards_binary(self):
        """Rewards should be binary."""
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert np.all((data["rewards"] == 0) | (data["rewards"] == 1))

    def test_asymmetric_learning(self):
        """Dual LR should produce different value updates for pos/neg PEs."""
        model = DualLearningRate(alpha_pos=0.8, alpha_neg=0.1, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, reward_prob=0.7, rng=rng)
        pes = data["prediction_errors"]
        # There should be both positive and negative PEs
        assert np.sum(pes > 0) > 10
        assert np.sum(pes < 0) > 10

    def test_simulate_deterministic_with_seed(self):
        """Same seed should produce same results."""
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        data1 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        data2 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        np.testing.assert_array_equal(data1["choices"], data2["choices"])
        np.testing.assert_array_equal(data1["rewards"], data2["rewards"])


class TestDualLRFitting:
    """Test parameter recovery."""

    def test_parameter_recovery(self):
        """Fitted parameters should be within 20% of true values."""
        true_alpha_pos = 0.6
        true_alpha_neg = 0.3
        true_beta = 5.0
        model = DualLearningRate(
            alpha_pos=true_alpha_pos,
            alpha_neg=true_alpha_neg,
            beta=true_beta,
        )
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=500, reward_prob=0.7, rng=rng)

        fit_model = DualLearningRate(
            alpha_pos=0.5, alpha_neg=0.5, beta=3.0
        )
        result = fit_model.fit(
            data["choices"], data["rewards"],
            n_starts=20,
            rng=np.random.RandomState(99),
        )

        apos_error = abs(result["alpha_pos"] - true_alpha_pos) / true_alpha_pos
        aneg_error = abs(result["alpha_neg"] - true_alpha_neg) / true_alpha_neg
        beta_error = abs(result["beta"] - true_beta) / true_beta
        assert apos_error < 0.20, (
            f"Alpha_pos recovery: true={true_alpha_pos}, "
            f"recovered={result['alpha_pos']:.4f}, error={apos_error:.2%}"
        )
        assert aneg_error < 0.20, (
            f"Alpha_neg recovery: true={true_alpha_neg}, "
            f"recovered={result['alpha_neg']:.4f}, error={aneg_error:.2%}"
        )
        assert beta_error < 0.20, (
            f"Beta recovery: true={true_beta}, "
            f"recovered={result['beta']:.4f}, error={beta_error:.2%}"
        )

    def test_fit_returns_required_keys(self):
        """Fit result should contain expected keys."""
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        result = model.fit(
            data["choices"], data["rewards"],
            rng=np.random.RandomState(99),
        )
        assert "alpha_pos" in result
        assert "alpha_neg" in result
        assert "beta" in result
        assert "nll" in result
        assert "bic" in result

    def test_fit_params_in_bounds(self):
        """Fitted parameters should be within valid bounds."""
        model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        result = model.fit(
            data["choices"], data["rewards"],
            rng=np.random.RandomState(99),
        )
        assert 0 < result["alpha_pos"] < 1
        assert 0 < result["alpha_neg"] < 1
        assert result["beta"] > 0


class TestDualLREdgeCases:
    """Test edge cases."""

    def test_invalid_alpha_pos_raises(self):
        """Alpha_pos out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            DualLearningRate(alpha_pos=0.0)
        with pytest.raises(ValueError):
            DualLearningRate(alpha_pos=1.0)

    def test_invalid_alpha_neg_raises(self):
        """Alpha_neg out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            DualLearningRate(alpha_neg=0.0)
        with pytest.raises(ValueError):
            DualLearningRate(alpha_neg=1.0)

    def test_invalid_beta_raises(self):
        """Beta <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            DualLearningRate(beta=0.0)
        with pytest.raises(ValueError):
            DualLearningRate(beta=-1.0)

    def test_equal_learning_rates(self):
        """Equal alpha_pos and alpha_neg should behave like single LR."""
        model = DualLearningRate(alpha_pos=0.4, alpha_neg=0.4, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert not np.any(np.isnan(data["values"]))

    def test_extreme_asymmetry(self):
        """Very asymmetric learning rates should still work."""
        model = DualLearningRate(alpha_pos=0.99, alpha_neg=0.01, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        assert not np.any(np.isnan(data["values"]))
        assert not np.any(np.isinf(data["values"]))
