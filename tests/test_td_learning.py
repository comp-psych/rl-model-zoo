"""Tests for the Temporal Difference Learning model."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.td_learning import TDLearning


class TestTDLearningSimulation:
    """Test simulation functionality."""

    def test_simulate_runs_without_errors(self):
        """Simulation should complete without raising exceptions."""
        model = TDLearning(alpha=0.3, gamma=0.9)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, n_states=5, rng=rng)
        assert "states" in data
        assert "rewards" in data
        assert "td_errors" in data
        assert "value_history" in data
        assert "choices" in data

    def test_simulate_output_lengths(self):
        """Should return correct number of episodes."""
        model = TDLearning(alpha=0.3, gamma=0.9)
        rng = np.random.RandomState(42)
        n_trials = 100
        data = model.simulate(n_trials=n_trials, rng=rng)
        assert len(data["states"]) == n_trials
        assert len(data["rewards"]) == n_trials
        assert data["value_history"].shape[0] == n_trials

    def test_simulate_states_valid(self):
        """States should be valid indices."""
        n_states = 5
        model = TDLearning(alpha=0.3, gamma=0.9)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=100, n_states=n_states, rng=rng)
        for ep_states in data["states"]:
            assert np.all(ep_states >= 0)
            assert np.all(ep_states < n_states)

    def test_simulate_value_convergence(self):
        """State values should converge with enough trials."""
        model = TDLearning(alpha=0.2, gamma=0.9)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=500, n_states=5, rng=rng)
        # Values for later states should be higher (closer to reward)
        final_values = data["value_history"][-1]
        # Terminal state gets reward, so later states should generally be valued higher
        assert final_values[-1] > 0

    def test_simulate_deterministic_with_seed(self):
        """Same seed should produce same results."""
        model = TDLearning(alpha=0.3, gamma=0.9)
        data1 = model.simulate(n_trials=50, rng=np.random.RandomState(123))
        data2 = model.simulate(n_trials=50, rng=np.random.RandomState(123))
        for i in range(len(data1["states"])):
            np.testing.assert_array_equal(data1["states"][i], data2["states"][i])


class TestTDLearningFitting:
    """Test parameter recovery."""

    def test_parameter_recovery(self):
        """Fitted parameters should be within 20% of true values."""
        true_alpha = 0.3
        true_gamma = 0.85
        model = TDLearning(alpha=true_alpha, gamma=true_gamma)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=400, n_states=5, rng=rng)

        fit_model = TDLearning(alpha=0.5, gamma=0.5)
        result = fit_model.fit(
            data["states"], data["rewards"], data["choices"],
            n_states=5, n_starts=15,
            rng=np.random.RandomState(99),
        )

        alpha_error = abs(result["alpha"] - true_alpha) / true_alpha
        gamma_error = abs(result["gamma"] - true_gamma) / true_gamma
        assert alpha_error < 0.20, (
            f"Alpha recovery: true={true_alpha}, recovered={result['alpha']:.4f}, "
            f"error={alpha_error:.2%}"
        )
        assert gamma_error < 0.20, (
            f"Gamma recovery: true={true_gamma}, recovered={result['gamma']:.4f}, "
            f"error={gamma_error:.2%}"
        )

    def test_fit_returns_required_keys(self):
        """Fit result should contain expected keys."""
        model = TDLearning(alpha=0.3, gamma=0.9)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        result = model.fit(
            data["states"], data["rewards"], data["choices"],
            n_states=5, rng=np.random.RandomState(99),
        )
        assert "alpha" in result
        assert "gamma" in result
        assert "nll" in result
        assert "bic" in result


class TestTDLearningEdgeCases:
    """Test edge cases."""

    def test_invalid_alpha_raises(self):
        """Alpha out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            TDLearning(alpha=0.0)
        with pytest.raises(ValueError):
            TDLearning(alpha=1.0)

    def test_invalid_gamma_raises(self):
        """Gamma out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            TDLearning(gamma=0.0)
        with pytest.raises(ValueError):
            TDLearning(gamma=1.5)

    def test_gamma_one_is_valid(self):
        """Gamma of 1.0 should be valid (no discounting)."""
        model = TDLearning(alpha=0.3, gamma=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=100, rng=rng)
        assert not np.any(np.isnan(data["value_history"]))
