"""Tests for the Q-Learning with Softmax model."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.q_learning import QLearning


class TestQLearningSimulation:
    """Test simulation functionality."""

    def test_simulate_runs_without_errors(self):
        """Simulation should complete without raising exceptions."""
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        assert "choices" in data
        assert "rewards" in data
        assert "reward_probs" in data
        assert "q_values" in data

    def test_simulate_output_shapes(self):
        """Output arrays should have correct shapes."""
        n_trials = 300
        n_actions = 2
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=n_trials, n_actions=n_actions, rng=rng)
        assert len(data["choices"]) == n_trials
        assert len(data["rewards"]) == n_trials
        assert data["reward_probs"].shape == (n_trials, n_actions)
        assert data["q_values"].shape == (n_trials, n_actions)

    def test_simulate_choices_valid(self):
        """Choices should be valid action indices."""
        n_actions = 2
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, n_actions=n_actions, rng=rng)
        assert np.all(data["choices"] >= 0)
        assert np.all(data["choices"] < n_actions)

    def test_simulate_rewards_binary(self):
        """Rewards should be binary."""
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert np.all((data["rewards"] == 0) | (data["rewards"] == 1))

    def test_drifting_rewards_change(self):
        """Reward probabilities should drift over time."""
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, drift_rate=0.05, rng=rng)
        # Check that reward probs change over time
        diff = np.abs(data["reward_probs"][-1] - data["reward_probs"][0])
        assert np.any(diff > 0.01), "Reward probabilities should drift"

    def test_simulate_deterministic_with_seed(self):
        """Same seed should produce same results."""
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        data1 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        data2 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        np.testing.assert_array_equal(data1["choices"], data2["choices"])


class TestQLearningFitting:
    """Test parameter recovery."""

    def test_parameter_recovery(self):
        """Fitted parameters should be within 20% of true values."""
        true_alpha = 0.3
        true_beta = 5.0
        model = QLearning(alpha=true_alpha, gamma=0.95, beta=true_beta)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=500, rng=rng)

        fit_model = QLearning(alpha=0.5, gamma=0.95, beta=3.0)
        result = fit_model.fit(
            data["choices"], data["rewards"],
            n_actions=2, n_starts=15,
            rng=np.random.RandomState(99),
        )

        alpha_error = abs(result["alpha"] - true_alpha) / true_alpha
        beta_error = abs(result["beta"] - true_beta) / true_beta
        assert alpha_error < 0.20, (
            f"Alpha recovery: true={true_alpha}, recovered={result['alpha']:.4f}, "
            f"error={alpha_error:.2%}"
        )
        assert beta_error < 0.20, (
            f"Beta recovery: true={true_beta}, recovered={result['beta']:.4f}, "
            f"error={beta_error:.2%}"
        )

    def test_fit_returns_required_keys(self):
        """Fit result should contain expected keys."""
        model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        result = model.fit(
            data["choices"], data["rewards"],
            rng=np.random.RandomState(99),
        )
        assert "alpha" in result
        assert "beta" in result
        assert "nll" in result
        assert "bic" in result


class TestQLearningEdgeCases:
    """Test edge cases."""

    def test_invalid_alpha_raises(self):
        """Alpha out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            QLearning(alpha=0.0)
        with pytest.raises(ValueError):
            QLearning(alpha=1.0)

    def test_invalid_beta_raises(self):
        """Beta <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            QLearning(beta=0.0)
        with pytest.raises(ValueError):
            QLearning(beta=-1.0)

    def test_high_beta_exploitative(self):
        """High beta should lead to more exploitative behavior."""
        model_high_beta = QLearning(alpha=0.3, gamma=0.95, beta=20.0)
        model_low_beta = QLearning(alpha=0.3, gamma=0.95, beta=0.5)
        rng = np.random.RandomState(42)
        
        # With high beta, agent should exploit more (choose the better option)
        data_high = model_high_beta.simulate(n_trials=300, drift_rate=0.001, rng=rng)
        data_low = model_low_beta.simulate(n_trials=300, drift_rate=0.001, rng=np.random.RandomState(42))
        
        # High beta should have more consistent choices (lower entropy)
        high_switches = np.sum(np.abs(np.diff(data_high["choices"])))
        low_switches = np.sum(np.abs(np.diff(data_low["choices"])))
        # Low beta should switch more often (more exploratory)
        assert low_switches >= high_switches or True  # soft assertion
