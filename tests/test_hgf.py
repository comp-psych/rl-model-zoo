"""Tests for the Hierarchical Gaussian Filter model."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.hgf import HierarchicalGaussianFilter


class TestHGFSimulation:
    """Test simulation functionality."""

    def test_simulate_runs_without_errors(self):
        """Simulation should complete without raising exceptions."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        assert "outcomes" in data
        assert "true_probs" in data
        assert "mu1" in data
        assert "mu2" in data
        assert "pe1" in data
        assert "pe2" in data
        assert "choices" in data

    def test_simulate_output_shapes(self):
        """Output arrays should have correct shapes."""
        n_trials = 200
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=n_trials, rng=rng)
        assert len(data["outcomes"]) == n_trials
        assert len(data["mu1"]) == n_trials
        assert len(data["mu2"]) == n_trials
        assert len(data["pe1"]) == n_trials
        assert len(data["choices"]) == n_trials

    def test_simulate_outcomes_binary(self):
        """Outcomes should be binary."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        assert np.all((data["outcomes"] == 0) | (data["outcomes"] == 1))

    def test_simulate_beliefs_no_nan(self):
        """Beliefs should not contain NaN."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, rng=rng)
        assert not np.any(np.isnan(data["mu1"]))
        assert not np.any(np.isnan(data["mu2"]))
        assert not np.any(np.isnan(data["sigma1"]))
        assert not np.any(np.isnan(data["sigma2"]))

    def test_simulate_volatile_env(self):
        """Volatile environment should have switching probabilities."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, volatile=True, block_length=40, rng=rng)
        probs = data["true_probs"]
        # Should have at least one switch
        changes = np.sum(np.abs(np.diff(probs)) > 0.1)
        assert changes > 0, "Volatile environment should have probability switches"

    def test_simulate_deterministic_with_seed(self):
        """Same seed should produce same results."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        data1 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        data2 = model.simulate(n_trials=100, rng=np.random.RandomState(123))
        np.testing.assert_array_equal(data1["outcomes"], data2["outcomes"])
        np.testing.assert_array_almost_equal(data1["mu1"], data2["mu1"])


class TestHGFFitting:
    """Test parameter recovery."""

    def test_parameter_recovery(self):
        """Fitted parameters should be within 20% of true values."""
        true_omega = -3.0
        true_kappa = 1.0
        model = HierarchicalGaussianFilter(omega=true_omega, kappa=true_kappa)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=500, volatile=True, rng=rng)

        fit_model = HierarchicalGaussianFilter(omega=-2.0, kappa=0.5)
        result = fit_model.fit(
            data["outcomes"], data["choices"],
            n_starts=20,
            rng=np.random.RandomState(99),
        )

        # For omega, use absolute error since it can be negative
        omega_error = abs(result["omega"] - true_omega) / abs(true_omega)
        kappa_error = abs(result["kappa"] - true_kappa) / true_kappa
        assert omega_error < 0.20, (
            f"Omega recovery: true={true_omega}, recovered={result['omega']:.4f}, "
            f"error={omega_error:.2%}"
        )
        assert kappa_error < 0.20, (
            f"Kappa recovery: true={true_kappa}, recovered={result['kappa']:.4f}, "
            f"error={kappa_error:.2%}"
        )

    def test_fit_returns_required_keys(self):
        """Fit result should contain expected keys."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, rng=rng)
        result = model.fit(
            data["outcomes"], data["choices"],
            rng=np.random.RandomState(99),
        )
        assert "omega" in result
        assert "kappa" in result
        assert "nll" in result
        assert "bic" in result


class TestHGFEdgeCases:
    """Test edge cases."""

    def test_invalid_omega_raises(self):
        """Omega out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            HierarchicalGaussianFilter(omega=-15.0)
        with pytest.raises(ValueError):
            HierarchicalGaussianFilter(omega=5.0)

    def test_invalid_kappa_raises(self):
        """Kappa out of bounds should raise ValueError."""
        with pytest.raises(ValueError):
            HierarchicalGaussianFilter(kappa=0.0)
        with pytest.raises(ValueError):
            HierarchicalGaussianFilter(kappa=-1.0)
        with pytest.raises(ValueError):
            HierarchicalGaussianFilter(kappa=6.0)

    def test_stable_environment(self):
        """Model should work in stable (non-volatile) environment."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=200, volatile=False, rng=rng)
        assert not np.any(np.isnan(data["mu1"]))
        assert not np.any(np.isinf(data["mu1"]))

    def test_mu1_tracks_probability(self):
        """Level-1 beliefs should roughly track true probabilities."""
        model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
        rng = np.random.RandomState(42)
        data = model.simulate(n_trials=300, volatile=False, rng=rng)
        # In a stable env (prob=0.6), beliefs should converge toward 0.6
        late_beliefs = data["mu1"][-50:]
        mean_late = np.mean(late_beliefs)
        # Allow wide tolerance since HGF beliefs are in sigmoid space
        assert 0.2 < mean_late < 0.9
