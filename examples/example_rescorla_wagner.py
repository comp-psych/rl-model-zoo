"""
Example: Rescorla-Wagner Model

Demonstrates simulation and parameter recovery for the Rescorla-Wagner
associative learning model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from models.rescorla_wagner import RescorlaWagner


def main():
    # Set random seed for reproducibility
    rng = np.random.RandomState(42)

    # Create model with known parameters
    true_alpha = 0.4
    model = RescorlaWagner(alpha=true_alpha)
    print(f"True alpha: {true_alpha}")

    # Simulate data
    data = model.simulate(n_trials=300, reward_prob=0.7, rng=rng)
    print(f"Simulated {len(data['choices'])} trials")
    print(f"Mean reward: {np.mean(data['rewards']):.3f}")
    print(f"Choice rate: {np.mean(data['choices']):.3f}")

    # Fit model to recover parameters
    fit_model = RescorlaWagner(alpha=0.5)  # start from different value
    result = fit_model.fit(
        data["choices"], data["rewards"],
        n_starts=10, rng=np.random.RandomState(99)
    )

    print(f"\nRecovered alpha: {result['alpha']:.4f}")
    print(f"Negative log-likelihood: {result['nll']:.2f}")
    print(f"BIC: {result['bic']:.2f}")

    error = abs(result["alpha"] - true_alpha) / true_alpha
    print(f"Recovery error: {error:.1%}")


if __name__ == "__main__":
    main()
