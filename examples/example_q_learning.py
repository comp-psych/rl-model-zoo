"""
Example: Q-Learning with Softmax

Demonstrates simulation and parameter recovery for the Q-learning model
with drifting two-armed bandit task.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from models.q_learning import QLearning


def main():
    rng = np.random.RandomState(42)

    true_alpha = 0.3
    true_beta = 5.0
    model = QLearning(alpha=true_alpha, gamma=0.95, beta=true_beta)
    print(f"True alpha: {true_alpha}, True beta: {true_beta}")

    # Simulate drifting bandit
    data = model.simulate(n_trials=400, n_actions=2, drift_rate=0.02, rng=rng)
    print(f"Simulated {len(data['choices'])} trials")
    print(f"Mean reward: {np.mean(data['rewards']):.3f}")

    # Fit
    fit_model = QLearning(alpha=0.5, gamma=0.95, beta=3.0)
    result = fit_model.fit(
        data["choices"], data["rewards"],
        n_actions=2, n_starts=10,
        rng=np.random.RandomState(99)
    )

    print(f"\nRecovered alpha: {result['alpha']:.4f}")
    print(f"Recovered beta: {result['beta']:.4f}")
    print(f"NLL: {result['nll']:.2f}, BIC: {result['bic']:.2f}")


if __name__ == "__main__":
    main()
