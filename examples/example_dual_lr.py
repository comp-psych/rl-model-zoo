"""
Example: Dual Learning Rate Model

Demonstrates simulation and parameter recovery for the dual learning rate
model with asymmetric positive/negative prediction error processing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from models.dual_lr import DualLearningRate


def main():
    rng = np.random.RandomState(42)

    true_alpha_pos = 0.6
    true_alpha_neg = 0.3
    true_beta = 5.0
    model = DualLearningRate(
        alpha_pos=true_alpha_pos, alpha_neg=true_alpha_neg, beta=true_beta
    )
    print(f"True alpha+: {true_alpha_pos}, alpha-: {true_alpha_neg}, "
          f"beta: {true_beta}")

    # Simulate
    data = model.simulate(n_trials=400, reward_prob=0.7, rng=rng)
    print(f"Simulated {len(data['choices'])} trials")
    print(f"Mean reward: {np.mean(data['rewards']):.3f}")
    
    pes = data["prediction_errors"]
    print(f"Positive PEs: {np.sum(pes > 0)}, Negative PEs: {np.sum(pes < 0)}")

    # Fit
    fit_model = DualLearningRate(alpha_pos=0.5, alpha_neg=0.5, beta=3.0)
    result = fit_model.fit(
        data["choices"], data["rewards"],
        n_starts=15, rng=np.random.RandomState(99)
    )

    print(f"\nRecovered alpha+: {result['alpha_pos']:.4f}")
    print(f"Recovered alpha-: {result['alpha_neg']:.4f}")
    print(f"Recovered beta: {result['beta']:.4f}")
    print(f"NLL: {result['nll']:.2f}, BIC: {result['bic']:.2f}")


if __name__ == "__main__":
    main()
