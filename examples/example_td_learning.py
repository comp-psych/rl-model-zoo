"""
Example: Temporal Difference Learning

Demonstrates simulation and parameter recovery for the TD learning model
in a multi-step sequential task.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from models.td_learning import TDLearning


def main():
    rng = np.random.RandomState(42)

    true_alpha = 0.3
    true_gamma = 0.85
    model = TDLearning(alpha=true_alpha, gamma=true_gamma)
    print(f"True alpha: {true_alpha}, True gamma: {true_gamma}")

    # Simulate
    data = model.simulate(n_trials=300, n_states=5, rng=rng)
    print(f"Simulated {len(data['states'])} episodes")
    print(f"Final state values: {data['value_history'][-1]}")

    # Fit
    fit_model = TDLearning(alpha=0.5, gamma=0.5)
    result = fit_model.fit(
        data["states"], data["rewards"], data["choices"],
        n_states=5, n_starts=10,
        rng=np.random.RandomState(99)
    )

    print(f"\nRecovered alpha: {result['alpha']:.4f}")
    print(f"Recovered gamma: {result['gamma']:.4f}")
    print(f"NLL: {result['nll']:.2f}, BIC: {result['bic']:.2f}")


if __name__ == "__main__":
    main()
