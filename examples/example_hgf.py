"""
Example: Hierarchical Gaussian Filter

Demonstrates simulation and parameter recovery for the two-level HGF
in a volatile binary outcome task.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from models.hgf import HierarchicalGaussianFilter


def main():
    rng = np.random.RandomState(42)

    true_omega = -3.0
    true_kappa = 1.0
    model = HierarchicalGaussianFilter(omega=true_omega, kappa=true_kappa)
    print(f"True omega: {true_omega}, True kappa: {true_kappa}")

    # Simulate volatile environment
    data = model.simulate(
        n_trials=400, volatile=True, block_length=40, rng=rng
    )
    print(f"Simulated {len(data['outcomes'])} trials")
    print(f"Number of probability switches: "
          f"{np.sum(np.abs(np.diff(data['true_probs'])) > 0.1)}")

    # Fit
    fit_model = HierarchicalGaussianFilter(omega=-2.0, kappa=0.5)
    result = fit_model.fit(
        data["outcomes"], data["choices"],
        n_starts=15, rng=np.random.RandomState(99)
    )

    print(f"\nRecovered omega: {result['omega']:.4f}")
    print(f"Recovered kappa: {result['kappa']:.4f}")
    print(f"NLL: {result['nll']:.2f}, BIC: {result['bic']:.2f}")


if __name__ == "__main__":
    main()
