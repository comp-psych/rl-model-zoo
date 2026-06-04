"""
Hierarchical Gaussian Filter (HGF) for Computational Psychiatry.

Mathematical Model
------------------
The two-level HGF models belief updating under uncertainty with a
hierarchical structure:

Level 1 (Value level, x1):
    Tracks the probability of an outcome via a sigmoid transform:
    mu1_hat(t) = sigmoid(mu1(t-1))
    
    Update: mu1(t) = mu1(t-1) + (sigma1(t) * delta1(t))
    where delta1 = y(t) - mu1_hat(t), the prediction error

Level 2 (Volatility level, x2):
    Tracks the log-volatility (how much x1 changes):
    mu2(t) = mu2(t-1) + (sigma2(t) * delta2(t))

    Volatility coupling: sigma1(t) depends on exp(kappa * mu2(t) + omega)

Parameters:
    - omega (tonic volatility): baseline rate of environmental change
    - kappa (coupling strength): how much x2 influences x1 uncertainty

Clinical Relevance
------------------
- **Psychosis**: Aberrant precision weighting at higher hierarchical levels,
  leading to excessive influence of prediction errors (delusions).
- **Anxiety**: Overestimation of environmental volatility (elevated x2),
  leading to excessive uncertainty and hypervigilance.
- **Autism**: Inflexible precision estimates, difficulty adapting to
  changing contingencies.

References
----------
Mathys, C. D., et al. (2014). Uncertainty in perception and the Hierarchical
Gaussian Filter. Frontiers in Human Neuroscience, 8, 825.
Mathys, C., et al. (2011). A Bayesian foundation for individual learning
under uncertainty. Frontiers in Human Neuroscience, 5, 39.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional


class HierarchicalGaussianFilter:
    """Two-level Hierarchical Gaussian Filter.

    Parameters
    ----------
    omega : float
        Tonic volatility parameter. Controls the baseline rate of
        environmental change. Typically in (-6, 0).
    kappa : float
        Coupling strength between levels. Controls how much the
        volatility level influences value-level uncertainty.
        Typically in (0, 2).

    Attributes
    ----------
    omega : float
        Tonic volatility.
    kappa : float
        Coupling strength.
    mu1 : float or None
        Level-1 mean after fitting/simulation.
    mu2 : float or None
        Level-2 mean after fitting/simulation.
    """

    def __init__(self, omega: float = -3.0, kappa: float = 1.0):
        if not -10 < omega < 2:
            raise ValueError(f"omega should be in (-10, 2), got {omega}")
        if not 0 < kappa < 5:
            raise ValueError(f"kappa should be in (0, 5), got {kappa}")
        self.omega = omega
        self.kappa = kappa
        self.mu1: Optional[float] = None
        self.mu2: Optional[float] = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def _hgf_update(
        self,
        y: float,
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
    ) -> Dict[str, float]:
        """Perform a single HGF update step.

        Parameters
        ----------
        y : float
            Binary observation (0 or 1).
        mu1 : float
            Level-1 mean (in logit space).
        sigma1 : float
            Level-1 variance.
        mu2 : float
            Level-2 mean (log-volatility).
        sigma2 : float
            Level-2 variance.

        Returns
        -------
        updates : dict
            Updated mu1, sigma1, mu2, sigma2, pe1, pe2.
        """
        # Level 1 prediction
        mu1_hat = self._sigmoid(mu1)
        mu1_hat = np.clip(mu1_hat, 1e-8, 1.0 - 1e-8)

        # Level 1 variance update
        # Prediction variance depends on level 2
        pi1_hat = 1.0 / (sigma1 + np.exp(self.kappa * mu2 + self.omega))
        sigma1_new = 1.0 / pi1_hat

        # Level 1 prediction error (surprise)
        pe1 = y - mu1_hat

        # Level 1 mean update using prediction error weighted by precision
        # The update in logit space uses the derivative of sigmoid
        mu1_new = mu1 + sigma1_new * pe1

        # Level 2: volatility update
        # Precision-weighted volatility PE
        v_pe = (sigma1_new + (mu1_new - mu1) ** 2) / \
               (sigma1 + np.exp(self.kappa * mu2 + self.omega)) - 1.0

        # Level 2 variance update
        pi2_hat = 1.0 / (sigma2 + np.exp(self.kappa * mu2 + self.omega))
        sigma2_new = max(1.0 / pi2_hat, 1e-8)

        # Level 2 mean update
        pe2 = self.kappa * sigma2_new * v_pe / 2.0
        mu2_new = mu2 + pe2

        return {
            "mu1": mu1_new,
            "sigma1": sigma1_new,
            "mu2": mu2_new,
            "sigma2": sigma2_new,
            "pe1": pe1,
            "pe2": pe2,
            "mu1_hat": mu1_hat,
        }

    def simulate(
        self,
        n_trials: int = 300,
        volatile: bool = True,
        block_length: int = 40,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate a volatile binary outcome task.

        Generates a sequence of binary outcomes where the underlying
        probability switches between blocks, creating a volatile
        environment.

        Parameters
        ----------
        n_trials : int
            Number of trials.
        volatile : bool
            Whether to use a volatile schedule with switching probabilities.
        block_length : int
            Length of each block before probability switch.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'outcomes': binary outcome array
            - 'true_probs': true generating probabilities
            - 'mu1': level-1 mean trajectory (belief about probability)
            - 'mu2': level-2 mean trajectory (belief about volatility)
            - 'sigma1': level-1 variance trajectory
            - 'sigma2': level-2 variance trajectory
            - 'pe1': level-1 prediction errors
            - 'pe2': level-2 prediction errors
            - 'choices': model choices based on beliefs
        """
        if rng is None:
            rng = np.random.RandomState()

        # Generate volatile probability schedule
        true_probs = np.zeros(n_trials)
        if volatile:
            high, low = 0.8, 0.2
            current_prob = high
            for t in range(n_trials):
                if t > 0 and t % block_length == 0:
                    current_prob = low if current_prob == high else high
                true_probs[t] = current_prob
        else:
            true_probs[:] = 0.6

        # Generate outcomes
        outcomes = (rng.random(n_trials) < true_probs).astype(float)

        # Initialize beliefs
        mu1 = 0.0   # logit(0.5) = 0
        sigma1 = 1.0
        mu2 = 0.0   # initial log-volatility
        sigma2 = 1.0

        # Storage
        mu1_traj = np.zeros(n_trials)
        mu2_traj = np.zeros(n_trials)
        sigma1_traj = np.zeros(n_trials)
        sigma2_traj = np.zeros(n_trials)
        pe1_traj = np.zeros(n_trials)
        pe2_traj = np.zeros(n_trials)
        choices = np.zeros(n_trials, dtype=int)

        for t in range(n_trials):
            # Generate choice based on current belief
            p_choose_1 = self._sigmoid(mu1)
            choices[t] = int(rng.random() < p_choose_1)

            # Update beliefs based on outcome
            result = self._hgf_update(outcomes[t], mu1, sigma1, mu2, sigma2)

            mu1_traj[t] = result["mu1_hat"]
            mu2_traj[t] = result["mu2"]
            sigma1_traj[t] = result["sigma1"]
            sigma2_traj[t] = result["sigma2"]
            pe1_traj[t] = result["pe1"]
            pe2_traj[t] = result["pe2"]

            mu1 = result["mu1"]
            sigma1 = result["sigma1"]
            mu2 = result["mu2"]
            sigma2 = result["sigma2"]

        self.mu1 = mu1
        self.mu2 = mu2

        return {
            "outcomes": outcomes,
            "true_probs": true_probs,
            "mu1": mu1_traj,
            "mu2": mu2_traj,
            "sigma1": sigma1_traj,
            "sigma2": sigma2_traj,
            "pe1": pe1_traj,
            "pe2": pe2_traj,
            "choices": choices,
        }

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        outcomes: np.ndarray,
        choices: np.ndarray,
    ) -> float:
        """Compute negative log-likelihood for HGF.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [omega, kappa].
        outcomes : np.ndarray
            Binary outcome array.
        choices : np.ndarray
            Binary choice array.

        Returns
        -------
        nll : float
            Negative log-likelihood.
        """
        omega, kappa = params

        if not (-10 < omega < 2) or not (0 < kappa < 5):
            return 1e10

        def sigmoid(x):
            if x >= 0:
                return 1.0 / (1.0 + np.exp(-x))
            else:
                exp_x = np.exp(x)
                return exp_x / (1.0 + exp_x)

        mu1, sigma1 = 0.0, 1.0
        mu2, sigma2 = 0.0, 1.0
        nll = 0.0

        for t in range(len(outcomes)):
            # Choice probability
            p_choose_1 = sigmoid(mu1)
            p_choose_1 = np.clip(p_choose_1, 1e-8, 1.0 - 1e-8)

            if choices[t] == 1:
                nll -= np.log(p_choose_1)
            else:
                nll -= np.log(1.0 - p_choose_1)

            # HGF update
            mu1_hat = sigmoid(mu1)
            mu1_hat = np.clip(mu1_hat, 1e-8, 1.0 - 1e-8)

            pi1_hat = 1.0 / (sigma1 + np.exp(kappa * mu2 + omega))
            sigma1_new = 1.0 / pi1_hat
            pe1 = outcomes[t] - mu1_hat
            mu1_new = mu1 + sigma1_new * pe1

            v_pe = (sigma1_new + (mu1_new - mu1) ** 2) / \
                   (sigma1 + np.exp(kappa * mu2 + omega)) - 1.0
            pi2_hat = 1.0 / (sigma2 + np.exp(kappa * mu2 + omega))
            sigma2_new = max(1.0 / pi2_hat, 1e-8)
            pe2 = kappa * sigma2_new * v_pe / 2.0
            mu2_new = mu2 + pe2

            mu1 = mu1_new
            sigma1 = sigma1_new
            mu2 = mu2_new
            sigma2 = sigma2_new

        return nll

    def fit(
        self,
        outcomes: np.ndarray,
        choices: np.ndarray,
        n_starts: int = 15,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """Fit HGF parameters to behavioral data.

        Parameters
        ----------
        outcomes : np.ndarray
            Binary outcome array.
        choices : np.ndarray
            Binary choice array.
        n_starts : int
            Number of random starting points.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'omega': fitted tonic volatility
            - 'kappa': fitted coupling strength
            - 'nll': negative log-likelihood
            - 'bic': Bayesian Information Criterion
        """
        if rng is None:
            rng = np.random.RandomState()

        best_nll = np.inf
        best_params = None

        bounds = [(-8.0, 1.0), (0.1, 4.0)]

        for _ in range(n_starts):
            x0 = [rng.uniform(-6.0, -1.0), rng.uniform(0.2, 2.0)]
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    x0,
                    args=(outcomes, choices),
                    bounds=bounds,
                    method="L-BFGS-B",
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
            except Exception:
                continue

        if best_params is None:
            raise RuntimeError("Optimization failed from all starting points")

        n = len(outcomes)
        k = 2
        bic = k * np.log(n) + 2 * best_nll

        self.omega = best_params[0]
        self.kappa = best_params[1]
        return {
            "omega": best_params[0],
            "kappa": best_params[1],
            "nll": best_nll,
            "bic": bic,
        }
