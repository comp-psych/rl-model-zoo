"""
Dual Learning Rate Model for Computational Psychiatry.

Mathematical Model
------------------
The dual learning rate model extends the Rescorla-Wagner framework by using
separate learning rates for positive and negative prediction errors:

    If PE >= 0:  V(t+1) = V(t) + alpha_pos * PE(t)
    If PE < 0:   V(t+1) = V(t) + alpha_neg * PE(t)

    where PE(t) = R(t) - V(t)

Action selection uses softmax:
    P(choose) = exp(beta * V) / (exp(beta * V) + exp(0))
              = 1 / (1 + exp(-beta * V))

Parameters:
    - alpha_pos: learning rate for positive prediction errors (0, 1)
    - alpha_neg: learning rate for negative prediction errors (0, 1)
    - beta: inverse temperature (> 0)

Clinical Relevance
------------------
- **Depression**: Blunted alpha_pos (reduced learning from positive
  outcomes) paired with enhanced alpha_neg (heightened learning from
  negative outcomes), reflecting the negative bias in depression.
- **Mania**: Enhanced alpha_pos with reduced alpha_neg, reflecting
  excessive optimism and poor learning from negative feedback.
- **Addiction**: Elevated alpha_pos for drug cues, reflecting enhanced
  reward learning for substance-related stimuli.
- **Anxiety**: Enhanced alpha_neg reflecting heightened threat learning.

References
----------
Frank, M. J., et al. (2007). Genetic triple dissociation reveals multiple
roles for dopamine in reinforcement learning. PNAS, 104(41), 16311-16316.
Niv, Y., et al. (2012). Neural prediction errors reveal a risk-sensitive
reinforcement-learning process in the human brain.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional


class DualLearningRate:
    """Dual Learning Rate model with asymmetric PE processing.

    Parameters
    ----------
    alpha_pos : float
        Learning rate for positive prediction errors, in (0, 1).
    alpha_neg : float
        Learning rate for negative prediction errors, in (0, 1).
    beta : float
        Inverse temperature for softmax, must be > 0.

    Attributes
    ----------
    alpha_pos : float
        Positive PE learning rate.
    alpha_neg : float
        Negative PE learning rate.
    beta : float
        Inverse temperature.
    values : np.ndarray or None
        Learned values after simulation or fitting.
    """

    def __init__(
        self,
        alpha_pos: float = 0.6,
        alpha_neg: float = 0.3,
        beta: float = 5.0,
    ):
        if not 0 < alpha_pos < 1:
            raise ValueError(
                f"alpha_pos must be in (0, 1), got {alpha_pos}"
            )
        if not 0 < alpha_neg < 1:
            raise ValueError(
                f"alpha_neg must be in (0, 1), got {alpha_neg}"
            )
        if not beta > 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.beta = beta
        self.values: Optional[np.ndarray] = None

    def simulate(
        self,
        n_trials: int = 300,
        reward_prob: float = 0.7,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate a probabilistic reward task.

        On each trial, the agent chooses between two options. Option 1
        delivers reward with probability reward_prob. Option 0 delivers
        reward with probability 1 - reward_prob.

        Parameters
        ----------
        n_trials : int
            Number of trials.
        reward_prob : float
            Probability of reward for the better option.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'choices': array of binary choices
            - 'rewards': array of rewards
            - 'values': array of value estimates for each option
            - 'prediction_errors': array of prediction errors
        """
        if rng is None:
            rng = np.random.RandomState()

        # Two options with separate value tracking
        values = np.zeros((n_trials + 1, 2))
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)
        pes = np.zeros(n_trials)

        for t in range(n_trials):
            # Softmax choice between two options
            dv = self.beta * (values[t, 1] - values[t, 0])
            dv = np.clip(dv, -500, 500)
            p_choose_1 = 1.0 / (1.0 + np.exp(-dv))
            choices[t] = int(rng.random() < p_choose_1)

            # Generate reward
            if choices[t] == 1:
                rewards[t] = float(rng.random() < reward_prob)
            else:
                rewards[t] = float(rng.random() < (1.0 - reward_prob))

            # Prediction error
            pe = rewards[t] - values[t, choices[t]]
            pes[t] = pe

            # Dual learning rate update
            if pe >= 0:
                alpha = self.alpha_pos
            else:
                alpha = self.alpha_neg

            values[t + 1] = values[t].copy()
            values[t + 1, choices[t]] = values[t, choices[t]] + alpha * pe

        self.values = values[:-1]
        return {
            "choices": choices,
            "rewards": rewards,
            "values": values[:-1],
            "prediction_errors": pes,
        }

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        choices: np.ndarray,
        rewards: np.ndarray,
    ) -> float:
        """Compute negative log-likelihood.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [alpha_pos, alpha_neg, beta].
        choices : np.ndarray
            Array of binary choices.
        rewards : np.ndarray
            Array of rewards.

        Returns
        -------
        nll : float
            Negative log-likelihood.
        """
        alpha_pos, alpha_neg, beta = params

        if not (0 < alpha_pos < 1) or not (0 < alpha_neg < 1) or not (beta > 0):
            return 1e10

        values = np.zeros(2)
        nll = 0.0

        for t in range(len(choices)):
            # Softmax probability
            dv = beta * (values[1] - values[0])
            dv = np.clip(dv, -500, 500)
            p_choose_1 = 1.0 / (1.0 + np.exp(-dv))
            p_choose_1 = np.clip(p_choose_1, 1e-8, 1.0 - 1e-8)

            if choices[t] == 1:
                nll -= np.log(p_choose_1)
            else:
                nll -= np.log(1.0 - p_choose_1)

            # Update
            pe = rewards[t] - values[choices[t]]
            if pe >= 0:
                values[choices[t]] += alpha_pos * pe
            else:
                values[choices[t]] += alpha_neg * pe

        return nll

    def fit(
        self,
        choices: np.ndarray,
        rewards: np.ndarray,
        n_starts: int = 15,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """Fit model parameters to behavioral data.

        Parameters
        ----------
        choices : np.ndarray
            Array of binary choices.
        rewards : np.ndarray
            Array of rewards.
        n_starts : int
            Number of random starting points.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'alpha_pos': fitted positive PE learning rate
            - 'alpha_neg': fitted negative PE learning rate
            - 'beta': fitted inverse temperature
            - 'nll': negative log-likelihood
            - 'bic': Bayesian Information Criterion
        """
        if rng is None:
            rng = np.random.RandomState()

        best_nll = np.inf
        best_params = None

        bounds = [(0.001, 0.999), (0.001, 0.999), (0.1, 30.0)]

        for _ in range(n_starts):
            x0 = [
                rng.uniform(0.01, 0.99),
                rng.uniform(0.01, 0.99),
                rng.uniform(0.5, 15.0),
            ]
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    x0,
                    args=(choices, rewards),
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

        n = len(choices)
        k = 3  # alpha_pos, alpha_neg, beta
        bic = k * np.log(n) + 2 * best_nll

        self.alpha_pos = best_params[0]
        self.alpha_neg = best_params[1]
        self.beta = best_params[2]
        return {
            "alpha_pos": best_params[0],
            "alpha_neg": best_params[1],
            "beta": best_params[2],
            "nll": best_nll,
            "bic": bic,
        }
