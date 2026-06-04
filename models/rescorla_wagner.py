"""
Rescorla-Wagner Model for Computational Psychiatry.

Mathematical Model
------------------
The Rescorla-Wagner model describes associative learning through prediction
error-driven value updates:

    V(t+1) = V(t) + alpha * (R(t) - V(t))

where:
    - V(t) is the expected value at trial t
    - alpha is the learning rate (0 < alpha < 1)
    - R(t) is the reward received at trial t
    - (R(t) - V(t)) is the prediction error (PE)

Clinical Relevance
------------------
- **Depression**: Reduced learning rates (alpha) for positive outcomes,
  reflecting blunted reward sensitivity and anhedonia.
- **Addiction**: Elevated learning rates for drug-related cues, leading to
  over-valuation of substance-associated stimuli.
- **Anxiety**: Altered PE processing for threat-related stimuli.

References
----------
Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional


class RescorlaWagner:
    """Rescorla-Wagner associative learning model.

    Parameters
    ----------
    alpha : float
        Learning rate, must be in (0, 1). Controls how quickly the model
        updates its value estimates in response to prediction errors.

    Attributes
    ----------
    alpha : float
        The learning rate parameter.
    values : np.ndarray or None
        Learned value estimates after simulation or fitting.
    prediction_errors : np.ndarray or None
        Prediction errors from the last simulation or fitting.
    """

    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.values: Optional[np.ndarray] = None
        self.prediction_errors: Optional[np.ndarray] = None

    def _update(self, value: float, reward: float) -> Tuple[float, float]:
        """Perform a single RW update step.

        Parameters
        ----------
        value : float
            Current value estimate.
        reward : float
            Observed reward.

        Returns
        -------
        new_value : float
            Updated value estimate.
        pe : float
            Prediction error.
        """
        pe = reward - value
        new_value = value + self.alpha * pe
        return new_value, pe

    def simulate(
        self,
        n_trials: int = 200,
        reward_prob: float = 0.7,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate a binary reward task.

        The agent encounters a single stimulus on each trial and receives
        a binary reward with probability reward_prob. Choices are generated
        via a sigmoid function of the current value estimate.

        Parameters
        ----------
        n_trials : int
            Number of trials to simulate.
        reward_prob : float
            Probability of reward on each trial.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'rewards': array of rewards (0 or 1)
            - 'values': array of value estimates
            - 'prediction_errors': array of prediction errors
            - 'choices': array of choices (1 = choose stimulus, 0 = avoid)
        """
        if rng is None:
            rng = np.random.RandomState()

        rewards = np.zeros(n_trials)
        values = np.zeros(n_trials + 1)
        prediction_errors = np.zeros(n_trials)
        choices = np.zeros(n_trials, dtype=int)

        for t in range(n_trials):
            # Choice probability via sigmoid of value
            p_choose = 1.0 / (1.0 + np.exp(-5.0 * (values[t] - 0.5)))
            choices[t] = int(rng.random() < p_choose)

            # Reward only if stimulus is chosen
            if choices[t] == 1:
                rewards[t] = float(rng.random() < reward_prob)
            else:
                rewards[t] = 0.0

            values[t + 1], prediction_errors[t] = self._update(
                values[t], rewards[t]
            )

        self.values = values[:-1]
        self.prediction_errors = prediction_errors

        return {
            "rewards": rewards,
            "values": values[:-1],
            "prediction_errors": prediction_errors,
            "choices": choices,
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
            Model parameters [alpha].
        choices : np.ndarray
            Array of binary choices.
        rewards : np.ndarray
            Array of rewards received.

        Returns
        -------
        nll : float
            Negative log-likelihood.
        """
        alpha = params[0]

        # Boundary check
        if not 0 < alpha < 1:
            return 1e10

        n_trials = len(choices)
        value = 0.0
        nll = 0.0

        for t in range(n_trials):
            # Choice probability via sigmoid
            p_choose = 1.0 / (1.0 + np.exp(-5.0 * (value - 0.5)))
            p_choose = np.clip(p_choose, 1e-8, 1.0 - 1e-8)

            if choices[t] == 1:
                nll -= np.log(p_choose)
            else:
                nll -= np.log(1.0 - p_choose)

            # Update value
            pe = rewards[t] - value
            value = value + alpha * pe

        return nll

    def fit(
        self,
        choices: np.ndarray,
        rewards: np.ndarray,
        n_starts: int = 10,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """Fit model parameters to behavioral data.

        Uses maximum likelihood estimation with multiple random starting
        points to avoid local minima.

        Parameters
        ----------
        choices : np.ndarray
            Array of binary choices (0 or 1).
        rewards : np.ndarray
            Array of rewards received.
        n_starts : int
            Number of random starting points for optimization.
        rng : np.random.RandomState, optional
            Random state for reproducibility of starting points.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'alpha': fitted learning rate
            - 'nll': negative log-likelihood at optimum
            - 'bic': Bayesian Information Criterion
        """
        if rng is None:
            rng = np.random.RandomState()

        best_nll = np.inf
        best_params = None

        bounds = [(0.001, 0.999)]

        for _ in range(n_starts):
            x0 = rng.uniform(0.01, 0.99, size=1)
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
        k = 1  # number of parameters
        bic = k * np.log(n) + 2 * best_nll

        self.alpha = best_params[0]
        return {
            "alpha": best_params[0],
            "nll": best_nll,
            "bic": bic,
        }
