"""
Q-Learning with Softmax Action Selection for Computational Psychiatry.

Mathematical Model
------------------
Q-learning updates action-value estimates using the Bellman equation:

    Q(s, a) = Q(s, a) + alpha * [R + gamma * max_a'(Q(s', a')) - Q(s, a)]

Action selection uses a softmax (Boltzmann) policy:

    P(a|s) = exp(beta * Q(s, a)) / sum_a'(exp(beta * Q(s, a')))

where:
    - Q(s, a) is the value of taking action a in state s
    - alpha is the learning rate (0 < alpha < 1)
    - gamma is the discount factor (0 < gamma <= 1)
    - beta is the inverse temperature (beta > 0); higher beta = more exploitative
    - R is the reward received

Clinical Relevance
------------------
- **Anxiety**: Elevated inverse temperature (beta) reflects excessive
  exploitation and reduced exploration, consistent with threat avoidance.
- **Schizophrenia**: Disrupted action-value computation leads to random
  or perseverative behavior (abnormal beta).
- **ADHD**: Altered explore-exploit tradeoffs with impulsive exploration.
- **OCD**: Excessive model-free habitual control overriding goal-directed behavior.

References
----------
Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279-292.
Daw, N. D., et al. (2006). Cortical substrates for exploratory decisions in humans.
Nature, 441(7095), 876-879.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional


class QLearning:
    """Q-Learning model with softmax action selection.

    Implements a two-armed bandit task with drifting reward probabilities,
    commonly used in computational psychiatry research.

    Parameters
    ----------
    alpha : float
        Learning rate, must be in (0, 1).
    gamma : float
        Discount factor, must be in (0, 1].
    beta : float
        Inverse temperature for softmax, must be > 0.

    Attributes
    ----------
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    beta : float
        Inverse temperature.
    q_values : np.ndarray or None
        Learned Q-values after simulation or fitting.
    """

    def __init__(
        self, alpha: float = 0.3, gamma: float = 0.95, beta: float = 5.0
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if not beta > 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.q_values: Optional[np.ndarray] = None

    @staticmethod
    def _softmax(q_values: np.ndarray, beta: float) -> np.ndarray:
        """Compute softmax action probabilities.

        Parameters
        ----------
        q_values : np.ndarray
            Q-values for each action.
        beta : float
            Inverse temperature.

        Returns
        -------
        probs : np.ndarray
            Action probabilities.
        """
        scaled = beta * q_values
        scaled = scaled - np.max(scaled)  # numerical stability
        exp_q = np.exp(scaled)
        return exp_q / np.sum(exp_q)

    @staticmethod
    def _generate_drifting_rewards(
        n_trials: int,
        n_actions: int = 2,
        drift_rate: float = 0.02,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Generate drifting reward probabilities via Gaussian random walk.

        Parameters
        ----------
        n_trials : int
            Number of trials.
        n_actions : int
            Number of actions/arms.
        drift_rate : float
            Standard deviation of the Gaussian random walk step.
        rng : np.random.RandomState, optional
            Random state.

        Returns
        -------
        reward_probs : np.ndarray
            Array of shape (n_trials, n_actions) with reward probabilities.
        """
        if rng is None:
            rng = np.random.RandomState()

        reward_probs = np.zeros((n_trials, n_actions))
        reward_probs[0] = rng.uniform(0.3, 0.7, size=n_actions)

        for t in range(1, n_trials):
            noise = rng.normal(0, drift_rate, size=n_actions)
            reward_probs[t] = np.clip(reward_probs[t - 1] + noise, 0.1, 0.9)

        return reward_probs

    def simulate(
        self,
        n_trials: int = 300,
        n_actions: int = 2,
        drift_rate: float = 0.02,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate a two-armed bandit with drifting rewards.

        Parameters
        ----------
        n_trials : int
            Number of trials.
        n_actions : int
            Number of available actions.
        drift_rate : float
            Drift rate for reward probability random walk.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'choices': array of chosen actions
            - 'rewards': array of received rewards
            - 'reward_probs': array of true reward probabilities
            - 'q_values': array of Q-value history
        """
        if rng is None:
            rng = np.random.RandomState()

        reward_probs = self._generate_drifting_rewards(
            n_trials, n_actions, drift_rate, rng
        )

        q_values = np.zeros(n_actions)
        q_history = np.zeros((n_trials, n_actions))
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)

        for t in range(n_trials):
            q_history[t] = q_values.copy()

            # Softmax action selection
            probs = self._softmax(q_values, self.beta)
            choices[t] = rng.choice(n_actions, p=probs)

            # Get reward
            rewards[t] = float(rng.random() < reward_probs[t, choices[t]])

            # Q-learning update (single-state bandit, so gamma * max Q(s') = 0
            # since we treat this as a stateless bandit, simplifying to:
            # Q(a) = Q(a) + alpha * (R - Q(a))
            # For the multi-state extension, gamma would matter)
            pe = rewards[t] + self.gamma * 0.0 - q_values[choices[t]]
            q_values[choices[t]] += self.alpha * pe

        self.q_values = q_values
        return {
            "choices": choices,
            "rewards": rewards,
            "reward_probs": reward_probs,
            "q_values": q_history,
        }

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        choices: np.ndarray,
        rewards: np.ndarray,
        n_actions: int,
    ) -> float:
        """Compute negative log-likelihood.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [alpha, beta].
        choices : np.ndarray
            Array of chosen actions.
        rewards : np.ndarray
            Array of received rewards.
        n_actions : int
            Number of available actions.

        Returns
        -------
        nll : float
            Negative log-likelihood.
        """
        alpha, beta = params

        if not (0 < alpha < 1) or not (beta > 0):
            return 1e10

        q_values = np.zeros(n_actions)
        nll = 0.0

        for t in range(len(choices)):
            # Softmax probabilities
            scaled = beta * q_values
            scaled = scaled - np.max(scaled)
            exp_q = np.exp(scaled)
            probs = exp_q / np.sum(exp_q)
            probs = np.clip(probs, 1e-8, 1.0)

            nll -= np.log(probs[choices[t]])

            # Q update
            pe = rewards[t] - q_values[choices[t]]
            q_values[choices[t]] += alpha * pe

        return nll

    def fit(
        self,
        choices: np.ndarray,
        rewards: np.ndarray,
        n_actions: int = 2,
        n_starts: int = 10,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """Fit model parameters to behavioral data.

        Parameters
        ----------
        choices : np.ndarray
            Array of chosen actions.
        rewards : np.ndarray
            Array of received rewards.
        n_actions : int
            Number of available actions.
        n_starts : int
            Number of random starting points.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'alpha': fitted learning rate
            - 'beta': fitted inverse temperature
            - 'nll': negative log-likelihood
            - 'bic': Bayesian Information Criterion
        """
        if rng is None:
            rng = np.random.RandomState()

        best_nll = np.inf
        best_params = None

        bounds = [(0.001, 0.999), (0.1, 30.0)]

        for _ in range(n_starts):
            x0 = [rng.uniform(0.01, 0.99), rng.uniform(0.5, 15.0)]
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    x0,
                    args=(choices, rewards, n_actions),
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
        k = 2  # alpha and beta (gamma not separately identifiable in bandit)
        bic = k * np.log(n) + 2 * best_nll

        self.alpha = best_params[0]
        self.beta = best_params[1]
        return {
            "alpha": best_params[0],
            "beta": best_params[1],
            "nll": best_nll,
            "bic": bic,
        }
