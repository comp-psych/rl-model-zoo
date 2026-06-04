"""
Temporal Difference (TD) Learning Model for Computational Psychiatry.

Mathematical Model
------------------
TD learning updates state value estimates using the temporal difference error:

    V(s_t) = V(s_t) + alpha * [R(t) + gamma * V(s_{t+1}) - V(s_t)]

where:
    - V(s_t) is the value of state s at time t
    - alpha is the learning rate (0 < alpha < 1)
    - gamma is the discount factor (0 < gamma <= 1)
    - R(t) is the reward at time t
    - [R(t) + gamma * V(s_{t+1}) - V(s_t)] is the TD error (delta)

Clinical Relevance
------------------
- **Addiction**: Dopaminergic prediction error signals map onto TD errors.
  Substance use disorders show aberrant temporal credit assignment.
- **Depression**: Impaired prospective valuation (reduced gamma) leads to
  temporal myopia and difficulty sustaining goal-directed behavior.
- **Schizophrenia**: Disrupted dopamine signaling alters TD error computation.

References
----------
Sutton, R. S. (1988). Learning to predict by the methods of temporal differences.
Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of
prediction and reward. Science, 275(5306), 1593-1599.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple


class TDLearning:
    """Temporal Difference learning model with multi-step sequential tasks.

    Parameters
    ----------
    alpha : float
        Learning rate, must be in (0, 1).
    gamma : float
        Discount factor, must be in (0, 1].

    Attributes
    ----------
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    state_values : np.ndarray or None
        Learned state values after simulation or fitting.
    """

    def __init__(self, alpha: float = 0.3, gamma: float = 0.9):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.alpha = alpha
        self.gamma = gamma
        self.state_values: Optional[np.ndarray] = None

    def simulate(
        self,
        n_trials: int = 200,
        n_states: int = 5,
        reward_states: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate a multi-step sequential task.

        The agent traverses a chain of states from state 0 to state n_states-1.
        At each state, there is a probability of transitioning forward or
        staying. Rewards are delivered at specified terminal states.

        Parameters
        ----------
        n_trials : int
            Number of episodes to simulate.
        n_states : int
            Number of states in the chain.
        reward_states : dict, optional
            Mapping of state index to reward magnitude. Defaults to
            reward of 1.0 at the final state.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'states': list of arrays, state sequences per episode
            - 'rewards': list of arrays, reward sequences per episode
            - 'td_errors': list of arrays, TD errors per episode
            - 'value_history': array of shape (n_trials, n_states),
              state values after each episode
            - 'choices': list of arrays, action choices per episode
        """
        if rng is None:
            rng = np.random.RandomState()

        if reward_states is None:
            reward_states = {n_states - 1: 1.0}

        values = np.zeros(n_states)
        value_history = np.zeros((n_trials, n_states))
        all_states = []
        all_rewards = []
        all_td_errors = []
        all_choices = []

        for ep in range(n_trials):
            states = [0]
            rewards_ep = []
            td_errors_ep = []
            choices_ep = []
            s = 0

            while s < n_states - 1:
                # Action: move forward (1) or stay (0)
                # Softmax-like choice based on value difference
                v_forward = values[min(s + 1, n_states - 1)] if s + 1 < n_states else 0
                v_stay = values[s]
                p_forward = 1.0 / (1.0 + np.exp(-2.0 * (self.gamma * v_forward - v_stay)))
                action = int(rng.random() < p_forward)
                choices_ep.append(action)

                if action == 1:
                    s_next = s + 1
                else:
                    # Staying has small chance of forced transition
                    s_next = s + 1 if rng.random() < 0.1 else s

                r = reward_states.get(s_next, 0.0)

                # TD update
                if s_next >= n_states - 1:
                    td_error = r - values[s]
                else:
                    td_error = r + self.gamma * values[s_next] - values[s]

                values[s] = values[s] + self.alpha * td_error

                states.append(s_next)
                rewards_ep.append(r)
                td_errors_ep.append(td_error)
                s = s_next

            all_states.append(np.array(states))
            all_rewards.append(np.array(rewards_ep))
            all_td_errors.append(np.array(td_errors_ep))
            all_choices.append(np.array(choices_ep))
            value_history[ep] = values.copy()

        self.state_values = values
        return {
            "states": all_states,
            "rewards": all_rewards,
            "td_errors": all_td_errors,
            "value_history": value_history,
            "choices": all_choices,
        }

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        all_states: list,
        all_rewards: list,
        all_choices: list,
        n_states: int,
    ) -> float:
        """Compute negative log-likelihood for TD model.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [alpha, gamma].
        all_states : list
            List of state sequences per episode.
        all_rewards : list
            List of reward sequences per episode.
        all_choices : list
            List of choice sequences per episode.
        n_states : int
            Number of states.

        Returns
        -------
        nll : float
            Negative log-likelihood.
        """
        alpha, gamma = params

        if not (0 < alpha < 1) or not (0 < gamma <= 1):
            return 1e10

        values = np.zeros(n_states)
        nll = 0.0

        for ep_idx in range(len(all_states)):
            states = all_states[ep_idx]
            rewards_ep = all_rewards[ep_idx]
            choices_ep = all_choices[ep_idx]

            for step in range(len(choices_ep)):
                s = states[step]
                s_next = states[step + 1]
                r = rewards_ep[step]
                action = choices_ep[step]

                # Compute choice probability
                v_forward = values[min(s + 1, n_states - 1)] if s + 1 < n_states else 0
                v_stay = values[s]
                p_forward = 1.0 / (1.0 + np.exp(-2.0 * (gamma * v_forward - v_stay)))
                p_forward = np.clip(p_forward, 1e-8, 1.0 - 1e-8)

                if action == 1:
                    nll -= np.log(p_forward)
                else:
                    nll -= np.log(1.0 - p_forward)

                # TD update
                if s_next >= n_states - 1:
                    td_error = r - values[s]
                else:
                    td_error = r + gamma * values[s_next] - values[s]
                values[s] = values[s] + alpha * td_error

        return nll

    def fit(
        self,
        all_states: list,
        all_rewards: list,
        all_choices: list,
        n_states: int = 5,
        n_starts: int = 10,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """Fit model parameters to behavioral data.

        Parameters
        ----------
        all_states : list
            List of state sequences per episode.
        all_rewards : list
            List of reward sequences per episode.
        all_choices : list
            List of choice sequences per episode.
        n_states : int
            Number of states in the task.
        n_starts : int
            Number of random starting points.
        rng : np.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'alpha': fitted learning rate
            - 'gamma': fitted discount factor
            - 'nll': negative log-likelihood
            - 'bic': Bayesian Information Criterion
        """
        if rng is None:
            rng = np.random.RandomState()

        best_nll = np.inf
        best_params = None

        bounds = [(0.001, 0.999), (0.001, 0.999)]

        for _ in range(n_starts):
            x0 = rng.uniform(0.01, 0.99, size=2)
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    x0,
                    args=(all_states, all_rewards, all_choices, n_states),
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

        # Count total number of choice observations
        n_obs = sum(len(c) for c in all_choices)
        k = 2  # number of parameters
        bic = k * np.log(n_obs) + 2 * best_nll

        self.alpha = best_params[0]
        self.gamma = best_params[1]
        return {
            "alpha": best_params[0],
            "gamma": best_params[1],
            "nll": best_nll,
            "bic": bic,
        }
