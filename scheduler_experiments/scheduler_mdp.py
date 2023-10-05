"""
grid world mdp class
"""
import numpy as np

# from gymnasium import Env, spaces
from numpy.random import randn


class SchedulerMDP:
    """
    Scheduler MDP where the exogenous states are auto-regressive
    processes and the reward is proportional to their sum.
    """

    def __init__(
        self,
        d: int = 2,
        phi: float = 0.9,
        scale: float = 0.1,
        budget: int = 5,
        horizon=100,
    ):
        super().__init__()
        self.d = d
        self.dims = d + 1  # considering the remaining actions
        self.scale = scale
        self.phi = phi
        self.budget = budget
        self.horizon = horizon
        self.action_space = [0, 1]
        self.min_state = -1
        self.max_state = 1
        self.reset()
        self._returns_buffer = []

    def reset(self) -> np.ndarray:
        self._states = [self.scale * randn(self.d)]
        for _ in range(self.horizon):
            next_state = self.phi * self._states[-1] + randn(self.d) * self.scale
            self._states.append(next_state)
        self._states = np.array(self._states)
        self._t = 0
        self._cum_reward = 0
        self._done = False
        self._remaining_actions = self.budget
        self._last_action = None
        self._potential_rewards = self._states.sum(axis=1)
        # get the indices of the top k=budget potential rewards
        self._top_k = np.argsort(self._potential_rewards)[-self.budget :]
        self._action_indicator = np.zeros(self.horizon + 1)
        self._action_indicator[self._top_k] = 1
        return self.state()

    def state(self):
        frac_remaining = self._remaining_actions / self.budget
        return np.array([*np.tanh(self._states[self._t]), frac_remaining])

    def step(self, action: np.ndarray) -> tuple:
        # update state as AR process
        if self._remaining_actions == 0:
            action = 0
        elif action == 1:
            self._remaining_actions -= 1
        self._last_action = action
        self._t += 1
        self.done()
        return self.state(), self.reward(), self._done, {}

    def done(self):
        self._done = self._t >= self.horizon
        if self._done:
            self._returns_buffer.append(self._cum_reward)

    def reward(self):
        pr = self._potential_rewards[self._t - 1]
        reward = 0 if self._last_action == 0 else pr
        self._cum_reward += reward
        return reward

    def sample_target_state(self):
        # sample random t
        t = np.random.randint(self.horizon)
        exo_states = np.tanh(self._states[t + 1])
        frac_remaining = 1 - sum(self._action_indicator[:t]) / self.budget
        return np.array([*exo_states, frac_remaining])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    env = SchedulerMDP()
    done = False
    states = []
    rewards = []
    while not done:
        a = int(np.random.rand() < 0.1)
        s, r, done, _ = env.step(a)
        states.append(s)
        rewards.append(r)
    states = np.array(states)
    rewards = np.array(rewards)

    plt.plot(states)
    plt.show()

    plt.plot(np.cumsum(rewards))
    plt.show()

    sys.exit(0)
