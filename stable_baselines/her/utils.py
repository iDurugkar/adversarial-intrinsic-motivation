from collections import OrderedDict

import numpy as np
from gym import spaces
# import pdb
# Important: gym mixes up ordered and unordered keys
# and the Dict space may return a different order of keys that the actual one
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())
        # Check that all spaces are of the same type
        # (current limitation of the wrapper)
        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        return np.concatenate([obs_dict[key] for key in KEY_ORDER])

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:self.obs_dim]),
            ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.convert_dict_to_obs(obs), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.convert_dict_to_obs(self.env.reset())

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

class SMiRLHERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env, eval=False):
        super(SMiRLHERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())
        self.eval = eval
        # Check that all spaces are of the same type
        # (current limitation of the wrapper)
        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.x_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.x_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"
        self.obs_dim = self.x_dim * 3 + 1
        self.default_std = 1e-4
        # self.t = 0
        self.mean = None
        self.std = None
        self.obs = []
        self.max_t = 50
        self.reward_weight = .0003

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            smirl_lows = [env.observation_space.spaces['observation'].low,
                          np.zeros_like(env.observation_space.spaces['observation'].low), [0.]]
            smirl_highs = [env.observation_space.spaces['observation'].high,
                           np.ones_like(env.observation_space.spaces['observation'].high) * np.inf, [1.]]
            lows = [space.low for space in self.spaces] + smirl_lows
            highs = [space.high for space in self.spaces] + smirl_highs
            lows = np.concatenate(lows)
            highs = np.concatenate(highs)
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

    def update_smirl(self, state):
        self.obs.append(state)
        self.mean = np.mean(self.obs, axis=0)
        if len(self.obs) > 1:
            self.std = np.std(self.obs, axis=0) + self.default_std
        else:
            self.std = np.ones_like(state) * self.default_std
        # np.mean(np.square(np.ndarray(self.obs) - [self.mean]), axis=0)
        # pdb.set_trace()
        return np.concatenate([state, self.mean, self.std, [len(self.obs) / self.max_t]])

    def reward(self, state):
        if self.eval:
            return 0.
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            return 0.
        log = np.log(self.std) + np.square(state[:self.x_dim] - self.mean) / (2. * np.square(self.std))
        rew = - np.sum(log)
        # pdb.set_trace()
        return rew

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        obses = [obs_dict[key] for key in KEY_ORDER]
        if obses[0].shape[0] == self.x_dim:
            obses[0] = self.update_smirl(obses[0])
        return np.concatenate(obses)

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:self.obs_dim]),
            ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.convert_dict_to_obs(obs)
        reward = reward + self.reward_weight * self.reward(obs)
        return obs, reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.convert_dict_to_obs(self.env.reset())

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
