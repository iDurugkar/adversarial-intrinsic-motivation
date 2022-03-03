"""
grid world mdp class
"""
import numpy as np


class GridMDP:
    """
    Grid world MDP
    """
    def __init__(self):
        self.x_dim = 10
        self.y_dim = 10
        self.num_states = self.x_dim * self.y_dim
        self.start_state = np.asarray([0, 0])
        self._state = self.start_state
        self.action_space = [0, 1, 2, 3]
        self.target_state = [np.asarray([9, 9])]
        self.time_limit = 15
        self.t = 0
        self._done = False
        self._rewards = np.zeros(shape=(self.x_dim, self.y_dim), dtype=np.float)
        for t in self.target_state:
            self._rewards[t[0], t[1]] = 1.
        self.dims = 2
        self.min_state = np.asarray([0, 0])
        self.max_state = np.asarray([9, 9])

    def target_distribution(self, weight=3.) -> np.ndarray:
        """
        Return the target distribution
        :return:
        """
        # dist = np.exp(self._rewards * weight)
        dist = (self._rewards) * weight
        dist /= np.sum(dist)
        return dist

    def reset(self) -> np.ndarray:
        """
        reset state to 0
        :return:
        """
        self._state = np.copy(self.start_state)
        self.t = 0
        self.done()
        return self._state

    def step(self, act):
        """
        take action and return next state
        :param act:
        :return:
        """
        act = int(act)
        if self.t >= self.time_limit:
            print("Time limit exceeded. Reset for new episode")
            raise Exception()

        if self._done:
            print("Episode ended. Reset for new episode")
            raise Exception()

        assert act in self.action_space

        if np.random.random() < 0.:
            return self._state, self.reward(), self._done

        if act < 2:
            self._state[0] += act * 2 - 1
            self._state[0] = np.clip(self._state[0], 0, self.x_dim - 1)
        else:
            act = (act - 2) * 2 - 1
            self._state[1] += act
            self._state[1] = np.clip(self._state[1], 0, self.y_dim - 1)
        self.t += 1
        self.done()
        return self._state, self.reward(), self._done, {}

    def done(self):
        """
        check if episode over
        :return:
        """
        if self.t >= self.time_limit:
            self._done = True
        elif any(np.array_equal(self._state, x) for x in self.target_state):
            self._done = True
        else:
            self._done = False
        return

    def reward(self):
        """
        Reward function
        :return:
        """
        return self._rewards[self._state[0], self._state[1]]
        # if self._state == self.target_state:
        #     return 1.
        # return 0.


class MazeWorld(GridMDP):
    """
    Maze World is the same grid world as above
    But it has multiple goals, one of which is unreachable
    """
    def __init__(self):
        super(MazeWorld, self).__init__()
        self.start_state = np.asarray([1, 1])
        self._state = self.start_state
        self.target_state = [np.asarray([1, 4])]  # , np.asarray([8, 1]), np.asarray([6, 6]), np.asarray([4, 9])]
        self.time_limit = 100
        self._rewards = np.zeros(shape=(self.x_dim, self.y_dim), dtype=np.float)
        for c, t in enumerate(self.target_state):
            self._rewards[t[0], t[1]] = 1. / (c + 1.)

    def sample_transitions(self, num=50):
        """
        sample valid transitions from the mdp
        """
        transitions = []
        temp_state = self._state
        temp_time = self.t
        temp_done = self._done
        self._done = False
        self.t = 0
        while len(transitions) < num:
            state = np.random.randint(self.min_state, self.max_state)
            act = np.random.choice(self.action_space)
            self._state = np.copy(state)
            next_state, _, _, _ = self.step(act)
            if not np.equal(state, next_state).all():
                transitions.append([state, next_state])
            self.t = 0
            self._done = False
        self._state = temp_state
        self._done = temp_done
        self.t = temp_time
        states, next_states = (np.asarray(x) for x in zip(*transitions))
        return states, next_states

    def step(self, act):
        """
        take action and return next state
        Maze is as follows:
        -  -  -  -  -  -  -  -  -  -

        -  -  -  -  -  -  -  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  1  -  -| -  -  -  -  -  -
        __________|
        -  -  -  -  -  -  -  -  -  -

        -  A  -  -  -  -  -  -  -  -

        -  -  -  -  -  -  -  -  -  -

        :param act:
        :return:
        """
        act = int(act)
        if self.t >= self.time_limit:
            print("Time limit exceeded. Reset for new episode")
            raise Exception()

        if self._done:
            print("Episode ended. Reset for new episode")
            raise Exception()

        assert act in self.action_space

        if np.random.random() < 0.:
            return self._state, self.reward(), self._done

        if act == 0:
            if not (self._state[0] == 4 and (2 < self._state[1] < 8)):
                self._state[0] -= 1
            self._state[0] = np.clip(self._state[0], 0, self.x_dim - 1)
        elif act == 1:
            if not (self._state[0] == 3 and (2 < self._state[1] < 8)):
                self._state[0] += 1
            self._state[0] = np.clip(self._state[0], 0, self.x_dim - 1)
        elif act == 2:
            if not (self._state[1] == 3 and self._state[0] < 4):
                self._state[1] -= 1
            self._state[1] = np.clip(self._state[1], 0, self.y_dim - 1)
        else:
            if not (self._state[1] == 2 and self._state[0] < 4):
                self._state[1] += 1
            self._state[1] = np.clip(self._state[1], 0, self.y_dim - 1)
        self.t += 1
        self.done()
        return self._state, self.reward(), self._done, np.copy(self.target_state[0])


class ToroidWorld(GridMDP):
    """
    Toroidal grid world is same as grid world
    Except the grid bends in on itself
    so walking off one side causes transitions to the other side
    """
    def __init__(self):
        super(ToroidWorld, self).__init__()
        self.start_state = np.asarray([2, 2])
        self._state = self.start_state
        self.target_state = [np.asarray([self.x_dim - 3, self.y_dim - 3])]  # , np.asarray([8, 1]), np.asarray([6, 6]), np.asarray([4, 9])]
        self.time_limit = 40
        self._rewards = np.zeros(shape=(self.x_dim, self.y_dim), dtype=np.float)
        for c, t in enumerate(self.target_state):
            self._rewards[t[0], t[1]] = 1. / (c + 1.)

    def step(self, act):
        """
        take action and return next state
        :param act:
        :return:
        """
        act = int(act)
        if self.t >= self.time_limit:
            print("Time limit exceeded. Reset for new episode")
            raise Exception()

        if self._done:
            print("Episode ended. Reset for new episode")
            raise Exception()

        assert act in self.action_space

        if np.random.random() < 0.:
            return self._state, self.reward(), self._done

        if act < 2:
            self._state[0] += act * 2 - 1
            self._state[0] %= self.x_dim
        else:
            act = (act - 2) * 2 - 1
            self._state[1] += act
            self._state[1] %= self.y_dim
        self.t += 1
        self.done()
        return self._state, self.reward(), self._done, {}

class WindyMazeWorld(MazeWorld):
    """
    Windy maze world is same as Maze world
    Except in part of the world there is a wind
    that causes stochastic transitions for certain actions
    """
    def __init__(self):
        super(WindyMazeWorld, self).__init__()

    def step(self, act):
        """
        take action and return next state
        Maze is as follows:
        -  -  -  -  -  -  -  -  -  -

        -  -  -  -  -  -  -  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  -  -  -| -  -  1  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  -  -  -| -  -  -  -  -  -
                  |
        -  1  -  -| -  -  -  -  -  -
        __________|
        -  -  -  -  -  -  -  -  -  -

        -  A  -  -  -  -  -  -  1  -

        -  -  -  -  -  -  -  -  -  -

        :param act:
        :return:
        """
        act = int(act)
        if self.t >= self.time_limit:
            print("Time limit exceeded. Reset for new episode")
            raise Exception()

        if self._done:
            print("Episode ended. Reset for new episode")
            raise Exception()

        assert act in self.action_space

        if np.random.random() < 0.:
            return self._state, self.reward(), self._done

        if act == 0:
            if not (self._state[0] == 4 and (2 < self._state[1] < 8)):
                self._state[0] -= 1
            # Move down due to wind
            if 4 <= self._state[0] and np.random.rand() < 0.4:
                self._state[1] = np.minimum(self._state[1] - 1, 0)
            self._state[0] = np.clip(self._state[0], 0, self.x_dim - 1)
        elif act == 1:
            if not (self._state[0] == 3 and (2 < self._state[1] < 8)):
                self._state[0] += 1
            # Move down due to wind
            if 4 <= self._state[0] < self.x_dim and np.random.rand() < 0.4:
                self._state[1] = np.minimum(self._state[1] - 1, 0)
            self._state[0] = np.clip(self._state[0], 0, self.x_dim - 1)
        elif act == 2:
            if not (self._state[1] == 3 and self._state[0] < 4):
                self._state[1] -= 1
            self._state[1] = np.clip(self._state[1], 0, self.y_dim - 1)
        else:
            if not (self._state[1] == 2 and self._state[0] < 4):
                self._state[1] += 1
            # Move down due to wind
            if 4 <= self._state[0] < self.x_dim and np.random.rand() < 0.4:
                self._state[1] -= 1
            self._state[1] = np.clip(self._state[1], 0, self.y_dim - 1)
        self.t += 1
        self.done()
        return self._state, self.reward(), self._done, np.copy(self.target_state[0])
