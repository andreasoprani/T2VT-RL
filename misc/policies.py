import numpy as np
from scipy import special


class Policy:
    """Base class for all policies"""

    def sample_action(self, s):
        """Samples from \pi(a|s)"""
        pass


class EpsilonGreedy(Policy):
    """An epsilon-greedy policy"""

    def __init__(self, Q, actions, epsilon=0):
        self._q = Q
        self._e = epsilon
        self._actions = actions

    def sample_action(self, s):
        q_a = self._q.value_actions(s)
        max_a = np.argmax(q_a)

        t = np.random.rand(1)
        if t < 1-self._e:
            return self._actions[max_a]
        else:
            return self._actions[np.random.randint(0, len(self._actions))]


class ScheduledEpsilonGreedy(Policy):
    """An epsilon-greedy policy with scheduled epsilon"""

    def __init__(self, Q, actions, schedule):
        self._q = Q
        self._schedule = schedule
        self._actions = actions
        self._h = 0
        self._H = schedule.shape[0]

    def sample_action(self, s):
        q_a = self._q.value_actions(s)
        max_a = np.argmax(q_a)

        t = np.random.rand(1)
        eps = self._schedule[self._h] if self._h < self._H else self._schedule[-1]
        self._h += 1
        if t < 1-eps:
            return self._actions[max_a]
        else:
            return self._actions[np.random.randint(0, len(self._actions))]

class Gibbs(Policy):
    '''A Boltzmann/Gibbs Policy'''
    def __init__(self, Q, actions, tau=1):
        self._q = Q
        self._tau = tau
        self._actions = actions

    def sample_action(self, s):
        q_a = self._q.value_actions(s).ravel()
        if self._tau == np.inf:
            return self._actions[np.argmax(q_a)]
        if self._tau == 0:
            return self._actions[np.random.randint(0, len(self._actions))]
        q_a = q_a*self._tau
        s_max = special.softmax(q_a)
        s_max = s_max/np.sum(s_max)  # for numerical stability
        return self._actions[np.argmax(np.random.multinomial(1, s_max, 1))]