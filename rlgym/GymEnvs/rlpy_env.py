from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box, Discrete
import rlpy.Domains.RCCar as RC
import numpy as np


class RLPyEnv(Env, Serializable):

    def __init__(self, domain=None):
        if domain is None:
            domain = RC()

        self.domain = domain
        self.domain.logger = None

        self._low = domain.statespace_limits[:, 0]
        self._high = domain.statespace_limits[:, 1]
        Serializable.quick_init(self, locals())
        self.state = None
        self.domain_fig = None

    def reset(self):
        state, term, p_actions = self.domain.s0()
        self.state = state.copy()
        return np.array(self.state)


    def step(self, action):
        reward, next_state, done, np_actions = self.domain.step(action)
        # print reward
        self.state = next_state.copy()
        return Step(observation=self.state, reward=reward, done=done)

    @property
    def horizon(self):
        return self.domain.episodeCap
    

    @property
    def action_space(self):
        return Discrete(self.domain.actions_num)

    @property
    def observation_space(self):
        return Box(self._low, self._high)

    def render(self):
        # print 'current state:', self.state
        self.domain.showDomain(0)
        # pass