from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box, Discrete
import rlpy.Domains.RCCar as RC
import numpy as np


class HRLEnv(Env):

    def __init__(self, domain, policies):
        #assert domain is rllab domain
        self.domain = domain
        self.domain.logger = None
        self.policies = policies

        # Serializable.quick_init(self, locals())
        self.state = None
        self.domain_fig = None

    def reset(self):
        state = self.domain.reset()
        self.state = state.copy()
        return np.array(self.state)


    def step(self, agent_number):
        assert agent_number < len(self.policies)
        agent = self.policies[agent_number]
        action, _ = agent.get_action(self.state)
        step_object = self.domain.step(action)
        # print reward
        self.state = step_object.observation.copy()
        return step_object

    @property
    def horizon(self):
        return self.domain.horizon
    

    @property
    def action_space(self):
        return Discrete(len(self.policies))

    @property
    def observation_space(self):
        return self.domain.observation_space

    def render(self):
        # print 'current state:', self.state
        self.domain.render()
        # pass