from rlpy.Agents.Agent import Agent
# from MetaAgent import MetaAgent
from itertools import combinations
import numpy as np
import time

class NoopAgent(Agent):
    def __init__(self, policy, representation, discount_factor=0, **kwargs):
        if hasattr(policy, "subagents"):
            self.subagents = policy.subagents
        super(
            NoopAgent,
            self).__init__(policy=policy,
            representation=representation, discount_factor=discount_factor, **kwargs)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

        # print "Learning is called on X agents"
        # if terminal:
        return