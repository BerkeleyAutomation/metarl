from rlpy.Agents import Q_Learning
from rlpy.Domains.Domain import Domain
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import combinations
from rlpy.Tools import plt
# import rllab
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class RLPyEnv(gym.Env):
    """For RLPy Domains"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, domain):
        low = domain.statespace_limits[:, 0]
        high = domain.statespace_limits[:, 1]

        self.action_space = spaces.Discrete(domain.actions_num)
        self.observation_space = spaces.Box(low, high)
        self.domain = domain
        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        pass
        # self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        reward, next_state, done, np_actions = self.domain.step(a)
        self.state = next_state.copy()
        return np.array(self.state), reward, done, {}

    def _reset(self):
        state, term, p_actions = self.domain.s0()
        self.state = state.copy()
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        pass

        # return self.viewer.render(return_rgb_array = mode=='rgb_array')