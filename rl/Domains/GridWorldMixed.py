"""Gridworld Domain."""
from GridWorldModified import GridWorldModified
from rlpy.Tools import plt, FONTSIZE, linearMap
import numpy as np
from rlpy.Domains.Domain import Domain
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os
import random
import time

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class GridWorldMixed(GridWorldModified):
    restarted = False

    def __init__(self, terrain_augmentation=True, **kwargs):
        #Instantiation of terrain statespace details are done in GridWorldModified because
        # GridWorldmap is instantiated there.
        self.terrain_augmentation = terrain_augmentation

        self.terrain_count = 4
        print "We are using a mixed domain with %d terrains" % self.terrain_count
        time.sleep(1)
        super(GridWorldMixed, self).__init__(terrain_augmentation=terrain_augmentation, **kwargs) # terrain is passed in to augment statespace details
        if terrain_augmentation:
            assert len(self.statespace_limits) == 3
        else:
            print "[WARNING] You are hiding the terrain information."


    def step(self, a):
        self.restarted = False
        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())
        # Take action
        state = self.state.copy()
        ns = self._dynamics(state, a)
        self.state = ns.copy()


        # Compute the reward
        r = self.STEP_REWARD
        if self.map[ns[0], ns[1]] == self.GOAL:
            r = self.GOAL_REWARD
        if self.map[ns[0], ns[1]] == self.PIT:
            r = self.PIT_REWARD

        terminal = self.isTerminal()
        return r, self.terrain_state(ns), terminal, self.possibleActions()

    def _dynamics(self, state, a):
        assert len(state) < 3, "in taking next step, state length not right"
        actions = self.terrain_actions(state)
        ns = self.state + actions[a]
        # Check bounds on state values
        if (ns[0] < 0 or ns[0] >= self.ROWS or
                ns[1] < 0 or ns[1] >= self.COLS or
                self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = state
        return ns

    def get_terrain(self, state):
        assert len(state) < 3, "state length is not 2"
        # return int(state[0] < 5)
        return int(state[0]) % 4
        # return int(state[1] < 4) + int(state[1] < 6) + int(state[1] < 9) 

    def terrain_actions(self, state):
        terr = self.get_terrain(state)
        
        return np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        # if terr == 0:
        #     return self.ACTIONS
        # elif terr == 1:
        #     return np.array([[+1, 0], [-1, 0], [0, -1], [0, +1]]) #up down switched
        # elif terr == 2: 
        #     return np.array([[0, -1], [0, +1], [+1, 0], [-1, 0]]) # more switching
        # elif terr == 3: 
        #     return np.array([[0, -1], [+1, 0], [0, +1],  [-1, 0]]) # more switching

    def s0(self):
        self.state = self.start_state.copy()
        self.restarted = True
        if self.random_start:
            choices = np.argwhere(self.map == self.EMPTY)
            self.state = random.choice(choices)
            # print "Starting from this state: {}".format(self.state)
            
        return self.terrain_state(self.state), self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        if self.map[s[0], s[1]] == self.GOAL:
            return True
        if self.map[s[0], s[1]] == self.PIT:
            return True
        return False
    

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)

        actions = self.terrain_actions(s)

        for a in xrange(self.actions_num):
            ns = self.state + actions[a]
            if (
                    ns[0] < 0 or ns[0] >= self.ROWS or
                    ns[1] < 0 or ns[1] >= self.COLS or
                    self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA


    def terrain_state(self, state):
        # method to augment state representation
        if self.terrain_augmentation:
            return np.append(state, self.get_terrain(state)) 
        else:
            return state
