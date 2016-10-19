from GridWorldModified import GridWorldModified
from GridWorldMixed import GridWorldMixed
from rlpy.Tools import plt, FONTSIZE, linearMap
import numpy as np
from rlpy.Domains.Domain import Domain
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os
import random
import time


class GridWorld2(GridWorldMixed):

    def __init__(self, **kwargs):
        super(GridWorld2, self).__init__( **kwargs) 

    def get_terrain(self, state):
        assert len(state) < 3, "state length is not 2"
        return int(state[0]) % 2

    def terrain_actions(self, state):
        terr = self.get_terrain(state)
        if terr == 0:
            return self.ACTIONS
        else:
            return np.append(self.ACTIONS[-1:], self.ACTIONS[:-1], axis=0)

class GridWorld3(GridWorldMixed):

    def __init__(self, **kwargs):
        super(GridWorld3, self).__init__( **kwargs) 

    def get_terrain(self, state):
        assert len(state) < 3, "state length is not 2"
        # return int(state[0] < 4) + int(state[1] < 6)
        return int(state[0]) % 2

    def terrain_actions(self, state):
        terr = self.get_terrain(state)
        if terr < 4:
            return np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)

class GridWorld4(GridWorldMixed):

    def __init__(self, **kwargs):
        super(GridWorld4, self).__init__( **kwargs) 

    def get_terrain(self, state):
        assert len(state) < 3, "state length is not 2"
        # return int(state[0] < 5)
        return int(state[0]) % 2
        # return int(state[1] < 4) + int(state[1] < 6) + int(state[1] < 9) 

    def terrain_actions(self, state):
        terr = self.get_terrain(state)
        if terr < 4:
            return np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        elif terr == 4:
            return np.array([[0, -2], [-1, 0], [+1, 0], [0, +2]])
        elif terr == 5:
            return np.array([[-2, 0], [0, -1], [0, +1], [+2, 0]])
        else:
            print "ERROR"


class GridWorld6(GridWorldMixed):

    def __init__(self, **kwargs):
        super(GridWorld6, self).__init__( **kwargs) 

    def get_terrain(self, state):
        assert len(state) < 3, "state length is not 2"
        # return int(state[0] < 5)
        return int(state[0]) % 3 + int(state[1]) % 3 
        # return int(state[1] < 4) + int(state[1] < 6) + int(state[1] < 9) 

    def terrain_actions(self, state):
        terr = self.get_terrain(state)
        if terr < 4:
            return np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        elif terr == 4:
            return np.array([[0, -2], [-1, 0], [+1, 0], [0, +2]])
        elif terr == 5:
            return np.array([[-2, 0], [0, -1], [0, +1], [+2, 0]])
        else:
            print "ERROR"
            import ipdb; ipdb.set_trace()  # breakpoint cd24c511 //
