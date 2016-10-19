"""Gridworld Domain."""
from GridWorldModified import GridWorldModified
from rlpy.Tools import plt, FONTSIZE, linearMap
import numpy as np
from rlpy.Domains.Domain import Domain
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os
import random

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

# _fig = plt.gcf()

class GridWorldReversed(GridWorldModified):
    # Modified for different starting points
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])

    def __init__(self, **kwargs):
        # self.ACTIONS = np.array([[+1, 0], [-1, 0], [0, -1], [0, +1]]) # down and up shifted
        terr = 1
        self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        super(GridWorldReversed, self).__init__(**kwargs)

class GridWorldReversed2(GridWorldModified):
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])

    def __init__(self, **kwargs):
        # self.ACTIONS = np.array([[0, -1], [0, +1], [+1, 0], [-1, 0]])
        terr = 2
        self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        super(GridWorldReversed2, self).__init__(**kwargs)

class GridWorldReversed3(GridWorldModified):
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])

    def __init__(self, **kwargs):
        # self.ACTIONS = np.array([[0, -1], [+1, 0], [0, +1],  [-1, 0]])
        terr = 3
        self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        super(GridWorldReversed3, self).__init__(**kwargs)

class GridWorldReversed4(GridWorldModified):
    ACTIONS = np.array([[-1, 0], [0, -1], [+1, 0], [0, +1]])

    def __init__(self, **kwargs):
        terr = 1
        self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        super(GridWorldReversed4, self).__init__(**kwargs)

class GridWorldReversed5(GridWorldModified):
    ACTIONS = np.array([[-1, 0], [0, -1], [+1, 0], [0, +1]])

    def __init__(self, **kwargs):
        terr = 2
        self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
        super(GridWorldReversed5, self).__init__(**kwargs)

### OTHER WAY 

class GridWorldSkip1(GridWorldModified):

    def __init__(self, **kwargs):
        self.ACTIONS = np.array([[0, -2], [-1, 0], [+1, 0], [0, +2]])
        super(GridWorldSkip1, self).__init__(**kwargs)

class GridWorldSkip2(GridWorldModified):

    def __init__(self, **kwargs):
        self.ACTIONS = np.array([[-2, 0], [0, -1], [0, +1], [+2, 0]])
        super(GridWorldSkip2, self).__init__(**kwargs)

# class GridWorldReversed2(GridWorldModified):
#     ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])

#     def __init__(self, **kwargs):
#         # self.ACTIONS = np.array([[0, -1], [0, +1], [+1, 0], [-1, 0]])
#         terr = 2
#         self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
#         super(GridWorldReversed2, self).__init__(**kwargs)

# class GridWorldReversed3(GridWorldModified):
#     ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])

#     def __init__(self, **kwargs):
#         # self.ACTIONS = np.array([[0, -1], [+1, 0], [0, +1],  [-1, 0]])
#         terr = 3
#         self.ACTIONS = np.append(self.ACTIONS[-terr:], self.ACTIONS[:-terr], axis=0)
#         super(GridWorldReversed3, self).__init__(**kwargs)

# class GridWorldSkip(GridWorldModified):

#     def __init__(self, **kwargs):
#         self.ACTIONS = np.array([[0, -2], [+2, 0], [0, +1],  [-1, 0]])
#         super(GridWorldReversed2, self).__init__(**kwargs)