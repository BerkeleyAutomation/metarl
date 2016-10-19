
from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from rlpy.Domains.Domain import Domain
import numpy as np
from RCCarDynamics import RCCarDynamics
from RCCarBarriers import RCCarBarriers
import logging 

LEFT_TURN = 0.03
RIGHT_TURN = -0.03
SLIDE_LEFT = -0.03
# HARD_SLIDE_LEFT = -0.03
SLIDE_RIGHT = 0.03

class RCCarBar_Bias(RCCarBarriers): # set problem structure

    ROOM_WIDTH = 6  # in meters
    ROOM_HEIGHT = 5  # in meters
    XMIN = -ROOM_WIDTH / 2.0
    XMAX = ROOM_WIDTH / 2.0
    YMIN = -ROOM_HEIGHT / 2.0
    YMAX = ROOM_HEIGHT / 2.0
    ACCELERATION = .1
    # GOAL_ANGLE = np.pi / 2
    INIT_STATE = np.array([-2., -2., 0, 0])
    GOAL = [-1., 1., 0. ]
    GOAL_RADIUS = .2
    episodeCap = 500
    STEP_REWARD = -0.01
    def __init__(self, wall_array=[(XMIN, -1., 3.5, 0.2)], **kwargs):
        super(RCCarBar_Bias, self).__init__(wall_array=wall_array, **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        raise NotImplementedError


    def _reward(self, ns, terminal):
        quad_reward = -(np.linalg.norm(ns[:2] - self.GOAL[:2]) / ((self.XMAX - self.XMIN)**2+(self.YMAX - self.YMIN) **2))

        r = self.GOAL_REWARD if terminal else quad_reward

        # Collision to wall => set reward to bad
        if self._bumped(ns) or ns[0] == self.XMIN or ns[0] == self.XMAX or ns[1] == self.YMIN or ns[1] == self.YMAX:
            r = -20
        return r

    def isTerminal(self):
        nx, ny, _, _ = self.state
        if nx == self.XMIN or nx == self.XMAX or ny == self.YMIN or ny == self.YMAX:
            return True

        return np.linalg.norm(self.state[:2] - self.GOAL[:2]) < self.GOAL_RADIUS \
                 and abs(self.state[2]) < 0.2

class RCCarBar_SlideLeft(RCCarBar_Bias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_slide(state, acc, turn, bias=SLIDE_LEFT)
    # set dynamics structure


class RCCarBar_LeftTurn(RCCarBar_Bias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_turn(state, acc, turn, bias=LEFT_TURN)


class RCCarBar_RightTurn(RCCarBar_Bias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_turn(state, acc, turn, bias=RIGHT_TURN)