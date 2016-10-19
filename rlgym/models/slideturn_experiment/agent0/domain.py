
from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from rlpy.Domains.Domain import Domain
import numpy as np
from RCCarDynamics import RCCarDynamics
from RCCarBarriers import RCCarBarriers
import logging 

LEFT_TURN = 0.03
RIGHT_TURN = -0.03
SLIDE_LEFT = -0.02
SLIDE_RIGHT = 0.03

class RCCarBias(RCCarDynamics):

    GOAL = [1., 0, 0. ]
    GOAL_RADIUS = .2
    GOAL_ANGLE = np.pi / 2
    INIT_STATE = np.array([-2., 0, 0, 0])
    episodeCap = 150

    def __init__(self, **kwargs):
        super(RCCarBias, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        raise NotImplementedError

    def _reward(self, ns, terminal):
        quad_reward = -(np.linalg.norm(ns[:3] - self.GOAL) / ((self.XMAX - self.XMIN)**2+(self.YMAX - self.YMIN) **2) + self.SPEEDMAX ** 2) 
        # r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        r = self.GOAL_REWARD if terminal else quad_reward

        # Collision to wall => set reward to bad
        if ns[0] == self.XMIN or ns[0] == self.XMAX or ns[1] == self.YMIN or ns[1] == self.YMAX:
            r = quad_reward
        return r

    def isTerminal(self):
        nx, ny, _, _ = self.state
        if nx == self.XMIN or nx == self.XMAX or ny == self.YMIN or ny == self.YMAX:
            return True

        return np.linalg.norm(self.state[:2] - self.GOAL[:2]) < self.GOAL_RADIUS and \
                abs(self.state[2]) < 0.2

class RCCarLeft(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_slide(state, acc, turn, bias=SLIDE_LEFT)


class RCCarRight(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_slide(state, acc, turn, bias=SLIDE_RIGHT)


class RCCarSlideTest(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        if np.linalg.norm(state[0:2] - self.GOAL) < self.GOAL_RADIUS * 5:
            return self._dynamics_slide(state, acc, turn, bias=0.03) # SLIDE RIGHT
        else:
           return self._dynamics_slide(state, acc, turn, bias=-0.03) # SLIDE LEFT


class RCCarLeftTurn(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_turn(state, acc, turn, bias=LEFT_TURN)


class RCCarRightTurn(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_turn(state, acc, turn, bias=RIGHT_TURN)


class RCCarTurnTest(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        if np.linalg.norm(state[0:2] - self.GOAL) < self.GOAL_RADIUS * 5:
            return self._dynamics_turn(state, acc, turn, bias=0.01) # SLIDE RIGHT
        else:
           return self._dynamics_turn(state, acc, turn, bias=-0.01) # SLIDE LEFT


class RCCarSlideTurn(RCCarBias):

    def _action_dynamics(self, state, acc, turn):
        if np.linalg.norm(state[0:2] - self.GOAL[:2]) < self.GOAL_RADIUS *  10:
            return self._dynamics_slide(state, acc, turn, bias=SLIDE_LEFT) # SLIDE LEFT
        else:
            return self._dynamics_turn(state, acc, turn, bias=RIGHT_TURN) # TURN RIGHT

class RCCar2(RCCarDynamics):

    def __init__(self, **kwargs):
        super(RCCar2, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        x, y, speed, heading = state
        if y > 0:
            return self._dynamics_regular(state, acc, turn)
        else:
        	return self._dynamics_regular(state, acc, -turn)

class RCCarInverted(RCCarDynamics):

    def __init__(self, **kwargs):
        super(RCCarInverted, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_regular(state, acc, -turn)

#### WITH BARRIERS


class RCCarBarriersInverted(RCCarBarriers):

    def __init__(self, **kwargs):
        super(RCCarBarriersInverted, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_regular(state, acc, -turn)


class RCCarBarrier_2(RCCarBarriers):

    def __init__(self, wall_array=[[-0.5, -2, 0.2, 2.5]], **kwargs):
        super(RCCarBarrier_2, self).__init__(wall_array=wall_array, **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        
        x, y, speed, heading = state
        if x > 0:
            return self._dynamics_regular(state, acc, turn)
        else:
            return self._dynamics_regular(state, acc, -turn)