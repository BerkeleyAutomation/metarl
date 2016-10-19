
from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from rlpy.Domains.Domain import Domain
import numpy as np
from RCCarDynamics import RCCarDynamics
from RCCarBarriers import RCCarBarriers
import logging 


class RCCarLeft(RCCarDynamics):

    def __init__(self, **kwargs):
        super(RCCarLeft, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
    	return self._dynamics_slide_left(state, acc, turn)

class RCCarRight(RCCarDynamics):

    def __init__(self, **kwargs):
        super(RCCarRight, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
    	return self._dynamics_slide_right(state, acc, turn)

class RCCarTurnLeft(RCCarDynamics):

    def __init__(self, **kwargs):
        super(RCCarTurnLeft, self).__init__( **kwargs) 

    def _action_dynamics(self, state, acc, turn):
        return self._dynamics_turn_left(state, acc, turn)


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