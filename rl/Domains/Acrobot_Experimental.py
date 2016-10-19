"""classic Acrobot task"""
from rlpy.Tools import wrap, bound, lines, fromAtoB, rk4
from rlpy.Domains import Acrobot
import numpy as np
import matplotlib.pyplot as plt

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

class ModifiedAcrobot(Acrobot):
    episodeCap = 500

    def __init__(self, **kwargs):
        self.counter = 0
        super(ModifiedAcrobot, self).__init__()

    def step(self, a):
        s = self.state
        torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.random_state.uniform(-
                                                self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns.copy()
        terminal = self.isTerminal()
        reward = self._reward_function(terminal)
        return reward, ns, terminal, self.possibleActions()

    def _reward_function(self, terminal):
        return -1. if not terminal else 0.

    def showDomain(self, a=0):
        self.counter += 1
        if self.counter % 2:
            return
        super(ModifiedAcrobot, self).showDomain(a)
        plt.pause(0.001)


class Acrobot_Mass1(ModifiedAcrobot):
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 0.1  #: [kg] mass of link 1
    LINK_MASS_2 = 1.

class Acrobot_Mass2(ModifiedAcrobot):
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.5