"""Gibbs policy"""

from rlpy.Policies import GibbsPolicy
import numpy as np
import logging

class ModifiedGibbsPolicy(GibbsPolicy):

    """
    Gibbs policy for finite number of actions

    Warning: assumes that the features for each action are stacked, i.e.,
    a feature vector consists of |A| identical stacked vectors.
    """
    def __init__(self, representation, tau=1.):
        self.tau = tau
        self._explore = False
        super(ModifiedGibbsPolicy, self).__init__(representation)

    def turnOffExploration(self):
        self._explore = False
        logging.debug("Turning off exploration")

    def turnOnExploration(self):
        self._explore = True
        logging.debug("Turning on exploration")


    def probabilities(self, s, terminal):
        if not self._explore:
            q_values = self.representation.Qs(s, terminal)
            r = np.zeros(len(q_values))
            r[q_values.argmax()] = 1
            return r
        phi = self.representation.phi(s, terminal)
        n = self.representation.features_num
        v = np.exp(np.dot(self.representation.weight_vec.reshape(-1, n), phi) / self.tau)
        v[v > 1e50] = 1e50
        r = v / v.sum()
        assert not np.any(np.isnan(r))
        # print "Probabilities are: ", r
        return r
