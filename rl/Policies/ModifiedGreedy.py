"""epsilon-Greedy Policy"""
import os
from rlpy.Policies.Policy import Policy
import numpy as np
import pickle
from collections import defaultdict
from copy import deepcopy
import scipy.stats as stats
import numpy as np
import joblib 

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class ModifiedGreedy(Policy):
    """ Greedy policy with epsilon-probability for uniformly random exploration.

    Modifications:
    - History storage and loading
    - Biased action implementation

    From a given state, it selects the action with the highest expected value
    (greedy with respect to value function), but with some probability 
    ``epsilon``, takes a random action instead.
    This explicitly balances the exploration/exploitation tradeoff, and 
    ensures that in the limit of infinite samples, the agent will
    have explored the entire domain.

    """
    # Probability of selecting a random action instead of greedy\

    DISTRIBUTION_FILE = "state_dist.p"
    epsilon         = None
    # Temporarily stores value of ``epsilon`` when exploration disabled
    old_epsilon     = None
    # This boolean variable is used to avoid random selection among actions
    # with the same values
    forcedDeterministicAmongBestActions = None
    historyfile = "history.p"
    last_run = None

    def __init__(self, representation, epsilon = .1, biasedaction=None,
                 forcedDeterministicAmongBestActions = False, seed=1):
        # self.history = defaultdict(int)
        # self.history_on = True

        self.transition_history = []

        self.biasedaction = biasedaction
        self.epsilon = epsilon
        self._explore = True
        self.seed = seed
        self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
        self._s = self._a = self._ns = None
        super(ModifiedGreedy, self).__init__(representation,seed)


    def pi(self, s, terminal, p_actions):
        # if self._explore and self.history_on:
        #     self.increase_confidence(s)
        coin = self.random_state.rand()
        if coin < self.epsilon:
            # print "chose random action"
            # if self.biasedaction in p_actions:
            #     return self.biasedaction
            return self.random_state.choice(p_actions)
        else:
            b_actions = self.representation.bestActions(s, terminal, p_actions)
            if self.forcedDeterministicAmongBestActions:
                return b_actions[0]
            else:
                return self.random_state.choice(b_actions)

    # def increase_confidence(self, s):
    #     f_num = self.hash_state(s)
    #     self.history[f_num] += 1


    def prob(self, s, terminal, p_actions):
        p = np.ones(len(p_actions)) / len(p_actions)
        p *= self.epsilon
        b_actions = self.representation.bestActions(s, terminal, p_actions)
        if self.forcedDeterministicAmongBestActions:
            p[b_actions[0]] += (1 - self.epsilon)
        else:
            p[b_actions] += (1 - self.epsilon) / len(b_actions)
        return p

    def turnOffExploration(self):
        self._explore = False
        self.old_epsilon = self.epsilon
        self.epsilon = 0

    def turnOnExploration(self):
        self._explore = True
        self.epsilon = self.old_epsilon

    def save_history(self, file_dir):
        with open(os.path.join(file_dir, self.historyfile), "w") as f:
            pickle.dump(self.transition_history, f)
        print "[Log] Saved History for {}".format(file_dir)

    def load_history(self, file_dir):
        with open(os.path.join(file_dir, self.historyfile), "r") as f:
            self.transition_history = pickle.load(f)
        print "[INFO] Loaded History for {}".format(file_dir)

    # def generate_transition_estimator(self):
    #     assert hasattr(self, "transition_history") and self.transition_history is not None
    #     reshape(self.transition_history)
    #     kde = stats.gaussian_kde(self.transition_history)
    #     self.transition_estimator = kde
        # kde.set_bandwidth(bw_method=1.43 * 4)

    # def hash_state(self, s):
    #     return self.representation.hashState(s)


class ConfidenceGreedy(ModifiedGreedy):
    """Implements confidence testing"""

    def __init__(self, representation, dim=None, **args):
        self._estimator = None
        self.dims_estimated = dim
        self.performance_domain = None
        super(ConfidenceGreedy, self).__init__(representation, **args)

    def get_confidence(self, s):
        return self._estimator.evaluate(s[:self.dims_estimated])

    def performanceRun(self, visualize=False):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        """

        # Set Exploration to zero and sample one episode from the domain
        assert self._estimator is None # we want to generate state distribution without confidence)

        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.turnOffExploration()
        if self.performance_domain is None:
            self.performance_domain = deepcopy(self.representation.domain)

        performance_domain = self.performance_domain
        s, eps_term, p_actions = performance_domain.s0()
        trajectory = s
        # print sum(agent.representation.weight_vec)

        while not eps_term and eps_length < performance_domain.episodeCap:
            a = self.pi(s, eps_term, p_actions)
            if visualize:
                performance_domain.showDomain(a)

            r, ns, eps_term, p_actions = performance_domain.step(a)
            # _gather_transition_statistics(s, a, ns, r, learning=False)
            s = ns
            eps_return += r
            eps_discount_return += performance_domain.discount_factor ** eps_length * \
                r
            eps_length += 1
            trajectory = np.vstack((trajectory, s))
        if visualize:
            performance_domain.showDomain(a)
        # performance_domain.close_figure()
        self.turnOnExploration()        
        print eps_return, eps_length, eps_term, eps_discount_return

        return trajectory, eps_return, eps_term

    def sample_state_distribution(self, agent_path=None, force=False):
        """:param agent_path: Path to directory where agent can be loaded
        Only saves states that have performed well"""
        trajectories = []
        for i in xrange(100):
            trj, ret, term = self.performanceRun(visualize=0)
            if ret > 0:
                trajectories.append(trj)

        trajectories = (np.vstack(trajectories))
        if agent_path is not None or force:
            joblib.dump(trajectories, 
                        os.path.join(agent_path, self.DISTRIBUTION_FILE))
        return trajectories


    def load_confidence_estimator(self, path_to_distribution=None, force=False):
        if path_to_distribution is not None and not force:
            print "Generating and loading confidence distribution from {}".format(path_to_distribution)
            try:
                distribution = joblib.load(os.path.join(path_to_distribution, self.DISTRIBUTION_FILE))
            except IOError:
                distribution = self.sample_state_distribution(path_to_distribution)
        else:
            print "Sampling disribution"
            distribution = self.sample_state_distribution(path_to_distribution)
        print "Shape of distribution is", distribution.shape

        kde = stats.gaussian_kde(distribution.T[:self.dims_estimated])
        # kde.set_bandwidth(bw_method=1.43 * 4)
        print "Set bandwidth to {}".format(kde.factor)
        self._estimator = kde
        return self._estimator


class DynamicsConfidenceGreedy(ConfidenceGreedy):
    """Implements confidence testing for dynamics"""

    def __init__(self, representation, **args):
        """
        :params representation: Internal representation
        :params dim: number of dimensions we want to calculate confidence for
           
        """
        self.alpha = 10e-8
        self.cache = {}
        print "Starting ", self.__class__.__name__
        super(DynamicsConfidenceGreedy, self).__init__(representation, dim=None, **args)

    # def _augment_distribution(self, distribution):
    #     size = distribution.shape[0]
    #     distribution = np.hstack((distribution, np.ones((size, 1)) * self.dynamics_encoding))
    #     return distribution

    def _estimate(self, sample):
        t = tuple(sample.tolist())
        
        if t not in self.cache:
            value = self._estimator.evaluate(sample)[0]
            self.cache[t] = value
        else:
            value = self.cache[t]
        return value

    def get_confidence(self, s, a, ns):
        assert len(s) == self.representation.domain.state_space_dims
        assert len(ns) == self.representation.domain.state_space_dims
        transition = np.hstack((s, a, ns))
        # if s[0] == 5 and s[1] == 8:

        domain = self.representation.domain
        condition = 0
        for d in [(-1,0), (1,0), (0,0), (0,1), (0,-1)]:
            test_state = [s[0] + d[0], s[1] + d[1]]
            if self.representation.domain.check_valid_state(test_state):
                if test_state[0] < 0:
                    print "ERRR0RRRR"
                sample = np.hstack((s, a, test_state))
                est = self._estimate(sample) 
                # print est
                condition += est + self.alpha
        conditional_confidence = (self._estimate(transition) + self.alpha) / condition 
        # could probably cache but who cares
        return conditional_confidence
        # return self._estimator.evaluate(transition)

    def performanceRun(self, visualize=False):
        """
        Saves trajectory rollouts

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        """

        # Set Exploration to zero and sample one episode from the domain
        assert self._estimator is None # we want to generate state distribution without confidence)

        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.turnOffExploration()
        if self.performance_domain is None:
            self.performance_domain = deepcopy(self.representation.domain)

        performance_domain = self.performance_domain
        s, eps_term, p_actions = performance_domain.s0()
        # print sum(agent.representation.weight_vec)
        trajectory = None

        while not eps_term and eps_length < performance_domain.episodeCap:
            a = self.pi(s, eps_term, p_actions)
            if visualize:
                performance_domain.showDomain(a)

            r, ns, eps_term, p_actions = performance_domain.step(a)

            transition = np.hstack((s, a, ns)) #KEY DIFFERENCE

            s = ns
            eps_return += r
            eps_discount_return += performance_domain.discount_factor ** eps_length * \
                r
            eps_length += 1

            if trajectory is None:
                trajectory = transition
            else:
                trajectory = np.vstack((trajectory, transition))
        if visualize:
            performance_domain.showDomain(a)
        # performance_domain.close_figure()
        self.turnOnExploration()        
        print "Performance run: ", eps_return, eps_length, eps_term, eps_discount_return

        return trajectory, eps_return, eps_term
