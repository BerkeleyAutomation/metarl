from rlpy.Agents import Q_Learning
from rlpy.Domains.Domain import Domain
import numpy as np
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import combinations
from rlpy.Tools import plt
# import rllab

class PolicyMixer(Domain):
    
    # This class takes in an index now, which references a policy specifically within the mix
    # agents = []

    episodeCap = None
    _s = None
    _p_actions = None
    _a = None
    _r = None
    _ns = None
    _np_actions = None
    _na = None
    _term = None
    
    performance_domain = False
    last_run = None
    valueFunction_fig = None

    def __init__(self, domain, subagents):

        self.statistics = []
        self.curr_run = []
        self.actual_domain = domain
        self.subagents = np.array(subagents)

        self.steps = 0
        self.episodeCap = domain.episodeCap
        assert not any([agent.policy.epsilon for agent in self.subagents])
        self.actions_num = len(self.subagents)
        self.statespace_limits = self.actual_domain.statespace_limits
        super(PolicyMixer, self).__init__()

    def s0(self):
        self.statistics.append(self.curr_run)
        self.curr_run = []
        self._s = None

        self._ns, self._term, self._np_actions = self.actual_domain.s0() # this is different from given
        # print s
        # print "S0: After: %f" % sum(self.subagents[0].policy.representation.weight_vec)

        return self._ns, self._term, np.r_[:len(self.subagents)]

    def isTerminal(self):
        return self.actual_domain.isTerminal()

    def showDomain(self, a):
        # print "Current action is {}".format(a)
        return self.actual_domain.showDomain(a)

    def showLearning(self, representation):
        pass
        # self.get_value_function(representation)
        # return self.gridworld_showlearning(representation)


    def step(self, a): #a here is for agent
        assert a < len(self.subagents)
        # print self._ns
        assert np.array_equal(self._ns, self.actual_domain.state), "States not consistent"

        major_subagents = None
        cur_agent = self.subagents[a]
        self._na = cur_agent.policy.pi(self.actual_domain.state, 
                        self.isTerminal(), 
                        self._np_actions)

        self._s, self._p_actions, self._a = self._ns, self._np_actions, self._na

        self._r, self._ns, self._term, self._np_actions = self.actual_domain.step(self._a)


        self.curr_run.append((self.actual_domain.state, a, self._a))
        self.steps += 1

        return self._r, self._ns, self._term, np.r_[:len(self.subagents)]
        # get all other actions from 


    def gridworld_showlearning(self, representation):
        dom = self.actual_domain
        if self.valueFunction_fig is None:
            plt.figure("Value Function")
            self.valueFunction_fig = plt.imshow(
                dom.map,
                cmap='ValueFunction',
                interpolation='nearest',
                vmin=dom.MIN_RETURN,
                vmax=dom.MAX_RETURN)
            plt.xticks(np.arange(dom.COLS), fontsize=12)
            plt.yticks(np.arange(dom.ROWS), fontsize=12)
            # Create quivers for each action. 4 in total
            plt.show()
        plt.figure("Value Function")

        V = self.get_value_function(representation)
        # print Acts
        # Show Value Function
        self.valueFunction_fig.set_data(V)
        plt.draw()
        # plt.pause(0.1)

    def get_value_function(self, representation):
        dom = self.actual_domain
        V = np.zeros((dom.ROWS, dom.COLS))
        Acts = np.zeros((dom.ROWS, dom.COLS))

        for r in xrange(dom.ROWS):
            for c in xrange(dom.COLS):
                if dom.map[r, c] == dom.BLOCKED:
                    V[r, c] = 0
                    Acts[r, c] = 9
                if dom.map[r, c] == dom.GOAL:
                    V[r, c] = dom.MAX_RETURN
                if dom.map[r, c] == dom.PIT:
                    V[r, c] = dom.MIN_RETURN
                if dom.map[r, c] == dom.EMPTY or dom.map[r, c] == dom.START:
                    s = np.array([r, c])
                    As = range(2) # dom.possibleActions(s)
                    terminal = dom.isTerminal(s)
                    Qs = representation.Qs(s, terminal)
                    bestA = representation.bestActions(s, terminal, As)
                    V[r, c] = max(Qs[As])
                    Acts[r, c] = np.argmax(Qs[As])

                    # if r == 4 and c == 8: #and Acts[4, 8] == 0:
                    #     print "Transition at 4, 8 is ", Qs
        # print Acts
        return V
    # def _majority_vote(self):
    #     """:returns action chosen, subagents_correct: Action chosen and a list of indices
    #     of agents that correctly chose this action
        
    #     This method should be AGNOSTIC to learning. Performance
    #         should not change over time """

    #     agent_actions = np.array([agent.policy.pi(self.actual_domain.state, 
    #                         self.isTerminal(), 
    #                         self._np_actions)
    #                         for agent in self.agents])
    #     # if not all(agent_actions == agent_actions[0]):
    #     #     print "Randomness here"
            
    #     votes = Counter(agent_actions)
    #     actions, counts = zip(*votes.items()) 
    #     action_chosen = self._choose_boltzmann(actions, counts)
    #     return action_chosen, np.where(agent_actions==action_chosen)[0]

    # def choose_confident_action(self):
    #     """:returns action: Picked from boltzmann distribution created based on 
    #     number of times an agent has reached a certain state"""
    #     s = self.actual_domain.state
    #     # import ipdb; ipdb.set_trace()
    #     agent_actions = np.array([agent.policy.pi(s,
    #                         self.isTerminal(), 
    #                         self._np_actions)
    #                         for agent in self.agents])
    #     confidences = np.array([agent.policy.get_confidence(s) for agent in self.agents])
    #     # import ipdb; ipdb.set_trace()
    #     confidences = confidences / np.mean(confidences) # hack

    #     # return self._choose_boltzmann(agent_actions, confidences)

    #     action_dict = defaultdict(int)
    #     for a, c in zip(agent_actions, confidences):
    #         action_dict[a] += c

    #     a = self._choose_boltzmann(action_dict.keys(), action_dict.values())

    #     if rnd.random() > 0.999:
    #         print s, confidences, a

    #     return a, np.where(agent_actions==a)[0]

    #     # try voting stuff
    #     # print confidences 


    # def _choose_boltzmann(self, agent_actions, values):
    #     assert len(agent_actions) == len(values)
    #     # import ipdb; ipdb.set_trace()
    #     boltz = np.array([np.exp(float(cnt) / self.tau) 
    #                         for cnt in values])
    #     boltz /= sum(boltz)
    #     return rnd.choice(agent_actions, p=boltz)


    # def subagent_learn_all(self):
    #     self.subagent_learn(None, self._s, self._p_actions, self._a, self._r, 
    #                             self._ns, self._np_actions, self._na, self._term)


    # def subagent_learn(self, possibleagents, s, p_actions, a, r, ns, np_actions, na, terminal):
    #     """:params possibleagents: list of indices"""
    #     if possibleagents is None:
    #         possibleagents = range(len(self.agents))
    #     learning_agents = self.agents[possibleagents]
    #     for agent in learning_agents:
    #         agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
    #     # if self.voting:

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # import ipdb; ipdb.set_trace() 
        for k, v in self.__dict__.items():
            print k, v
            if k == "agents" or k == "logger" or k == "subagents":
                continue
            try:
                setattr(result, k, deepcopy(v, memo))
            except TypeError:
                import ipdb; ipdb.set_trace()  # breakpoint f758fad4 //

        result.subagents = self.subagents
        result.performance_domain = True
        return result
