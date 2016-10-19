
from rlpy.Policies.Policy import Policy
import numpy as np
import pickle
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import joblib
import time

class MultiAgentPolicy(Policy):
    # The policy is in charge of mapping state to action

    learn_graph = None

    def __init__(self, representation, subagents, epsilon=0.0, tau=1.0, decay=False,
                        forcedDeterministicAmongBestActions=False, seed=1):
        print "MultiAgentPolicy epsilon set to 0"
        self.seed = seed
        self.subagents = subagents
        self.tau = tau
        self.initial_tau = self.tau
        self.decay_count = 0
        self.decay = decay
        for i, agent in enumerate(self.subagents):
            if not any(agent.representation.weight_vec):
                print "[INFO] Agent (%d) is Blank", i
            else:
                print "[INFO] Agent (%d) has experience: %f" %( i, sum(agent.representation.weight_vec))
        self._explore = True
        self.epsilon = epsilon
        self.seed = seed #random_state is initialized in the superclass
        self._trajectory = defaultdict(list)
        self._learnrates = []
        self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
        super(MultiAgentPolicy, self).__init__(representation, seed)

    def pi(self, s, terminal, p_actions):
        pass

    def terminal(self):
        self._trajectory = defaultdict(list)

    def save_learning_rates(self):
        joblib.dump(self._learnrates, "tmp/learning_rates.jb")

    def get_learning_rates(self):
        rates = np.array([agent.learn_rate for agent in self.subagents])
        if any(rates != 1):
            pass
        return rates


    def tau_decay(self):
        # self.decay_count += 1
        # self.tau = self.initial_tau * (8000 + 1.) / \
        #         (8000 + (self.decay_count / 100 + 1) ** 1.1)
        pass

    def prob(self, s, terminal, p_actions):
        p = np.ones(len(p_actions)) / len(p_actions)
        return p

    def turnOffExploration(self):
        self._explore = False
        for agent in self.subagents:
            # print "Agent policy exploration turned off"
            agent.policy.turnOffExploration()
            # print "Epsilon set to %f" % agent.policy.epsilon

    def turnOnExploration(self):
        self._explore = True
        for agent in self.subagents:
            # print "Agent policy exploration turned on"
            agent.policy.turnOnExploration()
            # import ipdb; ipdb.set_trace()
            # print "Epsilon set to %f" % agent.policy.epsilon

    def _choose_boltzmann(self, agent_actions, values):
        assert len(agent_actions) == len(values)
        # import ipdb; ipdb.set_trace()
        boltz = np.array([np.exp(float(cnt) / self.tau) 
                            for cnt in values])
        boltz /= sum(boltz)
        # if self._explore and self.decay:
        #     self.tau_decay()
        return self.random_state.choice(agent_actions, p=boltz)


    def agent_epsilons(self):
        return [agent.policy.epsilon for agent in self.subagents]

    def _debug(self):
        return zip(
            self._trajectory['options'], 
            self._trajectory['chosen'])


class MultiAgentVoting(MultiAgentPolicy):

    def __init__(self, representation, subagents, **kwargs):
        # import ipdb; ipdb.set_trace()
        self.printing = False
        super(MultiAgentVoting, self).__init__(representation, subagents, **kwargs)

    def pi(self, s, terminal, p_actions):

        """:returns action chosen
        """

        agent_actions = np.array([agent.policy.pi(s, 
                            terminal, 
                            p_actions)
                            for agent in self.subagents])
        # if not all(agent_actions == agent_actions[0]):
        #     print "Randomness here"

        votes = Counter(agent_actions)
        actions, counts = zip(*votes.items()) 

        # if rnd.random() > 0.999:
        #     print "[VOTING]", actions, counts

        action_chosen = self._choose_boltzmann(actions, counts)

        self._trajectory['options'].append(agent_actions)
        self._trajectory['chosen'].append(action_chosen)
        # self._learnrates.append(self.get_learning_rates())

        if self.printing:
            # time.sleep(0.5)
            print "{0} - {1}".format(agent_actions, np.argwhere(agent_actions == action_chosen).T)

        # if terminal: This is done in the experiment code bc policy has no access to reset
        #     self._trajectory = defaultdict(list)

        return action_chosen #, np.where(agent_actions==action_chosen)[0]

    def turn_on_printing(self):
        self.printing = True

    def turn_off_printing(self):
        self.printing = False

class MultiAgentConfidence(MultiAgentPolicy):

    def __init__(self, representation, subagents,  **kwargs):
        # import ipdb; ipdb.set_trace()
        super(MultiAgentConfidence, self).__init__(representation, subagents, **kwargs)

    def pi(self, s, terminal, p_actions):

        """:returns action chosen
        """

        agent_actions = np.array([agent.policy.pi(s, 
                            terminal, 
                            p_actions)
                            for agent in self.subagents])

        confidences = np.array([agent.policy.get_confidence(s) for agent in self.subagents])
        
        # action_dict = defaultdict(int)
        # for a, c in zip(agent_actions, confidences):
        #     action_dict[a] += c

        action_chosen = agent_actions[np.argmax(confidences)]
        # print np.argmax(confidences)
        action_chosen = 0

        # if self.random_state.rand() > 0.999:
        #     print s, confidences, action_chosen

        # self._trajectory['options'].append(agent_actions)
        # self._trajectory['chosen'].append(action_chosen)

        # if terminal:
        #     self._trajectory = defaultdict(list)

        return action_chosen #, np.where(agent_actions==action_chosen)[0]

class MultiAgentQ(MultiAgentPolicy):

    def __init__(self, representation, subagents,  **kwargs):
        # import ipdb; ipdb.set_trace()
        super(MultiAgentQ, self).__init__(representation, subagents, **kwargs)

    def pi(self, s, terminal, p_actions):
        agents_maxq = np.array([agent.representation.Qs(s, terminal) 
                                    for agent in self.subagents])
        import ipdb; ipdb.set_trace()  # breakpoint 272c50a8 //
        return 1
        # use q in boltzmann