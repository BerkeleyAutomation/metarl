from rlpy.Agents.Agent import Agent
from itertools import combinations
import numpy as np
import time

class MetaAgent(Agent):
    def __init__(self, policy, representation, discount_factor=0.9, lambda_=0, no_learning=False, **kwargs):
        assert hasattr(policy, "subagents")
        self.lambda_ = lambda_
        self.subagents = policy.subagents
        self.no_learning = no_learning
        super(
            MetaAgent,
            self).__init__(policy=policy,
            representation=representation, discount_factor=discount_factor, **kwargs)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

        # print "Learning is called on X agents"
        # if terminal:
        if self.no_learning:
            return
        for agent in self.policy.subagents:
            agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)

        # for agent1, agent2 in combinations(self.policy.subagents, 2):
        #     assert all(agent1.representation.weight_vec == agent2.representation.weight_vec)


class MetaDynamicsAgent(MetaAgent):

    def __init__(self, policy, representation, discount_factor=0.9, lambda_=0, no_learning=False, **kwargs):
        assert type(policy).__name__ == "MultiTerrainPolicy" #hacks to make this work
        assert no_learning == True, "Learning is turned on!"
        super(
            MetaDynamicsAgent,
            self).__init__(policy=policy,
            representation=representation, discount_factor=discount_factor, **kwargs)

        self.classes = np.r_[:len(self.subagents)]
        self.training_examples = []
        self.training_labels = []


    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        
        assert hasattr(self.policy, "classifier")
        # assert False # Currently fault, as Q vs V logic isn't quite right
        dim = self.subagents[0].representation.state_space_dims
        t_sas = self.policy.get_subagent_transition_probabilities(s[:dim], a, ns[:dim])
        # print "Transition probabilities: ", t_sas

        try:
            # candidate_idx = np.argwhere(self.policy.prev_state_votes[1] == a)[0]
            pass
        except ValueError:
            pass
        best_transition = np.argmax(t_sas)

        # if s[0] == 0 and s[1] == 3:

        # ## DEBUG ##
        # print ">>>>>>> For transition ", s, a, ns
        # print "Following agents voted for this action ", candidate_idx
        # print "Agent predictions regarding transition: ", t_sas

        X, Y = [], []

        if np.std(np.log(t_sas)) > 1: # if discrepancy is too high, have to do some correction
            # for i in candidate_idx: # only deal with agents that care about this state
            for i, transition_prob in enumerate(t_sas):
                if transition_prob < np.mean(t_sas):
                    X.append(np.append(s, [a, i])) 
                    Y.append(0)
                    # print "!! Punishing agent %d for being completely wrong" % i
                else:
                    # if not (s[-1] == i):
                    # if s[0] == 4:


                    X.append(np.append(s, [a, i])) # which agent - may also want to encode action in case that agent is not optimal
                    Y.append(1)
                    # print "!! Rewarding agent %d for being accurate" % i

            if len(self.training_examples) % 5 == 1:
                self.policy.show_learning(0)
                self.policy.show_learning(1)


            X = np.asarray(X)
            Y = np.asarray(Y)        
            self.policy.classifier.partial_fit(X, Y, classes=self.classes)
            self.training_examples.append(X)
            self.training_labels.append(Y)


class MetaQClassifierAgent(MetaAgent):

    def __init__(self, policy, representation, discount_factor=0.9, lambda_=0, no_learning=False, **kwargs):
        assert type(policy).__name__ == "MultiTerrainQPolicy" #hacks to make this work
        assert no_learning == False, "No learning is turned on!"
        super(
            MetaQClassifierAgent,
            self).__init__(policy=policy,
            representation=representation, discount_factor=discount_factor, **kwargs)

        self.classes = np.r_[:len(self.subagents)]

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        assert hasattr(self.policy, "classifier")
        assert False # Currently fault, as Q vs V logic isn't quite right
        dim = self.subagents[0].representation.state_space_dims
        V_values = [agent.representation.V(ns[:dim], terminal, np_actions) for agent in self.policy.subagents]
        Q_values = [agent.representation.Q(s[:dim], terminal, a) for agent in self.policy.subagents]
        diffs = np.subtract(V_values, Q_values)

        best_agent = np.argmax(diffs) # expected that if you were not trained on this domain, you will not each a better state than expected.
        if self.policy.discrep:
            self.policy.classifier.partial_fit(s.reshape(1, -1), np.array([best_agent]), classes=self.classes)


        # get V(ns) for each subagent
        # find agent with largest positive difference between V(ns) and Q(s, a)
