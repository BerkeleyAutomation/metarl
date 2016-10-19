import sys
import os 
cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
sys.path.append(cur_dir)

# from PolicyMixer import PolicyMixer
from rlpy.Agents import Q_Learning, SARSA
from rlpy.Domains import GridWorld
from rlpy.Representations import Tabular, IncrementalTabular, RBF
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
# from GridWorldModified import GridWorldModified

import os
from copy import deepcopy, copy 
import joblib
import AgentGenerator
from hyperopt import hp
import numpy as np
from datetime import datetime
import yaml
import pickle 

import scipy.stats as stats

VALUES_FILE = "weights.p" # count might not be needed
# history_str = "history.p"
DISTRIBUTION_FILE = "state_dist.p"

def load_all_agents(agent_paths, pretrained=True, load_confidence=False, yaml_file="params.yml"):
	return [agent_from_path(pth, pretrained=pretrained, load_confidence=load_confidence, yaml_file=yaml_file) for pth in agent_paths]

def agent_from_path(agent_path, pretrained=True, load_confidence=False, yaml_file="params.yml"):
	exp = AgentGenerator.make_experiment(agent_path, yaml_file=yaml_file)
	if load_confidence:
		assert pretrained
	if pretrained:
		exp.agent.representation.load_from_directory(agent_path)
		if load_confidence:
			exp.agent.policy.load_confidence_estimator(agent_path)
	return exp.agent
	#todo - get agent from directory given from path

	
# def loadagent(agentstr, ver=None):
# 	"""DEPRECATED"""
# 	exp = AgentGenerator.make_experiment(agentstr, epsilon=0.0)
# 	weightstr = "/weights.p"
# 	if ver is not None:
# 		weightstr = "/weights{}.p".format(ver)
# 	weights = joblib.load(cur_dir + agentstr + weightstr)
# 	exp.agent.representation.weight_vec = weights 
# 	return exp.agent

def create_csv(stats, fname="teststats_pit.csv"):
	import csv
	with open(fname, "w") as f:
		write = csv.writer(f)
		write.writerow(["tnum", "y", "x", "agent", "action"])
		for i, trajectory in enumerate(stats):
			for state, agent, action in trajectory:
				write.writerow([i, state[0], state[1], agent, action])

	print "DONE"

def load_yaml(file_path):
	with open(file_path, 'r') as f:
		ret_val = yaml.load(f)
	return ret_val

def get_time_str():
	return datetime.now().strftime('%h%d_%I-%M-%f')


# def performanceRun(agent, visualize=False):
#     """
#     Execute a single episode using the current policy to evaluate its
#     performance. No exploration or learning is enabled.

#     :param total_steps: int
#         maximum number of steps of the episode to peform
#     :param visualize: boolean, optional
#         defines whether to show each step or not (if implemented by the domain)
#     """

#     # Set Exploration to zero and sample one episode from the domain
#     eps_length = 0
#     eps_return = 0
#     eps_discount_return = 0
#     eps_term = 0

#     agent.policy.turnOffExploration()
#     performance_domain = agent.representation.domain


#     s, eps_term, p_actions = performance_domain.s0()
#     trajectory = s
#     # print sum(agent.representation.weight_vec)

#     while not eps_term and eps_length < performance_domain.episodeCap:
#         a = agent.policy.pi(s, eps_term, p_actions)
#         if visualize:
#             performance_domain.showDomain(a)

#         r, ns, eps_term, p_actions = performance_domain.step(a)
#         # _gather_transition_statistics(s, a, ns, r, learning=False)
#         s = ns
#         eps_return += r
#         eps_discount_return += performance_domain.discount_factor ** eps_length * \
#             r
#         eps_length += 1
#         trajectory = np.vstack((trajectory, s))
#     if visualize:
#         performance_domain.showDomain(a)
#     agent.policy.turnOnExploration()        
#     # print eps_return, eps_length, eps_term, eps_discount_return

#     return trajectory, eps_return, eps_term

# def sample_state_distribution(test_agent, save=True):
#     """:param agent_path: Path to directory where agent can be loaded
#     Only saves states that have performed well"""
#     trajectories = []
#     for i in xrange(100):
#         trj, ret, term = performanceRun(test_agent, visualize=0)
#         if ret > 0:
#             trajectories.append(trj)

#     trajectories = (np.vstack(trajectories))
#     if save:
#         joblib.dump(trajectories, 
#                     os.path.join(agent_path, DISTRIBUTION_FILE))
#     return trajectories

# def state_distribution_model(agent_path, dim=2):
#     test_agent = agent_from_path(agent_path, 
#                                 False, 
#                                 pretrained=True)
#     distribution = sample_state_distribution(test_agent).T[:dim]
#     kde = stats.gaussian_kde(distribution)
#     return kde