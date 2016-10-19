import sys
import os 
cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
sys.path.append(cur_dir)

from PolicyMixer import PolicyMixer
from rlpy.Agents import Q_Learning, SARSA
from rlpy.Domains import GridWorld
from rlpy.Representations import Tabular, IncrementalTabular, RBF
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
from GridWorldModified import GridWorldModified

import os
from copy import deepcopy, copy 
import joblib
import AgentGenerator
from hyperopt import hp
import numpy as np

param_space = {'lambda_': hp.uniform("lambda_", 0., 1.),
				'discount_factor': hp.uniform("discount_factor", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}

map_dir = os.path.expanduser("~/work/clipper/models/rl/GridworldMaps/")

def loadagent(agentstr, ver=None):
	exp = AgentGenerator.make_experiment(agentstr)
	weightstr = "/weights.p"
	if ver is not None:
		weightstr = "/weights{}.p".format(ver)
	weights = joblib.load(cur_dir + agentstr + weightstr)
	exp.agent.representation.weight_vec = weights 
	return exp.agent

def create_csv(stats, fname="teststats_pit.csv"):
	import csv
	with open(fname, "w") as f:
		write = csv.writer(f)
		write.writerow(["tnum", "y", "x", "agent", "action"])
		for i, trajectory in enumerate(stats):
			for state, agent, action in trajectory:
				write.writerow([i, state[0], state[1], agent, action])

	print "DONE"


def make_experiment(exp_id=2,
					path="./Results/MetaRL/",
					boyan_N0=680.7, 
					discount_factor=0.572,
					initial_learn_rate=0.4665,
					lambda_=0.106):
	opt = {}
	opt["path"] = path
	opt["exp_id"] = exp_id
	opt["max_steps"] = 100000
	opt["num_policy_checks"] = 50
	opt["checks_per_policy"] = 100

	# Logging

	# Domain:
	# MAZE                = '/Domains/GridWorldMaps/1x3.txt'
	maze = os.path.join(map_dir, '11x11-Paths.txt')
	actual_domain = GridWorldModified(maze, random_start=True, noise=0.1)	

	# agent_1 = loadagent("QL") # most likely preloaded
	# agent_2 = loadagent("SARSA")
	# agent_3 = loadagent("NAC")
	# agents = [agent_1, agent_2, agent_3]
	agents = [loadagent("SARSA", i) for i in range(4)]

	domain = PolicyMixer(actual_domain, agents, seed=exp_id)
	representation = Tabular(domain)
	meta_policy = eGreedy(representation, epsilon=0.3)
	opt['agent'] = Q_Learning(representation=representation,
							 policy=meta_policy,
							 learn_rate_decay_mode="boyan",
							 boyan_N0=boyan_N0,
							 lambda_=lambda_,
							 initial_learn_rate=initial_learn_rate,
							 discount_factor=discount_factor)
	opt['domain'] = domain
	experiment = Experiment(**opt)

	return experiment

if __name__ == '__main__':
	experiment = make_experiment()
	experiment.run(visualize_steps=False,  # should each learning step be shown?
				   visualize_learning=False,  # show performance runs?
				   visualize_performance=False)  # show value function?
	# import pickle
	# with open("teststats_pit.p", "w") as f:
	# 	pickle.dump(domain.statistics, f)
	# create_csv(domain.statistics)
	# experiment.plot()
	experiment.save()