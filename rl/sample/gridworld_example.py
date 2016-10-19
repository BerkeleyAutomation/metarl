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
from utils import loadagent, load_all_agents

param_space = {'lambda_': hp.uniform("lambda_", 0., 1.),
				'discount_factor': hp.uniform("discount_factor", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


AGENT = "SARSA"

def make_experiment(exp_id=2,
					agent_paths=None,
					mapname="12x12-Bridge.txt",
					path="./Results/MetaRLSarsa",
					boyan_N0=1000, 
					discount_factor=0.8286376417073243,
					initial_learn_rate=0.5,
					lambda_=0.2):
	opt = {}
	opt["path"] = path
	opt["exp_id"] = exp_id
	opt["max_steps"] = 30000
	opt["num_policy_checks"] = 5
	opt["checks_per_policy"] = 10
	# start_at = np.array([4, 5])

	# Logging

	# Domain:
	# MAZE                = '/Domains/GridWorldMaps/1x3.txt'
	maze = os.path.join(map_dir, mapname)

	map_dir = os.path.expanduser("~/work/clipper/models/rl/GridworldMaps/")

	domain = GridWorldModified(maze, 
									# random_start=True, 
									noise=0.1,
									# start_at=np.array([4,6])
									)	

	# agent_1 = loadagent("QL") # most likely preloaded
	# agent_2 = loadagent("SARSA")
	# agent_3 = loadagent("NAC")
	# agents = [agent_1, agent_2, agent_3]
	# agents = load_all_agents(agent_paths)

	# domain = PolicyMixer(actual_domain, agents, seed=exp_id)
	representation = Tabular(domain)
	policy = eGreedy(representation, epsilon=0.3)
	opt['agent'] = Q_Learning(representation=representation,
							 policy=policy,
							 learn_rate_decay_mode="boyan",
							 boyan_N0=boyan_N0,
							 lambda_=lambda_,
							 initial_learn_rate=initial_learn_rate,
							 discount_factor=discount_factor)
	opt['domain'] = domain
	experiment = Experiment(**opt)

	return experiment

if __name__ == '__main__':
	path = "./Results/DiffAgents/MetaRLSarsa2/" + AGENT
	experiment = make_experiment(path=path,)
	experiment.run(visualize_steps=False,  # should each learning step be shown?
				   visualize_learning=False,  # show performance runs?
				   visualize_performance=False)  # show value function?
	# import pickle
	# with open("teststats_pit.p", "w") as f:
	# 	pickle.dump(domain.statistics, f)
	# create_csv(experiment.domain.statistics)
	# experiment.plot()
	# experiment.save()