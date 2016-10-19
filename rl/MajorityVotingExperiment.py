import sys
import os 
cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
sys.path.append(cur_dir)
from PolicyMixer import PolicyMixer
from rlpy.Agents import Q_Learning
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
from utils import load_all_agents

param_space = { 'temp': hp.uniform("temp", 0., 1.),
				# 'lambda_': hp.uniform("lambda_", 0., 1.),
				# 'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
				# 'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))
				}

map_dir = os.path.expanduser("~/work/clipper/models/rl/GridworldMaps/")

AGENT = "SARSA"

def make_experiment(agent_paths="./", 
					pretrained=True,
					sublearning=False,
					mapname="12x12-Bridge.txt",
					exp_id=3,
					path="./Results/TestVoting/",
					temp=0.10517212721767522,
					boyan_N0=100, 
					discount_factor=0.7,
					initial_learn_rate=0.1,
					lambda_=0.0):
	opt = {}
	opt["path"] = path
	opt["exp_id"] = exp_id
	opt["max_steps"] = 100000
	opt["num_policy_checks"] = 10
	opt["checks_per_policy"] = 100
	# start_at = np.array([4, 6])

	# Logging

	# Domain:
	# MAZE                = '/Domains/GridWorldMaps/1x3.txt'
	maze = os.path.join(map_dir, mapname)
	actual_domain = GridWorldModified(maze,
									noise=0.1,
									random_start=True,
									# start_at=start_at
									)	

	# agent_1 = loadagent("QL") # most likely preloaded
	# agent_2 = loadagent("SARSA")
	# agent_3 = loadagent("NAC")
	# agents = [agent_1, agent_2, agent_3]
	agents = load_all_agents(agent_paths, pretrained=pretrained)

	domain = PolicyMixer(actual_domain, agents, 
						voting=True, seed=exp_id, temp=temp, sublearning=sublearning)
	representation = Tabular(domain)
	meta_policy = eGreedy(representation, epsilon=0.3)
	opt['agent'] = Q_Learning(representation=representation,
							 policy=meta_policy,
							 learn_rate_decay_mode="const",
							 boyan_N0=0,
							 lambda_=0,
							 initial_learn_rate=0.0,
							 discount_factor=discount_factor)
	opt['domain'] = domain
	experiment = Experiment(**opt)
	return experiment
	

if __name__ == '__main__':
	path = "./Results/DiffAgents/VotingUpdate/" + AGENT

	experiment = make_experiment(agent_paths=["./votingupdate/"] * 5, path=path,)
	experiment.run(
					visualize_steps=False,  # should each learning step be shown?
				   visualize_learning=False,  # show performance runs?
				   visualize_performance=False)  # show value function?
	# import pickle
	# with open("teststats_pit.p", "w") as f:
	# 	pickle.dump(domain.statistics, f)
	# create_csv(domain.statistics)
	experiment.plot()
	experiment.save()