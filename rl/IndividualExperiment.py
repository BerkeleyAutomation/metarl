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


def load_experiments(agent_path):
	exp = AgentGenerator.make_experiment(agent_path) #, epsilon=0.0)
	weightstr = "/weights.p"
	weights = joblib.load(agent_path+ weightstr)
	exp.agent.representation.weight_vec = weights 
	return exp

def make_experiment(path="./Results/DiffAgents/Individual/", 
					agent_path=".", mapname='12x12-Bridge.txt'):
	# Logging

	# Domain:
	# MAZE                = '/Domains/GridWorldMaps/1x3.txt'

	experiment = load_experiments(agent_path)
	experiment.path = path
	maze = os.path.join(map_dir, mapname)
	actual_domain = GridWorldModified(maze, 
									random_start=True, 
									noise=0.1,
									# start_at=start_at
									)	
	experiment.domain = actual_domain

	return experiment

if __name__ == '__main__':
	agent = "SARSA"
	ver = 4
	path = "./Results/DiffAgents/Individual/" + agent + str(ver)
	experiment = make_experiment(path, agent, ver)

	experiment.max_steps=100000
	experiment.num_policy_checks = 50
	experiment.checks_per_policy = 200


	experiment.run(visualize_steps=False,  # should each learning step be shown?
				   visualize_learning=False,  # show performance runs?
				   visualize_performance=False)  # show value function?
	# import pickle
	# with open("teststats_pit.p", "w") as f:
	# 	pickle.dump(domain.statistics, f)
	# create_csv(experiment.domain.statistics)
	experiment.plot()
	experiment.save()