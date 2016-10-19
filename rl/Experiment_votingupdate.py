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
from Domains import RCCarModified

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
					pretrained=False,
					sublearning=False,
					yaml_file=None,
					exp_id=3,
					path="./Results/TestVoting/",
					temp=0.90517212721767522,
					boyan_N0=100, 
					discount_factor=0.7,
					initial_learn_rate=0.1,
					lambda_=0.0,
					init_state=None):
	opt = {}
	opt["path"] = path
	opt["exp_id"] = exp_id
	opt["max_steps"] = 50000
	opt["num_policy_checks"] = 50
	opt["checks_per_policy"] = 10
	# start_at = np.array([4, 6])

	# Logging

	# Domain:
	# MAZE                = '/Domains/GridWorldMaps/1x3.txt'
	actual_domain = RCCarModified(noise=0.1,
									# random_start=True,
									init_state=init_state
									)	

	# agent_1 = loadagent("QL") # most likely preloaded
	# agent_2 = loadagent("SARSA")
	# agent_3 = loadagent("NAC")
	# agents = [agent_1, agent_2, agent_3]
	agents = load_all_agents(agent_paths, eps=0.0, pretrained=pretrained, yaml_file=yaml_file)

	for i, a in enumerate(agents):
		# import ipdb; ipdb.set_trace()
		a.policy.epsilon = i * 0.05 + 0.1

	# assert agents[1].policy.eps == 0.25

	assert agents[0].policy.representation.random_state.get_state()[2] == 432
	assert agents[0].policy.representation.random_state.get_state()[1][0] == 2308721491
	assert agents[0].policy.random_state.get_state()[1][0] == 1
	assert agents[0].policy.random_state.get_state()[2] == 624

	print "ASSERTIONS PASSED"

	# import ipdb; ipdb.set_trace()
	domain = PolicyMixer(actual_domain, agents, 
						voting=True, seed=exp_id, temp=temp, sublearning=sublearning)

	# import ipdb; ipdb.set_trace()
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


	#Seeding to match standard experiment
	for agent in agents:
	    agent.random_state = np.random.RandomState(
	        experiment.randomSeeds[experiment.exp_id - 1])
	    agent.init_randomization()
	    agent.representation.random_state = np.random.RandomState(
	        experiment.randomSeeds[experiment.exp_id - 1])
	    agent.representation.init_randomization()
	    agent.policy.random_state = np.random.RandomState(
	        experiment.randomSeeds[experiment.exp_id - 1])
	    agent.policy.init_randomization()

	return experiment
	

if __name__ == '__main__':
	path = "./Results/DiffAgents/VotingUpdate/" + AGENT

	experiment = make_experiment(agent_paths=["./params/"] * 5,
								 yaml_file="voting_params.yml",
								 sublearning=True,
								 path=path,
								 pretrained=False)
	experiment.run(
					visualize_steps=False,  # should each learning step be shown?
				   visualize_learning=False,  # show performance runs?
				   visualize_performance=True)  # show value function?
	# import pickle
	# with open("teststats_pit.p", "w") as f:
	# 	pickle.dump(domain.statistics, f)
	# create_csv(domain.statistics)
	experiment.plot()
	experiment.save()