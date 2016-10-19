import sys
import os 
cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
sys.path.append(cur_dir)
from Agents import MetaAgent
from Policies import MultiAgentVoting
# from rlpy.Agents import Q_Learning
# from rlpy.Domains import GridWorld
from rlpy.Representations import Tabular, IncrementalTabular, RBF
# from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
from Domains import RCCarModified

import os
import shutil
import inspect
from copy import deepcopy, copy 
import joblib
import AgentGenerator
from hyperopt import hp
import numpy as np
from utils import load_all_agents, get_time_str

import pprint
pp = pprint.PrettyPrinter(indent=4)

param_space = { 'temp': hp.uniform("temp", 0., 1.),
				# 'lambda_': hp.uniform("lambda_", 0., 1.),
				# 'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
				# 'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))
				}

# map_dir = os.path.expanduser("~/work/clipper/models/rl/GridworldMaps/")

def make_experiment(agent_paths="./", 
					pretrained=False,
					sublearning=False,
					yaml_file=None,
					exp_id=3,
					path="./Results/TestVoting/",
					init_state=None):
	opt = {}
	opt["path"] = path
	opt["exp_id"] = exp_id
	opt["max_steps"] = 80000
	opt["num_policy_checks"] = 40
	opt["checks_per_policy"] = 10
	# start_at = np.array([4, 6])

	# Logging

	# Domain:
	# MAZE                = '/Domains/GridWorldMaps/1x3.txt'
	domain = RCCarModified(noise=0.1,
									# random_start=True,
									# init_state=init_state
									)	

	# agent_1 = loadagent("QL") # most likely preloaded
	# agent_2 = loadagent("SARSA")
	# agent_3 = loadagent("NAC")
	# agents = [agent_1, agent_2, agent_3]
	agents = load_all_agents(agent_paths, pretrained=pretrained, yaml_file=yaml_file)

	for i, a in enumerate(agents):
		# import ipdb; ipdb.set_trace()
		a.policy.epsilon = i * 0.03 + 0.1

	# import ipdb; ipdb.set_trace()
	representation = IncrementalTabular(domain) #This doesn't matter
	policy = MultiAgentVoting(representation, agents, tau=.7)
	opt['agent'] = MetaAgent(representation=representation,
							 policy=policy,)
	opt['domain'] = domain
	experiment = Experiment(**opt)


	#Seeding to match standard experiment
	for agent in agents:
	    agent.random_state = np.random.RandomState(
	        experiment.randomSeeds[experiment.exp_id - 1])
	    agent.init_randomization()
	    # agent.representation.random_state = np.random.RandomState(
	    #     experiment.randomSeeds[experiment.exp_id - 1])
	    # agent.representation.init_randomization() #init_randomization is called on instantiation
	    agent.policy.random_state = np.random.RandomState(
	        experiment.randomSeeds[experiment.exp_id - 1])
	    agent.policy.init_randomization()


	for i, a in enumerate(agents):
		# import ipdb; ipdb.set_trace()
		print a.policy.epsilon 

	path_join = lambda s: os.path.join(opt["path"], s)
	if not os.path.exists(opt["path"]):
		os.makedirs(opt["path"])

	param_path = os.path.join(agent_paths[0], yaml_file)
	shutil.copy(param_path, path_join("params.yml"))
	shutil.copy(inspect.getsourcefile(inspect.currentframe()), path_join("experiment.py"))
  
	# import ipdb; ipdb.set_trace()

	return experiment
	

if __name__ == '__main__':
	path = "./Results/DiffAgents/VotingUpdateMaha/" + get_time_str()

	experiment = make_experiment(agent_paths=["./params/"] * 5,
								 yaml_file="individ_params.yml",
								 sublearning=True,
								 path=path,
								 pretrained=False)

	pp.pprint(experiment.__dict__)

	pp.pprint(experiment.domain.__dict__)
	pp.pprint(experiment.agent.__dict__)
	# sys.exit()

	experiment.run(visualize_steps=False,  # should each learning step be shown?
				   visualize_learning=False,  # show performance runs?
				   visualize_performance=0)  # show value function?
	# import pickle
	# with open("teststats_pit.p", "w") as f:
	# 	pickle.dump(domain.statistics, f)
	# create_csv(domain.statistics)
	print experiment.path
	experiment.save()
	experiment.plot()