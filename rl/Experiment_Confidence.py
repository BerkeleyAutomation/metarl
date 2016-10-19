import sys
import os 
cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
sys.path.append(cur_dir)
from Agents import MetaAgent
from Policies import MultiAgentConfidence, MultiAgentVoting
# from rlpy.Agents import Q_Learning
# from rlpy.Domains import GridWorld
from rlpy.Representations import Tabular, IncrementalTabular, RBF
# from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
from Domains.RCCarModified import RCCarModified

import os
from copy import deepcopy, copy 
import joblib
import AgentGenerator
from hyperopt import hp
import numpy as np
from utils import load_all_agents, get_time_str

param_space = { 'temp': hp.uniform("temp", 0., 1.),
				# 'lambda_': hp.uniform("lambda_", 0., 1.),
				# 'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
			# 'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))
				}

# map_dir = os.path.expanduser("~/work/clipper/models/rl/GridworldMaps/")

AGENT = "SARSA"

def make_experiment(agent_paths=["./"], 
					sublearning=False,
					exp_id=3,
					path="./Results/Confidence2/",
					temp=0.10517212721767522,
					discount_factor=0.7,
					lambda_=0.0,
					init_state=None):
	opt = {}
	opt["path"] = os.path.join(path, get_time_str())
	opt["exp_id"] = exp_id
	opt["max_steps"] = 40000
	opt["num_policy_checks"] = 10
	opt["checks_per_policy"] = 20
	# start_at = np.array([4, 6])

	agents = load_all_agents(agent_paths, pretrained=True, load_confidence=True)

	for i, a in enumerate(agents):
		# import ipdb; ipdb.set_trace()
		a.policy.epsilon = i * 0.02 + 0.1
		# a.learn_rate_decay_mode = 'boyan'
		# a.learn_rate = a.initial_learn_rate = 0.9
		# a.boyan_N0 = 3000
		a.learn_rate_decay_mode = 'dabney'

	domain = RCCarModified(noise=0.1, init_state = (-2, 0.8, 0, 2.5))
	representation = IncrementalTabular(domain)
	policy = MultiAgentConfidence(representation, agents, tau=.1)
	print "$" * 10
	print "You are currently running {}".format(policy.__class__)
	opt['agent'] = MetaAgent(representation=representation,
							 policy=policy,
							 learn_rate_decay_mode="const")
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