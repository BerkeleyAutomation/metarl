import sys; sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")
from hyperopt import hp
# from rlpy.Representations import KernelizediFDD
from Domains import *
# from Policies import MultiTerrainPolicy, MultiTerrainQPolicy, MAgentMultinomial
from rlpy.Policies import eGreedy, GibbsPolicy
from rlpy.Agents import Q_Learning
from Agents import *
from utils import get_time_str, load_all_agents
import os
import inspect, shutil
import numpy as np
from Representations import *
# from Experiments import *
from Policies import *
from rlpy.Experiments import Experiment
import Domains

param_space = {
    "num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(1e4), 1),
    'resolution': hp.quniform("resolution", 3, 30, 1),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}

def make_experiment(exp_id=1, path="./Results/Temp/MetaHP/", unique=False, max_episode=300, 
		num_rbfs=4000,
		initial_learn_rate=0.9,
		lambda_=0.7,
		resolution=20):
	opt = {}
	if unique:
		opt["path"] = os.path.join(path, get_time_str())
	else:
		opt["path"] = path

	opt["exp_id"] = exp_id
	opt["max_steps"] = 30000
	# opt["max_episode"] = max_episode
	opt["num_policy_checks"] = 10
	opt["checks_per_policy"] = 1
	# start_at = np.array([4, 6])
	from Domains import RCCarSlideTurn

	expdomain = Domains.RCCarSlideTurn
	agent_paths =['/home/jarvis/work/clipper/models/rl/Results/Mixed_ActionsB/agent0/Aug21_11-38-389943', 
				'/home/jarvis/work/clipper/models/rl/Results/Mixed_ActionsB/agent1/Aug21_11-43-003799']

	agents = load_all_agents(agent_paths, pretrained=True, load_confidence=False)
	for a in agents:
		a.policy.epsilon = 0
		# a.policy.turnOffExploration()
	if expdomain:
		actual_domain = expdomain(noise=0.)
	else:
		actual_domain = RCCarModified(noise=0.1)
	domain = RCPolicyMixer(actual_domain, agents)
	representation = MahaRBF(domain, 
						num_rbfs=int(num_rbfs),
						# state_dimensions=np.array([0,1,3]), 
						const_feature=False, 
						resolution_min=resolution,
						resolution_max=resolution,
						include_border=True,
						normalize=True,
						seed=exp_id)
	policy = eGreedy(representation) #, epsilon=0.1) # , tau=.1)
	opt['agent'] = Q_Learning(policy, representation, discount_factor=0.8, 
							initial_learn_rate=initial_learn_rate,
							 lambda_=lambda_, 
							 learn_rate_decay_mode= 'const')
	opt['domain'] = domain
	experiment = Experiment(**opt)
	
	# path_join = lambda s: os.path.join(opt["path"], s)
	# if not os.path.exists(opt["path"]):
	# 	os.makedirs(opt["path"])

	# shutil.copy(inspect.getsourcefile(inspect.currentframe()), path_join("experiment.py"))

	return experiment

# if __name__ == '__main__':
"""
	from rlpy.Tools.hypersearch import find_hyperparameters
	best, trials = find_hyperparameters(
	    "./MetaRCHPOpt.py",
	    "./Results/RBFs_hypersearch",
	    max_evals=20, parallelization="joblib",
	    trials_per_point=10)
	print best
	"""