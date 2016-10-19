from AgentGenerator import make_experiment
from Domains import *
# from Policies import MultiTerrainPolicy, MultiTerrainQPolicy, MAgentMultinomial
from rlpy.Policies import eGreedy, GibbsPolicy
from rlpy.Agents import Q_Learning, SARSA
from Agents import *
from utils import get_time_str, load_all_agents
import os
import inspect, shutil
import numpy as np
from Representations import *
from Experiments import *
from Policies import *
from rlpy.Representations import IncrementalTabular, Tabular, Fourier
from rlpy.Representations.LocalBases import NonparametricLocalBases, RandomLocalBases
from rlpy.Representations.kernels import *
from rlpy.Experiments import Experiment
import logging
import joblib

mapname = '11x11-Rooms3.txt'
{'initial_learn_rate': 0.3414408997566183, 'lambda_': 0.38802888678400627, 'resolution': 21.0, 'num_rbfs': 6988.0}


def generate_meta_experiment(exp_id, agent_paths, path, unique=True, expdomain=None, max_episode=5000):
	opt = {}
	if unique:
		opt["path"] = os.path.join(path, get_time_str())
	else:
		opt["path"] = path

	opt["exp_id"] = exp_id
	# opt["max_steps"] = 50000
	opt["max_episode"] = max_episode
	opt["num_policy_checks"] = 50
	opt["checks_per_policy"] = 1
	# start_at = np.array([4, 6])


	agents = load_all_agents(agent_paths, pretrained=True, load_confidence=False)
	for a in agents:
		a.policy.epsilon = 0
		a.policy.turnOffExploration()
	if expdomain:
		actual_domain = expdomain(noise=0.1)
	else:
		actual_domain = RCCarModified(noise=0.1)
	domain = RCPolicyMixer(actual_domain, agents)
	# representation = MahaRBF(domain, 
	# 					num_rbfs=3000,
	# 					# state_dimensions=np.array([0,1,3]), 
	# 					const_feature=False, 
	# 					resolution_min=21,
	# 					resolution_max=21,
	# 					include_border=True,
	# 					normalize=True,
	# 					seed=exp_id)
	# representation = RandomLocalBases(domain, gaussian_kernel,
	# 					 				num=100,
	# 									normalization=True,  
	# 									resolution_max=20, 
	# 									seed=exp_id)
	representation = NonparametricLocalBases(domain, gaussian_kernel, normalization=True)
	policy = eGreedy(representation, 0.05) #, epsilon=0.1) # , tau=.1)
	# policy = GibbsPolicy(representation)
	opt['agent'] = Q_Learning(policy, representation, discount_factor=0.8, 
							initial_learn_rate=.8,
							 lambda_=0.1, 
							 learn_rate_decay_mode= 'boyan', 
							 boyan_N0=500)
	opt['domain'] = domain
	experiment = ExperimentMod(**opt)
	
	
	path_join = lambda s: os.path.join(opt["path"], s)
	if not os.path.exists(opt["path"]):
		os.makedirs(opt["path"])

	shutil.copy(inspect.getsourcefile(inspect.currentframe()), path_join("experiment.py"))

	return experiment

def run_exp_trials(exp_generator, agent_paths, path, num=10, **kwargs):
	for i in range(1, num + 1):
		experiment = exp_generator(i, agent_paths, path, unique=False)
		experiment.run(visualize_performance=1, **kwargs)
		experiment.save()
	return path

def train_agent(param_folder, agent_number, path="./Results/Mixed_ActionsB/"):
	result_path = path + "agent{}".format(agent_number)

	exp0 = make_experiment(param_folder, yaml_file="agent{}.yml".format(agent_number), 
							result_path=result_path, save=True) #TODO
	exp0.run()
	# representation = exp0.agent.representation
	# representation.dump_to_directory(exp0.full_path) # TODO
	return exp0.full_path



############
def run_2_agents(num=1):
	from Domains import RCCar2

	agent_paths = []
	for i in range(2):
		p = train_agent("params/Car/", i)
		agent_paths.append(p)
	print agent_paths

	# agent_paths = ['./Results/Mixed_Actions3/agent0/Aug08_04-26-747713', './Results/Mixed_Actions3/agent1/Aug08_04-26-027814']

	# result_path = "./Results/CarMixed2/combine/" + get_time_str()
	# for i in range(1, 1+ num):
	# 	exp = generate_meta_experiment(i, agent_paths, result_path, expdomain=RCCar2, unique=False)
	# 	exp.run(visualize_steps=0, debug_on_sigurg=True)
	# 	# exp.plot()
	# 	exp.save()
	# print result_path

	# dom = exp.domain
	# joblib.dump(dom.statistics, os.path.join(result_path, "runs.pk"))

def run_2_agents_easy(num=1):
	"""one trained agent, one non-trained agent
		opt['domain'] = domain
	"""
	from Domains import RCCarBarriers

	# agent_paths = []
	# for i in range(3):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_ActionsB/agent0/Aug16_03-49-898192', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', ]

	result_path = "./Results/CarMixed2/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths[:1], result_path, expdomain=RCCarBarriers, 
			unique=False, max_episode=2000)
		exp.run(visualize_steps=0, visualize_performance=0, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path
	dom = exp.domain
	return result_path


# Good result!
def run_2_agents_barriers(num=1):
	from Domains import RCCarBarrier_2

	# agent_paths = []
	# for i in range(3):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_ActionsB/agent0/Aug16_03-49-898192', 
					'./Results/Mixed_ActionsB/agent1/Aug16_03-50-596823', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324']

	result_path = "./Results/CarMixed2/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths[:2], result_path, expdomain=RCCarBarrier_2, unique=False)
		exp.run(visualize_steps=0, visualize_performance=1, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

	dom = exp.domain
	return result_path
	# joblib.dump(dom.statistics, os.path.join(result_path, "runs.pk"))

def run_3_agents_barriers(num=1):
	from Domains import RCCarBarrier_2

	# agent_paths = []
	# for i in range(3):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_ActionsB/agent0/Aug16_03-49-898192', 
					'./Results/Mixed_ActionsB/agent1/Aug16_03-50-596823', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', #untrained
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324']

	result_path = "./Results/CarMixed2/combine/" + get_time_str()
	for i in range(2, 2+ num):
		exp = generate_meta_experiment(i, agent_paths[:3], result_path, expdomain=RCCarBarrier_2, unique=False)
		exp.run(visualize_steps=0, visualize_performance=1, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

	dom = exp.domain
	return result_path

def run_2_agents_barriers_2(num=1):
	from Domains import RCCarBarrier_2

	# agent_paths = []
	# for i in range(3):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_ActionsB/agent0/Aug16_03-49-898192', 
					'./Results/Mixed_ActionsB/agent1/Aug16_03-50-596823', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324']

	result_path = "./Results/CarMixed2/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths[:4], result_path, expdomain=RCCarBarrier_2, unique=False)
		exp.run(visualize_steps=0, visualize_performance=0, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

	dom = exp.domain
	return result_path

def run_2_agents_barriers_3(num=1):
	from Domains import RCCarBarrier_2

	# agent_paths = []
	# for i in range(3):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_ActionsB/agent0/Aug16_03-49-898192', 
					'./Results/Mixed_ActionsB/agent1/Aug16_03-50-596823', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324', 
					'./Results/Mixed_ActionsB/agent2/Aug16_03-50-036324']

	result_path = "./Results/CarMixed2/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths, result_path, expdomain=RCCarBarrier_2, unique=False)
		exp.run(visualize_steps=0, visualize_performance=0, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

	dom = exp.domain
	return result_path

def run_2_slideturn_slide(num=1):
	from Domains import RCCarLeft#, RCCarSlideInvert
	# agent_paths = []
	# for i in range(2):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths =['./Results/Mixed_ActionsB/agent0/Aug21_11-38-389943', './Results/Mixed_ActionsB/agent1/Aug21_11-43-003799']

	result_path = "./Results/CarSlideTurn/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths, result_path, max_episode=500, expdomain=RCCarLeft, unique=False)
		exp.run(visualize_steps=0, visualize_performance=0, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

	# dom = exp.domain
	# return result_path


def run_2_slideturn_turn(num=1):
	# agent_paths = []
	# for i in range(2):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths =['./Results/Mixed_ActionsB/agent0/Aug21_11-38-389943', './Results/Mixed_ActionsB/agent1/Aug21_11-43-003799']

	result_path = "./Results/CarSlideTurn/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths, result_path, max_episode=1000, expdomain=RCCarRightTurn, unique=False)
		exp.run(visualize_steps=0, visualize_performance=0, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

# Good result!
def run_2_slideturn_mixed(num=1):
	"""One agent trained on RightTurn, other on LeftSlide - comparing on stitched domain"""
	from Domains import RCCarSlideTurn#, RCCarSlideInvert
	# agent_paths = []
	# for i in range(2):
	# 	p = train_agent("params/Car/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths =['./Results/Mixed_ActionsB/agent0/Aug21_11-38-389943', 
					'./Results/Mixed_ActionsB/agent1/Aug21_11-43-003799']

	result_path = "./Results/CarSlideTurn/localrbf/increasedres/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths, result_path, max_episode=1000, expdomain=RCCarSlideTurn, unique=False)
		exp.run(visualize_steps=0, visualize_performance=0, debug_on_sigurg=True)
		# exp.plot()
		exp.save()
	print result_path

if __name__ == '__main__':
	results = []
	s = run_2_agents_easy(1)
	# run_2_slideturn_mixed(1)
	# results.append(s)
	# # s = run_2_agents_barriers_1(5)
	# # results.append(s)
	# # s = run_2_agents_barriers_2(5)
	# # results.append(s)
	# s = run_2_agents_barriers_3(10)
	# results.append(s)
	# print results
	# exp0 = make_experiment("params/gridworld/", yaml_file="agent0.yml", 
	# 						result_path="./Results/Mixed/agent0", save=True) #TODO
	# assert mapname in exp0.domain.mapname, "Not using correct map!"
	# exp0.run(visualize_performance=0)
	# representation = exp0.agent.representation
	# representation.dump_to_directory(exp0.full_path) # TODO


	# result_path1 = "./Results/Mixed/agent1"
	# exp1 = make_experiment("params/gridworld/", yaml_file="agent1.yml", result_path=result_path1, save=True)
	# assert mapname in exp1.domain.mapname, "Not using correct map!"
	# exp1.run(visualize_performance=0)
	# representation = exp1.agent.representation
	# representation.dump_to_directory(exp1.full_path)



	# agent_paths = [exp0.full_path, exp1.full_path]
	# print agent_paths

	# agent_paths4 =  ['./Results/Mixed_Actions3/agent0/Aug02_05-53-219522', 
	# 	'./Results/Mixed_Actions3/agent1/Aug02_05-53-805251', 
	# 	'./Results/Mixed_Actions3/agent2/Aug02_05-53-439359', 
	# 	'./Results/Mixed_Actions3/agent3/Aug02_05-53-025208']

	# result_path = "./Results/Meta/3agentmaze/"
	# # FOR MANY TRIALS
	# path = run_exp_trials(generate_meta_experiment, agent_paths4, result_path + get_time_str())
	# print path

	# exp = generate_meta_experiment(1, agent_paths4, result_path, unique=True)
	# exp.run(visualize_performance=0, debug_on_sigurg=True)
	# exp.save()
	# exp.plot()
	# import joblib
