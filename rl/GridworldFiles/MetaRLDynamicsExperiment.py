from AgentGenerator import make_experiment
from Domains import GridWorldMixed, PolicyMixer
# from Policies import MultiTerrainPolicy, MultiTerrainQPolicy, MAgentMultinomial
from rlpy.Policies import eGreedy
from rlpy.Agents import Q_Learning
from Agents import *
from utils import get_time_str, load_all_agents
import os
import inspect, shutil
import numpy as np
from rlpy.Representations import IncrementalTabular, Tabular
from rlpy.Experiments import Experiment
import logging

mapname = '11x11-Rooms3.txt'

def generate_meta_experiment(exp_id, agent_paths, path, unique=True, expdomain=None):
	opt = {}
	if unique:
		opt["path"] = os.path.join(path, get_time_str())
	else:
		opt["path"] = path

	opt["exp_id"] = exp_id
	opt["max_steps"] = 10000
	opt["num_policy_checks"] = 40
	opt["checks_per_policy"] = 20
	# start_at = np.array([4, 6])


	agents = load_all_agents(agent_paths, pretrained=True, load_confidence=True)
	for a in agents:
		a.policy.epsilon = 0
	if expdomain:
		actual_domain = expdomain(mapname=mapname, terrain_augmentation=False, noise=0.1)
	else:
		actual_domain = GridWorldMixed(mapname=mapname, terrain_augmentation=False, noise=0.1)
	domain = PolicyMixer(actual_domain, agents)
	representation = Tabular(domain)
	policy = eGreedy(representation) # , tau=.1)
	opt['agent'] = Q_Learning(policy, representation, discount_factor=0.9, 
							initial_learn_rate=0.8,
							 lambda_=0.5, 
							 learn_rate_decay_mode= 'boyan', 
							 boyan_N0= 2380)
	opt['domain'] = domain
	experiment = Experiment(**opt)
	
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

def train_agent(param_folder, agent_number, path="./Results/Mixed_Actions3/"):
	result_path = path + "agent{}".format(agent_number)

	exp0 = make_experiment(param_folder, yaml_file="agent{}.yml".format(agent_number), 
							result_path=result_path, save=True) #TODO
	assert mapname in exp0.domain.mapname, "Not using correct map!"
	exp0.run()
	representation = exp0.agent.representation
	representation.dump_to_directory(exp0.full_path) # TODO
	return exp0.full_path



############
def run_2_agents(num=1):
	from Domains import GridWorld2
	# agent_paths = []
	# for i in range(4):
	# 	p = train_agent("params/gridworld/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_Actions3/agent0/Aug02_11-11-650788', './Results/Mixed_Actions3/agent1/Aug02_11-11-866788'] #, './Results/Mixed_Actions3/agent2/Aug02_11-11-932882', './Results/Mixed_Actions3/agent3/Aug02_11-12-741840']
	result_path = "./Results/Mixed2/combine/" + get_time_str()
	for i in range(1, 1+ num):
		exp = generate_meta_experiment(i, agent_paths, result_path, expdomain=GridWorld2, unique=False)
		exp.run(visualize_performance=0, debug_on_sigurg=True)
		exp.save()
	print result_path



def run_3_agents(num=1):
	from Domains import GridWorld3
	# agent_paths = []
	# for i in range(4):
	# 	p = train_agent("params/gridworld/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_Actions3/agent0/Aug02_11-11-650788', './Results/Mixed_Actions3/agent1/Aug02_11-11-866788','./Results/Mixed_Actions3/agent2/Aug02_11-11-932882'] #, './Results/Mixed_Actions3/agent3/Aug02_11-12-741840']
	result_path = "./Results/Mixed3/combine/" + get_time_str()
	for i in range(1, 1 + num):
		exp = generate_meta_experiment(i, agent_paths, result_path, expdomain=GridWorld3, unique=False)
		exp.run(visualize_performance=0, debug_on_sigurg=True)
		exp.save()
	print result_path

def run_4_agents(num=1):
	from Domains import GridWorld4
	# agent_paths = []
	# for i in range(4):
	# 	p = train_agent("params/gridworld/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_Actions3/agent0/Aug02_11-11-650788', './Results/Mixed_Actions3/agent1/Aug02_11-11-866788', './Results/Mixed_Actions3/agent2/Aug02_11-11-932882', './Results/Mixed_Actions3/agent3/Aug02_11-12-741840']
	result_path = "./Results/Mixed4/combine/" + get_time_str()
	for i in range(1, 1 + num):
		exp = generate_meta_experiment(i, agent_paths, result_path, expdomain=GridWorld4, unique=False)
		exp.run(visualize_steps=0, debug_on_sigurg=True)
		exp.save()

	print result_path

def run_6_agents():
	from Domains import GridWorld6
	# agent_paths = ['./Results/Mixed_Actions3/agent0/Aug02_11-11-650788', './Results/Mixed_Actions3/agent1/Aug02_11-11-866788', './Results/Mixed_Actions3/agent2/Aug02_11-11-932882', './Results/Mixed_Actions3/agent3/Aug02_11-12-741840']
	# for i in range(4,6):
	# 	p = train_agent("params/gridworld/", i, path="./Results/Mixed6/")
	# 	agent_paths.append(p)
	# print agent_paths
	agent_paths = ['./Results/Mixed_Actions3/agent0/Aug02_11-11-650788', './Results/Mixed_Actions3/agent1/Aug02_11-11-866788', 
				'./Results/Mixed_Actions3/agent2/Aug02_11-11-932882', './Results/Mixed_Actions3/agent3/Aug02_11-12-741840', 
				'./Results/Mixed6/agent4/Aug02_11-30-202414', './Results/Mixed6/agent5/Aug02_11-30-991817']
	exp = generate_meta_experiment(2, agent_paths, "./Results/Mixed6/combine", expdomain=GridWorld6)
	exp.run(visualize_performance=0, debug_on_sigurg=True)



if __name__ == '__main__':
	run_2_agents(5)

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
