from AgentGenerator import make_experiment
from Domains import GridWorldMixed
from Policies import MultiTerrainPolicy, MultiTerrainQPolicy, MAgentMultinomial
from Agents import *
from utils import get_time_str, load_all_agents
import os
import inspect, shutil
import numpy as np
from rlpy.Representations import IncrementalTabular
from rlpy.Experiments import Experiment

mapname = '11x11-Rooms3.txt'

def generate_multinomial_experiment(exp_id, agent_paths, path, unique=True, expdomain=None):
	opt = {}
	if unique:
		opt["path"] = os.path.join(path, get_time_str())
	else:
		opt["path"] = path

	opt["exp_id"] = exp_id
	opt["max_steps"] = 8000
	opt["num_policy_checks"] = 20
	opt["checks_per_policy"] = 1

	agents = load_all_agents(agent_paths, pretrained=True, load_confidence=True) 

	for a in agents:
		assert type(a.policy).__name__ == "DynamicsConfidenceGreedy"
		a.policy.epsilon = 0

	if expdomain:
		domain = expdomain(mapname=mapname, noise=0.1)
	else:
		domain = GridWorldMixed(mapname=mapname, noise=0.1)
	representation = IncrementalTabular(domain)
	policy = MAgentMultinomial(representation, agents) # , tau=.1)
	
	print "$" * 10
	print "You are currently running {}".format(policy.__class__)
	opt['agent'] = NoopAgent(representation=representation,
							 policy=policy)
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

#### #### #### #### #### #### #### #### #### #### 
#### #### #### #### EXPERIMENTS:#### #### #### 
#### #### #### #### #### #### #### #### #### #### 

def run_4_agents():
	from Domains import GridWorld4
	# agent_paths = []
	# for i in range(4):
	# 	p = train_agent("params/gridworld/", i)
	# 	agent_paths.append(p)
	# print agent_paths

	agent_paths = ['./Results/Mixed_Actions3/agent0/Aug02_10-59-195632', './Results/Mixed_Actions3/agent1/Aug02_11-00-449265', 
					'./Results/Mixed_Actions3/agent2/Aug02_11-00-925019', './Results/Mixed_Actions3/agent3/Aug02_11-00-203614']
	exp = generate_multinomial_experiment(1, agent_paths, "./Results/Mixed4/combine", expdomain=GridWorld4)
	exp.run(visualize_performance=0, debug_on_sigurg=True)

def run_6_agents():
	agent_paths = []
	for i in range(8):
		p = train_agent("params/gridworld/", i, path="./Results/Mixed8/")
		agent_paths.append(p)
	print agent_paths

	exp = generate_multinomial_experiment(1, agent_paths, "./Results/Mixed8/combine")
	exp.run(visualize_performance=0, debug_on_sigurg=True)



if __name__ == '__main__':
	run_4_agents()

	# result_path2 = "./Results/Mixed_Actions3/agent2"
	# exp2 = make_experiment("params/gridworld/", yaml_file="agent2.yml", result_path=result_path2, save=True)
	# assert mapname in exp2.domain.mapname, "Not using correct map!"
	# exp2.run()
	# representation = exp2.agent.representation
	# representation.dump_to_directory(exp1.full_path)


	# agent_paths = [exp0.full_path, exp1.full_path]
	# print agent_paths # # run_exp_trials(generate_multinomial_experiment, agent_paths, "./Results/Mixed2/Easier11x11/" + get_time_str())