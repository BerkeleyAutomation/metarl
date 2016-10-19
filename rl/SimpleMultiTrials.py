#!/usr/bin/env python

import rlpy
from rlpy.Tools import deltaT, clock, hhmmss
# from .. import visualize_trajectories as visual
import os
import yaml
import shutil
import inspect
import numpy as np
# from rlpy.CustomDomains import RCIRL, Encoding, allMarkovReward
import Domains
import Policies
import Representations
import Experiments

from datetime import datetime
import sys
import yaml
import pickle 

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        ret_val = yaml.load(f)
    return ret_val

def get_time_str():
    return datetime.now().strftime('%h%d_%I-%M-%f')

def make_experiment(param_root, 
                    yaml_file="params.yml", 
                    result_path='./GeneratedAgents/RCCarMaha/', save=False, other_domain=None):
    """
    :params param_root: location of parameter
    :params result_path: location of results/weights
    """

    param_path = os.path.join(param_root, yaml_file) 
    print "[INFO] Loading agent from " + param_path
    params = type("Parameters", (), load_yaml(param_path))
    if other_domain:
        domain = other_domain(**params.domain_params)
    else: 
        domain = eval(params.domain)(**params.domain_params)
    # domain = eval(params.domain)()

    #Load Representation
    representation = eval(params.representation)(
                domain, 
                **params.representation_params)
    policy = eval(params.policy)(
                representation, 
                **params.policy_params)
    agent = eval(params.agent)(
                policy, 
                representation,
                discount_factor=domain.discount_factor, 
                **params.agent_params)
    opt = {}
    opt["exp_id"] = params.exp_id if params.exp_id else np.random.randint(100)
    opt["path"] = result_path
    # opt["max_steps"] = 10000 #params.max_steps
    opt["max_episode"] = 4000
    # opt["num_policy_checks"] = params.num_policy_checks
    opt["num_policy_checks"] = 25

    opt["checks_per_policy"] = params.checks_per_policy
    opt["domain"] = domain
    opt["agent"] = agent

    return eval(params.experiment)(**opt)

def rerun_agent(agent_path, domain, n=5):
    ### After training, trying it on a new domain
    result_path = "./Results/Car/Agent1Post/" +  get_time_str()
    for i in range(n):
        exp = make_experiment(agent_path, yaml_file="params.yml", result_path=result_path, other_domain=domain)
        exp.agent.representation.load_from_directory(agent_path)
        exp.agent.policy.epsilon = 0
        exp.agent.initial_learn_rate = 0

        exp.run(debug_on_sigurg=True, visualize_steps=1)
        exp.save()
    # change domain
    return result_path

def run_many_from_scratch(n=10):
    param_root = "params/Car/"
    yaml_file = "agent1.yml"
    result_path = "./Results/Car/Agent1/" +  get_time_str()
    for i in range(n):
        exp = make_experiment(param_root, yaml_file=yaml_file, result_path=result_path)

        exp.run(visualize_performance=0)
        # exp.plot()
        exp.save()
    print result_path



if __name__ == '__main__':
    # run_many_from_scratch(n=5)
    agent_paths = ['./Results/Mixed_ActionsB/agent0/Aug21_10-20-139246', './Results/Mixed_ActionsB/agent1/Aug21_10-21-223945']

    res = []
    for paths in agent_paths[1:]:
        s = rerun_agent(paths, Domains.RCCarRightTurn)
        break
        res.append(s)

    print res