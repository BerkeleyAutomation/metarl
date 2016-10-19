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
                    result_path='./GeneratedAgents/RCCarMaha/', save=False):
    """
    :params param_root: location of parameter
    :params result_path: location of results/weights
    """

    param_path = os.path.join(param_root, yaml_file) 
    print "[INFO] Loading agent from " + param_path
    params = type("Parameters", (), load_yaml(param_path))
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
    opt["max_steps"] = params.max_steps
    opt["num_policy_checks"] = params.num_policy_checks
    opt["checks_per_policy"] = params.checks_per_policy
    opt["domain"] = domain
    opt["agent"] = agent

    return eval(params.experiment)(**opt)

def rerun_agent(agent_path):
    pass

def run_many(param_root, yaml_file, result_path, n=10):
    for i in range(n):
        exp = make_experiment(param_root, yaml_file=yaml_file, result_path=result_path)

        exp.run(visualize_performance=1)
        exp.plot()
        exp.save()


if __name__ == '__main__':
    param_root = "params/gridworld/"
    yaml_file = "individ_params.yml"
    result_path = "./Results/Gridworld/Agent0/" +  get_time_str()
    run_many(param_root, yaml_file, result_path)
