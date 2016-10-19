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
import Experiments
import Representations
import Agents

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



{'initial_learn_rate': 0.19731292271762133, 
'lambda_': 0.5723135366314989, 
'resolution': 6.0, 'num_rbfs': 1724.0}

def make_experiment(param_root, 
                    yaml_file="params.yml", 
                    result_path='./GeneratedAgents/RCCarSlide/', save=False):
    """
    :params param_root: location of parameter
    :params result_path: location of results/weights
    """

    param_path = os.path.join(param_root, yaml_file) 
    print "[INFO] Loading agent from " + param_path
    params = type("Parameters", (), load_yaml(param_path))

    # params.domain_params['mapname'] = os.path.join(os.path.expanduser(params.map_dir),
    #                                                 params.domain_params['mapname'])
    # import ipdb; ipdb.set_trace()

    domain = eval(params.domain)(**params.domain_params)
    # domain = eval(params.domain)()

    #Load Representation
    if params.representation == "rlpy.Representations.RandomLocalBases":
        representation = eval(params.representation)(
                    domain, 
                    eval(params.representation_kernel),
                    **params.representation_params)
    else:
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
    opt["exp_id"] = params.exp_id if params.exp_id else np.random.randint(100) + 1
    # print "using seed of %d" % opt["exp_id"]
    # import ipdb; ipdb.set_trace()
    opt["path"] = os.path.join(result_path, get_time_str())
    # opt["max_eps"] = params.max_eps

    opt["max_episode"] = params.max_episode
    opt["max_steps"] = params.max_steps
    opt["num_policy_checks"] = params.num_policy_checks
    opt["checks_per_policy"] = params.checks_per_policy

    opt["domain"] = domain
    opt["agent"] = agent

    if save:

        path_join = lambda s: os.path.join(opt["path"], s)
        if not os.path.exists(opt["path"]):
            os.makedirs(opt["path"])

        shutil.copy(param_path, path_join("params.yml"))
        shutil.copy(inspect.getsourcefile(eval(params.domain)), path_join("domain.py"))
        shutil.copy(inspect.getsourcefile(inspect.currentframe()), path_join("experiment.py"))
        

    return eval(params.experiment)(**opt)


if __name__ == '__main__':
    experiment = make_experiment(sys.argv[1], yaml_file="params1.yml", save=True)

    final_path = experiment.path
    experiment.run(visualize_steps=0,  # should each learning step be shown?
                   visualize_learning=False,
                    debug_on_sigurg=True,
                   visualize_performance=0)  # show policy / value function?
                   # saveTrajectories=False)  # show performance runs


    print final_path
    # experiment.domain.showLearning(experiment.agent.representation)

    # experiment.plotTrials(save=True)
    # experiment.plot(save=True, x = "learning_episode") #, y="reward")
    representation = experiment.agent.representation
    experiment.save()
    experiment.plot()
