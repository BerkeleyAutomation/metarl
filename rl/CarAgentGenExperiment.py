#!/usr/bin/env python
import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")
from Domains import RCCarLeftTurn
from rlpy.Domains import GridWorld, RCCar
from rlpy.CustomDomains import RCCarModified
from rlpy.Agents import LSPI, Q_Learning, SARSA, NaturalActorCritic
from rlpy.Representations import Tabular, RBF
from rlpy.Policies import eGreedy, GibbsPolicy
from rlpy.Experiments import Experiment
from rlpy.Representations.LocalBases import NonparametricLocalBases, RandomLocalBases
from rlpy.Representations.kernels import *
# from GridWorldModified import GridWorldModified
# from Domains import *
# # print Domains
# RCCarModified = Domains.RCCarModified
import os
# import joblib 
import numpy as np
from hyperopt import hp

param_space = {
    "num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(1e4), 1),
    'resolution': hp.quniform("resolution", 3, 30, 1),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}

def make_experiment(exp_id=1,
        path="./Results/Temp", 
        initial_learn_rate=.40,
        lambda_=0.,
        resolution=25, num_rbfs=300):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.
    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    # import sys
    # import os 
    # cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
    # sys.path.append(cur_dir)
    # from Domains import RCCarModified
    # from Policies import RCCarGreedy

    # Experiment variables
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 200000
    opt["num_policy_checks"] = 15
    opt["checks_per_policy"] = 2
    # Logging

    domain = RCCarLeftTurn(noise=0.)
    opt["domain"] = domain

    # Representation
    kernel = gaussian_kernel
    representation = RandomLocalBases(domain, gaussian_kernel,
                                        num=int(num_rbfs),
                                        normalization=True,  
                                        resolution_max=resolution, 
                                        seed=exp_id)

    policy = eGreedy(representation, epsilon=0.15)
    # if biasedaction > -1:
    #     print "No Random starts with biasing {}".format(i % 4)
    #     policy = BiasedGreedy(representation, epsilon=0.5, biasedaction=biasedaction)

    # Agent

    opt["agent"] = Q_Learning(policy, representation, domain.discount_factor,
        initial_learn_rate=initial_learn_rate,
        lambda_=lambda_,
        learn_rate_decay_mode="const")

    experiment = Experiment(**opt)


    return experiment

if __name__ == '__main__':
    path = "./Results/Temp_car/{domain}/{agent}/{representation}/"


    experiment = make_experiment(path=path)
    # import ipdb; ipdb.set_trace()

    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  
                   visualize_performance=False) 
