# from rllab.algos.cem import CEM
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv

import numpy as np
import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")

# import Domains.RCCarModified as RCCarModified
from Policies import PolicyLoader
from Policies import ModifiedGibbs

from Domains import RCCarSlideTurn, RCCarBarriers
import Domains


# sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/GymEnvs")
from GymEnvs import RLPyEnv
from GymEnvs import HRLEnv

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.misc.instrument import stub, run_experiment_lite
import matplotlib.pyplot as plt

stub(globals())
import datetime, os

def rollout(env, policy, N=10):
    for _ in xrange(N):

        observations = []
        actions = []
        rewards = []

        observation = env.reset()

        for _ in xrange(T):
            # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contains
            # sufficient statistics for the action distribution. It should at least contain entries that would be
            # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
            # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
            # not needed.
            env.domain.slips = []
            env.render()
            action, _ = policy.get_action(observation)
            # print action            
            # print _
            # action = policy.action_space.sample()
            # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
            # case it is not needed.
            next_observation, reward, terminal, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = next_observation
            # totalobvs.append(observation)
            if terminal:
                # Finish rollout if terminal state reached
                break
        x_list, y_list = zip(*[(obs[0], obs[1]) for obs in observations])
        print observations
        print actions
        print rewards, sum(rewards)
        # plt.figure(1)
        plt.plot(x_list, y_list, 'x')
        plt.show()
        import ipdb; ipdb.set_trace()  # breakpoint 1d0ad163 //

        print sum(rewards)


T = 1000
# policies = [PolicyLoader("/home/jarvis/work/clipper/models/rl/" +path) for path in ['Results/Mixed_ActionsB/agent0/Aug21_11-38-389943', 
#                                                                                     'Results/Mixed_ActionsB/agent1/Aug21_11-43-003799', 
#                                                                                     # 'Results/Mixed_ActionsB/agent2/Aug16_03-50-036324',
#                                                                                     # 'Results/Mixed_ActionsB/agent2/Aug16_03-50-036324'
#                                                                                     ] ]
# policies = [PolicyLoader("models/noisy/" + path) for path in ['good'] ]

# stub(globals())
def good_inverted_car(directory="./Results/Car/Noisy_500/"):
    rccar = RCCarBarriers(noise=0.)
    policies = [PolicyLoader("models/noisy/" +path) for path in ['good','other'] ]
    domain = RLPyEnv(rccar)
    env = HRLEnv(domain, policies)
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=500,
        discount=0.9,
        step_size=0.001,
        # plot=True,
    )
    assert False, "Make sure to change logging directory before rerunning this experiment"

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        script="scripts/run_experiment_lite_rl.py",

        log_dir=directory,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )

def good_bad_car(num=1, directory="./Results/Car/CarNoisy/", exp_name="Noisy_untrained", save=True):

    rccar = RCCarBarriers(noise=0.1)
    policies = [PolicyLoader("models/noisy/" +path) for path in ['good','untrained'] ]
    domain = RLPyEnv(rccar)
    env = HRLEnv(domain, policies)
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64,32)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for i in range(num):
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=500,
            discount=0.9,
            step_size=0.0001,
            # plot=True,
        )

        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name + timestamp,
            log_dir=os.path.join(directory, exp_name + timestamp) if save else "./Results/Tmp",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )


def good_double_bad_car(num=1, directory="./Results/Car/CarNoisy/", exp_name="Noisy_2_untrained", save=False):
    rccar = RCCarBarriers(noise=0.1)
    policies = [PolicyLoader("models/noisy/" +path) for path in ['good','untrained', 'untrained'] ]
    domain = RLPyEnv(rccar)
    env = HRLEnv(domain, policies)
    # policy = CategoricalMLPPolicy(
    #     env_spec=env.spec,
    # )
    # baseline = LinearFeatureBaseline(env_spec=env.spec)
    # algo = TRPO(
    #     env=env,
    #     policy=policy,
    #     baseline=baseline,
    #     batch_size=4000,
    #     max_path_length=env.horizon,
    #     n_itr=500,
    #     discount=0.9,
    #     step_size=0.001,
    #     # plot=True,
    # )
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64,32)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for i in range(num):
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=600,
            discount=0.9,
            step_size=0.0001,
            # plot=True,
        )

        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name + timestamp,
            log_dir=os.path.join(directory, exp_name + timestamp) if save else "./Results/Tmp",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

def good_3_bad_car(num=5, directory="./Results/Car/CarNoisy/", exp_name="Noisy_3_untrained", save=False):
    # assert False, "Make sure to change directories!"
    rccar = RCCarBarriers(noise=0.1)
    policies = [PolicyLoader("models/noisy/" +path) for path in ['good','untrained', 'untrained', 'untrained'] ]
    domain = RLPyEnv(rccar)
    env = HRLEnv(domain, policies)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for i in range(1, num+1):
        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64,32)
        )
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=1000,
            discount=0.9,
            step_size=0.0001,
            # plot=True,
        )

        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name + timestamp,
            log_dir=os.path.join(directory, exp_name + timestamp) if save else "./Results/Tmp",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

def good_x_cars(num_agents=5, directory="./Final_Results/Car/CarNoisyAgents/", exp_name="NoisyTest", save=False):

    rccar = RCCarBarriers(noise=0.1)
    policies = [PolicyLoader("models/noisy/" +path) for path in ['good','untrained', 'untrained', 'untrained', 'untrained', 'untrained']][: 1 + num_agents]
    domain = RLPyEnv(rccar)
    env = HRLEnv(domain, policies)
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32,32)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    exp_name=exp_name + str(num_agents)
    directory = os.path.join(directory, exp_name)
    for i in range(3):
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2000,
            max_path_length=env.horizon,
            n_itr=500,
            discount=.995,
            step_size=0.001,
            # plot=True,
        )
        # algo.train()
        # rollout(env, policy)
        try:
            os.mkdir(directory)
        except Exception:
            pass
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name,
            log_dir=os.path.join(directory, timestamp) if save else "./Results/Tmp",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

if __name__ == '__main__':
    for i in range(1, 6):
        good_x_cars(num_agents=i, save=True)
    # good_bad_car(5, directory="./Results/Car/CarNoisy_v2",save=True)
    # good_double_bad_car(5, directory="./Results/Car/CarNoisy_v2", save=True)
    # good_3_bad_car(5, directory="./Results/Car/CarNoisy_v2", save=True)