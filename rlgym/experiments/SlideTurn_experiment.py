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

from Domains import * # RCCarSlideTurn, RCCarTurn_SWITCH, RCCarSlide_SWITCH
import Domains


# sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/GymEnvs")
from GymEnvs import RLPyEnv
from GymEnvs import HRLEnv

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())
import os, datetime
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
            env.render()
            action, _ = policy.get_action(observation)
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

        print sum(actions) * 1. / len(actions)


T = 1000

def slideturn_mixed(num=1, directory="./Final_Results/Car/SlideTurn/", exp_name="Mixed", save=False):
    policies = [PolicyLoader("models/slideturn_experiment/" + path) for path in ['agent0','agent1'] ]
    directory = os.path.join(directory, exp_name)
    for i in range(num):
        rccar = RCCarSlideTurn(noise=0.1)
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

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
            batch_size=1000,
            max_path_length=env.horizon,
            n_itr=500 + 1,
            discount=0.995,
            step_size=0.001,
            # plot=True,
        )
        # algo.train()
        # rollout(env, policy)
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name,
            log_dir=os.path.join(directory, timestamp) if save else './Results/Tmp',
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

def slideturn_slide_only(num=1, directory="./Final_Results/Car/SlideTurn/", exp_name="Slide", save=True):
    policies = [PolicyLoader("models/slideturn_experiment/" + path) for path in ['agent0'] ]
    directory = os.path.join(directory, exp_name)
    for i in range(num):
        rccar = RCCarSlideTurn(noise=0.1)
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

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
            batch_size=1000,
            max_path_length=env.horizon,
            n_itr=500 + 2,
            discount=0.995,
            step_size=0.001,
            # plot=True,
        )
        # algo.train()
        # rollout(env, policy)
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name,
            log_dir=os.path.join(directory, timestamp) if save else './Results/Tmp',
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

def slideturn_turn_only(num=1, directory="./Final_Results/Car/SlideTurn/", exp_name="Turn", save=True):
    policies = [PolicyLoader("models/slideturn_experiment/" + path) for path in ['agent1'] ]
    directory = os.path.join(directory, exp_name)
    for i in range(num):
        rccar = RCCarSlideTurn(noise=0.1)
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

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
            batch_size=1000,
            max_path_length=env.horizon,
            n_itr=500 + 3,
            discount=0.995,
            step_size=0.001,
            # plot=True,
        )
        # algo.train()
        # rollout(env, policy)
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name,
            log_dir=os.path.join(directory, timestamp) if save else './Results/Tmp',
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

def slideturn_from_scratch(num=1, directory="./Final_Results/Car/SlideTurn/", exp_name="Base_tmp", save=False):
    rccar = RCCarSlideTurn(noise=0.1)
    env =  RLPyEnv(rccar)
    dir_name = os.path.join(directory, exp_name)
    for i in range(num):
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1000,
            max_path_length=env.horizon,
            n_itr=500,
            discount=0.995,
            step_size=0.001,
            # plot=True,
        )
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name + str(i),
            log_dir=os.path.join(dir_name, timestamp) if save else './Results/Tmp',
            # plot=True,
        )

def generate_turn_model(num=1, directory="./Results/Car/Turn/", exp_name="Base", save=False):
    rccar = RCCarRightTurn(noise=0.)
    env =  RLPyEnv(rccar)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=8000,
        max_path_length=env.horizon,
        n_itr=200,
        discount=0.9,
        step_size=0.01,
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
        log_dir=os.path.join(directory, exp_name) if save else './Results/Tmp',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )

def generate_slide_model(num=1, directory="./Results/Car/Slide/", exp_name="Base", save=False):
    rccar = RCCarLeft(noise=0.)
    env =  RLPyEnv(rccar)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=8000,
        max_path_length=env.horizon,
        n_itr=200,
        discount=0.9,
        step_size=0.01,
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
        log_dir=os.path.join(directory, exp_name) if save else './Results/Tmp',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )

def switch_from_turn(exp_name="Switch_Turn", num=1, directory="./Results/Car/SlideTurn/", save=True):
    """Environment begins with Turn bias, switches to mixed after 1e5 calls to the step function"""
    rccar = RCCarSlideTurn(noise=0.1)
    env =  RLPyEnv(rccar)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    import joblib
    data = joblib.load("Results/Car/Turn/Base/params.pkl") # LOAD POLICY
    policy = data['policy']
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=100,
        discount=0.9,
        step_size=0.01,
        # plot=True,
    )
    # algo.train()
    # rollout(env, policy)

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        script="scripts/run_experiment_lite_rl.py",
        exp_name=exp_name + timestamp,
        log_dir=os.path.join(directory, exp_name) if save else './Results/Tmp',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )

def switch_from_slide(exp_name="Switch_Slide", num=1, directory="./Results/Car/SlideTurn/", save=True):
    rccar = RCCarSlideTurn(noise=0.1)
    env =  RLPyEnv(rccar)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    import joblib
    data = joblib.load("Results/Car/Slide/Base/params.pkl") # LOAD POLICY
    policy = data['policy']
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=100,
        discount=0.9,
        step_size=0.01,
        # plot=True,
    )
    # algo.train()
    # rollout(env, policy)

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        script="scripts/run_experiment_lite_rl.py",
        exp_name=exp_name + timestamp,
        log_dir=os.path.join(directory, exp_name) if save else './Results/Tmp',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )


if __name__ == '__main__':
    slideturn_from_scratch(num=5, exp_name="base_tmp", save=True)
    # slideturn_mixed(num=20, save=True)
    # slideturn_slide_only(num=20, save=True)
    # slideturn_turn_only(num=20, save=True)