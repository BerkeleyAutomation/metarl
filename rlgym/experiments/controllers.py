from rllab.algos.cem import CEM
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv

import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")

import Domains.RCCarModified as RCCarModified
from Domains.RCCar_Experimental_2 import *
from Domains.Acrobot_Experimental import ModifiedAcrobot
from rlpy.Domains import Acrobot
from rllab.envs.gym_env import GymEnv

from rllab.misc.instrument import stub, run_experiment_lite

# sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/GymEnvs")
from GymEnvs import RLPyEnv, ControllerEnv, BandedControllerEnv, YBandedControllerEnv, StandardControllerEnv
# from GymEnvs import hrl_env
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy

stub(globals())
import os, datetime

def rollout(env, policy, N=1, force_act=None):
    import numpy as np

    for _ in xrange(N):
        observations = []
        actions = []
        rewards = []

        observation = env.reset()

        for _ in xrange(env.horizon):
            # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contains
            # sufficient statistics for the action distribution. It should at least contain entries that would be
            # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
            # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
            # not needed.
            env.render()
            action, _ = policy.get_action(observation)
            if force_act is not None:
                action = force_act
            observations.append(observation)
            actions.append(action)
            # action = policy.action_space.sample()
            # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
            # case it is not needed.
            next_observation, reward, terminal, _ = env.step(action)
            rewards.append(reward)
            observation = next_observation
            # totalobvs.append(observation)
            if terminal:
                observations.append(observation)
                # Finish rollout if terminal state reached
                break
        x_list, y_list = zip(*observations)
        print observations
        print actions
        print rewards, sum(rewards)
        plt.figure(1)
        plt.plot(x_list, y_list, 'x')
        plt.show()


# rl = ModifiedAcrobot()
# rl = RCCarBar_SlideLeft()
def experiment_multiple_agents():
    for k in [4, 6, 8, 10, 16]:
        for _ in range(2):
            env = ControllerEnv(k=k, noise=0.05, num_dynamics=4, num_points=12)
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32,32,),
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,
                max_path_length=env.horizon,
                n_itr=200,
                discount=0.995,
                step_size=0.001,
                plot=False,
            )
            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=4,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # script="scripts/run_experiment_lite_rl.py",
                script="scripts/run_experiment_lite.py",
                log_dir=os.path.join("Results/Controls4/", timestamp)
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                # plot=True,
            )

def experiment_compare_baselines():
    # for act in range(4):
        # env = ControllerEnv(k=4, noise=0.05, num_dynamics=4, num_points=12, force_act=act)
    env = ControllerEnv(k=4, noise=0.05, num_dynamics=4, num_points=12)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32,32,),
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        max_path_length=env.horizon,
        n_itr=100,
        discount=0.995,
        step_size=0.001,
        plot=False,
    )
    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # script="scripts/run_experiment_lite_rl.py",
        script="scripts/run_experiment_lite.py",
        log_dir=os.path.join("Results/Controls_Compare_Baseline/", timestamp)
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        # plot=True,
    )

def experiment_partitioning(itr=3):
    for bw in [1, 1.5, 2, 3]:
        for _ in range(itr):
            env = BandedControllerEnv(bands=bw, k=4, noise=0.05, num_dynamics=4)
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32,32,),
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,
                max_path_length=env.horizon,
                n_itr=100,
                discount=0.995,
                step_size=0.001,
                plot=False,
            )
            # algo.train()
            # rollout(env, policy)

            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=4,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # script="scripts/run_experiment_lite_rl.py",
                script="scripts/run_experiment_lite.py",
                log_dir=os.path.join("Results/Controls/Controls_Partitions/XBand", timestamp)
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                # plot=True,
            )

def experiment_ypartitioning(itr=3):
    for bw in [1, 1.5, 2, 3]:
        for _ in range(itr):
            env = YBandedControllerEnv(bands=bw, k=4, noise=0.05, num_dynamics=4)
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32,32,),
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,
                max_path_length=env.horizon,
                n_itr=100,
                discount=0.995,
                step_size=0.001,
                plot=False,
            )
            # algo.train()
            # rollout(env, policy)

            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=4,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # script="scripts/run_experiment_lite_rl.py",
                script="scripts/run_experiment_lite.py",
                log_dir=os.path.join("Results/Controls/Controls_Partitions/YBand", timestamp)
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                # plot=True,
            )

def experiment_increase_points(itr=3):
    # for act in range(4):
        # env = ControllerEnv(k=4, noise=0.05, num_dynamics=4, num_points=12, force_act=act)
    for k in [10, 30, 50, 100]:
        for _ in range(itr):
            env = ControllerEnv(k=4, noise=0.05, num_dynamics=4, num_points=k)
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32,32,),
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,
                max_path_length=env.horizon,
                n_itr=100,
                discount=0.995,
                step_size=0.001,
                plot=False,
            )
            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=4,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # script="scripts/run_experiment_lite_rl.py",
                script="scripts/run_experiment_lite.py",
                log_dir=os.path.join("Results/Controls/Increasing_Points/", timestamp)
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                # plot=True,
            )

def experiment_scratch_baseline():
    # k = 100

    for seed in [10, 30, 50, 100]:
        for _ in range(4):
            env = StandardControllerEnv(k=4, noise=0.05, num_dynamics=4, num_points=k)
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32,32,),
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,
                max_path_length=env.horizon,
                n_itr=100,
                discount=0.995,
                step_size=0.001,
                plot=False,
            )
            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=4,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # script="scripts/run_experiment_lite_rl.py",
                script="scripts/run_experiment_lite.py",
                exp_name=os.path.join("Baseline %d" % k, timestamp),
                log_dir=os.path.join("Results/Controls/Increasing_Points/Baseline", timestamp)
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                # plot=True,
            )

def experiment_compare_scratch_100():
    # k = 100

    for seed in range(1, 10):
        env = StandardControllerEnv(k=4, seed=seed, noise=0.05, num_dynamics=4, num_points=100)
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32,32,),
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1000,
            max_path_length=env.horizon,
            n_itr=100,
            discount=0.995,
            step_size=0.001,
            plot=False,
        )
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # script="scripts/run_experiment_lite_rl.py",
            script="scripts/run_experiment_lite.py",
            exp_name=os.path.join("Baseline %d" % seed, timestamp),
            log_dir=os.path.join("Results/Controls/Seed_Baseline/Baseline/%d" % seed, timestamp)
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

        env = ControllerEnv(k=4, seed=seed, noise=0.05, num_dynamics=4, num_points=100)
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32,32,),
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1000,
            max_path_length=env.horizon,
            n_itr=100,
            discount=0.995,
            step_size=0.001,
            plot=False,
        )
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # script="scripts/run_experiment_lite_rl.py",
            script="scripts/run_experiment_lite.py",
            exp_name=os.path.join("Meta %d" % seed, timestamp),
            log_dir=os.path.join("Results/Controls/Seed_Baseline/Meta/%d" % seed, timestamp)
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

if __name__ == '__main__':
    # experiment_partitioning()
    experiment_compare_scratch_100()
    # experiment_ypartitioning()