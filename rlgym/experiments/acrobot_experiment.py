# from rllab.algos.cem import CEM
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv

import numpy as np
import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")
from rlpy.Domains import Acrobot

# import Domains.RCCarModified as RCCarModified
from Policies import PolicyLoader
from Policies import ModifiedGibbs

from Domains import * # RCCarSlideTurn, RCCarTurn_SWITCH, RCCarSlide_SWITCH
import Domains


# sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/GymEnvs")
from GymEnvs import RLPyEnv
from GymEnvs import HRLEnv

from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.misc.instrument import stub, run_experiment_lite

# stub(globals())
import os, datetime
def rollout(env, policy, N=10):
    T = 500
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
            # print policy.get_action(observation)
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

        print "Reward for trial: %f" % sum(rewards)

def test(num=1, path="./Results/Tmp", save=False):
    policies = [PolicyLoader("models/Acrobot/" + path) for path in ['Mass2_Light','Mass1_Light', 'Mass1_Heavy'][:] ]
    # directory = os.path.join(directory, exp_name)
    acrobot = ModifiedAcrobot()
    acrobot.dt = 0.1
    acrobot.episodeCap = 1000
    # acrobot.torque_noise_max = 0.05
    domain = RLPyEnv(acrobot)
    env = HRLEnv(domain, policies)
    # env = DoublePendulumEnv()
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(8,8)
    )
    # rollout(env, policy)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=50,
        discount=0.995,
        step_size=0.001,
        # plot=True,
    )
    algo.train()
    # # rollout(env, policy)
    # run_experiment_lite(
    #     algo.train(),
    #     # Number of parallel workers for sampling
    #     n_parallel=4,
    #     # Only keep the snapshot parameters for the last iteration
    #     snapshot_mode="last",
    #     script="scripts/run_experiment_lite_rl.py",
    #     exp_name=exp_name,
    #     log_dir=os.path.join(directory, timestamp) if save else './Results/Tmp',
    #     # Specifies the seed for the experiment. If this is not provided, a random seed
    #     # will be used
    #     # plot=True,
    # )


if __name__ == '__main__':
    test()