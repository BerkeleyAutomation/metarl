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
            # env.render()
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

def slideturn_reward(val=0.1, directory="./Results/Car/RewardVarying/", exp_name="Cap_", save=False):
    policies = [PolicyLoader("models/slideturn_experiment/" + path) for path in ['agent0','agent1'] ]
    rccar = RCCarST_Truncator(pnlty_trunc=val, noise=0.1)

    domain = RLPyEnv(rccar)
    env = HRLEnv(domain, policies)
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for i in range(5):
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
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
        dir_name = os.path.join(directory, exp_name) 
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            script="scripts/run_experiment_lite_rl.py",
            exp_name=exp_name + timestamp,
            log_dir=os.path.join(dir_name, timestamp) if save else './Results/Tmp',
            use_gpu=False,
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # plot=True,
        )

if __name__ == '__main__':
    slideturn_reward(val=0.01, exp_name="Exp1", save=True)
    slideturn_reward(val=0.1, exp_name="Exp2", save=True)
    slideturn_reward(val=0.2, exp_name="Exp3", save=True)
    slideturn_reward(val=0.3, exp_name="Exp4", save=True)
    slideturn_reward(val=0.5, exp_name="Exp5", save=True)
    # slideturn_reward(val=0.7, exp_name="Exp5", save=True)