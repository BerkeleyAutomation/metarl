from rllab.algos.cem import CEM
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv

import sys
from utils import rollout
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")

import Domains.RCCarModified as RCCarModified
from Domains.RCCar_Experimental import *
from Domains.Acrobot_Experimental import ModifiedAcrobot
from rlpy.Domains import Acrobot
from rllab.envs.gym_env import GymEnv

from rllab.misc.instrument import stub, run_experiment_lite


# sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/GymEnvs")
from GymEnvs import RLPyEnv, ControllerEnv
# from GymEnvs import hrl_env
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy

# stub(globals())

def rollout(env, policy, N=1, T=1000):
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
            observations.append(observation)
            actions.append(action)
            # action = policy.action_space.sample()
            # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
            # case it is not needed.
            action = 4
            next_observation, reward, terminal, _ = env.step(action)
            rewards.append(reward)
            observation = next_observation
            # totalobvs.append(observation)
            if terminal:
                # Finish rollout if terminal state reached
                break
        x_list, y_list = zip(*observations)
        print observations
        print actions
        plt.plot(x_list, y_list)
        plt.show()


# rl = ModifiedAcrobot()
rc = RCCarSlideLeftGradient()
env = RLPyEnv(rc)
# env = ControllerEnv(k=10)
policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32,32,),
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=3000,
    max_path_length=env.horizon,
    n_itr=100,
    discount=0.995,
    step_size=0.01,
    plot=False,
)
algo.train()
rollout(env, policy)

with open("models/rc_gradient/agentturn" + "policy.pkl", "w") as f:
    f.dump(policy)

# run_experiment_lite(
#     algo.train(),
#     # Number of parallel workers for sampling
#     n_parallel=4,
#     # Only keep the snapshot parameters for the last iteration
#     snapshot_mode="last",
#     script="scripts/run_experiment_lite_rl.py",
#     # script="scripts/run_experiment_lite.py",
#     log_dir="Results/Tmp",
#     # Specifies the seed for the experiment. If this is not provided, a random seed
#     # will be used
#     # plot=True,
# )
