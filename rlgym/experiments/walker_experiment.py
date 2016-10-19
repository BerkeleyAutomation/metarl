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
from rllab.envs.gym_env import GymEnv

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

def baseline():    # Heurisic: suboptimal, have no notion of balance.
    env = GymEnv("BipedalWalkerPit-v2")
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break

def test(num=1, path="./Results/Tmp", save=False):
    # env = normalize(GymEnv("BipedalWalkerPit-v2"))
    env = normalize(GymEnv("BipedalWalker-v2", record_video=False))
    # env = DoublePendulumEnv()
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )
    rollout(env, policy)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=env.horizon,
        n_itr=100,
        discount=0.995,
        step_size=0.01,
        # plot=True,
    )
    # run_experiment_lite(
    #     algo.train(),
    #     # Number of parallel workers for sampling
    #     n_parallel=6,
    #     # Only keep the snapshot parameters for the last iteration
    #     snapshot_mode="last",
    #     # script="scripts/run_experiment_lite_rl.py",
    #     # exp_name=exp_name,
    #     log_dir=os.path.join(directory, timestamp) if save else './Results/Tmp',
    #     # Specifies the seed for the experiment. If this is not provided, a random seed
    #     # will be used
    #     # plot=True,
    # )


# if __name__ == '__main__':
if __name__=="__main__":
    test()