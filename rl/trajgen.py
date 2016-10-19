from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import pickle as p
import time
import numpy as np 

ALGOFILE="trpo_cartpole_model.p"
DUMPFILE="cem_train.p"
N = 100
discount = 0.99
T = 100
import ipdb; ipdb.set_trace()

with open(ALGOFILE, "r") as f:
    algo = p.load(f)

policy = algo.policy
env = normalize(CartpoleEnv())

paths = []
totalobvs = []

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
        # action = policy.action_space.sample()
        # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
        # case it is not needed.
        next_observation, reward, terminal, _ = env.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        observation = next_observation
        totalobvs.append(observation)
        if terminal:
            # Finish rollout if terminal state reached
            break

    # We need to compute the empirical return for each time step along the
    # trajectory
    returns = []
    return_so_far = 0
    for t in xrange(len(rewards) - 1, -1, -1):
        return_so_far = rewards[t] + discount * return_so_far
        returns.append(return_so_far)
    # The returns are stored backwards in time, so we need to revert it
    returns = returns[::-1]

    paths.append(dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        returns=np.array(returns)
    ))

import ipdb; ipdb.set_trace()

with open(DUMPFILE, "w") as f:
    p.dump(totalobvs, f)