from rllab.sampler.utils import rollout
from rllab.misc import special
import argparse
import joblib
import uuid
import os
import matplotlib.pyplot as plt


import numpy as np
import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")
sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/")

# import Domains.RCCarModified as RCCarModified
from Policies import PolicyLoader
from Policies import ModifiedGibbs
from GymEnvs import RLPyEnv

from Domains import *
import Domains


# sys.path.insert(0, "/home/jarvis/work/clipper/models/rlgym/GymEnvs")
# from GymEnvs import RLPyEnv
# from GymEnvs import HRLEnv

filename = str(uuid.uuid4())
from scipy.stats import mode

def visualize_distribution(env, policy):

    try:
        xmax, ymax, smax, amax = env.observation_space.high
        xmin, ymin, smin, amin = env.observation_space.low
        xmin, xmax = -3.5, 3.5
        ymin, ymax = -2, 2
        X, Y, V, T = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, smax/2:smax:10j, (amin/5):0:10j]
        positions = np.vstack([X.ravel(), Y.ravel(), V.ravel(), T.ravel()]).T
        positions = positions.reshape(-1, 100, 4)
        Z = []
        counts = []
        sums = []
        for xy_pos in positions:
            actions, _ = policy.get_actions(xy_pos)
            Z.append(mode(actions)[0][0])
            counts.append(mode(actions)[1][0])
            sums.append(sum(actions))
        Z = np.array(Z).reshape((100, 100))
        sums = np.array(sums).reshape((100, 100))
        counts = np.array(counts).reshape((100, 100))
        env.reset()
        env.render()
        # plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
        # plt.imshow(np.rot90(counts), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
        # plt.imshow(Z, cmap=plt.cm.gist_earth_r,
        #           extent=[xmin, xmax, ymin, ymax])
        # plt.show()
        return sums
        # x = None
    except Exception:
        pass

def visualize_reward(env):
    dom = env.domain.domain
    xmin, xmax = -3.5 + 0.01, 3.5 - 0.01
    ymin, ymax = -2 + 0.01, 2 - 0.01
    X, Y,= np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
    positions = np.vstack([X.ravel(), Y.ravel(), np.zeros(400), np.zeros(400)]).T
    Z = []

    for xyz_pos in positions:
        dom.state = xyz_pos
        reward = -dom._reward(xyz_pos, dom.isTerminal())
        Z.append(reward if abs(reward) < 1 else 1)
    Z = np.array(Z).reshape((20, 20))
    plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    import ipdb; ipdb.set_trace()  # breakpoint 8ca5a182 //

    # plt.imshow(np.rot90(counts), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    # plt.imshow(Z, cmap=plt.cm.gist_earth_r,
    #           extent=[xmin, xmax, ymin, ymax])
    # plt.show()
    return sums
    # x = None

# def visualize_3d_distribution(env, policy):
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     try:
#         xmax, ymax, smax, amax = env.observation_space.high
#         xmin, ymin, smin, amin = env.observation_space.low
#         xmin, xmax = -3.5, 3.5
#         ymin, ymax = -2, 2
#         X, Y, V, T = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, smax/2:smax:10j, (amin/5):0:10j]
#         positions = np.vstack([X.ravel(), Y.ravel(), V.ravel(), T.ravel()]).T
#         positions = positions.reshape(-1, 100, 4)
#         Z = []
#         counts = []
#         sums = []
#         for xy_pos in positions:
#             actions, _ = policy.get_actions(xy_pos)
#             Z.append(mode(actions)[0][0])
#             counts.append(mode(actions)[1][0])
#             sums.append(sum(actions))
#         Z = np.array(Z).reshape((100, 10, 10))
#         sums = np.array(sums).reshape((100, 10, 10))
#         counts = np.array(counts).reshape((100, 10, 10))
#         env.reset()
#         env.render()
#         # plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
#         # plt.imshow(np.rot90(counts), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
#         # plt.imshow(Z, cmap=plt.cm.gist_earth_r,
#         #           extent=[xmin, xmax, ymin, ymax])
#         # plt.show()
#         return sums
#         # x = None
#     except Exception:
#         pass

def rollout(env, policy, N=100, force_act=None):
    # if env.__class__.__name__ == "StandardControllerEnv" or env.__class__.__name__ == "ControllerEnv":
    #     import numpy as np
    #     xmax, ymax = env.observation_space.high
    #     xmin, ymin = env.observation_space.low
    #     X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #     positions = np.vstack([X.ravel(), Y.ravel()])
    #     Z = np.reshape(env._regions.predict(positions.T), X.shape)
    #     plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    #               extent=[xmin, xmax, ymin, ymax])
    # visualize_reward(env)
    # sums = visualize_distribution(env, policy)
    T = env.horizon

    xmin, xmax = -3.5, 3.5
    ymin, ymax = -2, 2
    env = RLPyEnv(RCCarSlideGradient())
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!! Make sure you have right environment"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!! Make sure you have right environment"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!! Make sure you have right environment"
    import ipdb; ipdb.set_trace()  # breakpoint cc480b65 //
    def show_interactive_traj(observations, num=0, color='r', style='x'):
        traj_x, traj_y = zip(*observations)
        try:
            line = plt.axes().lines[num]
            if len(line.get_xdata()) != len(traj_x): # if plot has discrepancy from data
                line.set_xdata(traj_x)
                line.set_ydata(traj_y)
        except IndexError:
            plt.plot(traj_x, traj_y, style, color=color)

    def show_trajectory(observations, actions, style='o'):
        try:
            observations_0 = [xa[0] for xa in zip(observations, actions) if xa[1] == 0]
            traj_x, traj_y = zip(*observations_0)
            plt.plot(traj_x, traj_y, style, color='r', alpha=0.05)
        except Exception:
            pass
        try:
            observations_1 = [xa[0] for xa in zip(observations, actions) if xa[1] == 1]
            traj_x, traj_y = zip(*observations_1)
            plt.plot(traj_x, traj_y, style, color='b', alpha=0.05)
            plt.show()
        except Exception:
            pass


    # plt.imshow(np.rot90(sums), cmap=plt.cm.PuOr, extent=[xmin, xmax, ymin, ymax])
    for _ in xrange(N):
        full_observations = []
        observations = []
        actions = []
        rewards = []

        observation = env.reset()
        T = env.horizon
        # env.render()

        for _ in xrange(T):
            # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contains
            # sufficient statistics for the action distribution. It should at least contain entries that would be
            # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
            # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
            # not needed.
            env.render()
            action, _ = policy.get_action(observation)
            if force_act is not None:
                action = force_act
            full_observations.append(observation)
            observations.append(observation[:2])
            actions.append(action)
            # action = policy.action_space.sample()
            # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
            # case it is not needed.
            next_observation, reward, terminal, _ = env.step(action)
            rewards.append(reward)
            observation = next_observation
            # observation_0 = [xa[0] for xa in zip(observations, actions) if xa[1] == 0]
            # observation_1 = [xa[0] for xa in zip(observations, actions) if xa[1] == 1]
            # if len(observation_0):
            #     show_interactive_traj(observation_0)
            # if len(observation_1):
            #     show_interactive_traj(observation_1, num=1, color='b', style='o')

            # totalobvs.append(observation)
            if terminal:
                observations.append(observation[:2])
                # Finish rollout if terminal state reached
                break
        print sum(rewards)
        # show_trajectory(observations, actions)

        # x_list, y_list = zip(*observations)
    print observations

    # plt.gca().add_patch(
    #     plt.Circle(
    #         env.domain.domain.GOAL,
    #         radius=env.domain.domain.GOAL_RADIUS * 5,
    #         hatch='x',
    #         color='gray',
    #         alpha=.4))
    # import ipdb; ipdb.set_trace()  # breakpoint 85c3b260 //

    #     # obs = np.array(observations)
    #     # print "Longest step taken: {}".format(max(np.linalg.norm(obs[:-1] - obs[1:], axis=1)))
    #     # print actions
    #     # print rewards, sum(rewards)
    #     # plt.plot(x_list, y_list, 'x', linestyle="solid", )
    #     # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--loop', type=int, default=1,
                        help='# of loops')
    args = parser.parse_args()

    policy = None
    env = None
    while True:
        if ':' in args.file:
            # fetch file using ssh
            os.system("rsync -avrz %s /tmp/%s.pkl" % (args.file, filename))
            data = joblib.load("/tmp/%s.pkl" % filename)
            if policy:
                new_policy = data['policy']
                policy.set_param_values(new_policy.get_param_values())
                path = rollout(env, policy, max_path_length=args.max_length,
                               animated=True, speedup=args.speedup)
            else:
                policy = data['policy']
                env = data['env']
                path = rollout(env, policy, max_path_length=args.max_length,
                               animated=True, speedup=args.speedup)
        else:
            data = joblib.load(args.file)
            policy = data['policy']
            env = data['env']
            path = rollout(env, policy )
        break