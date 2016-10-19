import pickle

def rollout(env, policy, show=False):
    T = env.horizon
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
        if show:
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
    return sum(rewards), terminal


def load_rllab_policy(path_to_dump):
    import joblib
    data = joblib.load(path_to_dump)
    return data['policy']

def load_rllab_pkl(path_to_dump):
    with open(path_to_dump, "r") as f:
        policy = pickle.load(f)
    return policy