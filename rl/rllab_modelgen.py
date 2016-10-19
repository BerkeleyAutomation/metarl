from rllab.algos.cem import CEM
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(CartpoleEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = CEM(
    env=env,
    policy=policy,
    baseline=baseline,
    n_itr=200,
)
algo.train()

import pickle as p
with open("good_cem_cartpole.p", "w") as f:
    p.dump(algo, f)
