import numpy as np 
from collections import defaultdict
from rllab.sampler.utils import rollout

import numpy as np
import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")

# import Domains.RCCarModified as RCCarModified
from Policies import PolicyLoader
from Domains import * 
from GymEnvs import RLPyEnv

class Bandits():

	def __init__(self, policies, env, N=100, rmax=100, rmin = -100):
		self.policies = policies
		self.env = env
		self.rmax = rmax
		self.rmin = rmin
		self.iterations = N
		self.UCB = defaultdict(list)
		self.choices = []
		for i in range(len(policies)):
			self.UCB[i] = []

	def get_max_UCB(self, itr):
		if itr == 0:
			return np.random.choice(len(self.policies))
		mus = np.array([np.mean(v)  if len(v) else 1e6 for k, v in self.UCB.items()])
		deltas = np.array([abs(self.rmax - self.rmin) if len(v) else 0 for k, v in self.UCB.items()])
		values = mus + deltas / np.sqrt(2 * itr) 
		return np.argmax(values)

	def run(self):

		for i in range(self.iterations):
			chosen_policy = self.get_max_UCB(i)
			results = rollout(self.env, self.policies[chosen_policy], self.env.horizon)

			reward = sum(results['rewards']) - self.rmin # constant offset
			self.UCB[chosen_policy].append(reward)
			self.choices.append((len(results['rewards']), chosen_policy))# print self.UCB

		chosen_policy = self.get_max_UCB(self.iterations)
		return chosen_policy

def two_policy_bandits():
	rccar = RCCarSlideTurn(noise=0.1) # remove process noise
	domain = RLPyEnv(rccar)
	policies = [PolicyLoader("models/slideturn_experiment/" + path) for path in ['agent0','agent1'] ]
	band = Bandits(policies, domain, N=100, rmax=rccar.GOAL_REWARD, rmin=-rccar.episodeCap)
	chosen = band.run()
	import joblib; joblib.dump(band.choices, "Results/Bandits/TwoPolicy")
	return band


def noisy_bandits():
	rccar = RCCarBarriers(noise=0.1)
	policies = [PolicyLoader("models/noisy/" +path) for path in ['good','untrained', 'untrained', 'untrained'] ]
	domain = RLPyEnv(rccar)
	band = Bandits(policies, domain, N=100, rmax=rccar.GOAL_REWARD, rmin=-rccar.episodeCap - 20)
	chosen = band.run()
	import joblib; joblib.dump(band.choices, "Results/Bandits/Noisy")
	return band

def frac(arr, ans=0):
	steps, arr = zip(*arr)
	arr = np.array(arr) == ans
	percentage = []
	for i in range(len(arr)):
		percentage.append(np.mean(arr[:i+1]))
	import matplotlib.pyplot as plt
	plt.plot(np.cumsum(steps), percentage)
	plt.show()


if __name__ == '__main__':
	band = two_policy_bandits()
	frac(band.choices, ans=1)