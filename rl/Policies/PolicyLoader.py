import numpy as np
from collections import defaultdict
import os
import sys
sys.path.insert(0, "/home/jarvis/work/clipper/models/rl/")
import Domains
import Representations
import yaml 
from rlpy.Policies.Policy import Policy


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        ret_val = yaml.load(f)
    return ret_val

class MockDomain():
	def __init__(self, domain):
		self.actions_num = domain.actions_num
		self.state_space_dims = domain.state_space_dims
		self.continuous_dims = domain.continuous_dims
		self.statespace_limits = domain.statespace_limits


class PolicyLoader(Policy):
	representation = None

	def __init__(self, path_to_exp):
		self.path_to_exp = path_to_exp

	def load_representation(self):
		param_path = os.path.join(self.path_to_exp, "params.yml")
		print "[INFO] Loading agent from " + param_path
		params = type("Parameters", (), load_yaml(param_path))

		domain = eval(params.domain) # may need to fix because path may not work		
		mock_domain = MockDomain(domain)
		Representation = eval(params.representation)
		self.representation = Representation(mock_domain, **params.representation_params)
		self.representation.load_from_directory(self.path_to_exp)
		self._p_actions = range(domain.actions_num)
		self.representation.logger = None

	def pi(self, state):
		if self.representation is None:
			self.load_representation()
		action_list = self.representation.bestActions(state, False, self._p_actions)
		return np.random.choice(action_list)

	def get_action(self, state):
		return self.pi(state), {'mean': 0}

	def reset(self):
		pass


if __name__ == '__main__':
	tryit = PolicyLoader('/home/jarvis/work/clipper/models/rl/Results/Mixed_ActionsB/agent1/Aug12_01-10-612622/')
	import pickle as pkl
	with open("tmp/testdump.pk", "w") as f:
		pkl.dump(tryit, f)

	with open("tmp/testdump.pk", "r") as f:
		rep = pkl.load(f)

	print rep.pi(np.array([0, 0, 1, 1]))