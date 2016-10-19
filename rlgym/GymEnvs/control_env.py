from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box, Discrete
import rlpy.Domains.RCCar as RC
import controlpy
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides

class ControllerEnv(Env, Serializable):
    figure = None
    _regions = None
    XMAX = 8
    XMIN = -4

    def __init__(self, k=20, noise=0.1, seed=1, num_dynamics=5, num_points=10, force_act=None, normalize_action=False):
        self.force_act = force_act
        self.seed = seed
        np.random.seed(self.seed)
        self.NOISE = noise
        self._high = np.array([self.XMAX, self.XMAX])
        self._low = np.array([self.XMIN, self.XMIN])

        self._init_state = np.array([5, 5])
        self.normalize_action = normalize_action
        self.num_dynamics = num_dynamics
        self.num_points = num_points
        self.init_controllers(k)
        self.init_regions()

        np.random.seed() #set random seed
        Serializable.quick_init(self, locals())
        self.state = None
        self.domain_fig = None

    def reset(self):
        self.state = self._init_state.copy()
        return np.array(self._init_state)


    def step(self, action):
        A, B = self._dynamics[self._get_dynamics_id()]
        if self.force_act is not None:
            action = self.force_act
        chosen_controller = self._controllers[action]
        process_noise = np.random.standard_normal(size=A.shape) * self.NOISE
        control_gain = (A - B.dot(chosen_controller)) + process_noise
        if self.normalize_action:
            assert False
            gradient = control_gain.dot(self.state)
            next_state = 0.2 * gradient / np.linalg.norm(gradient) + self.state
        else:
            next_state = 0.2 * control_gain.dot(self.state / np.linalg.norm(self.state)) + self.state
        reward = -np.linalg.norm(next_state) ** 2 # quad reward
        if self.observation_space.contains(next_state):
            done = abs(reward) < 0.4 # will change
        else:
            done = True
        # print reward
        self.state = next_state.copy()
        return Step(observation=self.state, reward=reward, done=done)

    def _get_dynamics_id(self, state=None):
        assert self._regions is not None
        if state is None:
            state = self.state
        return self._regions.predict(state.reshape(1, -1))[0]

    @property
    def horizon(self):
        return 100

    @property
    def action_space(self):
        return Discrete(len(self._controllers))

    @property
    def observation_space(self):
        return Box(self._low, self._high)

    def render(self):
        x,y = zip(*[self.state])
        if self.figure is None:
            self.figure = plt.figure()
        plt.plot(x, y)
        plt.draw()
        # dom.gcf.canvas.draw()
        plt.pause(0.001)

    def init_controllers(self, k_):
        assert type(k_) == int

        def check_valid(A, B, K):
            feedback = (A - B.dot(K))
            x = [2, 0]
            for i in range(20):
                x = feedback.dot(x)
                if np.linalg.norm(x) > 2.5: # hack, shouldn't take this happen if contractive
                    return False
            return True

        _dynamics0 = self._generate_random_matrices(k_)
        _controllers0 = [self.get_gains(AB) for AB in _dynamics0]
        self._dynamics = []
        self._controllers = []
        for i in range(k_):
            A, B = _dynamics0[i]
            K = _controllers0[i]
            while not check_valid(A, B, K): # i < self.num_dynamics and
                A, B = self.generate_matrix(), self.generate_matrix()
                K = self.get_gains((A,B))
                logging.warn("Getting rid of this %d matrix..." % i)
            self._dynamics.append((A, B))
            self._controllers.append(K)
        print self._controllers
        return self._dynamics

    def init_regions(self):
        np.random.seed(self.seed )
        points = [self.observation_space.sample() for _ in range(self.num_points)]
        # print points
        self._regions = KNeighborsClassifier(n_neighbors=1)
        assignments = np.random.choice(range(self.num_dynamics), self.num_points)
        self._regions.fit(points, assignments)
        np.random.seed() #set random seed

    def generate_matrix(self):
        A = np.random.randn(2,2)
        normalized_A = A / np.linalg.norm(A)
        return normalized_A

    def _generate_random_matrices(self, n):
        return [(self.generate_matrix(), self.generate_matrix()) for i in xrange(n)]
    
    def get_gains(self, AB):
        A, B = AB
        Q, R = np.identity(2), np.identity(2)
        gain, X, egv = controlpy.synthesis.controller_lqr(A, B, Q, R)
        return gain

    def render(self):
        if self.figure is None:
            xmax, ymax = self.observation_space.high
            xmin, ymin = self.observation_space.low
            X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            # Z = np.reshape([self._get_dynamics_id(p) for p in positions.T], X.shape) # check if this works
            Z = np.reshape(self._regions.predict(positions.T), X.shape)
            plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                      extent=[xmin, xmax, ymin, ymax])
            self.figure = plt.figure(1)

class StandardControllerEnv(ControllerEnv, Serializable):

    def step(self, action):
        A, B = self._dynamics[self._get_dynamics_id()]
        if self.force_act is not None:
            assert False, "Turn off force_act "
            action = self.force_act
        process_noise = np.random.standard_normal(size=A.shape) * self.NOISE
        x_t = self.state / np.linalg.norm(self.state)
        control_gain = 0.2 * (A.dot(x_t) + (B + process_noise).dot(action)) 
        next_state = control_gain + self.state
        reward = -np.linalg.norm(next_state) ** 2 # quad reward
        if self.observation_space.contains(next_state):
            done = abs(reward) < 0.4 # will change
        else:
            done = True
        # print reward
        self.state = next_state.copy()
        return Step(observation=self.state, reward=reward, done=done)

    @property
    def action_space(self):
        return Box(low=np.array([-1, 1]), high=np.array([-1, 1]))

class NoisyObsController(ProxyEnv, Serializable):

    def __init__(self, env, obs_noise=0.1):
        super(NoisyObsController, self).__init__(env)
        Serializable.quick_init(self, locals())
        self.obs_noise = obs_noise

    def get_obs_noise_scale_factor(self, obs):
        # return np.abs(obs)
        return np.ones_like(obs)

    def inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * \
            np.random.normal(size=obs.shape)
        return obs + noise

    @overrides
    def step(self, action):
        step_obj = self._wrapped_env.step(action)
        next_obs, reward, done, info = step_obj
        if self._wrapped_env._get_dynamics_id() < 2:
            return Step(self.inject_obs_noise(next_obs), reward, done, **info)
        else:
            return step_obj
#     #


class BandedControllerEnv(ControllerEnv, Serializable):

    def __init__(self, bands=1, **kwargs):
        self._bands = bands
        Serializable.quick_init(self, locals())
        super(BandedControllerEnv, self).__init__(**kwargs)

    def init_regions(self):
        pass

    def _get_dynamics_id(self, state=None):
        if state is None:
            state = self.state
        x = state[0]
        return int((x - self.XMIN) * self._bands ) % self.num_dynamics

    def render(self):
        if self.figure is None:
            xmax, ymax = self.observation_space.high
            xmin, ymin = self.observation_space.low
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape([self._get_dynamics_id(p) for p in positions.T], X.shape) # check if this works
            # Z = np.reshape(self._regions.predict(positions.T), X.shape)
            plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                      extent=[xmin, xmax, ymin, ymax])
            self.figure = plt.figure(1)


class YBandedControllerEnv(BandedControllerEnv):

    def _get_dynamics_id(self, state=None):
        if state is None:
            state = self.state
        y = state[1]
        return int((y - self.XMIN) * self._bands ) % self.num_dynamics