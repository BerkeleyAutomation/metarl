"""Standard Experiment for Learning Control in RL."""

import logging
from rlpy.Tools import plt
import numpy as np
from copy import deepcopy
import re
import argparse
from rlpy.Tools import deltaT, clock, hhmmss
from rlpy.Tools import className, checkNCreateDirectory
from rlpy.Tools import printClass
import rlpy.Tools.results
from rlpy.Experiments import Experiment
from rlpy.Tools import lower
import os
import rlpy.Tools.ipshell
import json
from collections import defaultdict, Counter

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class ExperimentMod(Experiment):

    """
    The Experiment controls the training, testing, and evaluation of the
    agent. Reinforcement learning is based around
    the concept of training an :py:class:`~Agents.Agent.Agent` to solve a task,
    and later testing its ability to do so based on what it has learned.
    This cycle forms a loop that the experiment defines and controls. First
    the agent is repeatedly tasked with solving a problem determined by the
    :py:class:`~Domains.Domain.Domain`, restarting after some termination
    condition is reached.
    (The sequence of steps between terminations is known as an *episode*.)

    Each time the Agent attempts to solve the task, it learns more about how
    to accomplish its goal. The experiment controls this loop of "training
    sessions", iterating over each step in which the Agent and Domain interact.
    After a set number of training sessions defined by the experiment, the
    agent's current policy is tested for its performance on the task.
    The experiment collects data on the agent's performance and then puts the
    agent through more training sessions. After a set number of loops, training
    sessions followed by an evaluation, the experiment is complete and the
    gathered data is printed and saved. For each section, training and
    evaluation, the experiment determines whether or not the visualization
    of the step should generated.

    The Experiment class is a base class that provides
    the basic framework for all RL experiments. It provides the methods and
    attributes that allow child classes to interact with the Agent
    and Domain classes within the RLPy library.

    .. note::
        All experiment implementations should inherit from this class.
    """

    #: The Main Random Seed used to generate other random seeds (we use a different seed for each experiment id)
    mainSeed = 999999999
    #: Maximum number of runs used for averaging, specified so that enough
    #: random seeds are generated
    maxRuns = 1000
    # Array of random seeds. This is used to make sure all jobs start with
    # the same random seed
    randomSeeds = np.random.RandomState(mainSeed).randint(1, mainSeed, maxRuns)

    #: ID of the current experiment (main seed used for calls to np.rand)
    exp_id = 1

    # The domain to be tested on
    domain = None
    # The agent to be tested
    agent = None

    #: A 2-d numpy array that stores all generated results.The purpose of a run
    #: is to fill this array. Size is stats_num x num_policy_checks.
    result = None
    #: The name of the file used to store the data
    output_filename = ''
    # A simple object that records the prints in a file
    logger = None

    max_steps = 0  # Total number of interactions
    # Number of Performance Checks uniformly scattered along timesteps of the
    # experiment
    num_policy_checks = 0
    log_interval = 0  # Number of seconds between log prints to console
    best_return = None

    log_template = '{total_steps: >6}: E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'
    performance_log_template = '{episode: >6}: >>> [{total_steps}] Return={totreturn: >10.4g}, Stdev={stdreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'

    def __init__(self, agent, domain, exp_id=1, max_episode=50,
                 config_logging=True, num_policy_checks=10, log_interval=1,
                 path='Results/Temp',
                 checks_per_policy=1, stat_bins_per_state_dim=0, **kwargs):
        """
        :param agent: the :py:class:`~Agents.Agent.Agent` to use for learning the task.
        :param domain: the problem :py:class:`~Domains.Domain.Domain` to learn
        :param exp_id: ID of this experiment (main seed used for calls to np.rand)
        :param max_steps: Total number of interactions (steps) before experiment termination.

        .. note::
            ``max_steps`` is distinct from ``episodeCap``; ``episodeCap`` defines the
            the largest number of interactions which can occur in a single
            episode / trajectory, while ``max_steps`` limits the sum of all
            interactions over all episodes which can occur in an experiment.

        :param num_policy_checks: Number of Performance Checks uniformly
            scattered along timesteps of the experiment
        :param log_interval: Number of seconds between log prints to console
        :param path: Path to the directory to be used for results storage
            (Results are stored in ``path/output_filename``)
        :param checks_per_policy: defines how many episodes should be run to
            estimate the performance of a single policy

        """
        self.trials = defaultdict(list)
        self.best_representation = None
        self.max_episode = max_episode

        super(ExperimentMod, self).__init__(agent, domain, exp_id=exp_id,# max_steps=0, not used
                 config_logging=config_logging, num_policy_checks=num_policy_checks, log_interval=log_interval,
                 path=path,
                 checks_per_policy=checks_per_policy, stat_bins_per_state_dim=stat_bins_per_state_dim, **kwargs)

    def seed_components(self):
        self.trials_filename = '{:0>3}-trials.json'.format(self.exp_id)
        super(ExperimentMod, self).seed_components()


    def update_best_representation(self, total_steps, episode_number, visualize_performance):
        window = 30
        cur_return = np.mean(self.trials["return"][-window:])

        if len(self.trials['return']) < window:
            return
        elif len(self.trials['return']) == window:
            #correct for the first x trials
            for i in range(window):
                self.trials["best_return"].append(cur_return)

            self.logger.info("Updating first representationto %f..." % ( cur_return))
            if hasattr(self.agent.representation, "dump_to_directory", ):
                self.agent.representation.dump_to_directory(self.full_path)
            self.best_return = cur_return
            # self.evaluate(total_steps, episode_number, visualize_performance)

        elif cur_return > self.best_return + 1:

            self.logger.info("Updating best representation from %f to %f..." % (self.best_return, cur_return))
            if hasattr(self.agent.representation, "dump_to_directory", ):
                self.agent.representation.dump_to_directory(self.full_path)
            self.best_return = cur_return
            # self.evaluate(total_steps, episode_number, visualize_performance)


    def run(self, visualize_performance=0, visualize_learning=False,
            visualize_steps=False, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results

        :param visualize_performance: (int)
            determines whether a visualization of the steps taken in
            performance runs are shown. 0 means no visualization is shown.
            A value n > 0 means that only the first n performance runs for a
            specific policy are shown (i.e., for n < checks_per_policy, not all
            performance runs are shown)
        :param visualize_learning: (boolean)
            show some visualization of the learning status before each
            performance evaluation (e.g. Value function)
        :param visualize_steps: (boolean)
            visualize all steps taken during learning
        :param debug_on_sigurg: (boolean)
            if true, the ipdb debugger is opened when the python process
            receives a SIGURG signal. This allows to enter a debugger at any
            time, e.g. to view data interactively or actual debugging.
            The feature works only in Unix systems. The signal can be sent
            with the kill command:

                kill -URG pid

            where pid is the process id of the python interpreter running this
            function.

        """

        if debug_on_sigurg:
            rlpy.Tools.ipshell.ipdb_on_SIGURG()
        self.performance_domain = deepcopy(self.domain)
        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id

        self.trials = defaultdict(list)
        self.trials["seed"] = self.exp_id

        total_steps = 0
        eps_steps = 0
        eps_return = 0
        episode_number = 0

        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.showLearning(self.agent.representation)

        # Used to bound the number of logs in the file
        start_log_time = clock()
        # Used to show the total time took the process
        self.start_time = clock()
        self.elapsed_time = 0
        # do a first evaluation to get the quality of the inital policy
        self.evaluate(total_steps, episode_number, visualize_performance)
        self.total_eval_time = 0.
        terminal = True
        # while total_steps < self.max_steps:
        while episode_number < self.max_episode:
            if terminal or eps_steps >= self.domain.episodeCap:
                counter = defaultdict(int)
                # if len(self.agent.policy._trajectory['options']):
                #     sames = [max(Counter(x).values()) for x in self.agent.policy._trajectory['options']]
                #     print "Variance of unanimous choices: {}".format(np.mean(sames))
                self.agent.policy._trajectory = defaultdict(list)
                s, terminal, p_actions = self.domain.s0()
                # import ipdb; ipdb.set_trace()
                a = self.agent.policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.agent.representation)

                # Output the current status if certain amount of time has been
                # passed
                eps_return = 0
                eps_steps = 0
                episode_number += 1
            # Act,Step
            r, ns, terminal, np_actions = self.domain.step(a)
            counter[a] += 1
            # print "Next state: (%0.3f, %0.3f, %0.3f, %0.3f) : %0.5f," % (ns[0], ns[1], ns[2], ns[3], self.agent.representation.V(ns, terminal, np_actions))
            # if any(self.agent.representation.weight_vec > 0):
            #     wns = np.argmax(self.agent.representation.weight_vec)
            #     print self.agent.representation.weight_vec[wns]
                

            self.logger.debug(s, self.agent.representation.Qs(s, False))
            # print ns, self.agent.representation.Qs(ns, False)
            # print "*" * 10

            # self._gather_transition_statistics(s, a, ns, r, learning=True)
            na = self.agent.policy.pi(ns, terminal, np_actions)

            total_steps += 1
            eps_steps += 1
            eps_return += r

            # if total_steps == 60000:

            # Print Current performance
            if (terminal or eps_steps == self.domain.episodeCap):
                self.trials["learning_steps"].append(total_steps)
                self.trials["eps_steps"].append(eps_steps)
                self.trials["return"].append(eps_return)
                # print episode_number, eps_return
                self.trials["num_feat"].append(self.agent.representation.features_num)
                self.trials["learning_episode"].append(episode_number)
                self.trials["Action_0"].append(float(counter[0]) / eps_steps)
                self.trials["Action_1"].append(float(counter[1]) / eps_steps)
                if self.best_return is not None:
                    self.trials["best_return"].append(self.best_return)

                self.update_best_representation(
                        total_steps,
                        episode_number,
                        visualize_performance)
                # Check Performance
                if episode_number % (self.max_episode / self.num_policy_checks) == 0:
                    self.elapsed_time = deltaT(
                        self.start_time) - self.total_eval_time

                    # show policy or value function
                    if visualize_learning:
                        self.domain.showLearning(self.agent.representation)

                    # self.agent.policy.turn_on_printing()

                    self.evaluate(
                        total_steps,
                        episode_number,
                        visualize_performance)


                    # self.agent.policy.turn_off_printing()
                    self.total_eval_time += deltaT(self.start_time) - \
                        self.elapsed_time - \
                        self.total_eval_time
                    start_log_time = clock()

            self.agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
            s, a, p_actions = ns, na, np_actions
            # Visual
            if visualize_steps:
                self.domain.show(a, self.agent.representation)


        # Visual
        if visualize_steps:
            self.domain.show(a, self.agent.representation)
        self.logger.info("Total Experiment Duration %s" % (hhmmss(deltaT(self.start_time))))

    def evaluate(self, total_steps, episode_number, visualize=0):
        """
        Evaluate the current agent within an experiment

        :param total_steps: (int)
                     number of steps used in learning so far
        :param episode_number: (int)
                        number of episodes used in learning so far
        """
        print "Stepsize: %f" % self.agent.learn_rate
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        random_state = np.random.get_state()
        #random_state_domain = copy(self.domain.random_state)
        elapsedTime = deltaT(self.start_time)
        performance_return = 0.
        performance_steps = 0.
        performance_term = 0.
        performance_discounted_return = 0.
        performance_return_squared = 0.
        for j in xrange(self.checks_per_policy):

            p_ret, p_step, p_term, p_dret = self.performanceRun(
                total_steps, visualize=visualize > j)
            print j, p_ret
            performance_return += p_ret
            performance_steps += p_step
            performance_term += p_term
            performance_discounted_return += p_dret
            performance_return_squared += p_ret **2
        performance_return /= self.checks_per_policy
        performance_return_squared /= self.checks_per_policy
        performance_steps /= self.checks_per_policy
        performance_term /= self.checks_per_policy
        performance_discounted_return /= self.checks_per_policy

        std_return = np.sqrt(performance_return_squared - (performance_return) ** 2)

        self.result["learning_steps"].append(total_steps)
        self.result["return"].append(performance_return)
        self.result["return_std"].append(std_return)
        self.result["learning_time"].append(self.elapsed_time)
        self.result["num_features"].append(self.agent.representation.features_num)
        self.result["steps"].append(performance_steps)
        self.result["terminated"].append(performance_term)
        self.result["learning_episode"].append(episode_number)
        self.result["discounted_return"].append(performance_discounted_return)
        # reset start time such that performanceRuns don't count
        self.start_time = clock() - elapsedTime
        if total_steps > 0:
            remaining = hhmmss(
                elapsedTime * (self.max_steps - total_steps) / total_steps)
        else:
            remaining = "?"
        self.logger.info(
            self.performance_log_template.format(episode=episode_number,
                                                total_steps=total_steps,
                                                 # elapsed=hhmmss(
                                                 #     elapsedTime),
                                                 # remaining=remaining,
                                                 totreturn=performance_return,
                                                 stdreturn=std_return, #TODO
                                                 steps=performance_steps,
                                                 num_feat=self.agent.representation.features_num))

        np.random.set_state(random_state)
        #self.domain.rand_state = random_state_domain

    def save(self):
        """Saves the experimental results to the ``results.json`` file
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        trials_fn = os.path.join(self.full_path, self.trials_filename)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        with open(results_fn, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True)

        with open(trials_fn, "w") as f:
            json.dump(self.trials, f, indent=4, sort_keys=True)

    def _gather_transition_statistics(self, s, a, ns, r, learning=True):
         # s, a, self.agent.representation.Qs(s, False)
        pass

    def load(self):
        """loads the experimental results from the ``results.txt`` file
        If the results could not be found, the function returns ``None``
        and the results array otherwise.
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        self.results = rlpy.Tools.results.load_single(results_fn)
        return self.results

    def plot_trials(self, y="eps_return", x="learning_steps", average=10, save=False):
        """Plots the performance of the experiment
        This function has only limited capabilities.
        For more advanced plotting of results consider
        :py:class:`Tools.Merger.Merger`.
        """
        def movingaverage(interval, window_size):
            window= np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'same')

        labels = rlpy.Tools.results.default_labels
        performance_fig = plt.figure("Performance")
        trials = self.trials
        y_arr = np.array(trials[y])
        if average:
            assert type(average) is int, "Filter length is not an integer!"
            y_arr = movingaverage(y_arr, average)
        plt.plot(trials[x], y_arr, '-bo', lw=3, markersize=10)
        plt.xlim(0, trials[x][-1] * 1.01)
        m = y_arr.min()
        M = y_arr.max()
        delta = M - m
        if delta > 0:
            plt.ylim(m - .1 * delta - .1, M + .1 * delta + .1)
        xlabel = labels[x] if x in labels else x
        ylabel = labels[y] if y in labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            path = os.path.join(
                self.full_path,
                "{:3}-trials.pdf".format(self.exp_id))
            performance_fig.savefig(path, transparent=True, pad_inches=.1)
        plt.ioff()
        plt.show()