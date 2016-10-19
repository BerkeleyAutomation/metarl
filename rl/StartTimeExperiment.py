from IndividualExperiment import make_experiment as individual_experiment
from Experiment_votingupdate import make_experiment as voting_experiment
from MetaRLSarsaExperiment import make_experiment as metarl_experiment
from Experiment_Confidence import make_experiment as confidence_experiment

import AgentGenerator
import Representations
import matplotlib.pyplot as plt
from utils import get_time_str
import numpy as np
import joblib
import os
import yaml
import pickle
import shutil
import logging

class StartTimeExperiment():

    base_yaml_file = "base_params.yml"
    def __init__(self, final_results_path="./Results/StartTimeExperimentMaha/",
                    subagent_path_root='./GeneratedAgents/RCCarConfidenceMaha/',
                    seed=1):
        self.seed = seed
        self.logger = logging.getLogger("StartingExperiment")
        self.all_agent_paths = []
        self.delay = 10000 # max_steps after beginning meta experiments
        self.max_start_time = 10000
        self.num_policy_checks = 25 # how many start times we evaluate
        self.meta_iter_per_check = 200 # for the meta agents
        self.results = []
        self.final_results_path = os.path.join(final_results_path, get_time_str())
        try:
            os.makedirs(self.final_results_path)
        except Exception:
            print "[ERR]: Path not created"
            pass

        self.subagent_path_root = subagent_path_root

        steps = self.max_start_time / self.num_policy_checks
        self.start_times = np.r_[steps :self.max_start_time + 1:steps]

    def run_starttime_experiment(self, save=True):
        for start_time in self.start_times:
            current_agents = self.train_subagents(start_time)
            results = {}
            results['start_time'] = start_time
            results['individual'] = self.run_individual(current_agents).result
            # print "[BEGIN] VOTING EXPERIMENT"
            # results['voting'] = self.run_voting(current_agents).result
            print "[BEGIN] META EXPERIMENT"
            results['meta'] = self.run_meta(current_agents).result
            self.results.append(results)
            # import ipdb; ipdb.set_trace()

        with open(os.path.join(self.final_results_path, "results.p"), "w") as f:
            pickle.dump(self.results, f)

    # def run_confidence_experiment(self, num_pretrain_steps=None, preload=False):
    #     self.base_yaml_file = "individ_params.yml"
    #     if not preload:
    #         assert len(self.all_agent_paths) < 1
    #         self.train_subagents(steps=num_pretrain_steps, base_yaml_root="./params/", save_history=True)
    #     import ipdb; ipdb.set_trace()
    #     current_agents = self.all_agent_paths[0]
    #     exp = confidence_experiment(agent_paths=current_agents, temp=5,
    #                                 sublearning=True, init_state=(0,1.3, 0, 0))
    #     exp.run(visualize_performance=True)
    #     # set exp.path
    #     if save:
    #         exp.save()
    #     return exp

    def run_confidence_experiment(self, agent_paths, save=False):
        exp = confidence_experiment(agent_paths=agent_paths, 
                                    sublearning=True)
        exp.run(visualize_steps=False, visualize_performance=10, debug_on_sigurg=True)
        # set exp.path
        if save:
            exp.save()
        return exp

    # def run_full_individual(self, num_pretrain_steps, save=False):
    #     self.base_yaml_file = "individ_params.yml"
    #     current_agents = self.train_subagents(num_pretrain_steps, base_yaml_root="./params/", save_history=True)
    #     # exp = confidence_experiment(agent_paths=current_agents, mapname=self.mapname, sublearning=True)
    #     # exp.run(visualize_performance=True)
    #     return exp

    def run_individual(self, agent_paths):
        exp = individual_experiment(agent_path=agent_paths[0], mapname=self.mapname)
        exp.max_steps = self.delay
        exp.checks_per_policy = self.meta_iter_per_check
        exp.num_policy_checks = self.num_policy_checks
        exp.run()
        return exp

    def run_votingupdate_experiment(self, agent_paths, paramfile="individ_params.yml", pretrained=False):
        # import ipdb; ipdb.set_trace()
        self.logger.info("Running voting update experiment")
        exp = voting_experiment(agent_paths=agent_paths, temp=1,
                                    pretrained=pretrained,
                                    yaml_file=paramfile,
                                    sublearning=True, init_state=(0,1.3, 0, 0),
                                    exp_id=1)
        # exp.max_steps = self.delay
        # exp.checks_per_policy = self.meta_iter_per_check
        # exp.num_policy_checks = 2
        exp.run(visualize_performance=0)
        return exp

    def run_meta(self, agent_paths):
        exp = metarl_experiment(agent_paths=agent_paths, mapname=self.mapname)
        exp.max_steps = self.delay
        exp.checks_per_policy = self.meta_iter_per_check
        exp.num_policy_checks = self.num_policy_checks
        exp.run(visualize_performance=False)
        return exp

    def train_subagents(self, steps=None, base_yaml_root="./params/", save_history=False):
        base_yaml_file = "individ_params.yml"
        yaml_file = open(os.path.join(base_yaml_root, base_yaml_file), "r")
        # import ipdb; ipdb.set_trace()
        shutil.copy(yaml_file.name, os.path.join(self.final_results_path, "params.yml"))

        base_yaml = yaml.load(yaml_file)

        starts = [(-2, 0.8, 0, 2.5), (1.3,0.6, 0, -1.5), (-1.3,-.4, 0, 0.15), (1.5,-1.97, 0, -1.45), (0.6,-1.9, 0, -0.8), ]
        paths = []
        try:
            os.mkdir(self.subagent_path_root)
        except Exception:
            pass
        for pt in starts:
            base_yaml['domain_params']['init_state'] = pt
            # base_yaml['domain_params']['mapname'] = "12x12-BridgeSpecial.txt" if pt[0] == 3 else "12x12-BridgeBad.txt"
            temp_path_root = "./tmp/"
            with open(os.path.join(temp_path_root, "params.yml"), "w") as f:
                yaml.dump(base_yaml, f)
            agent_exp = AgentGenerator.make_experiment(
                                            temp_path_root, 
                                            result_path=os.path.join(self.subagent_path_root, base_yaml['domain']),
                                            save=True)

            if steps:
                agent_exp.max_steps = steps
            print "Steps: {}".format(agent_exp.max_steps)
            paths.append(agent_exp.path)
            agent_exp.run(visualize_performance=False, debug_on_sigurg=True)

            # if save_history:
            #     agent_exp.agent.policy.save_history(agent_exp.path)

            representation = agent_exp.agent.representation

            assert hasattr(representation, "dump_to_directory")
            representation.dump_to_directory(agent_exp.path)
            print representation.rbfs_mu[0], representation.rbfs_sigma[0]            

        yaml_file.close()
        # import ipdb; ipdb.set_trace()
        self.all_agent_paths.append(paths)
        return paths

    def load_results(self, resultpath):
        with open(resultpath, "r") as f:
            self.results = pickle.load(f)

    def load_agents(self, agent_paths):
        print "[LOAD] Loading agents into all_agent_paths"
        self.all_agent_paths.append(agent_paths) 
        return self.all_agent_paths       

    def plot_delay(self):
        individual = []
        voting = []
        meta = []
        fig = plt.figure()
        for result in self.results:
            idx = result['meta']['learning_steps'].index(self.delay)
            individual.append(result['individual']['return'][idx])
            # voting.append(result['voting']['return'][idx])
            meta.append(result['meta']['return'][idx])

        plt.plot(self.start_times, individual, label='individual')
        # plt.plot(self.start_times, voting, label='voting')
        plt.plot(self.start_times, meta, label='meta')
        plt.show()
        plt.legend(loc='upper center',
                                       bbox_to_anchor=(0.5, -0.15),
                                       fancybox=True, shadow=True, ncol=2)
        plt.title("Evaluating after {} steps".format(self.delay))

    def plot_agent_experience(self, agents):
        fig = plt.figure()
        for agent in agents:
            hist = agent.history
            states = ((x, y) for x, y, theta in hist.keys())
            plt.scatter(*zip(*states))

        plt.show()


if __name__ == '__main__':
    # Trained for 40000 - regular, nonvoting
 #    agents = ['./GeneratedAgents/RCCarConfidence/Domains.RCCarModified/Jul11_06-51-369621',
 # './GeneratedAgents/RCCarConfidence/Domains.RCCarModified/Jul11_06-54-712478',
 # './GeneratedAgents/RCCarConfidence/Domains.RCCarModified/Jul11_06-54-981478',
 # './GeneratedAgents/RCCarConfidence/Domains.RCCarModified/Jul11_06-55-419938',
 # './GeneratedAgents/RCCarConfidence/Domains.RCCarModified/Jul12_06-56-228023']
    train = False

    paths = ['./GeneratedAgents/RCCarConfidenceMaha/Domains.RCCarModified/Jul21_02-20-285239',
              './GeneratedAgents/RCCarConfidenceMaha/Domains.RCCarModified/Jul21_02-21-443177',
              './GeneratedAgents/RCCarConfidenceMaha/Domains.RCCarModified/Jul21_02-23-322976', 
              './GeneratedAgents/RCCarConfidenceMaha/Domains.RCCarModified/Jul21_02-24-460192', 
              './GeneratedAgents/RCCarConfidenceMaha/Domains.RCCarModified/Jul21_02-25-750812']

    exp = StartTimeExperiment()
    import ipdb; ipdb.set_trace()  # breakpoint d750fda4 //
    
    if train:
        paths = exp.train_subagents(steps=80000)
        print paths
    else:
        exp.load_agents(paths)
        conf = exp.run_confidence_experiment(exp.all_agent_paths[0])
        # vot = exp.run_votingupdate_experiment(exp.all_agent_paths[0], pretrained=False)
        # exp.run()
        # vot.plot()
        # vot.save()
        # conf.save()
        conf.plot()
        # import ipdb; ipdb.set_trace()

# []
