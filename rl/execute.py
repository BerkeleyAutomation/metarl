import logging
import joblib

class ExperimentGenerator():

    def __init__(self, yaml_file_path):
        self.logger = logging.getLogger('ExperimentCreator')
        self.logger.info("Loading parameters from {yaml_file_path}")
        self.yaml_file_path = yaml_file_path
        self.params = load_yaml(yaml_file_path)
        self.experiment = None

    def create_domain(self, params):
        domain_params = params
        domain_cls = eval(domain_params['class'])
        domain = domain_cls(**domain_params['params'])
        return domain

    def create_representation(self, params):
        representation_params = params 
        representation_cls = eval(representation_params['class'])
        domain = self.create_domain(representation_params['domain'])
        representation = representation_cls(domain, **representation_params['params'])
        return representation

    def create_policy(self, params):
        policy_params = params 
        policy_cls = eval(policy_params['class'])
        representation = self.create_representation(policy_params['representation'])
        policy = policy_cls(representation, **policy_params['params'])
        return policy

    def create_agent(self, params):
        agent_params = params 
        agent_cls = eval(agent_params['class'])
        policy = self.create_policy(agent_params['policy'])
        agent = agent_cls(policy, policy.representation, **agent_params['params'])
        return agent

    def make_experiment(self):
        self.logger.info("Creating experiment from {yaml_file_path}")
        exper_params = self.params['experiment']
        experiment_cls = eval(exper_params['class'])
        exper_params['path'] = X TODO
        exper_params['agent'] = self.create_agent() TODO
        exper_params['domain'] = exper_params['agent'].representation
        self.experiment = experiment_cls(**exper_params)
        return self.experiment



class PreloadingExperiment(ExperimentGenerator):

    def __init__(self, yaml_file_path):
        super(PreloadingExperiment, self).__init__(yaml_file_path)

    def create_subagents(self, params, pretrained=False):
        all_subagent_params = params
        agents = []
        for agent_param in all_subagent_params.values():
            subagent = self.create_agent(agent_param)

            if pretrained:
                subagent = self.load_experience(subagent)

            agents.append(subagent)

        return agent

    def create_policy(self, params):
        policy_params = params 
        policy_cls = eval(policy_params['class'])
        representation = self.create_representation(policy_params['representation'])
        subagents = self.create_subagents(params['subagents'], pretrained=False)
        policy = policy_cls(representation, subagents, **policy_params['params'])
        return policy

    def load_experience(self):
        pass

    def make_experiment(self):

        experiment = super.make_experiment()
        for agent in experiment.agent.policy.subagents:
            agent.random_state = np.random.RandomState(
                experiment.randomSeeds[experiment.exp_id - 1])
            agent.init_randomization()
            # agent.representation.random_state = np.random.RandomState(
            #     experiment.randomSeeds[experiment.exp_id - 1])
            # agent.representation.init_randomization() #init_randomization is called on instantiation
            agent.policy.random_state = np.random.RandomState(
                experiment.randomSeeds[experiment.exp_id - 1])
            agent.policy.init_randomization()
        self.experiment = experiment
        return experiment



def make_experiment():
    """
    :params param_root: location of parameter
    :params result_path: location of results/weights
    """

    param_path = os.path.join(param_root, yaml_file) 
    print "[INFO] Loading agent from " + param_path
    params = type("Parameters", (), load_yaml(param_path))

    # params.domain_params['mapname'] = os.path.join(os.path.expanduser(params.map_dir),
    #                                                 params.domain_params['mapname'])
    # import ipdb; ipdb.set_trace()

    domain = eval(params.domain)(**params.domain_params)
    # domain = eval(params.domain)()

    #Load Representation
    representation = eval(params.representation)(
                domain, 
                **params.representation_params)
    policy = eval(params.policy)(
                representation, 
                **params.policy_params)
    agent = eval(params.agent)(
                policy, 
                representation,
                discount_factor=domain.discount_factor, 
                **params.agent_params)

    opt = {}
    opt["exp_id"] = params.exp_id if params.exp_id else np.random.randint(100)
    print "using seed of %d" % opt["exp_id"]
    # import ipdb; ipdb.set_trace()
    opt["path"] = os.path.join(result_path, get_time_str())
    # opt["max_eps"] = params.max_eps

    opt["max_steps"] = params.max_steps
    opt["num_policy_checks"] = params.num_policy_checks
    opt["checks_per_policy"] = params.checks_per_policy

    opt["domain"] = domain
    opt["agent"] = agent

    if save:

        path_join = lambda s: os.path.join(opt["path"], s)
        if not os.path.exists(opt["path"]):
            os.makedirs(opt["path"])

        shutil.copy(param_path, path_join("params.yml"))
        shutil.copy(inspect.getsourcefile(eval(params.domain)), path_join("domain.py"))
        shutil.copy(inspect.getsourcefile(inspect.currentframe()), path_join("experiment.py"))
        

    return eval(params.experiment)(**opt)