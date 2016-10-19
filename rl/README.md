# Keeping track of Experiment files:
Update as of 8/9: Gridworld experiments have been moved into `GridWorldFiles`.
For the car experiment, main files are:

### [MetaRCCarDynamics.py](MetaRCCarDynamics.py) 
Has experiments for running the MetaRL agent and training subagents, (ie `run_2_agents`)

### [SimpleMultiTrials.py](SimpleMultiTrials.py)
Has experiments for training and plotting performance of individual agents, also loading trained agents onto new domain.

### [AgentGenerator.py](AgentGenerator.py)
Generates an agent (along with environment, policy, representation) from yaml file. Used in multiple files.

### [plot_results.py](plot_results.py)
Uses rlpy's pretty plotting to get performance + confidence intervals.

### [params/Car/](params/Car/)
YML files for subagents used. `agent0.yml` and `agent1.yml` are the ones of interest.

A lot of OOP is used, so some digging around may be needed.

## In GRIDWORLDFILES, following files are found:

### `Mixed_Experiment.py`
This file includes experiments for mixing domains with different dynamics.
 - (`generate_Qexperiment`) Using Q values for consistency
 - (`generate_mixed_experiment`) Generating confidence values and fitting classifier over them
   - MultiTerrainPolicy, MetaDynamicsAgent
 - (`generate_multinomial_experiment`) Using a multinomial distribution to choose action based on previously observed transition
  - MAgentMultinomial, NoopAgent

### `MetaRLDynamicsExperiment.py`
Has experiments for Gridworld running different meta agents and different dynamics 

### `GW_ExperParams.yml`
Contains parameters for all gridworld experiments.

	
