from rlpy.Tools.results import MultiExperimentResults, load_single, save_figure
import glob, os, sys
import numpy as np
################
#Visualization for Meta RL using 11x11-Rooms3.txt and 2 robots. Agent0 is normal and Agent1 is reversed
#Final Meta Map is setup such that rows < 5 is reversed.
# Reversed means UP -> DOWN and DOWN -> UP. 
	# opt["max_steps"] = 8000
	# opt["num_policy_checks"] = 20
	# opt["checks_per_policy"] = 40
#################
# paths = {"Meta RL": "./Results/Meta/Harder11x11maze/Aug01_10-04-831235",
# 		 "Agent0": "./Results/Gridworld/Agent0/Aug01_10-46-626985/",
# 		 "Agent1": "./Results/Gridworld/Agent0/Aug01_11-17-419997/",
# 		 "Multinomial": "./Results/Mixed/Easier11x11/Aug01_11-49-391406/"}


################
#Visualization for Meta RL using 11x11-Rooms.txt and 2 robots. Agent0 is normal and Agent1 is reversed
#Final Meta Map is setup such that rows < 5 is reversed.
# Reversed means UP -> DOWN and DOWN -> UP. 
#################
# paths = {"Meta RL": "./Results/Meta/Easier11x11maze/Aug01_11-29-876491", # FOR JARVIS
# 		 "Agent0": "./Results/Gridworld/Agent0/Aug01_11-32-629441/",
# 		 "Agent1": "./Results/Gridworld/Agent0/Aug01_11-27-450888/",
# 		 "Multinomial": "./Results/Mixed/Easier11x11/Aug01_11-44-170586/"}

# paths = {"Meta": "./Results/CarMixed2/combine/Aug08_04-45-385689",
# 		 "Agent1": "./Results/Car/Agent1/Aug08_04-17-792181",
# 		 "Agent0": "./Results/Car/Agent0/Aug08_04-13-758570"}

def movingaverage(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

def load_trials(path):
	"""
	returns a dictionary with the results of each run of an experiment stored
	in path
	The keys are the seeds of the single runs
	"""
	trials = {}
	for fn in glob.glob(os.path.join(path, '*-trials.json')):
		cur_trial = load_single(fn)

		cur_trial['avg_return'] = movingaverage(cur_trial['return'], 2)
		cur_trial['avg_0'] = movingaverage(cur_trial['Action_0'], 10)
		cur_trial['avg_1'] = movingaverage(cur_trial['Action_1'], 50)
		print cur_trial['avg_0'][-10:]
		print cur_trial['avg_1'][-10:]
		trials[cur_trial["seed"]] = cur_trial
	return trials

class MultiExperimentTrials(MultiExperimentResults):

	"""provides tools to analyze, compare, load and plot results of several
	different experiments each stored in a separate path"""

	def __init__(self, paths):
		"""
		loads the data in paths
		paths is a dictionary which maps labels to directories
		alternatively, paths is a list, then the path itself is considered
		as the label
		"""
		self.data = {}
		if isinstance(paths, list):
			paths = dict(zip(paths, paths))
		for label, path in paths.iteritems():
			self.data[label] = load_trials(path)



agents = ['./Results/Car/Agent1Post/Aug16_08-24-086841', './Results/Car/Agent1Post/Aug16_08-33-831932']

paths = {"Meta0": './Results/CarMixed2/combine/Aug17_12-06-345773',
		 # "Meta1": "./Results/CarMixed2/combine/Aug16_04-59-246235/",
		 # "Meta2": "./Results/CarMixed2/combine/Aug16_05-05-251391/",
		 "Meta4": './Results/CarMixed2/combine/Aug17_12-34-416129',
		 "rerun0": agents[0],
		 "rerun1": agents[1]
		}
paths = {"mixed": sys.argv[1], #for slide left
		 "slide only": "./Results/CarSlideTurn/combine/Aug21_01-11-478424",
		 "turn only": "./Results/CarSlideTurn/combine/Aug21_01-09-255914"}

paths = {"mixed": sys.argv[1], #for slide left
		 "slide only": "./Results/CarSlideTurn/combine/Aug21_01-25-746419",
		 "turn only": "./Results/CarSlideTurn/combine/Aug21_01-38-448884"}

paths = {"mixed": sys.argv[1], #for mixed
		 # "random_local": "./Results/CarSlideTurn/combine/Aug21_09-22-664025",
		 # "turn only": ""
		 }
# paths = {"temp": "./GeneratedAgents/RCCarSlide/Aug19_05-26-824448"}
# paths = { #"QLearning": "./Results/Gridworld/Agent0/Aug03_10-47-459778/",
# 		 "2 agents": "./Results/Mixed2/combine/Aug04_02-24-597933", 
# 		 "3 agents": "./Results/Mixed3/combine/Aug04_02-11-618909",
# 		 "4 agents": "./Results/Mixed4/combine/Aug04_02-21-237975"}

merger = MultiExperimentTrials(paths)
fig = merger.plot_avg_sem("learning_episode", "avg_return")
save_figure(fig, "tmp/test2.pdf")
