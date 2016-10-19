from rlpy.Tools.results import MultiExperimentResults, load_single
from rlpy.Tools import results as rt
import sys
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
# agents = ['./Results/Car/Agent1Post/Aug16_08-24-086841', './Results/Car/Agent1Post/Aug16_08-33-831932']

# paths = {"Meta0": "./Results/CarMixed2/combine/Aug16_04-53-934254/",
# 		 "Meta1": "./Results/CarMixed2/combine/Aug16_04-59-246235/",
# 		 "Meta2": "./Results/CarMixed2/combine/Aug16_05-05-251391/",
# 		 "Meta4": "./Results/CarMixed2/combine/Aug16_05-13-666296/",
# 		 "rerun0": agents[0],
# 		 "rerun1": agents[1]
# 		}

agents = ['./Results/Car/Agent1Post/Aug16_08-24-086841', './Results/Car/Agent1Post/Aug16_08-33-831932']

paths = {"Meta0": './Results/CarMixed2/combine/Aug17_12-06-345773',
		 # "Meta1": "./Results/CarMixed2/combine/Aug16_04-59-246235/",
		 # "Meta2": "./Results/CarMixed2/combine/Aug16_05-05-251391/",
		 "Meta4": './Results/CarMixed2/combine/Aug17_12-34-416129',
		 "rerun0": agents[0],
		 "rerun1": agents[1]
		}

# paths = {"egreed": "/home/jarvis/work/clipper/models/rl/Results/CarSlide/combine/Aug19_01-45-210261/"}
paths = {"mixed": sys.argv[1]}

# paths = { #"QLearning": "./Results/Gridworld/Agent0/Aug03_10-47-459778/",
# 		 "2 agents": "./Results/Mixed2/combine/Aug04_02-24-597933", 
# 		 "3 agents": "./Results/Mixed3/combine/Aug04_02-11-618909",
# 		 "4 agents": "./Results/Mixed4/combine/Aug04_02-21-237975"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_episode", "return")
text = "These experiments used the following: " + "\n" + "\n ".join(paths.values()) + "\n\n2 trained agents and X Noop agents were used."
fig.text(.1, -0.5, text)
rt.save_figure(fig, "tmp/test.pdf")
