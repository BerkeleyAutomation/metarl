from rlpy.Tools.run import run
import rlpy.Tools.results as rt
import os
import sys

if __name__ == '__main__':
	experiment_name = sys.argv[1]

	experiment_dir = os.path.expanduser("~/work/clipper/models/rl/")
	result_dir =  "./Results/" + experiment_name + "/"
	
	run(experiment_dir + experiment_name + "Experiment.py", result_dir,
	    ids=range(6), parallelization="joblib")

	paths = {experiment_name: result_dir}

	merger = rt.MultiExperimentResults(paths)
	fig = merger.plot_avg_sem("learning_steps", "return")
	rt.save_figure(fig, result_dir + "plot.pdf")