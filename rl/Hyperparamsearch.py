from rlpy.Tools.hypersearch import find_hyperparameters
import os
import sys

# cur_dir = os.path.expanduser("~/work/clipper/models/rl/")
# sys.path.append(cur_dir)

if __name__ == '__main__':
	experiment_name = sys.argv[1]
	best, trials = find_hyperparameters(
	    experiment_name + "Experiment.py",
	    "./Results/"+experiment_name+"/paramsearchnext",
	    max_evals=10, parallelization="joblib",
	    trials_per_point=6)
	print "============== This is the best parameters"
	print best
