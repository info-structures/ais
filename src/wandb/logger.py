import os
import numpy as np
import argparse
import json
import wandb

parser = argparse.ArgumentParser(description='This script uploads the performance and args to wandb for a given run')

parser.add_argument("--results_dir", help="Directory containing results of a single run (and args.json)")
parser.add_argument("--project_name", help="Name of project to log data on wandb", default="ais")
parser.add_argument("--run_name", help="Name of run to log data on wandb", default="exp1")
parser.add_argument("--wandb_mode", help="Log wandb stuff `online` or `offline`", default="online")

args = parser.parse_args()

wandb.init(
	project=args.project_name,
	name=args.run_name,
	dir=args.results_dir,
	mode=args.wandb_mode
)

with open(os.path.join(args.results_dir, 'args.json')) as args_json:
	args_data = json.load(args_json)
	wandb.config.update(args_data)

perf_data = np.load(os.path.join(args.results_dir, 'perf.npz'))

for i in range(len(perf_data['perf'])):
	wandb.log({"Performance": perf_data['perf'][i], "Samples": perf_data['samples'][i]})