#!/usr/bin/env bash
userhome=~
vm_dir=${userhome}/python-vms
project_name=ais
experiment_name=exp1
virtualhome=${vm_dir}/${project_name}

results_dir=${userhome}/scratch/${project_name}/${experiment_name}/results

source $virtualhome/bin/activate
cd $results_dir

for d in */ ; do
	python src/wandb/logger.py --results_dir ${results_dir}/${d} --project_name ${project_name} --run_name ${experiment_name} --wandb_mode online
done