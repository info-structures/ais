#!/usr/bin/env bash
userhome=~
vm_dir=${userhome}/python-vms
project_name=ais
experiment_name=exp1
virtualhome=${vm_dir}/${project_name}

mkdir -p ${userhome}/scratch/${project_name}/${experiment_name}/output_logs

MEMORY=2G
TIME=01:00:00
CPUS_PER_TASK=1
CC_ACCOUNT=def-adityam

output_dir=${userhome}/scratch/${project_name}/${experiment_name}/results
mkdir -p ${output_dir}

eval_frequency=100
N_eps_eval=10
beta=0.99
batch_size=200
num_batches=1000

env_names=(MiniGrid-Empty-8x8-v0)

#lmbdas is for the reward weight
lmbdas=(0.1 0.01)

#LRs are given as corresponding pairs
ais_LRs=(0.001)
policy_LRs=(0.0006)

AIS_state_size=(128)
seeds=(0 1)

echo 'Submitting SBATCH jobs...'
for env_name in ${env_names[@]}
do
	for lmbda in ${lmbdas[@]}
	do
		for ((i_LR = 0; i_LR < ${#ais_LRs[@]}; i_LR++))
		do
			ais_LR=${ais_LRs[$i_LR]}
			policy_LR=${policy_LRs[$i_LR]}
			for ais_ss in ${AIS_state_size[@]}
			do
				for seed in ${seeds[@]}
				do
					echo "#!/bin/bash" >> temprun.sh
					echo "#SBATCH --account=${CC_ACCOUNT}" >> temprun.sh
					echo "#SBATCH --job-name=env${env_name}_Lmbda${lmbda}_AIS-LR${ais_LR}_policy-LR${policy_LR}_AIS-SS${ais_ss}_seed${seed}" >> temprun.sh
					echo "#SBATCH --output=$userhome/scratch/${project_name}/${experiment_name}/output_logs/%x-%j.out" >> temprun.sh
					echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> temprun.sh
					echo "#SBATCH --mem=${MEMORY}" >> temprun.sh
					echo "#SBATCH --time=${TIME}" >> temprun.sh

					echo "source $virtualhome/bin/activate" >> temprun.sh
					echo "cd $virtualhome/code/ais" >> temprun.sh
					echo "python src/main.py \
					--env_name=${env_name}\
					--AIS_state_size=${ais_ss}\
					--seed=${seed}\
					--output_dir=${output_dir}\
					--eval_frequency=${eval_frequency}\
					--N_eps_eval=${N_eps_eval} --beta=${beta}\
					--lmbda=${lmbda}\
					--policy_LR=${policy_LR}\
					--ais_LR=${ais_LR}\
					--batch_size=${batch_size}\
					--num_batches=${num_batches}"  >> temprun.sh

					eval "sbatch temprun.sh"
					#added to submit job again if slurm error occurs (timeout error send/recv)
					while [ ! $? == 0 ]
					do
						eval "sbatch temprun.sh"
					done

					# sleep 1
					rm temprun.sh
				done
			done
		done
	done
done
