#!/bin/bash

OUTPUT_DIR=Results
ENV_NAME=RockSampling-v0
EVAL_FREQ=500
N_EPS_EVAL=100
BETA=0.99
LMBDA=0.0001
POLICY_LR=0.0007
AIS_LR=0.003
BATCH_SIZE=200
NUM_BATCHES=100000
AIS_SS=128
IPM=KL
SEED=42

python src/main.py --output_dir ${OUTPUT_DIR} \
					--env_name ${ENV_NAME} \
					--eval_frequency ${EVAL_FREQ} \
					--N_eps_eval ${N_EPS_EVAL} \
					--beta ${BETA} \
					--lmbda ${LMBDA} \
					--policy_LR ${POLICY_LR} \
					--ais_LR ${AIS_LR} \
					--batch_size ${BATCH_SIZE} \
					--num_batches ${NUM_BATCHES} \
					--AIS_state_size ${AIS_SS} \
					--IPM ${IPM} \
					--seed ${SEED}