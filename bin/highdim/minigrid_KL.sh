#!/bin/bash

#list of environments used
#Simple crossing: MiniGrid-SimpleCrossingS9N1-v0, MiniGrid-SimpleCrossingS9N2-v0, MiniGrid-SimpleCrossingS9N3-v0, MiniGrid-SimpleCrossingS11N5-v0
#Lava crossing: MiniGrid-LavaCrossingS9N1-v0, MiniGrid-LavaCrossingS9N2-v0
#Key corridor: MiniGrid-KeyCorridorS3R1-v0, MiniGrid-KeyCorridorS3R2-v0, MiniGrid-KeyCorridorS3R3-v0
#Obstructed maze: MiniGrid-ObstructedMaze-1Dl-v0, MiniGrid-ObstructedMaze-1Dlh-v0
#Misc: MiniGrid-Empty-8x8-v0, MiniGrid-DoorKey-8x8-v0, MiniGrid-FourRooms-v0

OUTPUT_DIR=Results
ENV_NAME=MiniGrid-SimpleCrossingS9N1-v0
EVAL_FREQ=1000
N_EPS_EVAL=20
BETA=0.99
LMBDA=0.1
POLICY_LR=0.0007
AIS_LR=0.001
BATCH_SIZE=200
NUM_BATCHES=100000
AIS_SS=128
AIS_PN=5
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
					--seed ${SEED} \
					--AIS_pred_ncomp ${AIS_PN}