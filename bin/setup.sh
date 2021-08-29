#!/bin/bash
userhome=~
VMS_DIR=${userhome}/python-vms
PROJECT=ais
my_repo=https://github.com/info-structures/ais

mkdir -p ${VMS_DIR}
virtualenv --no-download -p python3 ${VMS_DIR}/${PROJECT}
source ${VMS_DIR}/${PROJECT}/bin/activate
pip install --upgrade pip

cd ${VMS_DIR}/${PROJECT}
mkdir code
cd code
git clone ${my_repo}
git checkout ap1_ap2

AIS_DIR=${VMS_DIR}/${PROJECT}/code/ais/lib/gym/envs

# Install libraries
pip install numpy torch torchvision matplotlib wandb

# Install Gym
mkdir -p ${VMS_DIR}/${PROJECT}/libraries 
cd ${VMS_DIR}/${PROJECT}/libraries
git clone https://github.com/openai/gym
cd gym
pip install -e .

# Install Gym-minigrid
cd ${VMS_DIR}/${PROJECT}/libraries
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install -e .

# Install AIS
mkdir -p ${VMS_DIR}/${PROJECT}/libraries/gym/gym/envs/pomdp
cp ${AIS_DIR}/pomdp/*.py ${VMS_DIR}/${PROJECT}/libraries/gym/gym/envs/pomdp
cat ${AIS_DIR}/__init__.py >> ${VMS_DIR}/${PROJECT}/libraries/gym/gym/envs/__init__.py
