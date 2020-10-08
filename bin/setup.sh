#!/bin/bash

VMS_DIR=${PWD}/python-vms
PROJECT=ais
AIS_DIR=${PWD}/lib/gym/envs

mkdir -p ${VMS_DIR}
virtualenv --no-download -p python3 ${VMS_DIR}/${PROJECT}
source ${VMS_DIR}/${PROJECT}/bin/activate
pip install --upgrade pip

# Install libraries
pip install numpy torch torchvision matplotlib scikit-image pygame

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
