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
pip install gymnasium

# Install minigrid
pip install minigrid


GYM_PATH=$(python -c "import gymnasium, os; print(os.path.dirname(gymnasium.__file__))")
mkdir -p ${GYM_PATH}/envs/pomdp
cp ${AIS_DIR}/pomdp/*.py ${GYM_PATH}/envs/pomdp
cat ${AIS_DIR}/__init__.py >> ${GYM_PATH}/envs/__init__.py