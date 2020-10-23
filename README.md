
# AIS based PORL
This repository contains the code for the PORL (partially observed reinforcement learning) experiments presented in the paper
> J. Subramanian, A. Sinha, R. Seraj, and A. Mahajan, "[Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observable Environments][paper]", 2020. 

[paper]: http://arxiv.org/abs/2010.08843

Three classes of experiments are presented (with their `gym-environment-names`)
* **Low-dimensional environments**
  - Tiger: `Tiger-v0`
  - Voicemail: `Voicemail-v0`
  - Cheese Maze: `CheeseMaze-v0`
 * **Moderate-dimensional environments**
   - Rock Sample: `RockSampling-v0`
   - Drone Surveillance: `DroneSurveillance-v0`
 * **High-dimensional environments**

    Various grid-world models from [gym-minigrid](https://github.com/maximecb/gym-minigrid) (used in the BabyAI platform) including
    - Simple crossing: `MiniGrid-SimpleCrossingS9N1-v0`, `MiniGrid-SimpleCrossingS9N2-v0`, `MiniGrid-SimpleCrossingS9N3-v0`, `MiniGrid-SimpleCrossingS11N5-v0`
    - Lava crossing: `MiniGrid-LavaCrossingS9N1-v0`, `MiniGrid-LavaCrossingS9N2-v0`
    - Key corridor: `MiniGrid-KeyCorridorS3R1-v0`, `MiniGrid-KeyCorridorS3R2-v0`, `MiniGrid-KeyCorridorS3R3-v0`
    - Obstructed maze: `MiniGrid-ObstructedMaze-1Dl-v0`, `MiniGrid-ObstructedMaze-1Dlh-v0`
    - Misc `MiniGrid-Empty-8x8-v0`, `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-FourRooms-v0`


### Installation

To install all the dependencies of the code in a virtual environment, run the setup
script: 

    bash bin/setup.sh

## Usage

Fist activate the virtual environment using:

    source python-vms/ais/bin/activate

To run AIS training algorithm for an environment, say `Tiger-v0`, run:

    python src/main.py --env_name Tiger-v0 

This program accepts the following command line arguments:

| Option          | Description |
| --------------- | ----------- |
| `--output_dir` | The results are stored in this directory. |
| `--env_name` | The environment name (in open-ai gym format) |
| `--eval_frequency` | Number of batch iterations per evaluation step. |
| `--N_eps_eval` | Number of episodes to evaluate in an evaluation step. |
| `--beta` | Discount Factor |
| `--lmbda` |  Trade-off between the next reward loss and next observation loss. It generally helps to keep this value low if the rewards of the environment are high. |
| `policy_LR` |  Learning rate used by the ADAM optimizer for the policy. |
| `ais_LR` | Learning rate used by the ADAM optimizer for the ais. |
| `batch_size` | Number of samples used in a batch for every optimization step. |
| `num_batches` |  Number of batches to train on. |
| `AIS_state_size` | Size of the vector used to represent the approximate information state. |
| `--AIS_pred_ncomp` | Number of components in the GMM for MiniGrid with the KL IPM. |
| `--IPM` |  The IPM can be specified using this argument. In this code, `MMD` can be used to use the L2-norm squared form of the kernel based IPM. Or `KL` can be used to indirectly optimize for the Wasserstein IPM. |
| `--seed` | Random seed used. |
| `--models_folder` |  Directory to save/load models |

## Reproducing results in the paper

The results presented in the paper can be obtained by running the following
wrapper scripts:

* **Low-dimensional environments**
  - Tiger: `sh bin/lowdim/tiger_MMD.sh` and `sh bin/lowdim/tiger_KL.sh`
  - Voicemail: `sh bin/lowdim/voicemail_MMD.sh` and `sh bin/lowdim/voicemail_KL.sh`
  - Cheese Maze: `sh bin/lowdim/cheesemaze_MMD.sh` and `sh bin/lowdim/cheesemaze_KL.sh`
* **Moderate-dimensional environments**
  - Rock Sample: `sh bin/moddim/rocksampling_MMD.sh` and `sh bin/moddim/rocksampling_KL.sh`
  - Drone Surveillance: `sh bin/moddim/dronesurveillance_MMD.sh` and `sh bin/moddim/dronesurveillance_KL.sh`

* **High-dimensional environments**
  - `sh bin/highdim/minigrid_MMD.sh` and `sh bin/highdim/minigrid_KL.sh`

      This runs `SimpleCrossingS9N1` environment. To run other environments,
      the name of the environment must be changed in
      `bin/highdim/minigrid_*.sh` files.

# Citation
Please use the following citation to refer to the paper:

```
@misc{AIS,
      title={Approximate information state for approximate planning and reinforcement learning in partially observed systems}, 
      author={Jayakumar Subramanian and Amit Sinha and Raihan Seraj and Aditya Mahajan},
      year={2020},
      note={arXiv:2010.08843},
      url={https://arxiv.org/abs/2010.08843},

}
```
