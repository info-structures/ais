import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import gym
import time
from sklearn.cluster import k_means as k_means
from sklearn.manifold import TSNE
from train_manager import batch_creator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import MSELoss as MSELoss
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

parser = argparse.ArgumentParser(description='This code runs AIS using the Next Observation prediction version')

parser.add_argument("--output_dir", help="Directory to store output results in", default="results")
parser.add_argument("--env_name",
                    help="Gym Environment to Use. Options: `Tiger-v0`, `Voicemail-v0`, `CheeseMaze-v0`\n `DroneSurveillance-v0`, `RockSampling-v0`",
                    default="CheeseMaze-v0")
parser.add_argument("--eval_frequency", type=int, help="Number of Batch Iterations per Evaluation", default=100)
parser.add_argument("--N_eps_eval", type=int, help="Number of Episodes per Evaluation", default=500)
parser.add_argument("--beta", type=float, help="Discount Factor", default=0.95)
parser.add_argument("--lmbda", type=float,
                    help="lambda value (Trade off between next reward and next observation prediction)", default=0.0001)
parser.add_argument("--policy_LR", type=float, help="Learning Rate for Policy Network", default=0.0007)
parser.add_argument("--ais_LR", type=float, help="Learning Rate for AIS Netowrk", default=0.003)
parser.add_argument("--batch_size", type=int, help="Number of samples per batch of training", default=200)
parser.add_argument("--num_batches", type=int, help="Number of batches used in training", default=2000)
parser.add_argument("--AIS_state_size", type=int, help="Size of the AIS vector in the neural network", default=15)
parser.add_argument("--IPM", help="Options: `KL`, `MMD`", default='MMD')
parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
parser.add_argument("--models_folder", type=str, help='Pretrained model (state dict)',
                    default="results/eqn_60/CheeseMaze-v0_500_50_0.7_0.0001_0.0006_0.003_200_20000_15_KL/models/seed42")
parser.add_argument("--AIS_pred_ncomp", type=int,
                    help="Number of Components used in the GMM to predict next AIS (For MiniGrid+KL)", default=5)

args = parser.parse_args()

env = gym.make(args.env_name)
eval_frequency = args.eval_frequency
N_eps_eval = args.N_eps_eval
beta = args.beta  # beta is the discount factor
lmbda = args.lmbda
policy_LR = args.policy_LR
ais_LR = args.ais_LR
batch_size = args.batch_size
num_batches = args.num_batches
AIS_SS = args.AIS_state_size  # d_{\hat Z}
AIS_PN = args.AIS_pred_ncomp  # this arg is only used for MiniGrid with the KL IPM
IPM = args.IPM
seed = args.seed

# set random seeds
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)


def eval_performance(bc, greedy=False, n_episodes=1):
    with torch.no_grad():
        returns = torch.Tensor([])
        obs = []
        actions = []
        states = []
        initial_state = []
        aiss = []
        for n_eps in range(n_episodes):
            reward_episode = []

            if bc.args.env_name[:8] == 'MiniGrid':
                current_obs = bc.env.reset()['image']
                current_obs = bc.get_encoded_obs(current_obs)
            else:
                next_obs = bc.env.reset()
                current_obs = bc.convert_int_to_onehot(next_obs, bc.env.observation_space.n)
            action = torch.zeros(bc.env.action_space.n)
            reward = 0.
            hidden = None

            initial_state.append(bc.env.current_state)

            obs_history = []
            action_history = []
            state_history = []
            for j in range(1000):
                rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
                ais_z, hidden = bc.rho(rho_input, hidden)
                ais_z = ais_z.reshape(-1)

                # visualize_maze(env.current_state, ais_z.detach().numpy())

                policy_probs = bc.policy(ais_z.detach())
                if greedy:
                    c = Categorical(policy_probs)
                    _, action = policy_probs.max(0)
                else:
                    c = Categorical(policy_probs)
                    action = c.sample()

                obs_history.append(next_obs+1)  # + 1 to shift the index starting from 1
                action_history.append(action.item())

                next_obs, reward, done, _ = bc.env.step(action.item())

                action = bc.convert_int_to_onehot(action, bc.env.action_space.n)
                reward_episode.append(reward)
                state_history.append(bc.env.current_state)

                if bc.args.env_name[:8] == 'MiniGrid':
                    current_obs = next_obs['image']
                    current_obs = bc.get_encoded_obs(current_obs)
                else:
                    current_obs = next_obs
                    current_obs = bc.convert_int_to_onehot(current_obs, bc.env.observation_space.n)

                if done:
                    obs.append(obs_history)
                    actions.append(action_history)
                    states.append((state_history))
                    aiss.append(ais_z.detach().numpy())
                    break

    return obs, actions, states, np.array(initial_state), np.array(aiss)


def plot_single_obs(bc, action_dict):
    ais = {}
    for i in range(7):
        ais[i] = np.zeros((4,AIS_SS))
    for initial_state in range(10):
        for a in range(4):
            hidden = None
            bc.env.reset()
            next_obs = bc.env.set(initial_state)
            current_obs = bc.convert_int_to_onehot(next_obs, bc.env.observation_space.n)
            action = bc.convert_int_to_onehot(a, bc.env.action_space.n)
            rho_input = torch.cat((current_obs, action, torch.Tensor([0]))).reshape(1, 1, -1)
            ais_z, hidden = bc.rho(rho_input, hidden)
            ais_z_value = ais_z.detach().numpy().reshape(-1)
            next_obs, reward, done, _ = bc.env.step(a)
            ais[next_obs][a,:] = ais_z_value
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        for a in range(4):
            plt.plot(ais[i][a,:], label=action_dict[a])
            mean = np.mean(ais[i], axis=0)
            error = np.linalg.norm(ais[i] - mean, 'fro')
        # plt.legend(loc="best")
        plt.xlabel("AIS dim")
        plt.ylabel("AIS value")
        plt.title("Observation " +str(i+1) +", single action")
    plt.show()
    return ais

def plot_trajectory_single_obs(bc, ais, action_dict, vis_maze=False):
    plt.subplot(2, 1, 1)
    for a in range(4):
        plt.plot(ais[0][a, :], label=action_dict[a])
    plt.legend(loc="best")
    plt.xlabel("AIS dim")
    plt.ylabel("AIS value")
    plt.title("Observation 1, single action")

    # Start predefined trajectory 1
    initial_states = [9, 8, 8, 5]
    actions_sequences = [[0, 0, 3, 3, 1, 0, 3, 3],
                         [0, 0],
                         [0, 0, 2, 2, 3, 3],
                         [1, 0, 0, 2, 3]]
    for i in range(len(initial_states)):
        bc.env.reset()
        next_obs = bc.env.set(initial_states[i])
        current_obs = bc.convert_int_to_onehot(next_obs, bc.env.observation_space.n)
        actions = actions_sequences[i]
        reward = 0.
        hidden = None
        aiss = []
        plt.subplot(2, 1, 2)
        for a in actions:
            action = bc.convert_int_to_onehot(a, bc.env.action_space.n)
            rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
            ais_z, hidden = bc.rho(rho_input, hidden)
            ais_z = ais_z.reshape(-1)
            aiss.append(ais_z)

            if vis_maze:
                visualize_maze(env.current_state, ais_z.detach().numpy())
                time.sleep(0.025)

            next_obs, reward, done, _ = bc.env.step(a)

            if bc.args.env_name[:8] == 'MiniGrid':
                current_obs = next_obs['image']
                current_obs = bc.get_encoded_obs(current_obs)
            else:
                current_obs = next_obs
                current_obs = bc.convert_int_to_onehot(current_obs, bc.env.observation_space.n)
        assert(next_obs+1==1)
        assert(bc.env.current_state==0)
        plt.plot(ais_z.detach().numpy(), label="trajectory " + str(i))

    plt.xlabel("AIS dim")
    plt.ylabel("AIS value")
    plt.title("AIS value for current Observation 1 with action sequences")
    plt.legend(loc="best")
    plt.show()


def kmeans_cluster(ais_z, actions, obs, num_group=10):
    _, label, inertia = k_means(ais_z, num_group)
    for i in range(num_group):
        ind = np.where(label==i)
        plot_action_obs_aiz(ind[0], actions, obs, ais_z)


def plot_action_obs_aiz(ind,actions, obs, ais_z):
        fig, axs = plt.subplots(3, 1)
        axs[0].set_xlabel('time step (reverse)')
        axs[0].set_ylabel('Action')
        axs[0].set_title('Action in reverse time step')
        axs[1].set_xlabel('time step (reverse)')
        axs[1].set_ylabel('Obs')
        axs[1].set_title('Obs in reverse time step')
        axs[2].set_xlabel('AIS dim')
        axs[2].set_ylabel('AIS value')
        axs[2].set_title('AIS')
        for j in ind:
            if len(actions[j])<=1:
                plt.sca(axs[0])
                axs[0].scatter(0,actions[j])
                plt.sca(axs[1])
                axs[1].scatter(0,obs[j])
                plt.sca(axs[2])
                if len(ind) < 5:
                    axs[2].plot(ais_z[j], label=str(j))
                else:
                    axs[2].plot(ais_z[j])
            else:
                if len(ind)<5:
                    plt.sca(axs[0])
                    axs[0].plot(actions[j][::-1], label=str(j))
                    plt.sca(axs[1])
                    axs[1].plot(obs[j][::-1], label=str(j))
                    plt.sca(axs[2])
                    axs[2].plot(ais_z[j], label=str(j))
                else:
                    plt.sca(axs[0])
                    axs[0].plot(actions[j][::-1])
                    plt.sca(axs[1])
                    axs[1].plot(obs[j][::-1])
                    plt.sca(axs[2])
                    axs[2].plot(ais_z[j])
        if len(ind) < 5:
            axs[0].legend(loc='best')
            axs[1].legend(loc='best')
            axs[2].legend(loc='best')
        plt.show()


def plot_aiz_initial_state(initial_state, obs, ais_z):
    means = []
    for i in range(10):
        ind = np.where(initial_state==i)[0]
        mean = np.mean(ais_z[ind], axis=0)
        means.append(mean)
        error = np.linalg.norm(ais_z[ind]-mean,'fro')
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("AIS error is " + str(error))
        # plt.text(2, 3, "AIS error is " + str(error))
        axs[0].set_xlabel('AIS dim')
        axs[0].set_ylabel("AIS value starting from state "+str(i))
        # axs[0].set_title('AIS starting from state '+str(i))
        axs[1].set_xlabel('time step')
        axs[1].set_ylabel('Obs')
        axs[1].set_title('Obs in normal time step')
        for j in ind:
            plt.sca(axs[0])
            axs[0].plot(ais_z[j])
            plt.sca(axs[1])
            axs[1].plot(obs[j])
        plt.show()

    for i in range(len(means)):
        plt.plot(means[i], label=str(i))
    plt.legend()
    plt.show()


def draw_tsne(ais_z, perplexity=20):
    coord = TSNE(n_components=2, perplexity=perplexity).fit_transform(ais_z)
    plt.scatter(coord[:,0], coord[:,1])
    plt.show()


def visualize_maze(state, ais_z):
    plt.figure(0)
    plt.subplot(121)
    maze = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]])
    if state <= 4:
        maze[0, state] = 2
    elif state == 5:
        maze[1, 0] = 2
    elif state == 6:
        maze[1, 2] = 2
    elif state == 7:
        maze[1, 4] = 2
    elif state == 8:
        maze[2, 0] = 2
    elif state == 9:
        maze[2, 4] = 2
    elif state == 10:
        maze[2, 2] = 2
    plt.imshow(maze)
    plt.subplot(122)
    plt.plot(ais_z)
    plt.xlabel("AIS dim")
    plt.ylabel("AIS value")
    plt.show()


if __name__ == "__main__":
    bc = batch_creator(args, env, None, False)
    bc.load_models_from_folder(args.models_folder)
    # obs, actions, states, initial_state, ais_z = eval_performance(bc, greedy=False, n_episodes=N_eps_eval)
    # draw_tsne(ais_z)
    # kmeans_cluster(ais_z, actions, obs)
    # plot_aiz_initial_state(initial_state, obs, ais_z)
    action_dict = {0:"N",1:"S",2:"E",3:"W"}
    ais = plot_single_obs(bc, action_dict)
    plot_trajectory_single_obs(bc, ais, action_dict, False)



