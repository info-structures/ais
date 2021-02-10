import matplotlib.pyplot as plt
import numpy as np
import argparse
import gym
from random import sample
import learn_graph as lg
from train_manager import batch_creator

import torch
from torch.distributions import Categorical


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
parser.add_argument("--AIS_state_size", type=int, help="Size of the AIS vector in the neural network", default=25)
parser.add_argument("--IPM", help="Options: `KL`, `MMD`", default='MMD')
parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
parser.add_argument("--load_policy", help="load the AIS model for trained optimal policy",action="store_true")
parser.add_argument("--load_graph", help="load the parameters learned for the graph",action="store_true")
parser.add_argument("--save_graph", help="load the parameters learned for the graph",action="store_true")
parser.add_argument("--minimize", help="Run model minimization on learned graph",action="store_true")
parser.add_argument("--models_folder", type=str, help='Pretrained model (state dict)',
                    default="results/CheeseMaze-v0_500_50_0.7_0.0001_0.0006_0.003_200_15000_25_KL_1/models/seed42")

parser.add_argument("--AIS_pred_ncomp", type=int,
                    help="Number of Components used in the GMM to predict next AIS (For MiniGrid+KL)", default=5)

args = parser.parse_args()

args.pomdp = True
args.short_traj = False
args.env_not_terminate = False
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

def collect_sample(na, n_episodes=500, n_history=15):
    y = []
    a = []
    states = []
    O = []
    for n_eps in range(n_episodes):
        next_obs = bc.env.reset()
        reward = 0
        action = 0   # In reality, undefined; just place holder

        obs_history = []
        action_history = []
        state_history = []
        O_history = []

        obs_history.append(next_obs)  # + 1 to shift the index starting from 1
        action_history.append(action)
        O_history.append((reward, next_obs))
        state_history.append(bc.env.current_state)

        for j in range(n_history):
            action = np.random.randint(0,na)
            next_obs, reward, done, _ = bc.env.step(action)

            obs_history.append(next_obs)  # + 1 to shift the index starting from 1
            action_history.append(action)
            O_history.append((reward,next_obs))
            state_history.append(bc.env.current_state)

            # if done:
            #     break

        y.append(obs_history)
        a.append(action_history)
        O.append(O_history)
        states.append(state_history)

    return y, a, O, states

def initialization(O, n_episodes):
    ny = 7 #len(list(set(y[0])))
    na = 4 #len(list(set(a[0])))
    Ot = []
    for r in range(n_episodes):
        Ot += O[r]
    # list of unique (reward, observation) pairs
    Ot = list(set(Ot))
    # sort paris by observation
    Ot.sort(key=lambda x: x[1])
    # remove the undefined (reward, observation) at time=0
    Ot.remove((0,6))
    nO = len(Ot)

    # Initialization
    # Transition Probabilities
    A = np.ones((nz, na, nz))
    A = A / nz

    # Emission Probabilities
    B = np.random.random((nz, ny))
    B = B / np.sum(B, axis=1).reshape((-1, 1))

    initial_distribution = np.ones(nz)/nz

    return A, B, initial_distribution, ny, na, nz, nO, Ot


def baum_welch(y, a, O, A, B, initial_distribution, ny, na, nz, nO, Ot, n_iter=100, epsilon=1e-8, pred_O=True):
    for n in range(n_iter):
        print('Iteration: ', n)
        if n%10 == 0 and args.save_graph:
            save_graph(A, B, initial_distribution, Ot, nz, seed)
        A_num = np.zeros((nz, na, nz))
        A_den = np.zeros((nz, na))
        if pred_O:
            B_num = np.zeros((nz, nO))
        else:
            B_num = np.zeros((nz, ny))
        B_den = np.zeros(nz)
        initial_distribution_num = np.zeros(nz)
        n_episodes = len(y)
        R = n_episodes
        for r in range(n_episodes):
            yr = np.array(y[r])
            ar = np.array(a[r])
            Or = O[r]
            T = len(y[r])
            if T<=1:
                R -= 1
            else:
                alpha = forward(A, B, initial_distribution, nz, yr, ar, Or, Ot, pred_O)
                beta = backward(A, B, nz, yr, ar, Or, Ot, pred_O)

                xi = np.zeros((nz, nz, T-1))
                for t in range(T-1):
                    denominator = np.dot(alpha[:, t], beta[:, t])
                    for i in range(nz):
                        if pred_O:
                            numerator = alpha[i, t] * A[i, ar[t + 1], :] * B[:, Ot.index(Or[t+1])] * beta[:, t + 1]
                        else:
                            numerator = alpha[i, t] * A[i, ar[t+1], :] * B[:, yr[t+1]] * beta[:, t+1]
                        if denominator == 0:
                            xi[i, :, t] = np.zeros(nz)
                        else:
                            xi[i, :, t] = numerator / denominator

                gamma = np.sum(xi, axis=1)

                initial_distribution_num += gamma[:,0]


                for k in range(na):
                    A_num[:,k,:] += np.sum(xi[:,:,(ar[1:T]==k)],axis=2)
                    A_den[:,k] += np.sum(gamma[:,(ar[1:T]==k)], axis=1)

                # Add gamma at T
                if np.dot(alpha[:, -1], beta[:, -1]) ==0:
                    gammaT = np.zeros(nz).reshape((-1,1))
                else:
                    gammaT =  np.array(alpha[:,-1]*beta[:,-1]/np.dot(alpha[:, -1], beta[:, -1])).reshape((-1,1))
                gamma = np.hstack((gamma, gammaT))

                if pred_O:
                    for j in range(nO):
                        match_O = np.array(Or) == np.array(Ot)[j]
                        ind_O = match_O[:, 0] * match_O[:, 1]
                        B_num[:, j] += np.sum(gamma[:,ind_O], axis=1)
                        B_den[:] += np.sum(gamma, axis=1)
                else:
                    for j in range(ny):
                        B_num[:, j] += np.sum(gamma[:,(yr==j)], axis=1)
                        B_den[:] += np.sum(gamma, axis=1)


        for i in range(nz):
            for j in range(na):
                for k in range(nz):
                    if A_den[i, j]<epsilon:
                        A[i, j, k] = 0
                    else:
                        A[i,j,k] = A_num[i,j,k]/A_den[i,j]
        # Check marginalization
        # assert ((np.absolute(np.sum(A,axis=2)- np.ones((nz, ny, na)))< epsilon).all())

        for i in range(nz):
            if pred_O:
                for j in range(nO):
                        if B_den[i] < epsilon:
                            B[i, j] = 0
                        else:
                            B[i, j] = B_num[i,j]/B_den[i]
            else:
                for j in range(ny):
                        if B_den[i] < epsilon:
                            B[i, j] = 0
                        else:
                            B[i, j] = B_num[i,j]/B_den[i]
        # Check marginalization
        # assert ((np.absolute(np.sum(B, axis=1) - np.ones((nz, na))) < epsilon).all())
        # Re-normalize
        B = B / np.sum(B, axis=1)[0]

        initial_distribution = initial_distribution_num/R

    return A, B, initial_distribution


def forward(A, B, initial_distribution, nz, y, a, O, Ot, pred_O):
    T = len(y)
    alpha = np.zeros((nz, T))
    if pred_O:
        try:
            alpha[:, 0] = initial_distribution * B[:,Ot.index(O[0])]
        except:
            # O = (r,y). Only y0 known. Sample the index with y0
            index = [i for i, value in enumerate(Ot) if value[1]==O[0][1]]
            ind = sample(index,1)[0]
            alpha[:, 0] = initial_distribution * B[:, ind]
    else:
        alpha[:,0] = initial_distribution * B[:, y[0]]

    for t in range(1,T):
        for j in range(nz):
            if pred_O:
                alpha[j, t] = alpha[:, t - 1].dot(A[:, a[t], j]) * B[j, Ot.index(O[t])]
            else:
                alpha[j,t] = alpha[:,t-1].dot(A[:,a[t],j])*B[j,y[t]]

    return alpha


def backward(A, B, nz, y, a, O, Ot, pred_O):
    T = len(y)
    beta = np.zeros((nz, T))
    beta[:,T-1] = np.ones(nz)

    for t in range(T-2,-1,-1):
        for j in range(nz):
            if pred_O:
                beta[j, t] = (beta[:, t + 1] * B[:, Ot.index(O[t+1])]).dot(A[j, a[t + 1], :])
            else:
                beta[j, t] = (beta[:, t+1]*B[:, y[t+1]]).dot(A[j, a[t+1], :])

    return beta


def minimize(A, B, Ot, nz, na, epsilon=1e-8):
    goal_obs = 6

    goal_ind = B[:,goal_obs] > epsilon
    z2goal = np.arange(nz)[goal_ind]

    z = list(range(nz))
    # Split the state_to_goal and all other states
    for i in z2goal:
        z.remove(i)
    partition = [z, list(z2goal)]

    stable = False
    while not stable:
        L = partition.copy()
        for block_B in L:
            old_split = []
            partition.remove(block_B)
            for block_C in L:
                for i in range(na):
                    # T(p in B, a, q in C)
                    T = A[block_B, i, :][:,block_C]
                    # T(p in B, (y,a), C)
                    T = np.sum(T, axis=1)
                    T[T<epsilon]=0    # eliminate floating number error
                    T[np.absolute(T-1)<epsilon] = 1
                    split_T = lg.group_same_entry(T, block_B)

                    O = []
                    for p in block_B:
                        O_ind = np.where(B[p, :]>epsilon)[0]
                        O.append(O_ind)
                    split_O = lg.group_same_entry(O, block_B)

                    new_split = lg.intersect_partition(split_T, split_O)

                    if old_split == []:
                        old_split = new_split.copy()
                    else:
                        split = lg.intersect_partition(old_split, new_split)
                        old_split = split.copy()
            for s in split:
                partition.append(s)
        if lg.compare_partition(L, partition):
            stable = True
    return partition


def reduce_A_B(A, B, initial_distribution, partition):
    delete_ind = []
    Ar = A.copy()
    Br = B.copy()
    initial_distribution_r = initial_distribution.copy()
    for block in partition:
        if len(block)>1:  # Equivalent states
            # Sum over probabilities transition to equivalent states
            Ar[:, :, block[0]] = np.sum(Ar[:, :, block], axis=2)
            # Delete equivalent states except for the first state
            delete_ind += block[1:]
    Ar = np.delete(Ar,delete_ind, axis=0)
    Ar = np.delete(Ar, delete_ind, axis=2)
    Br = np.delete(Br, delete_ind, axis=0)
    initial_distribution_r = np.delete(initial_distribution_r, delete_ind)
    nz = Br.shape[0]
    return Ar, Br, initial_distribution_r, nz


def plot_A(A,na, obs=None, Ar=None):
    """
    obs: if not None, only plot A with observation y=obs; else plot all A's
    Ar: if not None, plot A with reduced Ar for comparison
    """
    if Ar is None:
        for i in range(na):
            plt.imshow(A[:,i,:])
            plt.title('P(z(t+1)|z(t), a(t)) with a: {}'.format(action_dict[i]))
            plt.xlabel("z(t+1)")
            plt.ylabel("z(t)")
            plt.colorbar()
            plt.show()
    else:
        for i in range(a):
            plt.subplot(1,2,1)
            plt.imshow(A[:, i, :])
            plt.title('P(z(t+1)|z(t), a(t)) with a: {}'.format(action_dict[i]))
            plt.xlabel("z(t+1)")
            plt.ylabel("z(t)")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(Ar[:, i, :])
            plt.title('P(z(t+1)|z(t), a(t)) with a: {}, reduced A'.format(action_dict[i]))
            plt.xlabel("z(t+1)")
            plt.ylabel("z(t)")
            plt.colorbar()
            plt.show()


def plot_B(B):
    plt.imshow(B)
    plt.title("P(y(t)|z(t)")
    plt.xlabel("y(t)")
    plt.ylabel("z(t)")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    bc = batch_creator(args, env, None, True)
    nz = 20
    na = 4
    action_dict = {0: "N", 1: "S", 2: "E", 3: "W"}
    epsilon = 1e-8

    if args.load_graph:
        A, B, initial_distribution, Ot = lg.load_graph(nz, seed, args)
        y_a, y, a, O = lg.load_trajectory(args)
        Ot = lg.to_tuple(Ot)
    else:
        if args.load_policy:
            bc.load_models_from_folder(args.models_folder)
        y, a, O, states = collect_sample(na, n_episodes=500)
        A, B, initial_distribution, ny, na, nz, nO, Ot = initialization(O, N_eps_eval)
        A, B, initial_distribution = baum_welch(y, a, O, A, B, initial_distribution, ny, na, nz, nO, Ot,
                                                100, epsilon, pred_O=True)
        lg.save_trajectory(y, a, O, None, args)
        if args.save_graph:
            lg.save_graph(A, B, initial_distribution, Ot, nz, seed, args)
    if args.minimize:
        partition = minimize(A, B, Ot, nz, na, epsilon=1e-5)
        Ar, Br, initial_distribution_r, nzr = reduce_A_B(A, B, initial_distribution, partition)
        Ar, Br, initial_distribution = baum_welch(y, a, O, Ar, Br, initial_distribution, ny, na, nzr, nO, Ot,
                                                  50, epsilon)
    plot_A(A, na)
    plot_B(B)






