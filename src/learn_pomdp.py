import matplotlib.pyplot as plt
import numpy as np
import argparse
import gym
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
parser.add_argument("--models_folder", type=str, help='Pretrained model (state dict)',
                    default="results/CheeseMaze-v0_500_50_0.7_0.0001_0.0006_0.003_200_15000_25_KL_1/models/seed42")
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


def initialization(y, a, O, n_episodes):
    ny = 7 #len(list(set(y[0])))
    na = 4 #len(list(set(a[0])))
    Ot = []
    for r in range(n_episodes):
        Ot += O[r]
    # list of unique (reward, observation) pairs
    Ot = list(set(Ot))
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


def baum_welch(y, a, O, A, B, initial_distribution, ny, na, nz, nO, Ot, n_episodes, n_iter=100, epsilon=1e-8):
    for n in range(n_iter):
        print('Iteration: ', n)
        if n%10 == 0 and args.save_graph:
            save_graph(A, B, initial_distribution, Ot, nz, seed)
        A_num = np.zeros((nz, na, nz))
        A_den = np.zeros((nz, na))
        B_num = np.zeros((nz, ny))
        B_den = np.zeros(nz)
        initial_distribution_num = np.zeros(nz)
        R = n_episodes
        for r in range(n_episodes):
            yr = np.array(y[r])
            ar = np.array(a[r])
            Or = O[r]
            T = len(y[r])
            if T<=1:
                R -= 1
            else:
                alpha = forward(A, B, initial_distribution, nz, yr, ar, Or, Ot)
                beta = backward(A, B, nz, yr, ar, Or, Ot)

                xi = np.zeros((nz, nz, T-1))
                for t in range(T-1):
                    denominator = np.dot(alpha[:, t], beta[:, t])
                    for i in range(nz):
                        numerator = alpha[i, t] * A[i, ar[t], :] * B[:, yr[t+1]] * beta[:, t+1]
                        if denominator == 0:
                            xi[i, :, t] = np.zeros(nz)
                        else:
                            xi[i, :, t] = numerator / denominator

                gamma = np.sum(xi, axis=1)

                initial_distribution_num += gamma[:,0]


                for k in range(na):
                    A_num[:,k,:] += np.sum(xi[:,:,(ar[0:T-1]==k)],axis=2)
                    A_den[:,k] += np.sum(gamma[:,(ar[0:T-1]==k)], axis=1)

                # Add gamma at T
                if np.dot(alpha[:, -1], beta[:, -1]) ==0:
                    gammaT = np.zeros(nz).reshape((-1,1))
                else:
                    gammaT =  np.array(alpha[:,-1]*beta[:,-1]/np.dot(alpha[:, -1], beta[:, -1])).reshape((-1,1))
                gamma = np.hstack((gamma, gammaT))

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
            for j in range(ny):
                    if B_den[i] < epsilon:
                        B[i, j] = 0
                    else:
                        B[i, j] = B_num[i,j]/B_den[i]
        # Check marginalization
        # assert ((np.absolute(np.sum(B, axis=1) - np.ones((nz, na))) < epsilon).all())

        initial_distribution = initial_distribution_num/R

    return A, B, initial_distribution


def forward(A, B, initial_distribution, nz, y, a, O, Ot):
    T = len(y)
    alpha = np.zeros((nz, T))
    alpha[:,0] = initial_distribution * B[:, y[0]]

    for t in range(1,T):
        for j in range(nz):
            alpha[j,t] = alpha[:,t-1].dot(A[:,a[t-1],j])*B[j,y[t]]

    return alpha


def backward(A, B, nz, y, a, O, Ot):
    T = len(y)
    beta = np.zeros((nz, T))
    beta[:,T-1] = np.ones(nz)

    for t in range(T-2,-1,-1):
        for j in range(nz):
            beta[j, t] = (beta[:, t+1]*B[:, y[t+1]]).dot(A[j, a[t], :])

    return beta


def minimize(A, B, Ot, nz, na, epsilon=1e-8):
    goal_obs = 6

    goal_ind = np.any(np.absolute(B[:,goal_obs]-1) < epsilon, axis=1)
    z2goal = np.arange(nz)[goal_ind]

    z = list(range(nz))
    # Split the state_to_goal and all other states
    for i in z2goal:
        z.remove(i)
    partition = [z, list(z2goal)]

    stable = False
    while not stable:
        L = partition.copy()
        block_B = partition.pop(0)
        old_split = []
        for block_C in L:
            for i in y_a:
                # T(p in B, (y,a), q in C)
                T = A[block_B, i[0], i[1], :][:,block_C]
                # T(p in B, (y,a), C)
                T = np.sum(T, axis=1)
                T[T<epsilon]=0    # eliminate floating number error
                T[np.absolute(T-1)<epsilon] = 1
                split_T = group_same_entry(T, block_B)

                O = []
                for p in block_B:
                    O_ind = np.where(B[p, i[1], :]>epsilon)[0]
                    O.append(O_ind)
                split_O = group_same_entry(O, block_B)

                new_split = intersect_partition(split_T, split_O)

                if old_split == []:
                    old_split = new_split.copy()
                else:
                    split = intersect_partition(old_split, new_split)
                    old_split = split.copy()
        for s in split:
            partition.append(s)
        if compare_partition(L, partition):
            stable = True
    return partition


def reduce_A_B(A, B, partition):
    delete_ind = []
    for block in partition:
        if len(block)>1:  # Equivalent states
            # Sum over probabilities transition to equivalent states
            A[:, :, :, block[0]] = np.sum(A[:, :, :, block], axis=3)
            # Delete equivalent states except for the first state
            delete_ind += block[1:]
    A = np.delete(A,delete_ind, axis=0)
    A = np.delete(A, delete_ind, axis=3)
    B = np.delete(B, delete_ind, axis=0)
    nz = B.shape[0]
    return A, B, nz


def group_same_entry(X, block_B):
    dict = {}
    for i in range(len(X)):
        if str(X[i]) not in dict.keys():
            dict[str(X[i])] = [block_B[i]]
        else:
            dict[str(X[i])].append(block_B[i])
    split = []
    for key in dict.keys():
        split.append(dict[key])
    return split


def intersect_partition(s1, s2):
    s1_copy = s1.copy()
    for b1 in s1_copy:
        if b1 not in s2:
            s1.remove(b1)
            for b2 in s2:
                i = intersection(b1, b2)
                if i !=[]:
                    s1.append(i)
    return s1


def compare_partition(p1, p2):
    p2_set = [set(s) for s in p2]
    for p in p1:
        if set(p) not in p2_set:
            return False
    return True

def intersection(b1, b2):
    return list(set(b1)&set(b2))


def to_tuple(X):
    return [(i[0], i[1]) for i in X]


def observation_count(y):
    dict = {}
    num_total_obs = 0
    for obs in y:
        for o in obs:
            num_total_obs += 1
            if o not in dict.keys():
                dict[o] = 0
            else:
                dict[o] += 1
    for key in dict.keys():
        print("Obs {} appear {}".format(key+1, dict[key]/num_total_obs))


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


def save_graph(A, B, initial_distribution, Ot, nz, seed):
    if args.load_policy:
        np.save("graph/pomdp/optimal_policy/A_{}_{}".format(nz, seed), A)
        np.save("graph/pomdp/optimal_policy/B_{}_{}".format(nz, seed), B)
        np.save("graph/pomdp/optimal_policy/initial_distribution_{}_{}".format(nz, seed), initial_distribution)
        np.save("graph/pomdp/optimal_policy/Ot_{}_{}".format(nz, seed), Ot)
    else:
        np.save("graph/pomdp/A_{}_{}".format(nz, seed), A)
        np.save("graph/pomdp/B_{}_{}".format(nz, seed), B)
        np.save("graph/pomdp/initial_distribution_{}_{}".format(nz, seed), initial_distribution)
        np.save("graph/pomdp/Ot_{}_{}".format(nz, seed), Ot)


def load(nz, seed, args):
    if args.load_policy:
        A = np.load("graph/pomdp/optimal_policy/A_{}_{}.npy".format(nz, seed))
        B = np.load("graph/pomdp/optimal_policy/B_{}_{}.npy".format(nz, seed))
        initial_distribution = np.load("graph/pomdp/optimal_policy/initial_distribution_{}_{}.npy".format(nz, seed))
    else:
        A = np.load("graph/pomdp/A_{}_{}.npy".format(nz, seed))
        B = np.load("graph/pomdp/B_{}_{}.npy".format(nz, seed))
        initial_distribution = np.load("graph/pomdp/initial_distribution_{}_{}.npy".format(nz, seed))
    return A, B, initial_distribution


if __name__ == "__main__":
    bc = batch_creator(args, env, None, True)
    nz = 11
    action_dict = {0: "N", 1: "S", 2: "E", 3: "W"}
    epsilon = 1e-8
    if args.load_graph:
        A, B, initial_distribution, Ot = load(nz, seed, args)
    else:
        if args.load_policy:
            bc.load_models_from_folder(args.models_folder)
        y, a, O, states = lg.collect_sample(bc, greedy=False, n_episodes=N_eps_eval)
        A, B, initial_distribution, ny, na, nz, nO, Ot = initialization(y, a, O, N_eps_eval)
        A, B, initial_distribution = baum_welch(y, a, O, A, B, initial_distribution, ny, na, nz, nO, Ot,
                                                N_eps_eval, 100, epsilon)
        # partition = minimize(A, B, Ot, nz, a, epsilon)
        # Ar, Br, nzr = reduce_A_B(A, B, partition)
        # Ar, Br, initial_distribution = baum_welch(y, a, O, Ar, Br, initial_distribution, ny, na, nzr, nO, Ot,
        #                                         N_eps_eval, 50, epsilon)
        if args.save_graph:
            save_graph(A, B, initial_distribution, Ot, nz, seed)
    plot_A(A, na)
    plot_B(B)






