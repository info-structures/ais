import matplotlib.pyplot as plt
import numpy as np
import argparse
import gym
from train_manager import batch_creator
import itertools

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
parser.add_argument("--load_policy", help="Load the AIS model for trained optimal policy",action="store_true")
parser.add_argument("--load_graph", help="Load the parameters learned for the graph",action="store_true")
parser.add_argument("--save_graph", help="Save the parameters learned for the graph",action="store_true")
parser.add_argument("--short_traj", help="Collect samples for short trajectory(not until reaching the goal)",action="store_true")
parser.add_argument("--minimize", help="Run model minimization on learned graph",action="store_true")
parser.add_argument("--env_not_terminate", help="Simulation does not terminate when goal state is reached",action="store_true")
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


def collect_sample(bc, greedy=False, n_episodes=1):
    y = []
    a = []
    states = []
    O = []
    for n_eps in range(n_episodes):

        if bc.args.env_name[:8] == 'MiniGrid':
            current_obs = bc.env.reset()['image']
            current_obs = bc.get_encoded_obs(current_obs)
        else:
            next_obs = bc.env.reset()
            current_obs = bc.convert_int_to_onehot(next_obs, bc.env.observation_space.n)
        action = torch.zeros(bc.env.action_space.n)
        reward = 0.
        hidden = None

        obs_history = []
        action_history = []
        state_history = []
        O_history = []
        for j in range(15): #range(1000):
            rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
            ais_z, hidden = bc.rho(rho_input, hidden)
            ais_z = ais_z.reshape(-1)

            policy_probs = bc.policy(ais_z.detach())
            if greedy:
                c = Categorical(policy_probs)
                _, action = policy_probs.max(0)
            else:
                c = Categorical(policy_probs)
                action = c.sample()

            next_obs, reward, done, _ = bc.env.step(action.item())

            obs_history.append(next_obs)  # + 1 to shift the index starting from 1
            action_history.append(action.item())
            O_history.append((reward,next_obs))
            state_history.append(bc.env.current_state)

            action = bc.convert_int_to_onehot(action, bc.env.action_space.n)


            if bc.args.env_name[:8] == 'MiniGrid':
                current_obs = next_obs['image']
                current_obs = bc.get_encoded_obs(current_obs)
            else:
                current_obs = next_obs
                current_obs = bc.convert_int_to_onehot(current_obs, bc.env.observation_space.n)

            # if done:
        y.append(obs_history)
        a.append(action_history)
        O.append(O_history)
        states.append(state_history)
                # break

    return y, a, O, states


def collect_sample_short_trajectory(na, n_history=10):
    ns = 10
    y = []
    a = []
    states = []
    O = []
    action_sequence = list(itertools.product(range(na), repeat=n_history))
    # for s in range(ns):
    for i in action_sequence:
        bc.env.reset()
        # bc.env.set(s)

        obs_history = []
        action_history = []
        state_history = []
        O_history = []

        for action in i:
            next_obs, reward, done, _ = bc.env.step(action)

            obs_history.append(next_obs)  # + 1 to shift the index starting from 1
            action_history.append(action)
            O_history.append((reward,next_obs))
            state_history.append(bc.env.current_state)

            if done:
                break

        y.append(obs_history)
        a.append(action_history)
        O.append(O_history)
        states.append(state_history)

    return y, a, O, states


def initialization(y, a, O, n_episodes):
    ny = 7 #len(list(set(y[0])))
    na = 4 #len(list(set(a[0])))
    Ot = []
    y_a = []
    for r in range(n_episodes):
        Ot += O[r]
        y_a += list(zip(y[r],a[r]))
    # list of unique (reward, observation) pairs
    Ot = list(set(Ot))
    y_a = list(set(y_a))
    nO = len(Ot)

    # Initialization
    # Transition Probabilities
    A = np.ones((nz, ny, na, nz))
    A = A / nz

    # Using prior knowledge about (y,a) pairs to update A
    # for j in range(ny):
    #     for k in range(na):
    #         if (j,k) not in y_a:
    #             A[:,j,k,:] = np.zeros((nz,nz))

    # Emission Probabilities
    B = np.random.random((nz, na, nO))
    # np.sum(B, axis=2)

    initial_distribution = np.ones(nz)/nz

    return A, B, initial_distribution, ny, na, nz, nO, Ot, y_a


def baum_welch(y, a, O, A, B, initial_distribution, ny, na, nz, nO, Ot, y_a , n_iter=100, epsilon=1e-8):
    for n in range(n_iter):
        print('Iteration: ', n)
        if n%10 == 0 and args.save_graph:
            save_graph(A, B, initial_distribution, Ot, nz, seed, args)
        A_num = np.zeros((nz, ny, na, nz))
        A_den = np.zeros((nz, ny, na))
        B_num = np.zeros((nz, na, nO))
        B_den = np.zeros((nz, na))
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
                alpha = forward(A, B, initial_distribution, nz, yr, ar, Or, Ot)
                beta = backward(A, B, nz, yr, ar, Or, Ot)

                xi = np.zeros((nz, nz, T-1))
                for t in range(T-1):
                    denominator = np.dot(alpha[:, t], beta[:, t])
                    for i in range(nz):
                        numerator = alpha[i, t] * A[i, yr[t], ar[t], :] * B[:, ar[t+1], Ot.index(Or[t+1])] * beta[:, t+1]
                        if denominator == 0:
                            xi[i, :, t] = np.zeros(nz)
                        else:
                            xi[i, :, t] = numerator / denominator

                gamma = np.sum(xi, axis=1)

                initial_distribution_num += gamma[:,0]

                for j in range(ny):
                    for k in range(na):
                        A_num[:,j,k,:] += np.sum(xi[:,:,(yr[0:T-1]==j)*(ar[0:T-1]==k)],axis=2)
                        A_den[:,j,k] += np.sum(gamma[:,(yr[0:T-1]==j)*(ar[0:T-1]==k)], axis=1)

                # Add gamma at T
                if np.dot(alpha[:, -1], beta[:, -1]) ==0:
                    gammaT = np.zeros(nz).reshape((-1,1))
                else:
                    gammaT =  np.array(alpha[:,-1]*beta[:,-1]/np.dot(alpha[:, -1], beta[:, -1])).reshape((-1,1))
                gamma = np.hstack((gamma, gammaT))

                for j in range(na):
                    for k in range(nO):
                        match_O = np.array(Or)==np.array(Ot)[k]
                        ind_O = match_O[:,0]*match_O[:,1]
                        B_num[:, j, k] += np.sum(gamma[:,(ar==j)*(ind_O)], axis=1)
                        B_den[:, j] += np.sum(gamma[:,ar==j], axis=1)


        for i in range(nz):
            for j in range(ny):
                for k in range(na):
                    for l in range(nz):
                        if A_den[i,j,k]<epsilon:
                            A[i, j, k, l] = 0
                        else:
                            A[i,j,k,l] = A_num[i,j,k,l]/A_den[i,j,k]
        # Check marginalization
        # assert ((np.absolute(np.sum(A,axis=3)- np.ones((nz, ny, na)))< epsilon).all())

        for i in range(nz):
            for j in range(na):
                for k in range(nO):
                    if B_den[i,j] < epsilon:
                        B[i, j, k] = 0
                    else:
                        B[i,j,k] = B_num[i,j,k]/B_den[i,j]
        # Check marginalization
        # assert ((np.absolute(np.sum(B, axis=2) - np.ones((nz, na))) < epsilon).all())
        # Re-normalize
        B = B/np.sum(B, axis=2)[0,0]

        initial_distribution = initial_distribution_num/R

    return A, B, initial_distribution


def forward(A, B, initial_distribution, nz, y, a, O, Ot):
    T = len(y)
    alpha = np.zeros((nz, T))
    alpha[:,0] = initial_distribution * B[:, a[0], Ot.index(O[0])]

    for t in range(1,T):
        for j in range(nz):
            alpha[j,t] = alpha[:,t-1].dot(A[:,y[t-1],a[t-1],j])*B[j,a[t],Ot.index(O[t])]

    return alpha


def backward(A, B, nz, y, a, O, Ot):
    T = len(y)
    beta = np.zeros((nz, T))
    beta[:,T-1] = np.ones(nz)

    for t in range(T-2,-1,-1):
        for j in range(nz):
            beta[j,t] = (beta[:,t+1]*B[:,a[t+1],Ot.index(O[t+1])]).dot(A[j,y[t],a[t],:])

    return beta


def minimize(A, B, Ot, nz, y_a, epsilon=1e-8):
    goal_obs = (1,6)
    goal_obs_ind = to_tuple(Ot).index(goal_obs)

    goal_ind = np.any(np.absolute(B[:,:,goal_obs_ind]-1) < epsilon, axis=1)
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


def reduce_A_B(A, B, initial_distribution, partition):
    delete_ind = []
    Ar = A.copy()
    Br = B.copy()
    initial_distribution_r = initial_distribution.copy()
    for block in partition:
        if len(block)>1:  # Equivalent states
            # Sum over probabilities transition to equivalent states
            Ar[:, :, :, block[0]] = np.sum(Ar[:, :, :, block], axis=3)
            # Delete equivalent states except for the first state
            delete_ind += block[1:]
    Ar = np.delete(Ar,delete_ind, axis=0)
    Ar = np.delete(Ar, delete_ind, axis=3)
    Br = np.delete(Br, delete_ind, axis=0)
    initial_distribution_r = np.delete(initial_distribution_r, delete_ind)
    nz = Br.shape[0]
    return Ar, Br, initial_distribution_r, nz


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


def plot_A(A,y_a, obs=None, Ar=None):
    """
    obs: if not None, only plot A with observation y=obs; else plot all A's
    Ar: if not None, plot A with reduced Ar for comparison
    """

    if obs is None:
        if Ar is None:
            for i in y_a:
                plt.imshow(A[:,i[0],i[1],:])
                plt.title('P(z(t+1)|z(t)) with y: {}, a: {}'.format(i[0]+1, action_dict[i[1]]))
                plt.xlabel("z(t+1)")
                plt.ylabel("z(t)")
                plt.colorbar()
                plt.show()
        else:
            for i in y_a:
                plt.subplot(1,2,1)
                plt.imshow(A[:,i[0],i[1],:])
                plt.title('P(z(t+1)|z(t)) with y: {}, a: {}'.format(i[0]+1, action_dict[i[1]]))
                plt.xlabel("z(t+1)")
                plt.ylabel("z(t)")
                plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.imshow(Ar[:, i[0], i[1], :])
                plt.title('P(z(t+1)|z(t)) with y: {}, a: {}, reduced A'.format(i[0] + 1, action_dict[i[1]]))
                plt.xlabel("z(t+1)")
                plt.ylabel("z(t)")
                plt.colorbar()
                plt.show()
    else:
        for i in y_a:
            if i[0]==obs:
                plt.imshow(A[:,i[0],i[1],:])
                plt.title('P(z(t+1)|z(t),y(t),a(t)) with y: {}, a: {}'.format(i[0]+1, action_dict[i[1]]))
                plt.xlabel("z(t+1)")
                plt.ylabel("z(t)")
                plt.colorbar()
                plt.show()


def plot_B(B,nz, Ot, state=None):
    """
    state: if not None, plot the B transition probability only for z = state
    """
    print("Order of Ot (r,y): ", Ot)
    if state is None:
        for i in range(nz):
            plt.imshow(B[i,:,:])
            plt.title("P(O(t)|z(t),a(t)) with z{}".format(i))
            plt.xlabel("O(t) index")
            plt.ylabel("a(t)")
            plt.colorbar()
            plt.show()
    else:
        plt.imshow(B[state, :, :])
        plt.title("P(O(t)|z(t),a(t)) with z{}".format(state))
        plt.xlabel("O(t) index")
        plt.ylabel("a(t)")
        plt.colorbar()
        plt.show()


def save_graph(A, B, initial_distribution, Ot, nz, seed, args):
    if args.load_policy:
        folder = "graph/optimal_policy/"
    elif args.short_traj:
        folder = "graph/short_traj/"
    elif args.env_not_terminate:
        folder = "graph/env_not_terminate/"
    else:
        folder = "graph/"
    np.save(folder+"A_{}_{}".format(nz, seed), A)
    np.save(folder+"B_{}_{}".format(nz, seed), B)
    np.save(folder+"initial_distribution_{}_{}".format(nz, seed), initial_distribution)
    np.save(folder+"Ot_{}_{}".format(nz, seed), Ot)


def save_trajectory(y, a, O, y_a, args):
    if args.load_policy:
        folder = "graph/optimal_policy/"
    elif args.short_traj:
        folder = "graph/short_traj/"
    elif args.env_not_terminate:
        folder = "graph/env_not_terminate/"
    else:
        folder = "graph/"
    np.save(folder + "y", y)
    np.save(folder + "action", a)
    np.save(folder + "O", O)
    np.save("graph/y_a", y_a)


def load_graph(nz, seed, args):
    if args.load_policy:
        folder = "graph/optimal_policy/"
    elif args.short_traj:
        folder = "graph/short_traj/"
    elif args.env_not_terminate:
        folder = "graph/env_not_terminate/"
    else:
        folder = "graph/"
    A = np.load(folder+"A_{}_{}.npy".format(nz, seed))
    B = np.load(folder+"B_{}_{}.npy".format(nz, seed))
    initial_distribution = np.load(folder+"initial_distribution_{}_{}.npy".format(nz, seed))
    Ot = np.load(folder+"Ot_{}_{}.npy".format(nz, seed))

    return A, B, initial_distribution, Ot


def load_trajectory(args):
    if args.load_policy:
        folder = "graph/optimal_policy/"
    elif args.short_traj:
        folder = "graph/short_traj/"
    elif args.env_not_terminate:
        folder = "graph/env_not_terminate/"
    else:
        folder = "graph/"
    y = np.load(folder + "y.npy", allow_pickle=True)
    a = np.load(folder + "action.npy", allow_pickle=True)
    O = np.load(folder + "O.npy", allow_pickle=True)
    y_a = np.load("graph/y_a.npy")
    return y_a, y, a, O


if __name__ == "__main__":
    bc = batch_creator(args, env, None, True)
    nz = 20
    na = 4
    ny = 7
    nO = 7
    action_dict = {0: "N", 1: "S", 2: "E", 3: "W"}
    epsilon = 1e-8

    if args.load_graph:
        A, B, initial_distribution, Ot  = load_graph(nz, seed, args)
        y_a, y, a, O = load_trajectory(args)
        Ot = to_tuple(Ot)
    else:
        if args.load_policy:
            bc.load_models_from_folder(args.models_folder)
        if args.short_traj:
            y, a, O, states = collect_sample_short_trajectory(na, n_history=10)
        else:
            y, a, O, states = collect_sample(bc, greedy=False, n_episodes=5000)
        A, B, initial_distribution, ny, na, nz, nO, Ot, y_a = initialization(y, a, O, N_eps_eval)
        A, B, initial_distribution = baum_welch(y, a, O, A, B, initial_distribution, ny, na, nz, nO, Ot, y_a,
                                                100, epsilon)
        save_trajectory(y, a, O, y_a, args)
        if args.save_graph:
            save_graph(A, B, initial_distribution, Ot, nz, seed, args)
    if args.minimize:
        partition = minimize(A, B, Ot, nz, y_a, epsilon)
        Ar, Br, initial_distribution_r, nzr = reduce_A_B(A, B, initial_distribution, partition)
        Ar, Br, initial_distribution_r = baum_welch(y, a, O, Ar, Br, initial_distribution_r, ny, na, nzr, nO, Ot, y_a,
                                                    50, epsilon)
    plot_A(A, y_a, Ar=Ar)
    plot_B(B, nz, Ot)
    plot_B(Br, nzr, Ot)






