import numpy as np
import learn_graph as lg
import argparse
import random

parser = argparse.ArgumentParser(description='This code runs AIS using the Next Observation prediction version')
parser.add_argument("--short_traj", help="Collect samples for short trajectory(not until reaching the goal)",action="store_true")
parser.add_argument("--env_not_terminate", help="Simulation does not terminate when goal state is reached",action="store_true")
args = parser.parse_args()
args.load_policy = False

def value_iteration(A, B, nz, na, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.

    Args:
        A: transition probabilities of the environment P(z(t+1)|z(t), y(t), a(t)).
        B: transition probabilities P(O(t)|z(t),a(t))
        nz: number of AIS in the environment.
        na: number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(V, a, z):

        prob_O = B[z,a,:]
        prob_z = A[z,Ot[:,1].astype(int),a,:]
        v = np.sum(prob_O * (Ot[:,0] + discount_factor * prob_z@V))

        return v

    # start with inital value function and intial policy
    V = np.zeros(nz)
    policy = np.zeros([nz, na])

    n = 0
    # while not the optimal policy
    while True:
        print('Iteration: ', n)
        # for stopping condition
        delta = 0

        # loop over state space
        for z in range(nz):

            actions_values = np.zeros(na)

            # loop over possible actions
            for a in range(na):
                # apply bellman eqn to get actions values
                actions_values[a] = one_step_lookahead(V, a, z)

            # pick the best action
            best_action_value = max(actions_values)

            # get the biggest difference between best action value and our old value function
            delta = max(delta, abs(best_action_value - V[z]))

            # apply bellman optimality eqn
            V[z] = best_action_value

            # to update the policy
            best_action = np.argmax(actions_values)

            # update the policy
            policy[z] = np.eye(na)[best_action]


        # if optimal value function
        if (delta < epsilon):
            break
        n += 1

    return policy, V


def eval_performance(policy, A, n_episodes=100, epsilon=1e-8, beta=0.95):
    returns = []
    for n_eps in range(n_episodes):
        reward_episode = []
        y = lg.env.reset()
        while True:
            # sample z from initial distribution
            z = np.where(np.random.multinomial(1,initial_distribution)==1)[0][0]
            # check z agrees with the first observation
            if (A[:,y,:, z]>epsilon).any():
                break


        for j in range(1000):
            action = np.arange(na)[policy[z].astype(bool)][0]

            y, reward, done, _ = lg.env.step(action)
            reward_episode.append(reward)

            z = np.where(np.random.multinomial(1, A[z, y, action, :]) == 1)[0][0]

            if done:
                break

        rets = []
        R = 0
        for i, r in enumerate(reward_episode[::-1]):
            R = r + beta * R
            rets.insert(0, R)
        returns.append(rets[0])

    return np.mean(returns)

if __name__ == "__main__":
    nz = 15
    seed = 60
    A, B, initial_distribution, Ot = lg.load_graph(nz, seed, args)
    na = B.shape[1]
    assert(na==4)
    policy, v = value_iteration(A, B, nz, na, discount_factor = 0.95)
    returns = eval_performance(policy, A)
    print("Performance: ",returns)
    # lg.plot_B(B, nz, Ot)