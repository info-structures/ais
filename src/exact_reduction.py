import numpy as np
import argparse
import cvxpy as cp
import learn_graph as lg
import gurobipy as gp

from gurobipy import GRB


parser = argparse.ArgumentParser(description='This code runs AIS using the Next Observation prediction version')
parser.add_argument("--save_graph", help="Save the transition probabilities", action="store_true")
parser.add_argument("--load_graph", help="Load the transition probabilities", action="store_true")
args = parser.parse_args()


def runGUROBIImpl(n, theta=0.5, nIter=10):
    for i in range(nIter):
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                C = np.random.rand(n, n)
                DCG = np.random.rand(n, n)
                IDCG = np.random.rand()

                P = model.addVars(n, n, vtype=GRB.BINARY)
                obj = sum([C[i, j] * P[i, j] for i in range(n) for j in range(n)])
                constrLHS = sum([DCG[i, j] * P[i, j] for i in range(n) for j in range(n)])

                model.setObjective(obj, GRB.MINIMIZE)
                model.addConstr(constrLHS >= theta * IDCG)
                model.addConstrs((P.sum("*", i) == 1 for i in range(n)))
                model.addConstrs((P.sum(i, "*") == 1 for i in range(n)))

                model.optimize()

                if not (model.status == GRB.OPTIMAL): print("unsuccessful...")

def runCVXPYImpl(nz, nb, nu, C, R):
    Q1 = cp.Variable((nz, nb), nonneg=True)
    Q2 = cp.Variable((nz, nb), nonneg=True)
    Q3 = cp.Variable((nz, nb), nonneg=True)
    Q4 = cp.Variable((nz, nb), nonneg=True)
    Q = [Q1, Q2, Q3, Q4]
    D = cp.Variable((nz, nb), boolean=True)
    r = cp.Variable((nb, nu))
    loss = 0
    constraints = []
    for i in range(nu):
        for j in range(nb):
            b_one_hot = np.zeros(nb)
            b_one_hot[j] = 1
            # Match transition distributions
            loss += cp.norm(cp.matmul(Q[i], b_one_hot)-cp.matmul(D, C[:, :, i]@b_one_hot))
            # Match reward
            loss += cp.norm(R[j, i] - cp.matmul(r[:, i], b_one_hot))
            constraints += [cp.sum(D) == nz,
                            cp.matmul(np.ones((1, nz)), D) <= 1,
                            cp.matmul(np.ones((1, nz)), D) == cp.matmul(np.ones((1, nz)), Q[i]),
                            cp.matmul(D, np.ones((nb, 1))) == np.ones((nz, 1)), ]


    objective = cp.Minimize(loss)

    problem = cp.Problem(objective, constraints)


    # solve problem
    # problem.solve(solver=cp.GUROBI, verbose=False)
    problem.solve(solver=cp.GUROBI, verbose=False)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        print("loss ", loss.value)
    return np.array([Q1.value, Q2.value, Q3.value, Q4.value]), D.value, r.value


def value_iteration(B, r, nz, na, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.

    Args:
        B: numpy array of size(na, nz, nz). transition probabilities of the environment P(z(t+1)|z(t), a(t)).
        r: numpy array of size (nz, na). reward function r(z(t),a(t))
        nz: number of AIS in the environment.
        na: number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(V, a, z):
        z_one_hot = np.zeros(nz)
        z_one_hot[z] = 1
        z_next = B[a, :, :]@z_one_hot
        # v = np.sum(prob_O * (Ot[:,0] + discount_factor * prob_z@V))

        return 0

    # start with initial value function and initial policy
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


def save_reduction_graph(Q, D, r, nz):
    np.save("src/Q_{}".format(nz), Q)
    np.save("src/D_{}".format(nz), D)
    np.save("src/r_fit_{}".format(nz), r)


def load_reduction_graph(nz):
    Q = np.load("src/Q_{}.npy".format(nz))
    D = np.load("src/D_{}.npy".format(nz))
    r = np.load("src/r_fit_{}.npy".format(nz))
    return Q, D, r


if __name__ == "__main__":
    np.random.seed(0)
    nz = 11
    nb = 15
    nu = 4
    C = np.load("src/C.npy")
    R = np.load("src/R.npy")

    if args.load_graph:
        Q, D, r = load_reduction_graph(nz)
    else:
        Q, D, r = runCVXPYImpl(nz, nb, nu, C, R)
        if args.save_graph:
            save_reduction_graph(Q, D, r, nz)

    B = Q@D.T
    value_iteration(B, r, nz, nu)




