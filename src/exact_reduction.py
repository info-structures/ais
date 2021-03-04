import numpy as np
import argparse
import cvxpy as cp
import learn_graph as lg
import gurobipy as gp

from gurobipy import GRB


parser = argparse.ArgumentParser(description='This code runs AIS using the Next Observation prediction version')
parser.add_argument("--save_graph", help="Save the transition probabilities", action="store_true")
parser.add_argument("--load_graph", help="Load the transition probabilities", action="store_true")
parser.add_argument("--AIS_state_size", type=int, help="Load the transition probabilities", default=11)
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

def runCVXPYImpl(nz, nb, nu, C, R, C_det=None, P_y_b_u=None):
    Q1 = cp.Variable((nz, nb), nonneg=True)
    Q2 = cp.Variable((nz, nb), nonneg=True)
    Q3 = cp.Variable((nz, nb), nonneg=True)
    Q4 = cp.Variable((nz, nb), nonneg=True)
    Q = [Q1, Q2, Q3, Q4]
    D = cp.Variable((nz, nb), boolean=True)
    r_bar = cp.Variable((nb, nu))
    Q_det = []
    P_y_z_u_bar = []
    loss = 0
    constraints = []
    for j in range(nb):
        b_one_hot = np.zeros(nb)
        b_one_hot[j] = 1
        for i in range(nu):
            # Match transition distributions
            loss += cp.norm(cp.matmul(Q[i], b_one_hot)-cp.matmul(D, C[:, :, i]@b_one_hot))
            # Match reward
            loss += cp.norm(R[j, i] - cp.matmul(r_bar[:, i], b_one_hot))
            constraints += [cp.sum(D) == nz,
                            cp.matmul(np.ones((1, nz)), D) <= 1,
                            cp.matmul(np.ones((1, nz)), D) == cp.matmul(np.ones((1, nz)), Q[i]),
                            cp.matmul(D, np.ones((nb, 1))) == np.ones((nz, 1)), ]

            if P_y_b_u is not None:
                P_y_z_u_bar.append(cp.Variable(P_y_b_u.shape[0], nb))
                loss += cp.norm(cp.matmul(P_y_z_u_bar[i], b_one_hot) - P_y_b_u[:, :, i]@ b_one_hot)
                constraints += [cp.matmul(np.ones((1, nz)), D) == cp.matmul(np.ones((1, nz)), P_y_z_u_bar[i]), ]

        if C_det is not None:
            for k in range(C_det.shape[2]):
                Q_det.append(cp.Variable((nz, nb), boolean=True))
                loss += cp.norm(cp.matmul(Q_det[k], b_one_hot) - cp.matmul(D, C_det[:, :, k] @ b_one_hot))
                # B_det is also part of permutation matrix
                constraints += [cp.matmul(np.ones((1, nz)), D) == cp.matmul(np.ones((1, nz)), Q_det[k]), ]

    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)

    # solve problem
    problem.solve(solver=cp.GUROBI, verbose=False)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        print("loss ", loss.value)

    if C_det is not None:
        Q_det_out = []
        for i in range(len(Q_det)):
            Q_det_out.append(Q_det[i].value)
        return np.array(Q_det_out), np.array([Q1.value, Q2.value, Q3.value, Q4.value]), D.value, r_bar.value
    else:
        return np.array([Q1.value, Q2.value, Q3.value, Q4.value]), D.value, r_bar.value


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
        # ind = (z_next > epsilon)
        v = r[a, :]@z_next + discount_factor * z_next@V

        return v

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


def eval_performance(policy, D, C_det, V, V_b, y_a, na, nb, B_det=None, n_episodes=100, epsilon=1e-8, beta=0.95):
    returns = []
    Vs = []
    V_bs = []
    for n_eps in range(n_episodes):
        reward_episode = []
        y = lg.env.reset()

        uniform_distribution = np.ones(nb) / nb
        while True:
            # sample b from initial distribution
            ind_b = np.where(np.random.multinomial(1, uniform_distribution) == 1)[0][0]
            # check b agrees with the first observation
            if (C_det[ind_b, :, y == y_a[:, 0]] > epsilon).any():
                b_one_hot = np.zeros(nb)
                b_one_hot[ind_b] = 1
                break
        z_one_hot = D@b_one_hot

        for j in range(1000):
            try:
                ind_z = np.where(z_one_hot == 1)[0][0]
            except:
                print("No corresponding z")
            Vs.append(V[ind_z])
            V_bs.append(V_b[b_one_hot == 1])

            action = np.arange(na)[policy[ind_z].astype(bool)][0]

            y, reward, done, _ = lg.env.step(action)
            reward_episode.append(reward)

            ind_ya = np.where((y == y_a[:, 0])*(action == y_a[:, 1]))[0][0]
            b_one_hot = C_det[:, :, ind_ya]@b_one_hot
            if B_det is not None:
                z_one_hot = B_det@z_one_hot
            else:
                z_one_hot = D@b_one_hot

            if done:
                break

        rets = []
        R = 0
        for i, r in enumerate(reward_episode[::-1]):
            R = r + beta * R
            rets.insert(0, R)
        returns.append(rets[0])

    average_return = np.mean(returns)
    V_mse = np.norm(np.array(Vs)-np.array(V_bs))
    print("Average reward: ", )
    print("V mse: ", V_mse)
    return average_return, V_mse


def save_reduction_graph(Q, D, r_bar, nz, Q_det=None):
    np.save("src/Q_{}".format(nz), Q)
    np.save("src/D_{}".format(nz), D)
    np.save("src/r_fit_{}".format(nz), r_bar)
    if Q_det is not None:
        np.save("src/Q_det_{}".format(nz), Q_det)


def load_reduction_graph(nz, det=False):
    Q = np.load("src/Q_{}.npy".format(nz))
    D = np.load("src/D_{}.npy".format(nz))
    r_bar = np.load("src/r_fit_{}.npy".format(nz))
    if det:
        Q_det = np.load("src/Q_det{}.npy".format(nz))
        return Q_det, Q, D, r_bar
    else:
        return Q, D, r_bar


if __name__ == "__main__":
    np.random.seed(0)
    nz = lg.args.AIS_state_size
    nb = 15
    nu = 4
    C = np.load("src/C.npy")
    C_det = np.load("src/C_det.npy")
    R = np.load("src/R.npy")
    y_a = np.load("graph/y_a.npy")

    if args.load_graph:
        Q, D, r_bar = load_reduction_graph(nz)
    else:
        Q_det, Q, D, r_bar = runCVXPYImpl(nz, nb, nu, C, R, C_det)
        if args.save_graph:
            save_reduction_graph(Q, D, r_bar, nz, Q_det)

    B = Q@D.T
    r = r_bar.T@D.T
    policy, V = value_iteration(B, r, nz, nu)
    policy_b, V_b = value_iteration(np.einsum('ijk->kij', C), R.T, nb, nu)
    eval_performance(policy, D, C_det, V, V_b, y_a, nu, nb)




