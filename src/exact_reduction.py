import numpy as np
import cvxpy as cp
import gurobipy as gp

from gurobipy import GRB

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

    if not (problem.status == cp.OPTIMAL): print("unsuccessful...")

# np.random.seed(0)
# start = datetime.now()
# runGUROBIImpl(n=100, theta=0.5, nIter=300)
# end = datetime.now()
# print((end - start).total_seconds())

np.random.seed(0)
nz = 11
nb = 15
nu = 4
C = np.load("src/C.npy")
R = np.load("src/R.npy")
runCVXPYImpl(nz, nb, nu, C, R)

