import numpy as np
import learn_graph as lg

nx = 11
nu = 4
ny = 7
P_xu = np.zeros((nu, nx, nx))
current_ind = list(range(nx))
next_indices = [[0, 1, 2, 3, 4, 0, 2, 4, 5, 7, 6],
                [5, 1, 6, 3, 7, 8, 10, 9, 8, 9, 10],
                [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10],
                [0, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10]]

for i in range(nu):
    next_ind = next_indices[i]
    assert(len(next_ind) == len(current_ind))
    P_xu[i, next_ind, current_ind] = 1

D = np.zeros((ny, nx, nx))
ind = [[0],
       [1, 3],
       [2],
       [4],
       [5, 6, 7],
       [8, 9],
       [10]]
for i in range(ny):
    D[i, ind[i], ind[i]] = 1

nb = 15
uniform_distribution = np.ones(nx)/nx
initial_distribution = np.random.rand(nx, nb)
initial_distribution = initial_distribution/np.sum(initial_distribution,axis=0)
b0 = np.hstack((uniform_distribution.reshape((-1,1)), initial_distribution))


y_a, y, a, O = lg.load_trajectory(lg.args)

b = []
for r in range(len(y)):
    yr = np.array(y[r])
    ar = np.array(a[r])
    Or = O[r]
    bn = b0
    for t in range(len(yr)):
        bn = D[yr[t]]@P_xu[ar[t]]@bn
        # bn = bn/np.sum(bn)
        bn = bn / np.sum(bn, axis=0)
        b.append(bn)

np.unique(b, axis=1)

