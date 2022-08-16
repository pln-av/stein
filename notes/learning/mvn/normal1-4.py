import numpy as np
import matplotlib.pyplot as plt 
import seaborn 
import scipy as sp
import scipy.optimize

import mvn 

def objective(z, srk, points):
    # multivariate object
    # note that i have defined most variables as (dx1) arrays, but 
    # the scipy optimisation routines require 1 dimension arrays. 
    # that explains the reshape here
    z_array = np.reshape(z, (z.size, 1))
    out = 0.5*srk(z_array, z_array)
    if points is not None:
        for xi in points:
            out += srk(xi, z_array)
    return out
    
if __name__ == '__main__':

    print('Multivariate Normal Example 1-4')

    # construct the target multivariate normal
    mu = np.array([[-0.5],[0.5]], dtype=float)
    sigsq = np.array([[1.1, 0.5], [0.5, 1.2]], dtype=float)
    normal = mvn.Normal(mu, sigsq)

    # construct IMQ kernel
    alpha = 1.0
    beta = -0.5
    kernel = mvn.IMQ(alpha, beta)

    # construct the stein reproducing kernel SRK
    srk = mvn.SRK(normal, kernel)

    # initialise the Stein Points with the distribution mode.  
    # again cheat and just set it using mean=mode
    # lets see what values are returned
    m = 10
    x = [mu]
    proposals = []
    for mi in np.arange(m):
        out = scipy.optimize.dual_annealing(objective, bounds=[(-5,5),(-5,5)], args=(srk, x))
        proposals.append(out.x)

    # plot the first objective function.  how bad is it?
    x = y = np.linspace(-3, 3, 51)
    X, Y = np.meshgrid(x, y)
    rows, cols = X.shape[0], X.shape[1]
    surf = np.zeros(shape=X.shape, dtype=float)
    for r in np.arange(rows):
        for c in np.arange(cols):
            p = np.array([[X[r, c]],[ Y[r,c]]], dtype=float)
            surf[r, c] = objective(p, srk, [mu])

   

    plt.contourf(X, Y, surf, levels=30)
    for p in proposals:
        plt.scatter(p[0], p[1], color='k', marker='x')
    plt.xlabel('z_x', fontsize=10); plt.ylabel('z_y', fontsize=10)
    plt.title('Greedy Objective for Second Stein Point')