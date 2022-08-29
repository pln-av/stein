import argparse
import numpy as np
import matplotlib.pyplot as plt 
import seaborn
import scipy.optimize
import time 

# imports for stein computations in python
import target
import kernel 
import stein 
import util

# export PYTHONPATH=/Users/patricknoble/Documents/Projects/stein/code/python/lib 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()

    print('Computing first %d Stein Points for MVN Example 1.' % args.n)

    # normal distribution target
    mu = np.array([-0.5, 0.5], dtype=float)
    sigsq = np.array([[1.0, 0.5],[0.5, 2.0]], dtype=float)
    normal = target.MVN(mu, sigsq)

    # imq kernel
    dimension = 2 
    alpha = 1.0
    beta = -0.50
    imq = kernel.IMQ(dimension, alpha, beta)

    # srk 
    srk = stein.SRK(normal, imq)

    # first point is mode.  using the same global optimizer for consistency
    # dual_annealing requires bounding box
    bounds = [(-5,5), (-5,5)]
    neg_obj = lambda z : -normal(z)
    s = time.time()
    opt = scipy.optimize.dual_annealing(neg_obj, bounds=bounds)
    e = time.time()
    print(' ** Stein Point (1) found in %s (H:M:S) at x:' % util.time_string(s, e), opt.x)
    x = [opt.x]
    t = [e-s]

    s0 = time.time()
    while len(x) < args.n:

        # continue until we have all the points we want
        point_idx = len(x)

        s = time.time()
        opt = scipy.optimize.dual_annealing(stein.greedy_objective, bounds=bounds, args=(srk, x))
        e = time.time()

        print(' ** Stein Point (%d) found in %s (H:M:S) at x:' % (point_idx+1, util.time_string(s, e)), opt.x)
        x.append(opt.x)
        t.append(e-s)

    fig, axs = plt.subplots(2, figsize=(8.27, 11.69))
    e0 = time.time()
    print('Total Compute time: %s' % util.time_string(s0, e0))
    
    # plot the distribution and points
    xp = yp = np.linspace( bounds[0][0], bounds[0][1], 201)
    X, Y, Z = util.fill_grid( normal, xp, yp)
    axs[0].contourf(X, Y, Z, levels=20)
    axs[0].set_title('Stein Points vs Target Distribution')
    axs[0].set_xlabel('x') 
    axs[0].set_ylabel('y')
    for xi in x:
        axs[0].scatter(xi[0], xi[1], marker='x', color='r')

    # plot the KSD
    n = len(x)
    ksd = stein.KSD(srk)
    axs[1].plot(np.arange(1,n), [ksd(x[:i]) for i in np.arange(1, n)])
    axs[1].set_xlabel('Number of Stein Points')
    axs[1].set_ylabel('KSD')
    axs[1].set_title('KSD vs Number of Points')
    axs[1].grid()

    # plot computation time
    #axs[2].plot(t)
    #axs[2].set_title('Computation Time vs Number of Points')
    #axs[2].set_ylabel('Computation Time Per Point (Secs)')
    #axs[2].set_xlabel('Number of Stein Points')
    #axs[2].grid()

    plt.savefig('normal18.png', bbox_inches='tight')
    plt.close()
    
