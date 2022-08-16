import numpy as np
import matplotlib.pyplot as plt 
import seaborn 
import scipy as sp
import scipy.optimize
import time 
import datetime
import mvn 
import util

def objective(z, srk, points, verbose=False):
    # multivariate object
    # note that i have defined most variables as (dx1) arrays, but 
    # the scipy optimisation routines require 1 dimension arrays. 
    # that explains the reshape here
    z_array = np.reshape(z, (z.size, 1))
    out = 0.5*srk(z_array, z_array)
    if verbose:
        print('Initial Calc: %.5f' % out)
    if points is not None:
        if verbose:
            print('length: %d' % len(points))
        for xi in points:
            tmp = srk(xi, z_array)
            if verbose:
                print(' -- adding %.5f' % tmp)
            out += tmp
    return out

if __name__ == '__main__':

    n = 10
    print('Multivariate Normal Example 1-4')

    # construct the target multivariate normal
    mu = np.array([[-0.0],[0.0]], dtype=float)
    sigsq = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
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
    mins = lambda s,e : (e-s)/60.0
    xp = yp = np.linspace(-5, 5, 51)
    x = [mu]
   
    while len(x) < n:

        # minimise with scipy, and record info about solution
        s = time.time()
        opt = scipy.optimize.differential_evolution(objective, bounds=[(-5,5),(-5,5)], args=(srk, x))
        e = time.time()
        spopt_x, spopt_y = opt.x
        spopt_p = opt.fun
        spopt_t = mins(s, e)
        print(' *** Scipy found minimum %.5f after %s (H:M:S)' % (spopt_p, datetime.timedelta(minutes=spopt_t)))

        # do grid search and record similar
        s = time.time()
        f = lambda z : objective(z, srk, x)
        X, Y, Z = util.fill_grid(f, xp, yp)
        gs_x, gs_y, gs_z = util.grid_search(X, Y, Z, obj_type='min')
        e = time.time()
        print(' *** Brute force minimum %.5f after %s (H:M:S)' % (gs_z, datetime.timedelta(minutes=mins(s, e))))

        # decide which to add to set, and continue
        print(' -- sp min=%.5f | bf min=%.5f' % (spopt_p, gs_z))

        if spopt_p < gs_z:
            print(' -- choosing sp solution.')
            xopt = np.array([spopt_x, spopt_y])
        else:
            print(' -- choosing brute solution.')
            xopt = np.array([gs_x, gs_y])
        
        # print(' **** Found Stein Point %d at z=(%.3f,%.3f) after %s (H:M:S) **** ' % (i+1, opt.x[0], opt.x[1], datetime.timedelta(minutes=mins(s, e)) ))
        x.append(xopt)

    # plot the first objective function.  how bad is it?
    # note that if you fill the surface with objective(z, srk, x[:-1])
    # you are looking at the surface for which x[-1] was the minimum
    
    if True:
        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, levels=30)
        cbar = fig.colorbar(contour)
        plt.scatter(gs_x, gs_y, marker='o', color='r', label='Grid Search Solution')
        plt.scatter(spopt_x, spopt_y, color='k', marker='x', label='Scipy Opt Solution')
        plt.legend()

    if False:
        f = lambda z : normal.pdf(z)
        X, Y, Z = fill_grid(f, xp, yp)
        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, levels=30)
        cbar = fig.colorbar(contour)

        x_val, y_val, z_val = grid_search(X, Y, Z, obj_type='max')
        plt.scatter(x_val, y_val, marker='o', color='r', label='x_1 = mode')
        for xi in x:
            plt.scatter(xi[0], xi[1], marker='x', color='k')
    
    #plt.show()
    """
    #for p in x:
    #    plt.scatter(p[0], p[1], color='k', marker='x')
    #plt.xlabel('z_x', fontsize=10); plt.ylabel('z_y', fontsize=10)
    #plt.title('Target PDF')
    """