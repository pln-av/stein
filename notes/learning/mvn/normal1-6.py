import numpy as np
import matplotlib.pyplot as plt 
import seaborn 
import scipy as sp
import scipy.optimize
import time 
import datetime
import mvn 
import util



if __name__ == '__main__':

    n = 9
    print('Multivariate Normal Example 1-4')

    # construct the target multivariate normal
    mu = np.array([-0.50 , 0.50])
    sigsq = np.array([[1.0,0.50], [0.50, 2.0]], dtype=float)
    normal = mvn.Normal(mu, sigsq)

    # construct IMQ kernel
    alpha = 1.00
    beta = -0.50
    kernel = util.IMQ(alpha, beta)

    # x, y limits for plots etc etc
    x_low, x_high = -4.0, 4.0
    y_low, y_high = x_low, x_high 
    n_points = 51
    xp = np.linspace(x_low, x_high, n_points)
    yp = np.linspace(y_low, y_high, n_points)
    x_bounds, y_bounds = (x_low, x_high), (y_low, y_high)

    # construct the stein reproducing kernel SRK
    srk = util.SRK(normal, kernel)

    # initialise the Stein Points with the distribution mode.  
    # again cheat and just set it using mean=mode
    # lets see what values are returned
    mins = lambda s,e : (e-s)/60.0
    x = [mu]
    idx = 0
    fig, axs = plt.subplots(4, 2, figsize=(8.27, 11.69))
    fig.suptitle('Stein Points for Bivariate Normal Example')
    plot_idx = np.arange(n-1)
    row_idx = np.array(plot_idx/2, dtype=int)
    col_idx = np.array(plot_idx % 2, dtype=int)
    while len(x) < n:

        # minimise with scipy, and record info about solution
        point_id = len(x)

        s = time.time()
        #opt = scipy.optimize.differential_evolution(util.objective, bounds=[x_bounds, y_bounds], args=(srk, x))
        opt = scipy.optimize.dual_annealing(util.objective, bounds=[x_bounds, y_bounds], args=(srk, x))
        e = time.time()
        spopt_x, spopt_y = opt.x
        spopt_p = opt.fun
        spopt_t = mins(s, e)
        print(' *** Scipy found minimum %.5f after %s (H:M:S)' % (spopt_p, datetime.timedelta(minutes=spopt_t)))

        if True:

            # lets just plot the objective surface
            f = lambda z : util.objective(z, srk, x)
            X, Y, Z = util.fill_grid(f, xp, yp)

            ri, ci = row_idx[idx], col_idx[idx]
            axi = axs[ri, ci]
            contour = axi.contourf(X, Y, Z, levels=30)
            axi.set_xticks([],[])
            axi.set_yticks([],[])
            axi.scatter(opt.x[0], opt.x[1], marker='x', color='r', label='Minimum')
            axi.set_title('Stein Point %d' % point_id) 
            idx += 1

        # add point to set
        print(' -- Point %d found at (%.4f, %.4f)' % (len(x), spopt_x, spopt_y))
        x.append(opt.x)

    plt.savefig('normal16.png', bbox_inches='tight')
    plt.close()
