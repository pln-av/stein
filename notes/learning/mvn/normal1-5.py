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

    n = 20
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
    n_points = 201
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
    
    

    while len(x) < n:

        s = time.time()
        opt = scipy.optimize.dual_annealing(util.objective, bounds=[x_bounds, y_bounds], args=(srk, x))
        e = time.time()
        spopt_x, spopt_y = opt.x
        spopt_p = opt.fun
        spopt_t = mins(s, e)
        print(' *** Scipy found point (%.2f. %.2f) after %s (H:M:S)' % (spopt_x, spopt_y, datetime.timedelta(minutes=spopt_t)))
        x.append(opt.x)

    # construct a plot of the target pdf, and computed Stein Points
    X, Y, Z = util.fill_grid( normal, xp, yp)
    plt.contourf(X, Y, Z, levels=30)
    plt.title('Bivariate Normal Example with Stein Points')
    for idx, xi in enumerate(x):
        if idx==0:
            plt.scatter(xi[0], xi[1], marker='x', color='r', label='Stein Points')
        else:
            plt.scatter(xi[0], xi[1], marker='x', color='r')
    plt.savefig('normal17.png', bbox_inches='tight')
    plt.close()
