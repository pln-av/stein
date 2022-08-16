import numpy as np
import matplotlib.pyplot as plt 
import seaborn

import time
import datetime

import uvn

def objective(z, ksr, points):
    # evaluate the greedy objective for this point z \in R
    # where points is a sequence of previously computed points
    # use ksr(x,y) to evaluate ksr between any two points
    out = 0.5*ksr(z, z)
    if points is not None:
        for xi in points:
            out += ksr(xi, z)
    return out 

def ecdf(points, z):
    return np.array([np.sum(points<zi) for zi in z], dtype=float)/len(points)

if __name__ == '__main__':

    n = 50
    print('Univariate Normal Example 1-2')
    print(' -- compute the first n=%d Stein Points' % n)
    print(' -- compute and compare empirical cdf' )
    
    # construct the target (ie the normal)
    mu = 3.14
    sig = 2;
    normal = uvn.Normal(mu, sig)

    print('Target is normal pdf with parameters:')
    print(' --mu: %.2f' % normal.mu)
    print(' --sigsq: %.2f' % normal.sigsq)

    # construct the IMQ kernel
    alpha = 1.0
    beta = -0.50
    kernel = uvn.IMQ(alpha, beta)

    # construct the KSR
    ksr = uvn.KSR(normal, kernel)
    
    # initial condition is at distribution mode (which we know so just set it)
    x = [normal.mu]

    # do exhaustive grid search for minima.  this is not feasible in higher dimension, but
    # lets focus on the KSD and associated Stein Points, not the challenge of the optimisation problem.
    mins = lambda s, e : (e-s)/60.0
    if True:

        # plot data
        mult, psize = 12.0, 10001
        zmin, zmax = normal.mu-mult*normal.sig, normal.mu+mult*normal.sig
        zz = np.linspace(zmin, zmax, psize)

        start_time = time.time()
        for i in range(1, n):
            objs = np.array([objective(z,ksr,x) for z in zz], dtype=float)

            # multimodal? take first for now 
            idx = np.argmin(objs)
            ostar, zstar = objs[idx], zz[idx]
            e = time.time()
            print(' **** Found Stein Point %d at z=%.3f after %s (H:M:S) **** ' % (i+1, zstar, datetime.timedelta(minutes=mins(start_time,e)) ))
 
            # add this point
            x.append(zz[idx])

        e = time.time()
        print('Computation complete.  Found:')
        print(' -- %d Stein Points.' % len(x))
        print(' -- Wall Time: %s (H:M:S)' % datetime.timedelta(minutes=mins(start_time, e) ) )

        # here we calculate and compare 
        # 1.  the empirical cdf of stein points
        # 2.  a couple (m) of potential empirical cdf from n normal samples from numpy generator
        # 3.  the 'true' cdf
        m = 2
        stein_ecdf = ecdf(x, zz)
        plt.plot(zz, stein_ecdf, label='Stein Empirical CDF')
        for mi in range(m):
            plt.plot(zz, ecdf( np.random.normal(loc=mu, scale=sig, size=n), zz), label='Numpy Sample %d' % (mi+1) )

        # here is an approx 'exact' cdf
        plt.plot(zz, np.cumsum( normal.pdf(zz) )/np.sum(normal.pdf(zz)), label='Exact CDF' )

        plt.xlabel('x', fontsize=15)
        plt.ylabel('probability', fontsize=15)
        plt.title('Comparison of ECDFs (n=%d)' % n, fontsize=20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.xlim([-4.0, 10])
        plt.legend()
