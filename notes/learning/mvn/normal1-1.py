import numpy as np
import matplotlib.pyplot as plt 
import seaborn

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

if __name__ == '__main__':

    n = 9
    print('Univariate Normal Example 1-1')
    print(' -- compute and plot greedy objective for first n=%d Stein Points.' % n)

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

    # lets plot the greedy objective for various candidate points
    if True:

        # plot data
        mult, psize = 3.0, 10001
        zmin, zmax = normal.mu-mult*normal.sig, normal.mu+mult*normal.sig
        zz = np.linspace(zmin, zmax, psize)

        # put the initial point first
        plt.scatter(x[0], 0.0, marker='o', color='r', label='x_1')
        
        for i in range(1, n):
            objs = np.array([objective(z,ksr,x) for z in zz], dtype=float)
            plt.plot(zz, objs, label='x_%d' % (i+1))

            # multimodal? take first for now 
            idx = np.argmin(objs)
            ostar, zstar = objs[idx], zz[idx]
            print('Minimum=%.3f at z=%.3f' % (ostar, zstar))

            # add this point
            x.append(zz[idx])

            # plot it too
            plt.scatter(zstar, 0.0, color='k')

        plt.legend()
        plt.xlabel('z', fontsize=15); 
        plt.ylabel('objective', fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)        
        plt.title('Greedy Objective', fontsize=20)
        plt.grid()
   