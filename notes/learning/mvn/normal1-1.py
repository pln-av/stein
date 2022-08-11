import numpy as np
import matplotlib.pyplot as plt 
import seaborn
import scipy as sp

class Normal(object):
    def __init__(self, mu, sig):
        self.mu = mu 
        self.sig = sig 
        self.sigsq = sig*sig 
    
    def pdf(self, x):
        c = 1.0/np.sqrt(2.0*np.pi*self.sigsq)
        arg = (x-self.mu)/sig 
        return c*np.exp(-0.5*arg*arg)

    def log_grad(self, x):
        # compute the gradient of the log at this x
        return -(x - self.mu)/self.sigsq 

class IMQ(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta 

    def kernel(self, x, y):
        diff = x - y 
        return np.power( self.alpha + diff*diff, self.beta )

    def kernel_x(self, x, y):
        # derivative wrt x
        diff = x-y
        return 2*self.beta*diff*np.power(self.alpha + diff*diff, self.beta-1.0)

    def kernel_y(self, x, y):
        return -self.kernel_x(x, y)

    def kernel_xy(self, x, y):
        diff = x-y 
        arg = self.alpha + diff*diff 
        t1 = -2*self.beta*np.power(arg, self.beta-2)
        t2 = 2*(self.beta-1)*diff*diff + arg
        return t1*t2

    def kernel_yx(self, x, y):
        return self.kernel_xy(x, y)

class KSR(object):
    def __init__(self, target, kernel):
        self.target = target 
        self.kernel = kernel 

    def __call__(self, x, y):
        # evaluate the ksr for the scalar points x, y
        out = self.kernel.kernel_xy(x, y)

        dgdx = self.target.log_grad(x)
        dgdy = self.target.log_grad(y)
        out += self.kernel.kernel_x(x, y)*dgdy
        out += self.kernel.kernel_y(x, y)*dgdx
        out += self.kernel.kernel(x,y)*dgdx*dgdy
        return out

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

    print('Univariate Normal Example')

    n = 9
    print(' --compute n=%d points.' % n)

    # construct the target (ie the normal)
    mu = 3.14
    sig = 2;
    normal = Normal(mu, sig)

    print('Target is normal pdf with parameters:')
    print(' --mu: %.2f' % normal.mu)
    print(' --sigsq: %.2f' % normal.sigsq)

    # construct the IMQ kernel
    alpha = 1.0
    beta = -0.50
    kernel = IMQ(alpha, beta)

    # construct the KSR
    ksr = KSR(normal, kernel)
    
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
   
    if False:
        mult = 5.0
        xmin, xmax = normal.mu-mult*normal.sig, normal.mu+mult*normal.sig
        xx = np.linspace(xmin, xmax, 1001)
        pp = normal.pdf(xx)
        plt.plot(xx, pp)
        plt.xlim([xmin, xmax])
        plt.grid()
        plt.title('Target: Normal(%.2f, %.2f)' % (normal.mu, normal.sigsq))

        for xi in x:
            plt.plot([xi, xi],[0.0, normal.pdf(xi)], color='r')
            plt.scatter(xi, 0.0, color='k', marker='o')
