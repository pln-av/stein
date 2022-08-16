import numpy as np
import scipy as sp
import scipy.stats
import util 


class Normal(object):
    def __init__(self, mu, sigsq):

        # copy the scipy notation
        # d is dimension.  
        # mu is array of means
        # sigsq is a matrix size (dxd)
        self.mu = np.atleast_2d(mu).T
        self.sigsq = sigsq 
        self.inv_sigsq = np.linalg.inv(sigsq)
        self.d = self.mu.size

        # we can compute the constant
        arg = np.power(2.0*np.pi, self.d)*np.linalg.det(self.sigsq)
        self.constant = 1.0/ np.sqrt( arg )

    def __call__(self, z):

        z = np.atleast_2d(z).T
        err = z - self.mu 
        arg = -0.5*np.matmul( np.matmul(err.T, self.inv_sigsq), err).item()
        return self.constant*np.exp(arg)

    def log_grad(self, z):
        z = np.atleast_2d(z).T
        err = z - self.mu
        return -np.matmul(err.T, self.inv_sigsq)[0]



if __name__ == '__main__':

    # construct normal pdf for d=2
    d = 2
    mu = np.array([0.0 , 0.0])
    sigsq = np.array([  [1.0,0.0],
                        [0.0, 1.0]
                    ], dtype=float)
    normal = Normal(mu, sigsq)

    alpha = 1.00
    beta = -0.50
    imq = util.IMQ(alpha, beta)
    print('Checking derivative calculations vs numeric derivatives for:')
    print(' --> IMQ kernel')
    print(' --> normal pdf')

    eps = 1.0E-5
    if True:
        print('\nTesting Normal PDF Calculations vs Scipy and Numerics')

        sp_normal = scipy.stats.multivariate_normal(mean=mu, cov=sigsq)
        m = 2 # number of test points
        z = -1 + 2*np.random.rand(m, d)

        print(' --> Test 1: PDF calculation.')
        for idx in range(m):
            p = z[idx, :]
            print(' * Point %d: Scipy: %.7f vs MVN: %.7f' % (idx, sp_normal.pdf(p), normal(p)))
        print('')
        print(' --> Test 2: Log Gradient Calculation')
        g = lambda z : np.log(sp_normal.pdf(z))

        for idx in range(m):
            p = z[idx, :]
            dgdx = normal.log_grad(p)
            print(' ** Point %d ** ')
            
            for i in range(d):
                v = np.zeros(d, dtype=float)
                v[i] = 1.0
                grad = util.grad_1d(g, p, v)
                print('     * Point %d: d log p / d_x%d: Scipy: %.7f vs MVN: %.7f' % (idx, i, grad, dgdx[i]))

        print('')
        print(' --> Test 3: Kernel Derivative Calculation.')
        z1 = z 
        z2 = np.random.rand(m, d)
        for idx in range(m):
            px = z1[idx, :]
            py = z2[idx, :]

            dkdx = imq.grad_x(px, py)
            dkdy = imq.grad_y(px, py)
            d2k  = imq.grad_xy(px, py)
            for i in range(d):
                v = np.zeros(d, dtype=float)
                v[i] = 1.0
                gradx, grady, gradxy = util.grad_2d(imq, px, py, v)
                
                print(' ** Point %d **' % idx)
                print('     * Points %d: dkd_x%d: Numeric: %.7f | Analytic: %.7f' % (idx, i, gradx, dkdx[i]))
                print('     * Points %d: dkd_y%d: Numeric: %.7f | Analytic: %.7f' % (idx, i, grady, dkdy[i]))
                print('     * Points %d: d2k_%d: Numeric: %.7f | Analytic: %.7f' % (idx, i, gradxy, d2k[i]))

        
    