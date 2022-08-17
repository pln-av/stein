import target_base 
import numpy as np 
import util 

class MVN(target_base.Target):
    def __init__(self, mu, sigsq):
        self.mu = mu 
        self.sigsq = sigsq
        self.inv_sigsq = np.linalg.inv(sigsq)
        self.constant = 1.0/ np.sqrt( np.power(2.0*np.pi, mu.size)*np.linalg.det(self.sigsq) )

    @property
    def dimension(self):
        return self.mu.size

    def __call__(self, z):

        err = z - self.mu
        arg = -0.5*np.matmul( np.matmul(err.T, self.inv_sigsq), err)
        return self.constant*np.exp(arg)

    def log_grad(self, z):

        err = z - self.mu
        return -np.matmul(err.T, self.inv_sigsq)

if __name__ == '__main__':

    import scipy.stats

    # construct normal pdf for d=2
    d = 2
    mu = np.array([0.50 , 0.20])
    sigsq = np.array([  [1.0, 0.20],
                        [0.20, 1.0]
                    ], dtype=float)
    normal = MVN(mu, sigsq)

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
    