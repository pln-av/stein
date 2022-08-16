import numpy as np
import scipy as sp

def norm2(x):
    return np.sum(x*x)

class IMQ(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x, y):
        return np.power( self.alpha + norm2(x-y), self.beta)

    def grad_x(self, x, y):
        diff = x - y
        arg = self.alpha + norm2(diff)
        return 2.0*self.beta*diff*np.power(arg, self.beta - 1.0)

    def grad_y(self, x, y):
        return -self.grad_x(x, y)

    def grad_xy(self, x, y):
        diff = x-y 
        arg = self.alpha + norm2(diff)
        t1 = 2.0*(self.beta-1)*diff*diff*np.power(arg, self.beta-2.0)
        t2 = np.power(arg, self.beta - 1.0)
        return -2.0*self.beta*(t1 + t2)

    def grad_yx(self, x, y):
        return self.grad_xy(x, y)

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

class SRK(object):
    def __init__(self, target, kernel):
        self.target = target 
        self.kernel = kernel 

    def __call__(self, x, y):

        # evaluate the ksr for the (array) points x, y
        dgdx = self.target.log_grad(x)
        dgdy = self.target.log_grad(y)
        t1 = np.sum( self.kernel.grad_xy(x, y) )
        t2 = np.sum( self.kernel.grad_x(x, y)*dgdy )
        t3 = np.sum( self.kernel.grad_y(x, y)*dgdx )
        t4 = self.kernel.kernel(x, y) * np.sum( dgdx*dgdy ) 
        t = t1 + t2 + t3 + t4
        return t

if __name__ == '__main__':

    # construct normal pdf for d=2
    d = 2
    mu = np.random.rand(d)
    sigsq = np.array([[3.0,0.6],[0.6,6.0]], dtype=float)
    normal = Normal(mu, sigsq)

    alpha = 1.0
    beta = -0.50
    imq = IMQ(alpha, beta)
    print('Checking derivative calculations vs numeric derivatives for:')
    print(' --> IMQ kernel')
    print(' --> normal pdf')

    eps = 1.0E-7 
    if True:
        print('\nTesting Normal PDF Calculations vs Scipy and Numerics')

        sp_normal = sp.stats.multivariate_normal(mean=mu, cov=sigsq)
        m = 5 # number of test points
        z = -1 + 2*np.random.rand(m, d)


        print(' --> Test 1: PDF calculation.')
        for idx in range(m):
            p = z[idx, :]
            print(' * Point %d: Scipy: %.7f vs MVN: %.7f' % (idx, sp_normal.pdf(p), normal(p)))
        print('')
        print(' --> Gradient Calculation')
        for idx in range(m):
            p = z[idx, :]
            dgdx = normal.log_grad(p)
            for i in range(d):
                v = np.zeros(d, dtype=float)
                v[i] = 1.0
                p_u = p + eps*v
                grad = ( np.log( sp_normal.pdf(p_u)) - np.log( sp_normal.pdf(p)) )/eps
                print(' * Point %d: d log p / d_x%d: Scipy: %.7f vs MVN: %.7f' % (idx, i, dgdx[i], grad))

    """
    # construct IMQ kernald
   

    x = np.array([[1.0], [6.0]], dtype=float)
    y = np.array([[2.0], [-1.0]], dtype=float)

    # kernel gradients
    d2kdxy = kernel.kernel_xy(x, y)
    dkdx = kernel.kernel_x(x, y)
    dkdy = kernel.kernel_y(x, y)

    # log target gradients
    dgdx = normal.log_grad(x)
    dgdy = normal.log_grad(y)

    srk = SRK(normal, kernel)
    """
    