import numpy as np
import scipy.stats

def norm2(x):
    return np.sum(x*x)

class Kernel(object):
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
        # evaluate the ksr for the scalar points x, y
        out = self.kernel.kernel_xy(x, y)

        dgdx = self.target.log_grad(x)
        dgdy = self.target.log_grad(y)
        out += self.kernel.kernel_x(x, y)*dgdy
        out += self.kernel.kernel_y(x, y)*dgdx
        out += self.kernel.kernel(x, y)*dgdx*dgdy

if __name__ == '__main__':

    # normal object
    mu = np.array([2.0, -3], dtype=float)
    sigsq = np.array([[3.0,0.6],[0.6,6.0]], dtype=float)
    normal = Normal(mu, sigsq)

    # kernel object
    alpha = 1.0
    beta = -0.5
    k = Kernel(alpha, beta)
    
    srk = SRK(normal, k)
    n = 2
    x = np.array([1.0, 6.0], dtype=float)
    y = np.array([2.0, -1.0], dtype=float)

    #x = np.random.rand(n)
    #y = np.random.rand(n)
    kxy = k(x, y)
    dgdx = normal.log_grad(x)
    dgdy = normal.log_grad(y)

    t1 = np.sum( k.grad_xy(x, y) )
    t2 = np.sum( k.grad_x(x, y)*dgdy )
    t3 = np.sum( k.grad_y(x, y)*dgdx )
    t4 = np.sum( kxy * np.sum( dgdx*dgdy ) )
    t = t1 + t2 + t3 + t4


    z = np.random.rand(2)
    
    eps = 1.0E-6

    # compare log gradient
    if False:
        g = lambda x : np.log( normal(x) )
        lg = normal.log_grad(z)
        for i in range(z.size):
            v = np.zeros(z.size, dtype=float)
            v[i] = 1.0
            z_up = z + eps*v
            z_down = z - eps*v
            grad = (g(z_up) - g(z_down))/(2.0*eps)
            print('Gradient(z_%d): %.6f | Analytic: %.6f' % (i,grad, lg[i]))
        
    if False:
       
        n = 3
        x = np.random.rand(n)
        y = np.random.rand(n)

        # partial derivatives in x
        gx = k.grad_x(x, y)
        gy = k.grad_y(x, y)
        gxy = k.grad_xy(x, y)
        print('-------------------------------------------------------')
        for i in range(x.size):
            v = np.zeros(x.size)
            v[i] = 1.0

            # partial wrt x
            x_up = x + eps*v
            grad = (k(x_up, y) - k(x, y))/eps
            print('Gradient(x_%d): %.6f | Analytic: %.6f' % (i, grad, gx[i]))

            # partial wrt y
            y_up = y + eps*v 
            grad = (k(x, y_up) - k(x, y))/eps
            print('Gradient(y_%d): %.6f | Analytic: %.6f' % (i, grad, gy[i]))

            # mixed wrt x, y
            grad = (k(x_up, y_up) - k(x_up, y) - k(x, y_up) + k(x, y)) / (eps*eps)
            print('Gradient(x_%d,y_%d): %.6f | Analytic: %.6f' % (i, i, grad, gxy[i]))
            print('-------------------------------------------------------')
    
