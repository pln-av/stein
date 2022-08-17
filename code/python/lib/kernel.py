import numpy as np
import kernel_base
import util 

# currently only contains IMQ kernel, but consider adding more
class IMQ(kernel_base.Kernel):
    def __init__(self, dimension, alpha, beta):
        
        self.alpha = alpha 
        self.beta = beta 
        self.d = dimension

    @property
    def dimension(self):
        return self.d 

    def __call__(self, x, y):
        return np.power( self.alpha + util.norm2(x-y), self.beta)

    def grad_x(self, x, y):
        diff = x - y
        arg = self.alpha + util.norm2(diff)
        return 2.0*self.beta*diff*np.power(arg, self.beta - 1.0) 

    def grad_y(self, x, y):
        return -self.grad_x(x, y) 

    def grad_xy(self, x, y):
        diff = x-y 
        arg = self.alpha + util.norm2(diff)
        t1 = 2.0*(self.beta-1)*diff*diff*np.power(arg, self.beta-2.0)
        t2 = np.power(arg, self.beta - 1.0)
        return -2.0*self.beta*(t1 + t2)

    def grad_yx(self, x, y):
        return self.grad_xy(x, y) 

if __name__ == '__main__':

    # test kernel codes
    d = 5
    alpha = 1.5
    beta = -0.50
    imq = IMQ(d, alpha, beta)

    m = 2 # number of test points
    print(' --> Test 1: Kernel Derivative Calculation.')
    z1 = np.random.rand(m, d)
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
