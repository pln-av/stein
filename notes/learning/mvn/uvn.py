import numpy as np

class Normal(object):
    def __init__(self, mu, sig):
        self.mu = mu 
        self.sig = sig 
        self.sigsq = sig*sig 
    
    def pdf(self, x):
        c = 1.0/np.sqrt(2.0*np.pi*self.sigsq)
        arg = (x-self.mu)/self.sig 
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