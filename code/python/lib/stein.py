import numpy as np

class SRK(object):

    """
    Stein Reproducing Kernel
    """
    def __init__(self, target, kernel):
        self.target = target 
        self.kernel = kernel 

    def __call__(self, x, y):

        # evaluate the ksr for the (array) points x, y
        # each of dgdx, dkdy etc etc are arrays of size d
        dgdx = self.target.log_grad(x)
        dgdy = self.target.log_grad(y)
        
        dkdx = self.kernel.grad_x(x, y)
        dkdy = self.kernel.grad_y(x, y)

        out = self.kernel.grad_xy(x, y)
        out += dkdx*dgdy
        out += dkdy*dgdx 
        out += self.kernel(x, y)*dgdx*dgdy 

        return np.sum(out)

class KSD(object):

    """
    Kernel Stein Discrepency
     -> compute the KSD for a sequence of points
    """

    def __init__(self, srk):
        self.srk = srk

    def __call__(self, points):

        if not points:
            return np.nan
             
        out = 0.0
        for p1 in points:
            for p2 in points:
                out += self.srk(p1, p2)
        n = len(points)
        return np.sqrt( out / (n*n) )


def greedy_objective(z, srk, points):
    # multivariate greedy objective function
    out = 0.5*srk(z, z)
    if points is not None:
        for xi in points:
            out += srk(xi, z)
    return out
