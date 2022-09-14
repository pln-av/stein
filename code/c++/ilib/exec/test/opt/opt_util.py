import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plotter(x, y, f, use_log=False):

    # create 2d surface plot for objective f
    # this is fixed for 2d functions only
    X = np.meshgrid(x, y)
    Z = np.log(f(X)) if use_log else f(X)

    # filled contour plot
    fig = plt.figure()
    plt.contourf(x, y, Z, levels=30)
    xmin, ymin = f.min()
    plt.scatter(xmin, ymin, marker='x', color='r', s=30)
    return fig

def to_matrix(x):

    # x is a container of points from optimiser
    # place these into a matrix for plotting
    if not x:
        return None
    n, m = len(x), x[0].size
    out = np.zeros(shape=(n, m), dtype=float)
    for i in np.arange(n):
        out[i, :] = x[i]
    return out

# interface for objective functions
class ObjFunction2D(object):
    def __init__(self, ):
        pass
    def __call__(self, x):
        raise NotImplementedError("__call__ must be implemented.")
    def gradient(self, x):
        raise NotImplementedError("gradient(x) must be implemented.")
    def min(self,):
        raise NotImplementedError("min() must be implemented.")

# interface for minimisers
class Criteria(object):
    # container for stop critera
    def __init__(self, max_iters=1000, g_tol=1E-5, f_atol=1E-5, f_rtol=1E-5):
        self.max_iters=max_iters
        self.g_tol = g_tol
        self.f_atol = f_atol
        self.f_rtol = f_rtol
        
class Minimiser(object):
    def __init__(self, criteria):
        self.criteria = criteria
        
    def minimise(self, f, ic):
        # minimise function f from initial condition
        raise NotImplementedError("minimise(f, ic) must be implemented.")
