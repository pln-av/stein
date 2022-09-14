# place objective functions we want to play with here
import numpy as np
import opt_util

class Bowl(opt_util.ObjFunction2D):
    def __init__(self, ):
        super().__init__()
        
    def __call__(self, x):
        x0, x1 = x
        return 2.0*x0*x0 + x1*x1*x1*x1

    def gradient(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=float)
        g[0] = 4*x0
        g[1] = 4*x1*x1*x1
        return g

    def min(self, ):
        return (0.0, 0.0)

class Rosenbrock(opt_util.ObjFunction2D):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, x):
        x0, x1 = x
        return np.power(1-x0, 2) + self.k*np.power(x1 - x0*x0, 2)

    def gradient(self, x):
        x0, x1 = x
        arg = x1 - x0*x0
        g = np.zeros(2, dtype=float)
        g[0] = -2*(1-x0) - 4*self.k*x0*arg
        g[1] = 2*self.k*arg
        return g
    
    def min(self, ):
        return (1.0, 1.0)
    
if __name__ == '__main__':

    bowl = Bowl()
    k = 20
    rosenbrock = Rosenbrock(k)
    n = 1001
    xx = np.linspace(-3, 3, n)
    yy = xx.copy()
    fig = opt_util.plotter(xx, yy, bowl)
