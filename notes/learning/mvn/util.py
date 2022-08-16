import numpy as np 

def norm2(x):
    return np.sum(x*x)

def fill_grid(f, xp, yp):

    # create 2d grid using arrays xp, yp 
    # fill using function f
    X, Y = np.meshgrid(xp, yp)
    rows, cols = X.shape[0], X.shape[1]
    surf = np.zeros(shape=X.shape, dtype=float)
    for r in np.arange(rows):
        for c in np.arange(cols):
            p = np.array([X[r, c],Y[r,c]], dtype=float)
            surf[r, c] = f(p)
    return X, Y, surf 

def grid_search(X, Y, Z, obj_type='max'):
    
    # return x,y positions of max/min value in surface
    loc_func = np.argmax if obj_type == 'max' else np.argmin
    idx, idy = np.unravel_index( loc_func(Z), Z.shape)
    return X[idx, idy], Y[idx, idy], Z[idx, idy]

def objective(z, srk, points):
    # multivariate greedy objective function
    out = 0.5*srk(z, z)
    if points is not None:
        for xi in points:
            out += srk(xi, z)
    return out

def grad_1d(f, x, v, eps = 1.0E-5):
    # 1st derivative of function of 1 variable with dimension d
    # use vector v to point in correct dimension
    x_u, x_d = x + eps*v, x - eps*v
    f_u, f_d = f(x_u), f(x_d)
    return 0.50*(f_u - f_d)/eps
    
def grad_2d(f, x, y, v, eps=1.0E-5):
    # 1st and mixed second derivatrives using v as above
    x_u, x_d = x + eps*v, x - eps*v
    y_u, y_d = y + eps*v, y - eps*v

    gradx  = 0.50*(f(x_u, y) - f(x_d, y))/eps
    grady  = 0.50*(f(x, y_u) - f(x, y_d))/eps
    gradxy = 0.25*(f(x_u, y_u) - f(x_d, y_u) - f(x_u, y_d) + f(x_d, y_d))/(eps*eps)
    return gradx, grady, gradxy 

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

class SRK(object):
    def __init__(self, target, kernel):
        self.target = target 
        self.kernel = kernel 

    def __call__(self, x, y):

        # evaluate the ksr for the (array) points x, y
        #import pdb 
        #pdb.set_trace()

        dgdx = self.target.log_grad(x)
        dgdy = self.target.log_grad(y)
        
        dkdx = self.kernel.grad_x(x, y)
        dkdy = self.kernel.grad_y(x, y)

        out = self.kernel.grad_xy(x, y)
        out += dkdx*dgdy
        out += dkdy*dgdx 
        out += self.kernel(x, y)*dgdx*dgdy 

        return np.sum(out)