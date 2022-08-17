import numpy as np 
import datetime as dt
def norm2(x):
    return np.sum(x*x)

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

def time_string(s, e):
    return dt.timedelta(minutes=(e-s)/60)