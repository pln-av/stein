import numpy as np 

def fill_grid(f, xp, yp):

    # create 2d grid using arrays xp, yp 
    # fill using function f
    X, Y = np.meshgrid(xp, yp)
    rows, cols = X.shape[0], X.shape[1]
    surf = np.zeros(shape=X.shape, dtype=float)
    for r in np.arange(rows):
        for c in np.arange(cols):
            p = np.array([[X[r, c]],[ Y[r,c]]], dtype=float)
            surf[r, c] = f(p)
    return X, Y, surf 

def grid_search(X, Y, Z, obj_type='max'):
    
    # return x,y positions of max/min value in surface
    loc_func = np.argmax if obj_type == 'max' else np.argmin
    idx, idy = np.unravel_index( loc_func(Z), Z.shape)
    return X[idx, idy], Y[idx, idy], Z[idx, idy]