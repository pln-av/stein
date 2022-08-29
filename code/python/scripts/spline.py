import numpy as np
import matplotlib.pyplot as plt

def horner(coeffs, x):
    # horner polynomial eval
    # note my coeffs are in increasing order
    # ie evaluate p(x) where 
    # p(x) = c[0] + c[1]*x + c[2]*x^2 + ...
    # which is opposite to np.polyval
    p = coeffs.size - 1
    out = coeffs[p]*x
    for i in range(p-1, -1, -1):
        arg = coeffs[i] + out
        mult = x if i>0 else 1.0
        out = mult*arg 
    return out 

def regression(X, y):
    lhs = np.matmul(X.transpose(), X)
    rhs = np.matmul(X.transpose(), y)
    coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return coeffs 

class Polynomial(object):
    def __init__(self,):
        self.coeffs = None

    def fit(self, p, x, y):
        # fit order p polynomial to (x, y)
        X = np.ones( shape=(n, p+1), dtype=float)
        for i in range(1, p+1):
            prev, curr = i-1, i
            X[:, curr] = x*X[:, prev]
        self.coeffs = regression(X, y)

    def predict(self, x):
        # use coeffs (if there are coeffs) to predict array of x
        if self.coeffs is None:
            return None
        return horner(self.coeffs, x) 

class RegressionSpline(object):
    def __init__(self,):
        self.coeffs = None 
        self.knots = None 

    def fit(self, p, knots, x, y):

        # construct regression matrix
        n = x.size
        m = (p + 1) + knots.size
        X = np.ones( shape=(n, m), dtype=float)
        # first the polynomial part
        for i in range(1, p+1):
            prev, curr = i-1, i
            X[:, curr] = x*X[:, prev]
        
        # no basis part
        for i in range(knots.size):
            col_idx = p + i + 1
            k_i = knots[i]
            basis_i = np.power(x - k_i, p)
            ind_i = np.array(x>k_i, dtype=float)
            X[:,col_idx] = basis_i*ind_i 
        self.coeffs = regression(X, y)
        self.knots = knots

    def predict(self, x):

        # use horner to get the first part 
        if self.coeffs is None or self.knots is None:
            return None 
        m = self.knots.size
        p = self.coeffs.size - m - 1 # remember the coeff for the constant
        out = horner(self.coeffs[:-m], x)

        # add on the knot part
        for i in range(m):
            k_i = self.knots[i]
            coeff_idx = (p + 1) + i 
            coeff_i = self.coeffs[coeff_idx]
            basis_i = np.power(x - k_i, p)
            ind_i = np.array(x>k_i, dtype=float)
            out += coeff_i*basis_i*ind_i 
        return out

    def __str__(self,):
        if self.knots is None or self.coeffs is None:
            return ''

        coeffs = '(%s)' % ','.join(['{:.7f}'.format(coeff) for coeff in self.coeffs])
        knots = '(%s)' % ','.join(['{:.7f}'.format(knot) for knot in self.knots]) 
        return 'knots=%s|coeffs=%s' % (knots, coeffs)

if __name__ == '__main__':

    print('Test Spline Code')
    def func(knots, x):
        out = 1
        out += 2*np.power(x, 1)
        out += -3*np.power(x, 2) 
        out += np.power(x, 3) 
        out += -0.5*np.power(x, 4)
        for ki in knots:
            out += 2.0*np.power(x-ki, 4)*np.array(x>ki, dtype=float) 
        return out 

    # generate x data
    knots = np.array([-1.0, 1.0], dtype=float)
    x_low, x_high, n = -np.pi, np.pi, 21
    x = np.linspace(x_low, x_high, n)
    y = func(knots, x)

    # first do pure polynomial regression
    p = 4 # this is hardcoded into func 
    poly = Polynomial()
    for p in range(3, p+1):
    
        poly.fit(p, x, y)
        yhat = poly.predict(x)
        plt.plot(x, yhat-y, alpha=0.5, label='Polynomial(%d)' % p)

    spline = RegressionSpline()
    spline.fit(p, knots, x, y)
    plt.plot(x, spline.predict(x) - y, linestyle='None', marker='o', label='Spline(%d)' % p)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Polynomial and Regression Spline Errors')
    plt.legend()
    plt.grid()
    plt.show()

