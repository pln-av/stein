import numpy as np
import scipy
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt
import seaborn


def pdf(x, nu):

    # compute constant
    c1 = scipy.special.gamma(0.5*(nu+1.0))
    c2 = np.sqrt( np.pi*nu) * scipy.special.gamma(0.5*nu)
    c = c1 / c2

    arg = 1 + x*x/nu
    return c * np.power(arg, -0.5*(nu + 1.0))

def spdf(x, nu):

    s = np.sqrt( (nu-2.0)/nu )
    c1 = scipy.special.gamma(0.5*(nu+1.0))
    c2 = np.sqrt( np.pi*nu) * scipy.special.gamma(0.5*nu)
    c = (1 / s)*(c1 / c2)

    y = x/s
    arg = 1 + y*y/nu
    return c * np.power(arg, -0.5*(nu+1.0))

    
if __name__ == '__main__':

    print('Misc program to check Tstd code with scipy')

    # generate pdf
    nu = 2.21
    xx = np.linspace(-10.0, 10.0, 101)
    yy = pdf(xx, nu)

    # compute numeric derivatives to check formulation

    f = lambda x : x*pdf(x, nu)
    integral, area = scipy.integrate.quad(f, -np.inf, np.inf)

    print('-- T Distribution Moments -- ')
    print('Numeric Mean(nu=%.2f): %.5f' % (nu, integral))
    f = lambda x : x*x*pdf(x, nu)
    integral, area = scipy.integrate.quad(f, -np.inf, np.inf)
    print('Numeric Variance(nu=%.2f): %.5f' % (nu, integral) )
    print('Analytic Variance(nu=%.2f): %.5f' % (nu, nu/(nu-2.0)))

    print('-- Tstd Distribution Moments --')
    f = lambda x : x*spdf(x, nu)
    integral, area = scipy.integrate.quad(f, -np.inf, np.inf)
    print('Numeric Mean(nu=%.2f): %.5f' % (nu, integral))
    f = lambda x : x*x*spdf(x, nu)
    integral, area = scipy.integrate.quad(f, -np.inf, np.inf)
    print('Numeric Variance(nu=%.2f): %.5f' % (nu, integral))
    print('Analytic Variance(nu=%.2f): 1.0' % nu)
    
    yy2 = spdf(xx, nu)
    plt.plot(xx, yy,  label='t(%.2f)' % nu)
    plt.plot(xx, yy2, label='tstd(%.2f)' % nu)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.title('t distribution')
    plt.legend()
    
