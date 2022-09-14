# put minimisers here
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import opt_util
import obj_fun

class HyperGradientDescent(opt_util.Minimiser):
    def __init__(self, criteria, alpha, beta):
        super().__init__(criteria)
        self.alpha = alpha
        self.beta = beta

    def minimise(self, f, ic):
        idx, xt, tol = 0, ic, 1.0E8
        fprev = f(xt)
        xstore = [xt]
        gtp = np.zeros(xt.size, dtype=float)
        tolic = np.linalg.norm( f.gradient(xt) )
        alpha = self.alpha
        
        while idx < self.criteria.max_iters and tol > self.criteria.g_tol:
            
            gt = f.gradient(xt)

            # alpha update.  can go wild with large gradients
            a_low, a_high = 0.9*alpha, 1.1*alpha
            alpha_adj = self.beta*np.dot(gt, gtp)
            alpha_hat = alpha + alpha_adj
            alpha = np.clip(alpha_hat, a_low, a_high)

            # position update
            xt = xt - alpha*gt

            # save results
            ft = f(xt)
            xstore.append(xt)
            tol = np.linalg.norm(gt)

            idx += 1
            fprev = ft
            gtp = gt
            
        return xstore, (tolic, tol), ft
    
class GradientDescent(opt_util.Minimiser):
    def __init__(self, criteria, alpha):
        super().__init__(criteria)
        self.alpha = alpha

    def minimise(self, f, ic):

        # initial setup
        idx, xt, tol = 0, ic, 1.0E8
        fprev = f(xt)
        xstore = [xt]

        # use gradient tolerance
        tolic = np.linalg.norm( f.gradient(xt) )
        while idx < self.criteria.max_iters and tol > self.criteria.g_tol:

            # do update
            gt = f.gradient(xt)
            ut = -self.alpha*gt
            xt = xt + ut

            # save results of this update
            ft = f(xt)
            xstore.append(xt)
            tol = np.linalg.norm(gt)

            # prepare for next iteration
            idx += 1
            fprev = ft

        return xstore, (tolic, tol), ft

class Nesterov(opt_util.Minimiser):
    def __init__(self, criteria, alpha, beta, gamma):
        super().__init__(criteria)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def minimise(self, f, ic):

        # initial setup
        idx, xt, tol = 0, ic, 1.0E8
        fprev = f(xt)
        xstore = [xt]
        vt = np.zeros(xt.size, dtype=float) 

        # reduce learning rate
        alpha_k = self.alpha
        tolic = np.linalg.norm( f.gradient(xt) )
        while idx < self.criteria.max_iters and tol > self.criteria.g_tol:

            # Nesterov projection
            xhat = xt + self.beta*vt
            ghat = f.gradient(xhat)
            gnorm = ghat / np.linalg.norm(ghat)

            # nesterov velocity
            vt = self.beta*vt - alpha_k*gnorm

            # position update
            xt = xt + vt

            # save results of this update
            ft = f(xt)
            xstore.append(xt)
            tol = np.linalg.norm(ghat)/xhat.size

            # prepare for next iteration
            idx += 1
            fprev = ft
            alpha_k *= self.gamma

        return xstore, (tolic, tol), ft

class Nesterov2(opt_util.Minimiser):
    def __init__(self, criteria, alpha, beta, gamma):
        super().__init__(criteria)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def minimise(self, f, ic):

        # initial setup
        idx, xt, tol = 0, ic, 1.0E8
        fprev = f(xt)
        xstore = [xt]
        vt = np.zeros(xt.size, dtype=float) 

        # reduce learning rate
        alpha_k = self.alpha
        tolic = np.linalg.norm( f.gradient(xt) )
        while idx < self.criteria.max_iters and tol > self.criteria.g_tol:

            # Nesterov projection
            xhat = xt + self.beta*vt
            ghat = f.gradient(xhat)
            gnorm = ghat / np.linalg.norm(ghat)

            # nesterov velocity
            vt = self.beta*vt - alpha_k*gnorm

            # position update
            xt = xt + vt
            ft = f(xt)

            # save results of this update
            xstore.append(xt)
            tol = np.linalg.norm(ghat)

            # prepare for next iteration
            idx += 1
            fprev = ft
            alpha_k *= self.gamma

        return xstore, (tolic, tol), ft
    
class Adam(opt_util.Minimiser):
    def __init__(self, criteria, alpha, gamma_v, gamma_s):
        super().__init__(criteria)
        self.alpha = alpha
        self.gamma_v = gamma_v
        self.gamma_s = gamma_s

    def minimise(self, f, ic):
        idx, xt, tol = 0, ic, 1.0E8
        fprev = f(xt)
        xstore = [xt]
        vt = np.zeros(xt.size, dtype=float)
        st = np.zeros(xt.size, dtype=float)
        tolic = np.linalg.norm( f.gradient(xt) )
        while idx < self.criteria.max_iters and tol > self.criteria.g_tol:
            gt = f.gradient(xt)
            vt = self.gamma_v*vt + (1-self.gamma_v)*gt
            st = self.gamma_s*st + (1-self.gamma_s)*gt*gt

            scale_v = np.power(self.gamma_v, idx+1)
            scale_s = np.power(self.gamma_s, idx+1)

            vhat = vt / (1.0 - scale_v)
            shat = st / (1.0 - scale_s)

            adj = - self.alpha*vhat / (1.0E-8 + np.sqrt(shat) )
            xt = xt + adj
            ft = f(xt)
            xstore.append(xt)
            tol = np.linalg.norm(gt)

            # prepare for next iteration
            idx += 1
        return xstore, (tolic, tol), ft

def do_min(minimiser, obj, ic):
    
    xpoints, tol, ft = minimiser.minimise(obj, xic)
    xpoints = opt_util.to_matrix(xpoints)
    xopt = xpoints[-1]
    xerr = np.linalg.norm(xopt - obj.min())
    ferr = np.linalg.norm(ft - obj(obj.min()))
    niters = xpoints.shape[0]
    print('Minimiser used:')
    print(' -- iterations: %d' % niters)
    print(' -- gtol: %.5f -> %.5f' % tol)
    print(' -- obj: %.5f' % ft )
    print(' -- xerr: %.5f' % xerr)
    print(' -- ferr: %.5f' % ferr)
    return xpoints, tol, ft

if __name__ == '__main__':


    # objective
    obj = obj_fun.Rosenbrock(100)
    #obj = obj_fun.Bowl()
    xic = np.array([-1.50, 0.50])
    
    # plot solution surface
    n = 1001
    xx = np.linspace(-5, 5, n)
    fig = opt_util.plotter(xx, xx, obj)
    plt.scatter(xic[0], xic[1], marker='o', color='b', label='IC')

    criteria = opt_util.Criteria(g_tol=0.01, max_iters=1000)
    # minimise with gd, and add solutions

    do_HGD = False
    if do_HGD:
        alpha = 0.001
        beta = 0.0001
        hgd = HyperGradientDescent(criteria, alpha, beta)
        print('HyperGradientDescent')
        xpoints, tol, ft = do_min(hgd, obj, xic)
        niters = xpoints.shape[0]
        plt.plot(xpoints[:,0], xpoints[:,1], marker='.', color='b', label='HGD(%d)' % niters)
        
    do_Adam = False
    if do_Adam:
        alpha = 0.25
        gamma_v = 0.9
        gamma_s = 0.999
        print('Adam')
        adam = Adam(criteria, alpha, gamma_v, gamma_s)
        xpoints, tol, ft = do_min(adam, obj, xic)
        niters = xpoints.shape[0]
        plt.plot(xpoints[:, 0], xpoints[:, 1], marker='.', color='y', label='Adam(%d)' % niters)

    do_Nesterov2 = False
    if do_Nesterov2:
        alpha = 0.2
        beta = 0.9
        gamma = 0.9
        print('Nesterov2')
        nesterov2 = Nesterov2(criteria, alpha, beta, gamma)
        xpoints, tol, ft = do_min(nesterov2, obj, xic)
        niters = xpoints.shape[0]
        plt.plot(xpoints[:,0], xpoints[:,1], marker='.', color='b', label='Nesterov2(%d)' % niters)
        f2 = ft
        
    do_Nesterov = True
    if do_Nesterov:
        alpha = 0.2
        beta = 0.8
        gamma = 0.9
        print('Nesterov')
        nesterov = Nesterov(criteria, alpha, beta, gamma)
        xpoints, tol, ft = do_min(nesterov, obj, xic)
        niters = xpoints.shape[0]
        plt.plot(xpoints[:, 0], xpoints[:, 1], marker='.', color='k', label='Nesterov(%d)' % niters)
        f1 = ft

    do_GD = False
    if do_GD:
        alpha = 0.001
        gd = GradientDescent(criteria, alpha)
        xpoints, tol, ft = do_min(gd, obj, xic)
        niters = xpoints.shape[0]
        plt.plot(xpoints[:, 0], xpoints[:, 1], marker='.', color='r', label='GD(%d)' % niters)

    plt.legend()
    plt.xlabel('x0'); plt.ylabel('x1')
    plt.title('Bowl Function')
    plt.show()
    
