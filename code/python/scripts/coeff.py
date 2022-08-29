import os
import argparse

import numpy as np
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt 
import seaborn 

import spline

def scale_factor(dof):
    return np.sqrt( (dof-2.0)/dof )
     
def packet(x, y, z):
    return '({:.2f},{:.5f},{:.5f})'.format(x, y,z)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--J', type=int, required=False, default=11)
    parser.add_argument('--p_min', type=float, required=False, default=0.01)
    parser.add_argument('--p_max', type=float, required=False, default=0.99)
    parser.add_argument('--knots', type=str, required=False, default='-0.5,0.50')
    parser.add_argument('--order', type=int, required=False, default=3)
    parser.add_argument('--nu', type=str, required=True)
    parser.add_argument('--spline_file', type=str, required=True)
    parser.add_argument('--data_file',  type=str, required=True)
    args = parser.parse_args()

    print(' --> computing spline fits for models with percentiles(%.3f, %.3f)' % (args.p_min, args.p_max))
    print(' --> percentile/grid space has %d points.' % args.J )

    dofs_string, knot_string = args.nu.split(','), args.knots.split(',')
    print(' --> computing %d models, one for each t-dist degree of freedom' % len(dofs_string))
    print(' --> Degrees of freedom considered: ')
    for idx, dof in enumerate(dofs_string):
        print('    --> nu[%d]: %s' % (idx, dof))
    print(' --> fitting regression splines of order %d, with knots:' % args.order)
    for idx, k in enumerate(knot_string):
        print('    --> knot[%d]: %s' % (idx, k))
    print(' --> spline coeffs to file %s' % args.spline_file)
    print(' --> raw data to file %s' % args.data_file)

    # want to use a set of percentiles for the ki, and use same ki for all nu.
    # so choose the ki based on min(nu).  could do this more generically for 
    # other distribution choices
    percentiles = np.linspace(args.p_min, args.p_max, args.J)
    dofs = np.array([float(ni) for ni in args.nu.split(',')])

    print('Using percentiles in range (%.2f, %.2f)' % (args.p_min, args.p_max) )
    print('Computing the above k range for nu in (%.2f, %.2f)' % (dofs[0], dofs[-1]) )

    # computing lookup table.  do it this way so meshgrid is right for plot
    if True:
        p_plot_c = np.zeros(shape=(dofs.size, args.J), dtype=float)
        p_plot_k = np.zeros(shape=(dofs.size, args.J), dtype=float)
        for i in np.arange(dofs.size):
            print('Compute integrals for dof: %.2f' % dofs[i])
            tdist = scipy.stats.t(df=dofs[i], loc=0.0, scale=scale_factor(dofs[i]))
            for j in np.arange(percentiles.size):

                # compute the knots k_i corresponding to this percentile
                k_i = tdist.ppf(percentiles[j])

                # compute required integral for this (dof, k_i) combination
                f = lambda x : tdist.pdf(x)*(x - k_i)*(x-k_i)
                integral, error = scipy.integrate.quad( f, k_i, np.inf )
                p_plot_c[i, j] = integral 
                p_plot_k[i, j] = k_i 

    with open(args.spline_file, 'w') as sf, open(args.data_file,'w') as df:
        
        #knots = np.array([float(k) for k in knot_string], dtype=float)
        order = args.order
        spline = spline.RegressionSpline()

        for idx in np.arange(dofs.size):

            # fit the spline for this dof and write to file
            x, y = p_plot_k[idx, :], p_plot_c[idx, :]
            #knots = np.percentile(x, [0.3, 0.7])
            knots = np.array([-0.5,0.5])
            spline.fit(order, knots, x, y)
            model_str = 'nu=%.3f|%s\n' % (dofs[idx], str(spline))
            sf.write(model_str)

            if False:
                plt.plot(x, y, label='Data', marker='o', alpha=0.5)
                yhat = spline.predict(xx)
                #err = yhat - y 

                xx = np.linspace(x[0], x[-1], 10001)
                plt.plot(xx, spline.predict(xx), label='Spline')
                print(np.min(spline.predict(xx)))
                plt.title('Coeffs vs Spline Fit for nu=%.3f' % (dofs[idx]))
                #plt.plot(x, 100.0*err/y, label='Spline')
                #plt.plot(x, err)

            # write data to file for analysis
            line = '|'.join([packet(dofs[idx], p_plot_k[idx, j], p_plot_c[idx, j]) for j in np.arange(p_plot_c.shape[1])])
            df.write(line + '\n')
        
    plt.xlabel('k')
    plt.ylabel('c')
    

