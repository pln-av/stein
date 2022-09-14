import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os, sys

import obj_fun
import opt_util

def read_results(results_file):
    if not os.path.exists(results_file):
        print('File %s not found.' % results_file)
        sys.exit(1)
    results = pd.read_csv(args.results, header=None)
    n, m = results.shape
    results.rename(columns={i:'x_%d' % i for i in range(m-1)}, inplace=True)
    results.rename(columns={m-1:'f(x)'}, inplace=True)
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    results = read_results(args.results)

    # make a plot of results
    k = 100.0
    obj = obj_fun.Rosenbrock(k)
    n = 1001
    xx = np.linspace(-2.0, 2.0, n)

    # contour plot
    fig = opt_util.plotter(xx, xx, obj, use_log=True)

    # plot initial conditions
    xopt, yopt = obj.min()
    plt.scatter(xopt, yopt, marker='x', color='r', label='Minimum')

    # plot solution trajectory
    plt.plot( results['x_0'], results['x_1'], marker='.', color='k', label='Nesterov' )

    plt.xlabel('x_0'); plt.ylabel('y_0')
    plt.title('Function %s' % args.name)
    
