import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import argparse

class Bowl(object):
    def __init__(self, ):
        pass

    def __call__(self, X, Y):
        return 2*np.power(X-1, 2) + np.power(Y, 4)

    def __str__(self, ):
        return 'Bowl'

    def min(self, ):
        return (1, 0, 0)

def plot_contourf(x, y, f):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    plt.contourf(X, Y, Z, levels=30)
    plt.title('Function %s' % f)
    return fig

if __name__ == '__main__':

    print('Plotting Results')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    args = parser.parse_args()

    # plot the objective function
    x_min, x_max, n = -15, 15, 1001
    xx = np.linspace(x_min, x_max, n)
    bowl = Bowl()
    fig = plot_contourf(xx, xx, bowl)
    
    df = pd.read_csv(args.results_file, header=None)
    iterations, cols = df.shape
    df.rename( columns={i:'x_%d' % i for i in range(cols-1)}, inplace=True )
    df.rename( columns={cols-1: 'f'}, inplace=True )
    
    plt.plot(df['x_0'], df['x_1'], marker='o', alpha=0.75, color='k', markersize=2, label=args.method)
    plt.scatter(bowl.min()[0], bowl.min()[1], marker='x', color='r')
    plt.grid()
    plt.xlabel('x');
    plt.ylabel('y')
    plt.legend()
    fig.show()
