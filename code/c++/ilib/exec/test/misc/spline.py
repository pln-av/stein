import numpy as np
import matplotlib.pyplot as plt
import seaborn

class Spline(object):
    def __init__(self, knots):

        
        self.knots = knots
        self.model = np.ones(knots.size, dtype=bool)
        self.n_active = np.sum(self.model)
        if self.knots.size < self.n_active:
            print('Not enough knots!!!')
        self.coeffs = np.zeros(3, dtype=float)
        self.spline_coeffs = np.zeros(self.knots.size, dtype=float)

    def check_setup(self):
        # fill this in!!
        return True
    
    def update_model(self, model):
        self.model = model
        self.n_active = np.sum(self.model)

    def update_coeffs(self, coeffs):
        self.coeffs = coeffs

    def update_spline_coeffs(self, spline_coeffs):
        self.spline_coeffs = spline_coeffs
    
    def evaluate(self, x):
        #if coeffs.size != self.n_active:
        #    print('Incorrect number of coeffs.')
        #    return None

        out = self.coeffs[0] + x*(self.coeffs[1] + x*self.coeffs[2])
        knot_idx, coeff_idx = 0, 0
        for i in np.arange(self.knots.size):

            #print('Knot: %d' % i)
            if not self.model[i]:
                #print('knot %d not active.' % i)
                knot_idx += 1
            else:

                # active knot
                beta_i = self.spline_coeffs[coeff_idx]
                knot_i = self.knots[knot_idx]
                out += 0.0 if x<knot_i else beta_i*(x-knot_i)*(x-knot_i)
                
                knot_idx += 1
                coeff_idx += 1
                
        return out
    
                
    
if __name__ == '__main__':

    knots = np.array([-2, -1,13, 2.6,4], dtype=float)
    s = Spline(knots)
    coeffs = np.array([1, 1.1, 0.3], dtype=float)
    s.update_coeffs(coeffs)

    # spline coeffs for the full model
    spline_coeffs = np.array([0.3, 1.5, 1.2, 1, 0.7], dtype=float)
    xx = np.linspace(-2, 2, 101)
    out = [s.evaluate(xi) for xi in xx]
    plt.plot(xx, out, label='model 1')
    
    model = np.array([1, 1, 0, 1, 0], dtype=bool)
    spline_coeffs = np.array([0.03, 0.15, 0.1], dtype=float)
    s.update_model(model)
    s.update_spline_coeffs(spline_coeffs)
    out = [s.evaluate(xi) for xi in xx]
    
    plt.plot(xx, out, label='model 2')
    #s.update_model(model)
    #s.evaluate(x, coeffs)
    
