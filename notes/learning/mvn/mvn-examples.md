\newpage

# Examples Part 1: Multivariate Normal Distribution with IMQ Kernel

In this section I will work through a few simple examples to clarify my understanding
of the problem.  These will focus on the IMQ kernel and the target distribution will
be the multivariate normal distribution.  The multivariate normal distribution with mean $\mu$, 
covariance $\Sigma$ and dimension $d$ is 
\eq{ p(\mb{z}) = \frac{1}{\sqrt{ \left( \left( 2 \pi\right)^d \textrm{det} \Sigma \right) }} \, \textrm{exp}\left[ -\frac{1}{2} \transpose{ \left(\mb{z}-\mu\right) }  \Sigma^{-1} \left(\mb{z}-\mu\right) \right] \nonumber }
with corresponding log target $g(\mb{z})$ 
\eq{ g(\mb{z}) = -\frac{1}{2} \textrm{log}\, \left( \left( 2 \pi\right)^d \textrm{det}\left( \Sigma \right)  \right) -\frac{1}{2} \transpose{ \left(\mb{z}-\mu\right) }  \Sigma^{-1} \left(\mb{z}-\mu\right) \nonumber }
where $\mb{z}\in \mbb{R}^d$ and of course the additive constant can be ignored for our purposes.  The 
IMQ kernel is
\eq{ k(\mb{x},\mb{y}) = \left( \alpha + \|\mb{x}-\mb{y} \|^2 \right)^\beta \nonumber }
with $\alpha>0$ and $-1 < \beta < 0$.

The multivariate normal is implemented in \verb|mvn.py| and the IMQ kernel in \verb|util.py|.
Note that when run \verb|mvn.py| tests the gradient calculations

* $\mb{\nabla} \, g(\mb{z})$ 
* $\mb{\nabla}_{x} \, k(\mb{x}, \mb{y})$ and $\mb{\nabla}_{y} \, k(\mb{x}, \mb{y})$
* The $d$ terms in $\mb{\nabla}_{x} \cdot \mb{\nabla}_{y} \, k(\mb{x}, \mb{y})$

against numeric approximations.  

\newpage 

## Example 1.1: Univariate Normal Distribution

In this example I take $d=1$ and investigate:

* What do the KSD and SRK formulas actually look like?
* What kind of objective functions result from the Greedy Minimisation strategy?

With $d=1$ the objective function for the greedy minimisation strategy are easy
to visualise.  This is the univariate normal target $p(x)$. This will be the only case where the bold 
font will not be used for the random variable.  The SRK is 
\eq{  
    k_0(x,y) = \frac{\partial^2 k }{\partial x \partial y} + \pwrt{k}{x} \pwrt{g}{y} +  \pwrt{k}{y} \pwrt{g}{x} + k(x,y)  \pwrt{g}{x} \pwrt{g}{y} 
}
The log gradient for this choice of $p$ is
\eq{
    \pwrt{g}{z} = - (z-\mu)/\Sigma
}
where both $\mu, \Sigma \in \mbb{R}$, and the kernel derivatives are
\eqa{
 \pwrt{k}{x} = & 2 \beta (x-y)  \left( \alpha + \|x-y \|^2 \right) ^{\beta-1}  \nonumber \\
 \pwrt{k}{y} = & -2 \beta (x-y) \left( \alpha + \|x-y \|^2 \right)^{\beta-1} \nonumber \\
 \frac{\partial^2 k }{\partial x \partial y} = & -2 \beta \left( \alpha + \|x-y \|^2 \right)^{\beta-2} \left[ 2(\beta-1)(x - y)^2  
 + \left( \alpha + \|x-y \|^2 \right) \right]
}
The code implementing these formulas is in \verb|stein/notes/learning/mvn/uvn.py|.

\newpage

### Plotting the Greedy Objective

Using the greedy optimisation method (Section 3.1 in Chen Et Al) we set the initial Stein Point to be $x_1=\mu$ ie the distribution mode.
Eventually this will require a global optimisation call but in this simple example lets just assign it.  For each 
subsequent $x_n$ for $n>1$ take $x_n$ to be the value which minimises the objective
\eq{ \mrt{argmin}_z \; \; \left\{ \frac{1}{2}k_0(z,z) + \sum_{j=1}^{n-1}k_0(x_j, z) \right\} }

Since we are in one dimension we can do an exhaustive grid search, and plot the objective function at each iteration 
to get a feel for the minimisation surface.  The script that creates this plot is
\verb|stein/notes/learning/mvn/normal1-1.py|.  Using the log gradient and SRK defined in \verb|uvn.py|
it is simple to write the greedy objective
\plist{}
```Python
def objective(z, ksr, points):
    # evaluate the greedy objective for this point z \in R
    # where points is a sequence of previously computed points
    # use ksr(x,y) to evaluate ksr between any two points
    out = 0.5*ksr(z, z)
    if points is not None:
        for xi in points:
            out += ksr(xi, z)
    return out 
``` 

In the plot below each curve plots the greedy objective for point $x_j$ 
where $j=2,3,\hdots, n$.  No objective is plotted for the initial point but the point itself is plotted in red.
Subsequent Stein Points are plotted in black.  This example would appear to be the *simplest possible example* of the 
computation of Stein Points and yet the scale of the optimisation problem is already apparent.

![Greedy Objective for a Univariate Normal Distribution](normal11.png)

\newpage

### Plotting Empirical CDFs and Comparing the numpy 

In \verb|/stein/notes/learning/mvn/normal1-2.py| we look at the empirical cdfs.  Running the script produces
output 
\blist{}
```bash
 **** Found Stein Point 2 at z=1.976 after 0:00:00.179566 (H:M:S) **** 
 **** Found Stein Point 3 at z=4.236 after 0:00:00.425367 (H:M:S) **** 
 **** Found Stein Point 4 at z=5.248 after 0:00:00.751985 (H:M:S) **** 
 **** Found Stein Point 5 at z=0.968 after 0:00:01.159569 (H:M:S) **** 
 ...
 ...
 **** Found Stein Point 18 at z=4.420 after 0:00:13.798995 (H:M:S) **** 
 **** Found Stein Point 19 at z=3.304 after 0:00:15.341222 (H:M:S) **** 
 **** Found Stein Point 20 at z=1.568 after 0:00:16.960261 (H:M:S) **** 
 Computation complete.  Found:
  -- 20 Stein Points.
  -- Wall Time: 0:00:16.960291 (H:M:S)
```
for the small $n=20$ case.  Using these samples we compare

* Stein Points
* Two samples of normal rvs generated from \verb|numpy.random.normal|
* True normal CDF

to gain insight into the quality of the convergence of the Stein Point approximation.  I do not know the 
method \verb|numpy| uses to generate normal rvs but I presume it is high quality.  No rigorous testing is
performed but from a cursory look the Stein Points appear to be a highly efficient approximation compared
to the normal rvs generated from \verb|numpy|.

![Empirical CDF Comparison for n=20](normal12.png)

The effectiveness of the Stein Points is even more clear after repeating the experiment with $n=50$. 
\blist{}
```bash
 **** Found Stein Point 2 at z=1.978 after 0:00:00.944684 (H:M:S) **** 
 **** Found Stein Point 3 at z=4.234 after 0:00:02.308628 (H:M:S) **** 
 ...
 ...
 **** Found Stein Point 48 at z=3.438 after 0:09:26.052780 (H:M:S) **** 
 **** Found Stein Point 49 at z=1.642 after 0:09:50.222214 (H:M:S) **** 
 **** Found Stein Point 50 at z=4.417 after 0:10:08.421444 (H:M:S) **** 
Computation complete.  Found:
 -- 50 Stein Points.
 -- Wall Time: 0:10:08.421573 (H:M:S)
```

![Empirical CDF Comparison for n=50](normal13.png)

I note that the greedy objective appears to get progressively slower as we compute more points.  This makes sense
as the objective requires repeated re-calculation of $k_0$ against previous points.  This appears to be 
a possible optimisation strategy for the future.  For example, when we minimize the objective for point $x_2$
we evaluate
\eq{ \frac{1}{2}k_0(z,z) + k_0(x_1, z) \nonumber }
but for each subsequent $x_i$ the expression $k_0(x_1,z)$ will be evaluated again.  It may be faster to 
evaluate $k_0(x_1,z)$ at more points than required by the $2^{\mrt{nd}}$ iteration of the minimiser, but 
save and tabulate them to be interpolated against later.  Further, say $x_{10} \approx x_1$, then can we 
interpolate from $k(x_1,z)$ to $k(x_{10},z)$?

\newpage

### Testing a scipy Global Optimiser

The \verb|scipy| package provides a suite of global optimizers in the \verb|scipy.optimize| namespace.  An
example is the \verb|dual_annealing| routine which is used below.  This routine appears to return essentially
the identical points to the direct search method above, and produces output for $n=20$
\blist{}
```bash
 **** Found Stein Point 2 at z=1.976 after 0:00:00.185857 (H:M:S) **** 
 **** Found Stein Point 3 at z=4.234 after 0:00:00.406755 (H:M:S) **** 
 ...
 ...
 **** Found Stein Point 18 at z=4.414 after 0:00:10.198553 (H:M:S) **** 
 **** Found Stein Point 19 at z=3.300 after 0:00:11.266055 (H:M:S) **** 
 **** Found Stein Point 20 at z=1.562 after 0:00:12.393725 (H:M:S) **** 
Computation complete.  Found:
 -- 20 Stein Points.
 -- Wall Time: 0:00:12.393767 (H:M:S)
```
The resulting ECDF is identical to that produced in the direct search in the line above.
![Empirical CDF Comparison for n=20 and Global Minimiser](normal14.png)

\newpage

## Example 2.1: Bivariate Normal Distribution

Here we continue on from Example 1.1 and compute Stein Points for a multivariate normal in $\mbb{R}^2$.
This is an interesting example for investigating more general formulas for the SRK and log gradients, as
well as how \verb|scipy| routines cope with the nasty optimisation problem in $d=2$.  The SRK is 
\eq{ 
    k_0(\mb{x},\mb{y}) = \sum_{i=1}^{d=2} \left( \frac{\partial^2 k }{\partial x_i \partial y_i} + \pwrt{k}{x_i} \pwrt{g}{y_i} + \pwrt{k}{y_i} \pwrt{g}{x_i} + k (\mb{x}, \mb{y}) \pwrt{g}{x_i} \pwrt{g}{y_i} \right) 
}
the kernel derivatives are
\eqa{
 \pwrt{k}{x_i} = & 2 \beta (x_i-y_i)  \left( \alpha + \|\mb{x}-\mb{y} \|^2 \right) ^{\beta-1}  \nonumber \\
 \pwrt{k}{y_i} = & -2 \beta (x_i-y_i) \left( \alpha + \|\mb{x}-\mb{y} \|^2 \right)^{\beta-1} \nonumber \\
 \frac{\partial^2 k }{\partial x_i \partial y_i} = & -2 \beta \left( \alpha + \|\mb{x}-\mb{y} \|^2 \right)^{\beta-2} \left[ 2(\beta-1)(x_i - y_i)^2  + \left( \alpha + \|\mb{x}-\mb{y} \|^2 \right) \right]
}
and finally the log gradients are 
\eq{ \mb{\nabla} \textrm{log}\,p(\mb{z}) = -\transpose{(\mb{z}-\mu)} \, \inverse{\Sigma} }
These formulas are implemented in \verb|stein/notes/learning/mvn/mvn.py| and are analgous to those computed in 
\verb|uvn.py| discussed previously.  The output is vectorised but the component wise nature of the calculations is clear.

### The d=2 Greedy Objective Function

As one expects the objective function in two dimensions is complicated.  In this example I take 
\eq{ \mu = \transpose{ \left( -0.5, 0.5 \right) } } and correlation 
\eq{ 
    \Sigma = \mat{ 1.0 & 0.5 \\ 0.5 & 2.0 }
}
As per the one dimension examples I choose the initial Stein Point to be the mode $\mb{x}_1 = \mu$, and use
the \verb|scipy| \verb|scipy.optimize.dual_annealing| minimiser which appears to also work well in two dimension.
The following plot shows contours of the objective for the first eight Stein Points (after the initial choice) and
the points chosen by the minimiser are the red crosses.  These points appear plausible (without doing rigorous 
checking).

![Countours of the Greedy Objective for the Bivariate Normal Target](normal16.png)

\newpage

### The First n=20 Stein Points for the Bivariate Normal

Using the same bivariate normal example from above, I compute the first 20 Stein Points.  This is
found in the script \verb|normal1-5.py|, which produces output like
\blist{}
```bash
 *** Scipy found point (0.01. 1.72) after 0:00:03.419930 (H:M:S)
 *** Scipy found point (-1.01. -0.72) after 0:00:04.777972 (H:M:S)
 *** Scipy found point (-1.38. 0.86) after 0:00:05.857179 (H:M:S)
 *** Scipy found point (0.43. 0.11) after 0:00:06.803905 (H:M:S)
 *** Scipy found point (-0.83. 1.87) after 0:00:08.909907 (H:M:S)
 *** Scipy found point (0.75. 1.20) after 0:00:09.986271 (H:M:S)
 ...
 ...
 ```

 A contour plot of the target pdf and $n=20$ Stein Points is plotted below.

![Countours of the Target PDF and Computed Stein Points](normal17.png)

\newpage 

## Example 2.2 Computing the KSD and Starting a Python Stein Point Library

The last example in this set is to outline the start of a simple Stein Point library
in python.  This is found at \verb|stein/code/python/lib| and currently consists
of four groups of files:

* \verb|kernel.py| and \verb|kernel_base.py| for defining general kernal functions $k(\mb{x},\mb{y})$
* \verb|target.py| and \verb|target_base.py| for defining general targets $p(\mb{x})$
* \verb|stein.py| where generic KSD, SRK and greedy objectives are defined
* \verb|util.py| where a couple of utility routines I seem to use regularly are kept.

The script \verb|stein/code/python/example/mvn1.py| demonstrates how to use the library,
and provides an example of using kernals, targets and \verb|scipy| global optimization routines
to compute Stein Points. It is run (for $n=25$ points for example) like
\blist{}
```bash
python mvn1.py --n 25
```
where \verb|PYTHONPATH=<path_to_stein>/stein/code/python/lib| should be set for the imports required.
This script produces output like
\blist{}
```bash
$ python mvn1.py --n 25
Computing first 25 Stein Points for MVN Example 1.
 ** Stein Point (1) found in 0:00:00.755472 (H:M:S) at x: [-0.50001774  0.49998333]
 ** Stein Point (2) found in 0:00:02.494034 (H:M:S) at x: [0.00604887 1.72167402]
 ** Stein Point (3) found in 0:00:03.273162 (H:M:S) at x: [-1.00729827 -0.72474337]
 ** Stein Point (4) found in 0:00:04.227296 (H:M:S) at x: [0.37904274 0.13130274]
 ** Stein Point (5) found in 0:00:04.875321 (H:M:S) at x: [-1.43584947  0.88528364]
 ** Stein Point (6) found in 0:00:03.568377 (H:M:S) at x: [0.7054779  1.23707363]
 ** Stein Point (7) found in 0:00:01.400034 (H:M:S) at x: [-0.88527567  1.87834818]
 ....
```
and writes a plot of the target pdf, Stein Points, and KSD which is shown below:

![Stein Points and KSD from Small Python Library](normal18.png)

