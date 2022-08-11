
\tableofcontents

# An Idiot's Guide to Stein Points

These are my (ie the idiot) introduction to Stein Points, and the two associated computational
problems I will investigate.  These will be fleshed out over time as I come to understand more
theory, and will include a number of examples that I found helpful on the way. Essentially 
everything in here is taken from Chen Et Al (2018). 

## Stein Point Concepts 

### Motivation 

A common statistical problem is to approximate a probability distribution
$P(\mb{x}), \, \mb{x}\in\mbb{R}^d$ with a sequence of points $\left\{\mb{x}_i \right\}_{i=1}^{n}$.
Such a sequence of points is an *empirical distribution* and is useful because it
permits efficient calculation of various quantities (moments, quantiles etc).  In 
addition they have well known and usefull theoretical properties.  The critical goal
is to produce a sequence which achieves *convergence* in the sense that
\eq{ \frac{1}{n} \sum_{i=1}^{n}h(\mb{x}_i) \; \rightarrow \int h dP }  
when $n\rightarrow\infty$, where we are subject to various technical conditions in the above.

This projects concerns a particular sequence $\left\{\mb{x}_i \right\}_{i=1}^{n}$ of points called 
**Stein Points**, which are chosen carefully to provide a *best approximation* to
$P(\mb{x})$, where we will define exactly what we mean by *best* in a later section.  There
are many techniques for generating convergent sequences of points from a given distribution, 
but most require all normalisation constants to be known.  This requirement is commonly 
violated in practice, where we know the *functional form* of the distribution but may not be
able to compute multiplicative constants.  MCMC techniques can circumvent this problem 
by defining transition probabilities as ratios, so that these unknown constants cancel.  
Stein Points also do not require multiplicative constants to be known, so exist in a smaller
set of plausible methods for this very common problem.

### Definitions

A *discrepency* quantifies how well the points $\left\{\mb{x}_i \right\}$ cover the domain
of the random variable $\mb{x}$ with respect to the distribution $p(\mb{x})$.  The set of 
points which minimise a particular type of discrepency (a *Kernel Stein Discrepency*) with
respect to the target $p(\mb{x})$ are referred to as *Stein Points*.  The Stein Point methodology 
exists in the more general framework of *Reproducing Kernel Hilbert Spaces* (RKHS), which is a 
popular framework because analytic formulas for discrepencies are available.  However, these 
general results suffer from the usual challenge that the target distribution's normalisation
constant is required.  It can be shows that particular choices of *Reproducing Kernels* (those
chosen from a *Stein Set*) both simplify the maths in the more general RHKS and do not require knowledge
of the normalisation constants.  Such discrepencies are called *Kernel Stein Discrepencies*
(KSD), and involve particular kernels referred to as *Stein Reproducing Kernels*.  In particular
the *Kernel Stein Discrepency* (KSD) is defined
\eq{ \mc{D} = \sqrt{ \frac{1}{n} \sum_i \sum_j k_0(\mb{x}_i, \mb{y}_j)} }
where $k_0(\mb{x},\mb{y})$ is the *Stein Reproducing Kernel* and is defined in a subsequent 
section.  

### Process 

The process of approximating a target $p(\mb{x})$ using Stein Points proceeds by

1. Choosing a particular kernel $k(\mb{x},\mb{y})$, which measures distances between points.  This choice
is subject to various technical conditions.
2. Choose a *Stein Operator* $\tau$, which should be chosen cleverly in conjunction with the kernel above.
3. Compute the *Stein Reproducing Kernel* (SRK) $k_0(\mb{x},\mb{y})$ by solving the integral equation 
$\int \tau[k]dP=0$
4. Use the *Stein Reproducing Kernel* (SRK) to compute the *Kernel Stein Discrepency*
\eq{ \boxed{ \mc{D} = \sqrt{ \frac{1}{n} \sum_i \sum_j k_0(\mb{x}_i, \mb{y}_j)} } \nonumber }
5. Choose an optimisation strategy to minimise the KSD above

Note that *Stein Operators* $\tau$ are chosen to

* Produce discrepencies that do not require normalisation constants
* Simplify formulas for discrepencies that are required when using the more general RKHS framework
* Combine with the choice of kernal $k(\mb{x},\mb{y})$ to guarantee that as the 
$\mrt{KSD} \rightarrow 0$ the empirical distribution produced by our point sequence is convergent
to the target $p(\mb{x})$.

After writing $k=k(\mb{x},\mb{y})$ to save notation, the particular choice of $\tau$ considered here is 
the *Langevin Stein Operator*
\eq{ \tau \left[ k \right] = \mb{\nabla} \cdot \left( p k \right) / p}
which generates a Stein Reproducing Kernel (SRK)
\eq{ \boxed{
    k_0 = \nabla_{\mb{x}} \cdot \nabla_{\mb{y}} \, k + \nabla_{\mb{x}} \, k \cdot \nabla_{\mb{y}} \, \mrt{log} \, p(\mb{y}) + \nabla_{\mb{y}} \, k \cdot \nabla_{\mb{x}} \, \mrt{log} \, p(\mb{x}) + k \nabla_{\mb{x}} \, \mrt{log} \, p(\mb{x}) \cdot \nabla_{\mb{y}} \, \mrt{log} \, p(\mb{y}) }
}

The \texttt{div} $\cdot$ \texttt{grad} structure of the SRK results in a computation where partial derivatives are evaluated at each of the $d$ components of $\mb{x}$ before being summed, that is, $k_0(\mb{x},\mb{y})$ can be written
\eq{ \boxed{
    k_0(\mb{x},\mb{y}) = \sum_{i=1}^{d} \left( \frac{\partial^2 k }{\partial x_i \partial y_i} + \pwrt{k}{x_i} \pwrt{g}{y_i} + \pwrt{k}{y_i} \pwrt{g}{x_i} + k (\mb{x}, \mb{y}) \pwrt{g}{x_i} \pwrt{g}{y_i} \right) }
}
where I have substituted $g=\textrm{log} \, p$ to again save notation.
