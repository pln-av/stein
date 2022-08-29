\tableofcontents

# SP-GARCH Model Notes

## Model Specification

The SP-GARCH model is 

\eqa{
    r_t & = \mu + y_t \nonumber \\
    y_t  & = \sigma_t \varepsilon_t \nonumber \\
    \sigma_{t}^{2} & = \omega + g(\varepsilon_{t-1})\sigma^{2}_{t-1}
}

where $\varepsilon \sim \textrm{F}_{\mrt{std}, \nu}$ is the standardised $t$ distribution with $\nu$
degrees of freedon.  Inf $z  sim \textrm{F}_{\mrt{std},\nu}$ then $z$ is related to the regular $t$ distribution
via 

\eq { z = \sqrt{\frac{\nu-2}{\nu}} x }

where $x \sim t_\nu$, the usual $t$ distribution with $\nu$ degrees of freedom.
