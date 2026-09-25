# SBETHE optical oscillator quadrature

Validation: `gos-optical-quadrature`. Independent verdict: rederived;
human sign-off remains pending.

## Source and intended quantity

SBETHE `OOS.dat` supplies transferred energy $W$ in eV, optical oscillator
strength density $g(W)$ in eV$^{-1}$, and cumulative strength per molecule.
The table can repeat $W$ at a shell edge, where the density jumps. The
intermediate model replaces each positive-width interval by one oscillator.

For linear $g$ between endpoints $(W_i,g_i)$ and $(W_{i+1},g_{i+1})$, integrate
the area and first moment:

$$
f_i=\frac{W_{i+1}-W_i}{2}(g_i+g_{i+1}),\qquad
\bar W_i=\frac{\int_{W_i}^{W_{i+1}}Wg(W)\,dW}{f_i}.
$$

Both $f_i$ and the cumulative f-sum are dimensionless; $\bar W_i$ is in eV.
When $g_i=g_{i+1}$, $\bar W_i=(W_i+W_{i+1})/2$. A repeated energy has zero
area. This quadrature preserves the optical f-sum, but replacing an interval
by its centroid does not preserve every nonlinear cross-section integral.

Writing $\Delta=W_{i+1}-W_i$ and $x=W-W_i$, the linear interpolant is
$g(W_i+x)=g_i+(g_{i+1}-g_i)x/\Delta$. Direct integration gives

$$
\int_{W_i}^{W_{i+1}}Wg(W)\,dW
=W_i f_i+\frac{\Delta^2}{6}(g_i+2g_{i+1}),\qquad
\bar W_i=W_i+\frac{\Delta(g_i+2g_{i+1})}{3(g_i+g_{i+1})}.
$$

For nonnegative endpoint densities and positive area, the centroid lies
inside the interval. Equal endpoint densities give its midpoint. A repeated
energy has $\Delta=0$ and contributes no oscillator, even if the density
jumps there.

Packaged Si, MoS2 and SiO2 recover the table's electron count and logarithmic
mean excitation energy within 0.1% by trapezoidal integration. This checks
the input distribution, not the finite-momentum GOS approximation.

## Source-to-code comparison

`_oscillator_quadrature` uses the trapezoidal $f_i$ and the first-moment
polynomial above, dividing only where $f_i>0$. Its ordered, nonnegative input
check and zero-width exclusion match the assumptions. The code agrees with the
independent expression; the packaged f-sum test anchors the input integral.
