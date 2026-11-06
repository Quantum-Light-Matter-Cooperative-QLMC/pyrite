# `cr-axial-depth-budget`: source-relative budget arithmetic

**Status:** task-owner derivation (2026-11-05), awaiting fresh-context verification.
This claim covers how supplied bounds combine and when the guard refuses. It
contains no measured bound and no calibrated depth; those belong to
[#993](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/993).

## Relative source error

Fix one observable and a depth interval $(a,b]$. Let $Y_\mathrm{ref}(z)$ and
$Y_\mathrm{model}(z)$ be photon yields per incident electron. Suppose that with
joint confidence $c$, for every $z\in(a,b]$, $Y_\mathrm{ref}(z)\ge L>0$ and the
model error decomposes into channels with $|e_k(z)|\le B_k$. The triangle inequality
gives

$$
\frac{|Y_\mathrm{model}-Y_\mathrm{ref}|}{Y_\mathrm{ref}}
\le\frac{\sum_k|e_k|}{Y_\mathrm{ref}}
\le\frac{\sum_kB_k}{L}.
$$

The decomposition must be exhaustive for the first inequality to hold. Overlapping
channels only loosen the bound, but a missing channel breaks it. The guard refuses
when the right side exceeds the tolerance.

## Paired probes

For bins $j$, reference means $r_j$ and a perturbed comparison $q_j$, with bounds
$s_j$ on the paired-difference estimate and $t_j$ on its drift across the interval,
each true bin difference satisfies $|\Delta_j|\le|q_j-r_j|+s_j+t_j$. Summing over
bins gives $B=\sum_j(|q_j-r_j|+s_j+t_j)$, which bounds the absolute error of the
integrated observable with no cancellation between bins. With one-sided allowances
$u_j,v_j$ on the reference, $L=\sum_j(r_j-u_j-v_j)$.

## Joint confidence

If interval $i$ holds with probability at least $c_i$, Boole's inequality bounds the
probability that any interval fails by $\sum_i(1-c_i)$, so all hold with probability at
least $1-\sum_i(1-c_i)$, with no independence assumption. Sampling risk $\delta_s$
and cap failure risk $\delta_c$ subtract from the source confidence the same way.

## Terminal-fate bound

Let $N$ independent unit-weight incidents each produce one terminal fate and depth.
For fate class $c$, $F_c(z)$ is the probability of a terminal event of class $c$ at
depth at most $z$, and $\hat F_c$ its empirical estimate. Each $\hat F_c(b)$ is a mean of
$N$ independent Bernoulli variables, so Hoeffding's one-sided bound gives

$$
P[F_c(b)-\hat F_c(b)\ge h]\le e^{-2Nh^2}.
$$

For $m$ interval stops and two classes, $2m$ events are bounded. Setting
$2m\,e^{-2Nh^2}=\delta$ yields

$$
h=\sqrt{\frac{\ln(2m/\delta)}{2N}}.
$$

$F_c$ is nondecreasing, so $F_c(z)\le F_c(b)$ for every $z\in(a,b]$, which makes the bound
uniform over the interval without interpolation. The union bound does not need
independence between depths, so reusing the same incidents at each stop is valid.
Multiplying $\min(1,\hat F_c(b)+h)$ by a per-fate cap $C_c$ bounds the incident-normalized
error when the cap holds for every fate of that class. A retained record must reach the
sampled maximum depth, or an incomplete history would understate $\hat F_c$.

## Limits

Zero channels give a zero bound. No observed events still leave $h\,C_c>0$. Disjoint
intervals with a gap refuse, because no bound covers a depth in the gap. The yield
of a cumulative observable vanishes at the entrance, so a useful first interval needs
$a>0$.
