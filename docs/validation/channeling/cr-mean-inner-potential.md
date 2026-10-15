# `cr-mean-inner-potential` re-derivation

Claim: `materials/channeling/entry.py::mean_inner_potential_eV`. Source: Ibers, Acta Cryst. 11, 178 (1958); Spence & Zuo, *Electron Microdiffraction* (1992).

## Independent derivation (before reading the body)

A neutral atom with nuclear charge $Ze$ and electron density $\rho(r)$ (normalized to $Z$) has electrostatic potential $\phi(q)=4\pi e\,[Z-f_0(q)]/q^2$ (Gaussian units), because $f_0(q)=\int\rho\,e^{i\mathbf q\cdot\mathbf r}\,d^3r$ and the Poisson source is $Ze\delta^3(\mathbf r)-e\rho$. The crystal-averaged potential is the $g\to0$ Fourier coefficient of the sum over the $N$ basis atoms divided by the cell volume $V$:

$$
\bar\phi=\frac1V\sum_j\lim_{q\to0}\frac{4\pi e\,[Z_j-f_{0,j}(q)]}{q^2}.
$$

Expanding $f_0(q)=Z-\tfrac16 Z\langle r^2\rangle q^2+O(q^4)$ gives

$$
\bar\phi=\frac{4\pi e}{V}\sum_j\frac{Z_j\langle r^2\rangle_j}{6},
\qquad
U_0=-e\bar\phi=-\frac{4\pi e^2}{V}\sum_j\frac{Z_j\langle r^2\rangle_j}{6}.
$$

With $e^2=14.3996\ \mathrm{eV\,\AA}$, $\langle r^2\rangle$ in $\mathrm{\AA^2}$ and $V$ in $\mathrm{\AA^3}$, $U_0$ is in eV. $U_0<0$ (attractive for electrons), so the kinetic energy inside is $T-U_0>T$.

Waasmaier–Kirfel: $f_0(s)=c+\sum_ia_ie^{-b_is^2}$ with $s=\sin\theta/\lambda=g/4\pi$ (xraydb's $q$). Then $Z-f_0=\sum_ia_ib_i\,s^2+O(s^4)$, so

$$
\frac{Z\langle r^2\rangle}{6}=\frac{\sum_ia_ib_i}{16\pi^2}\quad[\mathrm{\AA^2}].
$$

Filters: units eV pass; sign negative pass; $\langle r^2\rangle\to0$ gives $U_0\to0$ (point charge cancelled). Note the WK fit sums $c+\sum a_i$ to $13.9989$ for Si, not exactly $14$, so extrapolating $(Z-f_0)/g^2$ with the integer $Z$ would diverge at $g\to0$; the slope must be taken from differences (or from $\sum a_ib_i$).

## Independent numeric check

My own computation from the WK coefficients read from the xraydb table, with $a=5.4310$ (Si), $3.5668$ (C) $\mathrm{\AA}$, $N=8$:

| Crystal | $\langle r^2\rangle$ [$\mathrm{\AA^2}$] | $U_0$ mine [eV] | $U_0$ code [eV] |
| --- | --- | --- | --- |
| Si | 0.6559 | $-13.830$ | $-13.799$ |
| C (diamond) | 0.6528 | $-20.816$ | $-20.790$ |

Agreement to 0.22% (Si) and 0.12% (C). The residual is the two-node finite-difference error: with $f=Z-sg^2+tg^4$, the node slope is $s-t(g_1^2+g_2^2)$, and for Si at $g=0.05,0.1\ \mathrm{\AA^{-1}}$ it is $1.5269$ against the exact $1.5304\ \mathrm{\AA^2}$, a 2.3e-3 relative error. This is larger than the "below 1e-4" quoted in the docstring and ledger, but it is irrelevant at the required accuracy (about 0.03 eV of a 13.8 eV shift, refraction only). Plausibility: the IAM value $\approx13.8$ V is 10% above the measured $\approx12.6$ V, as expected from bonding charge redistribution.
