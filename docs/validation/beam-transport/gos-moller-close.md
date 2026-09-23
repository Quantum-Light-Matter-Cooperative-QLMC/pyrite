# GOS close Møller branch

Validation: `gos-moller-close`. Independent verdict: rederived;
human sign-off remains pending.

## Source and intended quantity

Equations 129 and 133 in the [Geant4 PENELOPE ionisation reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/penelope_ionisation.html)
give the close electron-electron branch. For $0<W\leq E/2$,

$$
\frac{d\sigma_{\mathrm{close}}}{dW}
=\frac{2\pi r_e^2m}{\beta^2W^2}\,F(E,W)
\sum_{W_i\leq W}f_i,
$$

where $m=m_ec^2$ and $F$ is the cited relativistic Møller factor. The
cross-section density has units cm$^2$/eV per molecule. The cumulative
oscillator strength jumps at each optical-bin resonance; integration splits
there rather than smoothing across the jump. The two outgoing electrons are
indistinguishable, so the primary's close transfer cannot exceed $E/2$.
For $W/E\to0$, $F\to1$; below the lowest oscillator, the active sum is zero.

This is an optical-bin approximation. `OOS.dat` gives no shell binding
energies, so this branch does not identify inner-shell vacancies or assign
shell-specific secondary energies.

Writing $y=W/(E-W)$ and $b=E/(E+m)$, the source gives

$$
F(E,W)=1+y^2-y+b^2\left(y+\frac{W^2}{E^2}\right).
$$

Thus $F\to1$ as $W/E\to0$. For $0\leq W/E\leq1/2$, $0\leq y\leq1$ and
$1-y+y^2>0$; the remaining term is nonnegative. Every oscillator with
$W_i\leq W$ contributes its full dimensionless strength to the close
density, which has units cm$^2$/eV per molecule. At $W>E/2$ the close
term vanishes by the indistinguishable-electron primary convention.

## Source-to-code comparison

`_moller_factor` is term-for-term identical to the source factor.
`build_gos_partition` multiplies it by $Af_{\leq W}/W^2$, using the
midpoint of each close bin to select the cumulative active oscillator
strength. It inserts every resonance and the $E/2$ ceiling into the grid,
so the active strength is constant within each open bin. Endpoint densities
are interpolated linearly within a bin; the resulting moments are exact for
that interpolation, while the smooth Møller factor is a numerical
approximation between grid points. No source-to-code term differs.
