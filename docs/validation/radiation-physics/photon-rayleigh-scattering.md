# Photon Rayleigh scattering

## Independent source-first derivation

Validation: `photon-rayleigh-scattering`. The verifier wrote this section after
reading the ledger, AST-extracted signatures/docstrings and original source,
before reading implementation bodies.

Source: [Geant4 Physics Reference Manual 11.4, Livermore Rayleigh final state](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/elastic/livermore_rayleigh.html),
Hubbell et al. (1979) and Cullen (1995). The data convention is pinned EPDL2025
MF=27 MT502 with lin-lin interpolation.

`sample_rayleigh(element, energy_eV, direction, uniforms)` samples one
conditional collision on a selected atom. With $\mu=\cos\theta$, the
unpolarized isolated-atom law is

$$
p(\mu)\propto(1+\mu^2)F(x,Z)^2,\qquad
x=\lambda^{-1}\sqrt{\frac{1-\mu}{2}},\qquad \lambda=hc/E.
$$

The source's density in angle includes $\sin\theta$; the density in $\mu$
above absorbs that Jacobian. The source's momentum transfer $2E\sin(\theta/2)$
must be converted to the EPDL inverse-angstrom argument $x$, not substituted
directly into its table.

Set $y=x^2$ and $k=E/(hc)$. Then $\mu=1-2y/k^2$,
$|d\mu/dy|=2/k^2$, and $0\leq y\leq k^2$. On each accessible lin-lin
form-factor interval, an exact envelope is

$$
M_i=\max(F_l,F_r)^2,\qquad
w_i=(y_r-y_l)M_i,\qquad
A(y)=\frac{F(\sqrt y)^2}{M_i}\frac{1+\mu(y)^2}{2}.
$$

Choosing interval $i$ with probability $w_i/\sum_jw_j$, then drawing $y$
uniformly inside it and accepting with $A$, produces the required density:
the interval width cancels and the surviving density is proportional to
$F(\sqrt y)^2(1+\mu^2)$. Both factors of $A$ are bounded by one for
nonnegative tabulated $F$. Zero total envelope mass means no event exists.
Clipped intervals must include their interpolated endpoints; constant endpoint
extensions, if used by the data owner, must retain their $y$ widths too.

Uniform azimuth and any transverse orthonormal pair give

$$
\hat{\mathbf n}'=\mu\hat{\mathbf n}
+\sqrt{(1-\mu)(1+\mu)}\,
(\mathbf a\cos\phi+\mathbf b\sin\phi).
$$

Infinite atomic mass gives $E'=E$ and zero recoil energy. Units pass:
$x$ has inverse-length units, $y$ inverse-length-squared units, and normalized
proposal and acceptance probabilities are dimensionless. Limits pass:
constant $F$ yields Thomson with $p(\mu)=3(1+\mu^2)/8$;
zero $F$ refuses an event; forward and backward endpoints have unit directions.
Signs and azimuth symmetry pass. Anomalous amplitudes, crystal coherence,
polarization and finite nuclear recoil are excluded. This conditional law
does not replace the independently tabulated integrated rate.

## Implementation comparison

Body inspection followed the written derivation. The code's `square_width`
equals $(x_r-x_l)(x_r+x_l)=y_r-y_l$, preserving narrow-interval precision.
Its interval weights, uniform $y$ proposal, lin-lin factor interpolation and
acceptance are exactly the derived expressions. There is no missing
$\sin\theta$ factor: it was absorbed by $d\mu$, followed by a constant
$d\mu/dy$. A nonmonotone positive form factor is still bounded by interval
endpoints; no monotonicity premise is required.

`clipped_factor_table` includes the interpolated last partial interval.
Zero-weight intervals are skipped by the strict inverse CDF; zero total mass
raises. `scatter_direction` uses the orthonormal rotation above with a least
aligned Cartesian axis. Energy is returned unchanged with zero electron
energy and no electron direction. Uniforms are explicit and counted, and
rejection-cap or tape exhaustion raises without forced acceptance.

## Independent numerical evidence

On 2026-10-19 the verifier independently unpacked the pinned MF=27 MT502
archive, SHA256
`d36a50d1add5c6f0b3df80a108e83aa8db8a4822445ca192534f6bc0a55bb166`.
It used direct archive interpolation, not production factor/CDF helpers.
The reference density in $x$ was proportional to
$x[1+(1-2x^2/k^2)^2]F(x,Z)^2$, integrated on 100,000 logarithmic points
plus all original accessible knots. Each row used 1,200 events along
incident direction $(2,-3,6)/7$, with the explicit local stream initialized
once at seed 943411 shared across alternating Rayleigh and Compton runs.

| Atom | Energy (eV) | Empirical CDF maximum distance |
|---|---:|---:|
| C | 1 | 0.03317 |
| C | 1,000 | 0.04757 |
| Si | 511,000 | 0.03713 |
| Pb | 100,000,000 | 0.01792 |
| Pb | 100,000,000,000 | 0.03190 |

All distances are below $2.2/\sqrt{1200}=0.06351$; this checks sampling of
the specified data model, not the physical accuracy of the input EPDL factors.
Every sampled photon retained its input energy exactly; maximum direction norm
error was $3.4\times10^{-16}$. Three orientations were separately checked at
$\delta\in\{0,10^{-30},10^{-15},1,2\}$; norm and transverse orthogonality
errors stayed below $5\times10^{-16}$, covering exact forward and backward
rotation and tiny forward angles.
All sampled tables start at zero and extend to
$10^9\,\mathrm{\mathring A}^{-1}$, exceeding the accessible
$8.07\times10^6\,\mathrm{\mathring A}^{-1}$ at the 100 GeV domain endpoint;
there is no unsupported endpoint extrapolation. An archive-wide check of all
200 MF=27 tables passed zero origin, strictly increasing knots, sufficient
upper support, and finite nonnegative factors.

## Verdict

Units, limiting cases, signs and conventions pass. The independent expression
matches the implementation: **rederived**, with no divergent factor or
Jacobian found. Suggested ledger change: set status to `rederived` and replace
the pending-verifier note with this derivation and independent numerical check.
Human sign-off and validation of the input atomic data remain separate claims.
