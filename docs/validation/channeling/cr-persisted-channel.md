# `cr-persisted-channel`: the persisted ensemble source

Implementation-context specification and independent verification for #507; anchored on the stated slab model and regression references.
The source equations are the expected-value estimator of
[cr-yield-assembly](cr-yield-assembly.md), Beer–Lambert exit attenuation of
`self-absorption`, and the one-way zone of [ADR-0018](../../adr/0018-quantum-channeling-zone.md).

The mean beam direction is $\mathbf b$ and the plane normal in slab coordinates
is $\mathbf n_p$. Construct a right-handed beam/plane frame

$$
\mathbf e_z=\mathbf b,\quad
\mathbf e_x=\frac{\mathbf n_p-(\mathbf n_p\cdot\mathbf b)\mathbf b}
 {|\mathbf n_p-(\mathbf n_p\cdot\mathbf b)\mathbf b|},\quad
\mathbf e_y=\mathbf e_z\times\mathbf e_x.
$$

Photon directions and the slab inward normal are transformed by the transpose
of this frame. Let the slab normal in the source frame be $\mathbf m$,
with $m_z>0$, and the normal thickness be $T$. The mean path is $L=T/m_z$.
At electron path $s$, its normal depth is $sm_z$; for an exit-side photon
direction $\hat{\mathbf q}$, its remaining material path is

$$
\ell(s,\hat{\mathbf q})=\frac{T-sm_z}{\hat{\mathbf q}\cdot\mathbf m}.
$$

The existing stack attenuation owner handles the analogous entrance-side ray.
Layer depths remain slab-normal depths. The Doppler and polarization angles
remain relative to the mean beam and the transverse plane normal, rather than
the slab normal. At zero tilt, this reduces to the diagnostic API convention.
As in the diagnostic API, the absorption geometry is a laterally infinite
slab. Finite side-face escape of photons is omitted. Electron face limits
below do not turn the photon absorption model into a finite-prism model;
side-escaping photon configurations remain outside its validity domain.

For each incident primary $j$, let $l_j$ be the path to its first physical face
or line-scoring energy cutoff, with zero for a footprint miss. Under the same
mean-field entrance approximation as the zone solve, the expected source is

$$
Y_c(\hat{\mathbf q})=\int_0^L ds\,
\left\langle d_c^\dagger d_c\right\rangle_{\rho(s)}
\exp[-\tau(sm_z,\hat{\mathbf q})]\,
\frac1N\sum_{j=1}^{N}\mathbf1_{s<l_j}.
$$

The density already contains dechanneling loss. Sampled fates must not weight
the source again. $N$ is the line-scoring incident population; the separate
bremsstrahlung trial count must not change its normalization. The implementation
inserts face/cutoff endpoints into the sampled depth history, linearly
interpolates the rates, and weights each trapezoid interval by the fraction of
entrance paths extending through that interval. If every limit is $L$, this
reduces to the unweighted diagnostic integral. Zero limits give zero source.

The resulting spectrum occupies only `spec_channeling`, in photons/eV/sr per
incident electron on `E_grid`. It includes the diagnostic bound lines and
tracked free-to-bound capture. `spec`, `spec_characteristic` and `brem` keep
their existing meanings. Common component owners include CR once in totals;
off omits it. `CHANNELING_MODEL` is
`planar-density-matrix-emission-v4` after #536 registered the provisional
MoS₂ directional valence loss (v3: #523 diamond electronic-kernel
input change), and applies only to enabled channeling identities.

Assumptions: the P3 diagnostic source remains at constant mean energy, while
transport includes continuous stopping. The profile, kernel, margin, mean
incidence and resolved normal divergence are shared; emission has its own
sampled density history. In-plane angular diffusion accumulated in the zone,
detector response convolution, and the source-audited measured-width gaps
remain omitted. Entrance-path limits are averaged independently of the
ensemble density and angular quadrature, so phase-space correlations beyond
the shared mean-field approximation are not reconstructed. These restrictions
are not claims of measured linewidth agreement.

Anchors: `tests/montecarlo/channeling/test_channeling_producer.py` (analytic integration,
frame covariance, public producer and real checkpoint round trip),
`test_tilted_surface_absorbs_over_the_physical_normal_depth` in the emission
tests (independent Beer–Lambert path), and the existing `test_spec_channeling`
storage/totals tests.

## Independent derivation (fresh context, 2026-10-16)

This section was written from the ledger, source write-ups, ADR-0018 and
signatures/docstrings before inspecting the implementation bodies.

Let $\hat{\mathbf b}$ be the normalized mean incoming beam and
$\hat{\mathbf p}$ the plane normal, both expressed in slab coordinates.
The transverse projection
$\mathbf v=\hat{\mathbf p}-(\hat{\mathbf p}\cdot\hat{\mathbf b})\hat{\mathbf b}$
defines $\hat{\mathbf x}=\mathbf v/\|\mathbf v\|$,
$\hat{\mathbf z}=\hat{\mathbf b}$ and
$\hat{\mathbf y}=\hat{\mathbf z}\times\hat{\mathbf x}$.
Thus $R=(\hat{\mathbf x},\hat{\mathbf y},\hat{\mathbf z})$ satisfies
$R^TR=I$ and $\det R=1$; source-frame vectors are $R^T\mathbf a$.
Simultaneous rigid rotation of beam, plane, photon and slab normal leaves
all source-frame scalar products unchanged. A beam parallel to the plane
normal leaves the transverse axis arbitrary; a physical result must not
depend on that choice in the unchanneled-entry limit.

For inward slab normal $\hat{\mathbf u}$, define
$\mathbf m=R^T\hat{\mathbf u}$ and $c=m_z>0$.
An electron mean path $s$ has normal depth $d=sc$ and traverses the
normal thickness $T$ in length $L=T/c$. For photon source-frame direction
$\hat{\mathbf q}$, let $a=\hat{\mathbf q}\cdot\mathbf m$.
The homogeneous slab escape length is

$$
\ell(s,\hat{\mathbf q})=
\begin{cases}
(T-sc)/a,&a>0,\\
sc/(-a),&a<0.
\end{cases}
$$

Beer--Lambert gives $\exp[-\mu(E_c)\ell]$. More generally each layer
contributes its nonnegative normal overlap divided by $|a|$.
The Doppler factor still uses the photon angle to the beam (and its
in-plane angular quadrature); replacing that angle by the photon angle
to the surface would be incorrect. The surface cosine belongs only in
the attenuation geometry. For a finite footprint the photon escape
length must instead stop at its first physical face if that geometry is
claimed; censoring the electron source does not by itself supply that
photon-side correction.

Write $r_c(s)=\operatorname{Tr}[d_c^\dagger d_c\rho(s)]$, with units
inverse angstrom. The unnormalized surviving ensemble density already
accounts for the loss of radiating states to the dechanneling sink.
For geometric/cutoff entrance-path limits $l_j$, linearity of the
expected-value source gives

$$
Y_c(\hat{\mathbf q})=
\frac1N\sum_{j=1}^{N}\int_0^{\min(L,l_j)}
r_c(s)\exp[-\tau(sc,E_c,\hat{\mathbf q})],ds
=\int_0^L r_c(s)\exp[-\tau],W(s),ds,
\qquad W(s)=\frac1N\sum_j\mathbf1_{s<l_j}.
$$

Here $N$ includes footprint misses, which have $l_j=0$. It is the incident
line-scoring population, independent of the bremsstrahlung sample size.
Multiplying by sampled bound/free fates or by another dechanneling
survival probability would count the same sink loss twice. The factorized
form assumes the geometry/cutoff ensemble can be averaged independently
of the common mean-energy density; correlated energy, angle and footprint
dependence is outside that approximation.

Insert all clipped limits into the sampled history. On an interval
containing no limit, $W$ is constant, so the trapezoid is

$$
\Delta Y_{c,k}=W_k\frac{f_c(s_k)+f_c(s_{k+1})}{2}
(s_{k+1}-s_k),\qquad f_c=r_c\exp(-\tau).
$$

Endpoint values of the step function have zero measure; midpoint survival
is unambiguous. Interpolation is exact for linearly sampled unabsorbed
rates; with absorption the remaining trapezoid error must converge as
the sampling interval shrinks. Integrating the diagnostic angular and
unit-area spectral mode kernels gives photons/eV/sr/incident electron.
The angular average over a detector aperture remains per steradian.

Cheap filters: rate times path is dimensionless; optical depth is
dimensionless and nonnegative. With no attenuation, all $l_j=L$ recover
the diagnostic source, all zero limits give zero, and duplicating every
incident path preserves the source. For constant rate $r$, no absorption,
$Y=rN^{-1}\sum_j\min(L,l_j)$. With constant rate and one full path,
exit-side absorption gives
$Y=r[1-\exp(-\mu T/a)]a/(\mu c)$ and tends to $rL$ as
$\mu\to0$. These filters pass for the stated estimator. Constant
emission energy despite transport stopping is an explicit approximation;
it does not certify changing-energy line positions or widths.

## Independent implementation comparison

- `source_frame` normalizes the beam, projects the plane normal transverse
  to it, and returns the three columns derived above. Multiplying a row
  photon vector by that frame is equivalent to applying $R^T$ to a column
  vector. `frame[2]` is exactly $R^T(0,0,1)$, the slab normal in the source
  frame. The right-handedness and simultaneous-rotation anchor pass.
- `case_channeling_spectrum` uses the leading `Ne_lines` entrance positions,
  directions and energies. The averaging denominator is the number of
  those path limits, including zero misses; no `Ne_brem` factor appears.
  It computes face limits from the original entrance trajectories and caps
  them at $\max(E_j-E_{\rm cut},0)/S(E_0)$. It never reads the sampled
  quantum fates or truncates at a sampled dechanneling depth. These terms
  match the declared mean-field estimator.
- `_History.censored` inserts clipped limits and linearly interpolates the
  rates. Its upper-endpoint `searchsorted(..., side="left")` counts paths
  whose limit equals the interval's upper endpoint as surviving that
  interval, which is the correct integral convention. `_History.integrate`
  multiplies each trapezoid by precisely that incident-path fraction.
  The analytic anchor with $r(s)=2+3s$ and limits $(0,2,7,10)$ gives
  $Y=66.875$ per electron. All-zero limits give exactly zero.
- `channeling_spectrum` constructs normal thickness $Lc$, transforms
  emission depths to $sc$, and calls `_stack_tau` with $a$. This matches
  the derived tilted infinite-slab optical depth on either exit side.
  The separately computed electron-frame angles remain the arguments of
  the Doppler and dipole kernels. The tilted-surface anchor independently
  assembles the Beer--Lambert weights and passes to relative $10^{-10}$.
- The producer returns the diagnostic spectrum directly; the runner stores
  it only as `spec_channeling`. The actual producer/checkpoint round trip
  and result-family tests verify that common totals add it once. The
  unchanged incident channels retain their separate stored arrays.

### Boundary of the attenuation result

The current photon attenuation call has no emission position, width or
height. Finite footprints affect the electron-path weight only. This
verifies tilted **infinite slab** photon escape, not the first-face photon
escape supported by the finite-prism branch of `self-absorption`.

For a concrete counterexample to finite-prism equivalence, take a normal
beam, $T=20000$ angstrom, width $100$ angstrom, an electron at the centre
of the entrance face, and photon direction $(0.6,0,0.8)$. The first
side-face path is $50/0.6=83.3333$ angstrom whereas the infinite-slab
path is $20000/0.8=25000$ angstrom. At an illustrative passive
$\mu=10^{-4}$ inverse angstrom, their respective survivals are
$0.991701$ and $0.0820850$. The exact divergent term is omission of
the minimum with the lateral photon face distances in $\ell$; it is
not a missing tilt cosine. The existing 5 mm footprint smoke test
cannot expose this boundary. Finite-prism correctness needs position-
conditioned escape. The task owner adjudicated infinite-slab photon
attenuation as the intended P3 model, with finite entrance-path censoring
applying to electrons only. This counterexample is a validity limit of
that explicitly restricted model, rather than a discrepancy with its
equation.

### Verification evidence

The focused command covering `test_channeling_producer.py`,
`test_channeling_emission.py`, `test_spec_channeling.py` and
`tests/dev/test_docs.py` passed: **52 passed in 9.28 s** (default slow
exclusions). The analytic integration, frame covariance, tilted escape,
storage and total anchors are suitable for the mathematical estimator
within its stated approximation. Additional producer anchors should pin
unequal line/bremsstrahlung counts, footprint misses, side exits and
energy cutoffs; those construction branches are currently inspected
symbolically rather than independently pinned by those tests.

### Independent verdict

- **Claim**: `cr-persisted-channel` —
  `montecarlo/runner/channeling_emission.py::{source_frame,case_channeling_spectrum}`;
  `montecarlo/channeling/emission.py::{_History.integrate,_History.censored,channeling_spectrum}`
  — `cr-yield-assembly`, Beer--Lambert and ADR-0018 P5.
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass` for
  the mean-field, constant-energy source and infinite-slab photon escape.
- **Re-derivation**: `matches` — no divergent factor, sign or convention
  within that model. Finite-prism photon escape is explicitly excluded.
- **Verdict**: `rederived`.
- **Write-up**: `docs/validation/channeling/cr-persisted-channel.md`.
- **Suggested ledger change**: change `filtered` to `rederived`; record
  the infinite-slab photon attenuation limit and completed independent
  comparison. The owner may then use `anchored` with the listed green
  analytic/frame/tilt/storage anchors. Human sign-off remains separate.

## Independent fixed-grid integration refinement

This derivation precedes inspection of the revised `_History` bodies.
It replaces insertion of every distinct incident-path endpoint with
integration of the original sampled, piecewise-linear integrand. The
underlying expected source and its incident-electron denominator are
unchanged.

On one original interval $[a,b]$, write $h=b-a>0$ and let
$f_a=f(a)$, $f_b=f(b)$. The integrand includes attenuation when present.
For incident path limit $l_j$, define
$u_j=\operatorname{clip}(l_j-a,0,h)$. Linear interpolation gives

$$
f(a+x)=f_a\left(1-\frac{x}{h}\right)+f_b\frac{x}{h},
\qquad 0\le x\le h.
$$

Its exact truncated integral is

$$
\int_0^{u_j}f(a+x)\,dx
=f_a\left(u_j-\frac{u_j^2}{2h}\right)
+f_b\frac{u_j^2}{2h}.
$$

Averaging all $N$ incident paths therefore yields fixed nodal
coefficients

$$
A=\frac1N\sum_j\left(u_j-\frac{u_j^2}{2h}\right),
\qquad B=\frac1N\sum_j\frac{u_j^2}{2h},
\qquad \Delta Y=A f_a+B f_b.
$$

These coefficients have units of length. They are nonnegative and obey
$A+B=N^{-1}\sum_j u_j$. A full surviving population gives
$A=B=h/2$, the ordinary trapezoid. Zero surviving path gives $A=B=0$;
a fraction $p$ of complete paths and otherwise misses gives
$A=B=ph/2$. A constant integrand integrates to its value times the
mean surviving length; an affine integrand integrates exactly to the
mean of its individual truncated antiderivatives. Endpoint equality
introduces no extra half weight.

Each original interval needs two scalar coefficients, independent of
the number of distinct limits. For $n$ sampled depths and $C$ lines,
the rate history remains $O(nC)$; retaining an $O(N)$ limit vector or
sorted cumulative moments does not create an $O(NC)$ rate array.
Attenuation must be evaluated on the original nodes before this linear
integrand approximation is integrated. In general this is not identical
at finite step size to inserting limits, linearly interpolating the
rates, reevaluating the exponential attenuation there, and applying a
new trapezoid. Both converge to the same expected path integral as the
original sampling interval tends to zero; this refinement changes the
quadrature approximation rather than the source equation.
