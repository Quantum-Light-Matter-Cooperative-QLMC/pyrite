# `line-grid-kinematic-bandwidth` — independent re-derivation

**Claim id:** `line-grid-kinematic-bandwidth`
**Code under review:** `src/pyrite/_cxr_grid_policy.py::kinematic_cxr_stop_eV`
(consumed by `campaign/sweep.py::_automatic_cxr_grid_policy`)
**Anchor:** `tests/energy-grid/test_line_grid_policy.py::test_kinematic_stop_bounds_the_resonance_over_random_directions`
**Verifier:** fresh context; derivation written from the ledger row and the
function signature/docstring before the implementation body was read.

## Claim as ledgered

Automatic case-local line grids put their upper edge at

$$
E_{\max}=\frac{\hbar c\,\beta\,\lvert\mathbf g\rvert_{\max}}{1-\beta},
$$

the direction-maximized form of the PXR/CBS resonance
$E_{\rm res}=\hbar c\,(\boldsymbol\beta\cdot\mathbf g)/(1-\hat{\mathbf n}\cdot\boldsymbol\beta)$,
evaluated at the incident speed over the case's own reflection set, rounded
up to `round_to_eV`.

Signature: `kinematic_cxr_stop_eV(g_magnitudes_invang, energy_keV, *, round_to_eV)`
returning eV. Inputs: reflection magnitudes $\lvert\mathbf g\rvert$ in
$\text{Å}^{-1}$ and electron kinetic energy in keV.

## Independent derivation

### Resonance condition

A charge moving uniformly with velocity $\mathbf v=\beta c\,\hat{\mathbf v}$
through a lattice sees a current with Fourier components
$\propto e^{i(\mathbf g\cdot\mathbf v)t}$. Emission of a photon
$(\omega,\mathbf k)$ with $\mathbf k=(\omega/c)\,n_r\hat{\mathbf n}$ is
phase-matched when

$$
\omega-\mathbf k\cdot\mathbf v=\mathbf g\cdot\mathbf v
\quad\Longrightarrow\quad
\hbar\omega=\frac{\hbar\,\mathbf g\cdot\mathbf v}{1-n_r\,\hat{\mathbf n}\cdot\mathbf v/c}
=\frac{\hbar c\,\beta\lvert\mathbf g\rvert\cos\psi}{1-n_r\beta\cos\theta},
$$

with $\cos\psi=\hat{\mathbf g}\cdot\hat{\mathbf v}$ and
$\cos\theta=\hat{\mathbf n}\cdot\hat{\mathbf v}$. In vacuum kinematics
$n_r=1$, which is the ledgered form. Here $\mathbf g$ carries the $2\pi$
($\lvert\mathbf g\rvert=2\pi/d_{hkl}$); with a $1/d$ convention the prefactor
is $hc$ instead of $\hbar c$.

### Upper bound over direction

Physical emission requires $\hbar\omega>0$. For $\beta<1$ and $n_r\le 1$ the
denominator obeys

$$
1-n_r\beta\cos\theta\;\ge\;1-\beta\;>\;0,
$$

so the sign of $\hbar\omega$ is that of the numerator. On the emitting branch
$0<\beta\lvert\mathbf g\rvert\cos\psi\le\beta\lvert\mathbf g\rvert$. For a
nonnegative numerator bounded above by $A$ and a positive denominator
bounded below by $B$, the ratio is at most $A/B$; the two maximizations need
not be achieved by the same direction for the inequality to hold. Hence

$$
\boxed{\;0<E_{\rm res}\le\frac{\hbar c\,\beta\lvert\mathbf g\rvert}{1-\beta}\;}
$$

for each reflection, and over the reflection set the maximum is attained at
$\lvert\mathbf g\rvert_{\max}$, because the bound increases monotonically in
$\lvert\mathbf g\rvert$. Negative-$\mathbf g$ partners ($-\mathbf g$, emitting where
$\cos\psi<0$) share the same $\lvert\mathbf g\rvert$ and are covered.

**Tightness.** Both extrema occur at $\hat{\mathbf n}=\hat{\mathbf v}=\hat{\mathbf g}$
(forward emission along a velocity parallel to $\mathbf g$). The two maximizations
are therefore compatible, and the bound is the supremum over unconstrained
directions, not merely a conservative bound. Treating them independently
costs nothing.

### Refractive index

If the kernel uses an in-medium $n_r=\operatorname{Re}n(E)$ in the denominator, then
$n_r\le 1$, which holds for X-ray energies where $\delta>0$, gives
$1-n_r\beta\cos\theta\ge 1-n_r\beta\ge 1-\beta$, so the vacuum bound still
holds and is looser. The vacuum form is valid only where
$\operatorname{Re}n\le 1$ at the resonance energy. At visible/UV energies,
$\operatorname{Re}n$ can exceed 1, but such resonances lie far below
$E_{\max}$ unless $\beta$ is tiny. This is an assumption to check against the
tables (see the comparison section).

### Speed along the trajectory

$E_{\max}$ increases strictly with $\beta$:
$\partial_\beta[\beta/(1-\beta)]=1/(1-\beta)^2>0$. It therefore bounds every
segment if the speed used is the maximum over every radiating trajectory.
Monotone slowing down (CSDA plus straggling losses; no energy gain in
transport) makes the initial speed the per-track maximum. Secondaries are
born below the primary energy. **Caveat:** if the beam samples a kinetic
energy spread around the nominal `energy_keV`, some electrons start above it.
The bound then holds only if the caller passes the maximum sampled energy, not
the nominal one. Within a sampled spread, the relative excess is
$\delta E_{\max}/E_{\max}=\delta\beta/[\beta(1-\beta)]$.

### Kinematics: kinetic energy to $\beta$

With $T$ the kinetic energy and $m_ec^2=510.998\,95\ \text{keV}$,

$$
\gamma=1+\frac{T}{m_ec^2},\qquad \beta=\sqrt{1-\gamma^{-2}},\qquad
\frac{1}{1-\beta}=\frac{1+\beta}{1-\beta^2}=\gamma^2(1+\beta)\xrightarrow{\beta\to1}2\gamma^2 .
$$

### Units and rounding

$\hbar c=1973.269\,804\ \text{eV·Å}$ times $\lvert\mathbf g\rvert\ [\text{Å}^{-1}]$ is in eV, and
$\beta$ is dimensionless. A bound must round **up**, so
$E_{\rm stop}=\lceil E_{\max}/\Delta\rceil\,\Delta$ with $\Delta=$`round_to_eV`.
Rounding to the nearest multiple could cut up to $\Delta/2$ below the
supremum.

### Cheap filters

| check | expectation | result |
| --- | --- | --- |
| units | $[\hbar c][\lvert\mathbf g\rvert]=\text{eV·Å}\cdot\text{Å}^{-1}=\text{eV}$ | pass |
| $\beta\to0$ | $E_{\max}\approx\hbar c\lvert\mathbf g\rvert\beta\to0$ | pass |
| $\beta\to1$ | $E_{\max}\to2\gamma^2\hbar c\lvert\mathbf g\rvert$, the relativistic Doppler limit | pass |
| $\lvert\mathbf g\rvert\to0$ | $E_{\max}\to0$; linear in $\lvert\mathbf g\rvert$, so the $n$-th order is $n\times$ | pass |
| sign | denominator $\ge1-\beta>0$; only the positive-numerator branch emits | pass |

### Independent numeric reference point

HOPG (004), with $c=6.711\ \text{Å}$, has $\lvert\mathbf g\rvert=2\pi/(c/4)=3.74501\ \text{Å}^{-1}$.
At $T=30\ \text{keV}$, $\gamma=1.058709$, $\beta=0.328376$ and $1/(1-\beta)=1.48893$, so
$\hbar c\lvert\mathbf g\rvert=7389.9\ \text{eV}$ and $E_{\max}=3613.1\ \text{eV}$. This rounds up to
$3700\ \text{eV}$ at $\Delta=100\ \text{eV}$.

## Comparison with implementation

*(Written after the derivation above was complete.)*

The body of `kinematic_cxr_stop_eV` is:

```text
finite = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
gamma = 1.0 + energy * 1.0e3 / _ELECTRON_REST_EV
beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
bound = _HBARC_EV_ANG * beta * float(finite.max()) / (1.0 - beta)
return float(math.ceil(bound / float(round_to_eV)) * float(round_to_eV))
```

with `_HBARC_EV_ANG = 1973.269804` and `_ELECTRON_REST_EV = 510998.95`.

| item | derivation | implementation | result |
| --- | --- | --- | --- |
| prefactor | $\hbar c$ in eV·Å | `_HBARC_EV_ANG` = 1973.269804 | match |
| $\lvert\mathbf g\rvert$ convention | $2\pi/d$ in Å$^{-1}$ | `reciprocal_g_vector` returns $2\pi/d_{hkl}$ (verified numerically, HOPG (004)) | match |
| $\gamma$, $\beta$ | $1+T/m_ec^2$, $\sqrt{1-\gamma^{-2}}$ | same; keV converted to eV | match |
| direction bound | $\beta\lvert\mathbf g\rvert/(1-\beta)$ | same | match |
| reflection set | $\max\lvert\mathbf g\rvert$ over the case set | `finite.max()`; caller passes the case `cp["hkl_list"]` after `max_reflections` truncation, the same list the case's line kernels receive | match |
| rounding | ceiling | `math.ceil` | match |
| speed used | **maximum** incident speed over the beam | nominal `energy_keV` from `_automatic_cxr_grid_policy` | **diverges when `energy_spread_frac` is set** (below) |

The kernel resonance (`montecarlo/spectrum/cxr/_kernels.py::_cxr_kin_core`
with `_in_medium_kinematics`) is
$E_{\rm res}=\hbar c\,(\boldsymbol\beta\cdot\mathbf g)/D$, where
$D=1-\boldsymbol\beta\cdot\hat{\mathbf n}-\delta\,\boldsymbol\beta\cdot\nabla L$,
$\nabla L=-\mathbf e/(\mathbf e\cdot\hat{\mathbf n})$ and $\delta=1-\operatorname{Re}n$. Roots with
$2\lvert\delta\rvert\lvert\nabla L\rvert^2\ge1$ are rejected. This is the ledgered vacuum form
plus the Snell term, which the bound must also cover.

### Snell term, $\delta>0$

In the exit plane, take $\hat{\mathbf n}$ at angle $\theta$ from the surface normal $\mathbf e$,
and $\hat{\mathbf v}$ at angle $\theta+a$. To second order,

$$
D\simeq 1-\beta+\tfrac12\beta a^2+\delta\beta\,(1-a\tan\theta)
\;\ge\;1-\beta+\delta\beta\Bigl(1-\frac{\delta}{2\cos^2\theta}\Bigr)
\;>\;1-\beta+\tfrac34\delta\beta ,
$$

where the last step uses the guard $2\delta/\cos^2\theta<1$. The vacuum
bound therefore holds with margin for $\delta>0$. A Nelder–Mead search over
$(\theta,a,\phi)$ at 300 keV with $\delta=10^{-5}$ and $10^{-3}$ found
$\min D\ge1-\beta$.

### Snell term, $\delta<0$ (finding, no practical effect)

The production tables (`use_henke=True`) give $\operatorname{Re}n>1$ in soft-X-ray
windows. Examples are $\delta=-3.1$ for BaF$_2$ between 14 and 780 eV,
$\delta<0$ up to about 584 eV for FeTe/HfTe$_2$, and HOPG at the carbon K edge.
Where $\delta<0$ the forward root has $D=1-\beta(1+\lvert\delta\rvert)<1-\beta$, so the
unrounded vacuum bound is not a strict bound there. A scan over every catalog
material's default reflection set covered
$T\in\{5,10,20,30,60,100,300\}$ keV. It asked whether any guard-admitted
($\lvert\delta\rvert<1/2$, normal exit) self-consistent root lies above $E_{\max}$.
Only one root qualified: PtTe$_2$ at 10 keV, at $575.2\ \text{eV}$ for $E_{\max}=575.2\ \text{eV}$, at the
boundary itself. That root sits below the rounded stop of $600\ \text{eV}$. No root exceeded a
rounded stop. The assumption $\operatorname{Re}n\le1$ should be stated in the
docstring/ledger, but it does not change any current grid.

### Incident speed with beam energy spread (discrepancy)

`initial_energies_keV` samples $E_e=E_0(1+f z_e)$ with $z_e\sim N(0,1)$ unbounded
when `Sweep.energy_spread_frac` $=f$ is set. `_automatic_cxr_grid_policy`
passes the nominal $E_0$ = `energy_keV`, and no code path widens the stop for
the spread. The docstring's premise that "`beta` is the incident speed, the
largest the electron ever has" is therefore false for about half the electrons
in a spread beam. The first divergent factor is $\beta(E_0)$ where the bound
needs $\beta(\max_e E_e)$. Measured with the independent formula at $f=0.01$ and
$+3\sigma$, the bound rises by 2.0 % at 30 keV, where the 3700 eV rounding
absorbs it. At 300 keV it rises by 3.3 %, from 25 690 to 26 534 eV, which
exceeds the rounded stop of 25 700 eV. Lines from above-nominal electrons near
$\hat{\mathbf n}\parallel\hat{\mathbf v}\parallel\hat{\mathbf g}$ can then lie above the
axis. The 1.6–2.8× practical margin recorded in the ledger makes clipping
unlikely in practice, but the claim is a strict bound, not an estimate. GPT/GDF
beams (`source="gpt_gdf"`) also carry per-electron energies from file. This
verification did not check them.

### Other findings

- The docstring cites `montecarlo/spectrum/lines/_kernels.py::_cxr_kin_core`.
  That file does not exist; the function is in
  `montecarlo/spectrum/cxr/_kernels.py` (the ledger row has it right).
- Tightness: the supremum is attained, not merely bounded. A random sample of
  $10^6$ $(\hat{\mathbf v},\hat{\mathbf n})$ pairs at 300 keV reaches $0.995\,E_{\max}$.

## Numerical evidence and anchors

The scratch script used only stdlib/NumPy constants, not implementation
helpers. The function under review was called only to compare its output.

| $T$ (keV) | independent $E_{\max}$ (eV), HOPG-like $\lvert\mathbf g\rvert=2\pi/1.677$ | independent ceil-100 | `kinematic_cxr_stop_eV` |
| --- | --- | --- | --- |
| 30 | 3614.755 | 3700 | 3700.0 |
| 60 | 5957.351 | 6000 | 6000.0 |
| 100 | 8971.450 | 9000 | 9000.0 |
| 300 | 25689.817 | 25700 | 25700.0 |
| 1000 | 118083.994 | 118100 | 118100.0 |

The catalog HOPG (004) magnitude from `reciprocal_g_vector` equals
$2\pi/(c/4)=3.745007\ \text{Å}^{-1}$ exactly.

Anchor: `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/energy-grid/test_line_grid_policy.py`
passed (32 tests). The anchor
`test_kinematic_stop_bounds_the_resonance_over_random_directions` checks the
vacuum form over random directions with exact rounding disabled. It is a
bound-property test, not a reference value. It does not exercise the Snell
term, energy spread, or the attaining direction.

## Adjudication

`discrepancy`, narrow in scope. For a monoenergetic beam, the closed form,
units, $\lvert\mathbf g\rvert$ convention, reflection set, ceiling rounding and all
limits match the independent derivation exactly. The joint direction
maximization is the attained supremum. With $\delta\ge0$ the in-medium Snell
term only lowers resonances. The required check "`beta` at incident energy is
the maximum over the trajectory" fails when `energy_spread_frac` is set:
`_automatic_cxr_grid_policy` evaluates $\beta$ at the nominal $E_0$, not at the
largest sampled incident energy. The owner can resolve this by either

1. scoping the claim to monoenergetic beams (then `rederived` applies as-is), or
2. evaluating the bound at a declared upper incident energy (for example the
   realized $\max_e E_e$, or $E_0(1+k f)$ with a stated $k$ and the residual
   probability recorded).

The $\operatorname{Re}n\le1$ assumption and the stale docstring path should be
recorded either way. Human sign-off remains pending; this verification does not
grant it.

## Independent re-verification (fresh context, 2026-11-11, declared upper energy)

Fresh context, commit `491208cde`. The ledger row, the signatures and the
docstrings were read first. The derivation below was written before the bodies
of `kinematic_cxr_stop_eV`, `_automatic_cxr_grid_policy`,
`line_window_ceiling_eV`, `upper_beam_energy_keV` and `initial_energies_keV`,
or any earlier section of this page, were opened. The separate
`line-grid-emission-ceiling` cap (energy conservation and the data onset) is
verified elsewhere. This section depends on it only because the same
`upper_keV` also feeds that cap.

### Cheap filters

- **Units.** $\hbar c$ [eV Å] $\times\lvert\mathbf g\rvert$ [Å$^{-1}$] gives
  eV, and $\beta/(1-\beta)$ is dimensionless. $E_{\rm up}=E_0(1+kf)$ is in keV
  because $f$ and $k$ are pure numbers. Pass.
- **Limits.** $f=0$ gives $E_{\rm up}=E_0$, so the bound reduces to the
  previously verified monoenergetic form. $\beta\to0$ gives $0$. The bound is
  linear in $\lvert\mathbf g\rvert_{\max}$. Pass.
- **Signs and conventions.** $f$ is an RMS *kinetic*-energy fraction, as for the
  sampler. A negative $f$ is harmless, because the sampler's law is symmetric
  in $z$ and $E_{\rm up}$ uses $\lvert f\rvert$. Pass.

### Independent derivation

Write $\mathbf v=\beta c\,\hat{\mathbf v}$ and the in-medium index $n$, with
$\operatorname{Re}n\le1$. For a line of reciprocal vector $\mathbf g$ seen
along $\hat{\mathbf n}$, the resonance is

$$
E_{\rm res}=\frac{\hbar c\,\beta\,\hat{\mathbf v}\cdot\mathbf g}
{1-\beta\,\operatorname{Re}n\,\hat{\mathbf n}\cdot\hat{\mathbf v}} .
$$

For $E_{\rm res}>0$, the numerator is at most $\beta\lvert\mathbf g\rvert$ and the
denominator is at least $1-\beta$. Both extremes occur together at
$\hat{\mathbf n}\parallel\hat{\mathbf v}\parallel\hat{\mathbf g}$, so

$$
\sup_{\hat{\mathbf v},\hat{\mathbf n},\,\mathbf g\in\mathcal G}E_{\rm res}
=B(\beta)\equiv\frac{\hbar c\,\beta\,\lvert\mathbf g\rvert_{\max}}{1-\beta}
$$

is attained (it is the supremum, not only an upper bound).

**Monotonicity.** $\mathrm dB/\mathrm d\beta=\hbar c\lvert\mathbf g\rvert_{\max}/(1-\beta)^2>0$.
With $\gamma=1+T/m_ec^2$, we have $\mathrm d\beta/\mathrm dT=1/(m_ec^2\gamma^3\beta)>0$, so
$B$ increases strictly with kinetic energy $T$. In the transport, every electron's
$T$ is non-increasing:

- Continuous and Urban-sampled losses are positive (`straggling.py` returns a
  positive loss, and the cores use `E -= |dEds| s`).
- Secondaries start below their parent.
- Pair positrons carry at most $E_\gamma-2m_ec^2$.

So $\max_{\text{trajectory}}B=B(\beta(T_{\rm start}))\le B(\beta(\max_eT_e))$, and
evaluating $B$ at any $E^\star\ge\max_e T_{e,\rm start}$ is a strict bound.

**Upper incident energy.** If $T_e=E_0(1+f z_e)$ with $\lvert z_e\rvert\le k$,
then $T_e\le E_0(1+k\lvert f\rvert)=E_{\rm up}$, so $E^\star=E_{\rm up}$
is valid. In floating point the sampler computes `E0 * (1.0 + f * z)` with
`z = ±6.0` at the clip. `upper_beam_energy_keV` computes
`E0 * (1.0 + 6.0 * |f|)`. IEEE multiplication is commutative, and $f\cdot(-6)$
is exactly $-6\lvert f\rvert$ when $f<0$. The two values are therefore
bit-identical at the clip, and the bound holds without a rounding margin. The
Gaussian law of the clamp is derived in
[`beam-energy-spread-injection`](beam-energy-spread-injection.md).

**Stop.** $S=\lceil B(\beta(E_{\rm up}))/100\ \text{eV}\rceil\cdot100\ \text{eV}$.

### Independent oracle

The oracle is `mpmath` at 40 digits and imports nothing from PyRITE. It uses
CODATA 2018 $\hbar c=1973.269804$ eV Å and $m_ec^2=510.99895$ keV.

| $\lvert\mathbf g\rvert$ (Å$^{-1}$) | $E_0$ (keV) | $f$ | raw $B(E_{\rm up})$ (eV) | stop (eV) | code |
|---|---|---|---|---|---|
| $2\pi/1.677$ | 30 | 0 | 3614.75 | 3700 | 3700 |
| $2\pi/1.677$ | 30 | 0.01 | 3762.16 | 3800 | 3800 |
| $2\pi/1.677$ | 30 | 0.05 | 4339.29 | 4400 | 4400 |
| $2\pi/1.677$ | 100 | 0.001 | 9016.84 | 9100 | 9100 |
| $2\pi/1.677$ | 100 | 0.01 | 9426.05 | 9500 | 9500 |
| $2\pi/1.677$ | 300 | 0 | 25689.82 | 25700 | 25700 |
| $2\pi/1.677$ | 300 | 0.001 | 25857.97 | 25900 | 25900 |
| $2\pi/1.677$ | 300 | 0.01 | 27386.83 | 27400 | 27400 |
| $2\pi/1.677$ | 300 | 0.05 | 34522.35 | 34600 | 34600 |
| $2\pi/1.677$ | 1000 | 0.01 | 128564.34 | 128600 | 128600 |
| $2\pi/1.677$ | 1000 | 0.05 | 174553.94 | 174600 | 174600 |
| 5.0 | 30 | 0.001 | 4843.70 | 4900 | 4900 |
| 5.0 | 300 | 0.01 | 36548.11 | 36600 | 36600 |
| 5.0 | 1000 | 0.05 | 232944.71 | 233000 | 233000 |

The "code" column is `kinematic_cxr_stop_eV(g, upper_beam_energy_keV(E0, f))`.
All 14 points agree exactly. The HOPG 300 keV, $f=0.01$ resonance at $+6\sigma$
is 27386.8 eV, which lies below the 27400 eV stop. The earlier counterexample at
$+3\sigma$, 26534 eV, is therefore covered.

### Comparison with the implementation

- `kinematic_cxr_stop_eV`: the code computes $\gamma=1+E\cdot10^3/m_ec^2$ and
  $\beta=\sqrt{1-\gamma^{-2}}$. It takes the bound
  `_HBARC_EV_ANG * beta * max|g| / (1 - beta)` over finite positive
  $\lvert\mathbf g\rvert$ and rounds it with `ceil`. The constants are the
  CODATA values above. This matches the derivation term for term, and the
  formula is unchanged by this remediation.
- `upper_beam_energy_keV`: $E_0(1+6\lvert f\rvert)$, with `None` or `0` giving
  exactly $E_0$. Matches.
- `_automatic_cxr_grid_policy`: calls `kinematic_cxr_stop_eV(magnitudes,
  upper_beam_energy_keV(energy_keV, sweep.beam.energy_spread_frac))` over the
  case `hkl_list`. Matches.
- `line_window_ceiling_eV`: uses the same $E_{\rm up}$ from
  `case.get("energy_spread_frac")` over every radiating layer. Matches.

### Spread routes to automatic grids

**Sweep path (end to end).** I built a real `Sweep` with
`BeamSpec(energy_spread_frac=...)` and ran it through `build_cases`, without the
`SimpleNamespace` stub the anchor uses. The case carries the same
`energy_spread_frac` (`campaign/beam_cases.py::beam_case_keys`, `float(b.energy_spread_frac)`)
and the same `E0_keV` that the policy saw.

| HOPG, $E_0$ (keV) | $f$ | `line_grid_policy` bandwidth `stop_eV` |
|---|---|---|
| 300 | none | 25700 |
| 300 | 0.01 | 27400 |
| 30 | 0.05 | 4400 |

All three equal the oracle values.

**Every sampler call site uses `initial_energies_keV`, so every one is clamped.**

- `transport/beam_entry.py` (`sample_primary_entry`)
- `transport/host_pass.py`
- `runner/block_transport.py` (block-addressed, bit-identical to the joined
  call)
- `transport/secondaries.py` (table range only)

Launched secondary, channeling and cone-split passes null
`energy_spread_frac` (`_BEAM_ARGS`, `_LAUNCH_BEAM_ARGS`, `_COPY_BEAM_ARGS`). No
other code adds energy to a transported electron: there is no `E += …` in
`montecarlo/transport/`.

**GPT/GDF.**

- `BeamSpec.gdf_beam` and `host_pass._gdf_entry` refuse `energy_spread_frac`
  with a GDF source.
- A non-shape-only sweep sets the case energy to `gdf.energy_keV.max()`
  (`_case_energies`), so the automatic stop is evaluated at the realized maximum
  and is still strict.
- A shape-only beam is monoenergetic at $E_0$.

The row's exclusion of GDF is therefore conservative for the sweep path.

**Channeling.**

- `IncidentEnergyEnsemble` (`cutoff_sigma`, default 4, user-settable) is a
  diagnostic API. Nothing in the runner or campaign imports it, so it feeds no
  automatic line grid.
- The channeling runner's analytic `energy_spread_eV` broadens channeling
  radiation, not a PXR/CBS resonance, so it is outside this claim.
- Its transport primaries come from the clamped sampler.

**Other consumers of the cap.** The `resonance-population` measured policy
(`runner/cxr_grid.py`, `case_cxr_stop_eV`) caps at `bandwidth["stop_eV"]`, which
is this same bound. No other caller of `kinematic_cxr_stop_eV` exists.

**Outside this claim.** Stored catalog `E_grid_line_by_energy` rows are taken
before automatic resolution and were derived at nominal energy. They are not
automatic grids. Their spread headroom is the `line_shift_fraction`/15 % margin
argument of `beam-energy-spread-injection`.

### Tests

The four listed modules ran via `pyrite-dev test`
(`tests/energy-grid/test_line_grid_policy.py`, `tests/montecarlo/transport/test_transverse.py`,
`tests/energy-grid/test_bounds.py`, `tests/montecarlo/spectrum/test_beam_energy_spread_grid.py`):
81 passed, 1 slow test skipped. The new anchor drives `_automatic_cxr_grid_policy`
through a `SimpleNamespace` sweep. The end-to-end `build_cases` probe above
closes that gap.

### Findings (non-blocking)

- The Notes on the row still say "Status stays `discrepancy`", and the
  docstring scope text says GDF is "not covered". The sweep path in fact covers
  GDF via the maximum energy. The scope text is conservative but imprecise.
- The `Re n <= 1` assumption is unchanged from the earlier review and was not
  re-examined here.

### Verdict

`rederived`. The first divergent factor from the earlier review was $\beta$ at
the nominal $E_0$. It is now $\beta$ at $E_{\rm up}=E_0(1+6\lvert f\rvert)$,
which bounds every analytic incident energy exactly (bitwise at the clip).
Combined with the strict monotonicity of $B$ in $T$, this makes the bound strict
for every analytic spread beam on an automatic grid. The verdict depends on the
`beam-energy-spread-injection` clamp (verified alongside). It does not certify
the `line-grid-emission-ceiling` cap. No human sign-off is implied.
