# Automatic line-grid emission ceiling

Validation: `line-grid-emission-ceiling`. Status: discrepancy.

## Independence and scope

Fresh-context verification for #1101, against base `fbd0958c9`.
The ledger record, relevant source records, cited atomic-data references and
AST-extracted signatures/docstrings were read before implementation bodies.
The independent expression below was written before reading those bodies or
previous derivation pages. This is an evidence document, not a physics change.
The supplied resonance cap is treated conditionally: the separate
`line-grid-kinematic-bandwidth` discrepancy is not resolved here.

## Sources and independent derivation

[Chantler (1995), Table 1](https://srd.nist.gov/jpcrdreprint/1.555974.pdf)
tabulates atomic factors over element-dependent finite energy intervals;
[NIST FFAST](https://physics.nist.gov/ffast) identifies the underlying dataset.
The local `henke_dispersion` interface states that unsupported energies become
NaN; its historical name does not mean Henke/CXRO data. Attenuation uses
[Cullen's EPDL2025](https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL),
not those anomalous factors. Finite dataset support is a numerical domain
restriction, not a physical cutoff in photon emission.

Energy conservation for one radiating electron with initial kinetic energy
$T_i$ and no external energy input gives

$$
E_\gamma+T_f+E_{\rm other}=T_i,
\qquad T_f\geq0,\quad E_{\rm other}\geq0,
\qquad E_\gamma\leq T_i.
$$

A nominal beam energy is a bound only if every radiating electron's energy is
bounded by it. A Gaussian energy distribution has no finite strict upper bound.
For a supplied, valid resonance cap $R$, and a route whose accepted line
centres are bounded by the common table-support cutoff $D$, the unrounded
policy ceiling and upward-rounded axis stop are

$$
C=\min(T_i,D,R),\qquad
S=\Delta\left\lceil\frac{C}{\Delta}\right\rceil,
\qquad \Delta=100\ {\rm eV},\qquad C\leq S<C+\Delta.
$$

Thus the axis endpoint may exceed the physical or numerical ceiling by less
than 100 eV. This is an axis policy, not a pointwise spectral-support theorem:
finite-time sinc-squared and characteristic Lorentzian profiles have tails.

For table $j$, let $D_j$ be its upper NaN onset. If an accepted route needs all
those tables evaluated at the *final* centre $E$, and rejects every nonfinite
coefficient, then $D=\min_jD_j$ suffices. That premise must be checked separately
for each route. Iterating a root with a refraction lookup at $E_m$ and returning
$E_{m+1}$ does not imply that refraction was evaluated at $E_{m+1}$. In
particular, a last update can cross $D_{\rm refraction}$ while the previously
read refraction and other coefficients stay finite. A minimum over all tables
then need not bound coherent accepted centres. Likewise, an independently
produced characteristic line does not automatically inherit a coherent
coupling-table cutoff.

For a sum over atomic basis contributions, any unsupported anomalous term
makes the sum nonfinite. Its upper support is the minimum endpoint of the
actually participating anomalous elements, not necessarily all absorber
endpoints. An absorber's attenuation support must be tracked independently.
Closed versus open endpoint inequalities matter: a finite attenuation value
at its last knot is not NaN at that knot.

## Initial filters

Units pass: all three caps and the rounding quantum are photon energies in eV;
beam keV must be multiplied by 1000. Limits pass conditionally: removing one
cap sends it to infinity; the smallest remaining cap binds. Rounding adds no
more than one quantum. Signs pass for positive beam energies and support
endpoints. The blanket table-rejection statement remains a condition to check,
not a consequence of dimensional analysis. No physical accuracy near recoil
or atomic-data endpoints follows from these filters.

## Implementation comparison

Definition owners resolve to `src/pyrite/campaign/sweep.py` and
`src/pyrite/montecarlo/spectrum/cxr/_kernels.py`. Serena refused the ignored
worktree path; scoped direct reads were used after the derivation was written.

The policy computes the rounded resonance cap first, selects the smaller of
nominal kinetic energy and `_cxr_emission_ceiling`, rounds that result up to
100 eV, and replaces the original stop only when the rounded cap lies strictly
between `start_eV` and the original stop. For the ordinary domain where the
cap exceeds the start, monotonic rounding gives the independently derived
minimum. The guard means the unqualified minimum formula is not a theorem for
a configuration whose proposed cap is at or below the axis start.
A tie between the kinetic and data limits records `energy conservation` by
dictionary insertion order. A cap tied with the existing rounded stop leaves
the original provenance unchanged. No new `sources["stop_eV"]` entry is added
when the kinematic cap remains binding.

`resonance-population` returns before these two additional caps are applied.
Its supplied resonance cap is unchanged. This is verified as bookkeeping,
without certifying the separate resonance bound for energy-spread beams.

The serialized policy includes both `bandwidth.stop_eV` and `sources`;
`_line_grid_case_keys` stores it in `line_grid_policy`, and
`campaign/profiles.py::case_content_key` hashes that mapping.
`identity_options.py::_CONTENT_KEY_DENYLIST` excludes neither the policy nor
the encoded line coordinates. The `attenuation_model` also joins identity.
Thus changed stops and changed recorded cap provenance are identity-bearing.
This establishes the mechanism, not historical equivalence of all old cases.

### First divergent domain/rejection premise

The `_cxr_table_nan_onsets` docstring groups `chi_g` and `U_g` as sharing the
anomalous endpoint. Actual `materials/crystal.py::U_g` and
`reflection_coupling_tables` use the static form factor for the electrostatic
potential; its tabulated real and imaginary rows remain finite at and above
that endpoint. The returned onset list has one entry for susceptibility,
one for refraction and one per absorber. It does not represent the support of
every coupling row individually. The numerical minimum is still the
susceptibility/refraction onset for default anomalous HOPG.

The first failure in the ledger's *universal rejection* implication is the
predicate: the coherent route tests the scalar positive prefactor `amp`, time,
and finite formation attenuation, **not** the complex coefficients
`pol_A`/`coefs`. See `_per_hkl.py::_formation_good` and
`_accumulate_reflection_coherent`, and `_batched.py::_batched_coherent_block`.
The current `coherent_jit_kernel.py` reads the polarization coefficients
without the ledger's claimed `amp != amp` rejection guard. A NaN coefficient
can therefore propagate into a field sum rather than being dropped.

The incoherent/component route does test the common total `weight` with
`isfinite(weight) & (weight > 0)`; the fused route applies the same kind of
predicate. The total susceptibility NaN poisons that total even when static
CBS coupling remains finite. This is a numerical rejection of that combined
model, not proof that physical CBS emission ceases there.

There is also a distinct evaluation-domain mismatch: coherent formation and
escape phases sample `delta_omega_grid` on the **output photon axis**.
Upward rounding puts part of the automatic axis beyond the finite refraction
support. Rejection of a line centre cannot ensure finite phase evaluation on
that suffix. This yields the end-to-end failure below even with a centre below
the endpoint.

Characteristic scoring is separate (`runner/emission.py` and
`spectrum/characteristic.py`). It uses EADL transition energies and EPDL
attenuation, not the crystal's susceptibility or refractive-index lookup at
its centre. It integrates Lorentzian bin mass and retains out-of-window tail
mass; unsupported attenuation raises through `_finite_mu_or_raise`. Thus it
must not be justified by the coherent coupling-mask claim. No characteristic
centre above the default carbon endpoint is asserted by this verification;
its tails alone show why “empty axis” is stronger than the centre-only claim.

## Independent numerical evidence

All probes used the CPU backend and the project environment, with no transport
simulation, remote compute or source modifications. These are deterministic
domain and evaluation checks; no tolerance was fitted to a spectrum.

1. Raw `xraydb.chantler_energies("C")` gives a maximum of 966266.74 eV.
   At the immediately preceding float, the endpoint and the immediately
   following float, `henke_dispersion` is finite, NaN, NaN respectively.
   The actual `(004)` susceptibility rows have the same finiteness pattern;
   both static potential rows stay finite at all three points.
2. `_cxr_table_nan_onsets` for default HOPG returns
   `[966266.74, 966266.74, 100000000000.00002]` eV. Direct EPDL attenuation
   for carbon at density 0.1 atoms/angstrom cubed is
   $1.28936865\times10^{-9}$ inverse angstrom at 966266.74 eV and
   $3.59032284\times10^{-10}$ inverse angstrom at $10^{11}$ eV, then NaN at
   the next representable float above $10^{11}$ eV. This independently checks
   the open anomalous endpoint versus closed attenuation endpoint convention.
3. Using $g_{004}=8\pi/(6.711\ {\rm \mathring A})$,
   $m_ec^2=510998.95\ {\rm eV}$ and
   $\hbar c=1973.269804\ {\rm eV\,\mathring A}$ directly, without the
   production cap helper, gives the following conditional policy oracle.

| Nominal beam energy (keV) | Supplied monoenergetic resonance cap before rounding (eV) | Independent and implemented stop (eV) |
| --- | ---: | ---: |
| 30 | 3613.138752 | 3700 |
| 300 | 25678.332697 | 25700 |
| 3000 | 686630.297618 | 686700 |
| 5000 | 1707961.985501 | 966300 |
| 10000 | 6242325.150619 | 966300 |
| 50000 | 144400151.463439 | 966300 |

The independent oracle uses

$$
\gamma=1+\frac{T}{m_ec^2},\quad
\beta=\sqrt{1-\gamma^{-2}},\quad
R=\frac{\hbar c\,\beta g_{004}}{1-\beta},\quad
S=100\ {\rm eV}\left\lceil
\frac{\min(T,966266.74\ {\rm eV},R)}{100\ {\rm eV}}\right\rceil.
$$

The rounded kinetic-only result at 50 MeV is covered by the focused anchor
with the data ceiling removed. The nominal-energy substitution is not an
independent repair of `line-grid-kinematic-bandwidth`.

### Final-root crossing

An independently manufactured table can make three finite reads return a
root beyond support. Let $D=966266.74\ {\rm eV}$, $d_0=0.25$ and
$N=d_0(D-4\ {\rm eV})$. Put finite knots at one-eV spacing below $D$ and
NaNs from $D$ onward. At the three input energies set

$$
\delta(D-4)=d_0-\frac{N}{D-3},\quad
\delta(D-3)=d_0-\frac{N}{D-2},\quad
\delta(D-2)=d_0-\frac{N}{D+0.25},
$$

with energies consistently in eV. Iterating $E_{m+1}=N/(d_0-\delta(E_m))$
gives 966262.74, 966263.74, 966264.74, 966266.99 eV. The production
`_in_medium_kinematics` returns the same final energy, with finite last
$\delta=1.09959257\times10^{-6}$ and a satisfied convergence guard.
There is no final lookup at 966266.99 eV. A finite formation predicate with
positive scalar amplitude accepts a row even when its complex coefficient
is NaN; multiplying that coefficient by a finite profile gives NaN, not zero.
This synthetic example isolates the logical rejection premise; it does not
claim this particular manufactured refraction table is physical carbon data.

### Actual coherent output-axis failure

A bounded, four-coordinate full scoring call uses one 5 MeV straight segment
of length 10 angstrom, midpoint $(4,0,5)$ angstrom, velocity direction
$(0,0,1)$, zero emission clock, a 10-angstrom slab and footprint, HOPG `(004)`,
and `B_ang2=0`. Choose the observation direction from the vacuum target
$E_{\rm vac}=966230\ {\rm eV}$:

$$
n_z=\frac{1-\hbar c\,\beta g_{004}/E_{\rm vac}}{\beta},\qquad
\hat{\mathbf n}=(\sqrt{1-n_z^2},0,n_z)
\simeq(0.08143625,0,0.99667855).
$$

Calling `mc_cxr_spectrum` on
`[966200, 966250, 966280, 966300]` eV gives:

```text
coherent=False: [3.22298587e-17, 3.22298607e-17, 3.22298522e-17, 3.22298425e-17]
coherent=True:  [3.22298587e-17, 3.22298607e-17, nan, nan]
```

The last two coordinates exceed the raw refraction endpoint while remaining
inside the rounded stop. This is an actual atomic-data failure, independent
of the manufactured root-crossing example. It disproves the unconditional
claim that trimming to the rounded emission ceiling merely removes empty
axis and that all routes safely drop unsupported factors.

## Focused checks and filter adjudication

The canonical command was
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test` with these four nodes
from `tests/energy-grid/test_line_grid_policy.py`:

- `test_automatic_stop_is_capped_at_the_atomic_data_emission_ceiling`
- `test_automatic_stop_keeps_the_kinematic_bound_when_it_is_lowest`
- `test_automatic_stop_is_capped_at_the_beam_kinetic_energy`
- `test_measured_bandwidth_keeps_the_kinematic_cap`

Result: **4 passed in 8.33 s**. These anchors test selection and provenance;
they do not exercise coherent scoring or independently validate data domains.

| Filter | Result |
| --- | --- |
| Dimensions, keV/eV boundary, 100-eV rounding | Pass |
| Minimum-limit cases and measured-policy bypass | Pass in the stated cap-above-start domain |
| Physical energy-conservation premise | Concern: applies per electron; nominal energy is not a bound on an unbounded spread |
| Notation and per-route support convention | Fail: static potential support is grouped with anomalous support; coherent coefficient finiteness is not the scalar-amplitude predicate |
| Atomic source and endpoint units | Pass for installed Chantler and EPDL domains; not Henke/CXRO or Elam attenuation |
| Reference-value transcription audit | Limited: raw installed carbon endpoint checked; full original Chantler paper fetch exceeded the browser size limit; no complete paper-table transcription audit claimed |
| Tolerances | Exact domain/rounding checks; printed numerical spectra are diagnostic values, not calibrated accuracy thresholds |
| External-code/parity evidence | None required or claimed; CPU probe is not GPU or FP32/FP64 parity |
| Documentation render/static checks | Task owner performs shared docs, ledger and positive HTML math checks |

The [EPICS2025 publisher page](https://nuclear.llnl.gov/EPICS/) identifies the
release and links the underlying [EPDL report](https://nuclear.llnl.gov/EPICS/DOCUMENTS/EPDL.rev1.pdf).
The report's section 3 (printed pages 5–6) distinguishes tabulated atomic
support from physical validity. The 100-GeV endpoint above is checked against
the installed data seam, not asserted as a newly audited original-file
transcription. No historical GPU evidence is used to claim a pass.

## Verdict and suggested ledger change

**Discrepancy.** Conditional minimum selection, rounding, cap provenance,
identity inclusion and measured-policy bypass match. The stronger universal
support/rejection premise fails at coherent coefficient and output-axis
refraction evaluation. The cap also inherits the unresolved nominal-energy
scope of the separate kinematic-bandwidth claim.

Suggested owner edit: set `line-grid-emission-ceiling` to `discrepancy`, link
this document, preserve the passing arithmetic anchors, and state the exact
coherent NaN-suffix and static-potential convention differences. Do not claim
physical emission vanishes at the dataset endpoint or nominal beam energy.
Remediation requires an owner decision: scope the claim to the arithmetic and
explicitly supported routes, or implement and independently verify a coherent
out-of-domain evaluation/rejection policy, including upward-rounded axis
coordinates. Correct the static-potential support description and retire the
obsolete CUDA guard assertion. Preserve the separate bandwidth discrepancy;
any energy-distribution bound or recoil model needs its own verification.
No implementation, test or ledger repair was made here. Agent status is never
human sign-off.

## Post-verification remediation (implementation context)

Applied after this verdict by the #1101 task owner; not part of the
independent derivation above and not yet re-verified in fresh context.

- A binding atomic-data ceiling now rounds **down** to the largest 100 eV node
  strictly below the onset; kinetic and resonance limits still round up.
  HOPG at 5, 10 and 50 MeV now stops at 966200 eV, so output-axis refraction
  stays inside its support. Those automatic cases change identity.
- Coherent per-hkl (`_formation_good`) and batched
  (`_batched_coherent_block`) predicates now also require finite complex
  polarization coefficients, covering the final-root crossing.
- The ledger claim and `_cxr_table_nan_onsets` docstring no longer group
  static `U_g` with anomalous support or cite the absent CUDA guard.

The nominal-energy scope inherited from `line-grid-kinematic-bandwidth` is
unchanged.

## Independent re-verification (fresh context, 2026-11-11, post-remediation)

Fresh context, worktree `issue-1101-ceiling-reverify` at `e45ebefbd`. Read
first: the ledger row's Claim/Code/Source/Checks/Anchor and the signatures
and docstrings of `campaign/sweep.py::_automatic_cxr_grid_policy`,
`_kernels.py::_cxr_emission_ceiling`, `::_cxr_table_nan_onsets` and
`_per_hkl.py::_formation_good`. The derivation below was fixed before the
implementation bodies and the earlier sections of this page were read. The
nominal-energy premise of `line-grid-kinematic-bandwidth` is a separate, known
discrepancy and is not adjudicated here; the energy-conservation cap below
inherits it (an energy-spread beam has electrons above the nominal $T$).

### Independent derivation

Let $T$ be the nominal beam kinetic energy, $R$ the supplied resonance bound,
and $\Delta=100$ eV the lattice quantum. For a monoenergetic electron with no
external energy input, energy conservation gives $E_\gamma\le T$.

Chantler/FFAST anomalous factors are evaluated on the strict interior
$(E_{\min},E_{\max})$ of each element's tabulated grid; at and above
$E_{\max}$ they are NaN. They enter the anomalous susceptibility
$\chi_{\mathbf g}$ and the refractive index $\operatorname{Re}n$, so both are
finite exactly on $E<D$, with $D=\min_{\rm basis}E_{\max}$ when
anomalous dispersion is on for all basis elements. The static potential
$U_{\mathbf g}$ uses only $f_0$ and stays finite past $D$. The EPDL2025
attenuation is finite on $E\le 10^{11}$ eV and never binds here. Hence

$$
E_{\rm data}=\min\Bigl(D,\ \operatorname{nextafter}(10^{11}\ {\rm eV},+\infty)\Bigr).
$$

Each limit has its own correct rounding direction. A bound used only to keep
physical emission on the axis, $T$ or $R$, may round up: nodes above it are
empty. A data-support bound must round down to the largest lattice node
strictly below the open endpoint, because the coherent and flight-grouped
routes evaluate $\delta(E)\,\omega(E)=[1-\operatorname{Re}n(E)]\,E/\hbar c$
at every output node, independent of any line centre. So

$$
\boxed{
S=\min\Bigl(\Delta\lceil R/\Delta\rceil,\
\Delta\lceil T/\Delta\rceil,\
\Delta\bigl(\lceil E_{\rm data}/\Delta\rceil-1\bigr)\Bigr)
}
$$

and $S<E_{\rm data}$ always, since the third term is strictly below it.
Rounding must be applied to each limit **before** taking the minimum. Rounding
the minimum of the raw limits differs whenever a raw $T$ or $R$ in
$(\Delta(\lceil E_{\rm data}/\Delta\rceil-1),\,E_{\rm data})$ binds: it rounds
up to $\Delta\lceil E_{\rm data}/\Delta\rceil\ge E_{\rm data}$.

Line rejection: a line whose centre reads a NaN coupling must not enter any
field or intensity sum. For the incoherent routes that means rejecting
nonfinite total weight. For coherent routes it means rejecting nonfinite
complex polarization coefficients $c_e=\sqrt{\alpha\omega/(4\pi^2\hbar c)}\,t_L A_e$,
because the root iteration can finish beyond $D$ after finite reads, leaving
a finite scalar amplitude and a NaN $\chi_{\mathbf g}$.

### Implementation comparison

`_cxr_table_nan_onsets` returns the $\chi_{\mathbf g}$ onset, the
$\operatorname{Re}n$ onset and one EPDL onset per absorber. For default HOPG
these are $[966266.74,\,966266.74,\,10^{11}{+}]$ eV, and
`_cxr_emission_ceiling` returns their minimum. This matches $E_{\rm data}$.
The docstrings no longer group static $U_{\mathbf g}$ with anomalous support.

`_automatic_cxr_grid_policy` takes the already rounded-up resonance stop as
`policy.stop_eV`. It then forms

```text
limits = {"energy conservation": energy_keV * 1e3,
          "atomic data ceiling": _cxr_emission_ceiling(...)}
label, cap = min(limits.items(), key=...)
steps = math.ceil(cap / STOP_ROUND_TO_EV)
if label == "atomic data ceiling":
    steps -= 1
cap = steps * STOP_ROUND_TO_EV
if not start < cap < policy.stop_eV: return policy
```

Comparing the resonance term after rounding is equivalent to the derivation
(a rounded $R$ above $E_{\rm data}$ is replaced by the rounded-down data
node). The data branch matches, including the exact-lattice case
($\lceil\cdot\rceil-1$ is strictly below). **The kinetic term differs:** the
code chooses between $T$ and $E_{\rm data}$ on raw values and rounds the
winner. If $\Delta(\lceil E_{\rm data}/\Delta\rceil-1)<T<E_{\rm data}$, the
label is `energy conservation` and the cap rounds up to
$\Delta\lceil T/\Delta\rceil\ge E_{\rm data}$. This leaves a node at which
$\operatorname{Re}n$ is NaN. For carbon-limited crystals the window is
$966200<T<966266.74$ eV. The resonance bound must also exceed that rounded cap,
which takes $|\mathbf g|_{\max}\gtrsim 32\ {\rm \mathring A}^{-1}$ at
$T\approx 966$ keV, a high-order reflection set. This is reachable but
narrow.

Rejection masks:

- Per-hkl coherent and grouped rows use `_formation_good`, which now also
  requires `isfinite` on every complex coefficient.
- `_batched_coherent_block` applies `good &= isfinite(c_s) & isfinite(c_p)`.
  It compacts every per-row record through that mask before the CPU or CUDA
  coherent reductions, so the kernels never see a rejected row.
- The incoherent per-hkl and batched paths mask
  `isfinite(weight) & (weight > 0)`. The fused CUDA elementwise kernel applies
  the same test. A NaN $\chi_{\mathbf g}$ poisons the total $|A|^2$, so the
  test also drops the component spectra.
- The temporal coherent rows reuse `good`.
- Table interpolation (`_interp_gather1d`) clamps above the last mesh node.
  The setup comment and `_cxr_setup_ceiling` say the mesh keeps its first NaN
  node, so a final root beyond $D$ reads NaN $\chi_{\mathbf g}$, and the
  masks above reject it. The mesh construction itself was not independently
  re-checked here.

The new anchor `test_coherent_formation_mask_drops_nonfinite_coefficients`
covers `_formation_good` only; the batched mask is verified here by
inspection and by the route-forced probe below.

`resonance-population` policies return before the two caps, as stated. This
verification does not show that a measured axis stays inside $E_{\rm data}$,
and the claim does not assert it.

### Numerical evidence (CPU, project environment, no transport)

Independent oracle: $D$ from raw `xraydb.chantler_energies("C")`,
$m_ec^2=510998.95$ eV, $\hbar c=1973.269804$ eV Å, $c=6.711$ Å, $R$ from
$\hbar c\,\beta|\mathbf g|_{\max}/(1-\beta)$, and $S$ from the boxed
formula. The policy was called through a sweep stub with
`cxr_grid_policy=None` and `composition=[("C", 0.1)]`.

| Case | Oracle $S$ (eV) | Implemented stop (eV), source | Stop $<D$ |
| --- | ---: | --- | --- |
| HOPG (004), 3 MeV | 686700 | 686700, resonance | yes |
| HOPG (004), 5 MeV | 966200 | 966200, atomic data ceiling | yes |
| HOPG (004), 10 MeV | 966200 | 966200, atomic data ceiling | yes |
| HOPG (004), 50 MeV | 966200 | 966200, atomic data ceiling | yes |
| HOPG (0 0 40), 966.19 keV | 966200 | 966200, energy conservation | yes |
| HOPG (0 0 40), 966.21 keV | 966200 | **966300**, energy conservation | **no** |
| HOPG (0 0 40), 966.25 keV | 966200 | **966300**, energy conservation | **no** |
| HOPG (0 0 40), 966.30 keV | 966200 | 966200, atomic data ceiling | yes |

Support check at $E=[966200,\ D^-,\ D,\ 966300]$ eV:
`emission_coupling_tables` gives $\chi_{\mathbf g}$ finiteness
`[T, T, F, F]` and $U_{\mathbf g}$ finiteness `[T, T, T, T]`;
`refractive_index` gives `[T, T, F, F]`.

The scoring probe used one 5 MeV straight segment: length 10 Å, midpoint
$(4,0,5)$ Å, velocity $(0,0,1)$, a 10 Å slab, HOPG (004) and `B_ang2=0`.
The observation direction $\hat{\mathbf n}$ was solved from the vacuum target
$E_{\rm vac}$ as $n_z=(1-\hbar c\beta g_{004}/E_{\rm vac})/\beta$. The route
was forced by patching `_needs_per_hkl_route`. Outputs are
$d^2N/dE\,d\Omega$ per electron:

| Axis (eV), $E_{\rm vac}$ | batched or per-hkl, incoherent | batched or per-hkl, coherent |
| --- | --- | --- |
| $[966000,966100,966150,966200]$, 966150 | all finite, about $3.2228\times10^{-17}$ | identical, all finite |
| $[961000,963000,965000,966000,966200]$, 966150 | all finite, $3.2121$ to $3.2228\times10^{-17}$ | identical, all finite |
| same wide axis, 966300 (centre beyond $D$, inside the 1040 eV keep pad) | all `0` | all `0`: no NaN |
| $[966100,966200,966250,966300]$, 966150 | all finite | last node `nan` in both routes |

The new stop of 966200 eV therefore scores finitely in every route. A line
whose vacuum centre lies beyond $D$ is dropped end to end without NaN. That
row does not isolate which guard drops it: root rejection on a NaN $\delta$ or
the coefficient mask. A 966300 eV node, which the kinetic edge case above
emits, returns NaN in coherent scoring.

Focused anchors, run as
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test <nodes>`: the four
`tests/energy-grid/test_line_grid_policy.py` policy anchors and
`tests/montecarlo/spectrum/test_spectrum_phases.py::test_coherent_formation_mask_drops_nonfinite_coefficients`
gave **5 passed**. The ceiling anchor takes its ceiling from the production
`_cxr_emission_ceiling`. It asserts `ceiling - 100 <= stop < ceiling`, so it
does not exercise the kinetic-term window.

### Filters

| Filter | Result |
| --- | --- |
| Units: beam keV to eV, onsets and quantum in eV | Pass |
| Limits: each cap binds when the others are removed; resonance-population bypass | Pass |
| Rounding direction for a binding data limit, including an exact lattice onset | Pass |
| Rounding direction for a binding kinetic limit within 100 eV below $E_{\rm data}$ | **Fail**: rounds up across the onset |
| Support conventions: open Chantler endpoint, closed EPDL endpoint, static $U_{\mathbf g}$ finite | Pass; claim text accurate |
| Coherent complex-coefficient rejection (per-hkl, batched, so CUDA inputs) | Pass by inspection and probe; anchor covers per-hkl only |
| Incoherent nonfinite-weight rejection | Pass |
| CUDA-guard wording | Pass: the claim no longer cites a kernel guard; it masks before the kernels |
| Nominal-energy premise | Inherited from `line-grid-kinematic-bandwidth`; not adjudicated |
| Tolerances | Exact domain and rounding checks; no fitted tolerance |

### Verdict

**Discrepancy, narrowed.** The remediation fixes the case that failed before.
A binding data limit now gives HOPG 966200 eV at 5, 10 and 50 MeV, and
coherent scoring is finite there. Both coherent predicates drop nonfinite
complex coefficients, and the claim text on static $U_{\mathbf g}$ and the
CUDA kernels is accurate.

The first divergent convention is the order of minimum and rounding in
`_automatic_cxr_grid_policy`. The code takes $\min(T,E_{\rm data})$ on raw
values and rounds the winner. When $T$ binds within one quantum below
$E_{\rm data}$, it rounds up to a node at or above the NaN onset (966300 eV for
$966200<T<966266.74$ eV). That breaks the claim that the automatic stop stays
strictly inside the refraction support evaluated on the output axis.

Suggested remediation (owner):

- Round each limit by its own rule before the minimum:
  $\min(\Delta\lceil T/\Delta\rceil,\ \Delta(\lceil E_{\rm data}/\Delta\rceil-1))$.
  Equivalently, after rounding the kinetic cap, also clamp it to the
  rounded-down data node, and record whichever limit binds.
- Add an anchor with $T$ in that window and a reflection set whose resonance
  bound exceeds it.
- Optionally, add a batched-route coefficient-mask anchor.

Re-verify afterwards. Suggested ledger edit: keep `discrepancy` and name this
kinetic-rounding window as the remaining divergence. The separate
`line-grid-kinematic-bandwidth` discrepancy is unchanged. No source, test or
ledger change was made here; this is not human sign-off.

### Owner follow-up to the post-remediation finding (implementation context)

Applied after the re-verification above; not yet re-verified in fresh
context. `_automatic_cxr_grid_policy` now rounds each limit by its own rule
before taking the minimum: the kinetic limit rounds up, the data limit to the
largest node strictly below its onset. A beam energy in
$(966200, 966266.74)\ \mathrm{eV}$ now stops at 966200 eV. Anchor:
`tests/energy-grid/test_line_grid_policy.py::test_kinetic_cap_just_below_the_data_onset_stays_below_it`.
The kinetic limit is now the declared upper incident energy
$E_0(1+6f)$ (see `line-grid-kinematic-bandwidth`).
