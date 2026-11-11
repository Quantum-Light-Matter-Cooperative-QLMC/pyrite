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
