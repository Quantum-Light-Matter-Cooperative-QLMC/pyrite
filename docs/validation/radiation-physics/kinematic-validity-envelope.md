# Kinematic validity envelope

Validation: `kinematic-validity-envelope`.

Independent verification for issue #342; derivation below was written before
inspection of the implementation body. The claim concerns a photon Born
screen, a resonant scalar two-beam exchange scale, and a thickness/absorption
ratio. It does not certify the full radiation model.

## Source, signature, and conventions

The source is [Feranchuk et al., *Physical Review E* **62**, 4225 (2000)](https://doi.org/10.1103/PhysRevE.62.4225),
photon eigenmode, as reproduced by [Zhai et al. (2025)](https://doi.org/10.1038/s41467-025-66063-6),
[supplementary information Eq. (3), p. 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-66063-6/MediaObjects/41467_2025_66063_MOESM1_ESM.pdf).
The locally supplied primary-source supplement was inspected independently;
the original Feranchuk full text was unavailable. Its reproduced diffracted
contribution is proportional to

$$
-\chi_{\mathbf g}
\frac{(\mathbf k+\mathbf g)(\mathbf g\cdot\hat{\mathbf e}_s)
-k^2\hat{\mathbf e}_s}{\lvert\mathbf k+\mathbf g\rvert^2-k^2},
\qquad k=\frac{\omega_{\rm physical}}{c}=\frac{E}{\hbar c}.
$$

`materials/crystal.py::kinematic_validity` receives `photon_E_eV`,
`chi_abs`, `detuning_inv_ang2`, `thickness_ang`, and
`absorption_length_ang`. Their units are eV, dimensionless, inverse square
angstroms, angstroms, and angstroms, respectively. Its local `omega`
denotes $k$, an inverse length, rather than angular frequency.

Assumptions: homogeneous crystal, vacuum photon wavevector,
scalar unit-polarization coupling, and a characteristic propagation length.
The reciprocal-vector convention in the source is explicitly $\mathbf k+\mathbf g$.
The diagnostic magnitude removes the susceptibility phase and detuning sign;
it does not remove the need to use the correct reciprocal-vector branch.

## Independent derivation

For $\mathbf k=k\hat{\mathbf n}$, expansion of the denominator gives

$$
\Delta=\lvert\mathbf k+\mathbf g\rvert^2-k^2
=g^2+2k\hat{\mathbf n}\cdot\mathbf g.
$$

Projection onto a transverse diffracted polarization removes the longitudinal
part of the numerator. The remaining scalar coupling scale is
$k^2\lvert\chi_{\mathbf g}\rvert$ when the polarization overlap is set to one.
The dimensionless Born mixing diagnostic is consequently

$$
D=\frac{k^2\lvert\chi_{\mathbf g}\rvert}{\lvert\Delta\rvert}.
$$

This is a scalar scale for the perturbative eigenmode correction, not the exact
vector correction or a relative error in emitted intensity.

Independently, inserting slowly varying plane-wave envelopes into the scalar
Helmholtz equation gives a first derivative term $2ik\,\partial_s A$.
The resonant off-diagonal term is $k^2\chi_{\mathbf g}B$, so the magnitude
of the coupled-wave rate is

$$
\kappa=\frac{k\lvert\chi_{\mathbf g}\rvert}{2},
\qquad L_{\rm ext}=\frac{1}{\kappa}
=\frac{2}{k\lvert\chi_{\mathbf g}\rvert}.
$$

In the lossless symmetric two-beam reduction the transferred intensity varies
as $\sin^2(\kappa s)$; $L_{\rm ext}$ therefore means an inverse exchange
rate, not a full Pendellösung period or the length for complete transfer.
Polarization factors, asymmetric direction cosines and refractive detuning
would change the physical coupled-wave problem; this diagnostic omits them.
The Born source equation alone does not supply a finite-thickness dynamical
solution.

Using absorption as a characteristic intensity attenuation length, the chosen
available-length screen is

$$
L_{\rm eff}=\min(t,L_{\rm abs}),
\qquad R=\frac{L_{\rm ext}}{L_{\rm eff}}.
$$

The minimum is a policy approximation to the available distance, not an exact
absorption-weighted path integral. With the stated scalar assumptions the units
are $[D]=[R]=1$ and $[L_{\rm ext}]={\rm \mathring A}$.

Zero coupling gives $D=0$ and infinite exchange length, including at zero
detuning. Nonzero coupling at exact zero detuning gives infinite $D$; the
Born pole indicates breakdown of this expansion, not divergent physical
intensity. Zero thickness gives infinite $R$. Infinite absorption length gives
$L_{\rm eff}=t$. Increasing thickness beyond the absorption length leaves
$R$ unchanged. Reversing detuning sign preserves the three scalar results.
Reversing $\mathbf g$ alone generally changes $\Delta$; reversing both
$\hat{\mathbf n}$ and $\mathbf g$ preserves it.

## Independent numeric anchors

Before implementation inspection, choose $E=1973.269804\ {\rm eV}$ and
$\hbar c=1973.269804\ {\rm eV\,\mathring A}$, hence
$k=1\ {\rm \mathring A}^{-1}$. With
$\lvert\chi_{\mathbf g}\rvert=10^{-5}$,
$\Delta=0.002\ {\rm \mathring A}^{-2}$,
$t=100000\ {\rm \mathring A}$ and
$L_{\rm abs}=200000\ {\rm \mathring A}$, the independent results are
$D=0.005$, $L_{\rm ext}=200000\ {\rm \mathring A}$ and $R=2$.

Changing only $\Delta$ to $0.0005\ {\rm \mathring A}^{-2}$ gives
$D=0.02$ with $R=2$. Changing the latter case to
$t=400000\ {\rm \mathring A}$ and
$L_{\rm abs}=300000\ {\rm \mathring A}$ gives $R=2/3$.
Thus a mixing-only flag is distinguishable from simultaneous mixing and
reachable-exchange flags without material-data assumptions.

## Runtime policy boundary

The stated warning predicate is the conjunction $D\ge0.01$ and $R\le1$.
These thresholds are project policy, not constants derived from the paper.
The first tests coupling relative to detuning; the second says only that a
resonant exchange length fits within the characteristic available distance.
Neither condition alone establishes a complete validity certificate.

The intended adapter retains the offline audit's positive-$\mathbf g$
emission resonance and plus-$\mathbf g$ photon detuning. From the source phase
$\omega_{\rm physical}-\mathbf v\cdot(\mathbf k+\mathbf g)=0$ the resonance
energy is

$$
E_p=\hbar c\frac{\boldsymbol\beta\cdot\mathbf g}
{1-\boldsymbol\beta\cdot\hat{\mathbf n}}.
$$

Positive energy therefore selects positive $\boldsymbol\beta\cdot\mathbf g$
for this branch. This is convention preservation for the current audit, not
a redesign of other emission-kernel signs. A central incident ray and scalar
detector direction cannot establish validity over electron scattering,
mosaic domains, energy spread, detector acceptance, or every electron flight.

## Implementation comparison

After the derivation was written, the helper body and runtime adapter were
inspected. `omega = photon_E_eV / HBARC_EV_ANG`, the squared-wavevector
numerator, absolute detuning, factor two in the exchange length, and
`min(thickness_ang, absorption_length_ang)` match term by term. The helper
correctly resolves zero coupling before the zero-detuning branch and handles
zero thickness and positive infinite absorption length. Its input checks
reject nonfinite energy, coupling, detuning and thickness, negative coupling
or thickness, nonpositive energy, and nonpositive or NaN absorption length.

The three independent numeric points above produce respectively
$D=0.005$, $D=0.02$, and $D=0.02$;
$L_{\rm ext}=199999.99999999997\ {\rm \mathring A}$ at all three points;
and $R=1.9999999999999998$, $R=1.9999999999999998$, and
$R=0.6666666666666665$. Differences from the hand values are roundoff.

`campaign/kinematic_validity.py::_radiator_optics` rotates the reciprocal
vector with the resolved crystal orientation, computes the positive-energy
resonance from the central beam and observation directions, and uses the
source's plus-reciprocal-vector detuning. The susceptibility modulus and
absorption length are evaluated at that photon energy. Nonpositive resonances
and optical-data misses remain explicit records. `case_kinematic_validity`
deduplicates selected reflections and uses each crystalline layer's own
thickness; amorphous layers are omitted. This is a thickness screen, not an
oblique photon escape-path calculation. Infinite diagnostics become JSON null
while decision booleans remain explicit.

The adapter subsequently batches each radiator's optical queries. Rows of
reciprocal vectors are rotated by `g @ rotation.T`, equivalent to applying
`rotation @ g` separately. The ordered positive-resonance index list selects
both coupling-table rows and energy columns; the table diagonal consequently
pairs each reflection with its own resonance. The same positive mask restores
those results to their original reflection positions. Vector detuning retains
the squared reciprocal norm and the plus-sign cross term. Energies remain in
eV and absorption lengths in angstroms.

The final dependency split caches these thickness-independent inputs in
`_radiator_optics`, keyed by crystal, ordered reflections, Debye–Waller factor,
incident energy, and the complete resolved orientation/direction parameters.
Thickness is intentionally absent from that key: the homogeneous bulk
susceptibility, attenuation coefficient, vacuum resonance and detuning do not
depend on thickness under the stated model. `_radiator_audit` copies each
cached optical record and applies `kinematic_validity` with the current
thickness. Thus the exchange ratio and its policy flag are recomputed without
mutating cached optical records or retaining another layer's thickness. The
split changes neither equation nor the interpolation context for a fixed
ordered reflection set.

An independent mixed-sign probe of silicon reflections `(220)`, `(-2-20)`,
`(440)`, `(222)` and HOPG `(002)`, `(00-2)`, `(004)` confirmed record order,
negative-resonance status, and equality of scalar and batched absorption
lengths. The allowed-reflection susceptibility magnitudes differed from the
scalar optical query by at most $1.25\times10^{-9}$ relatively in this probe.
This comparison is not an exact-roundoff guarantee: scalar and vector optical
queries can condition the underlying interpolation differently, depending on
their energy spans. The batched adapter uses the existing production coupling
table route. Extinct silicon `(222)` additionally illustrates cancellation:
susceptibilities are of order $10^{-20}$ and regrouping the basis sum changes
them by about $2.3\times10^{-21}$ absolutely, so relative errors there are
misleading. Neither effect changes the scalar screen's equation, reciprocal
sign convention, units, or array correspondence.
The scalar-versus-batched optical comparison does not guarantee identical
policy flags for inputs arbitrarily close to a threshold.
The task owner's separate 12-case diamond comparison found a larger optical
context effect: the `(11-1)` reflection's extinction ratio changed from
$6.44683$ in scalar queries to $6.48335$ in batched queries, a relative
difference of approximately $0.005665$. All flags agreed in those sampled
cases. These owner-supplied results were not independently reproduced here;
they bound the scalar-query equivalence claim rather than change this
derivation. In particular, scalar optical agreement at the silicon/HOPG
points above must not be generalized to the low-energy carbon-edge region.

`warn_kinematic_validity` applies the required conjunction; a reachable
exchange length alone causes no warning. Tests cover the scalar analytic
anchor, zero/transparent limits, a Bragg-matched thick silicon photon,
thin HOPG, resolved orientation, crystalline-layer thickness, standard-profile
warning behavior, duplicate reflections, nonpositive resonances, JSON
provenance, and the extinction-only off-Bragg case. The material silicon/HOPG
tests consume production optical-data helpers and therefore anchor the screen's
integration rather than independently validate those optical tables.

**Verdict: rederived.** Units, limits, and signs/conventions pass within the
stated scalar vacuum convention. No divergent factor, sign, exponent or unit
was found. Suggested owner ledger change: `unverified` to `rederived`; the
owner may advance to `anchored` after recording passing named regression
anchors. This verdict does not certify full-flight or acceptance validity,
quantify spectral error, or grant human sign-off.

## Verification commands

`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/materials/test_kinematic_validity.py tests/scan/test_kinematic_validity.py -k 'not standard_profile'`
passed: 13 tests, 50 catalog-profile parametrizations deselected. The task owner
owns the full catalog audit. `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/dev/test_docs.py`
passed: 6 tests.
