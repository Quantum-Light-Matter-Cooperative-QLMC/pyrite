# Mechanisms for phase retention

This survey separates candidates for a future omission certificate from the
[implemented cost gate](../../repo-design/compute/phase-retention.md). It proposes
no scattering-induced switch and no new production physics criterion.

## Keep the requested quantity explicit

For a fixed trajectory, summing amplitudes before squaring preserves its
interference. Averaging over a specified source ensemble can suppress particular
cross terms, but is a different quantity. Integrating over energy, angle or a
detector response is another operation whose domain and weights must be stated.
A reduction justified for one operation cannot be transferred to another merely
because the resulting plot appears smooth.

The primary-source
[bunch-radiation treatment](https://www.sciencedirect.com/science/article/pii/S016890020701861X)
expresses coherent enhancement through the Fourier transform of the specified
source distribution. That motivates analytic source averaging where the field
and source offsets factorize; it does not certify fixed-realization omission.
PyRITE's actual finite-footprint eligibility and tilted-face delay are documented
under [transverse bunch averaging](../../validation/radiation-physics/transverse-bunch-form-factor.md).

## Target in physically incoherent limits

The optimizer should stop computing interference terms when their contribution
to the requested observable is certified negligible and omitting them saves
work. The surviving intensity must be the one appropriate to that physical
limit. A small inter-electron contribution alone does not establish that all
coherent phase effects are small.

For the existing conditional Gaussian arrival-time average, increasing the
bunch duration drives the inter-electron factor to zero at positive photon
energy. The supported certificate tests the factor multiplied by the physical
pair population, $F(N-1)$, against the omission allowance. Thus a long bunch
can skip the complete cross-electron reduction while retaining the full
single-electron field. This statement is the existing
[validated flat-term criterion](../../validation/beam-transport/coherent-flat-term-omission.md),
not an additional source or transport approximation. A short bunch with a wide
recorded spot reaches the same endpoint through the row's transverse factor;
see the [transverse certificate](../../validation/beam-transport/coherent-transverse-flat-omission.md). Its analytic averaging
requires the finite-footprint eligibility contract; physical infinite slabs
use complete sampled fields instead and cannot reuse that factor.

| Physical limit | Surviving reduction | Decision boundary |
| --- | --- | --- |
| Independent broad arrival-time spread | Intensities of complete electron fields | Supported conditional Gaussian certificate; internal flight interference survives |
| Independent broad transverse position spread | Intensities of complete electron fields where the spatial characteristic function is small | Supported finite-footprint certificate on the row's directed $F_zF_\perp$ bound with the existing eligibility and population weight; spot size alone is insufficient (phase-matched rows stay coherent) |
| Averaging that suppresses interference between distinct physical flights | Intensities of complete flight fields | Needs a bound for every omitted flight-pair contribution over the specified ensemble or observation domain; not implemented |
| Both electron-pair and flight-pair interference negligible | Flight-grouped incoherent reduction | Desired fully incoherent limit, after both sectors satisfy a shared accuracy contract |

The final row retains the finite-time field integral within each physical
flight. Numerical subdivision does not create independent emitters: its
substep fields must still add before squaring. See
[physical flights and numerical substeps](../../physics/radiation-physics/coherent-radiation.md#physical-flights-and-numerical-substeps).
Changing to a per-substep intensity sum would make the result depend on
transport resolution.

Observation averaging can suppress a cross term through an oscillatory
integral. The [Fourier-transform definition and decay theorem](https://dlmf.nist.gov/1.14.i)
provide the mathematical starting point when the weighted amplitude product is
absolutely integrable and the phase is an appropriate Fourier variable.
Applying that theorem to the actual flight fields is a proposed derivation,
not a current certificate: narrow resonances, correlated amplitudes and
stationary or equal-phase pairs need explicit treatment. A broad observation
window alone supplies no finite-error bound.

The public-API regression
`test_bunch_spread_policy_preserves_the_correct_coherence_sector` exercises short
and long bunches on the same one- and two-flight trajectories, using physical
$N=10^8$. Short bunches retain the full cross-electron calculation; long bunches
skip every cross-electron energy node and reproduce an isolated electron's
coherent spectrum. The two-flight control remains distinct from the incoherent
policy. A one-flight constant-escape control reaches that policy; grooved
midpoint-versus-mean escape quadrature is not misclassified as interference.
Synthetic profitable costs exercise the decision gate only and establish no
runtime benefit. No intra-electron omission rule is enabled by these anchors.

## Candidate mechanisms

| Mechanism | Sector and useful scope | Existing information / missing contract | Omission supported now? |
| --- | --- | --- | --- |
| Initial longitudinal bunch spread | Inter-electron, evaluated energy | Gaussian RMS and sampled offsets exist; use the existing physical-population and footprint eligibility checks | Existing #362 certificate only |
| Initial transverse position spread | Inter-electron, energy and reflection/orientation/direction row | Tilted-face covariance and arrival delay exist; correlated slopes, energy spread and footprint selection are refused before the reducer | Finite footprint: [transverse certificate](../../validation/beam-transport/coherent-transverse-flat-omission.md); infinite slabs keep complete sampled fields |
| Angular or energy spread | Both sectors, conditional ensemble and row | Sampled initial states exist; joint correlations between amplitude, phase and trajectory need a specified ensemble | No |
| Multiple scattering / transport phase dispersion | Intra-electron flight/region and inter-electron ensemble | Segment phase exists after transport; random-looking phases do not bound the omitted cross terms | No |
| Formation and attenuation scales | Intra-electron piece/depth/energy | Formation integrals and escape data exist; require a bound on both removed fields and their interference with retained fields | No phase truncation rule |
| Energy-bin or detector/angular averaging | Either sector, complete weighted integration domain | Axes/directions/response weights exist; require domain-wide bounds and quadrature control, rather than point samples | No general rule |
| Geometry and finite observation distance | Either sector, row/direction/region | Footprint and far-field support checks exist; excluded geometry needs a new propagation model rather than an omission shortcut | No |

The [LPM derivation](https://arxiv.org/abs/hep-ph/9604327) is an explicit example
of multiple scattering entering a *coherent interference* calculation, with a
model-specific radiation consequence. It concerns bremsstrahlung in its stated
regime, not a transferable PXR/CBS omission certificate. It is evidence against
equating a scattering event with destruction of phase.

## Cross-boundary accounting

An early/late or shallow/deep split does not split intensity additively: the
retained and discarded complex fields interfere. Removing a weak field also
changes its cross term with the retained field. Treating both regions as
separate powers instead removes their mutual interference, even if neither
region's power is removed. Both proposals need a field-norm bound or an explicit
ensemble/integration estimate covering that cross term. Multiple partitions
need a shared error allocation; each pair may be charged only once.

The supported flat omission avoids this issue by retaining the *complete*
intra-electron amplitude in every grouped field and bounding the complete
inter-electron excess. It does not cut a trajectory into independent regions,
flights or segments. Constructive aligned-electron and destructive opposite-field
anchors exercise the two error signs under the existing certificate.

Future deterministic region certificates must start from the full coherent
field, include cross-boundary terms and preserve absorption/dispersion phase.
Statistical estimates need an explicit source/transport ensemble, confidence
contract and reviewed sampling model before use. Apparent phase dispersion,
a formation-length comparison, or separate-seed agreement supplies none of
those by itself. Independent source derivation and fixed-trajectory anchors
must precede activating a new criterion.
