# Optical material response

## Independent derivation

Validation: `optical-material-response`. This derivation was written in a fresh
verification context before inspecting the implementation body. The quantity is
the dimensionless, local, isotropic relative permittivity at positive photon
energy in eV. Magnetic response is assumed to be unity. The time convention is
$\exp(-i\omega t)$; only passive media are admitted.

For a plane wave propagating in the positive spatial direction, write its
complex refractive index as $N=n+ik$. The field has the factor

$$
\exp(iN\omega z/c-i\omega t)
=\exp(-k\omega z/c)\exp(in\omega z/c-i\omega t).
$$

Thus $k\geq0$ gives attenuation rather than gain. Maxwell's dispersion relation
for scalar relative permeability equal to unity gives

$$
\boxed{\epsilon=N^2=(n^2-k^2)+2ink.}
$$

Both $n$ and $k$ are dimensionless. For nonnegative $n,k$, the imaginary part is
nonnegative, as required by nonnegative cycle-averaged dissipated power,
$\omega\epsilon_0\operatorname{Im}(\epsilon)|\mathbf E|^2/2$.
In vacuum $n=1$, $k=0$, so $\epsilon=1$ exactly. The propagating positive-permittivity
lossless limit has $k=0$ and $\epsilon=n^2$; zero dissipation also permits $n=0$,
$k>0$, and $\epsilon=-k^2$, where spatial decay is evanescent rather than
absorptive. The implementation accepts both cases because
$\operatorname{Im}(\epsilon)=2nk=0$; negative real permittivity must not be rejected.

For tabulated vacuum wavelength in micrometers, the conversion is

$$
E_{\rm eV}=\frac{hc}{10^4\lambda_{\mu\mathrm m}},
$$

where `HC_EV_ANG` supplies $hc$ in eV angstrom. Increasing wavelength reverses
the energy ordering; interpolation therefore requires increasing energy after
conversion. At neighboring energy knots, the stipulated interpolation is

$$
w=\frac{E-E_j}{E_{j+1}-E_j},\qquad
\epsilon(E)=(1-w)\epsilon_j+w\epsilon_{j+1},\qquad 0\leq w\leq1.
$$

This is componentwise linear interpolation of permittivity, after squaring the
index, rather than linear interpolation of the index followed by squaring.
Weights are dimensionless. Convex interpolation preserves nonnegative imaginary
part and exactly reproduces constant vacuum response. Explicit disjoint closed
supported intervals restrict which pairs of knots may be joined. A query in a
gap or outside support has no defined value. Rejecting it prevents implicit
extrapolation. No conclusion about Kramers–Kronig consistency follows from a
piecewise linear finite-band interpolant alone.

## Source and scope

The provider uses the [pinned Rakić LD Au database record](https://github.com/polyanskiy/refractiveindex.info-database/blob/c5c2f188e848453def5970e347399d653df2ffc2/database/data/main/Au/nk/Rakic-LD.yml),
revision `c5c2f188e848453def5970e347399d653df2ffc2`, source SHA256
`a91fca57ee99efc77354908fa11b9c4822e8852394042069caaac7610c52e602`.
Its reference is [Rakić et al., Applied Optics 37, 5271–5283 (1998)](https://doi.org/10.1364/AO.37.005271).
The public [record-generation script](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-scripts/570e9d272ac7b164c8e67dbbade8a006684984d7/scripts/Rakic%201998%20-%20Au%20(LD%20model).py)
uses a Drude term plus five damped Lorentz oscillators, with positive damping
and oscillator strengths and denominators consistent with this time convention.
It samples 200 energies from 0.2 to 5 eV, then rounds the wavelength and index
columns. Therefore supported endpoints must follow the actual rounded table
and current conversion constant, rather than exactly 0.2 and 5 eV.

This is a model fit to multiple measurements, not an individual measurement
series. Source temperature is unspecified. The database record is CC0. This
claim covers conversion and bounded interpolation of that optical record; it
does not establish tensor, nonlocal, active-medium, X-ray, or production-model
validity. The paper itself was inaccessible through the browsing tool; the
public source-generation script was directly inspected. No paper equation
number is asserted without access to the full paper.

The final metadata contract was compared with the stage-1 Optical/IR material
response specification. It records a dataset identifier, pinned version/hash,
original reference, license, original wavelength/index units, normalized energy
units through the API, complex sign convention, temperature, orientation,
sample state, supported intervals, and uncertainty. The Au provider identifies
the DOI and source units explicitly; temperature and uncertainty are unknown,
and orientation is inapplicable to its isotropic model. Unknown uncertainty is
not a claim of zero error. These fields participate in the response digest.
No metadata requirement gap was found for this bounded scalar Au provider.
Phonon/tensor datasets, sample-specific film/substrate models, and a tested
optical–X-ray splice remain outside its declared support. The finite-band
causality limitation stated above remains applicable.

## Implementation comparison

The comparison below was performed after writing the independent derivation above.

The inspected definitions are
`src/pyrite/materials/optical.py::OpticalResponse`, its `epsilon` method,
`_validate_segment`, and `gold_rakic_response`. The provider uses the required
$10^4$ wavelength conversion, sorts energy increasingly, and computes the
complex square before interpolation. The segment validator requires finite,
positive, strictly increasing energy, matching scalar ordinates, strict
separation from the previous closed interval, and finite nonnegative imaginary
permittivity. The response rejects tensor symmetry and the opposite time
convention. Its query masks include endpoints and leave unsupported energies
as nonfinite sentinels, which are rejected before backend upload. Separate
real/imaginary `np.interp` calls exactly implement the derived convex expression.
The final conversion uses `xp.asarray(..., dtype=xp.complex128)`.

The shipped record contains 200 rows and its SHA256 matches the pinned source
hash above. Its first row has wavelength $0.24797\,\mu\mathrm m$,
$n=1.0941$, and $k=1.5478$. Direct arithmetic independently gives

$$
\epsilon=-1.19863003+3.38689596i.
$$

The provider's explicit complex square matches that value algebraically. Its
last wavelength is $6.1992\,\mu\mathrm m$; neither endpoint should be silently
rounded to an ideal energy endpoint. For an independent interpolation check,
knots $\epsilon_1=1+2i$ and $\epsilon_2=3+4i$ at energies 1 and 3 eV give
$\epsilon(2\,\mathrm{eV})=2+3i$. The inspected implementation has precisely
these coefficients. This is a symbolic and source-data arithmetic comparison;
no implementation execution or device equivalence test was performed here.

No divergent sign, factor, unit, or interpolation convention was found.
The independent review establishes `rederived`. The executed reference and
refusal regressions below support the integrated
ledger status `anchored`. The source-to-code review and execution evidence are
separate; the primary owns the latter.
Regression execution and rendered-math checks belong to the primary task owner
to avoid shared environment races. The provider, class, and interpolation method
carry the `Validation:` marker.
Only a human may apply `signed-off`.

## Final boundary and anchor review

After the tests became available, their reference formula was compared with the
public source-generation script. `_rakic_au` reproduces the plasma energy,
Drude strength/damping, all five Lorentz strengths/resonances/dampings, and the
negative imaginary damping sign of the oscillator denominators. This reference
is independent of the production table reader and interpolator, although both
represent the same fitted model; it cannot establish experimental fit accuracy.
The source uses scientific notation with four decimal places, hence five
significant digits in its tabulated columns. Source rounding and linear
interpolation error justify a finite tolerance. The 0.002 maximum
complex-relative-error gate over 2001 energies is a reasonable interpolation
anchor, strengthened by the separate stride-4/2/1 convergence test. Successful
execution is reported by the primary owner. Tests also exercise
closed endpoints, gaps, vacuum, active/tensor rejection, and scalar/array shape.

`case_layer_eps` now requires an explicit response for every layer when optical
responses are supplied. It preserves layer order and propagates unsupported-band
failures with `transition_radiation/layer[index]` context; it cannot replace a
failed optical lookup with an atomic table. Without explicit responses, it
verifies the existing per-medium X-ray
continuum floor before calling `composition_dielectric`. This prevents optical
material behavior from being inferred solely from atomic composition.
`case_transition_spectrum` passes the explicit layer values into its existing
TR scorer without changing their physical meaning. This is an explicitly
selected source-only seam, not evidence for production optical transport.

`ComponentSupport` and `component_grid` keep supported bands specific to each
required medium/component pair. Missing attenuation, bremsstrahlung, atomic,
or detector response cannot inherit the TR support declaration. Node evaluation
requires support only at nodes; bin representation additionally requires each
entire canonical midpoint bin to lie within one supported interval. Both
contracts correctly reject the gaps relevant to their respective representations.
No additional physics discrepancy was found in these final boundaries.

## Executed anchors and CUDA evidence

The primary ran 105 affected CPU tests (five existing skips) covering optical
response, supported bands, the TR runner, its DTR consumer and continuum-floor
regressions. A further 39 documentation/ledger/record tests passed. The strict
docs build passed; the rendered math contains no raw dollar delimiters.
The final metadata update additionally passed 58 material/grid/TR-runner tests. The wheel includes the unchanged optical record with the pinned hash.

On 2026-10-21 UTC, `pyrite remote sync` staged clean implementation checkpoints.
The exact tested revisions are preserved in the machine-readable records.
The check ran through PyRITE remote transport in a bounded three-minute,
one-GPU SLURM allocation, with the sync lock held shared. The staged revision
and clean-tree stamp were checked before executing
`pyrite-dev validation-records --output-dir jobs/issue-614-optical-response -- checks/optical_response_cuda.py`.
The [passing record](../check-records/optical_response_cuda.jsonl) names the full
source revision; [detailed measurements](../check-records/optical-response-cuda.json)
record the device, runtime, source hash, response digest and tolerances.

The NVIDIA GeForce RTX 5080 (compute capability 12.0, CuPy 14.2.0,
CUDA runtime 13.2) reproduced the 513 complex128 response values bit for bit.
This checks host interpolation followed by device upload, not device-side
interpolation. The response digest also agrees between the local ARM64 host and
the remote host. The planar optical slab's CPU/CUDA maximum absolute difference,
normalized by the CPU peak, was $1.61\times10^{-15}$. The established float64
TR parity criterion is relative tolerance $10^{-10}$ plus a peak-scaled absolute
floor $10^{-12}$, allowing LAPACK/cuSOLVER rounding in cancelled fields.
Unsupported 0.01 eV, 6 eV and NaN queries, a declared internal gap, tensor
symmetry and the missing `Au/eaglexo_qe` capability all failed closed.

This evidence anchors the bounded optical-record provider and its upload/refusal
path. It does not validate the fitted model against a new experiment or supply
optical transport, detector QE or arbitrary-band tensor response.
