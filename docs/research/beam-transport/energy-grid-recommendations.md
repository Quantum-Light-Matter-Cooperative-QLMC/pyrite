# Recommendation: energy grids across six decades

**Status:** Research recommendation, September 2026. Not implemented or an
extension of the validated physics range. Numerical examples are arithmetic
estimates, not transport benchmarks.

## Recommendation

Adopt logarithmic kinetic-energy grids for broad transport lookup tables and
logarithmic photon grids for smooth continua. Retain fine local energy windows
for narrow radiation lines, and define detector channels independently of source
evaluation nodes. Make interpolation and observable error the acceptance criteria;
a fixed bin count is only a resource constraint.

Changing grids does not make the current transport models valid at 100 MeV. See
[relativistic transport research](../physics/relativistic-electron-transport.md)
and the [transition-radiation recommendation](../physics/transition-radiation-recommendations.md).

## Current implementation

The catalog already accepts `values`, `arange`, `linspace`, and `logspace` in
`src/pyrite/materials/_catalog_decode.py`. The existing `logspace` descriptor uses
NumPy's exponent-based start/stop convention. The compact codec in
`src/pyrite/_energy_grid_encoding.py` preserves exact nonuniform arrays when a
legacy linear triple cannot reproduce them. These capabilities do not establish
end-to-end nonuniform-grid support.

`detectors/spec.py::EnergyBins` already separates a fine line grid, optionally
selected per incident beam energy, from a wide bremsstrahlung grid. Keep this
useful separation. Incident sweep energies, transport table nodes, radiation
evaluation nodes, and detector channel edges have different accuracy requirements.

The transport LUT in `montecarlo/transport/lut.py` targets 0.025 keV spacing,
with 256–16,384 points. Above the point cap, it silently increases the spacing.
For an illustrative 0.1–100,000 keV interval:

| Construction | Approximate size or spacing |
| --- | --- |
| Uniform 25 eV spacing | 4 million points |
| Current 16,384-point cap | 6.1 keV spacing, even near the low-energy cutoff |
| Geometric spacing with adjacent ratio at most 1.01 | 1,390 points |

This is a numerical scaling comparison only. The last row does not guarantee a
particular interpolation error. The capped linear grid loses precisely the
low-energy resolution needed when a high-energy electron slows down.

## Proposed grid policies

| Purpose | Policy | Required special treatment |
| --- | --- | --- |
| Incident beam-energy sweep | Geometric baseline | Add thresholds, model joins, and promising regions |
| Transport LUT | Uniform log kinetic energy | Validate off-grid rates, probabilities, stopping, and timing |
| Broad photon continuum | Geometric baseline | Refine near material edges and kinematic endpoints |
| PXR/CBS spectrum | Fine local windows | Resolve intrinsic lines and relevant interference fringes |
| Characteristic spectrum | Integrate known line profiles over bins | Preserve line mass and physical window truncation |
| Detector output | Instrument-defined channel edges | Apply response to integrated input-bin photons |

### Transport tables

Use a uniform coordinate in log kinetic energy with positive bounds. Indexing
then needs one logarithm, multiplication, and integer conversion rather than a
binary search. Update both `_lut_index_frac_scalar` and the corresponding
`_jit_device.py::_lut_lerp_at` path, including launch metadata and midpoint/cutoff
lookups. Preserve endpoint behavior consistently across backends.

Start a convergence study at 128 intervals per decade, comparing 256 and 512.
These are trial resolutions, not proposed universal defaults. For smooth positive
rates, compare interpolation of log values against interpolation of values in
log energy. Probabilities and cumulative probabilities need positivity,
monotonicity, and exact terminal normalization; do not blindly log every field.
Use local refinement or separate intervals around model joins, and never smooth
across a genuine physical discontinuity. If the memory cap prevents the requested
accuracy, report the unmet tolerance rather than silently coarsening.

This table change does not control integration error along a flight. Check
`energy_model`, `max_dE_frac`, and stopping/cutoff integration independently.

### Photon evaluation and detector bins

Use explicit bin edges for histogram quantities and explicit nodes for sampled
spectral functions. A density in photons/eV remains a density in photons/eV on a
log grid. Integrate in physical energy with local widths; integration in log
energy requires the energy Jacobian. Avoid treating a node-centered trapezoidal
integral and a sum of bin-integrated line masses as interchangeable conventions.

A logarithmic grid cannot contain zero. Choose a physically justified positive
photon floor from the modeled band and data support; if a zero-based detector
channel is needed, represent it with a separate explicit edge. Do not use an
arbitrarily small numerical epsilon as an unexamined infrared cutoff.

Retain narrow line windows with initially at least 8–10 samples across the
narrowest feature whose shape matters, then verify convergence. This count is a
starting heuristic, not a proof. If only integrated yield is needed, analytic
bin integration can be cheaper. A coarse pilot mesh alone can miss a narrow
resonance, so seed windows from kinematics and known atomic lines. Insert
appropriate one-sided edge information without duplicate energy coordinates.

Keep detector response and output channels separate from the source mesh. Rebin
conservatively using integrals, account explicitly for photons outside the
output window, and include the complete resolved grid in cache identity.

## Required consumer changes

Paths below are relative to `src/pyrite/`.

| Owner | Current constraint | Required change |
| --- | --- | --- |
| `montecarlo/spectrum/lines/_per_hkl.py` | Sinc-window bounds use a single `dE` | Search actual coordinates |
| `montecarlo/spectrum/lines/_kernels.py::_line_tabulation_grid` | Internal 1 eV mesh across the interval | Use validated sparse nodes plus native edge data |
| `detectors/_si_sensor.py::poisson_core` | Counts use the first spacing everywhere | Consume explicit bin masses or local widths |
| `detectors/_si_sensor.py::grid_key` | Size and endpoints identify a uniform grid | Hash complete coordinates and grid semantics |
| `detectors/timepix_response.py` | Rebinning uses one input-bin width | Conservative mapping to response input bins |
| `detectors/response.py`, `detectors/eaglexo_response.py` | Energy-resolution paths assume uniform sampling | Independent response channels or validated resampling |
| `results/metrics.py` | Peak widths/windows use sample coordinates | Convert interpolated crossings to physical energies |
| `energy_grid/derive.py`, `bounds.py`, `apply.py` | Linear descriptors and fixed diagnostic defaults | Preserve policy and exact resolved nonuniform coordinates |

The existing characteristic-line edge/width helper supports nonuniform centers;
reuse its integration ideas while making endpoint conventions explicit. Audit
all CUDA and fallback routes before enabling the policy. Keep adjacent nodes
distinct in backend precision; broad coverage does not justify float32 collapse
of locally fine windows.

The derivation default of 95% integrated coverage is a bandwidth policy, not a
1% accuracy guarantee. Set bandwidth and resolution separately for each
observable. Hard diagnostic ceilings must expand with the requested energy range.
Preserve old grid decoding and include changed numerical semantics in artifact,
case, and checkpoint identities where they affect computed results.

## Acceptance and rollout

1. Pin grid semantics and identify uniform-only consumers before changing defaults.
2. Implement and validate log transport tables independently of photon changes.
3. Enable nonuniform continua with conservative detector scoring and cache keys.
4. Add local line windows and adaptive refinement, then revise derivation defaults.

Validate LUTs against direct physics at off-grid points sampled throughout every
decade, with targeted points near cutoffs and joins. Preserve existing tested
accuracy where applicable: `tests/montecarlo/test_transport_lut.py` checks roughly
1e-5 relative agreement for selected quantities over 5–30 keV. That test does not
certify a six-decade interval or high-energy model validity.

For spectra, reuse identical trajectories to isolate grid error. Compare yield,
centroid, FWHM, detected counts, and line/background ratios under refinement.
An initial target is below 1% observable change, tightened for precision work;
allocate the total error budget across bandwidth, quadrature, interpolation,
transport, and Monte Carlo statistics. Cover near-zero spectra with absolute
tolerances. Record wall time, memory, and accuracy together. Heavy sweeps and GPU
measurements belong on the remote compute path.

This recommendation introduces numerical policies only; it certifies no new
transport or radiation equation. Production physics changes require the normal
source, limiting-case, ledger, and independent-validation workflow.
