# Validation: `brem-spectrum`

Implementation-context validation of the EEDL bremsstrahlung background.

- **Claim id:** `brem-spectrum`
- **Code:** `src/pyrite/montecarlo/spectrum/brem.py::load_bremsstrahlung_cross_sections`,
  `::_prepare_eedl_grid`, `::_eedl_brem_dsigma_dk`,
  `::_bremsstrahlung_dsigma_dk`, and `::mc_brem_spectrum`;
  `src/pyrite/montecarlo/spectrum/brem_jit_kernel.py::run_eedl_brem_reduction_kernel`
- **Source:** EEDL {cite:p}`perkins1991eedl`; ENDF-6 Files 23 and 26
  {cite:p}`trkov2018endf6`
- **Status:** `filtered`
- **Independence:** this update was checked against the implementation while it
  was being changed. A fresh-context re-derivation and human sign-off remain
  pending.

## Intended quantity

For incident electron kinetic energy $T$, photon energy $k$, and element $Z$,
the default differential cross section is

$$
\frac{d\sigma_Z}{dk}(T,k)
=\sigma_Z^{23,527}(T)P_Z^{26,527}(k\mid T),
\qquad
\int_0^T P_Z(k\mid T)\,dk=1.
$$

The ENDF-6 electro-atomic convention places the total bremsstrahlung cross
section in MF=23/MT=527. MF=26/MT=527 uses two subsections: the photon spectrum
first, followed by the electron average energy loss. The photon subsection is a
continuum distribution with `LAW=1`, `LANG=1`, `NA=0`; its $f_0$ values are the
energy probability density integrated over angle. The packaged tape also
declares `LEP=2` and incident-panel `INT=2`, both lin-lin.

For a segment of length $L$ in number density $n_Z$, PyRITE retains its current
isotropic estimator:

$$
\frac{d^2N}{dk\,d\Omega}
=\frac{1}{4\pi}\sum_Z n_Z L
\frac{d\sigma_Z}{dk}(T,k)\,T_{\rm abs}(k).
$$

After summing segments and dividing by the macro-electron count, the returned
units are photons/eV/sr/incident electron.

## Parser and units

`endf-parserpy` decodes the packaged `EEDL.endf`. The loader:

1. finds the material by $Z=\mathrm{round}(ZA/1000)$;
2. requires MF=23/MT=527 and MF=26/MT=527;
3. requires ordered, finite, non-negative values and the supported ENDF
   interpolation laws;
4. converts MF=23 barns to cm² using $1\ \mathrm{barn}=10^{-24}\ \mathrm{cm}^2$;
5. verifies every native photon density integrates to unity within the
   source-record precision, then removes that rounding residual by
   normalization; and
6. verifies the packaged bytes against the pinned SHA-256 before parsing.

The carbon MF=23 table gives, by declared lin-lin interpolation,

$$
\sigma_{\rm C}(30\ \mathrm{keV})
=42.8516263574\ \mathrm{barn}
=4.28516263574\times10^{-23}\ \mathrm{cm}^2.
$$

This value is pinned independently in the regression suite.

## Two-dimensional interpolation

Let $T_0\le T\le T_1$ bracket adjacent incident photon-spectrum panels and
$w=(T-T_0)/(T_1-T_0)$. Following the tape's `INT=2` law, PyRITE first evaluates
each piecewise-linear panel at the same absolute photon energy:

$$
\widetilde P(k\mid T)
=(1-w)P(k\mid T_0)+wP(k\mid T_1).
$$

Because the upper endpoints differ, this Cartesian interpolation can leave a
small tail at $k>T$. PyRITE enforces the physical support and restores the
normalization:

$$
P(k\mid T)=
\frac{\widetilde P(k\mid T)\,\mathbf 1_{0<k\le T}}
{(1-w)F_0(T)+wF_1(T)},
$$

where $F_i(T)$ is the exact integral of panel $i$'s piecewise-linear density up
to $T$. The denominator is evaluated analytically from native-grid trapezoid
segments, not from the caller's output grid. Thus an arbitrarily cropped or
coarse output grid cannot change the underlying normalization.

At a native incident panel, $w$ is zero or one, so the result reduces to that
native normalized spectrum. Across panels, a dense independent trapezoid
integration of $d\sigma/dk$ recovers the MF=23 total within the output-grid
quadrature tolerance.

## Backend and fallback behavior

`mc_brem_spectrum` defaults to `cross_section_model="eedl"`. The previous
Bethe--Heitler + Elwert helper is unchanged and remains selectable with
`cross_section_model="bethe-heitler"`.

An absent element, absent 527 section, or incident energy outside the common
MF=23/MF=26 range emits a `RuntimeWarning` and evaluates only the affected
element/segment rows with Bethe--Heitler. Structural violations and checksum
mismatches do not fall back: they indicate corrupt or unsupported input and
raise. The EEDL generation marker participates in dataset identities, case
content keys, and public result provenance, preventing reuse of spectra from the
retired default.

For each element and output-energy grid, `mc_brem_spectrum` evaluates the native
MF=26 panels on that grid once and stages the resulting panel matrix once. It
also computes the incident-panel bracket, interpolation fraction, total cross
section, and exact cutoff-normalization factor once per segment. Reusing these
arrays across reduction chunks avoids repeatedly parsing or transferring EEDL
data without changing the interpolation equations above.

On CUDA with float32 arrays, a fused reduction kernel consumes the staged panel
matrix and the one-dimensional per-segment state. Each block evaluates adjacent
EEDL panels (or the Bethe--Heitler fallback for an uncovered segment), applies
the existing layer escape factor, and reduces directly into the output-energy
bin. It does not materialize a segment-by-energy cross-section matrix. CPU,
ROCm, SYCL, float64, and unsupported CUDA specializations retain the portable
array implementation. Its chunk admission budgets eight dense intermediates
for EEDL rather than the analytic path's three.

CuPy can report an allocation failure only when a later operation synchronizes.
The backend therefore recognizes both memory-pool `OutOfMemoryError` and
`CUDARuntimeError`/`HIPRuntimeError` with the runtime memory-allocation status;
the runner tags either as a bremsstrahlung-phase OOM and retries with a smaller
chunk. Other runtime failures remain hard errors.

## Dimensional and limiting checks

- $sigma P$ has units cm²/eV.
- $n_Z[\mathrm{cm}^{-3}]L[\mathrm{cm}]\sigma P[\mathrm{cm}^2/\mathrm{eV}]$
  has units photons/eV.
- Division by $4\pi$ and the electron count gives photons/eV/sr/electron.
- Every parsed density and cross section is non-negative, so the unattenuated
  estimator is non-negative.
- $k\le0$ and $k>T$ are exactly zero.
- At a native panel, the interpolated distribution reduces to the parsed native
  distribution.
- Missing coverage reduces exactly to the historical Bethe--Heitler helper and
  emits a note through Python's warning system.

## Regression anchors

- `tests/montecarlo/test_bremsstrahlung_eedl.py`: carbon source value, native
  normalization, interpolated normalization, isotropic $nL\sigma P/(4\pi)$
  reduction, explicit analytic selection, fallback warnings, invalid-model
  rejection, and one-time panel staging across portable reduction chunks.
- `tests/montecarlo/test_chunk_invariance.py`: output independence from segment
  chunk size.
- `tests/montecarlo/test_spectrum_cuda_cheap_hoists.py`: CUDA-gated fused EEDL
  reduction against an independent NumPy reference.
- `tests/integration/test_backend_selection.py` and
  `tests/montecarlo/test_gpu_oom_retry.py`: delayed runtime-allocation
  classification, non-memory error exclusion, phase tagging, and retry sizing.
- `tests/montecarlo/test_groove.py`: unchanged Beer--Lambert geometry assembly
  using the explicitly selected legacy backend.
- `tests/montecarlo/test_radiation_error_estimators.py`: endpoint estimator
  follows the selected cross-section backend.
- `tests/materials/test_profiles.py` and `tests/scan/test_public_api.py`: cache
  identity and provenance markers.

## Verdict

Parser structure, units, interpolation, normalization, default selection,
explicit legacy selection, fallback notes, isotropic assembly, and cache
separation pass implementation-context filters. Status remains `filtered` until
an independent fresh-context verifier re-derives the interpolation and compares
it term-by-term without first reading the implementation. Only a human may mark
the row `signed-off`. The CUDA kernel is statically checked and has a hardware-
gated equivalence test, but that test could not run in the 2026-09-03 editing
environment because it had no CUDA device; hardware execution remains pending.
