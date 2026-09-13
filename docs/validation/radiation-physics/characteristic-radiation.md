# Validation: characteristic-radiation

## Source model

PyRITE reads electron-impact subshell-ionization cross sections from the
packaged 2025 Livermore Evaluated Electron Data Library (EEDL), distributed in
ENDF-6 form as NDS-IAEA-226. The parser accepts ENDF File 23, MT 534--572 TAB1
sections and converts their tabulated cross sections from barns to cm$^2$.
Each section declares interpolation law 2, so the incident-energy dependence
is evaluated piecewise linearly and set to zero outside the tabulated range.

EEDL supplies vacancy-production cross sections, but not the relaxation data
used here. Line energy $E_{ai\ell}$, fluorescence yield $\omega_{ai}$, and
conditional radiative intensity $I_{ai\ell}$ come from the Elam tables exposed
by xraydb. Natural initial- and final-hole widths come from xraydb's compiled
Krause--Oliver and Keski-Rahkonen--Krause tables. For segment $j$ in an
emitting material, the bin-averaged track-length estimator is

```{math}
:label: eq-characteristic-track-length

\left.\frac{d^2N}{dE\,d\Omega}\right|_b
=\frac{1}{4\pi N_e\,\Delta E_b}
\sum_{j,a,i,\ell}
n_a L_j\,\sigma_{ai}(T_j)\,
\omega_{ai}I_{ai\ell}\,
\exp[-\tau_j(E_{ai\ell})]\,\widehat q_{ai\ell b}.
```

Here $a$ is an element, $i$ an initially ionized subshell, $\ell$ a line from
that vacancy, $T_j$ the representative electron energy, and $b$ an energy bin.
The factor $\widehat q_{ai\ell b}$ is the natural Lorentzian mass in bin $b$,
renormalized over the requested grid:

```{math}
q_{\ell b}=\frac{1}{\pi}\left[
\tan^{-1}\!\frac{2(E_b^+-E_\ell)}{\Gamma_\ell}
-\tan^{-1}\!\frac{2(E_b^--E_\ell)}{\Gamma_\ell}
\right],\qquad
\widehat q_{\ell b}=\frac{q_{\ell b}}{\sum_c q_{\ell c}},\qquad
\Gamma_\ell=\Gamma_{\rm initial}+\Gamma_{\rm final}.
```

The result is photons eV$^{-1}$ sr$^{-1}$ per incident electron. The factor
$1/(4\pi)$ is isotropic emission; $\tau_j$ is the existing PyRITE
Beer--Lambert optical depth along slab, finite-prism, groove, or multilayer
escape geometry. Exact CDF differences avoid point-sampling line shapes much
narrower than a bin. The grid renormalization preserves the full transition
yield whenever its centre is in the requested window, matching the previous
delta-line window convention.

## Units and numerical conventions

- $n_a$ is stored in $\AA^{-3}$ and multiplied by $10^{24}$ to obtain
  cm$^{-3}$; $L_j$ is stored in $\AA$ and multiplied by $10^{-8}$ to obtain cm.
  Thus $n_aL_j\sigma_{ai}$ is a dimensionless expected vacancy count.
- Dividing the analytically integrated Lorentzian mass by $\Delta E_b$
  produces the spectral density represented on PyRITE's line-grid centres.
  Detector broadening remains a separate downstream operation.
- A line contributes only when its centre lies inside the requested line grid.
  Its Lorentzian tails are then normalized on that grid so truncation cannot
  change the integrated vacancy yield.
- The transition FWHM is the sum of the pertinent initial- and final-hole
  widths. Combined final labels such as `M4,5` use the mean available
  component width. A missing final width contributes zero; a missing initial
  width fails closed.
- Packaged EEDL bytes are verified before first use against SHA-256
  `f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c`.
  The resolved xraydb package version is included in the characteristic-model
  checkpoint marker.

## Assumptions and scope

The estimator treats independent atoms and isolated, directly created
vacancies. It multiplies the xraydb edge fluorescence yield by the conditional
line intensity for that same initial shell. It does not invent Auger-fed
daughter vacancies, Coster--Kronig redistribution, multiple-vacancy shifts, or
Auger-electron transport because xraydb's line API is not a complete cascade
model. An EEDL shell with a nonzero fluorescence yield but no xraydb line list
is retained, emits zero photons, and raises a `RuntimeWarning` rather than being
silently approximated.

Characteristic emission uses the bremsstrahlung electron population and its
default 1 keV transport cutoff, rather than the PXR/CBS population's default
5 keV cutoff. This preserves more low-energy ionization path while sharing the
same trajectories as the rest of a case. The remaining path below 1 keV is not
modeled unless the case lowers `E_cut_brem_keV`. The 50 eV relaxation cutoff is
a photon-line data cutoff and does not override the electron transport cutoff.

For multilayers, each layer emits using its own elemental composition and all
layers attenuate the escaping photon. Passive absorber elements are not loaded
as EEDL emitters for another layer. Atomic relaxation is incoherent, so the
same characteristic component is added once to both PyRITE's incoherent and
optional coherent PXR/CBS totals. It is retained as `spec_characteristic` for
audit and plotting, persisted in its own `characteristic.h5` component, and can
be hidden in the analysis app without modifying stored totals. The control
defaults to showing the component.

A single natural-width Lorentzian is used per xraydb transition. The empirical
multi-Lorentzian fits of Hölzer et al. demonstrate satellite and asymmetric
structure in 3d-transition-metal lines, but do not supply a universal
parameterization for the full EEDL element/shell domain. That finer structure,
chemical shifts, and multiple-vacancy broadening are intentionally excluded.

## Limits and regression evidence

- If density, segment length, cross section, fluorescence yield, or line
  intensity tends to zero, {eq}`eq-characteristic-track-length` tends to zero.
- With zero attenuation and one line, summing $\Delta E_b$ times the returned
  density recovers $nL\sigma\omega I/(4\pi N_e)$ to floating-point
  precision.
- Splitting a constant-energy segment preserves total yield when its
  attenuation weight is also fixed (in particular, with zero attenuation).
  Changing midpoint escape depth changes attenuation quadrature.
- Increasing optical depth suppresses the line monotonically through
  $\exp(-\tau)$.

`tests/montecarlo/test_characteristic.py` anchors the packaged carbon K-shell
EEDL value and natural carbon K-alpha width, exact Lorentzian bin integration
and yield preservation, segment-subdivision invariance, absorber/emitter
separation, and runner composition into both emission modes. Checkpoint/reline
tests cover independent `characteristic.h5` persistence, missing-component
compatibility, and identity boundaries. A small end-to-end HOPG simulation
also exercises ENDF parsing, electron transport, self-absorption, line
profiles, and result assembly.

## Validation status

The sections above preserve the implementation-context derivation and
regression record of 2026-09-10, when the ledger status was `filtered` and
independent verification was pending. The independent review below supersedes
that pending-review statement; human sign-off remains pending.

## Independent verification, 2026-09-13

### Derivation frozen before implementation inspection

This verifier did not implement this claim. The ledger entry, signatures and
AST-extracted docstrings were read first; implementation bodies and the prior
review above were withheld until the following derivation was written.

Let $n_a$ be the number density of element $a$, $\sigma_{ai}(T)$ its
shell-ionization cross section per atom, and $\Delta s_j$ a path increment of
one incident electron at kinetic energy $T_j$. The expected number of initial
vacancies is $n_a\sigma_{ai}(T_j)\Delta s_j$. A direct transition $i\to f$
produces $\omega_{ai}b_{aif}$ photons per vacancy, where $\omega$ is the
fluorescence yield and $b$ is conditional on radiative decay of that initial
shell. Thus the thin-path photon expectation is

$$
\Delta N_{jaif}=n_a\sigma_{ai}(T_j)\Delta s_j\omega_{ai}b_{aif}.
$$

The conditional branch probabilities sum to one before any line-energy
selection. Selected branches must not be renormalized: omitting a transition
reduces the emitted photon count. No shell occupancy multiplier belongs here
because the EEDL quantity is already a subshell cross section. The direct model
does not propagate daughter vacancies; a full cascade would require a vacancy
transfer matrix and its repeated action on the initial vacancy vector.

[The EEDL survey, IAEA-NDS-226](https://www-nds.iaea.org/epics/DOCUMENTS/EEDL.pdf)
describes shell-resolved electron data and binding-energy updates. The
[ENDF-6 manual](https://www.oecd-nea.org/dbdata/data/endf102.htm) specifies the
File 23 representation. The source-data year and report year are distinct:
NDS-226 is dated December 2017. With cross sections in barns, lengths in
angstroms and densities in atoms per cubic angstrom,

$$
n_a\,[\mathrm{\mathring A}^{-3}]\,
\sigma_{ai}\,[\mathrm{barn}]\,\Delta s_j\,[\mathrm{\mathring A}]
\times10^{-8}
$$

is a dimensionless vacancy expectation. Equivalently, convert density by
$10^{24}$ and path length by $10^{-8}$ to centimetre units, and convert barns
by $10^{-24}$. Incident energy conversion is $T_{\rm eV}=10^3T_{\rm keV}$.
Linear interpolation is valid only for an ENDF region declaring that law;
zero outside the table is a model boundary rather than a high-energy
extrapolation.

The [XrayDB API](https://xraypy.github.io/XrayDB/python.html) documents the
Elam-derived edge fluorescence yields and transition intensities, plus
core-hole widths in eV from Krause--Oliver and Keski-Rahkonen--Krause.
For independent exponentially decaying initial and final holes, convolution
of their Cauchy spectral densities adds their half widths. Consequently

$$
\Gamma_{aif}=\Gamma_{ai}+\Gamma_{af},\qquad
\gamma_{aif}=\frac{\Gamma_{aif}}{2},\qquad
L_{aif}(E)=\frac{\gamma_{aif}}
 {\pi[(E-E_{aif})^2+\gamma_{aif}^2]}.
$$

This agrees with the line-width addition rule in
[Krause and Oliver (1979), Eq. (2)](https://srd.nist.gov/jpcrdreprint/1.555595.pdf).
The profile has unit integral over the real axis, half its maximum at
$E-E_{aif}=\pm\gamma_{aif}$, and units of inverse eV. Averaging unresolved
final-level widths and assigning zero to a missing final width are explicit
approximations, not consequences of the lifetime convolution. A grouped
transition is generally a weighted sum of profiles rather than one profile
with an arithmetic-mean width. The single-line model does not reproduce the
empirical multi-Lorentzian fits of
[Hölzer et al. (1997)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.56.4554).

For bin edges $e_k,e_{k+1}$, the exact unconditioned bin probability is

$$
p_{aif,k}=\frac{1}{\pi}\left[
\arctan\frac{e_{k+1}-E_{aif}}{\gamma_{aif}}
-\arctan\frac{e_k-E_{aif}}{\gamma_{aif}}\right].
$$

The stated window-preserving convention instead uses

$$
P_{aif,W}=\sum_{k\in W}p_{aif,k},\qquad
w_{aif,k}=\frac{p_{aif,k}}{P_{aif,W}},\qquad
\sum_k w_{aif,k}=1,
$$

and includes a transition only if its centre lies in the admitted window.
This is a conditional, truncated Lorentzian. It preserves the complete line
yield on a chosen grid but differs from physically cropping an unconditioned
spectrum by the exact factor $1/P_{aif,W}$. Changing the window can therefore
change the density in overlapping bins. At a window edge $P_W$ approaches
one half for a distant opposite boundary, doubling the retained half-line.
This accepted numerical convention must not be called conservation under
physical aperture selection. The infinite-window limit removes the factor.

For isotropic photons the probability per solid angle is $1/(4\pi)$.
The escape probability from source position $\mathbf r_j$ along
$\hat{\mathbf n}$ is

$$
A_j(E)=\exp\left[-\sum_m\mu_m(E)\ell_{jm}(\hat{\mathbf n})\right],
$$

where each material path $\ell_{jm}$ is nonnegative. Passive absorbers
contribute to this exponent but not to $n_a$ in the source term. A
line-centre attenuation approximation takes $A_j(E)\simeq A_j(E_{aif})$;
it is accurate when attenuation varies little across the admitted profile.
The exact absorbed bin would integrate $L(E)A_j(E)$ instead. The estimator
expected under the stated discretization is

$$
\boxed{S_k=\frac{1}{4\pi N_e\Delta E_k}
\sum_{j,a,i,f}n_a\sigma_{ai}(T_j)\Delta s_j
\omega_{ai}b_{aif}A_j(E_{aif})w_{aif,k}.}
$$

$N_e$ counts all incident macro-electrons in the selected cohort, including
those producing no retained segment. This is a photon-number density per eV,
per sr and per incident electron; there is no photon-energy factor and no
extra speed factor. A track-length estimator integrates path length, not
residence time.

Cheap filters pass under those assumptions: zero density, path length, cross
section or fluorescence yield gives zero; absorption is positive and at most
one; an opaque path gives zero and zero optical depth gives unity. Splitting
a segment while holding energy, escape geometry and source composition fixed
preserves its yield. A zero-width line inside one bin tends to a delta mass;
on a bin boundary its symmetric limit divides between adjacent bins.
Unattenuated integration over bins and solid angle recovers the direct photon
expectation per incident electron. Real transport subdivision changes energy
and position quadrature and is not required to be exactly invariant.

Characteristic emission is statistically incoherent with PXR/CBS amplitudes.
If the two total-spectrum choices differ only in coherent versus incoherent
PXR/CBS treatment, the same characteristic intensity must be added once to
each. It must not be added to amplitudes or multiplied by an interference
factor.

### Comparison after the derivation was frozen

| Derived term or convention | Implementation evidence | Result |
|---|---|---|
| Subshell cross section per atom, barns to cm$^2$ | `_extract_eedl_subshell` uses `mt - 533`, requires law 2, and multiplies by `1.0e-24` | Matches |
| Incident energy conversion and bounded linear interpolation | `_interpolate_shell_cross_sections` multiplies keV by `1.0e3` and masks outside table endpoints | Matches |
| Direct branching $\omega_i b_{if}$ | `_parse_characteristic_file` stores `fluorescence_yield * intensity`; `_xraydb_line_yields` zeros excluded lines without renormalization | Matches |
| Natural FWHM $\Gamma_i+\Gamma_f$ | `_transition_fwhm_eV` sums the two resolved widths | Matches, with declared grouped/missing-width approximations |
| Conditional bin mass $p_k/P_W$ | `_lorentzian_bin_weights` evaluates the arctangent difference and divides by `captured` | Matches the declared window convention |
| Density, path and angular normalization | `mc_characteristic_spectrum` multiplies density by `1.0e24`, path by `1.0e-8`, then divides by bin width, `4*pi`, and incident `Ne` | Matches |
| Escape evaluated at line centre | `mc_characteristic_spectrum` evaluates material attenuation once per transition and uses the segment-midpoint escape path | Matches the stated quadrature approximation |
| Each layer emits from its own composition | `_characteristic_from_segments` selects segments by layer and retains the complete absorber stack | Matches |
| One common additive intensity | `runner/__init__.py` adds `spec_characteristic` once to `spec` and once to an existing `spec_coherent` | Matches |

The EEDL threshold and xraydb line/edge energies originate in different
compilations. They are joined by element and shell; this review verifies that
join and the direct-emission estimator, not a self-consistent energy-conserving
relaxation cascade or the physical accuracy of every atomic datum.

### Independent numerical evidence

All computations below used the project runner with `PYRITE_MC_BACKEND=cpu`.
No transport sweep or GPU calculation was performed.

- A small fixed-column reader, independent of `endf-parserpy` and the PyRITE
  parser, read packaged carbon MAT 600, MF 23, MT 534. Its raw header gives
  binding energy 288 eV and 25 samples with interpolation law 2. The bracketing
  records are 25118.9 eV / 59832.9 barns and 39810.7 eV / 42689.6 barns.
  Direct scalar linear interpolation at 30000 eV gives
  $5.4137330932220685\times10^{-20}\,\mathrm{cm}^2$; the implementation gives
  $5.4137330932220697\times10^{-20}\,\mathrm{cm}^2$ (relative difference
  $2.22\times10^{-16}$). An assertion with relative tolerance $10^{-13}$
  and **zero absolute tolerance** passes.
- For a 1000 eV line, FWHM 2 eV and bin edges 998 through 1002 eV, direct
  analytic integration captures 0.7048327646991335 of the full Lorentzian.
  The implementation sums to 1, exactly as the declared conditional model
  requires; its retained-bin density is larger by 1.4187762687605225.
  The independently computed conditional bin vector matches at relative
  tolerance $10^{-14}$ with zero absolute tolerance.
- For carbon at 30 keV, density $0.1\,\mathrm{\mathring A}^{-3}$, one
  100 angstrom track, two incident electrons and zero attenuation, direct
  multiplication of the raw EEDL cross section and XrayDB API yields predicts
  $3.0156786314065755\times10^{-7}$ photons per sr per incident electron.
  The integrated output is $3.0156786314065765\times10^{-7}$, agreeing to
  $4.44\times10^{-16}$ relatively. This also checks that the electron with
  no segment remains in the normalization.
- Carbon's API branch intensities sum to 1.0000000972. Using the idealized
  sum of exactly one initially produced a relative difference of
  $9.72\times10^{-8}$; retaining the rounded source intensities resolves it.
  The implementation accepts source sums within 0.99 to 1.01 and does not
  renormalize them. Thus photon-yield preservation is exact relative to the
  supplied branch sum, and only approximate relative to $\omega_i$ alone.
  Carbon K-alpha-1 has API widths $0.0868+0.0045=0.0913$ eV, matching
  the constructed transition width.
- The focused command
  `PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_characteristic.py`
  passed **7 tests**. These cover parsing/provenance, relaxation joining,
  line integration, transparent-track subdivision, absorber separation and
  addition to both emission modes.

The existing carbon cross-section assertion uses `np.isclose` with its
default absolute tolerance $10^{-8}\,\mathrm{cm}^2$. That tolerance is much
larger than the reference value: even zero passes this particular assertion.
It is therefore a weak regression anchor for the numerical cross section.
The independent zero-absolute-tolerance check above verifies today's value;
it does not repair the persistent test. A separate test improvement is
recommended before treating that reference as a strong CI anchor.

### Scope corrections and evidence limits

Constant energy alone does **not** guarantee subdivision invariance when
self-absorption varies along the track. For a uniform path with total optical
depth 2, one midpoint gives attenuation 0.36787944117144233, two equal
subsegment midpoints give 0.4148304099305316, and exact path integration gives
0.43233235838169365. This is ordinary midpoint-quadrature error, not a missing
normalization factor. The invariant limit holds with identical attenuation
weights or zero attenuation; the existing regression test explicitly sets
attenuation to zero.

At the time of this review, the maintained physics page
[Characteristic radiation](../../physics/radiation-physics/characteristic-radiation.md)
needs two exact documentation corrections: its density formula uses
$Yq_b/\Delta E_b$ after defining the unnormalized $q_b$, while the
implementation requires $Yq_b/(\Delta E_b\sum_cq_c)$; its subdivision bullet
needs the fixed-attenuation condition above. This verifier only edits the
ledgered validation document and leaves those corrections to the owning
context.

The positive-opacity checks assume finite nonnegative attenuation
coefficients. The implementation replaces nonfinite coefficients, including
positive infinity, by zero. Therefore a directly supplied infinite
coefficient becomes transparent instead of opaque; this behavior is outside
the finite-coefficient verdict. The physical opaque limit means increasing
finite optical depth, for which the exponential tends to zero. This review
did not establish reachability of nonfinite coefficients from valid bundled
atomic data.

Source access was bounded explicitly. The ENDF-6 manual, EEDL survey and
XrayDB API documentation were inspected online. The EEDL report is dated
2017; the packaged tape records evaluation in August 2023 and distribution in
January 2025. Its packaged-byte hash test passes, but this review did not
independently redownload the complete 2025 upstream tape. The Krause--Oliver
paper text, including section 2's Eq. (2) linewidth-sum rule, was retrieved
from an [Argonne-hosted copy of the primary paper](https://millenia.cars.aps.anl.gov/archives/list/ifeffit%40millenia.cars.aps.anl.gov/message/W3G725L4WFZ7M3ZUYX543D73VGSXICYR/attachment/7/Krause1979.pdf)
after the direct NIST PDF endpoint failed. Elam and Keski-Rahkonen--Krause
numerical data were checked through the primary XrayDB API documentation and
installed database, not by independently digitizing those original papers.
Hölzer's publisher abstract supports the stated multi-Lorentzian model
limitation; no reproduction of its measured spectra is claimed.

### Independent verdict and suggested ledger edit

**Re-derived:** the code matches the direct-vacancy, conditional-window,
line-centre-attenuation track-length estimator with finite nonnegative
attenuation coefficients. Units, signs and the scoped limiting cases pass.
This does not certify a physically cropped full Lorentzian spectrum, a
complete atomic relaxation cascade, empirical line-profile accuracy, or
subdivision invariance under changing attenuation weights. Human sign-off
remains pending.

Suggested human-applied ledger update: change `characteristic-radiation`
from `filtered` to `rederived`; replace the pending independent-review note
with this dated result and its conditional-window, midpoint-quadrature and
finite-attenuation scope; record the independent raw carbon interpolation
and incident-electron normalization checks; flag the overly permissive
cross-section test absolute tolerance. Do not apply `signed-off`.

### Coordinated documentation checks, 2026-09-13

Supplied by the coordinating context after this verification was written.
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs` builds this page
with no new Sphinx warning. `uv run pyrite-dev test tests/dev/test_docs.py
tests/dev/test_validation_ledger.py tests/dev/test_doc_blocks.py` passed
**68 tests**, covering documentation structure, ledger schema/anchor
resolution and doc code blocks. The rendered page carries 63 parsed math
nodes, with no unparsed dollar-delimited or backslash-parenthesis math left
as literal text in the HTML, so the display and inline math above render.

The two documentation corrections requested above were applied to the
maintained physics page by the coordinating context: its bin density now reads
$Y\widehat q_b/\Delta E_b$ with $\widehat q_b=q_b/\sum_cq_c$ stated explicitly,
and its subdivision bullet now carries the fixed-attenuation condition and the
optical-depth-2 midpoint-quadrature numbers. No code was changed. The weak
`np.isclose` absolute tolerance on the carbon cross-section assertion is left
as a recorded ledger finding for a separate test-improvement task.
