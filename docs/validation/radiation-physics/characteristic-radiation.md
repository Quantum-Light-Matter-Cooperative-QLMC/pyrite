# Validation: characteristic-radiation

## Source model

PyRITE reads electron-impact subshell-ionization cross sections from the packaged 2025 Livermore Evaluated Electron Data Library (EEDL), distributed in ENDF-6 form as NDS-IAEA-226. The parser accepts ENDF File 23, MT 534--572 TAB1 sections and converts their tabulated cross sections from barns to cm$^2$. Each section declares interpolation law 2, so the incident-energy dependence is evaluated piecewise linearly and set to zero outside the tabulated range.

EEDL supplies vacancy-production cross sections but no relaxation data. Since issue #91 (model `v7`), relaxation comes from the packaged EPICS2025 EADL, ENDF File 28, MT 533: per subshell $i$ its binding energy $B_i$, occupancy, and every radiative ($i\to j$) and nonradiative ($i\to(j,k)$) transition with energy and probability. Line energies are xraydb's (Elam) where xraydb tabulates the same level pair, EADL transition energies otherwise. Natural initial- and final-hole widths come from xraydb's compiled Krause--Oliver and Keski-Rahkonen--Krause tables. For segment $j$ in an emitting material, the bin-averaged track-length estimator is

```{math}
:label: eq-characteristic-track-length

\left.\frac{d^2N}{dE\,d\Omega}\right|_b
=\frac{1}{4\pi N_e\,\Delta E_b}
\sum_{j,a,p,\ell}
n_a L_j\,\sigma_{ap}(T_j)
\left[\sum_{i}V^a_{pi}\,R^a_{i\ell}\right]
\langle e^{-\tau_j(E_{a\ell})}\rangle\,q_{a\ell b}.
```

Here $a$ is an element, $p$ the subshell the incident electron ionized, $i$ any subshell the resulting cascade reaches, $\ell$ a line, $T_j$ the representative electron energy, and $b$ an energy bin. $R^a_{i\ell}$ is the probability that one decay of a vacancy in $i$ emits $\ell$, and $V^a$ the expected vacancy count

```{math}
:label: eq-characteristic-eadl-cascade

V^a=(\mathbb{1}-D^a)^{-1}=\sum_{m\ge0}(D^a)^m,
\qquad
D^a_{ij}=\sum_{i\to j}F+\sum_{i\to(j,k)}F+\sum_{i\to(k,j)}F,
```

with $F$ the EADL transition probability, so a nonradiative decay adds one hole to each of its two daughter subshells. Subshells with $B_i\leq B_{\rm cut}$ have zero rows in $D^a$ and $R^a$. In decreasing-binding-energy order (equal energies by designator) every EADL transition fills from a less tightly bound subshell, so $D^a$ is strictly upper-triangular and nilpotent and the series is a finite sum, evaluated exactly by forward substitution. With $B_{\rm cut}$ above every binding energy, $V^a=\mathbb{1}$ and the estimator is the direct-vacancy form with EADL branching. The factor $q_{a\ell b}$ is the unconditioned natural Lorentzian mass in bin $b$:

```{math}
q_{\ell b}=\frac{1}{\pi}\left[
\tan^{-1}\!\frac{2(E_b^+-E_\ell)}{\Gamma_\ell}
-\tan^{-1}\!\frac{2(E_b^--E_\ell)}{\Gamma_\ell}
\right],\qquad
P_{\ell,W}=\sum_{b\in W}q_{\ell b}\leq1,\qquad
\Gamma_\ell=\Gamma_{\rm initial}+\Gamma_{\rm final}.
```

The result is photons eV$^{-1}$ sr$^{-1}$ per incident electron. The factor $1/(4\pi)$ is isotropic emission; $\tau_j$ is the existing PyRITE Beer--Lambert optical depth along slab, finite-prism, groove, or multilayer escape geometry. Exact CDF differences avoid point-sampling line shapes much narrower than a bin. A finite requested window retains only the physical probability $P_{\ell,W}$, while an off-grid line centre retains its nonzero in-window tail. The infinite-window limit gives $P_{\ell,W}\to1$.

## Units and numerical conventions

- $n_a$ is stored in $\AA^{-3}$ and multiplied by $10^{24}$ to obtain cm$^{-3}$; $L_j$ is stored in $\AA$ and multiplied by $10^{-8}$ to obtain cm. Thus $n_aL_j\sigma_{ai}$ is a dimensionless expected vacancy count.
- Dividing the analytically integrated Lorentzian mass by $\Delta E_b$ produces the spectral density represented on PyRITE's line-grid centres. Detector broadening remains a separate downstream operation.
- Every line above the relaxation-data cutoff contributes its physical mass in the requested line grid, including tails from an off-grid centre. No finite-window renormalization is applied.
- The transition FWHM is the sum of the pertinent initial- and final-hole widths. Combined final labels such as `M4,5` use the mean available component width. A missing final width contributes zero; a missing initial width fails closed.
- Packaged EEDL bytes are verified before first use against SHA-256 `f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c`, and packaged EADL bytes against `78ccf8a4e07c1c120a2e3d94ff051aab2180d151f35e8bc3406d52df5af5e88c`. Both checksum prefixes, the endf-parserpy and xraydb versions, and the fluorescence-yield source are part of the characteristic-model checkpoint marker.
- EADL binding and transition energies are in eV and EADL probabilities are dimensionless, so $V^a$, $R^a$ and $V^aR^a$ are dimensionless counts per primary vacancy.

## Assumptions and scope

The estimator treats independent atoms and relaxes every primary vacancy bound above the 50 eV cutoff through {eq}`eq-characteristic-eadl-cascade`. The relaxation cutoff bounds vacancy propagation by binding energy; photons at or below it are also not scored.

Declared approximations and validity limits:

- Energy accounting: per primary vacancy, $B_p$ equals expected photon energy plus Auger-electron energy plus terminal-vacancy binding plus an EADL transition-energy defect, because EADL transition energies are not differences of its single-vacancy binding energies. The identity closes to the $10^{-5}$ branching-sum precision. The defect is under 1.5% of $B_p$ for every K primary and reaches about 6% for L primaries above 100 eV (Si L2, K L1).
- Auger electrons are not transported; their energy is booked but they do not ionize or radiate.
- Independent single-vacancy rates: no multiple-vacancy shifts, rate changes, or satellites.
- M and N lines absent from xraydb use calculated EADL energies, which can differ from measured values by tens of eV and carry no multiplet splitting.
- Secondary fluorescence: the escape factor is a pure sink, so an absorbed characteristic photon does not re-emit.
- EADL's L3 branch shape for 3d metals puts Lℓ/Lα1 at 1--3, against about 0.1 in xraydb; 3d-metal L-line ratios are not claimed.
- EADL and Elam (xraydb) disagree on radiative yields and Coster--Kronig branching; the default uses EADL throughout, and `fluorescence_yields="elam"` rescales only the radiative/nonradiative split.

Characteristic emission uses the bremsstrahlung electron population and its default 1 keV transport cutoff, rather than the PXR/CBS population's default 5 keV cutoff. This preserves more low-energy ionization path while sharing the same trajectories as the rest of a case. The stopping/scattering model is not validated below 1 keV, so characteristic scoring resolves an omitted cutoff to 1 keV and rejects `E_cut_brem_keV < 1`. Low-binding-energy vacancy production below that floor is omitted. The 50 eV relaxation cutoff bounds atomic relaxation and does not override the electron transport-validity boundary. Cascade M and N lines below 1 keV from parents ionized above the floor do not validate sub-keV transport.

For multilayers, each layer emits using its own elemental composition and all layers attenuate the escaping photon. Passive absorber elements are not loaded as EEDL emitters for another layer. Atomic relaxation is incoherent, so the same characteristic component is added once to whichever of PyRITE's incoherent or optional coherent PXR/CBS spectra a consumer selects. It is kept as its own array, `spec_characteristic`, persisted in its own `characteristic.h5` component, and can be hidden in the analysis app without modifying stored results. The control defaults to showing the component.

A single natural-width Lorentzian is used per transition. The empirical multi-Lorentzian fits of Hölzer et al. demonstrate satellite and asymmetric structure in 3d-transition-metal lines, but do not supply a universal parameterization for the full EEDL element/shell domain. That finer structure, chemical shifts, and multiple-vacancy broadening are intentionally excluded.

## Limits and regression evidence

- If density, segment length, cross section, fluorescence yield, or line intensity tends to zero, {eq}`eq-characteristic-track-length` tends to zero.
- With zero attenuation and one line, summing $\Delta E_b$ times the returned density gives $P_{\ell,W}nL\sigma\omega I/(4\pi N_e)$; it approaches the complete yield monotonically as the window tends to the whole energy axis.
- Narrowing a window cannot increase integrated yield and does not change the density in bins whose edges are unchanged. A line centred outside the window has a positive in-window tail.
- Splitting a constant-energy segment preserves total yield when its attenuation weight is also fixed (in particular, with zero attenuation). Changing midpoint escape depth changes attenuation quadrature.
- Increasing optical depth suppresses the line monotonically through $\exp(-\tau)$.

`tests/montecarlo/test_characteristic.py` anchors the packaged carbon K-shell EEDL value and natural carbon K-alpha width, exact Lorentzian bin integration, finite-window mass, off-grid tails, the 1 keV floor, segment-subdivision invariance, absorber/emitter separation, and runner composition into both emission modes. Checkpoint/reline tests cover independent `characteristic.h5` persistence, missing-component compatibility, and identity boundaries. A small end-to-end HOPG simulation also exercises ENDF parsing, electron transport, self-absorption, line profiles, and result assembly.

## Issue #88 implementation-context review, 2026-09-14

The v3 line-window change removes the historical conditioning factor $1/P_{\ell,W}$. This follows directly from the whole-line normalization $\int_{-\infty}^{\infty}L_\ell(E)\,dE=1$: integrating over a proper subset $W$ must give $P_{\ell,W}\leq1$. Positivity follows from monotonicity of the arctangent CDF. Expanding either window boundary can only add nonnegative mass, while bins common to two windows retain identical CDF differences. When the line centre lies outside $W$, strict CDF monotonicity leaves a positive tail for every finite-width line. These limits and the symmetric half-mass example $W=[E_\ell-\Gamma_\ell/2,E_\ell+\Gamma_\ell/2]$ are anchored without using the implementation to construct the expected values.

The low-energy choice is an enforced model boundary, not a new transport equation. A universal 1 keV cutoff omits some physically possible ionization for shells whose thresholds are lower, but evaluating those paths would exceed the currently validated stopping/scattering range. `E_cut_keV=None` therefore means 1 keV for this estimator and a lower explicit value fails closed. A future transport model validated below 1 keV may replace this conservative boundary with the lowest scored shell threshold, but that is outside issue #88.

The model marker changes from `lorentzian-v2` to `lorentzian-v3`, so checkpoint dataset identities and case-content hashes cannot reuse spectra calculated under conditional window renormalization. No separate grid-independent integrated-yield field is introduced; the stored `spec_characteristic` remains the spectrum restricted to its recorded line grid.

This is an implementation-context review only. The new v3 convention has units, signs, normalization, narrow/wide/off-grid limits, and a regression anchor, but still requires fresh-context source-to-code validation. The ledger status is therefore `filtered`; no human sign-off is claimed.

## Issue #101 implementation-context note, 2026-09-14

Two `v3` to `v4` changes, both grid bookkeeping rather than new source physics:

1. **Low bin-edge clamp.** `_energy_bin_edges_and_widths`'s outer edges mirror the adjacent spacing, {math}`E_0^-=E_0-\tfrac12(E_1-E_0)`. That reflection can go negative when the first spacing exceeds the first node -- an unphysical photon energy. `v4` clamps {math}`E_0^-=\max(0,E_0-\tfrac12(E_1-E_0))`; the high edge is never clamped since photon energy has no analogous upper bound here. This changes {math}`\Delta E_0` (and therefore the first bin's reported density) only for grids that hit the negative-edge case; an ordinary evenly- or slowly-varying grid is unaffected because its first edge is already non-negative. Limiting case: a grid whose first spacing does not exceed its first node reduces to the unclamped `v3` formula exactly, since the reflected edge is then never negative.
2. **Explicit truncation query.** `characteristic_line_window_mass` returns {math}`(P_{\ell,W},\,1-P_{\ell,W})` per line -- the same $P_{\ell,W}$ already in {eq}`eq-characteristic-track-length`, just returned as data instead of only implicitly shaping the density. `captured+truncated=1` follows directly from `_lorentzian_bin_weights` being the unconditioned CDF difference (same argument as the v3 review above). `mc_characteristic_spectrum` additionally warns when a line's centre is inside the requested grid but the grid still captures under `CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION` (50%) of that line's mass, so a window that truncates a line at its own edges is reported rather than left for the caller to notice only as a smaller-than-expected peak. A line centred outside the grid does not warn: its small in-window tail is the documented off-grid-line behaviour, not a misconfigured window.

Neither change alters the analytic Lorentzian CDF itself, the vacancy-yield estimator, or the transport-validity floor. The model marker moves to `lorentzian-v4` because the low-edge clamp can change a stored spectrum's first-bin value; checkpoints identity-fork accordingly. This is an implementation-context review only, covering units, the clamp's limiting case, and the truncation identity above; it does not independently verify the change. The ledger status remains `filtered`; no human sign-off is claimed.

## Historical v2 validation status

The independent review below records the v2 implementation verified on 2026-09-13. Its direct-vacancy, linewidth, attenuation, unit, and normalization work remains evidence for unchanged parts of v3. Its conditional-window verdict is historical and does not independently verify the issue #88 change. Human sign-off remains pending.

## Independent verification of v2, 2026-09-13

### Derivation frozen before implementation inspection

This verifier did not implement this claim. The ledger entry, signatures and AST-extracted docstrings were read first; implementation bodies and the prior review above were withheld until the following derivation was written.

Let $n_a$ be the number density of element $a$, $\sigma_{ai}(T)$ its shell-ionization cross section per atom, and $\Delta s_j$ a path increment of one incident electron at kinetic energy $T_j$. The expected number of initial vacancies is $n_a\sigma_{ai}(T_j)\Delta s_j$. A direct transition $i\to f$ produces $\omega_{ai}b_{aif}$ photons per vacancy, where $\omega$ is the fluorescence yield and $b$ is conditional on radiative decay of that initial shell. Thus the thin-path photon expectation is

$$
\Delta N_{jaif}=n_a\sigma_{ai}(T_j)\Delta s_j\omega_{ai}b_{aif}.
$$

The conditional branch probabilities sum to one before any line-energy selection. Selected branches must not be renormalized: omitting a transition reduces the emitted photon count. No shell occupancy multiplier belongs here because the EEDL quantity is already a subshell cross section. The direct model does not propagate daughter vacancies; a full cascade would require a vacancy transfer matrix and its repeated action on the initial vacancy vector.

[The EEDL survey, IAEA-NDS-226](https://www-nds.iaea.org/epics/DOCUMENTS/EEDL.pdf) describes shell-resolved electron data and binding-energy updates. The [ENDF-6 manual](https://www.oecd-nea.org/dbdata/data/endf102.htm) specifies the File 23 representation. The source-data year and report year are distinct: NDS-226 is dated December 2017. With cross sections in barns, lengths in angstroms and densities in atoms per cubic angstrom,

$$
n_a\,[\mathrm{\mathring A}^{-3}]\,
\sigma_{ai}\,[\mathrm{barn}]\,\Delta s_j\,[\mathrm{\mathring A}]
\times10^{-8}
$$

is a dimensionless vacancy expectation. Equivalently, convert density by $10^{24}$ and path length by $10^{-8}$ to centimetre units, and convert barns by $10^{-24}$. Incident energy conversion is $T_{\rm eV}=10^3T_{\rm keV}$. Linear interpolation is valid only for an ENDF region declaring that law; zero outside the table is a model boundary rather than a high-energy extrapolation.

The [XrayDB API](https://xraypy.github.io/XrayDB/python.html) documents the Elam-derived edge fluorescence yields and transition intensities, plus core-hole widths in eV from Krause--Oliver and Keski-Rahkonen--Krause. For independent exponentially decaying initial and final holes, convolution of their Cauchy spectral densities adds their half widths. Consequently

$$
\Gamma_{aif}=\Gamma_{ai}+\Gamma_{af},\qquad
\gamma_{aif}=\frac{\Gamma_{aif}}{2},\qquad
L_{aif}(E)=\frac{\gamma_{aif}}
 {\pi[(E-E_{aif})^2+\gamma_{aif}^2]}.
$$

This agrees with the line-width addition rule in [Krause and Oliver (1979), Eq. (2)](https://srd.nist.gov/jpcrdreprint/1.555595.pdf). The profile has unit integral over the real axis, half its maximum at $E-E_{aif}=\pm\gamma_{aif}$, and units of inverse eV. Averaging unresolved final-level widths and assigning zero to a missing final width are explicit approximations, not consequences of the lifetime convolution. A grouped transition is generally a weighted sum of profiles rather than one profile with an arithmetic-mean width. The single-line model does not reproduce the empirical multi-Lorentzian fits of [Hölzer et al. (1997)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.56.4554).

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

and includes a transition only if its centre lies in the admitted window. This is a conditional, truncated Lorentzian. It preserves the complete line yield on a chosen grid but differs from physically cropping an unconditioned spectrum by the exact factor $1/P_{aif,W}$. Changing the window can therefore change the density in overlapping bins. At a window edge $P_W$ approaches one half for a distant opposite boundary, doubling the retained half-line. This accepted numerical convention must not be called conservation under physical aperture selection. The infinite-window limit removes the factor.

For isotropic photons the probability per solid angle is $1/(4\pi)$. The escape probability from source position $\mathbf r_j$ along $\hat{\mathbf n}$ is

$$
A_j(E)=\exp\left[-\sum_m\mu_m(E)\ell_{jm}(\hat{\mathbf n})\right],
$$

where each material path $\ell_{jm}$ is nonnegative. Passive absorbers contribute to this exponent but not to $n_a$ in the source term. A line-centre attenuation approximation takes $A_j(E)\simeq A_j(E_{aif})$; it is accurate when attenuation varies little across the admitted profile. The exact absorbed bin would integrate $L(E)A_j(E)$ instead. The estimator expected under the stated discretization is

$$
\boxed{S_k=\frac{1}{4\pi N_e\Delta E_k}
\sum_{j,a,i,f}n_a\sigma_{ai}(T_j)\Delta s_j
\omega_{ai}b_{aif}A_j(E_{aif})w_{aif,k}.}
$$

$N_e$ counts all incident macro-electrons in the selected cohort, including those producing no retained segment. This is a photon-number density per eV, per sr and per incident electron; there is no photon-energy factor and no extra speed factor. A track-length estimator integrates path length, not residence time.

Cheap filters pass under those assumptions: zero density, path length, cross section or fluorescence yield gives zero; absorption is positive and at most one; an opaque path gives zero and zero optical depth gives unity. Splitting a segment while holding energy, escape geometry and source composition fixed preserves its yield. A zero-width line inside one bin tends to a delta mass; on a bin boundary its symmetric limit divides between adjacent bins. Unattenuated integration over bins and solid angle recovers the direct photon expectation per incident electron. Real transport subdivision changes energy and position quadrature and is not required to be exactly invariant.

Characteristic emission is statistically incoherent with PXR/CBS amplitudes. If the two total-spectrum choices differ only in coherent versus incoherent PXR/CBS treatment, the same characteristic intensity must be added once to each. It must not be added to amplitudes or multiplied by an interference factor.

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
| One common additive intensity | `runner/__init__.py` keeps `spec_characteristic` separate from `spec`/`spec_coherent`; `_spectral_components.line_spectrum` adds it once to the selected line spectrum | Matches |

The EEDL threshold and xraydb line/edge energies originate in different compilations. They are joined by element and shell; this review verifies that join and the direct-emission estimator, not a self-consistent energy-conserving relaxation cascade or the physical accuracy of every atomic datum.

### Independent numerical evidence

All computations below used the project runner with `PYRITE_MC_BACKEND=cpu`. No transport sweep or GPU calculation was performed.

- A small fixed-column reader, independent of `endf-parserpy` and the PyRITE parser, read packaged carbon MAT 600, MF 23, MT 534. Its raw header gives binding energy 288 eV and 25 samples with interpolation law 2. The bracketing records are 25118.9 eV / 59832.9 barns and 39810.7 eV / 42689.6 barns. Direct scalar linear interpolation at 30000 eV gives $5.4137330932220685\times10^{-20}\,\mathrm{cm}^2$; the implementation gives $5.4137330932220697\times10^{-20}\,\mathrm{cm}^2$ (relative difference $2.22\times10^{-16}$). An assertion with relative tolerance $10^{-13}$ and **zero absolute tolerance** passes.
- For a 1000 eV line, FWHM 2 eV and bin edges 998 through 1002 eV, direct analytic integration captures 0.7048327646991335 of the full Lorentzian. The implementation sums to 1, exactly as the declared conditional model requires; its retained-bin density is larger by 1.4187762687605225. The independently computed conditional bin vector matches at relative tolerance $10^{-14}$ with zero absolute tolerance.
- For carbon at 30 keV, density $0.1\,\mathrm{\mathring A}^{-3}$, one 100 angstrom track, two incident electrons and zero attenuation, direct multiplication of the raw EEDL cross section and XrayDB API yields predicts $3.0156786314065755\times10^{-7}$ photons per sr per incident electron. The integrated output is $3.0156786314065765\times10^{-7}$, agreeing to $4.44\times10^{-16}$ relatively. This also checks that the electron with no segment remains in the normalization.
- Carbon's API branch intensities sum to 1.0000000972. Using the idealized sum of exactly one initially produced a relative difference of $9.72\times10^{-8}$; retaining the rounded source intensities resolves it. The implementation accepts source sums within 0.99 to 1.01 and does not renormalize them. Thus photon-yield preservation is exact relative to the supplied branch sum, and only approximate relative to $\omega_i$ alone. Carbon K-alpha-1 has API widths $0.0868+0.0045=0.0913$ eV, matching the constructed transition width.
- The focused command `PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_characteristic.py` passed **7 tests**. These cover parsing/provenance, relaxation joining, line integration, transparent-track subdivision, absorber separation and addition to both emission modes.

The existing carbon cross-section assertion uses `np.isclose` with its default absolute tolerance $10^{-8}\,\mathrm{cm}^2$. That tolerance is much larger than the reference value: even zero passes this particular assertion. It is therefore a weak regression anchor for the numerical cross section. The independent zero-absolute-tolerance check above verifies today's value; it does not repair the persistent test. A separate test improvement is recommended before treating that reference as a strong CI anchor.

### Scope corrections and evidence limits

Constant energy alone does **not** guarantee subdivision invariance when self-absorption varies along the track. For a uniform path with total optical depth 2, one midpoint gives attenuation 0.36787944117144233, two equal subsegment midpoints give 0.4148304099305316, and exact path integration gives 0.43233235838169365. This is ordinary midpoint-quadrature error, not a missing normalization factor. The invariant limit holds with identical attenuation weights or zero attenuation; the existing regression test explicitly sets attenuation to zero.

At the time of this review, the maintained physics page [Characteristic radiation](../../physics/radiation-physics/characteristic-radiation.md) needs two exact documentation corrections: its density formula uses $Yq_b/\Delta E_b$ after defining the unnormalized $q_b$, while the implementation requires $Yq_b/(\Delta E_b\sum_cq_c)$; its subdivision bullet needs the fixed-attenuation condition above. This verifier only edits the ledgered validation document and leaves those corrections to the owning context.

The positive-opacity checks assume finite nonnegative attenuation coefficients. The implementation replaces nonfinite coefficients, including positive infinity, by zero. Therefore a directly supplied infinite coefficient becomes transparent instead of opaque; this behavior is outside the finite-coefficient verdict. The physical opaque limit means increasing finite optical depth, for which the exponential tends to zero. This review did not establish reachability of nonfinite coefficients from valid bundled atomic data.

Source access was bounded explicitly. The ENDF-6 manual, EEDL survey and XrayDB API documentation were inspected online. The EEDL report is dated 2017; the packaged tape records evaluation in August 2023 and distribution in January 2025. Its packaged-byte hash test passes, but this review did not independently redownload the complete 2025 upstream tape. The Krause--Oliver paper text, including section 2's Eq. (2) linewidth-sum rule, was retrieved from an [Argonne-hosted copy of the primary paper](https://millenia.cars.aps.anl.gov/archives/list/ifeffit%40millenia.cars.aps.anl.gov/message/W3G725L4WFZ7M3ZUYX543D73VGSXICYR/attachment/7/Krause1979.pdf) after the direct NIST PDF endpoint failed. Elam and Keski-Rahkonen--Krause numerical data were checked through the primary XrayDB API documentation and installed database, not by independently digitizing those original papers. Hölzer's publisher abstract supports the stated multi-Lorentzian model limitation; no reproduction of its measured spectra is claimed.

### Independent verdict and suggested ledger edit

**Re-derived:** the code matches the direct-vacancy, conditional-window, line-centre-attenuation track-length estimator with finite nonnegative attenuation coefficients. Units, signs and the scoped limiting cases pass. This does not certify a physically cropped full Lorentzian spectrum, a complete atomic relaxation cascade, empirical line-profile accuracy, or subdivision invariance under changing attenuation weights. Human sign-off remains pending.

Suggested human-applied ledger update: change `characteristic-radiation` from `filtered` to `rederived`; replace the pending independent-review note with this dated result and its conditional-window, midpoint-quadrature and finite-attenuation scope; record the independent raw carbon interpolation and incident-electron normalization checks; flag the overly permissive cross-section test absolute tolerance. Do not apply `signed-off`.

### Coordinated documentation checks, 2026-09-13

Supplied by the coordinating context after this verification was written. `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs` builds this page with no new Sphinx warning. `uv run pyrite-dev test tests/dev/test_docs.py tests/dev/test_validation_ledger.py tests/dev/test_doc_blocks.py` passed **68 tests**, covering documentation structure, ledger schema/anchor resolution and doc code blocks. The rendered page carries 63 parsed math nodes, with no unparsed dollar-delimited or backslash-parenthesis math left as literal text in the HTML, so the display and inline math above render.

The two documentation corrections requested above were applied to the maintained physics page by the coordinating context: its bin density now reads $Y\widehat q_b/\Delta E_b$ with $\widehat q_b=q_b/\sum_cq_c$ stated explicitly, and its subdivision bullet now carries the fixed-attenuation condition and the optical-depth-2 midpoint-quadrature numbers. No code was changed. The weak `np.isclose` absolute tolerance on the carbon cross-section assertion is left as a recorded ledger finding for a separate test-improvement task.

## Issue #91 implementation-context note, 2026-09-21

Superseded by the EADL cascade of 2026-09-27 below; kept as the record of `v5`/`v6`. The transfer those versions applied was

```{math}
:label: eq-characteristic-ck-transfer

T^a_{L_1L_1}&=1-f_{12}-f_{13}, & T^a_{L_1L_2}&=f_{12}, & T^a_{L_1L_3}&=f_{13},\\
T^a_{L_2L_2}&=1-f_{23}, & T^a_{L_2L_3}&=f_{23}, & T^a_{ii}&=1\ \text{otherwise},
```

with xraydb's total Coster--Kronig probabilities $f_{ij}$ and stored product $\sum_{i'}T^a_{ii'}\omega_{ai'}I_{ai'\ell}$.

`v4` to `v5` adds one physical process: L-shell Coster--Kronig redistribution of primary vacancies, {eq}`eq-characteristic-ck-transfer`, applied in `_l_shell_vacancy_transfer` and folded into `line_yield_per_vacancy` at table construction. Nothing in the Lorentzian convention, the window treatment, the bin-edge clamp, the escape geometry, or the transport floor changes, and the hot loop is untouched — `line_yield_per_vacancy` was already `(n_shell, n_line)` and simply becomes dense, so the change costs nothing per segment.

Why it is a correction rather than a refinement: the Elam $\omega_i$ are Krause pure subshell radiative yields — Au's 0.107 / 0.334 / 0.320 reproduce Krause & Oliver (1979) exactly — so Coster--Kronig transfer is by construction *absent* from them. Applying $\omega_i$ to the primary population therefore leaves the L holes in the subshell that created them, which is not where they radiate from.

Checks that do not use the implementation to construct their expected values:

- **Row normalization.** Each row of $T^a$ sums to one by inspection of {eq}`eq-characteristic-ck-transfer`. Sampled across $3\leq Z\leq98$, the L-shell outflow $f_{12}+f_{13}$ never exceeds one — the tightest case is Mn at 0.997, leaving $T^a_{L_1L_1}=0.003$ — so the diagonal never goes negative and the matrix is a vacancy distribution, not a rescale. A source table that violated this fails closed with a `ValueError`.
- **Triangularity.** Decay fills a hole from a less-bound subshell, so vacancies only move to higher indices and $T^a$ is upper triangular. This is the property that will let the full EADL cascade of #91 be solved exactly by a finite Neumann series rather than an iteration with a cutoff.
- **Limiting case.** xraydb tabulates no L Coster--Kronig for $Z\leq11$, so $T^a=\mathbb{1}$ there and the estimator reduces to the pre-cascade product exactly. The carbon anchors in `tests/montecarlo/test_characteristic.py` are unchanged, which is the regression evidence for that limit.
- **Magnitude, computed from the source tables alone.** With $n=T^{a\mathsf T}N$ from the Elam $f_{ij}$ and the EEDL primary populations, total L emission $\sum_i n_i\omega_i$ over $\sum_i N_i\omega_i$ is 1.235 (Cu, 30 keV), 1.232 (Cu, 15 keV), 1.114 (Mo, 60 keV), 1.079 (Au, 100 keV), 1.064 (Au, 30 keV) and 1.033 (Ta, 30 keV). The sign is forced: $\omega_{L_3}>\omega_{L_2}>\omega_{L_1}$ for every element with tabulated CK, and transfer only moves holes outward, so redistribution can only raise total L emission. Per-line ratios move much further than totals — Cu's L1-origin share of L photons falls from 0.031 to 0.0005, so Lβ3,4/Lα was previously wrong by more than 10x.
- **No double counting.** $T^a_{L_1L_3}=f_{13}$ is asserted to differ from $f_{13}+f_{12}f_{23}$, which is what the total-vs-direct confusion would produce.

This supersedes the "Direct branching $\omega_i b_{if}$" row of the frozen 2026-09-13 comparison table above: the stored product is now $\sum_{i'}T^a_{ii'}\omega_{ai'}I_{ai'\ell}$. The cutoff-zeroing behaviour of `_xraydb_line_yields` is unchanged.

The model marker moves to `l-shell-ck-lorentzian-v5`, so dataset identities and case-content keys fork from `v4`; `v4` records hold un-redistributed L line yields and are not the same spectrum. The shipped profile digest pins in `tests/materials/test_profiles.py` were re-minted accordingly.

This is an implementation-context review only. Units, row normalization, the $T^a=\mathbb{1}$ limit, the forced sign, and the double-counting guard all have anchors, but fresh-context source-to-code validation of the Coster--Kronig claim — in particular whether Elam's tabulated $f_{ij}$ are the total probabilities this derivation assumes, checked against Krause (1979) or Campbell (2003) directly rather than through xraydb — is still pending. The ledger status therefore stays `filtered`; no human sign-off is claimed.

## Issue #91 EADL cascade implementation-context note, 2026-09-27

`v6` to `v7` replaces the xraydb relaxation model with the EADL cascade of {eq}`eq-characteristic-eadl-cascade`. The L-shell-only transfer {eq}`eq-characteristic-ck-transfer` and its Elam $\omega_iI_{i\ell}$ product are removed. Changes in behaviour:

- **Cascade.** Every primary vacancy bound above 50 eV relaxes through all EADL radiative and nonradiative transitions: K-fed L, L-fed M/N, and L- and M-shell Coster--Kronig. `vacancy_cascade` builds $D^a$ and evaluates $V^a$ by forward substitution. `line_yield_per_vacancy` $=V^a[p,:]\,R^a$ is precomputed, so the hot loop's `shell_sigma_cm2 @ response` is unchanged.
- **Cutoff contract.** `relaxation_cutoff_eV` bounds propagation by binding energy: subshells at or below it neither decay nor act as primaries, and photons at or below it are still dropped. Previously it only filtered emitted-line energy.
- **Lines.** There is one line per EADL radiative transition above 50 eV. xraydb energy and label are used where xraydb tabulates the same level pair, including grouped `M4,5`-style levels, and the EADL energy and an IUPAC `i-j` label otherwise, recorded in `line_source`. Across $3\leq Z\leq98$ no level pair is claimed by two xraydb lines, and every EADL-only line above 50 eV has an xraydb initial-level width. Lines are not pruned. EADL-only lines carry 1.5% (Cu) to about 10% (U) of a representative weighted photon yield, and they raise the line count to 26 (Cu), 108 (Au) and 146 (U), which costs proportionally more in the per-line escape loop and in line-grid seeding.
- **Yields.** EADL radiative yields are the default. `fluorescence_yields="elam"` rescales each subshell's radiative branch to xraydb $\omega_i$ and its nonradiative branch to $1-\omega_i$. Both choices are part of the marker.

Checks built from data or algebra the implementation does not use for its own result:

- **EEDL sum rule.** $\sum_{MT=534}^{572}\sigma$ against MT=522 read directly from the tape agrees to $10^{-5}$ for C, Cu and Au at 1 keV, 10 keV, 100 keV and 1 MeV. This covers the designator map, the barn units, and missing or doubled subshells. It replaces the carbon regression pin as the independent evidence for the cross-section read.
- **Bethe band.** EEDL/Bethe with $b=0.9$, $c=0.65$ and EADL occupancies lies within 0.85--1.15 for Si K, Cu K/L2/L3 at 30 keV and Au L3/M5 at 100 keV.
- **EADL invariants, Z=1--100.** The worst per-subshell $|\sum F-1|$ is $1.7\times10^{-6}$; tolerance is $10^{-5}$. Occupancies sum to $Z$ exactly. Every daughter is present and strictly later in cascade order; the only equal-binding pairs are Mg, Al and Si $L_2/L_3$, ordered by designator. 204 nonradiative transitions carry ETR $=0$, EADL's clamp for energetically marginal super-Coster--Kronig electrons, so they are accepted. No radiative transition has a nonpositive energy.
- **Exactness.** $V^a$ matches `numpy.linalg.inv` of $\mathbb{1}-D^a$ to $10^{-12}$, and $(D^a)^n=0$ for Si, Cu and Au.
- **Probability.** Each decay row of $D^a$ sums to $\omega_i+2(1-\omega_i)$: one hole consumed, one or two created.
- **Energy.** The budget identity closes within $5\times10^{-6}$ relative for C, Si, Cu and Au K and for Cu and Au L primaries. The transition-energy defect is below 1.5% of $B_p$ for these cases (corrected after the independent verification below: the bound holds for all K primaries, while L primaries reach about 6%). Cu K emits 3521.7 eV of photons per 8986 eV hole, consistent with $\omega_K\approx0.43$.
- **Limits.** A cutoff above every binding energy gives $V=\mathbb{1}$. A 1.2 keV cutoff in Cu removes every L-origin line and leaves K lines at their per-decay EADL probabilities.
- **Source disagreement recorded, not reconciled.** EADL/Elam $\omega$ ranges over 0.87--1.20 for K and 0.47--2.0 for L subshells above 200 eV from about Ti up; lighter elements reach 7.2 (S L1), as found by the independent verification below. Cu Coster--Kronig is $f_{12}=0.240$, $f_{13}=0.572$ and $f_{23}=0.009$ in EADL, against 0.30, 0.681 (total) and 0.47 in xraydb's Elam tables. Resolving which is right needs measured L-line intensity ratios. This is the main open physics question for L spectra.

The marker moves to `...-eadl-cascade-eadl-yields-lorentzian-segment-escape-v7` and now includes the EADL checksum prefix. Shipped profile digest pins in `tests/materials/test_profiles.py` were re-minted for the identity fork.

Still pending: comparison of K, L and M cascade spectra against an independent cascade implementation (PENELOPE `pdrelax`, Geant4 G4EMLOW `fluor/`, or EGSnrc) for a low-, mid- and high-Z element, and fresh-context source-to-code validation of {eq}`eq-characteristic-eadl-cascade` and the ENDF MF=28 field mapping. This is an implementation-context review only; the ledger row stays `filtered` and no human sign-off is claimed.

## Independent verification of the issue #91 EADL cascade, 2026-09-27

Fresh-context verification of the `v7` cascade (`...-eadl-cascade-eadl-yields-lorentzian-segment-escape-v7`) at commit `eafc6581` (cascade files identical to `a655a414`). The verifier read the ledger row, the physics page, and this write-up, then derived the cascade and parsed the packaged `EADL2025.ALL` with a scratch fixed-column ENDF-6 reader that uses no PyRITE code. The implementation bodies of `eadl_relaxation.py` and `characteristic.py` were read only after the derivation and the independent numbers below were fixed.

### Sources consulted

- The packaged EPICS2025 EADL tape itself: raw MF=28/MT=533 records for C, Si, Cu, Au and U, and every element for Z=3--99.
- xraydb 4.5.8 (Elam, Ravel & Sieber 2002 tables): edges, fluorescence yields, Coster--Kronig probabilities, and line lists.
- Not accessible in this context: the ENDF-102 manual text for File 28, Perkins et al. UCRL-50400 Vol. 30 (EADL tables), and Krause (1979). ENDF field semantics are therefore established from the manual layout as recalled and from internal consistency of the tape, not from the manual text. The "Krause" numbers are xraydb's, which come from Elam's compilation.

### Derivation, fixed before reading the implementation

**MF=28/MT=533 layout.** A HEAD record carries NSS. Each subshell is a LIST record with $C_1={\rm SUBI}$, $N_1=6(N_{\rm TR}+1)$, and $N_2=N_{\rm TR}$. Its body is ${\rm EBI},{\rm ELN},0,0,0,0$, followed by $N_{\rm TR}$ sextets ${\rm SUBJ},{\rm SUBK},{\rm ETR},{\rm FTR},0,0$. SUBJ is the subshell whose electron fills the vacancy. SUBK is the subshell that ejects the electron, or $0$ for a radiative transition. ETR is the photon or electron energy in eV, and FTR is the fractional probability. Designators are sequential: $1={\rm K}$, $2={\rm L}_1$, $3={\rm L}_2$, $4={\rm L}_3$, $5={\rm M}_1$, and so on up to $39={\rm Q}_3$. The 39 designators match EEDL MT 534--572.

The tape confirms this reading without circularity. In Cu, SUBI $=1..10$ has occupancies $2,2,2,4,2,2,4,4,6,1$, which are $2j+1$ for K through N$_1$ and sum to $Z=29$. The EBI values 8986, 1103, 958, 938, 127, 82 and 80 eV match xraydb's K, L$_{1-3}$ and M$_{1-3}$ edges (8979, 1097, 952, 933, 122.5 eV) within the expected single-vacancy offset. Radiative ETR is close to $B_i-B_j$. For nonradiative transitions, ETR is close to $B_i-B_j-B_k$: Cu L$_1$--L$_3$M$_1$ has ETR 27.4 eV against 38 eV.

**Cu K spot values from the raw tape.** Kα$_1$ (K--L$_3$) has ETR 8005.71 eV and FTR 0.255668. Kα$_2$ (K--L$_2$) has 7984.67 eV and 0.131119. Kβ$_3$ and Kβ$_1$ (K--M$_{2,3}$) have 0.0158899 and 0.0310908, and K--M$_{4,5}$ has $4.38737\times10^{-5}$ in total. K--L$_1$ is absent. Hence $\omega_K=0.43381$ (xraydb: 0.441), Kα$_1$/Kα$_2=1.950$ (xraydb: 1.961), and EADL Kα energies are about 41 eV below the measured 8046.3/8026.7 eV, so preferring xraydb energies is justified.

**Vacancy counting.** Let $n_i$ be the expected number of vacancies that ever occupy $i$ for one primary vacancy in $p$. Each vacancy in a decaying subshell decays once. A radiative branch moves the hole to SUBJ. A nonradiative branch leaves holes in SUBJ and in SUBK, which is two holes in one subshell when SUBJ $=$ SUBK. Therefore

$$
D_{ij}=\sum_{b\in i,\ {\rm rad}}F_b\,\delta_{j,J_b}
+\sum_{b\in i,\ {\rm nonrad}}F_b\,(\delta_{j,J_b}+\delta_{j,K_b}),
\qquad
n^{\mathsf T}=e_p^{\mathsf T}+n^{\mathsf T}D
\;\Rightarrow\;
n^{\mathsf T}=e_p^{\mathsf T}(\mathbb 1-D)^{-1}.
$$

The row sum is $\sum_j D_{ij}=\omega_i+2(1-\omega_i)=2-\omega_i$. If every daughter is less bound than its parent, $D$ is strictly upper-triangular in decreasing-binding order and $D^n=0$. Then $V=\mathbb 1+VD$ can be solved column by column: $V_{\cdot c}=e_c+\sum_{q<c}V_{\cdot q}D_{qc}$. The result is exact, not a truncated series. A cutoff that makes subshell $i$ non-decaying zeroes row $i$ of $D$ and of $R$. With every row zeroed, $V=\mathbb 1$. The line yield is $Y_{p\ell}=\sum_i V_{pi}R_{i\ell}$.

**Energy budget.** For each decaying $i$, $\sum_b F_b=1$, and

$$
B_i=\sum_b F_b\big[E_b+B_{J_b}+[{\rm nonrad}]B_{K_b}+\delta_b\big],
$$

where $\delta_b$ is the defect of branch $b$. Weight by $V_{pi}$ and sum over the decaying subshells. The identity $\sum_i V_{pi}D_{ij}=V_{pj}-\delta_{pj}$ turns the daughter binding energies into a telescoping sum, which gives

$$
B_p=E_\gamma+E_e+\sum_{j\ \rm terminal}V_{pj}B_j+\Delta_p .
$$

A terminal subshell is below the cutoff *or* has no EADL transitions. Omitting the second case breaks the closure: U O$_{4,5}$ at 110/101 eV and P$_1$ at 52 eV have $N_{\rm TR}=0$. The scratch cascade first omitted that case and U failed to close by 4%. With the case included, it closed.

**Elam rescale.** Scale the radiative branch by $\omega^E/\omega^A$ and the nonradiative branch by $(1-\omega^E)/(1-\omega^A)$. The total is $\omega^E+(1-\omega^E)=1$. The same radiative factor must multiply $R$, and the $D$ row sum becomes $2-\omega^E$.

### Comparison with the implementation

| item | independent result | implementation | verdict |
| --- | --- | --- | --- |
| field map | SUBI, EBI, ELN, NTR, (SUBJ, SUBK, ETR, FTR); SUBK $=0$ is radiative; sequential designators | `_extract_eadl_relaxation` reads the same fields; `EEDL_SUBSHELL_LABELS` is 1=K, 2=L1, ... | pass |
| capacity | $2j+1$ with $j=\lfloor(m+1)/2\rfloor-\tfrac12$ | `2 * ((index + 1) // 2)` | pass |
| cascade order | $(-B_i,\ \text{designator})$; daughters strictly later | `_cascade_key`; rank check rejects violations | pass |
| $D$ | the equation above, including $+2F$ when $J=K$ | `daughters[row, first] += ...; daughters[row, second] += ...` | pass; max $\lvert\Delta D\rvert=0$ for C, Si, Cu, Au, U |
| $V$ | dense `inv` of $\mathbb 1-D$ from the scratch reader | forward substitution | pass; max $\lvert\Delta V\rvert\leq 9\times10^{-16}$ |
| $V=\mathbb 1$ limit | cutoff above every $B_i$ | `vacancy_cascade(r, 1e7)` | pass (exact identity) |
| budget identity | terminal $=$ below cutoff or $N_{\rm TR}=0$ | `decays = B > cutoff and (n_rad + n_aug)` | pass; closure $\leq 10^{-6}$ relative, including U |
| Elam rescale | radiative $\times\omega^E/\omega^A$, nonradiative $\times(1-\omega^E)/(1-\omega^A)$, same factor in $D$ and $R$ | `_branching_scales` (nonradiative denominator is $\sum F_{\rm aug}$, which equals $1-\omega^A$ to $10^{-5}$); applied in `vacancy_cascade` and to `emission` | pass; Cu and Au: $\sum_\ell R_{i\ell}=\omega^E_i$ and $D$ row sum $=2-\omega^E_i$ for every decaying subshell |
| line yields | Cu K primary: K--L$_3$ 0.255668, K--L$_2$ 0.131119, K--M$_{4,5}$ $4.38737\times10^{-5}$; cascade-fed L$_3$--M$_5$ 0.0035816, L$_3$--M$_1$ 0.0039117 | `line_yield_per_vacancy` K row: identical to printed precision | pass |

Line join (`_xraydb_line_index`, `_expand_levels`, `_parse_characteristic_file`): Kβ$_5$ (K--M$_{4,5}$) claims both K--M$_4$ and K--M$_5$, whose EADL probabilities sum into the one xraydb-energy column. Kα$_3$ (K--L$_1$, xraydb intensity $3\times10^{-4}$) has no EADL radiative branch and is omitted. Lines with no xraydb level pair take EADL `ETR` with `line_source="eadl"`: Cu has 14 of 26 lines from EADL, including the M1/E2 lines L$_2$--M$_3$ and L$_3$--M$_{2,3}$ and the intrashell L$_1$--L$_{2,3}$ lines at 128 and 149 eV. Tables for Z=3--98 load in both yield modes without any xraydb pair being claimed twice. Line counts are 26 (Cu), 108 (Au) and 146 (U), matching the note. The EADL-only share of the "representative weighted photon yield" (1.5% to 10%) was not reproduced, because that weighting is not defined in the write-up.

Reported numbers checked from raw data: Cu K photon energy 3521.7 eV per 8986 eV hole (pass). EADL Cu $f_{12}=0.2401$, $f_{13}=0.5721$, $f_{23}=0.00895$ (pass). xraydb Cu $0.30$, $0.681$, $0.47$ (pass, as xraydb values; the `total=True` query returns the same numbers). EADL/Elam $\omega_K$ over 0.867 (Ne) to 1.201 (C) for K subshells above 200 eV (pass).

### Discrepancies (documentation and declared bounds, not code)

1. **Transition-energy defect bound is false as stated.** The claim appears in the ledger Notes, the physics page ("For K and L primaries bound above about 100 eV the defect is under 1.5%"), this write-up's Assumptions and 2026-09-27 note, and the docstring of `test_cascade_energy_budget_closes_within_the_eadl_transition_defect`. An independent scan over Z=3--99 of K, L$_1$, L$_2$ and L$_3$ primaries bound above 100 eV found:
   - K: maximum $\lvert\Delta_p\rvert/B_p$ is 1.44%, so the claim holds for K.
   - L: 35 of the 353 K and L primaries exceed 1.5%, and all 35 are L primaries.
   - Worst above 100 eV: Si L$_2$ at 104 eV, $-6.3\%$ (the page already mentions this case).
   - Other L$_1$ and L$_{2,3}$ cases: K L$_1$ at 381 eV, $+5.96\%$; Ca L$_1$ at 441 eV, $+5.46\%$; P L$_2$ at 135 eV, $-5.0\%$; Ti L$_1$ at 567 eV, $+4.44\%$.
   - High-binding L$_1$ cases: Ga L$_1$ at 1302 eV, $+2.95\%$; Sr L$_1$ at 2219 eV, $+2.43\%$; Ba L$_1$ at 5991 eV, $+1.78\%$.
   - Failing L$_1$ primaries span Z=13--57.

   The test passes only because its parametrization (C, Si, Cu K; Cu L$_1$, L$_3$; Au K, L$_3$) misses these cases. The identity itself is exact; only the declared bound is wrong. Suggested wording: "K primaries under 1.5%; L primaries up to about 6% below 600 eV and up to 3% for L$_1$ through Z≈57."
2. **The L-subshell $\omega$ ratio range is scope-limited.** "EADL/Elam $\omega$ spans 0.47--2.0 for L subshells above 200 eV" holds only from about Ti upward. Lighter elements with L edges above 200 eV reach S L$_1$ 7.18, Cl L$_1$ 5.18, Ca L$_2$ 3.70, Ca L$_3$ 3.50 and K L$_1$ 3.32. The minimum, 0.471 (W L$_1$), is confirmed.
3. **Attribution.** The 0.30, 0.681 and 0.47 values are xraydb/Elam Coster--Kronig probabilities. This review could not check them against Krause (1979) directly, so "Krause's" should read "Elam/xraydb (after Krause)" unless someone checks the primary table.

### Validity-limit completeness

The stated limits are accurate: untransported Auger electrons, independent single-vacancy rates, EADL energies for M/N lines missing from xraydb, no secondary fluorescence, and the sub-keV caveat. Two limits are missing:

- **The shape of EADL radiative branches is not validated.** For 3d metals, EADL's L$_3$ radiative branching gives L$_3$--M$_1$/L$_3$--M$_5$ (Lℓ/Lα$_1$) of 2.93 (Fe), 1.49 (Ni), 1.09 (Cu) and 0.83 (Zn). xraydb gives 0.12, 0.09, 0.08 and 0.07. For Au the two sources agree (0.065 vs 0.052). The Elam option keeps the EADL branch shape, so it cannot correct this. Together with the Coster--Kronig disagreement, 3d-metal L-line ratios are unvalidated in both yield modes. The limit list should say so.
- **The Elam option also rescales M, N and O subshells** to xraydb yields. For outer levels these are coarse (Au O$_{1-3}$ $\omega=0$), and any subshell with EADL $\omega=0$ or no nonradiative branch is silently kept at EADL. This is documented in the `_branching_scales` docstring but not on the physics page.

Minor, not a defect:

- Primary selection uses the EEDL binding energy, while the cascade uses the EADL one. A mismatch across 50 eV only moves zero-yield rows.
- `relaxation_energy_budget` has no `fluorescence_yields` argument, so the energy identity is verified only for the default EADL mode.

### Test run

`uv run pyrite-dev test tests/montecarlo/test_eadl_relaxation.py tests/montecarlo/test_characteristic.py tests/montecarlo/test_eedl_ionization_consistency.py`: 60 passed.

### Verdict

- **Claim**: `characteristic-radiation` (issue #91 cascade). Code: `montecarlo/eadl_relaxation.py::{_extract_eadl_relaxation,vacancy_cascade,relaxation_energy_budget}` and `montecarlo/spectrum/characteristic.py::{_parse_characteristic_file,_cascade_yields}`. Source: EADL ENDF-6 MF=28/MT=533 and {eq}`eq-characteristic-eadl-cascade`.
- **Filters**: units pass; limits pass ($V=\mathbb 1$; exact nilpotent substitution); signs and conventions pass.
- **Re-derivation**: matches for the field map, $D$, $V$, $Y$, the budget identity and the Elam rescale. The difference is in declared bounds: the transition-energy defect bound for L primaries, where the first failing case is K L$_1$ at 381 eV with 5.96% against the 1.5% declared.
- **Verdict**: `rederived` for the cascade equations and their code. Documentation discrepancies 1--3 and the missing branch-shape limit need text fixes; none changes a computed number.
- **Suggested ledger change (human applies)**: change the status from `filtered` to `rederived` only after the defect-bound sentence is corrected to K < 1.5% and L up to about 6%, and the L-$\omega$ range is scoped to Z ≳ 22. Also record the independent MF=28 and cascade verification and remove "fresh-context source-to-code verification of the cascade are pending" from Notes. Comparison against PENELOPE/Geant4/EGSnrc and measured L-line ratios remains pending. No sign-off.
