# Optical, infrared and coherent beam-radiation requirements

Research investigation for [#533](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/533),
20 October 2026; implementation inventory at `389ad59c`. Includes diffracted
transition radiation (DTR). This note changes no production physics, continuum
floor, material tables or numerical defaults. Proposed claims below are
**unverified**; sources identify future verification work, not completed anchors.

Status update, 21 October 2026: Stage 1 (#614) has since landed.
`optical-material-response` is now an implemented, `anchored` claim in the
[crystallography and atomic-data ledger](../../validation/ledger-crystallography-atomic-data.md#optical-material-response)
(`materials/optical.py`, pinned CC0 Rakić Au record, about 0.2–5 eV). The
remaining six `optical-*` claims below are still research-only.

## Recommendation and band boundary

Build an opt-in optical diagnostic path around the existing planar TR/CTR
amplitudes, explicit measured dielectric data, and physical bunch normalization.
Add finite radiators and wave optics before advertising an instrument-level
bunch-length prediction. Keep the current X-ray path and its floor intact.
Start with separate cases sharing an electron-source description; a single
0.01 eV–100 keV case should follow only after component-specific validity and
integration contracts exist.

CTR is a classical correlation of radiation from different electrons. It does
not require a coherent quantum electron state. OTR angular distributions can
diagnose transverse beam properties even when the bunch has no optical temporal
coherence. Neither statement establishes that all optical cathodoluminescence
is TR: fluorescence, scintillation and plasmonic emission may need separately
bounded backgrounds.

The useful band depends on the bunch. With $E=\hbar\omega=hc/\lambda$,
0.05–3 eV corresponds to roughly 25–0.41 µm; 0.01 eV corresponds to 124 µm.
A smooth Gaussian with RMS duration $\sigma_t$ has temporal squared form factor

$$
B_t(E)=\exp[-(E\sigma_t/\hbar)^2],\qquad E_{1/e}=\hbar/\sigma_t.
$$

The catalog files `beams/gaussian_200fs.toml` and `beams/1_atto.toml` specify
`envelope_rms_fs = 200.0` and `0.001`, respectively, rather than FWHM durations.
Using $\hbar\simeq0.658212$ eV fs gives these *temporal-only* estimates:

| RMS duration | $E_{1/e}$ | Corresponding wavelength | Consequence |
| --- | --- | --- | --- |
| 200 fs | 0.00329 eV | 377 µm | $B_t(0.01\,\mathrm{eV})\simeq9.8\times10^{-5}$; at 0.05 eV it is below $10^{-100}$. Far-IR/THz coverage below 0.01 eV is needed to see the turnover. |
| 1 as | 658 eV | 1.88 nm | $B_t(3\,\mathrm{eV})\simeq0.99998$ and $B_t(12\,\mathrm{eV})\simeq0.99967$; optical/IR temporal coherence is nearly complete. |

Near-unity temporal coherence does not establish transverse coherence or
validated production of a 1 as bunch. These catalog beams are descriptions,
not measured beam-source guarantees. Conversely, a small form factor can
still matter when multiplied by a large physical population.

`Validation: optical-ctr-population` — research extension of the existing
`tr-coherent-bunch` and `coherent-physical-bunch-population` claims. The Gaussian
limit is a scale calculation, not a new simulation result.

## Implementation inventory

Paths in this section are relative to `src/pyrite/` unless stated otherwise.
These are candidate owners for follow-up work, not changes made here.

| Layer | Exists and reusable | Needs extension | New subsystem |
| --- | --- | --- | --- |
| Source fields | `montecarlo/spectrum/transition_radiation.py::planar_transition_radiation`: complex128, scalar complex permittivity per layer, outgoing Cartesian amplitude, infinite planar layers, globally straight passage. Crossing/history superposition already exists. | `montecarlo/runner/transition_emission.py::case_layer_eps` currently obtains Chantler independent-atom response and refuses gaps; explicit optical providers must reach this seam. Bound locally straight formation and fixed-trajectory energy loss. | Finite-source vector boundary fields; grating source fields if later selected. |
| Bunch statistics | `spectrum/coherent_population.py`: physical $N=Q/e$, distinct-pair scaling, sampling diagnostics; TR uses physical population in bunch mode. | Weighted correlated source ensembles, convergence for weak pair signals, phase-preserving optical propagation. | Instrument-specific inference and uncertainty, not another generic $N^2$ multiplier. |
| Materials | Composition, phase/orientation identity, provenance; TR accepts supplied scalar $\varepsilon(E)$. | Optical/IR dispersion provider, temperature, valid bands, interpolation, tensor capability declaration. | Tensor boundary solver for anisotropic production targets. |
| Propagation | Planar stack solve already includes Fresnel response, film interference and absorption; detector quadrature exists in `transition_radiation_detector.py`. | Preserve that boundary response without applying a second escape factor. Detector quadrature currently integrates a far-field source, even when its plane has a finite distance. | Finite-distance vector wavefronts, apertures, mirrors, lenses, chamber/windows, polarization transfer. |
| Detectors | Geometry, spectral arrays and persisted provenance can be reused. | Unit-safe energy/pulse and power outputs alongside photons; optical camera response. | Calibrated IR thermal sensors, spectrometer response and interferograms. |
| Grid/device | Continuum floor/support registry, bin/node semantics, coherent windows, CPU/CUDA TR machinery. | Component-specific optical support and independent angular/spatial/spectral convergence; device data upload. | GPU wave-optics operator and optical instrument reduction. |
| DTR | Experimental Born and two-beam amplitudes; joint-field accounting and transported-current diagnostic. | Independent validation remains absent; atomic response cannot be extrapolated to IR. | Optical grating diffraction, if wanted, needs its own macroscopic periodic boundary model. |

The current continuum floor is
`_photon_continuum_floor.py::photon_continuum_floor_eV`,
$\max(\hbar\omega_p,12\,\mathrm{eV})$. Its support registry records EEDL
photon spectra at 0.1 eV, EPDL photon cross sections at 1 eV, Chantler at
1.01 eV (strict interior in the dispersion accessor), and Eagle-XO measured
QE at 12 eV. Numerical table support is not optical validity: the existence
of a Chantler ordinate near 1 eV does not capture a solid's optical bands.
Changing just this floor would silently authorize unsupported emission,
attenuation and detectors.

## Mechanisms and source equations

### TR and CTR: first production target

For an ideal perfect conductor and a normally crossing electron, the backward
Ginzburg–Frank limit in SI units is

$$
\frac{d^2W_1}{d\omega\,d\Omega}
=\frac{e^2}{4\pi^3\epsilon_0c}
\frac{\beta^2\sin^2\theta}{(1-\beta^2\cos^2\theta)^2}.
$$

This is energy per angular frequency per solid angle, not photons/eV/sr.
The optical implementation must use the existing Maxwell continuity solve
with measured $\varepsilon(E)$ for real films; the perfect-conductor equation
is a limiting check. Assumptions: prescribed straight current, infinite flat
interfaces, local isotropic nonmagnetic media, outgoing passive waves. Matched
media and a vanishing film in identical exterior media must vanish; the
perfect-conductor angular pattern must be recovered in its limit.

`Validation: optical-planar-tr` — source: Ginzburg–Frank expression and the
Maxwell stack formulation already recorded in
[transition radiation](../../physics/radiation-physics/transition-radiation.md).
Published optical anchor: [Coenen, Vesseur and Polman (2011)](https://arxiv.org/abs/1107.3632),
30 keV electrons on gold, measured angular distribution. Reproduce its angular
pattern with collection/polarization response and stated optical constants;
an agreement with the ideal-conductor formula alone is insufficient.

CTR uses the same single-electron fields, summed with their actual crossing
time and position phases. It is not a new single-electron emission process.
For identical single-electron responses and independent offsets,

$$
\langle I_N\rangle=I_1\{N+N(N-1)\lvert F\rvert^2\},\qquad
F=\left\langle e^{i\omega t-i\mathbf k_\parallel\cdot\mathbf r_\parallel}\right\rangle.
$$

Here $t$ is the crossing time, not an equal-time longitudinal coordinate;
$\mathbf k_\parallel$ refers to the emitting interface and observed vacuum
wave. Face tilt and bunch-front tilt alter that mapping. Dividing by $N I_1$
gives enhancement $1+(N-1)\lvert F\rvert^2$, often approximated as
$N\lvert F\rvert^2$ only when the pair term dominates. The coherent *bunch*
energy instead scales approximately as $N^2\lvert F\rvert^2 I_1$.

`Validation: optical-ctr-population` — source: expansion of the squared
field sum, as in [Shkvarunets and Fiorito, Eq. (1)](https://arxiv.org/pdf/0712.2848).
Limits: $N=1$ retains only self radiation; $F=0$ gives $N I_1$; $F=1$ gives
$N^2 I_1$. The intended published bunch-diagnostic anchor is
[Schmidt et al. (2018), Figs. 35–38](https://arxiv.org/pdf/1803.00608), comparing
CTR reconstructions with an independent deflecting structure. These are
prospective comparisons; raw response/data availability must be checked.

### Finite targets, diffraction radiation and the pre-wave zone

A boundary can radiate even if the electron misses it: its evanescent field
drives currents around an aperture or edge (diffraction radiation, DR).
An electron-crossing ledger alone therefore cannot generate DR. Existing
finite electron geometry can describe the object, but does not solve its
electromagnetic boundary problem.

For a relativistic vacuum beam the characteristic transverse footprint is
of order $\gamma\beta\lambda/(2\pi)$; practical infinite-screen and far-zone
criteria can require sizes and distances of order $\gamma\lambda$ and
$\gamma^2\lambda$. Numerical factors depend on the required error and aperture.
Use these as screening scales, not universal acceptance tests. At $\gamma=100$
and $\lambda=10$ µm they are approximately 1 mm and 0.1 m using the latter
conservative scales. A finite target and short beamline can thus change the
detected spectrum even when the single-interface amplitude is correct.

Use vector equivalent surface currents with the finite-distance outgoing
Green function, retaining $e^{ikR}/R$ and polarization before taking any
far-field limit. [Shkvarunets and Fiorito, Eqs. (15), (17), (19)–(25)](https://arxiv.org/pdf/0712.2848)
provide a perfect-conductor boundary formulation. For the axisymmetric
relativistic disc limit, [Schmidt et al., Eqs. (11)–(15), Fig. 7](https://arxiv.org/pdf/1803.00608)
provide finite-radius and Fresnel-phase comparisons. These are theoretical
published anchors, not measured validation.

`Validation: optical-finite-boundary-field` — recommend a vector finite-foil
and aperture solver after planar TR. Limits: large screen and large distance
recover planar far-field TR; removing the screen removes radiation; grid
refinement conserves the collected flux. An experimental DR anchor is
[Measurements of coherent diffraction radiation (2001)](https://doi.org/10.1103/PhysRevE.63.056501).
Only its publisher abstract was accessible here; full geometry, calibration
and numerical data must be acquired before claiming reproduction.

### Smith–Purcell and existing grooves

For a beam moving parallel to a periodic grating of period $p$, vacuum order
$m>0$ obeys

$$
\lambda=\frac{p}{m}(\beta^{-1}-\cos\theta).
$$

This is a phase-matching relation, not an absolute-yield equation. Groove
shape, number of periods, conductivity, beam height and velocity distribution
determine coupling. Existing blazed-target geometry can supply dimensions;
its electron intersections and X-ray facet scoring do not calculate the
evanescent beam-to-grating field.
[Delerue et al. (2009), Eqs. (1)–(3)](https://arxiv.org/pdf/0912.1863)
describe a surface-current bunch-diagnostic model;
[Clarke et al. (2017)](https://arxiv.org/abs/1711.09018) address compound blazed
structures. Defer production Smith–Purcell: it needs a separate finite-period
Maxwell/current solver and specified beam clearance. A possible future
experimental anchor is [Doucas et al. (2002)](https://doi.org/10.1103/PhysRevSTAB.5.072802),
1.8 MeV electrons, millimetre wavelengths. That does not validate an optical
blazed grating. No implementation issue or new validated claim is asserted
for this deferred mechanism.

### Cherenkov and soft bremsstrahlung

Transparent layers can support Cherenkov light when $\beta n(E)>1$.
The lossless, isotropic, infinite-medium Frank–Tamm limit is

$$
\frac{d^2N}{dx\,d\lambda}
=\frac{2\pi\alpha}{\lambda^2}
\left(1-\frac{1}{\beta^2n^2(\lambda)}\right),\qquad
\cos\theta_C=\frac{1}{\beta n}.
$$

It vanishes below threshold. It is not a finite absorbing-film escape model;
bulk Cherenkov and TR must arise from a common boundary/current treatment
where their amplitudes overlap. Defer a production Cherenkov source until
transparent radiator materials and geometry are selected. The standalone
bulk source is already tracked by [#587](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/587);
it explicitly excludes detected optical/IR and finite-interface effects.
Do not duplicate that issue or gate its independent bulk calculation on an
optical instrument. The
[DIRC optical study](https://arxiv.org/abs/hep-ex/0305001) is a candidate anchor
for silica propagation/internal reflection, not an electron CTR benchmark.
No new Cherenkov claim is promoted by this inventory.

Low-energy bremsstrahlung cannot be obtained by extending the current
incoherent X-ray intensity table. The high-energy plasma approximation gives
Ter-Mikaelian suppression

$$
S(E)=\frac{E^2}{E^2+(\gamma\hbar\omega_p)^2}.
$$

[Anthony et al. (1996), Eqs. (4)–(9)](https://arxiv.org/pdf/hep-ph/9512381)
state that approximation and measured suppression for GeV electrons and
200 keV–20 MeV photons. Its assumptions fail as a universal optical dielectric
law. LPM, dielectric and finite-target effects cannot generally be combined
by multiplying independent suppression factors. Defer production optical
bremsstrahlung until a dispersive finite-history current model and a relevant
low-energy anchor exist. Bound it as an omitted background in each initial
TR diagnostic; the GeV experiment is context, not an optical validation.

### PXR, CBS and DTR at long wavelength

`Validation: optical-lattice-applicability` — research applicability claim.
For elastic diffraction of propagating waves, a nonzero atomic reciprocal
vector requires $\lvert\mathbf g\rvert\le2k$ in the isotropic lossless case.
With $\lvert\mathbf g\rvert=2\pi/d$ and $k=2\pi n/\lambda$, this requires
$\lambda\le2nd$. Optical/IR wavelengths greatly exceed ordinary atomic
spacings: ordinary propagating Bragg PXR and diffraction of free TR waves
by that atomic harmonic have no accessible Bragg order. The macroscopic
dielectric response is the useful starting point. Source: elastic wavevector
matching and the own/free-field distinction in
[Artru and Rullhusen (1998)](https://doi.org/10.1016/S0168-583X(98)00241-9),
also documented in [experimental DTR](../../physics/radiation-physics/diffracted-transition-radiation.md).
Only the paper's metadata/abstract, not a numerical reproduction, supports
the existing repository attribution.

This does **not** prove every lattice-induced contribution is exactly zero.
The charge field contains evanescent, off-shell spatial frequencies. Very
short passages, surfaces and grazing geometries can couple them without a
propagating incident Bragg photon. CBS involves the force frequency
$\omega(1-\hat{\mathbf n}\cdot\boldsymbol\beta)=\mathbf g\cdot\mathbf v$,
not just a photon Bragg condition; a nearly transverse reciprocal vector can
make its projection small. Such a tuned limit needs a finite-target and
possibly channeling treatment, not extrapolation of the current isolated
X-ray-line model or a declaration that all CBS vanishes.

The existing first-Born DTR model diffracts through an atomic susceptibility
harmonic; the two-beam model retains repeated coupling in one crystal layer.
Neither its independent-atom $\chi_{\mathbf g}$ nor its X-ray validity is
established below the continuum floor. Its ledger claims remain unverified
and its scorer is CPU-only. Do not relabel a surviving ordinary optical TR
field as DTR or add its signed correction as an independent positive source.

An engineered optical grating or multilayer with a wavelength-scale period
can diffract TR and out-couple an evanescent field. That is macroscopic
structure diffraction, with optical $\varepsilon(\mathbf r,E)$ and new
boundaries; passing an optical period as an atomic Miller reflection is
invalid. Defer a dedicated grating-DTR implementation until a target and
published optical benchmark are chosen. Use the finite-boundary/wave-optics
path for ordinary apertures. TR/CTR, optionally DR/CDR, replaces atomic
PXR/CBS lines as the initial optical bunch-diagnostic signal. Atomic DTR
validation remains owned by #581's existing claims; this note does not
upgrade them or propose an optical parity claim for that solver.

## Bunch inputs, sampling and phase retention

The factorized $F=F_tF_\perp$ requires independent crossing time and transverse
offsets, with the same single-electron response for each offset. For an
uncorrelated transverse Gaussian,
$\lvert F_\perp\rvert^2=\exp[-\mathbf k_\parallel^{\mathsf T}
\Sigma_\perp\mathbf k_\parallel]$.
Finite foils, beam clipping, energy-time chirp, divergence, tilt and optics can
make response and phase correlated. Then use the full response-weighted
single-history fields and their pair estimator, rather than a scalar Gaussian
factor. The existing longitudinal/transverse CXR windows have lattice-dependent
phase arguments; reusing their predicates for TR without deriving the optical
phase would be wrong.

For independent sampled histories with complex vector responses $A_j$, the
per-physical-electron estimator has the structure

$$
\widehat I=\frac1M\sum_j\lVert A_j\rVert^2+
\frac{N-1}{M(M-1)}\sum_{j\ne l} A_j\cdot A_l^*.
$$

This shows why the Monte Carlo history count $M$ cannot replace physical $N$.
The pair sum is real after conjugate partners are included. It needs $M>1$
when a nonzero pair contribution is estimated; unresolved pair estimates may
be negative and must not be clipped into false precision. Vary $M$ at fixed
$Q$, and $Q$ at fixed sampling law, as separate convergence checks.

Weighted beam-file particles are samples of a distribution, not automatically
individual electrons. Normalize positive sampling weights for the self term;
derive a distinct-pair importance estimator from the sampling law. Squaring
macro-particle charges adds an artificial self-noise term. A smooth density,
an i.i.d. electron ensemble and a correlated microbunched source have different
shot-noise/pair contracts and must be identified in provenance.

A CTR prediction needs bunch charge and repetition rate separately, the
crossing-time distribution or a resolved time/energy/position/direction source,
source bandwidth/resolution, jitter and uncertainty. For microbunching use
measured or accelerator-simulated modulation, with its sampling/noise model;
a smooth Gaussian is not evidence of that structure. Retain phases through
every optical element before squaring. Do not apply analytic offset factors
again when sampled fields already contain them. Space charge, wakefields,
collective bunch evolution and strong radiative depletion are outside this
initial passive-source proposal; initial fixed trajectories require an
application-specific energy budget even if the photon recoil is individually
small.

## Optical/IR material response

`Validation: optical-material-response` — provider contract, using
the passive $e^{-i\omega t}$ convention of the TR solver:

$$
\varepsilon(\omega)=\varepsilon_\infty-
\frac{\omega_D^2}{\omega(\omega+i\Gamma_D)}+
\sum_s\frac{f_s}{\omega_s^2-\omega^2-i\Gamma_s\omega},
\qquad \varepsilon=(n+i\kappa)^2.
$$

Positive damping/strengths give a passive local scalar model. This is a model
family, not a universal fit. Source and candidate metal anchors:
[Rakić et al. (1998)](https://doi.org/10.1364/AO.37.005271), Lorentz–Drude and
Brendel–Bormann optical models over their stated 0.1–6 eV interval.
Below that interval obtain additional data rather than extrapolating the fit.
Recover $\varepsilon\to\varepsilon_\infty$ at high frequency within the
mathematical model and the fitted DC conductivity in its metallic low-frequency
limit. Neither limit validates interband or phonon peaks.

The provider must cover free carriers/Drude damping, interband absorption,
IR-active phonons/Reststrahlen bands, measured tensor axes and temperature
where relevant. Surface oxide, film thickness, roughness, phase/polytype and
substrate can matter as much as nominal elemental composition. A crystal's
atomic number density is insufficient. Nonlocal response at very small
clearance needs a separate validity decision.

Select response per medium, component and band. Require dataset identifier,
version/hash, original reference, license, energy/wavelength units, complex
sign convention, temperature, orientation, sample state, range and uncertainty.
Reject gaps or unsupported tensors; do not average principal indices silently.
Interpolation must resolve resonances and preserve passivity. Check causality
with stated finite-band assumptions; a truncated Kramers–Kronig integral is
not a global certification. A band overlap with X-ray data needs an explicit
tested splice or disjoint validity, rather than arbitrary blending.

| Candidate source | Use | Redistribution decision |
| --- | --- | --- |
| [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database) | Pin individual measured $n,\kappa$ datasets; retain original references, phase, temperature, thickness and ranges. | Database is explicitly CC0. Check each selected record's notices and preserve attribution; this does not license copying publisher figures/PDFs. |
| Rakić metal models | Small explicit analytic benchmark/metal provider where fit coverage is suitable. | Publication access is not a data redistribution license. Prefer pinned CC0 records/parameters with provenance or obtain permission for other assets. |
| [NIST IR optical-property calibration](https://doi.org/10.6028/NIST.SP.250-94) and [index metrology](https://www.nist.gov/programs-projects/index-properties-optical-materials-012-mm-15-mm) | Window reflectance/transmittance and index/temperature uncertainty; these services are not a complete complex dielectric catalog. | Verify record-specific reuse terms and third-party notices before shipping data. |
| User ellipsometry/FTIR datasets | Actual radiator coatings, anisotropic crystals and windows. | Explicit owner license; retain raw-to-fit provenance and uncertainty. No default redistribution. |

Metals often reflect below their free-carrier plasma scale; transparent
dielectrics can transmit in the same band and reflect in phonon bands.
“Below plasma energy means opaque” is not a universal material rule.

## Propagation, detectors and observables

`Validation: optical-wavefront-propagation` — proposed vacuum propagation
equation for each transverse Fourier component:

$$
\widetilde{\mathbf E}(k_x,k_y;z+L)=
\widetilde{\mathbf E}(k_x,k_y;z)\exp(i k_zL),\quad
k_z=\sqrt{(\omega/c)^2-k_x^2-k_y^2},\quad\operatorname{Im}k_z\ge0.
$$

Use vector transversality and forward passive modes; evanescent components
decay. This is an angular-spectrum identity, not a model for an arbitrary
mirror. Source/anchor: [Schmidt et al., Appendix D](https://arxiv.org/pdf/1803.00608),
with finite-screen checks above. Required limits are identity at zero distance,
flux conservation for a lossless unobstructed propagating field, passive decay,
and agreement with the far-field transform after adequate sampling/padding.

Fresnel amplitudes, polarization rotations, optical path phase, total internal
reflection and thin-film interference replace a single Beer–Lambert escape
probability. Bulk attenuation may still be used inside a compatible wave
operator or an explicitly incoherent ray limit. The source-stack solver
already contains interface optics: applying the same interfaces again would
double count them. Define an exterior reference plane where the source ends
and the instrument begins.

An optical beamline description needs finite mirrors/lenses/apertures,
windows and coatings, chamber clearances, path lengths, atmospheric absorption,
alignment, collection NA and calibration. #341's Monte Carlo photon transport
can address incoherent intensities; sampled photon rays alone lose CTR phase
and diffraction. A distant angular detector remains useful as a source
observable when its asymptotic assumptions pass. It is insufficient as the
universal instrument abstraction in the pre-wave zone: use finite reference
planes and a calibrated optical transfer operator. Reject regimes beyond
paraxial/vector-FFT validity rather than approximating arbitrary chambers.

`Validation: optical-instrument-observable` — for spectral energy per pulse
$U_\omega$ incident on a calibrated detector, a linear energy-integrating
channel has signal $V=\int R_U(\omega)U_\omega\,d\omega$.
An ideal balanced two-port interferometer has the difference signal

$$
D(\tau)=\int_0^\infty R_U(\omega)U_\omega
\cos(\omega\tau)\,d\omega.
$$

Here $\tau$ is optical delay, with a moving-mirror displacement converted using
the actual double-pass geometry; gain/sign depend on port conventions.
A Martin–Puplett model must additionally propagate its wire-grid and roof-mirror
Jones operators, finite overlap and port imbalance. Source/experimental setup
anchor: [M. Geitz, *Bunch length measurements*, DIPAC 1999](https://proceedings.jacow.org/d99/papers/IT07.pdf),
Fourier-transform spectroscopy and Martin–Puplett layout. Ideal limits are an
even interferogram and equal/dephased ports at large delay for a smooth finite
band. This is field autocorrelation, not an intensity autocorrelation or a
unique longitudinal profile.

| Instrument | Required response/acceptance and calibration | Predicted quantity |
| --- | --- | --- |
| Pyroelectric detector | Energy-to-voltage response, absorber/window spectrum, pulse repetition/chopping, aperture, saturation and noise. | Voltage or energy/pulse; steady power alone does not describe its AC response. |
| Bolometer | Absorptance, power responsivity, thermal/electrical time constants, operating temperature and NEP. | Band-limited power or time-filtered signal. |
| Golay cell | Window and absorber bands, power responsivity, modulation bandwidth, aperture and background subtraction. | Calibrated modulated power/signal. |
| IR spectrometer | Throughput versus polarization/wavelength, slit/angular coupling, line-spread function, detector channels and wavelength calibration. | Energy/pulse or power per resolved channel. |
| Martin–Puplett | Both polarized arms, grids, optical delay, collection transfer, detector response and shot-to-shot charge/jitter normalization. | Two port signals, difference interferogram, recovered spectral magnitude. |
| Optical/IR camera | Spectral QE/responsivity, pixel geometry, imaging PSF/magnification, exposure, gain, dark/read noise, saturation. | Photoelectrons/counts per pixel or calibrated radiance. |

Photon number, joules per pulse, watts and ADC counts must remain distinct.
Mean power is repetition rate times energy/pulse for isolated identical
pulses; a phase-locked train can require an additional comb model.
Room-temperature thermal backgrounds matter particularly toward the IR/THz
end. No X-ray detector's thin-Si extrapolation substitutes for these responses.

Outputs should include source spectrum/angular polarization, collected
spectrum/energy, calibrated detector signal and interferogram, each with its
normalization, response digest and validity. Bunch-length inference must fit
through the same finite-source/optical/response model. Record spectral range,
missing-band extrapolation, charge and transverse-profile uncertainties,
timing jitter and response errors. Spectral magnitude alone loses Fourier
phase; Kramers–Kronig or iterative retrieval needs assumptions and can be
ambiguous ([Schmidt et al., §§3.3–3.5](https://arxiv.org/pdf/1803.00608)). Report
a constrained family or uncertainty, not a uniquely measured bunch shape.

## Grids, numerical validity and CUDA

Use component-specific supported bands instead of lowering the universal
floor. Resolve each selected mechanism, material, propagation operator and
detector over the requested band; reject unsupported intervals with the
responsible component named. An optical-only case need not request EEDL,
EPDL, atomic line tables or Eagle-XO QE. Multi-band results must carry separate
validity/provenance and combine only compatible quantities.

The 0.01 eV–100 keV span covers seven decades, but logarithmic coverage alone
does not resolve foil fringes, phonons, microbunching, detector channels or an
interferogram. Use band segmentation and local refinement; integrate bin
means explicitly and retain complex phase where later operators need it.
Re-derive coherent grid/window bounds using optical phase arguments. Avoid
materializing every X-ray reflection window in an optical continuum case.
For long bunches the initial grid proposal must permit sub-0.01 eV bands with
their own measured materials/detector support; that is not an automatic
extension of any optical fit.

Independently converge energy integration, angular acceptance, radiator
surface mesh, wavefront sampling/window/padding, propagation and delay range.
The maximum optical delay fixes spectral resolution; delay step fixes the
highest sampled frequency. Record whether spectra are per eV, angular
frequency, wavelength or wavenumber and apply the Jacobians. Use float64/
complex128 phases, relative reference clocks/planes, bounded passive
propagators and stable cancellation; do not infer accuracy from a finite value.

Every recommended production stage requires CUDA support and real hardware
evidence. Material selection/interpolation may prepare immutable host arrays
once per case, but field solve, crossing/history pair reduction, vector
surface integral, FFT propagation, polarization/element operators, spectral
quadrature and instrument reduction need a device-resident path without
per-crossing/element host traffic. Compare shared inputs and complex vectors
before squaring as well as final observables, using peak-scaled absolute
tolerances near zeros. Establish tolerances from convergence rather than
copying the current X-ray parity number. A small CPU reference is required;
production GPU sweeps/parity use `pyrite remote`, never a local heavy run.
Current homogeneous TR hardware evidence does not cover these optical bands,
finite optics or the experimental CPU DTR scorer.

## Requirements matrix and staging

Effort is relative architectural scope, not a delivery estimate: **M** extends
an existing owner; **L** adds a subsystem; **XL** needs both a new model and
geometry/validation work. Each row includes CUDA and independent validation.

| Mechanism / output | Layer and disposition | Effort | Dependency / decision |
| --- | --- | --- | --- |
| All optical sources | Material/band provider: needs extension | M | Explicit optical response; fail on gaps/tensors; no global floor change. |
| Planar OTR | Source: reusable Maxwell amplitude + provider extension | M | Material/band contract and completed #103 baseline. Anchor `optical-planar-tr`. |
| CTR, coherent emission generally | Statistics/phase: reusable pair estimator + correlated-input extension | M | Planar source, physical charge and resolved source law. Anchor `optical-ctr-population`. |
| Finite TR, DR/CDR | Source boundary fields: new subsystem | L | Planar normalization, finite geometry; source `optical-finite-boundary-field`. |
| Collected TR/CTR/DR | Polarized near-field optics: new subsystem | L | Finite-source fields, measured windows/elements; `optical-wavefront-propagation`. |
| Spectrum, camera, interferogram, bunch inference | Instrument response/inverse problem: new subsystem | L | Calibrated wavefront transfer and source uncertainty; `optical-instrument-observable`. |
| Atomic PXR/CBS/DTR | Applicability boundary; no optical production extrapolation | M investigation | Existing #581 work is separate; `optical-lattice-applicability` remains unverified. |
| Grating-diffracted TR / Smith–Purcell | New periodic source and finite geometry | XL | Deferred: choose optical target, clearance and published anchor first; existing grooves alone are insufficient. |
| Cherenkov | New finite dispersive source/escape coupling | L | Deferred: choose transparent radiator and anchor; avoid TR double counting. |
| Optical soft bremsstrahlung | New dispersive finite-history amplitude | XL | Deferred: relevant anchor and consistent formation/suppression treatment needed; bound omitted background initially. |

Stage 1: explicit material/band capability and optical provenance. Stage 2:
planar optical TR/CTR, correct bunch inputs and optical-band hardware parity.
Stage 3: finite-source fields and vacuum/polarized wave optics. Stage 4:
calibrated detector/interferometer forward model and uncertainty-aware bunch
inference. Separate cases are acceptable through these stages; a combined
optical–X-ray run is a later integration milestone. General grating sources,
Cherenkov and optical bremsstrahlung are not prerequisites for a bounded
planar CTR diagnostic if each omitted background is justified for its scene.

Implementation follow-ups (#614 landed; the rest open):

| Stage | Issue | Native prerequisite |
| --- | --- | --- |
| Optical materials and supported bands | [#614](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/614) | Closed #103 baseline. |
| Planar optical TR/CTR and bunch phases | [#615](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/615) | #614 and closed #103. |
| Finite-source TR/DR and polarized wave optics | [#616](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/616) | #615 and closed #103. |
| Calibrated instruments and bunch inference | [#617](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/617) | #616 and closed #103. |

Each is a native sub-issue of #533; the table explains staging, while GitHub's
native relations own dependency state. The closed #103/#581 issues are baseline
work, not evidence that their remaining independent-validation questions are
solved. The separate bulk Cherenkov source remains in #587. No additional
optical grating-DTR/Smith–Purcell or bremsstrahlung implementation is scheduled
without the geometry/data decisions described above.

## Evidence and verification limits

Zotero-first check: the local API and connector at `127.0.0.1:23119` were
unavailable (connection refused); no library records or attachment full text
were read or changed. Primary publisher/preprint and official dataset sources
were used instead. Available full text: Schmidt, Shkvarunets–Fiorito, Delerue
and Anthony; optical OTR/DR and DTR benchmark details still require acquisition
or extraction before implementation validation. No paper figure was digitized,
no experimental dataset reproduced and no solver/GPU run performed here.

This investigation recorded seven `optical-*` claims as research-only and
`unverified`, because no fresh-context derivation or independent reproduction
was performed. #614 has since implemented and anchored
`optical-material-response`; the other six remain `unverified`. Future implementation must add owning source markers,
limiting-case tests and independent verification to `rederived`, then a
published/experimental regression to `anchored` where specified. This note
does not alter existing claim statuses or imply human sign-off.
