# Transition radiation: omission bound and decisions

PyRITE emits transition radiation (TR) as an opt-in planar-stack component, including coherent bunch emission (CTR). See the [TR/CTR demo](../../guides/transition-radiation-demo.md) for a notebook and reference profiles, and [Run-case component](#run-case-component) for selection and limits. This page also records where the default omission is bounded below the statistical precision of a detected observable. The decision belongs to an observable, a band and a geometry, not to a beam energy. The small energy an electron loses to TR is never used as evidence: a few eV per interface can still dominate a weak soft-X-ray band.

The screening calculation lives in `pyrite.montecarlo.spectrum.transition_radiation`; the study that produces the numbers below is `checks/transition_radiation_omission.py`. Neither is a production emission component.

## Exact planar-stack TR

`Validation: tr-detected-omission-bound`

A point charge on a straight, uniform-velocity trajectory crosses a planar stack of homogeneous isotropic layers (permittivities $\varepsilon_j(\omega)$, thicknesses $d_j$) on a semi-infinite substrate. The radiation into the vacuum above the stack is the classical boundary-value problem of Ginzburg and Frank {cite:p}`ginzburg1990transition`, solved here without approximation (Gaussian units, $\hbar=c=1$).

For $\rho=e\,\delta(\mathbf r-\mathbf v t)$ and a transverse wavevector $\mathbf k_\perp$, the charge's own field in an unbounded medium $\varepsilon$ has the single longitudinal wavenumber $q=(\omega-\mathbf k_\perp\cdot\mathbf v_\perp)/v_z$ and

$$
\phi=\frac{4\pi e}{|v_z|\,\varepsilon\,(k^2-\varepsilon\omega^2)},\qquad
\mathbf E_p=i\phi\,(\varepsilon\omega\mathbf v-\mathbf k),\qquad
\mathbf H_p=i\varepsilon\phi\,(\mathbf k\times\mathbf v),\qquad
\mathbf k=(\mathbf k_\perp,q).
$$

Each layer adds free plane waves with $\mathbf K=(\mathbf k_\perp,\pm\lambda_j)$, $\lambda_j=\sqrt{\varepsilon_j\omega^2-k_\perp^2}$, $\operatorname{Im}\lambda_j\ge0$; the substrate and the vacuum carry only outgoing waves. Continuity of tangential $\mathbf E$ and $\mathbf H$ at every interface fixes all free amplitudes in one linear solve of size $4N_{\rm interface}$. Each wave is referenced to the interface it decays away from, so opaque layers do not overflow. Stationary phase at $\mathbf k_\perp=\omega\hat{\mathbf n}_\perp$ turns the vacuum amplitude $\mathbf a$ into the far field:

$$
\frac{d^2N}{dE\,d\Omega}=\frac{\omega\cos^2\theta\,|\mathbf a|^2}{16\pi^4}
\quad[\text{photons eV}^{-1}\,\text{sr}^{-1}\text{ per crossing electron}],
$$

summed over polarization. Interference between interfaces, formation zones, absorption ($\operatorname{Im}\varepsilon>0$), backward emission ($v_z<0$), oblique incidence, and a Cherenkov-active layer are all in the same solve. The permittivity is $\varepsilon=1-(r_e\lambda^2/\pi)\sum_i n_i(Z_i+f'_i-if''_i)$ from the Chantler factors, i.e. $1+\overline{\chi_0}$ of [the X-ray susceptibility](../../validation/radiation-physics/xray-chi-zero.md), extended to any catalog medium.

**Assumptions.** Classical emission (photon energy far below the electron kinetic energy); a straight trajectory at constant speed through the formation zones; flat, sharp, laterally infinite interfaces; local isotropic $\varepsilon(\omega)$. The independent-atom $\varepsilon$ is reliable well above $\hbar\omega_p$ and away from edges, and crude near the plasma energy, where it is still the medium response the rest of PyRITE's photon optics uses. Layers have no lattice periodicity, so diffracted TR is outside the solve (see below).

**Limiting cases** (all in `tests/montecarlo/spectrum/test_transition_radiation.py`):

- normal incidence on one interface reproduces the closed Ginzburg--Frank form in both directions, including a Cherenkov-active medium, to $10^{-9}$ (its angle integral is given for two arbitrary non-absorbing media by {cite:t}`deraad1978interference`);
- a conductor ($|\varepsilon|\to\infty$) at oblique incidence reproduces the sudden stop of the charge and its mirror image, $\alpha\,|\hat{\mathbf n}\times\boldsymbol\beta/(1-\hat{\mathbf n}\cdot\boldsymbol\beta)-\hat{\mathbf n}\times\boldsymbol\beta'/(1-\hat{\mathbf n}\cdot\boldsymbol\beta')|^2/(4\pi^2E)$, to $3\times10^{-4}$;
- zero contrast and a vanishing film radiate nothing; an opaque film reduces to its exit interface; a transparent film averages to twice one interface and peaks at four times;
- for $\gamma\gg1$ and $\omega\gg\omega_p$ the exit interface is Jackson's small-angle form {cite:p}`jackson1999classical` (Eq. 13.91) used by the channeling bound.

## Crossing superposition on transported tracks

`Validation: tr-crossing-superposition`

The solve is linear in its source. The right-hand side at interface $i$ is the jump $-(\mathbf P_{i+1}-\mathbf P_i)$ of the tangential own field $(\mathbf E_p,\mathbf H_p)_\perp$ between the media on either side, times $e^{iqz_i}$. For a charge through the origin at $t=0$ that phase equals $e^{i(\omega t_i-\mathbf k_\perp\cdot\mathbf r_{\perp,i})}$ at its crossing $(\mathbf r_i,t_i)$. The vacuum amplitude is therefore a sum over crossings,

$$
\mathbf a=\sum_c \mathsf G_{i_c}\,\bigl[-(\mathbf P_{i_c+1}-\mathbf P_{i_c})(\mathbf v_c)\bigr]\,e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})},
$$

where $\mathsf G_i$, the top-wave rows of the inverse continuity matrix, carries each layer's propagation, absorption and multiple reflection from interface $i$ to the vacuum. For a straight trajectory the sum reproduces the solve exactly (`interface_transition_radiation_amplitude`; agreement to $10^{-12}$ of the peak in both senses through an absorbing oblique stack).

On transported tracks, `transition_radiation_crossings.boundary_crossings` supplies $(\mathbf r_c,t_c,\mathbf v_c)$ from the row events (entry, composition-changing layer boundaries, top and bottom exits; substeps and equal-media boundaries never cross), and `crossing_transition_radiation` sums the crossings of one primary history coherently before squaring. Entry/exit formation-zone and multilayer interference then follow from the transported crossing times and points. Histories add incoherently by default; `coherence="bunch"` keeps the cross-history terms (CTR, [below](#coherent-bunches-ctr)). `coherence="crossing"` gives the independent-interface (dephased) limit, which a thickness average of the coherent film recovers.

**Assumptions.** Each crossing is a locally straight passage at its own velocity: $\mathbf P(\mathbf v_c)$ is the own field of an unbounded straight charge. Deflection between crossings radiates bremsstrahlung, which is not this term. The emitting face's vacuum surrounds the stack, interfaces are laterally infinite, and side exits and grooved faces carry no TR. Far-field origin: the transverse origin of the emitting face, time the transport clock $t_0+t$; the global $-i$ stationary-phase factor is dropped. Results change with numerical substeps only at quadrature level ($<10^{-3}$ in the smoke test).

**Published regular radiator.** For $N$ transparent foils in vacuum the small-angle density of {cite:t}`garibian1971transition` and {cite:t}`grichine2002generation` (the {cite:t}`geant4tr` regular-radiator stack factor),

$$
\frac{d^2N}{dE\,d\Omega}=\frac{2\alpha}{\pi^2E}\,\theta^2\Bigl(\frac1{A_2}-\frac1{A_1}\Bigr)^2\operatorname{Re}\langle R^{(N)}\rangle,\qquad
\langle R^{(N)}\rangle=N\frac{(1-F_1)(1-F_2)}{1-F}+\frac{(1-F_1)^2F_2(1-F^N)}{(1-F)^2},
$$

with $A_j=\gamma^{-2}+\theta^2+\omega_{p,j}^2/\omega^2$, $F_j=e^{-i\phi_j}$, $\phi_j=\omega t_jA_j/2$ and $F=F_1F_2$ ($2\operatorname{Re}\langle R^{(N)}\rangle=4\sin^2\frac{\phi_1}2\sin^2\frac{N(\phi_1+\phi_2)}2/\sin^2\frac{\phi_1+\phi_2}2$), is reproduced by `crossing_transition_radiation` on the stack's straight-track crossings within $2\times10^{-4}$ of the peak for ten 15 µm foils with 200 µm gaps at $\gamma=2000$, 2–40 keV and $\theta\le3/\gamma$. The residual is the formula's small-angle phase: with exact phases $t_j(\omega/v-\lambda_j)$ the agreement is $10^{-5}$. The absorbing-radiator flux form is not reproduced.

## Detector channels and acceptance

`Validation: tr-detector-integration`

Thin-film and stack TR has formation-zone fringes in energy and angle far narrower than a broad logarithmic grid. `transition_radiation_detector.detected_transition_radiation` therefore integrates the crossing-superposition density over each pixel of a planar detector (`PlanarPose` with a `PixelGrid`, or a bare plane) and each energy channel, instead of sampling it:

$$
N_{p,c}=\int_{E_c}^{E_{c+1}}dE\int_{\Omega_p}d\Omega\,\frac{d^2N}{dE\,d\Omega},
$$

with composite trapezoid nodes inside every channel and $s\times s$ point sub-pixels per pixel ($A\cos/r^2$). The energy and angular levels double one at a time, toward the axis whose last doubling changed the table most, until a third of the larger last change (both rules are second order) is within `rtol` of the table peak. That estimate is asymptotic, not a bound. The source is a point at the lab origin, directions map into the slab frame with transport's tilt, and the detector must lie beyond one face. Photons are incident on the plane; filters and detector response stay read-time factors. A single exit interface reproduces the closed Ginzburg--Frank form through the acceptance to $10^{-3}$, and a 50 µm film with about twenty fringes per channel matches a dense Gauss--Legendre reference to $5\times10^{-4}$, where the starting grid is off by more than 5%.

## Interface energy scale

`Validation: tr-interface-energy-scale`

Integrated over the forward hemisphere and all photon energies with the Drude $\varepsilon=1-(\omega_p/\omega)^2$, the exact single-interface emission gives $W/(\alpha\gamma\hbar\omega_p/3)=0.955$ at $\gamma=40$ and $0.991$ at $\gamma=197$: the PDG estimate {cite:p}`zyla2020passage` holds with an $O(1/\gamma)$ deficit (about $1.8/\gamma$) from large-angle, low-frequency emission. Emission below $3\hbar\omega_p$ carries about 35% of $W$ at $\gamma=40$, because the spectrum $E\,dN/dE\simeq(\alpha/\pi)\ln(\gamma^2\omega_p^2/\omega^2)$ is logarithmic below $\gamma\hbar\omega_p$. The scale is a total-energy statement only: it is not a photon count and not a bound on any detected band.

## Omission bound for a detected observable

Let $S(E)$ be the production density the detector scores at the observation direction and $T(E)$ an upper bound on the TR density over the acceptance. If $T(E)\le c\,S(E)$ at every energy an observable can draw on, then any positive linear response $R$ (filters, quantum efficiency, resolution and redistribution, solid-angle integration) gives $R[T]\le c\,R[S]$ pointwise. The relative change of every non-negative observable (window counts, sidebands, totals, a line/background ratio's background) is therefore at most $c$. The study takes $S$ as the bremsstrahlung continuum alone; lines and CXR only lower $c$.

$T$ is $5\times$ the maximum single-crossing TR over speeds $\le\beta_0$, crossing directions, and five acceptance directions (centre and half the 16.6° polar span in polar angle and azimuth). A primary crosses a flat face at most twice; an entry and an exit through the same face add as amplitudes, at most $4\times$ one crossing; escaping secondaries above the photon energy add fewer than one more crossing per primary (asserted, not computed). TR rises monotonically with speed, so the maximum over speeds is the top of the sampled range. The angle grids are sampled: a finer grid raises the maximum by 3.7% at most in the fresh-context check, inside every margin quoted below. Where transport is diffusive (tens of keV), every crossing direction up to 88° from the normal and both senses are allowed, together with the facing layer alone (reversing electrons) and the whole stack (straight-through electrons). For MeV films the crossing directions stay within $3\theta_0$ (Highland) of the beam.

Two cases are outside this argument:

- a net line after sideband subtraction is not non-negative; with the continuum-only $S$ used here its bias is at most $c\,B/L$ for window background $B$ and net line $L$ (with a total $S$, only $c\,(1+B/L)$);
- a coherent signal can interfere with a TR amplitude; see [diffracted TR](#diffracted-tr-and-pxr).

**Tolerance.** Omission is a deterministic bias $b$ added to a statistical error. It is accepted when $b\le\text{target\_rse}/3$, so that $\sqrt{\text{rse}^2+b^2}\le1.054\,\text{rse}$ (`omission_tolerance`). With the catalog adaptive-precision default `target_rse = 0.05` the tolerance is $b\le1.67\%$. A fixed-count run uses the observable's realized relative standard error instead; coherent-emission and physical-detector routes, which refuse adaptive precision, must do so. Bounds below are functions of the scene, so a tighter target is re-checked by rerunning the study.

Two statements the bound does **not** rely on, and which are false in general:

- **Summed single-interface estimates are not a bound for a coherent stack.** A resonant ten-foil Si radiator (100 MeV, 1.5 keV, 1.58 µm foils, 20 µm gaps) peaks at 156 times one interface's peak, against $4N=40$ for fully constructive pairs summed incoherently, and at 300 against 80 for twenty foils (`test_coherent_multilayer_exceeds_summed_interface_estimates`). Stacks must be solved coherently, as here.
- **All-angle radiated energy divided by a nominal photon energy does not bound the detected background.** The TR spectrum is $\propto1/E$ up to $\gamma\hbar\omega_p$ and its angular pattern differs from the source's; only a spectral-angular comparison through the acceptance, as above, bounds an observable.

## Results

`checks/transition_radiation_omission.py`, CPU, 100 trials per scene, brem grid from 1.01× the [continuum floor](spectral-observables.md) to the beam energy (30 keV cap), 30° tilt toward the detector. Bias bounds per band; **bold** fails the 1.67% tolerance. $E^\ast$ is the lowest energy above which every grid point passes.

| Scene | Detector faces | EUV (< 300 eV) | 0.3--1 keV | ≥ 1 keV | $E^\ast$ |
| --- | --- | --- | --- | --- | --- |
| Si, 10 keV, 10 µm | entrance | **1.4** | 8.7e-4 | 8.7e-7 | 213 eV |
| Si, 30 keV, 2 µm | exit | **15** | 1.5e-2 | 9.1e-6 | 307 eV |
| Si, 30 keV, 10 µm | entrance | **13** | 1.3e-2 | 3.3e-6 | 307 eV |
| Si, 60 keV, 2 µm | exit | **85** | **8.3e-2** | 6.6e-5 | 439 eV |
| HOPG, 30 keV, 2 µm | exit | **1.4e3** | **3.1e-2** | 3.2e-5 | 363 eV |
| MoS₂, 30 keV, 2 µm | exit | **61** | **3.0e-2** | 2.5e-5 | 386 eV |
| WSe₂, 30 keV, 2 µm | exit | **200** | **1.3** | 2.5e-4 | 743 eV |
| MoS₂ 0.2 µm on sapphire 50 µm, 30 keV | entrance | **26** | 1.3e-2 | 2.8e-5 | 327 eV |
| Si, 3 MeV, 2 µm, detector 90° | exit | **1.6e3** | **1.0** | 8.0e-4 | 625 eV |
| Si, 3 MeV, 2 µm, detector 1° | exit | **2.6e3** | **67** | **0.18** | 1.8 keV |
| Si, 10 MeV, 2 µm, detector 1° | exit | **6.5e3** | **1.0e3** | **15** | 7.3 keV |
| Si, 30 MeV, 2 µm, detector 1° | exit | **1.2e7** | **2.3e5** | **2.0e3** | 25 keV |
| Si, 100 MeV, 2 µm, detector 90° | exit | **2.9e10** | **2.0e5** | **14** | 3.6 keV |
| Si, 100 MeV, 2 µm, detector 1° | exit | **7.7e8** | **1.5e7** | **2.6e4** | > 30 keV |
| Si, 100 MeV, 100 µm, detector 1° | exit | **2.1e9** | **4.1e7** | **1.4e5** | > 30 keV |

The bound is conservative by construction (worst crossing direction and speed, factor 5, continuum-only denominator). The 30 keV Si scene at the actual beam exit direction gives TR/brem ≈ 0.73 at 32 eV, 0.03 at 50 eV and 0.03 at 150 eV, so the EUV failures are physical, not an artefact of the bound. Near the floor TR is a surface source, unattenuated, while the bremsstrahlung escaping at the same energy comes from the last tens of nanometres.

The 30 and 100 MeV scenes (#439) use the production coupled transport, so the bremsstrahlung denominator carries the Ter-Mikaelian factor $k^2/(k^2+(\gamma\hbar\omega_p)^2)$ ([bremsstrahlung](bremsstrahlung.md)); with $\gamma\hbar\omega_p=1.9$ and 6.1 keV in Si it suppresses the continuum by $(k/\gamma\hbar\omega_p)^2$ below those energies, while TR, the boundary term of the same dielectric response, is not suppressed. The EUV and soft-X-ray ratios therefore grow much faster than $\gamma$; at 100 MeV forward the ratio is still 0.12 (2 µm) and 0.09 (100 µm) at 30 keV, the grid cap. The cone is $3\theta_0$ (Highland): 0.21° at 30 MeV and 0.07°/0.6° at 100 MeV for 2/100 µm. The earlier angle-integrated screening estimate (Jackson Eq. 13.92 against thin-target complete-screening bremsstrahlung, no suppression) put the 1.67% crossover at 39 keV for 100 MeV and 2 µm and 12 keV for 30 MeV, consistent with these bounds.

## Diffracted TR and PXR

An [experimental first-Born field diagnostic](diffracted-transition-radiation.md)
now diffracts the stack's internal free TR waves through one crystal harmonic
and provides an alternating SiO₂/MoS₂/SiO₂ reference. It is not part of the
production component sum and has no CBS field or dynamical extinction solve;
the production interference limitations below still apply.

In a crystal the boundary term carries the susceptibility's Fourier components too: diffracted TR (DTR) emerges near the Bragg directions where PXR does, and its amplitude interferes with PXR rather than adding as an intensity {cite:p}`shevelev2024`. In the kinematic theory the PXR amplitude carries the medium-dressed virtual-photon denominator $\omega^2(\gamma^{-2}+\theta^2-\chi_0)$ {cite:p}`nitta1991kinematical` (Eq. 19), while the boundary term carries the difference between the vacuum and medium denominators, as ordinary TR does. Their ratio estimates

$$
\left|\frac{A_{\rm DTR}}{A_{\rm PXR}}\right|\sim\frac{|\chi_0|}{\gamma^{-2}+\theta^2}\le\beta^2\gamma^2|\chi_0| ,
$$

using $k^2-\omega^2\ge\omega^2/(\beta^2\gamma^2)$ for any virtual photon with $\mathbf k\cdot\mathbf v=\omega$. For tens-of-keV beams at keV lines ($\beta^2\gamma^2\approx0.1$, $|\chi_0|\lesssim10^{-4}$) the cross term is below $10^{-4}$ of the line, so additive PXR/CBS without DTR is consistent with the decisions below. At MeV energies and soft-X-ray lines ($\gamma^{-2}\sim|\chi_0|$) DTR is comparable to PXR and its cross term must be kept: that is #103 scope. This is an amplitude-ratio screening estimate, not a rigorous bound, and is not part of the ledgered claim.

## Coherent bunches (CTR)

Every bound on this page is per electron and incoherent. A bunch of $N$ electrons on identical trajectories radiates TR amplitudes whose phases differ only by arrival time and position. The spectrum is then $[N+N(N-1)|F(\omega)|^2]$ times the single-electron yield, with $|F|^2$ the bunch form factor the [coherent-emission](coherent-emission.md) path already uses ($e^{-(\omega\sigma_t)^2}$ for a Gaussian bunch). Relative to the incoherent yield used above, coherent TR (CTR) is enhanced by about $N|F(\omega)|^2$. Bremsstrahlung stays incoherent, so a bremsstrahlung-referenced bound fails as soon as $N|F|^2\gtrsim1$. Coherent PXR and CBS get the same longitudinal enhancement, so CTR relative to coherent lines is set by the incoherent ratio and the different angular spreads.

For the catalog beams at 1 pC ($N=6.2\times10^6$):

| Beam | $\sigma_t$ | $\omega\sigma_t$ at 31 eV | at 1 keV | $N\lvert F\rvert^2$ at 31 eV / 1 keV |
| --- | --- | --- | --- | --- |
| `default`, `gaussian_200fs` | 200 fs | $9.4\times10^3$ | $3.0\times10^5$ | 0 / 0 |
| `1_atto`, `coherent_demo` | 1 as | 0.047 | 1.5 | $6.2\times10^6$ / $6.2\times10^5$ |

So the omission decisions hold for incoherent emission and for bunches long compared with the photon period. They do **not** hold for `emission = coherent` (or `both`) runs whose form factor is non-negligible in the band: today the attosecond catalog beams across the whole EUV and soft-X-ray range. CTR there is computed from the crossings (next section). CTR at optical and infrared wavelengths, where it is used as a bunch-length diagnostic, is below the modelled X-ray band; the bounded opt-in optical path with explicit measured permittivities and a physical-bunch sampling contract is [optical planar TR and CTR](optical-transition-radiation.md).

### CTR from transported crossings

`Validation: tr-coherent-bunch`

With $\mathbf S_e$ the summed vacuum amplitude of history $e$ (all its crossings, phased with that electron's own arrival offset $t_0$ and transverse entry point), the ensemble power per physical electron of a bunch of $N$ independent electrons is $\mathrm E|\mathbf S|^2+(N-1)F\,|\mathrm E\mathbf S|^2$. `crossing_transition_radiation(..., coherence="bunch", physical_electrons=N)` estimates it from the $M$ sampled histories with the distinct-pair estimator of the coherent-emission path (`coherent_population.pair_scale`),

$$
\frac{d^2N}{dE\,d\Omega}=\frac{E\,n_z^2}{16\pi^4M}\Bigl[G+\frac{N-1}{M-1}F\,(P-G)\Bigr],\qquad
G=\sum_e|\mathbf S_e|^2,\quad P=\Bigl|\sum_e\mathbf S_e\Bigr|^2 .
$$

The sampled offsets already sit in the crossing phases $e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})}$, so $F=1$ and the sampled $|\langle e^{i(\omega t_0-\mathbf k_\perp\cdot\Delta\mathbf r_\perp)}\rangle|^2$ supplies the longitudinal and transverse form factors. An analytic `form_factor` (e.g. $e^{-(\omega\sigma_t)^2}$) is only for histories transported without those offsets; passing it with sampled offsets counts the form factor twice. `physical_electrons=None` takes the sample as the bunch ($N=M$, the plain coherent sum $P/M$).

**Limits** (`tests/montecarlo/spectrum/test_transition_radiation_ctr.py`). Identical in-phase histories give $1+(N-1)F$ per electron, so $N+N(N-1)F$ per bunch and $N^2$ for a point bunch; $N=1$ gives the per-history yield. Sampled Gaussian arrival times and transverse offsets reproduce $e^{-\omega^2\sigma_t^2-(\omega\sin\theta\,\sigma_x)^2}$ within three standard deviations of the pair estimate. A bunch long compared with the wavelength returns the incoherent sum on average, with speckle: a sampled bunch of $M=N$ electrons is one realization, not the ensemble.

**Assumptions.** Electrons are independent and identically distributed; with sampled offsets nothing else is needed. An analytic `form_factor` additionally requires transport that does not depend on the omitted offsets (true for the planar stack: arrival time does not enter transport and the slab is laterally infinite). It can stand in for one offset alone, e.g. pass only $e^{-(\omega\sigma_t)^2}$ for histories transported with a sampled spot but no arrival-time spread. The estimate is signed and unclipped: when $(N-1)/(M-1)\gg1$ the excess noise can exceed a small true excess and give negative bins, which raise `CoherentSamplingWarning`. No sampling-error certificate is produced.

## Run-case component

A run opts in with `Physics(transition_radiation="history" | "bunch")` (profile key `transition_radiation`, CLI `--transition-radiation`; default `"off"`). Lowering attaches the case record `transition_radiation = {"model": "planar-stack", "coherence": ...}` (`Case.transition_radiation`); `energy_model = "midpoint"` is required for the row events, and grooved faces, channeling, temporal profiles and adaptive precision are refused. The spectrum phase then adds `spec_transition` [photons/eV/sr per incident electron] on the line grid at the case direction, and `spec_transition_by_direction` for observation directions, from the same transported segments as every other component (`runner.transition_emission.case_transition_spectrum`). `_spectral_components.cxr_spectrum` and `incident_spectrum` add it once to either CXR mode (`transition=False` deselects it), for the reason given in the next section.

- **Population.** The electrons below `Ne_brem`, which transport follows down to the continuum cutoff `E_cut_brem_keV`; crossings after a history ends at the cutoff are not recorded. For a 1 keV cutoff the omitted exits have $\beta^2\le4\times10^{-3}$.
- **Media.** Each transport layer's permittivity is the independent-atom forward scattering of its composition with Chantler $f',f''$ (`composition_dielectric`); energies outside the table are refused. The [validity boundary](#validity-boundary) of that model applies: crude near and below the plasma energy and near edges.
- **Coherence.** `"bunch"` uses the physical population `bunch_charge_pc / e` with the sampled offsets (above); its signed estimate warns when undersampled.
- **Identity and provenance.** On the divergence-only rule: `"off"` keeps every dataset and case digest; `"history"`/`"bunch"` add `TRANSITION_MODEL` (`planar-stack-crossings-v1`) to the dataset identity, the case content key and the physics provenance (`emission.transition` descriptor).
- **Storage and consumers.** Checkpoints store it as the separate optional `transition.h5` component; reline rescores it from the same re-transport as the line kernel; full and streamed artifact replay score it (`"history"` sums whole-electron blocks, `"bunch"` refuses streaming). `Result.transition`, `SpatialResult.transition` and `spec_transition_by_direction` carry it to the API, planar observations and plots; `combined_cxr`, `cxr_total` and `cxr_spectrum` add it (each can exclude it).
- **Device.** On a CuPy (CUDA/ROCm) backend the component is scored on the device. The stack response is one batched CuPy solve per energy/direction set (`interface_response`). On CUDA a fused RawKernel (`transition_radiation_jit_kernel`) then evaluates every crossing amplitude in registers: each thread sums whole histories, and each block reduces in a fixed tree order with no atomics, so a launch geometry is bit-reproducible. Changing the geometry only reorders the sum. Host and device memory grow with the crossings and pairs, not with their product. Crossing records are reduced from the transport rows on the host (one bulk copy), and only the finished spectrum returns. Without CUDA the same `xp` code runs as CuPy array operations, or on the host with NumPy. Device and host spectra agree to `rtol = 1e-10`, with an absolute floor of `1e-12` times the peak for near-cancelled entries. All 16 tests passed on an RTX 5080; see the [hardware evidence](../../validation/radiation-physics/tr-cuda-parity.md).
- **Refused.** Adaptive precision and electron-block transport do not score it (`refuse_transition_radiation`). Finite footprints are allowed: side exits are counted and carry no TR.

## Interference with PXR and CBS

`Validation: tr-line-cross-term`

TR here is the $\mathbf g=0$ term of the polarization current: $\varepsilon$ of each layer is homogeneous. Every coherent PXR/CBS field of the [coherent-emission](coherent-emission.md) path belongs to a reciprocal vector $\mathbf g\neq0$ (slab frame) and carries the lattice phase $e^{-i\mathbf g\cdot\mathbf r_j}$ on top of the far-field phase $e^{i(\omega t-\omega\hat{\mathbf n}\cdot\mathbf r)}$ that the TR crossings also carry; both use the vacuum $\omega\hat{\mathbf n}$, and the line's in-medium escape phase depends only on depth. Translate one whole history by $\boldsymbol\Delta$ in the surface plane. Amorphous transport through a laterally infinite stack makes the translated history equally likely; at one detector direction the TR amplitude and the line field gain $e^{-i\omega\hat{\mathbf n}_\parallel\cdot\boldsymbol\Delta}$ and $e^{-i(\omega\hat{\mathbf n}+\mathbf g)_\parallel\cdot\boldsymbol\Delta}$, so their product $\mathbf S_{\rm TR}\cdot\mathbf S^\ast_{\mathbf g}$ gains $e^{i\mathbf g_\parallel\cdot\boldsymbol\Delta}$. Averaged over the beam's entry distribution $\rho(\boldsymbol\Delta)$, the single-electron cross term is multiplied by the characteristic function $\tilde\rho(\mathbf g_\parallel)$, which is $e^{-(g_\parallel\sigma_\perp)^2/2}$ for a Gaussian spot of per-axis r.m.s. $\sigma_\perp$: the cross term is linear in each field, so the factor enters once, not squared. With $g_\parallel\sim1$ Å$^{-1}$ it is numerically zero for any spot above a nanometre. Between two electrons of a bunch the TR–line term carries $\tilde\rho(\mathbf K)\,\tilde\rho^\ast(\mathbf K+\mathbf g_\parallel)$, with $\mathbf K=\omega\hat{\mathbf n}_\parallel$ at normal incidence with an untilted bunch front (in general shifted by the bunch-front tilt and face mapping, which moves both factors equally), at most $e^{-(g_\parallel\sigma_\perp)^2/4}$ per pair. That is suppressed too, but it is not the inter-electron line factor $e^{-(q_\perp\sigma_\perp)^2}$ of [The transverse partner](coherent-emission.md#the-transverse-partner), which has a different argument and power.

**Decision.** For reflections with $g_\parallel\sigma_\perp\gg1$, TR and the coherent lines are **additive** in the ensemble: TR is a separate component and no TR–line cross term is kept. The ensemble average of that cross term is reduced by $\tilde\rho(\mathbf g_\parallel)$ itself; it is not a term dropped for its size. The condition is $g_\parallel\sigma_\perp\gg1$, not $\mathbf g_\parallel\neq0$: a small miscut or a mosaic node tilted by a small angle from a surface-normal $\mathbf g$ can leave $g_\parallel\sigma_\perp\lesssim1$.

**Not covered by the homogeneous `history`/`bunch` component.** These cases
keep a TR–line cross term. The experimental opt-in
[`two-beam` transported-current path](diffracted-transition-radiation.md#opt-in-transported-current-production-scoring)
retains it for one perfect-crystal reflection, with independent validation
pending:

- $\mathbf g$ along the surface normal ($\mathbf g_\parallel=0$: planes parallel to the surface, symmetric Bragg geometry). This includes the default orientation of the layered catalog crystals: the (00l) rows of HOPG and hBN with [001] along the normal, and the central node of a mosaic average. Symmetric Laue geometry, with $\mathbf g$ in the surface plane, is the most strongly suppressed case, not an exception. At $\mathbf g_\parallel=0$ the translation phase is trivial. The cross term then depends on the surface termination (the lattice origin relative to the face), which PyRITE does not model, and on the absolute phase of $A_{\rm PXR}+A_{\rm CBS}$ relative to the Maxwell solve here, which has not been reconciled.
- A point spot (`PointSpot()`): every history enters at the same lattice point, so $\tilde\rho=1$. This is the degenerate pure-geometry limit, not a physical beam.
- Transport that is not translation invariant at the lattice scale or at the spot scale: channeling (the trajectory depends on the entry point modulo the lattice), finite footprints with entries near an edge, and grooved faces.

**Diffracted TR.** The additivity above is for the $\mathbf g=0$ TR only. Diffracted TR (DTR, the boundary term carrying $\chi_{\mathbf g}$, [above](#diffracted-tr-and-pxr)) exists for every $\mathbf g\neq0$, picks up the same translation phase as PXR, and so interferes with PXR for every reflection, without suppression. It is comparable to PXR when $\gamma^{-2}\sim|\chi_0|$ (MeV beams, soft-X-ray lines), where the coherent line itself is incomplete without it. DTR needs its own solve (two-beam dynamical boundary conditions on the stack faces) and is not implemented. At tens of keV and keV lines, the screening estimate above puts it below $10^{-4}$ of the line.

The bremsstrahlung continuum is incoherent in PyRITE and has no phased field to interfere with.

## Decisions

| Application (research-note priority table) | Decision | Bound and validity |
| --- | --- | --- |
| Tens-of-keV transport, keV X-rays (≥ 1 keV) | **Omit** | $b\le2.5\times10^{-4}$ over 10--60 keV, Si/HOPG/MoS₂/WSe₂, films, bulk, film-on-substrate, both faces; $\le7\times10^{-5}$ for every scene but WSe₂. Holds for observables whose source band starts ≥ 1 keV. |
| Tens-of-keV, soft X-ray 0.3--1 keV | **Per scene** | Passes for Si up to 30 keV and for the film-on-substrate scene; fails for HOPG, MoS₂, WSe₂ and Si at 60 keV. Use $E^\ast$ from the study, or escalate. |
| EUV / soft X-ray down to the continuum floor (any beam energy) | **Escalate to #103** | $b$ from 1.4 to 1400 below 300 eV in every scene. PyRITE models photons from $\max(\hbar\omega_p, 12~\text{eV})$, so this band cannot be scoped out. Below the floor (optical) is outside the model; optical TR is measured at 30 keV {cite:p}`coenen2011` and belongs to a future optical model. |
| MeV thin-film soft X-ray (outside channeling) | **Escalate to #103** below $E^\ast$; omit above | Off-axis 90° at 3 MeV: omit ≥ 625 eV ($b\le8.0\times10^{-4}$ for ≥ 1 keV). Forward detectors: fails to 1.8 keV (3 MeV) and 7.3 keV (10 MeV). The channeling row is decided separately (`cr-transition-radiation-bound`). |
| 30--100 MeV transport, keV X-rays | **Include** (opt-in TR, #103) below $E^\ast$; omit above | With supported coupled transport (#439): forward detectors fail to 25 keV at 30 MeV and beyond the 30 keV grid at 100 MeV, for 2 µm and 100 µm Si; off-axis 90° at 100 MeV omits ≥ 3.6 keV. Above 100 MeV: #440. |
| Purpose-built multilayer radiator | **Escalate to #103** | Essential source physics; coherent gain exceeds summed-interface estimates (above). |
| Coherent bunches (CTR) with non-negligible form factor | **Compute** (`coherence="bunch"`) | Enhancement $N\lvert F\rvert^2$ up to $6\times10^6$ for the 1 as catalog beams; incoherent bounds do not apply. TR adds to PXR/CBS for $g_\parallel\sigma_\perp\gg1$; surface-normal $\mathbf g$ (default (00l) rows) and DTR are not covered (#103). Optical/IR CTR: #533. |

## Validity boundary

The omission decisions hold for planar slabs and stacks, straight crossings at the stated speeds, the catalog media's independent-atom $\varepsilon$, and bremsstrahlung as the comparison floor. They do not cover grooved or rough entrance faces, laterally finite targets whose side faces the detector sees, coherent (`emission = coherent`) observables with a non-negligible bunch form factor (CTR, above), or a tighter precision target than 0.05 without rerunning the study. Transported delta rays are off by default in these scenes; that limitation is independent of TR.

## Explicit optical response for bounded diagnostics

`Validation: optical-material-response`

The provisional `pyrite.materials.optical.OpticalResponse` accepts explicit
local, isotropic, scalar relative permittivity. Photon energies are in eV;
permittivity is dimensionless and uses $\exp(-i\omega t)$, with
$\operatorname{Im}\epsilon\geq0$ for a passive medium. Each response records
its medium/sample label, sample state, temperature in kelvin (or explicitly
unspecified), dataset identifier, original reference, source units, orientation,
uncertainty, source URL, source SHA-256 and license. Isotropic orientation is
explicitly inapplicable; unspecified uncertainty is not a zero-error claim.
Its `digest` also covers
numeric values, support segmentation and interpolation policy. Unsupported
tensors, active response, nonfinite values and undeclared extrapolation fail.
Disjoint segments represent gaps; interpolation never joins across a gap.

`gold_rakic_response()` provides the pinned CC0 refractiveindex.info gold
Lorentz–Drude fit to measurements, referenced to Rakić et al.,
[Applied Optics 37, 5271 (1998)](https://doi.org/10.1364/AO.37.005271).
This record covers about **0.2–5 eV**, with actual endpoints determined by the
rounded wavelength table. It does not cover every range discussed in the
paper. Source temperature is unspecified. The shipped bytes, revision and hash
are recorded in `THIRD-PARTY-NOTICES.md`; loading verifies the hash offline.
For vacuum wavelength $\lambda$ in micrometers,

$$
E=\frac{hc}{10^4\lambda},\qquad
\epsilon=(n+ik)^2=n^2-k^2+2ink.
$$

Real and imaginary permittivity are interpolated linearly **versus energy**.
This convex interpolation preserves passivity and constant vacuum response;
it is not a claim of global Kramers–Kronig consistency. Interpolation runs in
NumPy; `epsilon(energy, xp=cupy)` validates on the host and uploads complex128
values. It does not silently select a backend or interpolate on the device.

A source-only diagnostic can associate one response with each transport layer
in order through `case_layer_eps(..., layer_responses=(...))`, or pass the same
keyword to `case_transition_spectrum`. Every layer must have an explicit
provider. The mapping is an assertion about the sample by the caller; a
chemical composition never selects an optical response automatically. Provider
failures name the TR component and layer and propagate without atomic fallback.
Without providers, `case_layer_eps` retains the independent-atom X-ray response
above each layer's existing continuum floor and refuses lower energies.
Production component dispatch, persisted cases and the shared X-ray grid floor
are unchanged. This API does not enable general optical source/instrument runs.

Use the provisional `energy_grid.component_support` resolver to validate all
components requested by a diagnostic:

```python
import numpy as np
from pyrite.energy_grid.component_support import ComponentSupport, component_grid
from pyrite.materials.optical import gold_rakic_response
from pyrite.montecarlo.spectrum.transition_radiation import planar_transition_radiation

response = gold_rakic_response()
energy = component_grid(
    np.geomspace(0.3, 4.0, 257),
    required=(("Au", "transition_radiation"),),
    support=(ComponentSupport("Au", "transition_radiation", response.bands_eV),),
)
eps = response.epsilon(energy)
photons_per_eV_sr = planar_transition_radiation(
    energy,
    np.array([0.6, 0.0, 0.8]),
    np.array([0.0, 0.0, 0.8]),
    [(eps, 1000.0)],  # gold slab thickness in angstroms, vacuum outside
)
```

A capability belongs to one named medium/component pair. Adding an attenuation,
bremsstrahlung or detector component to `required` demands that component's own
explicit capability; otherwise resolution fails naming the responsible pair.
An optical TR capability supplies no EPDL, EEDL, Chantler or Eagle-XO/Si-QE
support. The resolver preserves coordinates and the canonical midpoint edge
and width conversion. `representation="nodes"` checks evaluation nodes;
`representation="bins"` additionally requires the entire extent of every
midpoint bin to lie in one supported interval. It never clips or relocates a
grid to make coverage pass.

The [independent verification](../../validation/radiation-physics/optical-material-response.md)
separates optical-record interpolation accuracy from the accuracy of the fitted
model against experiments. `checks/optical_response_cuda.py` is the bounded
real-hardware parity/refusal check; it must run on a remote CUDA allocation.
