# Coherent PXR and CBS radiation

The line-spectrum kernel is the core radiation model of PyRITE. It uses vector PXR/CBS amplitudes with relativistic velocity and CBS corrections, drawing on {cite:t}`feranchuk2000` and {cite:t}`zhai2025` for van der Waals crystals; {cite:t}`baryshevsky2005` gives a book-length treatment. It combines parametric X-ray radiation (PXR) and coherent bremsstrahlung (CBS) amplitudes for a set of reciprocal-lattice reflections, evaluates them once per transported trajectory segment, applies the finite-flight-time line profile, attenuates the photon on its way out of the sample, and sums the result over segments, electrons, reflections, and crystallite orientations.

The kernel is kinematic (Born): the crystal supplies static Fourier components of its susceptibility and its electrostatic potential, the electron supplies a straight constant-velocity flight, and the emitted field is first order in both.

There is no fixed 300 keV beam-energy gate. The documented transport checks and scattering data extend to roughly 300 keV; higher-energy predictions, including 3–5 MeV beams, still need transport and radiation validation. See the [relativistic PXR/CBS assessment](../../research/physics/relativistic-pxr-cbs.md) for the remaining model limits.

## Emission geometry

All vectors live in the **sample frame**: $+z$ is the slab normal (into the sample), the slab surface is the $z=0$ plane, and the transport geometry places the beam and the detector in that frame. Observation is a single fixed far-field direction $\hat{\mathbf n}$ (unit), either given directly or built from a polar observation angle as $(\sin\theta_{\rm obs},0,\cos\theta_{\rm obs})$. Because $\hat{\mathbf n}$ is fixed for the whole sample, the model is a far-field one: every segment radiates toward the same direction and no per-segment detector parallax is taken.

The far-field expansion keeps the linear retardation $-\omega\hat{\mathbf n} \cdot\mathbf r$ exactly and drops only the curvature term $\omega r_\perp^2/2R$, where $R$ is the source-to-detector distance and $r_\perp$ is the component of the emission point transverse to $\hat{\mathbf n}$. A displacement parallel to the observation direction has no curvature term. Slab depth contributes to $r_\perp$ when observation is oblique, so the condition must use the full emitting volume:

```{math}
:label: eq-coherent-radiation-far-field

r_\perp\ll\sqrt{\frac{\lambda R}{\pi}},
```

where the right-hand side is the transverse radius at which the neglected phase reaches one radian. This is a phase-error condition; check it for the actual source extent, photon wavelength, and detector distance. Incoherent calculations also incur a direction error of order $r_\perp/R$, which must be small enough for the desired spectral resolution.

Reciprocal vectors are oriented before the sum. `beam_uvw` names the direct-lattice axis placed along $+z$; `surface_hkl` instead names the reciprocal-lattice plane normal placed along $+z$, which is the exact cleavage-plane contract for nonorthogonal cells. Either choice is applied as the minimal proper rotation, followed by `azimuth_rad` about $+z$ for the in-plane setting relative to the detector azimuth. `recip_miscut_rad` adds a further tilt to the reciprocal vectors **only**, leaving the transported slab normal untouched — useful for an asymmetric reflection where $\mathbf g$ is not parallel to $\hat{\mathbf n}$'s exit face, and not yet wired into any campaign grid. Conventions are shared with [Tilt convention](../geometry/tilt-convention.md).

Reflections arrive as a pinned catalog family list expanded to **both** reciprocal directions, $\pm\mathbf g$. Distinct reflections are treated as spectrally separated and therefore summed incoherently; the kernel does not evaluate cross-$\mathbf g$ interference, and the tested scope is recorded under [Validation: `cross-reflection-coherence`](../../validation/radiation-physics/cross-reflection-coherence.md). The bound covers the catalog basal-plane families, not arbitrary near-degenerate reflections.

## Resonance condition

For a segment with velocity $\mathbf v$ (in units of $c$) and a reflection $\mathbf g$, the stationary-phase condition on a straight flight fixes the emitted frequency {cite:p}`feranchuk2000`,

```{math}
:label: eq-coherent-radiation-resonance

\omega_{\rm res}=\frac{\mathbf v\cdot\mathbf g}
{1-\hat{\mathbf n}\cdot\mathbf v},
\qquad E_{\rm res}=\hbar c\,\omega_{\rm res},
```

with $\omega$ carried in inverse Ångström ($c=1$, lengths and times both in Å). Only positive roots radiate, so of the $\pm\mathbf g$ pair exactly one member satisfies the condition for a given flight; expanding families to both directions is what makes the harmonic-selection sign self-resolving rather than a hidden convention. The overall numerator sign depends jointly on the lattice harmonic convention $\exp(+i\mathbf g\cdot\mathbf r)$ and the outgoing-wave convention, and the independent derivation and the implementation currently disagree on it — this is the open `line-energy-dispersion` row. It fixes **where** each line sits, not how the amplitude is squared or summed.

The vacuum relation $k=\omega$ is replaced throughout by the bulk Maxwell relation $k=n(\omega)\,\omega$, which makes the resonance implicit; see [Photon escape and in-medium dispersion](photon-escape-and-dispersion.md).

## Crystal couplings

PXR couples through the X-ray susceptibility Fourier component

```{math}
:label: eq-coherent-radiation-chi-g

\chi_{\mathbf g}=-\frac{r_e\lambda^2}{\pi V_{\rm cell}}S_{\mathbf g},
\qquad
S_{\mathbf g}=\sum_j f_j(\mathbf g,E)\,
e^{i\mathbf g\cdot\mathbf r_j}\,e^{-W_j},
```

the Debye–Waller-weighted unit-cell structure factor over the *electron* density, with the per-element $f_j$ taken from the {cite:t}`chantler1995` tabulation. CBS couples through the screened crystal-potential component

```{math}
:label: eq-coherent-radiation-u-g

U_{\mathbf g}=\frac{4\pi e^2}{V_{\rm cell}\,g^2}
\sum_j\bigl[Z_j-f_j(\mathbf g)\bigr]
e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j},
```

the Mott–Bethe *net screened-nucleus* combination — {eq}`eq-coherent-radiation-u-g` is the Poisson solution for the screened lattice potential — returned in eV and used as the dimensionless $eU_{\mathbf g}/m_ec^2$. The two differ structurally: PXR carries $\sum_j f_j$ with a photon $1/k^2$ denominator, CBS carries $\sum_j(Z_j-f_j)$ with a momentum-transfer $1/g^2$ denominator. The repository's amplitude sign convention fixes their **relative** phase; only $|A_{\rm PXR}+A_{\rm CBS}|^2$ is observable, so the absolute sign of $U_{\mathbf g}$ is a convention.

$U_{\mathbf g}$ is a static electrostatic quantity, so in principle only the non-dispersive $f_0(\mathbf g)$ belongs in it. For edge-prone constituents the implementation uses $Z_j-(f_0+f')$, which makes $U_{\mathbf g}$ weakly photon-energy dependent near an edge; away from edges it is energy-independent as expected. This is a recorded, bounded inconsistency, not a factor error. The per-atom factors $f_j$ are defined in [Atomic form factors](../atomic-physics/atomic-form-factors.md), the cell sum $S_{\mathbf g}$ in [Structure factor and Debye--Waller](../materials/structure-factor.md), and the data provenance in [Atomic data sources](../atomic-physics/atomic-data-sources.md).

Both couplings, the attenuation coefficient, and the refractive index are evaluated on a shared tabulation grid — a 1 eV mesh unioned with the native Chantler nodes of every basis and absorber element, so absorption edges are densely sampled — and interpolated at each segment's own resonance energy {cite:p}`chantler1995,chantler2000`. The couplings are frozen at $\omega_{\rm res}$ across the line rather than re-evaluated per grid bin: the residual is odd about line centre and cancels to first order on integration.

## Amplitudes

For each reflection a polarization pair is built from the observation direction and $\mathbf g$: $\hat{\mathbf e}_s\propto\hat{\mathbf k}\times\mathbf g$ and $\hat{\mathbf e}_p=\hat{\mathbf e}_s\times\hat{\mathbf k}$ (a degenerate $\hat{\mathbf k}\parallel\mathbf g$ falls back to an arbitrary transverse seed). The two polarizations are orthogonal modes and always add **incoherently**.

Per polarization $\hat{\mathbf e}$, with $\mathbf k=k\hat{\mathbf n}$ and $k=\operatorname{Re}n(\omega)\,\omega$ in the implementation, detuning $\Delta=|\mathbf k+\mathbf g|^2-k^2=g^2+2\,\mathbf k\cdot\mathbf g$, and the transverse-to-velocity product $\{\mathbf a;\mathbf b\}=\mathbf a\cdot\mathbf b-(\mathbf a\cdot\mathbf v)(\mathbf b\cdot\mathbf v)$, the two amplitudes are {cite:p}`feranchuk2000`

```{math}
:label: eq-coherent-radiation-a-pxr

A_{\rm PXR}=\frac{\chi_{\mathbf g}}{\Delta}
\Bigl[\bigl(\mathbf v\cdot(\mathbf k+\mathbf g)\bigr)
(\mathbf g\cdot\hat{\mathbf e})-k^2(\mathbf v\cdot\hat{\mathbf e})\Bigr],
```

```{math}
:label: eq-coherent-radiation-a-cbs

A_{\rm CBS}=-\frac{eU_{\mathbf g}/m_ec^2}{\gamma\,(\mathbf v\cdot\mathbf g)}
\left[\{\mathbf g;\hat{\mathbf e}\}
+(\mathbf v\cdot\hat{\mathbf e})
\frac{\{\mathbf k;\mathbf g\}}{\mathbf v\cdot\mathbf g}\right],
\qquad \gamma=(1-\beta^2)^{-1/2}.
```

{eq}`eq-coherent-radiation-a-pxr` corresponds to Eq. (13) of {cite:t}`feranchuk2000` and {eq}`eq-coherent-radiation-a-cbs` to its Eq. (14), also given as Eq. (6) of the {cite:t}`zhai2025` supplement.

```{warning}
The equation numbers above are recorded from the ledger's `Source` fields, not
from the papers themselves — neither source text is held in the repository. What
the `cbs-amplitude` row certifies is that
{eq}`eq-coherent-radiation-a-cbs` is the correct CBS amplitude at the correct
relative normalization against {eq}`eq-coherent-radiation-a-pxr`, **not** that it
faithfully transcribes those numbered equations. A reader with the papers should
confirm the correspondence.
```

They are summed as a **complex** amplitude, $A=A_{\rm PXR}+A_{\rm CBS}$, so PXR/CBS interference is retained inside one segment/reflection contribution. The $1/\gamma$ on the CBS term is the formation-length suppression that becomes numerically important above roughly 100 keV ($1/\gamma=0.84$ at 100 keV); at the tens-of-keV design point $\gamma\approx1.06$ and the factor is a few percent. Both amplitudes are dimensionless.

## Finite interaction time

The finite flight replaces the infinite-crystal delta function of the idealized treatment by the finite-time factor of Eq. (8) of {cite:t}`feranchuk2000`. A segment of length $L$ traversed at speed $\beta$ radiates for $t_L=L/\beta$ (in Å, $c=1$). Integrating a constant amplitude over that centered duration gives the unsquared factor $Q=t_L\,{\rm sinc}(Pt_L/\pi)$ with $P=D(\omega-\omega_{\rm res})/2$, with the frozen resonance denominator $D=1-\operatorname{Re}n(\omega_{\rm res})\,\mathbf v\cdot\hat{\mathbf n}$. In the vacuum limit $D=1-\mathbf v\cdot\hat{\mathbf n}$. Thus the incoherent intensity carries

```{math}
:label: eq-coherent-radiation-lineshape

|Q|^2=t_L^2\,{\rm sinc}^2\!\left(\frac{P t_L}{\pi}\right),
```

using the normalized-sinc convention. At zero detuning {eq}`eq-coherent-radiation-lineshape` is $t_L^2$; in the long-duration limit $|Q|^2/(\pi t_L)\to\delta(P)$ distributionally. The half-width to the first zero is $W=2\pi\hbar c/(D\,t_L)$, which sets the simulated linewidth before any detector broadening. The factor assumes constant velocity and constant amplitude across the segment.

`sinc_cutoff=C` truncates each lineshape at $|Pt_L|>C$ and processes segments in resonance-sorted blocks against the relevant energy coordinates, including on nonuniform grids. It loses about $1/(\pi C)$ of each line's integral (0.3% at $C=100$), and leaves peak heights unaffected. The default evaluates every segment over the full grid exactly.

## Spectral weight and units

Collecting the pieces, one segment and reflection contributes — Eqs. (10) and (12) of {cite:t}`feranchuk2000` —

```{math}
:label: eq-coherent-radiation-spectral-weight

\frac{d^2N}{dE\,d\Omega}
=\frac{\alpha\,\omega_{\rm res}}{4\pi^2\hbar c}\,
|A|^2\,t_L^2\,
{\rm sinc}^2\!\left(\frac{Pt_L}{\pi}\right)T_{\rm abs},
```

summed over the two polarizations, all segments, all electrons, all reflections, and all mosaic orientations, then divided by the electron count. $T_{\rm abs}$ is the Beer–Lambert escape transmission from the segment midpoint. With $|A|^2$ dimensionless, $\omega$ in Å⁻¹, $t_L$ in Å and $\hbar c$ in eV·Å, the prefactor is $1/{\rm eV}$; the result is **photons per eV per steradian per incident electron**. Conventions for consuming that quantity — solid-angle integration, the PXR/CBS component split, external backgrounds — are collected in [Spectral observables](spectral-observables.md).

Segments whose resonance falls outside the grid are dropped, with a 20% pad so that sinc tails reaching into the window still count, and with a hard 10 eV lower floor.

## Physical flights and numerical substeps

When transport splits one physical flight into numerical substeps ([Electron transport](../beam-transport/electron-transport.md)), the physical emitter is the flight, not the row. The default `incoherent` policy therefore sums the rows of one `(electron_id, flight_id)` as a complex field before squaring, so only whole flights add incoherently, and evaluates each row at the flight's representative energy `E_repr_keV` rather than its start energy — which turns the one-point path integral into a midpoint rule.

Without the grouping, splitting a flight into $N$ substeps would give $N$ rows carrying $(t_L/N)^2$ in place of one carrying $t_L^2$, and the line peak would fall roughly as $1/N$: tightening a numerical tolerance would dismantle the line. At frozen energy and clock the grouped finite-time factor recovers the unsplit row exactly by the Dirichlet-kernel identity in the vacuum/zero-dispersion limit. With the production in-medium escape phase, the phase varies along the flight but is not part of that sinc factor; the grouped result therefore converges first-order under refinement rather than remaining exactly invariant.

The grouped reduction is host-only and non-batched. Substepped rows on a device backend, with `components=True`, or with refractive dispersion across `layers` raise rather than silently degrading to a row-incoherent sum.

The subsequent intensity sum over distinct physical flights is an explicit random-phase/independent-emission approximation. An elastic collision changes the trajectory and phase but does not by itself destroy coherence; neglecting cross-flight terms requires those phases to average away through angular or energy spread, formation-length separation, or unresolved environmental recoil. The optional `coherent` policy instead retains phase across the complete single-electron trajectory.

## Mosaicity

`mosaic_fwhm_rad` with `mosaic_nodes > 1` replaces the perfect crystal by an incoherent average over crystallite orientations drawn from a Gaussian mosaic, evaluated as a 2-D Gauss–Hermite product quadrature ($K=$ `mosaic_nodes`$^2$ evaluations per reflection). Each orientation tilts $\mathbf g$ and rebuilds its own polarization pair, resonance energy, amplitudes, and optical depth, so the average broadens PXR and CBS alike and yields the correct, generally asymmetric lineshape and integrated yield. This is mutually exclusive with the analytic energy-shift-only mosaic term applied at detector convolution; applying both double counts. See [Crystal mosaicity](../materials/crystal-mosaicity.md).

## Emission policies

The default `incoherent` policy sums intensities over flights and electrons under that random-phase approximation. `coherent` instead phase-sums segment fields using the emission midpoint position and time plus per-electron bunch timing, and `both` stores the two results side by side from a single electron transport. The cross-electron coherent path is opt-in and its human sign-off is still pending; see [Coherent emission](coherent-emission.md) for the phase convention and the validation boundary.

## Assumptions and limits

- kinematic/Born treatment: no dynamical diffraction, no extinction, no photon multiple scattering, and no depletion of the incident wave;
- straight, constant-velocity, constant-amplitude motion within each transport segment; the finite-time factor and the midpoint pairing both rest on this;
- reflections and polarizations add incoherently; cross-$\mathbf g$ interference is bounded only for the reflection families tested under `cross-reflection-coherence`;
- couplings frozen at $\omega_{\rm res}$ across each line;
- tabulated or CIF-derived crystal structure, form factors, Debye–Waller factors, and the selected orientation fully determine the coupling — a wrong $B$ factor or basis moves the yield directly;
- a single fixed far-field observation direction per evaluation; the finite detector face is handled by summing directions, not by parallax within one;
- Beer–Lambert attenuation handles escape, with no feedback on emission;
- the analytic finite-time profile has a known integral, but numerical spectra still have finite-window, grid, and segment-quadrature errors. Peak height also depends on the physical flight-length distribution;
- detector response is downstream and must not be folded into source physics.

## Validation

Ledger rows: `coherent-line-spectrum` (how $|A|^2$ is used, summed, and normalized), `pxr-amplitude` and `cbs-amplitude` (the couplings themselves), `finite-time-lineshape`, `line-energy-dispersion` (**open discrepancy** on the harmonic sign), `line-absorption-tabulation`, `self-absorption`, `mosaic-mc`, and — for the representative energy and flight grouping — `substep-radiation-invariance`. The evaluation-order rows `line-hkl-batch`, `coherent-line-hkl-batch`, `line-amplitude-fusion`, and `line-gemv-elementwise` cover the batched and fused arithmetic paths, which are algebraically identical to the reference loop up to float reassociation. `closed-form-flux` is `anchored`: the reference includes the exit-angle factor in its escape length. The corrected escape term still awaits an independent derivation check and human sign-off; this status does not settle all absolute comparisons with literature.

Follow those rows in the [validation ledger](../../validation/physics-validation-ledger.md) before scientific use; no row in this section is human `signed-off` yet. Implementation owners: `pyrite.materials.crystal.chi_g`, `pyrite.materials.crystal.U_g`, and `pyrite.montecarlo.spectrum.lines.mc_spectrum`.
