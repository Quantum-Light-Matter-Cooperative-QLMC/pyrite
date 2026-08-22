# Coherent-emission tracking

Emission is a **profile policy**, not a transient run flag: a profile carries
`emission = incoherent | coherent | both` (`pyrite profile set --emission ...`,
or unioned incrementally with `pyrite profile add/remove --coherent/--incoherent`).
Default scans remain incoherent and retain their historical checkpoint identity;
coherent scans use identity-qualified stems, so the two modes cannot overwrite
each other. `both` runs a **single** electron transport per case and stores both
the incoherent `spec` and a `spec_coherent` built from the same segments, so a
paired coherent-vs-incoherent comparison (peak and integrated-flux ratios) is one
checkpoint with two spectra instead of two separate runs. The analysis app
exposes an Emission selector, gated per checkpoint on the stored spectra, to
switch every plot between the two.

## Model

The coherent path reuses **every** per-segment quantity of the [line
kernel](coherent-radiation.md) — the resonance energy, the complex amplitude
$A=A_{\rm PXR}+A_{\rm CBS}$ per polarization, the finite-time factor, and the
escape transmission — and only changes where the square is taken. Instead of
accumulating $|A_j|^2|Q_j|^2$, it accumulates a complex field and squares at the
end. For one reciprocal vector and polarization,

$$
E_j(E)=\sqrt{\frac{\alpha\,\omega}{4\pi^2\hbar c}\,T_{{\rm abs},j}}\;
A_j\,Q_j(E)\,
\exp\!\bigl\{i\bigl[\omega\,t_{{\rm abs},j}
-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r_j\bigr]\bigr\},
$$

$$
E_{\rm tot}(E,\hat{\mathbf n})=\sum_j E_j(E),
\qquad
\frac{d^2N}{dE\,d\Omega}\propto\bigl|E_{\rm tot}\bigr|^2 .
$$

$Q_j=t_L\,{\rm sinc}(Pt_L/\pi)$ is the **unsquared** finite-segment amplitude,
whose modulus-square is the incoherent $t_L^2\,{\rm sinc}^2$.

## Phase convention

The temporal phase uses $t_{\rm abs}=t_{\rm ang}+L_{\rm ang}/(2\beta)+t_{0,\rm ang}$,
in Å with $c=1$:

- transport keeps `t_ang` as the segment-**start** age for schema compatibility;
- the half-flight term $L/(2\beta)$ moves it to the segment midpoint, pairing time
  with the stored midpoint position `r_mid` under the same constant-velocity
  assumption as the finite-time factor;
- $t_{0,\rm ang}$ is each electron's longitudinal bunch offset. All-zero $t_0$
  (`bunch_length_fs=None`) is the documented degenerate pure-geometry limit —
  still physics, but position-phase only.

The spatial phase carries $\omega\hat{\mathbf n}\cdot\mathbf r_j$ as far-field
retardation and $\mathbf g\cdot\mathbf r_j$ as the reciprocal-harmonic phase. The
$\mathbf g$ term follows the repository's structure-factor convention
$S(\mathbf g)=\sum F\exp(+i\mathbf g\cdot\mathbf R)$, whose susceptibility
harmonic is $\chi_{\mathbf g}\exp(-i\mathbf g\cdot\mathbf r)$; this mapping was
checked as part of the ledger re-derivation.

The segment-to-segment propagation phase rides the same dispersion relation:
each field picks up
$-\delta(E)\,\omega(E)\,L_{{\rm esc},j}$ over its in-crystal escape path — the
real partner of the Beer–Lambert amplitude factor already applied over that same
path. This is refused for layered absorbers, whose per-layer $\delta$ is not
modelled. See [Photon escape and in-medium
dispersion](photon-escape-and-dispersion.md).

## What stays incoherent

Polarizations, reciprocal vectors, and mosaic orientations still add as
intensities. The phased sum runs **within** one (reflection, orientation,
polarization) row, on the grounds that distinct reflections are spectrally
separated. That separation has not been quantified and is not yet a ledger row.

`components=True` is rejected under the coherent policy: once the
$A_{\rm PXR}A_{\rm CBS}^{*}$ cross term survives the sum, no uniquely additive
PXR/CBS split exists.

## Two coherence scales

One sum produces both:

- **intra-electron** — segments of a single trajectory, always present, and
  surviving even when bunch phases are fully scrambled;
- **inter-electron / superradiant** — the spread of $t_0$ across the bunch, whose
  $|\langle e^{i\omega t_0}\rangle|^2$ is the Gaussian bunch form factor
  $\exp[-(\omega\sigma_z)^2]$.

### The transverse partner

The inter-electron scale above, as originally implemented, covered only the
**longitudinal** offset $t_0$. A
bunch with a finite transverse spot (`beam_fwhm_mm`) also gives each electron a
constant transverse displacement $\Delta\mathbf r_\perp$, which the spatial
phase reads directly through $\mathbf r_j$. Every cross-electron term therefore
carries $\exp[-i(\omega\hat{\mathbf n}+\mathbf g)\cdot\Delta\mathbf r_\perp]$,
and averaging over a Gaussian spot of r.m.s. width $\sigma_\perp$ gives the
transverse form factor

$$
\bigl|\langle e^{-i\mathbf q_\perp\cdot\Delta\mathbf r_\perp}\rangle\bigr|^2
=\exp\bigl[-(q_\perp\sigma_\perp)^2\bigr],
\qquad
\mathbf q_\perp=\bigl(\omega\hat{\mathbf n}+\mathbf g\bigr)_\perp,
$$

the exact analogue of $\exp[-(\omega\sigma_z)^2]$. With
$q_\perp\sim1\,\text{Å}^{-1}$ and any spot above a nanometre this factor is
numerically zero, so the ensemble observable keeps **only** the intra-electron
sum $\sum_e\bigl|\sum_{j\in e}E_j\bigr|^2$.

**The implementation originally did not take that average.** It evaluated one
realization of the sampled offsets at one $\hat{\mathbf n}$, so the stored
spectrum was a single speckle draw rather than the ensemble mean. Measured on
the reference HOPG case, the coherent peak height scattered 30–41% seed to
seed with a finite spot against a 7% Monte Carlo counting-noise floor on the
incoherent path, and that contrast did **not** fall as the electron count
grew — speckle contrast is independent of the number of randomly phased
emitters. Averaging over seeds recovered the correct spot-independent
intra-electron enhancement (2.83–3.25 at $N_e=300$, stable across 1 µm, 50 µm
and 1 mm spots and across electron counts), which is what the form factor
delivers deterministically and for free.

Integrating over the detector face did not rescue it numerically either: the
speckle angular scale $\lambda/D_\perp\approx2\times10^{-4}$ rad is far finer
than any affordable tile spacing, so the face integral plateaued at ~11% of
the reference by `n_side=9` where the incoherent one had converged to 0.01%.
Convergence would need $n_{\rm side}\approx\Delta\theta/(\lambda/D_\perp)\approx180$,
some 32000 directions per case — tiling was never going to be the fix.

**As of 2026-08-21, both offsets are blended analytically.** The combined
longitudinal $\times$ transverse form factor, its fresh-context derivation,
and the implementation (an empirical characteristic function of the actual
sampled per-electron offsets rather than a closed-form-parametrized $\sigma$,
which extends unchanged to elliptical/Courant–Snyder spots) are in
[Validation: `coherent-inter-electron-decoherence`](../../validation/radiation-physics/coherent-inter-electron-decoherence.md).
Finite crystal footprints (`crystal_width_mm`/`crystal_height_mm`) need one
extra boundary. Their transverse offset also perturbs escape attenuation, so
the combined phase-only form-factor blend does not apply. Longitudinal arrival
time is independent of that geometry, however. The finite-footprint path keeps
each sampled electron's transverse phase, hit/miss history, and attenuation in
$S_e$, then applies the exact conditional Gaussian average
$(1-F_z)\sum_e|S_e|^2+F_z|\sum_eS_e|^2$, with
$F_z=\exp[-(\omega c\sigma_t)^2]$. A short bunch therefore retains
cross-electron enhancement; a long bunch continuously reaches the grouped
floor without the previous macroscopic longitudinal speckle. This conditional
result does not additionally average over transverse bunch/transport
realizations, so partially coherent finite-footprint output may retain
transverse diffraction/speckle. See
[Validation: `finite-footprint-longitudinal-decoherence`](../../validation/radiation-physics/finite-footprint-longitudinal-decoherence.md).
The `beam_fwhm_mm=None` point source remains what
it always was: the fully degenerate limit whose cross-electron terms are
maximally constructive, giving an enhancement that grows linearly with the
simulated electron count (33.2× at $N_e=80$, 107.5× at $N_e=300$) — a property
of the sampling, not of the physics, and not a stand-in for the analytic
average.

## Expected limits

- `incoherent` (the default) preserves the previous path bit-for-bit.
- One segment, or one electron, gives only the self-term and is identical in both
  modes.
- One straight constant-velocity flight is unchanged when represented by two
  contiguous half-segments.
- Identical in-phase emitters give the $N^2$ intensity limit before per-electron
  normalization.
- A bunch much longer than the wavelength — or independently scrambled
  per-electron $t_0$ — removes inter-electron cross terms and leaves
  $\sum_e|\sum_{j\in e}E_j|^2$. This equals the per-segment incoherent sum only
  when each electron contributes one segment or its internal segment phases also
  decohere; it is **not** identical to the incoherent policy in general.
- A Gaussian longitudinal distribution gives the bunch form factor
  $\exp[-(\omega\sigma_z)^2]$.

## Validation boundary

The `coherent-emission` and `coherent-segment-midpoint-time` rows are
`rederived`: fresh-context derivations reproduced the phase, the midpoint pairing,
and the decoherent limits with no divergent sign, factor, or unit, and the limits
above are anchored in `tests/montecarlo/test_coherent_emission.py`. The batched
evaluation route `coherent-line-hkl-batch` is only `filtered`. The combined
longitudinal/transverse decoherence blend is its own `rederived` row,
`coherent-inter-electron-decoherence`.

**Human sign-off is still pending on all of them**, and the following remain open
before scientific use:

1. cross-reflection ($\mathbf g\neq\mathbf g'$) coherence is dropped. It is now
   ledgered and bounded (`cross-reflection-coherence`, `filtered`) below ~1.6% of
   the integrated yield over the catalog's basal-plane families, but only
   `filtered`: the bound is worst-case in relative phase, and a reflection set
   with near-degenerate resonances at the observation angle is not covered;
2. physical-charge / macro-particle weighting of the bunch is a separate,
   unfinished claim — the stored spectrum is still per incident electron;
3. representative CPU/GPU cases are not anchored and the complex-grid memory cost
   is not quantified;
4. the coherent path inherits the open `line-energy-dispersion` harmonic-sign
   discrepancy, which sets where each line sits;
5. the combined longitudinal/transverse ensemble blend is valid for the
   infinite slab. A finite footprint instead uses the separately `filtered`
   longitudinal average conditional on one sampled transverse/transport
   realization; its transverse ensemble average remains open. The far-field
   phase itself is exact in the linear term; its dropped curvature is a
   separate, smaller effect bounded in
   [Coherent PXR and CBS radiation](coherent-radiation.md#emission-geometry).

Track status in the [`coherent-emission` ledger
row](../../validation/physics-validation-ledger.md) and the [validation
write-up](../../validation/radiation-physics/coherent-emission.md); sign-off
follows the [physics validation workflow](../../validation/methodology.md).
Implementation owner: `pyrite.montecarlo.spectrum.lines.mc_spectrum`
(`coherent=True`).
