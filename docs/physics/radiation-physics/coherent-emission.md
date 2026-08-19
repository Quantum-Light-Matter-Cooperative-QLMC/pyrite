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

Under `xray_dispersion="refractive"` the segment-to-segment propagation phase
moves onto the same dispersion relation: each field picks up
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
evaluation route `coherent-line-hkl-batch` is only `filtered`.

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
   discrepancy, which sets where each line sits.

Track status in the [`coherent-emission` ledger
row](../../validation/physics-validation-ledger.md) and the [validation
write-up](../../validation/radiation-physics/coherent-emission.md); sign-off
follows the [physics validation workflow](../../validation/methodology.md).
Implementation owner: `pyrite.montecarlo.spectrum.lines.mc_spectrum`
(`coherent=True`).
