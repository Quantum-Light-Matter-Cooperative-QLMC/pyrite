# Coherent-emission tracking

A profile selects `emission = incoherent | coherent | both`. The default is `incoherent`. Coherent outputs have distinct checkpoint identities. With `both`, one transport run supplies the segments for `spec` and `spec_coherent`, allowing paired comparisons from the same trajectories. The analysis app's Emission selector exposes the spectra stored in each checkpoint.

## Model

The coherent path reuses **every** per-segment quantity of the [line kernel](coherent-radiation.md) — the resonance energy, the complex amplitude $A=A_{\rm PXR}+A_{\rm CBS}$ per polarization, the finite-time factor, and the escape transmission — and only changes where the square is taken. Instead of accumulating $|A_j|^2|Q_j|^2$, it accumulates a complex field and squares at the end. For one reciprocal vector and polarization,

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

$Q_j=t_L\,{\rm sinc}(Pt_L/\pi)$ is the **unsquared** finite-segment amplitude, whose modulus-square is the incoherent $t_L^2\,{\rm sinc}^2$.

## Phase convention

The temporal phase uses $t_{\rm abs}=t_{\rm ang}+L_{\rm ang}/(2\beta)+t_{0,\rm ang}$, in Å with $c=1$:

- transport keeps `t_ang` as the segment-**start** age for each segment;
- the half-flight term $L/(2\beta)$ moves it to the segment midpoint, pairing time with the stored midpoint position `r_mid` under the same constant-velocity assumption as the finite-time factor;
- $t_{0,\rm ang}$ is each electron's longitudinal bunch offset. All-zero $t_0$ (`bunch_length_fs=None`) is the documented degenerate pure-geometry limit with no per-electron arrival-time spread.

The spatial phase carries $\omega\hat{\mathbf n}\cdot\mathbf r_j$ as far-field retardation and $\mathbf g\cdot\mathbf r_j$ as the reciprocal-harmonic phase. The $\mathbf g$ term is the conjugate-field phase of the harmonic that transfers $-\mathbf g$ to the photon. The amplitude it multiplies is built from the Friedel-mate coupling $\chi_{-\mathbf g}^{*}$, $U_{\mathbf g}$ of {eq}`eq-coherent-radiation-friedel-pairing`, not from the crystallographic $\chi_{\mathbf g}$. The two agree when $f''=0$ (the case the `coherent-emission` record assumed). See `line-energy-dispersion`.

The segment-to-segment propagation phase rides the same dispersion relation: each field picks up $-\delta(E)\,\omega(E)\,L_{{\rm esc},j}$ over its in-crystal escape path — the real partner of the Beer–Lambert amplitude factor already applied over that same path. This is refused for layered absorbers, whose per-layer $\delta$ is not modelled. See [Photon escape and in-medium dispersion](photon-escape-and-dispersion.md).

## What stays incoherent

Polarizations, reciprocal vectors, and mosaic orientations still add as intensities. The phased sum runs within one (reflection, orientation, polarization) row. The cross-reflection approximation is bounded for the tested basal-plane families under `cross-reflection-coherence` (`filtered`); near-degenerate reflection sets are outside that validation scope.

`components=True` is rejected under the coherent policy: once the $A_{\rm PXR}A_{\rm CBS}^{*}$ cross term survives the sum, no uniquely additive PXR/CBS split exists.

## Two coherence scales

One sum produces both:

- **intra-electron** — segments of a single trajectory, always present, and surviving even when bunch phases are fully scrambled;
- **inter-electron / superradiant** — the spread of $t_0$ across the bunch, whose $|\langle e^{i\omega t_0}\rangle|^2$ is the Gaussian bunch form factor $\exp[-(\omega\sigma_z)^2]$.

### The transverse partner

A bunch with a finite transverse spot (`beam_fwhm_mm`) also gives each electron a constant transverse displacement $\Delta\mathbf r_\perp$, which the spatial phase reads directly through $\mathbf r_j$. Every cross-electron term therefore carries $\exp[-i(\omega\hat{\mathbf n}+\mathbf g)\cdot\Delta\mathbf r_\perp]$, and averaging over a Gaussian spot of r.m.s. width $\sigma_\perp$ gives the transverse form factor

$$
\bigl|\langle e^{-i\mathbf q_\perp\cdot\Delta\mathbf r_\perp}\rangle\bigr|^2
=\exp\bigl[-(q_\perp\sigma_\perp)^2\bigr],
\qquad
\mathbf q_\perp=\bigl(\omega\hat{\mathbf n}+\mathbf g\bigr)_\perp,
$$

the exact analogue of $\exp[-(\omega\sigma_z)^2]$. With $q_\perp\sim1\,\text{Å}^{-1}$ and any spot above a nanometre this factor is numerically zero, so the ensemble observable keeps **only** the intra-electron sum $\sum_e\bigl|\sum_{j\in e}E_j\bigr|^2$.

For an infinite slab, the implementation blends the fully coherent intensity with the sum of per-electron coherent intensities. The blend factor is the squared empirical characteristic function of the sampled longitudinal and transverse offsets. This supports elliptical and Courant–Snyder spots without assuming a circular Gaussian width. The Gaussian expression above is the corresponding analytic limit.

The factorization assumes independent offsets and offset-independent transport and escape amplitudes. Its derivation and implementation are recorded in [Validation: `coherent-inter-electron-decoherence`](../../validation/radiation-physics/coherent-inter-electron-decoherence.md).

Finite crystal footprints (`crystal_width_mm`/`crystal_height_mm`) need one extra boundary. Their transverse offset also perturbs escape attenuation, so the combined phase-only form-factor blend does not apply. Longitudinal arrival time is independent of that geometry, however. The finite-footprint path keeps each sampled electron's transverse phase, hit/miss history, and attenuation in $S_e$, then applies the exact conditional Gaussian average $(1-F_z)\sum_e|S_e|^2+F_z|\sum_eS_e|^2$, with $F_z=\exp[-(\omega c\sigma_t)^2]$. A short bunch therefore retains cross-electron enhancement; a long bunch continuously reaches the grouped floor without sampling longitudinal phases explicitly. This conditional result does not additionally average over transverse bunch/transport realizations, so partially coherent finite-footprint output may retain transverse diffraction/speckle. See [Validation: `finite-footprint-longitudinal-decoherence`](../../validation/radiation-physics/finite-footprint-longitudinal-decoherence.md). With no transverse or longitudinal offset spread, the blend retains the full cross-electron sum. Production weights the inter-electron excess by the physical bunch population `bunch_charge_pc / e`, using distinct Monte Carlo pairs. The incident sample count normalizes the estimate and includes missed entries. Infinite slabs pair complete sampled fields without a separate empirical form factor; finite footprints keep the analytic Gaussian longitudinal average. Negative estimates refuse as insufficient sampling. See [physical population derivation and limitations](../../validation/radiation-physics/coherent-physical-bunch-population.md). Low-level `mc_spectrum` calls omitting `physical_electrons` retain the historical sampled-population blend.

## Expected limits

- One contributing physical flight gives the same result in both modes, including any coherently grouped numerical substeps. One electron with multiple flights can retain interference between those flights.
- Splitting a constant-velocity flight is exact for the vacuum finite-time integral at fixed amplitude. The production model samples escape phase and attenuation at segment midpoints, so subdivision instead has a residual quadrature error; see `coherent-segment-midpoint-time`.
- Identical in-phase emitters give the $N^2$ intensity limit before per-electron normalization.
- A bunch much longer than the wavelength — or independently scrambled per-electron $t_0$ — removes inter-electron cross terms and leaves $\sum_e|\sum_{j\in e}E_j|^2$. This equals the per-segment incoherent sum only when each electron contributes one segment or its internal segment phases also decohere; it is **not** identical to the incoherent policy in general.
- A Gaussian longitudinal distribution gives the bunch form factor $\exp[-(\omega\sigma_z)^2]$.

## Validation boundary

The `coherent-emission` and `coherent-segment-midpoint-time` rows are `rederived`: fresh-context derivations reproduced the phase, the midpoint pairing, and the decoherent limits with no divergent sign, factor, or unit, and the limits above are anchored in `tests/montecarlo/test_coherent_emission.py`. The batched evaluation route `coherent-line-hkl-batch` is only `filtered`. The combined longitudinal/transverse decoherence blend is its own `rederived` row, `coherent-inter-electron-decoherence`.

**Human sign-off is still pending on all of them**, and the following remain open before scientific use:

1. cross-reflection ($\mathbf g\neq\mathbf g'$) coherence is dropped. It is bounded (`cross-reflection-coherence`, `filtered`) below ~1.6% of the integrated yield over the catalog's basal-plane families, but only `filtered`: the bound is worst-case in relative phase, and a reflection set with near-degenerate resonances at the observation angle is not covered;
2. physical bunch charge weights the production cross-electron excess; the stored spectrum stays per incident electron. The distinct-pair estimator has limiting-case regressions but still needs independent verification and a production sampling-error certificate;
3. CUDA blend regression tests cover the fast paths, but do not establish validation across all materials, geometries, and precision settings;
4. the harmonic sign that places each line is resolved under `line-energy-dispersion`. The Bijvoet-sensitive coupling pairing {eq}`eq-coherent-radiation-friedel-pairing` is anchored for the incoherent path. The coherent phase sum reuses the same tables;
5. the combined longitudinal/transverse ensemble blend is valid for the infinite slab. A finite footprint instead uses the separately `filtered` longitudinal average conditional on one sampled transverse/transport realization; its transverse ensemble average remains open. The far-field phase itself is exact in the linear term; its dropped curvature is a separate, smaller effect bounded in [Coherent PXR and CBS radiation](coherent-radiation.md#emission-geometry).

Track status in the [`coherent-emission` ledger row](../../validation/physics-validation-ledger.md) and the [validation write-up](../../validation/radiation-physics/coherent-emission.md); sign-off follows the [physics validation workflow](../../validation/methodology.md). Implementation owner: `pyrite.montecarlo.spectrum.lines.mc_spectrum` (`coherent=True`).
