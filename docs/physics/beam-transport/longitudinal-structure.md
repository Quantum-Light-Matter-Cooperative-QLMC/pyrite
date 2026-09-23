# Longitudinal bunch structure

The transverse and energy phase space of the incident beam is documented in [Beam phase space](beam-phase-space.md). This page covers the remaining coordinate: **when** each electron arrives.

Arrival time matters only for coherent emission. The incoherent spectrum sums intensities and is blind to $t_0$; the coherent sum carries $e^{i\omega t_{\rm abs}}$ per segment, so the bunch's temporal distribution is exactly the quantity that decides whether emitters add in phase. See [Coherent-emission tracking](../radiation-physics/coherent-emission.md).

## Two clocks

Transport keeps two independent time-like quantities per segment:

```{list-table} Time quantities carried through transport.
:name: tbl-longitudinal-clocks

* - `t_ang` / `t_start_ang`
  - **Relative age.** Accumulated $\int ds/\beta$ since this electron entered,
    zero at entry. A property of the trajectory.
* - `t0_ang`
  - **Bunch offset.** This electron's arrival time relative to the bunch
    centroid. A property of the beam, constant along the trajectory.
```

Both are in $\AA$ with $c = 1$. The radiation kernels form the absolute emission time as $t_{\rm abs} = t_{\rm ang} + L_{\rm ang}/(2\beta) + t_{0,\rm ang}$ — age, half-flight correction to the segment midpoint, and bunch offset.

Input is in femtoseconds and converts once through

```{math}
:label: eq-longitudinal-time-unit

c = 2997.924580~\AA\text{\,fs}^{-1}.
```

## Policies

A beam's `longitudinal` sub-table selects one policy. All of them produce a per-electron offset array, centered on its own sample mean so that $t = 0$ is the realized centroid rather than the nominal one.

```{list-table} Longitudinal policies.
:name: tbl-longitudinal-policies
:header-rows: 1

* - `kind`
  - Distribution
  - Duration set by
* - `gaussian`
  - $\Delta t \sim \mathcal N(0,\sigma)$
  - `envelope_rms_fs` directly
* - `compressed`
  - one Gaussian at the derived microbunch width
  - target line and `retained_coherence`
* - `microtrain`
  - Gaussian envelope of discrete microbunches
  - envelope from `envelope_rms_fs`, microbunch width and spacing from the
    target line
```

`compressed` is terminology for a short bunch, not a chirp model: the energy spread of [Beam phase space](beam-phase-space.md#energy-spread) is drawn independently of the arrival time under every policy, so $\langle t\,\delta\rangle = 0$ holds by construction and the reported longitudinal emittance is exactly $\sigma_t\sigma_\delta$.

Two legacy spellings remain and are mutually exclusive with the resolved policy: `bunch_length_fs` with `long_shape = "gaussian" | "uniform"` (the uniform variant is a flat top of the *same* RMS, half-width $\sqrt3\,\sigma$), and `long_offsets_fs`, an explicit per-electron array for measured or externally generated profiles.

## Target-line resolution

`compressed` and `microtrain` do not take a duration; they take a *coherence target* and derive one. The resolution runs once per case, against that case's material, energy, and geometry.

For a basal reflection whose reciprocal vector is parallel to the catalog surface normal, the emitted photon wavenumber is

```{math}
:label: eq-longitudinal-photon-wavenumber

k_\gamma =
\frac{ \beta \, |\mathbf g| \cos\theta_\mathrm{tilt}}{1 - \beta\cos\theta_\mathrm{obs}}
\quad[\AA^{-1}],
```

giving photon energy $E_\gamma = \hbar c\,k_\gamma$, optical period $T = h/E_\gamma$, and angular frequency $\Omega = E_\gamma/\hbar = 2\pi/T$.

A Gaussian arrival-time distribution of RMS width $\sigma_t$ has intensity form factor

```{math}
:label: eq-longitudinal-form-factor

|F(\Omega)|^{2} = \exp\!\left[-(\Omega\sigma_t)^{2}\right],
```

so requiring that a fraction $\eta$ = `retained_coherence` survives fixes the width:

```{math}
:label: eq-longitudinal-sigma-from-eta

\sigma_t = \frac{\sqrt{-\ln\eta}}{\Omega}.
```

Microbunch spacing is `spacing_periods` optical periods, $T_{\rm spacing} = m T$. The dominant reflection is the first catalog-pinned positive reflection parallel to the orientation axis unless `target_reflection` pins one explicitly; the resolved record keeps the reflection, photon energy, wavelength, period, and a provenance string, so a checkpoint records what it targeted and why.

## Microtrain construction

A microtrain draw is built from three independent normal draws plus a grid snap:

```{math}
:label: eq-longitudinal-microtrain

\Delta t = m\,T_{\rm spacing}\;+\;\xi_{\rm micro}\;+\;\xi_{\rm jitter},
\qquad
m = \operatorname{round}\!\left(\frac{\xi_{\rm env}}{T_{\rm spacing}}\right),
```

with $\xi_{\rm micro}\sim\mathcal N(0,\sigma_{\rm micro})$, $\xi_{\rm jitter}\sim\mathcal N(0,\sigma_{\rm jitter})$, and $\xi_{\rm env}$ normal with the *deconvolved* envelope width

```{math}
:label: eq-longitudinal-center-variance

\sigma_{\rm centers}^{2} =
\sigma_{\rm env}^{2} - \sigma_{\rm micro}^{2} - \sigma_{\rm jitter}^{2}.
```

Subtracting the widths that {eq}`eq-longitudinal-microtrain` adds back keeps the *total* RMS duration equal to the requested envelope, so changing the modulation does not silently change the bunch length. A non-positive {eq}`eq-longitudinal-center-variance` is rejected rather than clamped: an envelope narrower than its own microbunches is not a train.

`modulation_depth` $D$ mixes the train against an unmodulated Gaussian of the full envelope width, per electron with probability $D$. $D = 1$ is a pure train, $D = 0$ is the plain Gaussian, and intermediate values model partial bunching.

### Consequences for bunching

- Snapping centers to the spacing grid adds a negligible $T_{\rm spacing}^2/12$ to the center variance.
- Timing jitter multiplies the coherent enhancement by $\exp[-\Omega^{2}\sigma_{\rm jitter}^{2}]$; jitter comparable to the optical period destroys the effect it was meant to expose.
- Partial depth gives approximately $D^{2}\eta$ when the envelope itself is decoherent, so a half-modulated train retains about a quarter of the enhancement, not half.

## Sampling and reproducibility

The bunch draw takes its own RNG child stream (`SeedSequence(seed).spawn(4)[3]`), disjoint from the free-path and scattering streams and from the transverse and energy-spread streams. Enabling or changing a bunch policy therefore never perturbs the trajectories themselves: the same electrons are transported, and only their $t_0$ labels change. The degenerate case (no policy, no legacy field) returns all-zero offsets bit-for-bit, the pure-geometry coherent limit.

## Limiting cases

- $\sigma_t \to 0$: point bunch, all electrons in phase, $|F|^2 \to 1$.
- $\Omega\sigma_t \gg 1$: {eq}`eq-longitudinal-form-factor` vanishes and the coherent sum degenerates to the incoherent one up to shot noise.
- $\eta \to 1$: {eq}`eq-longitudinal-sigma-from-eta` gives zero microbunch width, an unreachable ideal, and the reason `retained_coherence` is bounded strictly above zero and at most one.
- `spacing_periods = 1`: $\Omega T_{\rm spacing} = 2\pi$, adjacent microbunches exactly one optical cycle apart.

## Out of scope

Chirp (energy–time correlation), space charge and bunch evolution during drift, shot-to-shot centroid drift beyond the sampled jitter, and any absolute photons-per-second normalization from `bunch_charge_pc` and `rep_rate_hz`, which remain inert to the transport draw.

## Validation

`Validation: longitudinal-bunch-sampling` for the sampling laws and centering; `Validation: longitudinal-target-timing` for {eq}`eq-longitudinal-photon-wavenumber`–{eq}`eq-longitudinal-sigma-from-eta`. See the [physics validation ledger](../../validation/physics-validation-ledger.md) and the write-ups for [bunch sampling](../../validation/beam-transport/longitudinal-bunch-sampling.md) and [target timing](../../validation/beam-transport/longitudinal-target-timing.md).
