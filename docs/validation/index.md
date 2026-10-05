# Physics Validation

This section records the independent checks used to establish confidence in the physical models and numerical implementations used by Pyrite.

Validation may include comparison against analytic results, published literature, external databases or software, limiting cases, numerical convergence studies, and independent re-derivations.

Start with the generated [status summary](status-summary.md), then browse the [domain inventories](domain-inventories.md). The [detailed validation ledger] (physics-validation-ledger.md) remains the authoritative record.

```{toctree}
:maxdepth: 1
:caption: Validation overview

physics-validation-ledger
status-summary
domain-inventories
methodology
```

```{toctree}
:maxdepth: 1
:caption: Atomic physics

atomic-physics/atomic-form-factor
atomic-physics/structure-factor
```

```{toctree}
:maxdepth: 1
:caption: Beam physics and electron transport

beam-transport/beam-energy-spread-injection
beam-transport/beam-phase-space-injection
beam-transport/beam-phase-space-metrics
beam-transport/gpt-gdf-injection
beam-transport/coherent-line-grid-fringe-spacing
beam-transport/dielectric-bulk-loss
beam-transport/eedl-material-shell-rates
beam-transport/elsepa-elastic-sampling
beam-transport/elsepa-positron-elastic-sampling
beam-transport/bhabha-close
beam-transport/sbethe-positron-stopping
beam-transport/positron-brems-scaling
beam-transport/heitler-annihilation
beam-transport/positron-annihilation-at-rest
beam-transport/elsepa-muffin-tin-inputs
beam-transport/inelastic-angular-deflection
beam-transport/penelope-shell-oscillators
beam-transport/penelope-shell-oscillators-verification
beam-transport/penelope-shell-gos-moments
beam-transport/penelope-shell-gos-moments-verification
beam-transport/penelope-shell-rate-closure
beam-transport/penelope-shell-rate-closure-verification
beam-transport/penelope-shell-soft-hard-partition
beam-transport/penelope-shell-hard-loss-sampling
beam-transport/penelope-shell-hard-recoil
beam-transport/penelope-shell-secondary-direction
beam-transport/shell-soft-hard-transport
beam-transport/shell-secondary-transport
beam-transport/sbethe-atomic-shell-inputs
beam-transport/sbethe-atomic-shell-inputs-verification
beam-transport/gos-core-edge
beam-transport/electron-transport
beam-transport/energy-controlled-propagation
beam-transport/energy-loss-straggling
beam-transport/energy-step-convergence
beam-transport/gos-distant-response
beam-transport/gos-hard-recoil
beam-transport/gos-moller-close
beam-transport/gos-optical-quadrature
beam-transport/gos-soft-hard-partition
beam-transport/line-grid-sinc-convergence
beam-transport/line-grid-resonance-bandwidth
beam-transport/line-grid-resonance-local-spacing
beam-transport/line-spectrum-error-budget
beam-transport/line-window-seeding
beam-transport/legacy-stopping-comparison
beam-transport/longitudinal-bunch-sampling
beam-transport/longitudinal-target-timing
beam-transport/radiation-error-estimators
beam-transport/relativistic-bethe-stopping
beam-transport/sbethe-corrected-stopping
beam-transport/sbethe-material-inputs
beam-transport/substep-radiation-invariance
beam-transport/transport-midpoint-stopping
```

```{toctree}
:maxdepth: 1
:caption: Radiation physics

radiation-physics/absorption-length
radiation-physics/brem-source-comparison
radiation-physics/brem-spectrum
radiation-physics/bremslib-angular-model
radiation-physics/bremslib-coupled-expected-spectrum
radiation-physics/bremslib-radiative-event-spectrum
radiation-physics/bremslib-radiative-partition
radiation-physics/cbs-amplitude
radiation-physics/characteristic-radiation
radiation-physics/eedl-shell-ionization-comparison
radiation-physics/bremslib-angular-schiff
radiation-physics/closed-form-flux
radiation-physics/coherent-emission
radiation-physics/pair-production-sampling
radiation-physics/photon-pair-first-interaction
radiation-physics/coherent-formation-absorption
radiation-physics/coherent-inter-electron-decoherence
radiation-physics/coherent-line-spectrum
radiation-physics/coherent-segment-midpoint-time
radiation-physics/temporal-intensity-profile
radiation-physics/cross-reflection-coherence
radiation-physics/external-brem-subtraction
radiation-physics/finite-footprint-longitudinal-decoherence
radiation-physics/finite-time-lineshape
radiation-physics/line-absorption-tabulation
radiation-physics/kinematic-validity-envelope
radiation-physics/line-energy-dispersion
radiation-physics/narrow-beam-total-attenuation
radiation-physics/pxr-amplitude
radiation-physics/self-absorption
radiation-physics/segment-escape-average
radiation-physics/sinc-bin-far-envelope
radiation-physics/sinc-bin-integration
radiation-physics/sinc-bin-near-far
radiation-physics/transverse-bunch-form-factor
radiation-physics/xray-chi-zero
radiation-physics/xray-in-medium-propagation-phase
radiation-physics/xray-in-medium-resonance
radiation-physics/xray-refractive-index
radiation-physics/zhai-hbn-921-detected
```

```{toctree}
:maxdepth: 1
:caption: Materials and crystallography

materials/2h-tas2-debye-waller-002
materials/2ha-niobium-dichalcogenides
materials/crystal-db-comparison
materials/debye-waller-audit
materials/hbn-structure
materials/hfs2-structure
materials/hopg-debye-waller-00l
materials/mos2-debye-waller-00l
materials/mosaic-analytic
materials/mosaic-mc
materials/multilayer-stack
materials/oriented-v2o5-tis2
materials/pdte2-debye-waller-001
materials/surface-hkl-orientation
```

```{toctree}
:maxdepth: 1
:caption: Geometry

geometry/blazed-groove-geometry
geometry/finite-beam-size
geometry/finite-transverse-crystal
geometry/grazing-beam-projection
geometry/surface-hkl-orientation
```

```{toctree}
:maxdepth: 1
:caption: Detectors and optics

detectors/alexs-charge-diffusion
detectors/alexs-qe-absorption
detectors/detector-eaglexo
detectors/detector-line-broadening
detectors/detector-timepix
detectors/grazing-reflectivity
detectors/pixel-angular-interpolation
detectors/positioned-filter-attenuation
```

```{toctree}
:maxdepth: 1
:caption: Literature cross-checks

literature-ref/external-bremsstrahlung-comparison
literature-ref/zhai-supplementary
```

## Reading the validation record

Individual validation documents should answer four questions wherever possible:

1. **What is being validated?**
2. **What independent reference or derivation is used?**
3. **What constitutes acceptable agreement?**
4. **What was the result?**

A validation document is evidence about the implementation rather than the canonical description of the model itself. Where an associated physics reference page exists, the validation record should link to it rather than duplicate its explanation.

The generated inventories provide exhaustive compact coverage. The groups above organize the detailed evidence pages; the detailed ledger owns status and notes.
