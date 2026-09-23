# Recommendation: transition radiation through 100 MeV

**Status:** Research recommendation, September 2026. No production TR component or detector-specific omission bound is established here. Estimates below are illustrative, not simulated yields or independent validation.

## Recommendation

Defer a production transition-radiation (TR) component for the current tens-of-keV electron, keV-photon workflow, subject to checking representative detector-specific omission bounds. Prioritize a TR calculation for MeV thin-film and multilayer soft-X-ray studies. Do not use its small electron energy loss as evidence that it is a negligible photon background.

For general electron transport toward 100 MeV, validated high-energy scattering, collision stopping, radiative energy loss, and secondary-particle treatment take priority. For a weak soft-X-ray line measurement, TR can deserve attention earlier. The decision belongs to an observable and geometry, not a universal beam-energy threshold.

## Current repository context

Production spectra contain PXR/CBS, bremsstrahlung, and characteristic radiation. There is no dedicated TR emission component. Material-boundary navigation is not itself a radiation calculation. Existing bulk photon dispersion and attenuation likewise do not establish that boundary radiation is included.

The [channeling research note](channeling-radiation-physics.md) already identifies TR at entry/exit surfaces and requests an include-or-bound decision under the proposed `cr-transition-radiation-bound` validation item. Its order-of-magnitude photon-yield estimate motivates investigation; it is not a detector-integrated exclusion bound.

Current [stopping assumptions](../../physics/beam-transport/stopping-power.md) omit radiative feedback, transported delta rays, and density-effect corrections. The [elastic model](../../physics/beam-transport/elastic-scattering.md) includes a Browning fit whose stated range ends at 30 keV. These limitations remain even after improving the [energy grids](../beam-transport/energy-grid-recommendations.md).

## Energy scales and interpretation

TR occurs when a charged particle crosses a change in dielectric response. Optical TR has been measured from gold under 30 keV electron irradiation {cite:p}`coenen2011`. Therefore, deferring TR for keV X-ray predictions does not justify omitting it from a future optical or EUV model.

For a relativistic electron crossing an idealized vacuum/material interface, the PDG estimate gives the total radiated energy scale {cite:p}`zyla2020passage`:

```{math}
:label: eq-research-tr-interface-energy

W_{\mathrm{TR}} \simeq \frac{\alpha}{3}\gamma\hbar\omega_p,
\qquad \gamma = 1 + \frac{T}{m_e c^2}.
```

Here $T$ is electron kinetic energy, $m_e c^2$ its rest energy, $\alpha$ the fine-structure constant, and $\hbar\omega_p$ the material plasma energy. The associated photon roll-off scale is of order $\gamma\hbar\omega_p$, not a sharp cutoff. The simple plasma description suppresses detailed dielectric structure and absorption edges; a finite foil also requires interference between its two surfaces.

`Validation: tr-interface-energy-scale` — research-only claim, ledgered as unverified in the [core physics ledger](../../validation/ledger-core-coherent-physics.md). The expression applies to a relativistic interface estimate, not to the nonrelativistic limit. In the matched-medium limit the boundary contribution must vanish; the corresponding material-vacuum estimate vanishes as the plasma energy tends to zero. Do not extrapolate this formula to $T\to0$.

Using an illustrative Si-like plasma energy of 31 eV and $m_ec^2=0.51099895$ MeV:

| Electron kinetic energy (MeV) | Lorentz factor | Photon scale (keV) | Energy per interface (eV) |
| --- | --- | --- | --- |
| 10 | 20.6 | 0.64 | 1.6 |
| 50 | 98.8 | 3.1 | 7.5 |
| 100 | 196.7 | 6.1 | 14.8 |

These values are arithmetic evaluations of the stated approximation, without angular acceptance, absorption, or foil interference. They illustrate how a tiny fraction of electron kinetic energy can appear in the photon band of interest. They do not determine photon counts: counts require integrating the number spectrum over a stated finite energy interval.

## Determine relevance using detected observables

For representative materials and geometries, calculate the spectral-angular TR yield and propagate it through the same acceptance, filters, and response as the other radiation components. Compare contributions to integrated line windows, background sidebands, total detected counts, and inferred line/background ratios. An illustrative acceptance criterion is less than 1% change in the target observable; the experiment must set its actual tolerance.

Cover current tens-of-keV cases, MeV cases, thin free-standing films, substrates, and multiple interfaces. Include oblique incidence and the actual surface normal. Relativistic forward concentration does not justify a universal off-axis exclusion: backward radiation and diffracted TR require geometry-specific review.

For periodic structures, PXR and diffracted TR can occupy overlapping bands and directions; a 5.7 MeV multilayer study explicitly considers both {cite:p}`shevelev2024`. Review their amplitude relationship before treating them as independent additive intensities. A small omitted intensity can still alter a coherent signal through an interference term.

## Proposed implementation boundary

1. Start with planar slab/stack geometries and explicit dielectric-data validity. Support complex dielectric response in the relevant photon band; the high-frequency plasma approximation is not a universal optical/edge model.
2. Record actual boundary crossings: electron identity, position, direction, kinetic energy, time, surface normal, and adjacent media. Artificial numerical subdivisions must not emit TR.
3. Preserve phase across entry/exit surfaces and subsequent interfaces where coherence survives. Account for absorption and photon propagation. Independent interface intensities are a limiting approximation, not the default thin-film model. Geant4's treatment explicitly includes formation zones and multilayer interference {cite:p}`geant4tr`.
4. Expose a separate source contribution and provenance. If TR interferes with PXR/CBS, retain the cross term and document the component accounting instead of implying every displayed intensity is additive.
5. Initially assess spectra on existing trajectories only where the omitted TR energy loss is bounded. Add recoil/feedback if that bound fails. Do not infer that this justifies omitting bremsstrahlung feedback at high electron energy.
6. Resolve spectral fringes and angular structure separately from broad logarithmic grid coverage. Use exact or numerical integration through detector channels and acceptance.

Analytic single-interface scaling is useful for screening, but summing these estimates is not automatically a rigorous bound for a coherent multilayer. Nor is all-angle radiated energy divided by a nominal photon energy a bound on the detected spectral background.

## Validation and priority

Require a source derivation, implementation-linked ledger entries, and independent review before advertising support. Test zero dielectric contrast, the vanishing film in a common surrounding medium, and the dephased-interface limit. Check units, polarization, angular normalization, absorption, and invariance under transport substep changes. Reproduce a published spectral-angular result before comparing detector-integrated backgrounds in application scenes.

The priority is conditional:

| Application | Recommended TR priority |
| --- | --- |
| Current tens-of-keV transport / typical keV X-rays | Omission study first; implementation can wait if the bound passes |
| Optical/EUV radiation at the same electron energy | Reassess immediately |
| MeV thin-film soft-X-ray or channeling-line studies | Include or demonstrate a detector-specific exclusion |
| General transport approaching 100 MeV | Address major bulk transport deficiencies first; assess TR alongside spectral observables |
| Purpose-built multilayer radiator | Essential source physics |

This is a recommendation, not an approved implementation plan or a claim that the current repository supports transport through 100 MeV.
