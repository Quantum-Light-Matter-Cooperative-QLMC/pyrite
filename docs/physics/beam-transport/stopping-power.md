# Stopping power and the energy cutoff

Between elastic collisions PyRITE removes energy continuously rather than sampling individual inelastic events. This is the condensed-history (continuous-slowing-down) approximation: discrete excitation and ionization losses are replaced by their mean rate $dE/ds$, evaluated along the flight{cite:p}`nistestar,akkerman1978`.

The local scale-separation check is the fractional mean loss over an elastic free path, $|dE/ds|\lambda_{\rm el}/E$: when it is small, energy evolves slowly compared with the explicitly sampled directional changes. The inelastic and elastic mean free paths are distinct, material- and energy-dependent scales{cite:p}`akkerman1978,shinotsuka2015`. Optional straggling restores fluctuations around that mean without changing it.

## Current production model: SBETHE

Production transport resolves a material-level SBETHE collision-stopping table{cite:p}`salvat2024sbethe` for each layer. SBETHE evaluates the corrected Bethe expression with DHFS shell and Fano density-effect corrections. Its source states that the corrected expression applies above a material-dependent `ECUT` of about 1 keV for electrons; below `ECUT`, SBETHE uses an empirical extrapolation. PyRITE consumes the tabulated result rather than reproducing the Fortran expression in the transport kernel.

The catalog's atomic number densities $n_i$ (in Å$^{-3}$) set the bulk mass density and the compound mean excitation energy supplied to SBETHE:

```{math}
:label: eq-sbethe-material-inputs

\rho = \frac{10^{24}}{N_{\rm A}}\sum_i n_i A_i,
\qquad
\ln I = \frac{\sum_i n_i Z_i\ln I_i}{\sum_i n_i Z_i}.
```

Here $A_i$ is in g mol$^{-1}$, $I_i$ is the elemental mean excitation energy, and the resulting $\rho$ is in g cm$^{-3}$. The logarithmic rule assumes independent-atom Bragg additivity; a one-element composition returns its elemental $I_i$. Table identity includes the composition, density, $I$, optional band gap, source digest, and generation inputs.

For positive table nodes $(E_i,S_i)$, transport interpolates in log energy and log stopping magnitude:

```{math}
:label: eq-sbethe-interpolation

f = \frac{\ln E-\ln E_i}{\ln E_{i+1}-\ln E_i},
\qquad
\frac{dE}{ds} = -\exp\!\left[(1-f)\ln S_i+f\ln S_{i+1}\right].
```

$E$ is in keV and $S$ in keV Å$^{-1}$. This rule recovers each native node, keeps the rate negative, and is continuous between nodes; its slope need not be continuous at a node. Resolved tables span 1 keV to 1 GeV. The host rejects energies outside that range, so a transport cutoff below 1 keV is unsupported. The optional Urban fluctuations use the same SBETHE mean through a shared scale factor. `Validation: sbethe-material-inputs` and `Validation: sbethe-corrected-stopping` are `rederived`; CUDA stopping anchors have run on hardware. Human sign-off remains pending.

The former Joy–Luo/Berger–Seltzer splice is retained only for reference comparisons; its range and yield measurements do not describe production SBETHE transport. See the [historical equations and measurements](../../validation/beam-transport/legacy-stopping-comparison.md) and [independent splice validation](../../validation/beam-transport/relativistic-bethe-stopping.md).

## Evaluation along a flight

Where {eq}`eq-sbethe-interpolation` is evaluated is set by `energy_model`: the
frozen rule holds it at the flight-start energy, the midpoint rule
evaluates it at $(E_{\rm start}+E_{\rm end})/2$ through one predictor–corrector
pass. In the earlier splice's thick 5 keV carbon case, where
$|\frac{dE}{ds}|$ grows as $E$ falls, the frozen rule overstated the mean path
length by 1.2%; that figure has not been remeasured with SBETHE. See
[Electron transport](electron-transport.md#energy-controlled-propagation).

## Transport cutoff

Transport of an electron ends when its kinetic energy falls to `E_cut_keV`.
The line-radiation population defaults to 5 keV; the bremsstrahlung and
characteristic populations normally continue to 1 keV, the lower SBETHE
table boundary.

- segments below the cutoff radiate essentially nothing in the spectral window of
  interest, so the discarded path length does not carry the observables, while
  the elastic mean free path keeps shortening and the step count keeps growing;
- the remaining energy is deposited and not tracked in the default continuous
  mode. Opt-in shell soft/hard transport can launch secondaries above its
  configured threshold;
- historical cutoff-sensitivity measurements below 1 keV used the former
  stopping law and do not establish SBETHE behavior below its table boundary.

The frozen rule reaches the cutoff by overshooting it and clipping, which inflates
the CSDA range. The midpoint rule instead solves the truncation distance for
$E_{\rm end} = E_{\rm cut}$ exactly,

```{math}
:label: eq-stopping-cutoff-distance

s_{\rm cut} =
\frac{E_{\rm cut} - E_{\rm start}}
     {\frac{\mathrm{d}E}{\mathrm{d}s}\bigl((E_{\rm start}+E_{\rm cut})/2\bigr)},
```

so the overshoot disappears. `E_cut_by_electrons` allows a per-electron cutoff
where a study needs it.

## Assumptions and limits

- Transport uses continuous SBETHE **collision** stopping by default. When BremsLib tables are unavailable or uncoupled transport is selected, bremsstrahlung and characteristic radiation are scored from the electron histories and do not debit electron energy. The default [BremsLib soft/hard radiative mode](../radiation-physics/hard-bremsstrahlung-events.md) adds its separate soft loss and sampled hard-photon debits.
- With `straggling=True`, Urban fluctuations are scaled to the SBETHE mean; the default is deterministic. See `Validation: energy-loss-straggling` and the [derivation and observable checks](../../validation/beam-transport/energy-loss-straggling.md).
- The continuous model omits discrete knock-on electrons. The opt-in [shell soft/hard mode](shell-soft-hard-transport.md) samples hard inelastic transfers and can launch secondaries above its threshold; it uses SBETHE to close the mean collision loss.
- SBETHE includes shell and density-effect corrections above its material-dependent `ECUT`; below that it supplies an empirical extrapolation. PyRITE accepts its table only from 1 keV to 1 GeV and does not extrapolate outside those nodes.

## Validation

`Validation: sbethe-material-inputs` covers composition, density, and mean excitation energy; `Validation: sbethe-corrected-stopping` covers the SBETHE source quantity, table parsing, interpolation, domain, and CPU/CUDA use. Both ledger rows are `rederived`, with human sign-off pending. `Validation: transport-midpoint-stopping` covers the evaluation point and {eq}`eq-stopping-cutoff-distance`; its historical observable measurements used the earlier stopping law. See the [physics validation ledger](../../validation/physics-validation-ledger.md).
