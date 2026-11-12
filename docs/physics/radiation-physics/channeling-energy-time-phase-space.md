# Correlated energy–time phase space and temporal channeling radiation (issue #628)

**Experimental Python API** — [issue #628](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/628). It builds on [incident-energy ensembles (#626)](channeling-energy-ensemble.md) and [energy-resolved axial MCWF (#627)](channeling-energy-resolved-mcwf.md). It models a **classical, mutually incoherent** bunch: electrons with correlated arrival times, kinetic energies, transverse positions and slopes. It is not longitudinal quantum coherence or recoil (#629), and it does not change PyRITE's production electron transport, whose analytic beam still has no chirp ([beam phase space](../beam-transport/beam-phase-space.md#energy-spread)).

The modules are `pyrite.montecarlo.channeling.energy_time` (phase-space contract) and `pyrite.montecarlo.channeling.temporal_radiation` (diagnostics). Their symbols are also exported by `pyrite.montecarlo.channeling`. Units: time fs, energy eV, length Å, slopes rad.

## Incident phase space

An `IncidentPhaseSpace` holds $N$ entrance records $(t_i, T_i, x_i, y_i, x'_i, y'_i)$ with probabilities $w_i$, $\sum_i w_i = 1$. Here $t_i$ is the arrival time at the entrance face (a larger $t$ is the bunch tail), $(x_i, y_i)$ the face position and $(x'_i, y'_i) = (u_x/u_z, u_y/u_z)$ the slopes in the slab frame, whose $z$ is the mean beam axis. Records come from the existing beam samplers, never a parallel beam model:

- `sample_incident_phase_space(T0, n, seed, **beam_kwargs)` calls the transport primary sampler `sample_primary_entry` with the same seed. Positions, slopes, arrival offsets and the uncorrelated `energy_spread_frac` draw are therefore the transport primaries bit for bit.
- `IncidentPhaseSpace.from_gdf(GDFBeam, count, seed=..., z_origin_m=...)` uses the validated GPT/GDF importer (`GDFBeam.sample`). It resamples whole records by `nmacro`, projects them onto the entrance plane and keeps screen crossing times, so every imported correlation is preserved record by record ([GDF contract](../../reference/gdf-beams.md)).
- `from_entry_arrays(...)` adapts any transport-frame arrays (`c t` in Å); the constructor accepts explicit weighted records.

`covariance()` returns the weighted population moments
$C_{ab} = \sum_i w_i (q_{ia} - \langle q_a\rangle)(q_{ib} - \langle q_b\rangle)$ in the order `COORDINATES = (t_fs, T_eV, x_ang, y_ang, xp_rad, yp_rad)`; `correlation()` normalizes it. `arrival_rms_fs` is the bunch-envelope duration. `resample(n, seed=...)` draws equally weighted whole records from one Philox stream (prefix-stable). `to_dict()`/`from_dict()` use the strict JSON mapping `pyrite.channeling.incident-phase-space/1`, whose floats round-trip exactly; `digest()` hashes that mapping. This is an in-memory interchange contract, not a new file format or case identity.

### Linear energy chirp

`LinearEnergyChirp(h, reference_time_fs=None)` adds the one correlation the analytic beam lacks:

$$
T_i = T_i^{(0)} + h\,(t_i - t_\mathrm{ref}), \qquad t_\mathrm{ref} = \textstyle\sum_i w_i t_i \text{ by default},
$$

where $T_i^{(0)}$ already carries any uncorrelated spread $\delta T_u$. The relations

$$
\operatorname{Cov}(t, T) = \operatorname{Cov}(t, T^{(0)}) + h \operatorname{Var} t, \qquad
\operatorname{Var} T = \operatorname{Var} T^{(0)} + 2h \operatorname{Cov}(t, T^{(0)}) + h^2 \operatorname{Var} t
$$

hold exactly for the records, and the energy centroid is unchanged when $t_\mathrm{ref}$ is the mean. A chirp that would make any $T_i \le 0$ is refused, and an already chirped phase space is not chirped again implicitly. $h = 0$ is the identity. Validation: `cr-energy-time-phase-space`.

## Energy nodes and temporal slices

Independent electrons in a static crystal make every single-electron quantum history a function of $(T_i, x_i, y_i, x'_i, y'_i)$ only. Arrival time enters observables only through *which* electrons are grouped together. `quantize_energies(space, max_nodes=K)` keeps up to $K$ distinct record energies exactly. Otherwise it bins records on $K$ equal-width intervals and places each node at its records' weighted mean, so every bin and the ensemble keep their first moment. `quantization_rms_eV` reports the energy resolution given up by solving at node energies.

`time_slices(space, nodes, edges_fs)` partitions records by arrival time and returns the conditional node weights $P(k \mid s)$. A slice observable is

$$
O_s = \sum_k P(k \mid s)\, O(T_k).
$$

`planar_temporal_slices` and `axial_temporal_slices` evaluate $O(T_k)$ with the #626 per-energy solvers (`planar_energy_diagnostic`, `axial_energy_diagnostic`, with `retain_node_results=True`) and return trace, bound and band populations, fluxes and spatial density (planar), or density, survival and OAM populations (axial). The #626 result over the node marginal is in `integrated`. These deterministic paths use the configured **nominal** transverse entry for every record; their provenance says so.

**Limiting case.** Arrival-time spread alone, with every other initial condition identical, gives every slice exactly the monoenergetic populations. An uncorrelated energy spread gives every slice the same mixture. Only an energy–time correlation, such as a chirp, makes slices differ. Validation: `cr-temporal-slice-ensemble`.

## Retarded photon arrival

An electron entering at $(x_i, y_i, 0)$ at time $t_i$ moves at constant $\beta c$ along $z$ and emits at depth $z$. With the common far-field $R/c$ removed, a photon reaching a detector in direction $\hat n = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ arrives at

$$
t_\mathrm{obs} = t_i - \frac{\hat n_\perp \cdot \mathbf r_{\perp i}}{c} + \tau(z), \qquad
\tau(z) = \frac{z\,(1 - \beta\cos\theta)}{\beta c}.
$$

`retarded_delay_fs` evaluates $\tau$ with the cancellation-free Doppler denominator $1-\beta\cos\theta = 1/[\gamma^2(1+\beta)] + 2\beta\sin^2(\theta/2)$. `projected_arrival_fs` evaluates the first two terms. Limits: $\tau(0)=0$; forward emission gives $\tau = z/[\gamma^2(1+\beta)\beta c]$, while $\theta=\pi/2$ gives $z/(\beta c)$. Forward observation removes the transverse term.

Emission is resolved in depth windows $[z_{j-1}, z_j]$ and assumed uniform in depth, and hence in retarded time, inside each window. Each record then spreads window $j$'s emission uniformly over $t_\mathrm{obs} \in [t_i - \hat n_\perp\cdot\mathbf r_{\perp i}/c + \tau(z_{j-1}),\ \ldots + \tau(z_j)]$. Validation: `cr-retarded-photon-arrival`.

- `planar_time_resolved_spectrum(space, solver_factory, L, photon_eV, time_edges_fs, theta_rad=..., phi_rad=..., depth_edges_ang=...)` solves `channeling_spectrum` at each energy node for the cumulative depths, with exit absorption always through the **full** slab. Edges must lie on the shared estimator grid (multiples of `sample_ang` rounded to `dz_ang`), so successive differences are the window emissions exactly. Channels are `intensity`, `capture` and the Stokes numerators `I·Q`, `I·U`. The planar analytic `energy_spread_eV` broadening and `path_limits_ang` censoring are refused.
- `axial_time_resolved_spectrum(space, config, photon_edges_eV, time_edges_fs)` uses `propagate_axial` and the string-0 helicity line catalog at each node. The stored depths are the windows, and the forward line energies are deposited into bins like `axial_energy_spectrum`. Channels are `photons`, `helicity_plus` and `helicity_minus`.

`TimeResolvedSpectrum.spectra[m, c, e]` is per incident electron, integrated over time bin $m$; `rate_per_fs` divides by the bin width. `outside` holds emission arriving outside the time window, so `time_integrated = spectra.sum(0) + outside` is the ordinary energy-ensemble spectrum. `mean_photon_energy_eV` is the intensity-weighted centroid of each time bin, and `polarization` returns the per-bin normalized Stokes $(Q, U)$ (planar) or helicity $(N_+ - N_-)/N$ (axial).

### Bunch envelope versus photon coherence time

The output keeps the two time scales separate:

| Field | Meaning |
|---|---|
| `bunch_envelope_rms_fs` | RMS of $t_i$: the entrance bunch duration |
| `observed_envelope_rms_fs` | RMS of $t_i - \hat n_\perp\cdot\mathbf r_{\perp i}/c$: the envelope seen along $\hat n$ |
| `wavetrain_duration_fs[k]` | $\tau(z_J) - \tau(z_0)$ at node $k$: one electron's retarded emission span |
| `wavetrain_rms_fs[k]`, `mean_wavetrain_rms_fs` | emission-weighted RMS of that single-electron profile |

A single photon's coherence time is bounded by its single-electron wave train, not by the bunch envelope. Mutually incoherent electrons spread *when* photons arrive but do not broaden the time-integrated spectrum. A chirp produces time-dependent photon energies, since slice $s$ radiates at $E_\gamma(T_s)$, and so it broadens the time-integrated spectrum even when every slice is narrow. Near the centroid, $\mathrm dE_\gamma/\mathrm dt = h\, \mathrm dE_\gamma/\mathrm dT$. For a forward line, $E_\gamma \simeq 2\gamma^2 \Delta E_\perp(\gamma)$ gives $\mathrm d\ln E_\gamma/\mathrm d\ln\gamma = 2 + \mathrm d\ln \Delta E_\perp/\mathrm d\ln\gamma$, which is 1.5 in a harmonic well.

## Correlated axial MCWF

`axial_phase_space_mcwf(space, config, slice_edges_fs=..., impact="phase_space")` runs the #627 energy-resolved MCWF with one electron per equally weighted record (use `resample` for weighted imports). Each electron takes:

- its energy node, exact for discrete records, with `max_energy_nodes` bounding solver rebuilds;
- tilt `config.tilt_rad` plus its slope deviation from the centroid;
- with `impact="phase_space"`, the entry state translated by its centroid-relative position modulo the supercell. `impact="entry"` keeps the given entry.

This is the path that keeps the energy–transverse correlations of every record. Slices run separately with seeds keyed by slice index, so a slice is statistically, not bitwise, comparable with an unsliced run. The slices must cover every record. `integrated_*` are the count-weighted slice sums. MCWF radiation, when requested through `photon_energy_edges_eV`, is integrated through the final depth and indexed by *incident* arrival slice; use the deterministic axial path for retarded intra-crystal arrival.

## Example

```python
import numpy as np
from pyrite.quantum import planar_profile
from pyrite.montecarlo.channeling import (
    LinearEnergyChirp,
    planar_solver_factory,
    planar_temporal_slices,
    planar_time_resolved_spectrum,
    sample_incident_phase_space,
)

space = sample_incident_phase_space(
    18.5e6,
    2000,
    628,
    bunch_length_fs=300.0,
    energy_spread_frac=1e-4,
    chirp=LinearEnergyChirp(500.0),  # +500 eV/fs: the tail is more energetic
)
factory = planar_solver_factory(
    planar_profile("silicon", (1, 1, 0), 293.15), n_cells=2, max_order=5
)
slices = planar_temporal_slices(space, factory, 2000.0, 100.0, np.linspace(-900, 900, 7))
spectrum = planar_time_resolved_spectrum(
    space,
    factory,
    2000.0,
    np.linspace(1e3, 200e3, 400),
    np.linspace(-900, 900, 19),
    max_energy_nodes=9,
    dz_ang=100.0,
    sample_ang=500.0,
)
spectrum.mean_photon_energy_eV  # rises across the bunch
spectrum.bunch_envelope_rms_fs, spectrum.mean_wavetrain_rms_fs
```

## Limitations and decisions

- The timing assumes the entrance face is $z = 0$ and the mean velocity is along $z$: a tilted planar `surface_normal` is refused, and analytic records sampled with `beam_dir` or face tilts are outside the timing model (the axial MCWF path, which subtracts the centroid slope, is not affected). The axial path times its $4\pi$ line yields with the forward delay, exact for a forward detector only.
- Static crystal and constant electron energy inside the crystal: no energy loss and no energy-changing jumps. Emission inside a depth window is taken as uniform in depth. The deterministic paths use the nominal transverse entry, and one scalar detector direction per time-resolved spectrum (the aperture average is not retarded per sub-direction).
- Records are a sampled representation of the bunch. Energy-node quantization trades resolution (`quantization_rms_eV`) for solver cost; raise `max_energy_nodes` to check convergence.
- The chirp lives on the channeling-diagnostic phase space only. Adding it to `pyrite.Beam`/`BeamSpec` for production transport would change campaign configuration and case identity, and is left as a separate decision.

## Validation

`Validation: cr-energy-time-phase-space`, `cr-temporal-slice-ensemble`, `cr-retarded-photon-arrival` — see the [channeling ledger](../../validation/ledger-channeling.md). Anchors: `tests/montecarlo/channeling/test_channeling_energy_time.py`, `tests/montecarlo/channeling/test_channeling_temporal_radiation.py`, `tests/montecarlo/quantum/test_channeling_phase_space_mcwf.py`.
