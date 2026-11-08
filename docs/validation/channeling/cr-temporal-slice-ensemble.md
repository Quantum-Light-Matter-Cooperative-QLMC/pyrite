# `cr-temporal-slice-ensemble` — conditional-mixture arrival-time slices

- **Claim**: `cr-temporal-slice-ensemble` — `montecarlo/channeling/energy_time.py::{EnergyNodes,quantize_energies,TimeSlices,time_slices}`; `montecarlo/channeling/temporal_radiation.py::{_slice_mix,planar_temporal_slices,axial_temporal_slices,axial_phase_space_mcwf,planar_time_resolved_spectrum,axial_time_resolved_spectrum}` — incoherent ensemble average of epic #620 Stage I/III, $\bar O = \sum_i w_i O(T_i)$, conditioned on arrival-time slices; #626 node solvers and #627 `axial_energy_mcwf` with `entry_factory`.
- **Verifier**: fresh context; derivation written before the bodies were read.

## Independent derivation

Electrons are independent and the crystal is static. Each single-electron history is therefore a function of its entrance state $(T_i, \mathbf r_{\perp i}, \mathbf x'_i)$ only. Arrival time $t_i$ only labels which records are grouped together. A density-linear observable $O$ averaged over slice $S_s = \{i : t_i \in [e_s, e_{s+1})\}$ is

$$
O_s = \frac{\sum_{i\in S_s} w_i\, O(T_i)}{P(s)}, \qquad P(s) = \sum_{i\in S_s} w_i .
$$

If $O$ is evaluated only at node energies $T_k$, with record $i$ assigned node $k(i)$, this becomes

$$
O_s = \sum_k P(k\mid s)\, O(T_k), \qquad
P(k\mid s) = \frac{\sum_{i\in S_s,\,k(i)=k} w_i}{P(s)} .
$$

The law of total expectation gives $\sum_s P(s)\,O_s + \sum_{i\notin\cup S_s} w_i O(T_{k(i)}) = \sum_k P(k)\,O(T_k)$, which is the #626 result on the node marginal.

Quantization. With equal-width bins on $[\min T, \max T]$ and node $T_k = \sum_{i\in k} w_i T_i / \sum_{i\in k} w_i$, every bin keeps its first moment. Hence $\sum_k P(k)\,T_k = \langle T\rangle$. The lost resolution is $\sigma_q = [\sum_i w_i (T_i - T_{k(i)})^2]^{1/2}$. Discrete energies $\le K$ need no binning, so $\sigma_q = 0$.

Limits:

- An arrival spread alone (one $T$) gives $P(k\mid s) = \delta_{k0}$, so every slice equals the monoenergetic $O$ exactly.
- A product distribution $p(t,T) = p(t)\,p(T)$ gives $P(k\mid s) = P(k)$, the same mixture in every slice.
- A step chirp puts each slice on its own node.

MCWF mapping of a classical record onto the axial entry:

- *Energy.* The electron runs at its node energy, which is exact for discrete records.
- *Direction.* The axial carrier tilt is the electron's direction relative to the string axis (initial transverse momentum $k\,\boldsymbol\theta$; `axial_diagnostic.py:394`). If `config.tilt_rad` is defined as the **centroid** direction relative to the axis, a record with slope $\mathbf x'_i$ about the slab $z$ axis has $\boldsymbol\theta_i = \boldsymbol\theta_{\rm cfg} + (\mathbf x'_i - \langle\mathbf x'\rangle)$. Small-angle additivity of slopes holds to $O(\theta^3)$.
- *Impact.* For a beam much wider than a cell, the impact relative to the lattice matters only modulo the periodic supercell. The centroid is placed at the nominal entry and the others are shifted by $\psi(\mathbf r) \to \psi(\mathbf r - \mathbf d_i)$ with $\mathbf d_i = (\mathbf r_{\perp i} - \langle \mathbf r_\perp\rangle) \bmod \mathbf L$. On a periodic FFT grid the phase is $e^{-i\mathbf k\cdot\mathbf d}$ with $\mathbf k \in 2\pi\mathbb Z/\mathbf L$, so the $\bmod$ is exact.

For an equally weighted MCWF slice, the per-electron mean estimates $O_s$ without bias. The incident-normalized total is $\sum_s (n_s/N)\,O_s$ when the slices cover every record.

## Source-to-code diff

| Item | Derivation | Code | Result |
|---|---|---|---|
| slice index | $[e_s, e_{s+1})$, last closed | `searchsorted(...,'right')-1`, `t == edges[-1]` → last (energy_time.py:559-561) | match |
| $P(s)$, $P(k\mid s)$ | as above | `bincount`, `add.at(joint)`, divided by $P(s)$ (:565-573) | match |
| outside mass | $1 - \sum_s P(s)$ | :588 | match |
| node at bin mean | weighted mean | `first[keep] / mass[keep]` (:498-503) | match |
| $\sigma_q$ | as above | :506 | match |
| exact discrete | distinct $\le K$ | `np.unique`, `method="exact"` (:487-493) | match |
| $O_s$ | $\sum_k P(k\mid s)\,O_k$ | `tensordot(node_weights, stack, (1, 0))` (temporal_radiation.py:533) | match |
| node order | per-node result $k$ is node $k$ | `nodes.ensemble()` passes explicit nodes in order; `quadrature()` masks only $w = 0$, and `EnergyNodes` already drops empty bins (energy_ensemble.py:120-122) | match |
| per-node normalization | per incident electron | #626 `kept.append(result)` holds the raw single-energy solve, not weight-multiplied (energy_ensemble.py:372-373) | match |
| MCWF energy | node energy per record | `energies = nodes.energies_eV[nodes.node_index]` (:701) | match |
| MCWF tilt | $\boldsymbol\theta_{\rm cfg} + \mathbf x'_i - \langle\mathbf x'\rangle$ | :644-645 | match |
| MCWF impact | $\psi(\mathbf r - \mathbf d_i)$, $\bmod$ supercell | `translate(base, mod(r - <r>, extent))`; `Translated` is $\psi(\mathbf r - \mathbf d)$ (states.py:376); periodized sampling (realspace.py:111-114) | match |
| slice mean | per assigned electron | #627 divides bin sums by counts and uses weights `counts/n_electrons` (axial_energy_mcwf.py:914-924), with `n_electrons = records.size` (:725) | match |
| combined | $\sum_s (n_s/N)\,O_s$ | `probability = counts / space.size`, `tensordot` (:732, :748); coverage enforced (:699-700) | match |
| record ↔ electron | electron `eid` = `records[eid]` | factory checks the energy match (:715-719) | match |

No term diverges.

## Numerical checks

Script: the verifier's disposable `check_phase_space.py` (not committed).

- Binned quantization ($N = 4000$ weighted records, $K = 1, 5, 17$):
  - per-bin mass and mean equal brute force;
  - every node holds a single bin;
  - global first moment kept to $\le 1.1\times10^{-8}$ eV at 18 MeV (relative $6\times10^{-16}$);
  - $\sigma_q$ equals the independent formula.
- Discrete $\{17, 18, 19\}$ MeV gives `exact` with $\sigma_q = 0$.
- Slice mixture against brute-force conditional averages of random node observables: max error $8\times10^{-16}$.
- Outside mass matches.
- Law of total expectation reconstructs $\sum_k P(k)\,O_k$ to $3\times10^{-16}$; `_slice_mix` agrees bitwise.
- Anchors are green. They cover planar slices against direct `DensityMatrixSolver.propagate`, axial slices against `axial_diagnostic` bitwise, MCWF records against direct `AxialMCWF` runs with the translated entry and tilt, and the count-weighted slice reduction.

## Verdict

- **Filters**: units pass; limits pass (mono, product and step-chirp limits follow from $P(k\mid s)$; discrete records unsmeared); signs/conventions pass (tilt is a centroid-relative direction about the axis; translation is $\psi(\mathbf r - \mathbf d)$).
- **Re-derivation**: `matches`.
- **Verdict**: `rederived`.

## Findings (non-blocking)

1. *Contract edge.* Zero-weight records get `node_index = -1`, or a neighbouring node's index, because `remap = cumsum(keep) - 1` is applied to their dropped bin (energy_time.py:491-492, :500-501). For records with energies $\{1, 2, 2, 3, 3, 3\}$ MeV and $w_0 = 0$, the indices are `[-1, 0, 0, 1, 1, 1]`. Every moment is unaffected because the weight is zero, and `time_slices` only adds $0$ to the last column. It still contradicts the `EnergyNodes` docstring "`node_index[i]` is record `i`'s node". MCWF never sees it (uniform weights only).
2. *Implicit frame identity.* The slab-frame $(x, y)$ of positions and slopes is taken to be the axial projected-cell grid $(x, y)$ of `config` (supercell `period_ang * cells`). That is reasonable, but the ledger Assumptions do not state it.
3. *Convention.* The ledger could say explicitly that `config.tilt_rad` is the **centroid** tilt relative to the axis, which is why $\langle\mathbf x'\rangle$ is subtracted.
4. *Coverage.* `planar_temporal_slices` reports trace, bound, free and overflow fluxes per slice but omits `reprojection_flux` and `entry_flux`, so a per-slice probability balance cannot be formed. This is an omission, not an error.

## Resolution (implementer, 2026-11-07)

1. Zero-weight records whose bin carries no mass now map to the nearest retained node; anchored by `test_zero_weight_records_map_to_a_retained_node`.
2. and 3. The ledger Assumptions now state that slab `(x, y)` is the axial projected-cell grid frame and that `config.tilt_rad` is the centroid tilt.
4. `planar_temporal_slices` also reports `reprojection_flux` and `entry_flux`, so a per-slice balance can be formed (checked in the anchor).
