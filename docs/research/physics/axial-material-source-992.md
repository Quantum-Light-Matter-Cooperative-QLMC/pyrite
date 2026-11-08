# Axial material source: first Si evidence (#992)

Research evidence for [#992](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/992).
It does not admit an axial run. `cr-axial-persisted-channel` remains
`unverified`: persistence, exactly-once combination and native ownership
belong to #994.

## Estimator

`fiber_on_axis_source` (claim
[`cr-axial-fiber-regression`](../../validation/ledger-channeling.md#cr-axial-fiber-regression))
estimates the on-axis two-time source of the carrier-fibred absorbing model.
That model has continuum TDS strings and electronic kicks, and both move the
Bloch carrier. Each primary history carries companion vectors $J_b\hat u$,
which receive the same operators and norm factors. The dense one-sector
`regression_correlation` cannot represent these jumps. Emission coherence
survives a jump through the gain term, and the anchor's dense oracle
resolves that survival: a pure-damping model misses by more than 10 standard
errors.

## Material runs

`checks/axial_fiber_source.py` runs Si at normal incidence along the axis.
The setup uses one cell, continuum TDS strings, electronic kicks with
harmonics $|n|\le4$ and the provisional cut rule
$U_\mathrm{max}+(U_\mathrm{max}-U_\mathrm{min})$. Depth points sit every
15 Å over the slab. The ownership boundary is $U_\mathrm{max}$, so only final
states below it count. Photon intensity is attenuated by
$e^{-\mu(T-z)}$ (EPDL2025 $\mu$). Record:
[`axial-source/fiber-source-992-remote.json`](../../validation/check-records/axial-source/fiber-source-992-remote.json).

Yields are photons per steradian per incident electron on axis, integrated
over 0.8–8 keV, from the retained subspace only. Errors are batch-means SEMs
over 100-primary chunks. Peaks are taken on the spectrum smeared with the
detector resolution (200 eV FWHM) on a 50 eV grid, so peak positions carry a
±25 eV quantization. "Model/WK" is the model peak divided by the Watson & Koehler
forward-equivalent peak, minus one.

| Arm | Job | $N$ | Yield $(10^{-4}\,\mathrm{sr}^{-1})$ | Retained at exit | Model/WK peaks |
|---|---|---|---|---|---|
| ⟨111⟩ 2.99 MeV, main | 4284 | 4000 | 3.73(6) | 0.427 | 3.36→2.95 keV (−12%); 4.90→4.15 (−15%); 1.63 not resolved |
| ⟨110⟩ 2.99 MeV, main | 4285 | 4000 | 8.95(8) | 0.476 | 3.88→3.50 (−10%); 5.85 not resolved |
| ⟨111⟩, window $M=12$, box $B=24$ | 4286 | 2000 | 3.85(8) | 0.416 | 3.00 (−11%); 4.20 (−14%) |
| ⟨111⟩, depth step 10 Å | 4525 | 2000 | 3.74(7) | 0.419 | 2.95 (−12%); 4.15 (−15%) |
| ⟨111⟩, no ownership boundary | 4288 | 2000 | 4.65(10) | 0.419 | 2.95 (−12%); 4.15 (−15%) |
| ⟨111⟩, TDS only (no kicks) | 4289 | 2000 | 1.79(5) | 0.432 | 1.63→1.40 (−14%); 2.95 (−12%); 4.40 (−10%) |
| ⟨111⟩ 2.78 MeV | 4290 | 2000 | 2.95(7) | 0.425 | 3.03→2.60 (−14%); 4.43→3.60 (−19%) |
| ⟨111⟩ 2.37 MeV | 4291 | 2000 | 1.81(4) | 0.421 | 2.28→2.00 (−12%); 3.41 not resolved |

Findings:

- **Numerical convergence of the retained source.** A larger window and box
  changes the yield by $+3(3)\%$ (1.2σ) and the peaks by one grid bin. A finer
  depth grid changes the yield by $+0.4(25)\%$. Both are within statistics at
  this sample size. One-cell supercell and the Bloch-sector set were not
  varied: the fibre engine is one-cell only.
- **Line energies against measurement.** Every resolved model peak sits 10–19%
  below the Watson & Koehler forward-equivalent energy, with the same sign at
  all three energies and both axes. Andersen, Bonderup & Pantell judge Watson's
  energies 10–20% high from the detector angle. The model is therefore
  consistent with the measurement within that stated systematic. This test is
  weak: it cannot distinguish a model error of order 10% from the detector
  systematic, and the 4–5 keV maxima are 2–4 keV wide humps, not resolved lines.
- **Line widths.** The 3 keV peak has a smeared FWHM of 0.45–0.64 keV against
  0.2 keV resolution. No numerical measured axial width is available (see
  below), so widths are not compared.
- **Absolute yield.** The output is finite and incident-normalized, but no
  absolute measured axial yield is available to compare.
- **Physics sensitivity.** Electronic kicks double the retained source relative
  to TDS alone (1.79 → 3.73), and the K-edge-region 1.63 keV line is resolved
  only without kicks. Retained weight at exit is 0.42–0.48; the rest leaves as
  outgoing (40–53%) or unresolved (5–12%) records, whose later emission is
  native-owned or unrepresented.

## Measured data used

- Watson & Koehler, Phys. Rev. B 25, 3079 (1982), Table II
  (`tests/data/channeling/watson-koehler-1982-si-axial.json`; Zotero
  `DXH6CSY9`). Si ⟨111⟩, ⟨110⟩ and ⟨211⟩ at 1.96–3.09 MeV, 1.1 µm crystal,
  Si(Li) detector with 200 eV FWHM at 0.046 rad. The table gives
  forward-equivalent peak energies only; intensities are relative to the
  Si K line, with no absolute yield or line width.
- Andersen, Bonderup & Pantell, Ann. Rev. Nucl. Part. Sci. 33, 453 (1983)
  (Zotero `4V42ILVZ`). Figure 14 shows Si ⟨111⟩ at 4 MeV, 0.5 µm, as a 3-D
  surface with no tabulated values. Bound–bound line widths agree "within
  15%" with single-scattering calculations. That review judges Watson's line
  energies 10–20% high, from the detector angle.

No absolute measured axial yield or numerical axial line width was found in
the library. The original Andersen et al. measurement (the review's ref. 28,
Nucl. Instrum. Methods 194 (1982)) is not in the library.

## Photon refraction in the Doppler phase

With $n=1-\delta$ and $\delta=\omega_p^2/(2E^2)$ (free-electron form,
$\hbar\omega_p=31.05$ eV for Si), the phase $E(1-\beta n)/\hbar v$ moves a
line of rest gap $\Delta$ by
$|\Delta E|/E\le\beta\omega_p^2/(2E^2D)$, with $D=1-\beta$. At 2.99 MeV the
shift is at most 1.7% at 1.63 keV, 0.39% at 3.36 keV and 0.19% at 4.90 keV.
At 2.37 MeV the bounds are 1.1%, 0.27% and 0.12%. These shifts are small
next to the 10–20% spread between laboratories. Near the Si K edge
(1.839 keV), anomalous dispersion changes $\delta$ by an $O(1)$ factor,
which leaves the low line's bound at a few percent. Refraction is neglected
in the estimator.

## Capture and bound→bound

Capture (free→bound) and bound→bound share final states. Their cross term
therefore survives Hamiltonian evolution, and the ownership boundary keeps
both in the axial block. This needs a decision, with these options:

1. Report the axial block as one quantity, as now.
2. Split it by initial-state class into bound→bound, capture and a signed
   cross term. This means extending `fiber_lowering_currents` with an
   initial-state mask, and the cross term is reported, not dropped.
3. Score capture as an upper bound only (ADR-0018 v1).

**Decision (2026-11-08): option 1 for #992.** The axial block is reported as
one quantity that includes bound→bound, capture and their cross term. The
measured comparisons available (Watson & Koehler line energies; Andersen et
al. widths) do not separate capture from bound→bound, so a split would add no
testable content here. Dropping or bounding capture alone would discard a cross
term that the Hamiltonian evolution keeps. If a consumer needs the split,
option 2 is the route; it is not required for admission in #994.

## Native CBS and the free→free block

The fibred source with `boundary="max"` scores only final states below
$U_\mathrm{max}$, so it never scores free→free. Lowering currents need the
initial state above the final one, so the transitions that `boundary="none"`
adds are exactly the free→free elements inside the retained cut. Their share is
measured directly: at Si ⟨111⟩ 2.99 MeV the yield is
$4.65(10)\times10^{-4}$ /sr without a boundary and
$3.73(6)\times10^{-4}$ /sr with it. Free→free inside the cut is therefore
$0.92(11)\times10^{-4}$ /sr, about 20% of the retained-subspace source.

| Population | Native CBS | Fibred source (`boundary="max"`) | Free→free block |
|---|---|---|---|
| Zone row, exact axis (this run) | zero-order $\mathbf g$ has $\mathbf v\cdot\mathbf g=0$: no line | not scored | **omitted by both** |
| Zone row, tilted incidence | line above the native resonance-root threshold | not scored | supplied natively |
| Handed-off electron (left $P(\kappa)$) | tilted `on_shell_direction` row: line scored | emission after a record leaves the retained subspace is not represented | supplied natively, no overlap |
| Nonzero axial-order $\mathbf g$, incoherent brems | native | not scored | n/a (stays native, ADR-0018) |

Exact alignment grants no exemption. On the exact axis a production adapter
must either score the in-cut free→free block axially (`boundary="none"` on
zone rows) or accept a ~20% omission of the retained source. With tilted
incidence it must keep `boundary="max"`, or the block is counted twice.
Free→free above the retained cut is outside both estimators on the exact axis;
its size is not bounded here. This is input to #994 admission, not a
production rule.
