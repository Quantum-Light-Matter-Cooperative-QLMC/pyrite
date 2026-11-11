---
orphan: true
---

# Native CBS on planar zone rows against the free→free block (#1055)

**Status:** research evidence for #1055 and
[`cr-planar-free-free-block`](../../validation/ledger-channeling.md#cr-planar-free-free-block).
This note records a measurement. It changes no production rule; the ownership
decision belongs to the maintainer (#1055 Plan item 4).

## Question

Planar zone rows (ADR-0018 "Emission") are straight mean segments along the
entry direction, tagged `channeling_zone`. No scorer reads the tag, so the
native PXR/CBS line kernel scores every zone-row electron as a free plane
wave. Most of that population is bound or tracked above the barrier, and its
emission already goes to `spec_channeling` (bound→bound plus capture). The
axial case measured the same overlap at 5.6× its free→free block (#994).
Does planar double count, and by how much?

## Method

Driver: `checks/planar_zone_row_cbs.py`. For each case it transports the
production runner's planar case once and scores, on the beam axis, a zero
aperture and the same line grid:

- **native zone, own row**: the production native line kernel
  (`_cxr_for_segments`, incoherent) on the zone rows only, restricted to the
  channeling plane's own reciprocal row (`g` parallel to the plane normal; the
  case's `hkl_list` holds `±(220)`-type only). Per incident line electron.
  The standard error comes from 20 electron chunks.
- **native zone, other rows**: the same on every other reflection.
- **free→free (in cut)**: `channeling_spectrum(..., final_states="tracked")`
  through the production producer (`case_channeling_spectrum`), on the same
  censored paths. Its `free_free_spectrum` is the expected emission of
  lowering currents between tracked above-barrier states
  (`barrier ≤ E < barrier + margin`). It is the planar analogue of the axial
  fibre's `boundary="none"` minus `"max"`. It is an expected value, so it has
  no sampling error of its own.
- **CR**: the production `spec_channeling` (bound→bound plus capture).

The driver resets the cached zone solver to its `gamma0` basis before CR
scoring (#1090). It also checks that the tracked catalog leaves the bound and
capture part unchanged, to 1e-10 of peak.

Settings: 17 MeV electrons, 20 µm slab, plane-wave beam, `H = 16`, 16 Bloch
cells, `dz = 250 Å`, margin 300 eV, 293.15 K, 2000 line electrons, CPU. The
slab is tilted 3° from the catalog cut's surface axis (⟨110⟩ for Si, ⟨100⟩ for
diamond), far from any axial critical angle. The tilt azimuth is solved so
the vacuum angle to the planes is ψ. Planes: Si (1,−1,0) and diamond
(0,1,−1), both {110}, which contain the cut axis. Planar critical angles
(`ψ_c = √(2 U_depth / pv)`): Si 1.55 mrad (21.0 eV well), diamond 1.64 mrad
(23.6 eV).

Remote: qlmc cpu, job 10912 (SHA `b9b1327a2`, six cases) and job 10919 (SHA
`83e5826f1`, Si 4.0 mrad; the only change is the driver's SHA lookup).
Record: `docs/validation/check-records/planar-zone/zone-cbs-1055.json`.

## Results

Band integrals in photons/sr per incident line electron. `own_line` is the
window `[0.85 E_q, 1.15 E_cl]` around the own-row first-order line, with the
quantum `E_q = ħ²(2kg − g²)/(2γm)/(1 − β)` and the classical
`E_cl = ħcβ g ψ/(1 − β)`. The ratio is native own-row / free→free.

| Case | ψ (mrad) | ψ/ψ_c | native own-row (whole band) | free→free (whole band) | ratio, whole | ratio, own_line | free→free / CR |
|---|---|---|---|---|---|---|---|
| Si(110), 60 keV top | 0.0 | 0.00 | 0.000e+00 ± 0.0e+00 | 3.754e-02 | 0.00(0) | — | 0.34 |
| Si(110), 60 keV top | 0.8 | 0.52 | 2.194e-01 ± 2.7e-03 | 4.029e-02 | 5.44(7) | 63.3(8) | 0.37 |
| Si(110), 60 keV top | 2.2 | 1.42 | 8.027e-02 ± 1.1e-03 | 4.752e-02 | 1.69(2) | 4.6(1) | 0.77 |
| Si(110), 100 keV top | 4.0 | 2.58 | 3.483e-02 ± 7.3e-04 | 3.955e-02 | 0.88(2) | 2.6(1) | 1.18 |
| C(110), 100 keV top | 0.0 | 0.00 | 0.000e+00 ± 0.0e+00 | 6.062e-02 | 0.00(0) | — | 0.24 |
| C(110), 100 keV top | 1.0 | 0.61 | 4.168e-01 ± 3.3e-03 | 6.616e-02 | 6.30(5) | 61.7(5) | 0.24 |
| C(110), 120 keV top | 3.0 | 1.83 | 1.223e-01 ± 1.5e-03 | 9.555e-02 | 1.28(2) | 3.1(1) | 1.03 |

The whole band runs from 5 keV to the listed top. The CR column is the production `spec_channeling` over the same band. Every band (whole, 10–25 keV, 25–40 keV, `own_line`) with its standard error is in the record.

In every case the other reflections scored exactly zero on the band. Their
lines lie far above it at a 3° tilt.

## Reading

**Within ψ_c the double count is large.** Native zone-row CBS treats a
channeled electron as a free plane wave moving at the beam angle ψ. It puts a
sharp own-row line at `ħcβgψ/(1 − β)`: 12 keV for Si at 0.8 mrad and 23 keV
for diamond at 1.0 mrad, inside the CR band. Si 0.8 mrad gives 0.219 /sr
against a free→free block of 0.040 /sr (5.4×). That alone is 2.0× the whole
CR yield. In its own-line window it is 63× the block. Diamond 1.0 mrad gives
6.3× overall and 62× in window. This is the axial failure, larger.

**Beyond ψ_c the totals approach each other, but the spectrum is wrong.**
The whole-band ratio is 1.69(2) for Si at 2.2 mrad, 1.28(2) for diamond at
3.0 mrad and 0.88(2) for Si at 4.0 mrad. Most of the population is above the
barrier, so the free-wave picture is closer to right for the band total. The
spectral shape is not: the in-window ratio stays at 2.6–4.6. The native kernel puts all strength in
one line at the beam angle, 9% (Si) and 10% (diamond) above the quantum
energy, because the classical kernel omits the transverse recoil term
`−ħ²g²/(2γm)`. The free→free block spreads from 10 keV to the line, because
scattering lowers the transverse momentum of the tracked states. In 25–40 keV
the Si 2.2 mrad ratio is 3.9×. Below that, in 10–25 keV, the native kernel has
almost nothing, while the block has 1.3e-2 /sr.

**At exact alignment both owners omit the block.** At ψ = 0, `v·g = 0` for the
own row, so native zone rows score no line, as on the axial B case. The in-cut
block is real and is scored by nobody. It is 3.75e-2 /sr for Si (34% of CR)
and 6.06e-2 /sr for diamond (24% of CR).

**Size of the in-cut block relative to CR:** 24–37% within ψ_c and at 0, and
77–118% beyond ψ_c (Si 2.2: 77%, diamond 3.0: 103%, Si 4.0: 118%). A declared one-sided bound of
that size, as the axial rule carries, would be a weak statement for planar
results.

## Ownership options (maintainer decision)

1. **Axial rule.** Stop native PXR/CBS on planar zone rows and declare the
   in-cut block as a one-sided bound. This removes the double count. Here the
   other rows are zero in band, so the result equals option 2. The bound is
   24–118% of CR.
2. **Own row only.** Exclude the plane's own reciprocal row on zone rows and
   keep the other rows native. The planar potential holds only the own row,
   so other rows have no density-matrix counterpart and cannot overlap. This
   keeps native emission that nothing else scores. It still needs the bound.
3. **Option 2, plus score the block from the density matrix.** Production
   `spec_channeling` would use the `final_states="tracked"` catalog. The
   in-cut free→free block then has an owner, and only the above-cut part
   stays undeclared. This removes both the double count and the omission, and
   it is the only option that scores anything at ψ = 0. It costs a larger
   catalog (3.2–3.6× the lines at this grid: Si 1777 → 5729, about 3.5×
   the catalog build time, 2.5 s → 8.8 s) and needs `cr-planar-free-free-block`
   fresh-context validated and anchored before it goes into production.

**Recommendation: option 3.** Option 2 is the minimum if option 3 is
deferred. Any option changes default planar output, so it bumps
`CHANNELING_MODEL` and needs a no-double-count test (proposed claim
`cr-planar-zone-row-ownership`).

## Limits

- One thickness (20 µm), one energy (17 MeV), production numerics. The block
  is in cut only. It depends on the 300 eV margin, as capture does
  (`cr-capture-yield`).
- The native standard error is statistical over 20 chunks of 100 electrons.
  The free→free and CR values are expected values on the same paths.
- Dipole emission for free→free on the solver's Bloch basis, observed on
  axis. Transitions below 1 keV forward are dropped (`FREE_FREE_FLOOR_EV`).
- Coherent emission and finite apertures were not measured.
