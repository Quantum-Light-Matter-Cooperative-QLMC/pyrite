# issue-91-relaxation-cascade

Full radiative + nonradiative atomic-relaxation cascade, replacing the
direct-vacancy fluorescence estimator.
GitHub issue: #91. Branch/worktree: `issue-91-relaxation-cascade`
(`.worktrees/issue-91-relaxation-cascade`).

Baseline under review: `src/pyrite/montecarlo/spectrum/characteristic.py`
(`lorentzian-v4`, ledger row `characteristic-radiation`, status `filtered`).

## Implementation-context review, 2026-09-21

Scope of the review: are the EEDL subshell ionization cross sections — K
relative to the other shells — correct, and what does the cascade need?

### Cross sections: no defect found

Nothing shell-specific in the code; K is read, scaled, and joined exactly like
every other subshell, which is right for a direct-vacancy estimator.

- Designator map `characteristic.py:267` (`mt - 533`) matches the ENDF subshell
  designator list: 534=K, 535-537=L1-L3, ..., 572=Q3. Non-contiguous MT sets are
  handled because the loader iterates present MTs, not a range — Au ships
  `..., 553, 554, 559` (5d then 6s, no occupied 6p).
- Barn to cm^2 via 1e-24 (`:270`); interpolation law 2 enforced; zero at/below
  `EPE` and outside the tabulated range.
- **Independent internal check (new).** Sum of MF=23/MT=534-572 against
  MT=522 (total electroionization) gives ratio `1.000000` at T = 1 keV, 10 keV,
  100 keV, 1 MeV for Z = 6, 29, 79. Confirms the designator map, barn units, and
  that no subshell is missing or double counted. Not currently a test — see
  checklist.
- **Magnitude check.** Against Bethe
  `sigma_i = pi e^4 b n_i ln(c T / E_i) / (T E_i)`, `b=0.9`, `c=0.65`,
  `pi e^4 = 6.5138e-14 eV^2 cm^2`: EEDL/Bethe = 1.041 (Cu K, 30 keV), 1.007
  (Si K, 30 keV), 0.990/0.980 (Cu L3/L2, 30 keV), 0.907/1.073 (Au L3/M5,
  100 keV). L1 runs 0.58-0.78 and M 0.20-0.96, consistent with generic `b`,`c`
  not being tuned per subshell. K is not anomalously scaled and the shell
  ordering follows `n_i / (T E_i) * ln(c T / E_i)`.
- Cu at 30 keV: `sigma_K = 350.7 b`, `sigma_L(tot) = 4.405e4 b`, K/L = 0.8% —
  the ratio the `E_i^-1` scaling demands, not an error.
- xraydb conditional intensities sum to `1.000000` per initial level (C, Si, Cu,
  Mo, Ta, Au, U; worst case 0.99707 for M4), so `omega_i * I_il` is the correct
  product and the 0.99-1.01 guard at `:403` is load-bearing.

### Defects, all on the relaxation side

1. **Coster-Kronig omitted, data already installed.** Elam `omega_i` are Krause
   pure subshell radiative yields (Au 0.107/0.334/0.320 reproduces Krause 1979
   exactly), so CK transfer is by construction absent. `xraydb.ck_probability`
   ships CK and is unused. Effect on total L emission `sum_i sigma_i omega_i`:
   Cu 30 keV x1.235, Cu 15 keV x1.232, Mo 60 keV x1.114, Au 100 keV x1.079,
   Au 30 keV x1.064, Ta 30 keV x1.033. Subshell shares move much further than
   totals: Cu L1-origin photon share 0.031 -> 0.0005, i.e. Lbeta3,4/Lalpha wrong
   by more than 10x.
   L-shell CK sums are <= 1 for every Z in 3-98, so they are usable as
   probabilities. **M-shell CK from xraydb is not** — `sum_finals` reaches 3.82
   (Cr M1) with 134 violating (Z, initial) pairs. Gate M CK off; it needs EADL.
2. **No Auger-fed daughter vacancies — and the K feed is small.** At ~1.7 L
   vacancies per K vacancy, K-fed L vacancies are 0.66% (Cu 15 keV), 1.35%
   (Cu 30 keV), 1.85% (Mo 60 keV), 0.60% (Au 100 keV) of direct L vacancies,
   because `sigma_L >> sigma_K` whenever K is open. The cascade deficit that
   matters is M/N population fed by L Auger decay, which cannot be quantified
   without EADL.
3. **Elam has no M/N line lists for most elements**, so the `:384` warning path
   fires and those photons are zero: Au M1, M2, N1-N7 (`omega_M2 = 0.0423`
   entirely lost), Cu M1/M2, Mo M1-M4/N1. M spectra stay structurally incomplete
   even with a correct cascade.
4. The affected lines dominate production: Cu at 30 keV has K = 30.4% and
   non-K = 69.6% of `sum sigma omega I`. Self-absorption removes most of the
   soft branch from the escaped spectrum, but items 1 and 3 are not fringe.
5. `omega_i` is spread across only the Elam-tabulated lines. That conserves the
   total radiative yield but over-assigns intensity to tabulated lines wherever
   the table omits weak ones. Nowhere stated.
6. No secondary fluorescence: `exp(-tau)` deletes the photon with no
   re-emission. Weak for Kalpha inside its own element (below its own K edge),
   not weak for alloys or multilayers.
7. Auger electrons are neither produced nor transported; cascade energy balance
   is not closed.

## Implementation path

### Data

**Required beyond xraydb: EADL2025, MF=28/MT=533** (EPICS2025 `EADL2025.ALL`).
Same IAEA NDS / LLNL distribution and licence as the packaged EEDL, so the
existing checksum-pin plus `.gitattributes -text` path applies verbatim.
`characteristic.py` already *detects* the file (the MF=28/MT=533 error branch in
`_load_eedl_subshell_tables`), and endf-parserpy 0.17.0 parses it.

Per subshell EADL gives binding energy, electron occupancy, and the full
transition list: radiative `(SUBJ, ETR, FTR)` and nonradiative
`(SUBJ, SUBK, ETR, FTR)`, with `sum FTR = 1` per subshell — the directly
testable invariant for the probability-conservation acceptance bullet.

Keep from xraydb: line energies (Elam/experimental), core-hole widths
(Krause-Oliver, Keski-Rahkonen-Krause), Elam `omega_i` for optional
renormalization, L-shell `ck_probability`.

Neither source covers: experimental M/N line energies and intensities at high Z
(Elam sparse, EADL transition energies are binding-energy differences with tens
of eV error and no multiplet splitting) — Bearden/Deslattes or Campbell M
compilations if M spectroscopy becomes first-class; and multiple-vacancy
satellite structure, which has no portable table and stays excluded.

For the "independent cascade implementation" acceptance bullet, use PENELOPE
`pdrelax`, Geant4 G4EMLOW `fluor/`, or EGSnrc as *reference spectra only*, never
as packaged data sources — provenance and redistribution are cleaner via IAEA
EADL.

### Algorithm: deterministic, linear, precomputed per element

Zero per-segment cost; all work lands in table construction.

- Order shells by **binding energy**, not designator index. The 4f/5s and
  N6,7/O1 orderings invert for some Z.
- Build `D[i,j]` = expected daughter vacancies in `j` per one decay of a vacancy
  in `i`: a radiative `i->j` adds 1 to `j`; a nonradiative `i->(j,k)` adds 1 to
  each of `j`, `k`. Decay always fills the hole from a less-bound shell, so **D
  is strictly lower-triangular** and `n = (I - D^T)^-1 N` is exact via a finite
  Neumann series — no iteration cutoff, no convergence tolerance. The relaxation
  cutoff is applied by dropping rows below the binding-energy floor.
- `R[i,l]` = radiative `FTR`; line yield = `sum_i n_i R[i,l]`, replacing
  `omega_i * I_il` in `_xraydb_line_yields` (`:680`).
- **The hot loop needs no change.** `line_yield_per_vacancy` is already
  `(n_shell, n_line)` (`:117`) and `shell_sigma_cm2 @ response` (`:891`) simply
  goes dense.

Energies and widths: keep xraydb line energies and the Krause-Oliver /
Keski-Rahkonen-Krause widths wherever an EADL transition matches an xraydb
`(initial, final)` level pair; fall back to EADL `ETR` otherwise, and record
per-line provenance — mixing the two shifts peak positions inconsistently.

Yields: EADL `FTR` are self-consistent and probability-conserving, but its L/M
radiative totals differ from Krause's experimental `omega_i` by roughly 10-20%.
Recommend EADL topology and branching with an *optional* per-subshell rescale to
`sum_l R[i,l] = omega_i^Elam`; make it a flag, document it as an approximation,
and fold the choice into the model marker.

Deterministic vs stochastic: deterministic expectation propagation for spectrum
scoring. It matches the existing track-length estimator, adds no variance, and
is *exact* here because the system is linear and triangular. Stochastic sampling
is only required if Auger electrons are coupled back into transport — a separate
issue; declare untransported Auger electrons a validity limit.

### Interim slice, no new data

Apply L-shell CK from xraydb before EADL lands:

```
n_L1 = N_L1 * (1 - f12 - f13)
n_L2 = N_L2 * (1 - f23) + f12 * N_L1
n_L3 = N_L3 + f23 * N_L2 + f13 * N_L1
```

Trap: `xraydb.ck_probability` defaults to `total=True`, so `f13` already
contains the L1->L2->L3 route. Feed it from the **primary** `N_L1` and do not
also route `f12 * N_L1` through `f23`, or that path is double counted. Use
`total=False` for direct values with sequential application. Recovers the
1.03-1.24 factors above under its own model marker. Independently ownable —
split into its own issue if it is dispatched separately from the EADL work.

### Contract changes

- `relaxation_cutoff_eV` currently filters **emitted line energy**. A cascade
  needs it to bound **vacancy propagation** by binding energy. Different
  parameter meaning, new model marker, checkpoint identity fork.
- Model marker moves off `lorentzian-v4`; the EADL checksum and the
  yield-renormalization flag both belong in it, alongside the existing EEDL and
  xraydb version components.
- 1 keV transport floor: a cascade emits M/N lines well below 1 keV from parents
  ionized above it, partially compensating the direct sub-keV production the
  floor omits. That must not be written up as validating sub-keV transport.

## Checklist

- [x] Add the MT=522 sum rule (`sum MT 534-572 / MT 522 == 1`, Z = 6/29/79,
      1 keV-1 MeV) and a Bethe-ratio band as independent internal-consistency
      tests. `tests/montecarlo/test_characteristic.py:71` currently pins carbon
      `sigma_K` to `rtol=1e-13` against the implementation — a regression pin,
      not independent evidence, and it cannot catch a designator-map or unit
      error.
- [x] Interim L-shell CK redistribution from `xraydb.ck_probability`, M-shell CK
      gated off with the sum-violation evidence recorded.
- [x] Package and pin EADL2025 (`EADL2025.ALL`, MF=28/MT=533) next to
      `EEDL.endf`; resolve provenance/redistribution note in
      `src/pyrite/data/characteristic_cross_sections/README.md`.
- [x] EADL MF=28 parser + validation (`sum FTR = 1` per subshell, daughter
      designators in range, ETR positive, occupancy consistent).
- [x] Binding-energy-ordered vacancy-transfer matrix, `(I - D^T)^-1` Neumann
      solve, cutoff by binding energy.
- [x] Radiative emission matrix; xraydb energy/width join by `(initial, final)`
      level pair with per-line provenance recorded.
- [x] Optional `omega_i^Elam` renormalization flag; model marker updated;
      checkpoint identity fork documented.
- [x] `relaxation_cutoff_eV` contract migration.
- [x] Energy accounting: cascade energy balance per primary vacancy within the
      declared approximation.
- [ ] Reference-spectrum comparison (PENELOPE/Geant4/EGSnrc) for K, L and M
      primary vacancies on at least one low-Z, one mid-Z and one high-Z element.
- [x] Physics docs + validation write-up + ledger row updated; untransported
      Auger electrons, absent secondary fluorescence, missing experimental M/N
      line data, and multiple-vacancy effects recorded as explicit validity
      limits.
- [x] Items 5 and 6 of the review (yield spread over tabulated lines only;
      no secondary fluorescence) written into the scope section of
      `docs/physics/radiation-physics/characteristic-radiation.md` — they are
      currently unstated approximations, and they are the only review findings
      the existing docs do not already disclose.

## Out of scope here

- Auger-electron transport and the coupled source term (needs the stochastic
  route; own issue).
- Secondary fluorescence from reabsorbed characteristic photons (own issue; the
  present `exp(-tau)` is a pure sink).
- Multiple-vacancy shifts, satellites, chemical shifts.

## Progress, 2026-09-27

Landed on the branch: `montecarlo/eadl_relaxation.py` (checksummed MF=28
loader/validator, `vacancy_cascade`, `relaxation_energy_budget`) and
`characteristic.py` rewired to it (model `...-eadl-cascade-eadl-yields-...-v7`,
`fluorescence_yields="eadl"|"elam"`, binding-energy `relaxation_cutoff_eV`).
The interim xraydb L-shell CK transfer (v5/v6, landed on main earlier) is
removed; EADL supplies all CK. Evidence and numbers:
`docs/validation/radiation-physics/characteristic-radiation.md`, 2026-09-27
note.

Remaining:

- Reference-spectrum comparison (PENELOPE `pdrelax` / Geant4 `fluor/` /
  EGSnrc) for K, L, M primaries, low/mid/high Z. Needs external reference
  data; not packaged.
- Fresh-context `physics-validation` of the cascade and MF=28 field mapping.
- Open physics question: EADL vs Krause CK (Cu f23 0.009 vs 0.47) and
  omega (L subshells 0.47-2.0x). Needs measured L-line ratios to settle.
- Cost: line count 24 -> 95 for W; characteristic scoring ~2.5x on CPU and
  proportionally more line-grid seeds. Consider a yield floor for seeding only.
- Pre-existing, unrelated failures on main: `test_high_energy_profile_range_...`,
  `test_packaged_profiles_have_explicit_membership`, 1200-line budget for
  `transport/api.py`.
