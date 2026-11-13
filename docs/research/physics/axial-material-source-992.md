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
  this sample size. Supercell and Bloch sectors have no finite set to vary in
  the fibre model; window, cut and kick-harmonic convergence and the
  emitting-subspace closure are in the
  [next section](#emitting-subspace-closure-and-numerical-convergence).
- **Line energies against Watson & Koehler.** Every resolved model peak sits
  10–19% below the Watson & Koehler forward-equivalent energy, with the same
  sign at all three energies and both axes. The Andersen et al. (1982)
  comparison below resolves this: the model matches Andersen's forward-detector
  energies, so the offset is Watson's detector-angle systematic.
- **Physics sensitivity.** Electronic kicks double the retained source relative
  to TDS alone (1.79 → 3.73), and the K-edge-region 1.63 keV line is resolved
  only without kicks. Retained weight at exit is 0.42–0.48; the rest leaves as
  outgoing (40–53%) or unresolved (5–12%) records, whose later emission is
  native-owned or unrepresented.

## Emitting-subspace closure and numerical convergence

**Bloch sectors and supercell.** The finite-grid reference of #885 samples
carriers on a $c\times c$ grid, $\kappa_r=r\mathbf g/c$, so it has a
supercell and a Bloch-sector set to converge. The fibre model has neither. Its
carrier is a continuous classical label. TDS string jumps shift it by a
continuous $\Delta$ over the zone, and the $c\times c$ supercell is the
restriction of $\Delta$ to a grid
([`cr-axial-continuum-tds-jumps`](../../validation/channeling/cr-axial-continuum-tds-jumps.md),
supercell-equivalence check). Electronic kicks shift it by $q$ at uniform
azimuth. One cell with independent strings is therefore the
infinite-supercell limit, not a truncation. What remains numerical is the
order window $M$, the outgoing box $B$, the retained cut, the depth step and
the kick harmonic cutoff.

**Closure.** `emitting_subspace_closure` needs one finite-grid
`GlobalAxialSource` and a single-sector density. It cannot take the fibre
model, whose jumps change carrier: the same reason the dense
`regression_correlation` was replaced. Its question is still answerable
directly. The `reference` arm retains the whole order window ($P=R$, no cut),
so states above the cut keep emitting and can return below it. Arm minus
control is then the source the absorbing cut omits at that window, coherent
cross-cut terms included. This is the photon-yield analogue of the #882
return deficit.

All arms below share seed and 50- or 100-primary chunks, so differences are
paired per chunk ([record](../../validation/check-records/axial-source/closure-992-remote.json);
`checks/axial_fiber_source.py paired`). Jobs 5465–5469 and 5521
(⟨100⟩) and 5865–5871 (⟨111⟩), 2000 primaries per arm. Each value is the arm/control yield
ratio minus one, mean ± SEM, over the whole grid and over the main-line band
(⟨100⟩ 2.8–4.0 keV, the 2p→1s line; ⟨111⟩ 2.5–3.5 keV).

| Change | ⟨100⟩ 3.5 MeV, 0.3 µm, tilt 0.088° | band | ⟨111⟩ 2.99 MeV, 1.1 µm, 0° | band |
|---|---|---|---|---|
| Cut $U_\max+2\Delta U$ (vs $+\Delta U$) | −0.03(19)% | −0.15(27)% | +2.8(11)% | +4.0(18)% |
| $P=R$, $M=9$ (closure) | +0.10(31)% | +0.02(43)% | +1.8(12)% | +2.3(18)% |
| Window $M=11$ | +0.00(5)% | −0.02(6)% | +2.5(10)% | +1.2(15)% |
| $P=R$ at $M=11$ vs $M=11$ (closure) | +0.24(34)% | +0.24(47)% | +0.7(11)% | +0.6(15)% |
| $P=R$, $M=11$ vs control (all changes) | +0.24(33)% | +0.22(46)% | +3.2(12)% | +1.8(16)% |
| Kick harmonics $\lvert n\rvert\le6$ | +0.04(3)% | +0.05(4)% | −0.6(9)% | −1.0(12)% |
| Kick harmonics $\lvert n\rvert\le8$, 48² grid | | | −0.4(8)% | −0.6(12)% |

Retained weight at exit rises with the cut, by construction (⟨111⟩: 0.42
control, 0.52 doubled cut, 0.58 $P=R$, 0.65 $P=R$ at $M=11$). The yield
barely moves, because states above the cut feed the bound block weakly before
they leave. Dropped kick weight $\int\rho_q$ is $1.3\times10^{-3}$ and
$1.8\times10^{-2}$ at the default $\lvert n\rvert\le4$, against
$1.1\times10^{-4}$/$3.7\times10^{-3}$ at 6 and $9\times10^{-4}$ at 8 (48²).

Findings:

- **0.3 µm (Andersen Fig. 11 peak).** Every change is below 0.3% of the yield
  and the 2p→1s area. The absorbing cut omits no measurable source, and the
  Fig. 11 model/data ratio (1.15, ≈1.08 after divergence) is not a
  truncation effect.
- **1.1 µm (Watson main).** The closure, cut and window each move the yield by
  2–3%, at 1.5–2.5σ. All changes together give +3.2(12)%, so the
  absorbing $M=9$ control underestimates the converged source by about 2–5%
  (95%). The arms are not additive within errors (doubled cut exceeds $P=R$
  by 1.0(9)%), as expected from sign-indefinite cross-cut terms. This matches
  the #882 trend: return and window effects grow with depth, staying near 1%
  to about 1 µm and then rising.
- **Kick harmonics** are converged in both cases (≤1%, within 1σ).
- **Material completeness** is outside this table. Emission after a record
  leaves the retained subspace (outgoing, 2–51% of incidents here) or is unresolved (1–8%) is native-owned
  or unrepresented, as stated above.

These bounds are numerical closure at the stated cases, not a depth budget for
admission. For #994, the selected scope should stay at thicknesses of about
1 µm or less, or carry a ≈5% one-sided numerical allowance, until deeper
cases are run at $M\ge11$ with $P=R$.

## Andersen et al. (1982): energies, widths and absolute yield

Andersen, Bonderup, Laegsgaard, Marsh & Sørensen, Nucl. Instrum. Methods 194,
209 (1982) (Zotero `I99KB2XX`) measured Si with a forward detector
(~1e-6 × 4π sr, 280–300 eV resolution). Data in
`tests/data/channeling/andersen-1982-si-axial.json`: Table 2 transcribed;
Figs. 8 and 11 digitized from a 400 dpi render. Remote arms at `41796d0d0` (rebased onto main as `fc4ae9d97`; the incoming main changes touch no channeling, axial or materials code)
(jobs 4945–4962, 2000–4000 primaries) are in the
[record](../../validation/check-records/axial-source/fiber-source-992-remote.json)
under `andersen_1982`; `checks/axial_fiber_andersen.py` does the comparison.
All arms use `absorption=false`, because Andersen's line energies and areas
are absorption-corrected fits. Tilted incidence puts the entrance plane wave
at order $\mathbf n$, with carrier $\mathbf k_\perp-\mathbf n\mathbf g$
folded into the first zone.

**Line energies (Fig. 8; rest frame $E_R=E/2\gamma$, peaks of the 300 eV-smeared
model).**

| Axis, energy, tilt | 2p→1s meas → model (eV) | 3p→1s | 3d→2p | 2s→2p |
|---|---|---|---|---|
| ⟨111⟩ 4 MeV, 0° | 279 → 279.0 (0.0%) | 424 → 422.6 (−0.3%) | not resolved | 80 → 82.5 (+3%) |
| ⟨111⟩ 4 MeV, 0.06° | 279 → 280.6 (+0.6%) | 424 → 413.2 (−2.6%) | 133 → 129.6 (−2.6%) | 80 → 82.7 (+3%) |
| ⟨110⟩ 2.25 MeV, 0.07° | 212 → 203.7 (−3.9%) | 301 → 293.0 (−2.7%) | | |
| ⟨110⟩ 4 MeV, 0.07° | 336 → 323.4 (−3.8%) | 545 → 519.1 (−4.7%) | 175 → 178.3 (+1.9%) | 88 → 87.3 (−0.8%) |
| ⟨100⟩ 2.5 / 3.0 / 3.5 / 4.0 MeV | 157/184/212/234 → −0.4/+1.6/+2.5/+4.2% | 219/251/291/336 → −7.4/−2.3/+0.6/+5.5% | | |

All lines agree within 5%, except ⟨100⟩ 3p→1s at 2.5 MeV (−7%). On ⟨100⟩
the 3p state is barely bound, and Andersen's own curve misses those markers
by a similar amount. The static line catalog (one-string flat levels) gives
⟨111⟩/⟨110⟩ 2p→1s within 1% and ⟨100⟩ within +2–4%, pinned by
`test_silicon_axial_2p1s_matches_andersen_fig8`. The model's ⟨111⟩ 2.99 MeV
2p→1s (2.95 keV) is 12% below Watson's 3.36 keV but within 1–2% of
Andersen's curve at that $\gamma/d$. Watson's energies are high, as
Andersen, Bonderup & Pantell state.

**2p→1s widths (Table 2).** The unsmeared model spectrum is fitted around
2p→1s with Eq. (35): a finite-thickness Lorentzian of lifetime FWHM $\Gamma$
at slab length $l$, plus a linear background.

| Case | $\Gamma_\mathrm{exp}$ | Andersen theory (uncorrelated / total) | Model $\Gamma$ |
|---|---|---|---|
| ⟨111⟩ 4 MeV 0.5 µm, 0° | 950 | 595 / 807 | 813(14) (−14%) |
| ⟨111⟩ 4 MeV 0.5 µm, 0.06° | 950 | 595 / 807 | 724(5) (−24%) |
| ⟨100⟩ 4 MeV 2 µm, 0.1° | 688 | 466 / 588 | 524(5) (−24%) |

The model sits 14–24% below measurement, which carries an uncertainty of
"at least 10%". It lies between Andersen's uncorrelated thermal estimate and
their total. That total includes correlated displacements of neighbouring
string atoms, a factor $1+0.7(C-1)=1.18$–$1.28$ on the thermal part. The
model's independent-string TDS omits these correlations, so a deficit of this
size is expected. Measured widths are therefore reproduced to the stated
physics scope, not beyond it. The correlated-TDS extension is in
[Correlated-string TDS (#1045)](#correlated-string-tds-1045).
Fits at other energies have $\chi^2/\nu$ up to 20, because neighbouring
lines fall inside the window, so those widths are not used.

**Absolute 2p→1s line yield (Fig. 11; ⟨100⟩ 3.5 MeV, 0.3 µm).** Model line
areas (photons/sr/e⁻) are converted to Andersen's counts for 1 µC into
$0.64\times10^{-6}\times4\pi$ sr (1 count = 1.992e-8 sr⁻¹ e⁻¹). The
measured value is the mean of ±θ.

| Tilt (°) | 0 | 0.02 | 0.04 | 0.06 | 0.075 | 0.088 | 0.10 | 0.12 |
|---|---|---|---|---|---|---|---|---|
| Model (counts) | 4050(220) | 6580(160) | 9650(110) | 12470(110) | 14850(100) | 15510(90) | 14250(80) | 11780(80) |
| Measured | 4580 | 5850 | 9030 | 11890 | 13150 | 13500 | 13140 | 11410 |
| Ratio | 0.88 | 1.12 | 1.07 | 1.05 | 1.13 | 1.15 | 1.08 | 1.03 |

On the main lobe the model is 3–15% above the data; Andersen's own
many-beam curve peaks at 15,200, 12% above. Beam divergence (0.02° FWHM
collimation plus a similar magnet term) lowers the model peak by about 5%,
which leaves model/data ≈ 1.08 at the maximum. Data asymmetry about 0°
(peaks at −0.095° and +0.082°) and the unstated detector efficiency are
systematics of the same order. At exact alignment the model gives a nonzero
2p→1s yield (0.88 × measured, before divergence) from scattering into 2p
alone, because only s states are populated at entry. A 45° tilt azimuth
changes the 0.088° area by −2%. At 0.15° the free→2p line crosses the window
(Andersen fixed its ratio for this reason), and the single-line fit fails
(χ²/ν = 30). That point is not compared.

## Correlated-string TDS (#1045)

Claim `cr-axial-correlated-tds` adds the transverse displacement correlations
of neighbouring atoms in one string as extra continuum-TDS shells. The Si
input is Nielsen & Weber (1980), adiabatic bond-charge model at 300 K
(Zotero `PQ7CVU28`):

- (111) $c=0.240$, (400) 0.071 and (800) 0.028, digitized from Figs. 3(b) and 4(b);
- other string partners from the fitted envelope $c=0.381\,r^{-1.186}$.

Over the six nearest string atoms this gives Andersen's Table 2 $C$: 1.238
for ⟨100⟩ (Andersen: 1.25) and 1.420 for ⟨111⟩ (Andersen: 1.40). The
research arms keep the ten nearest partners, about $1/\alpha Z$ (`c10`).
The six-partner set (`c6`) is a range sensitivity check.

**Run setup.** Remote arms at `c29a7339f`, jobs 11467–11485, are in the
[record](../../validation/check-records/axial-source/correlated-tds-1045-remote.json).
They use the #992 Andersen settings plus `correlated_partners`, the same
seed and incident IDs, and `absorption=false`. The uncorrelated control
(`c0`) reproduces the #992 spectra to that record's 5-significant-figure
rounding. Ratios are paired c/c0 with a delete-group jackknife SE (10
groups of 100-primary chunks).

**2p→1s widths (Table 2), eV.**

| Case | $\Gamma_\mathrm{exp}$ | Andersen total | c0 | c6 | c10 | c10/c0 |
|---|---|---|---|---|---|---|
| ⟨100⟩ 4 MeV 2 µm, 0.1° | 688 | 588 | 524(5) | 567(6) | 570(7) (−17%) | 1.089(8) |
| ⟨111⟩ 4 MeV 0.5 µm, 0° | 950 | 807 | 813(14) | 914(17) | 931(18) (−2%) | 1.146(22) |
| ⟨111⟩ 4 MeV 0.5 µm, 0.06° | 950 | 807 | 724(5) | 829(7) | 845(7) (−11%) | 1.167(9) |

The correlations widen the line by 9% on ⟨100⟩ and 15–17% on ⟨111⟩.
Going from six to ten partners adds 1–2%. The ⟨111⟩ widths now lie within
2–11% of measurement, inside its stated "at least 10%" uncertainty. ⟨100⟩
2 µm stays 17% low. The gain is about what the 1s + 2p loss rates predict:
+11–14% (⟨100⟩) and +22–25% (⟨111⟩), diluted by the electronic and
thickness parts of the fitted width. Andersen's factor $1+0.7(C-1)$ is
17% and 29% on the thermal part.

**Absolute 2p→1s yield (Fig. 11; ⟨100⟩ 3.5 MeV 0.3 µm), counts.**

| Tilt (°) | 0.06 | 0.075 | 0.088 | 0.088 (az 45°) | 0.10 |
|---|---|---|---|---|---|
| Measured | 11890 | 13150 | 13500 | 13500 | 13140 |
| c0 | 12470(110) | 14850(90) | 15510(90) | 15230(90) | 14250(80) |
| c10 | 11300(110) | 14060(110) | 14740(100) | 14510(110) | 13440(90) |
| c10 / measured | 0.95 | 1.07 | 1.09 | 1.08 | 1.02 |
| c10/c0 area | 0.907(10) | 0.947(12) | 0.950(9) | 0.953(7) | 0.943(8) |
| c10 width (eV) | 564(6) | 597(5) | 617(4) | 597(5) | 597(4) |

Stronger incoherent scattering lowers the main-lobe line area by 5–9%.
Model/data moves from 1.05–1.15 to 0.95–1.09. With the ~5% beam-divergence
reduction estimated above, the remaining excess is ≤4%. Agreement is
research evidence. It does not change the claim's status, and the
production `AxialFibreCase` path and default output are unchanged.

**Remaining gap.**

- ⟨100⟩ 2 µm is the only Table 2 case still well below measurement.
- The sudden-approximation truncation stands in for longitudinal dephasing.
- Inter-string correlations are omitted.
- The Si input holds at 300 K only.

## Off-axis observation and polarization

`fiber_angular_source` (claim `cr-axial-fiber-angular`) scores the same
two-time source toward a direction $(\theta,\phi)$, with polarization basis
$(\hat\theta,\hat\phi)$. It uses the Liénard–Wiechert angular factor
evaluated at each line's resonance,
$A_\theta=\hat n_\perp\cdot\mathbf J(\cos\theta-\beta)/(1-\beta\cos\theta)$
and $A_\phi=\hat\phi\cdot\mathbf J$, together with the Doppler phase
$E(1-\beta\cos\theta)z/\hbar v$.

An earlier velocity/recoil form evaluated the photon-recoil term at the photon
energy. It kept the crystal-face boundary terms, which produced a continuum
growing in proportion to $E$, and it made the estimator heavy-tailed. It is
rejected; the [derivation](../../validation/channeling/cr-axial-fiber-angular.md)
records the reasons.

Three Si⟨100⟩ arms were run with a depth step of 7.5 Å, $\theta\gamma\le2$ and
16 azimuths (jobs 5405–5407, commit 819567a16; [check record](../../validation/check-records/axial-source/fiber-angular-992-remote.json)).
Intensities are integrated over the photon grid only.

| Arm | θγ | Line peak (eV) | $I/I(0)$ | cos 2φ contrast | Linear degree | $V/I$ |
|---|---|---|---|---|---|---|
| 3.5 MeV, 0.3 µm, tilt 0.088° | 0 | 3400 | 1 | 0 | 0.64 | −0.0006 |
| | 0.5 | 2720 | 0.43 | −0.30 | 0.70 | −0.0008 |
| | 1 | 1720 | 0.10 | −0.74 | 1.000 | 0.0000 |
| | 1.5 | 1080 | 0.042 | −0.59 | 0.87 | 0.0001 |
| 3.5 MeV, 0.3 µm, tilt 0 | 1 | 1720 | 0.092 | −0.0017 | 1.000 | 0.0000 |
| | 2 | — | 0.0099 | +0.0010 | 0.47 | 0.0010 |
| 10 MeV, 0.2 µm, tilt 0.046° (≈ψc/5) | 0 | 6600 | 1 | 0 | 0.36 | −0.0033 |
| | 1 | 3400 | 0.13 | −0.36 | 1.000 | 0.0000 |
| | 2 | — | 0.020 | −0.17 | 0.53 | 0.0035 |

How to read these numbers:

- **Doppler shift.** The 3.5 MeV 2p→1s peak follows $E(0)/(1+\gamma^2\theta^2)$: 3.40, 2.72, 1.70 and 1.05 keV are predicted, 3.40, 2.72, 1.72 and 1.08 keV are found.
- **Intensity at θ = 1/γ.** A transverse dipole line, averaged over azimuth, gives 0.125 of its forward value there. The grid integral is lower because off-axis lines move below the 0.8 keV grid edge.
- **Symmetry.** At exact alignment the map is azimuthally uniform within 0.3%, as $C_4$ symmetry requires.
- **Tilted beam.** Tilting the beam along $x$ aligns the 2p population along the tilt. That produces the expected null at $\phi=0$ when $\theta'=90°$, i.e. at $\theta\gamma=1$.
- **Polarization.** It is fully linear at $\theta\gamma=1$, where $\cos\theta'=0$ and only the $\hat\phi$ component survives. Circular polarization stays below 0.4% everywhere.

**Comparison with Korotchenko, Kunashenko & Dabagov (2022).** They give
theoretical predictions for Si⟨100⟩ at 10 MeV and $\vartheta_c/5$, with no
measured data (Eur. Phys. J. C 82, 196, Zotero `NH9SX2CA`).

- **Linear-only polarization:** reproduced.
- **Four-fold azimuthal peaks:** cannot be tested here. They come from a $\langle xy\rangle_{if}$ term, first order in the photon's transverse momentum times a quadrupole moment. At the transverse dipole order implemented here, a $C_4$ ensemble is azimuthally uniform, and the tilt adds only $\cos2\phi$. Testing their four-fold peaks needs the next order in $\mathbf q\cdot\mathbf r$, or exact recoil (a companion in fibre $\kappa-\mathbf q$).
- **10 MeV spectrum shape:** ours is a broad continuum to about 24 keV, peaking at 6.6 keV, with a 20–22 keV shoulder. That shoulder is consistent with extrapolating Andersen's Fig. 8 2p→1s trend to $\gamma/d\approx3.8$. Theirs is smooth to 33 keV with its maximum at 26.8 keV. Different potentials and matrix elements make that a qualitative comparison only.

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

- Andersen, Bonderup, Laegsgaard, Marsh & Sørensen, Nucl. Instrum. Methods
  194, 209 (1982) (Zotero `I99KB2XX`; the review's ref. 28): line energies,
  widths and absolute line yields, compared in the section above.

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
| Zone row, tilted incidence | line above the native resonance-root threshold; off for admitted fibre cases (#994) | not scored | supplied natively before #994; omitted and bounded for admitted fibre cases |
| Handed-off electron (left $P(\kappa)$) | tilted `on_shell_direction` row: line scored | emission after a record leaves the retained subspace is not represented | supplied natively, no overlap |
| Nonzero axial-order $\mathbf g$, incoherent brems | native | not scored | n/a (stays native, ADR-0018) |

Exact alignment grants no exemption. On the exact axis a production adapter
must either score the in-cut free→free block axially (`boundary="none"` on
zone rows) or accept a ~20% omission of the retained source. With tilted
incidence it must keep `boundary="max"`, or the block is counted twice.
Free→free above the retained cut is outside both estimators on the exact axis;
its size is not bounded here. This is input to #994 admission, not a
production rule.

**Si ⟨100⟩ 3.5 MeV, 0.3 µm (#994).** The in-cut block was measured for the
two #993 cases with a `boundary="none"` arm, chunk-paired with each #993
control (qlmc cpu at 07c0f5f3e, jobs 10497 and 10498, 20,000 primaries
each). Record: `docs/validation/check-records/axial-source/free-free-994.json`.

| Case | 800 eV–10 keV | 2.8–4 keV (2p→1s) |
|---|---|---|
| A, tilt 0.088° | +7.83(8)% | +3.63(4)% |
| B, 0° | +18.19(23)% | +10.95(16)% |

The maintainer's 2026-11-09 decisions on #994 treat A and B alike. Native
PXR/CBS line scoring is off on axial zone rows (`cr-axial-zone-row-native-lines`),
and each case omits the in-cut block and carries it as a declared one-sided
bound: 8.06% for A and 18.9% for B (mean + 3 SE). Above-cut free→free stays
unbounded for both. An earlier decision had A take the block from native CBS on
its tilted zone rows; the comparison below showed that native CBS does not
reproduce it, and that decision was superseded.

## CPU source cost and the GPU question (#1154)

The admitted A/B source is substantially cheaper than the older, thicker
research arms quoted above. On a Ryzen 9 9950X3D CPU node, with one BLAS/OpenMP
thread per worker, the measured source work is 0.112 s/history for A and
0.104 s/history for B. These are 32-history batches in four serial chunks of
eight, seed 1154, using the admitted physical inputs and numerical settings
through `axial_fiber_inputs`. The explicit photon grid has 185 points over
0.8–10 keV, with exit absorption enabled. This is source timing, excluding
native continuation and the rest of a production campaign.

| Case | Source work (s/history) | Warm batch wall (s) | Histories/wall-s | Peak worker RSS (MiB) |
|---|---:|---:|---:|---:|
| A, 0.088° | 0.112 | 4.390 | 7.29 | 340 |
| B, aligned | 0.104 | 4.110 | 7.79 | 361 |

Wall time includes pool startup, input construction, result hashing and return.
The first repeat is retained separately as cold; the table uses the median
of the next two identical-workload repeats. Peak RSS includes imports and
setup in each worker; it is not the simultaneous process-tree peak.
The CPU matrix runs serially on one node to avoid contention between arms.
The source itself is unchanged: profiling checks bitwise equality of the
spectra, spectral errors, complex correlations, retained traces and terminal
events. This engineering check changes no validation status or admission.

The entrance Hamiltonian has 361 order coefficients, 61 retained levels and
5 bound levels for both cases, with 672 electronic kick nodes and 201 recorded
depths. A fixed-depth batch could share channel tables and scoring geometry.
Jump depths, terminal fates and post-jump carrier sectors differ between
histories, so their carrier eigensystems and companion states cannot all share
one propagator. The short admitted workload must therefore be measured
separately from the thicker research cases before choosing a GPU port.

Reproduction: `checks/axial_source_performance.sbatch`, submitted with
`--array=0-23%1` from a clean, isolated checkout synced through `pyrite remote`.
The [CPU evidence record](../../validation/check-records/axial-source/source-performance-1154-cpu.json)
preserves commands, workload settings, source revision, cold/warm measurements,
result digests and profile timings. Published operator paths and node names
are placeholders; numerical measurements are unchanged.

### Where the time goes

Separate 32-history, single-call profiles give the following fractions of
`fiber_on_axis_source` time. The rows are disjoint; the jump row includes
constructing the new carrier's eigenbasis. Both full profiles reproduce their
unprofiled result digests exactly.

| Phase | A | B |
|---|---:|---:|
| Channel-engine construction | 9.5% | 10.5% |
| No-jump search, companion propagation and depth recording | 14.8% | 13.7% |
| TDS/electronic jump selection and new carrier basis | 45.4% | 43.2% |
| Companion jump maps | 8.4% | 8.7% |
| Weyl outgoing measurement | 0.08% | 0.18% |
| Line-grid scoring | 20.2% | 22.0% |
| Other source work | 1.7% | 1.8% |

The primary waiting-time search alone is about 0.2%. These admitted cases
use continuum-string TDS; they have no nonzero discrete lattice-Kraus-column
draw. Accelerating a fixed no-jump step alone would miss the main cost.
Line scoring already spends most of its time in NumPy contractions; adding
Numba to the small waiting/draw control loops cannot remove the carrier
eigensolves or these contractions.

### Numerical and process scaling

Each numerical probe changes one setting, with the same seed and incident IDs.
The following ratios use source work per history, excluding input construction;
they measure cost, not numerical adequacy of the overridden settings.

| Probe against admitted numerics | A cost ratio | B cost ratio |
|---|---:|---:|
| Order window M: 9 → 11 | 1.76 | 1.74 |
| Outgoing box B: 20 → 24 | 1.04 | 1.04 |
| Recorded-depth spacing: 15 → 7.5 Å | 1.73 | 1.76 |
| Recorded-depth spacing: 15 → 30 Å | 0.78 | 0.78 |
| Kick harmonic cutoff: 4 → 2 | 1.00 | 1.03 |
| Kick harmonic cutoff: 4 → 6 | 1.04 | 1.05 |

A separate matched Si ⟨111⟩, 2.99 MeV, aligned pair changes thickness
from 0.3 to 1.1 µm, keeping the other numerical controls fixed. Source cost
rises from 0.185–0.186 to 0.641–0.644 s/history, about 3.46×; peak worker
RSS rises from 359–368 to 807 MiB. The two repeats of this physical pair
come from identical overrides of the A and B input records; they are neither
additional admitted cases nor independent random ensembles.

For B, 64 histories with chunks of eight give the following process scaling:

| Workers | Warm wall (s) | Histories/wall-s | Speedup |
|---:|---:|---:|---:|
| 1 | 8.163 | 7.84 | 1.00× |
| 2 | 4.307 | 14.86 | 1.90× |
| 4 | 2.613 | 24.50 | 3.12× |
| 8 | 2.124 | 30.13 | 3.84× |

At four workers, chunks of two give 10.88 histories/s; chunks of sixteen
give 29.47/s. Larger chunks amortize input/channel construction. The existing
production adapter already calls the source over a whole case; this measured
parallelism is the research source-chunk path, not an implemented intra-case
production pool. The late thickness replicate, worker scaling, bound-population
arms and full profiles shared the node with another eight-CPU allocation.
These timings describe that shared-node workload, not exclusive-node capacity;
the close timing repeats do not measure uncertainty across seeds or node loads.

### GPU assessment and CPU-first decision

Retain the CPU production source and prioritize CPU work before a GPU port:
reuse immutable inputs across research chunks, avoid very small chunks, and
investigate a shared/precomputed scoring contraction and controlled history
parallelism. No optimization is implemented by these measurements. Any change
to production reduction order needs its own numerical-identity checks.

A partial GPU implementation that makes no-jump/companion propagation,
companion jumps and line scoring free, while leaving channel construction,
carrier solves and jump selection on one CPU controller, has an optimistic
B source-only ceiling of about 23 histories/s. This Amdahl estimate uses the
measured 44.4% accelerated share and the unprofiled 32-history single-call
rate; it excludes transfers, launch costs and input construction. It is already
below the measured 30 histories/s process-chunk throughput. A competitive
GPU port would need batched carrier eigensolves and jump maps as well, rather
than just a CuPy no-jump loop. Common channels/depths can be shared, but generic
post-jump sectors, waiting times and terminal fates still require independent
per-history state and stream handling. No achieved GPU fibre throughput is
claimed. A twofold improvement over the measured CPU batch would require the
entire 64-history GPU request to finish in roughly 1.06 s.

### Matched population observable

The comparison observable is incident-normalized weight in the instantaneous
Hamiltonian block below the potential maximum, at 0, 500, …, 3000 Å, and
its trapezoidal depth exposure. This is not a photon-yield comparison.
Three CPU fibre ensembles of 512 histories (seeds 1154–1156, four workers)
give final bound weight 0.5398 and exposure 2103(25) Å, where the parenthesized
error is the SEM across three seed means. Their wall times are 9.00–9.54 s
per ensemble, including startup/setup. The full per-seed/depth data are in the
evidence record; three seeds give limited precision for interval estimates.

The finite-grid arm starts from the full plane wave; the fibre starts from
its retained projection with the incident entry weight. The fibre absorbs
outgoing cut/window outcomes; a finite periodic supercell permits return flux
and discretizes transverse transfer/Bloch sectors, with its own out-of-band
sink. Thus equal observable definitions do not make their generators equal.
Other differences remain: the fibre truncates electronic density harmonics at
four on a 32-point density grid, while the real-space model uses its represented
density harmonics; continuum-string TDS and projected waiting-time evolution
also differ from discrete-supercell transfers and full-grid Strang/fixed-step
MCWF. A discrepancy cannot be assigned solely to return flux.

The completed RTX 5080 study uses 32 histories per seed, seeds 1154–1156,
and varies points per cell, supercell size and propagation step. All five
local refinement comparisons meet the predeclared criterion: an exposure
change below 5% and compatible with zero within twice the paired seed-mean
SEM. These are local refinement checks, not a rigorous numerical error bound
or a full joint Bloch-sector/grid-limit extrapolation.

| Refinement | Exposure change | Paired-difference SEM (Å) |
|---|---:|---:|
| 1×1 cells, 24 → 32 points/cell | +0.17% | 2.99 |
| 2×2 cells, 24 → 32 points/cell | +0.18% | 3.12 |
| 1×1 → 2×2 cells, 24 points/cell | −0.10% | 2.47 |
| 2×2 → 3×3 cells, 24 points/cell | −0.06% | 1.44 |
| 2×2 cells, step 5 → 2.5 Å | +0.02% | 0.31 |

Using 2×2 cells, 32 points/cell and a 5 Å propagation step as the grid
reference gives:

| Model | Total histories | Bound weight at 3000 Å | Bound exposure (Å) | Request wall per seed (s) |
|---|---:|---:|---:|---:|
| Fibre, CPU, four workers | 1536 | 0.540 ± 0.008 | 2103 ± 25 | 9.00–9.54 (512 histories) |
| Grid, RTX 5080 | 96 | 0.583 ± 0.067 | 2199 ± 141 | 121.85–156.27 (32 histories) |

Errors are SEMs across three seed means. The grid-minus-fibre exposure is
96 ± 117 Å using seed-paired differences, a central shift of +4.6% that is
unresolved at this precision. Matching integer seeds between unlike
unravellings does not make their trajectories shared. These results establish
neither statistical equivalence nor a photon-source error budget.

The grid request includes 120.79–155.19 s of CPU full-spectral projection and
host/device transfer; it is dominated by the diagnostic observable. Subtracting
that timer leaves about 1.1 s, including input/solver construction and device
work; this subtraction is not a kernel benchmark. Request timers exclude
driver imports and initial CUDA-context creation; the fibre timer includes its
internal worker-pool startup. Kernel caches may already be warm. Unequal
ensemble sizes, observable costs and cache conditions prevent interpreting
these arm costs as a source speedup comparison.

The [paired evidence record](../../validation/check-records/axial-source/grid-fiber-comparison-1154.json)
preserves all 18 CUDA runs, CPU ensembles, per-seed/depth observables, uncertainty
calculations, refinement results and the exact single-process launch script.
The initial forkserver launch failed before propagation; direct single-process
CUDA execution passed and produced this study. The CPU-first decision remains:
grid MCWF is diagnostic and needs independent photon-source/depth-budget
evidence before any production admission. No GPU gate, model marker, claim
status or admitted envelope changes here.
