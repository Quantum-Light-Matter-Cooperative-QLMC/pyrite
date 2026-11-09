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
physics scope, not beyond it. Correlated TDS would be a model extension.
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

**Si ⟨100⟩ 3.5 MeV, 0.3 µm (#994).** The in-cut block was measured for the
two #993 cases with a `boundary="none"` arm, chunk-paired with each #993
control (qlmc cpu at 07c0f5f3e, jobs 10497 and 10498, 20,000 primaries
each). Record: `docs/validation/check-records/axial-source/free-free-994.json`.

| Case | 800 eV–10 keV | 2.8–4 keV (2p→1s) |
|---|---|---|
| A, tilt 0.088° | +7.83(8)% | +3.63(4)% |
| B, 0° | +18.19(23)% | +10.95(16)% |

The maintainer's 2026-11-09 decisions on #994: A takes the block from native
CBS on its tilted zone rows. B omits it and carries a declared one-sided
bound of 18.9% (mean + 3 SE). Above-cut free→free stays unbounded for B.
