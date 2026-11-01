# MeV full-track transport against Geant4 TestEm5: independent verification

Ledger id: `mev-full-track-transport` (issue #439). The claim is a
measurement, not a formula, so this audit checks the definitions of the
observables, the estimators, the gates, the committed numbers and the
argument for the two excluded failures. Verifier: fresh context, 2026-10-23,
worktree `issue-439-mev-electrons` at `6131231a`.

## 1. What the claim requires (written before reading the scripts)

The cases are a pencil beam of $N$ primary electrons at normal incidence on a
slab of thickness $t$ (100 µm W, 1 mm Si) with a $1\times1$ cm² footprint.
The primary is followed down to a 10 keV stop with continuous loss only. No
secondaries are transported, and photons are scored at creation. For primary
$i$ let $E_i$ be its kinetic energy at the end of its track ($0$ if it
stopped), $\hat{\mathbf u}_i$ its final direction, $k_{ij}$ its photon energies
and $f_i$ its fate (transmitted through the back face, reflected, absorbed,
or side exit). Every gated observable is then the sample mean of a
per-primary quantity $x_i$:

| Observable | $x_i$ | Population |
|---|---|---|
| not-transmitted fraction | $\mathbb 1[f_i\neq T]$ | all $N$ |
| continuous deposit | $E_0-E_i-\sum_j k_{ij}$ | all $N$ |
| transmitted loss | $E_0-E_i$ | transmitted |
| mean exit angle | $\theta_i=\arccos(\hat{\mathbf u}_i\cdot\hat{\mathbf z})$ | transmitted |
| rms exit angle | $\theta_i^2$, reported as $\sqrt{\langle\theta^2\rangle}$ | transmitted |
| photon energy above $k_0$ | $\sum_j k_{ij}\,\mathbb 1[k_{ij}\ge k_0]$, $k_0=10,100$ keV | all $N$ |
| photon count above $k_0$ | $\sum_j \mathbb 1[k_{ij}\ge k_0]$ | all $N$ |

Here $\theta$ is the polar angle to the beam axis, not a projected angle.
Photon observables are per incident primary, so primaries that emit nothing
count as zero. The standard error of a mean is $s/\sqrt n$ with the sample
standard deviation $s$; for a fraction this is the binomial error. The rms
error follows from the delta method:

$$
\sigma_{\theta_{\rm rms}}=\frac{\sigma_{\langle\theta^2\rangle}}{2\,\theta_{\rm rms}} .
$$

The two runs are independent, so the difference $a-b$ (PyRITE minus Geant4)
has $\sigma=\sqrt{\sigma_a^2+\sigma_b^2}$. The gate is

$$
\lvert a-b\rvert\le\max\left(0.05\,\lvert b\rvert,\;3\sigma\right),
$$

which is equivalent to gating the ratio $a/b$ at $\max(5\,\%,\,3\sigma/b)$ when
$b\neq0$.

The same quantities must be measured on both sides. That needs the same
thickness and density, the same 10 keV stop, the same lateral footprint, no
loss fluctuations, no delta rays, primaries only, the same photon threshold
convention ($k\ge k_0$), and the photon floor below 10 keV on both sides
(PenBrem 990 eV, PyRITE $k_c$ = 1 keV). A primary stopped by the energy cut
must enter the deposit with its 10 keV residual on both sides.

## 2. Cheap filters: expected magnitudes

The scratch script `sanity.py` (PDG $X_0$: W 0.3504 cm, Si 9.370 cm) gives
$t/X_0$ = 0.02854 for W and 0.01067 for Si.

- **Highland.** The projected width is
  $\theta_0=\frac{13.6\ \rm MeV}{\beta cp}\sqrt{t/X_0}\,[1+0.038\ln(t/(X_0\beta^2))]$.
  For a 2-D Gaussian core the polar mean is $\theta_0\sqrt{\pi/2}$. Geant4's
  $\langle\theta\rangle$ is 0.92–1.22 of that value in W and 1.07–1.11 in Si.
  This is the expected excess from the single-scattering tail beyond
  Highland's 98 % core (accuracy about 11 %). At W 3 MeV the slab is thick in
  scattering terms (12 % backscatter), so the Gaussian picture fails there.
- **Radiative loss.** The photon energy per primary above 1 keV is
  0.79–0.91 of $E_0t/X_0$ at $E_0\ge10$ MeV, rising towards 1 as screening
  becomes complete. This is the expected sub-asymptotic approach. The count
  above 100 keV, estimated as
  $(t/X_0)[\tfrac43\ln(E_0/k_0)-\tfrac56]$, is 0.98–1.04 of Geant4 at
  $E_0\ge10$ MeV (W, Si 30–100 MeV).
- **Collision deposit.** The deposit divided by $\rho t$ is 1.26–1.42
  MeV cm²/g for W (10–100 MeV) and 1.71–1.92 for Si. It rises
  logarithmically and is consistent with unrestricted collision stopping
  times the mean path, which is slightly longer than $t$.
- **Energy closure.** On the Geant4 side, deposit plus photon energy matches
  the transmitted loss within the share carried by non-transmitted
  primaries: 0.3–1 % at $\ge10$ MeV.
- **Ter-Mikaelian.** $\hbar\omega_p$ = 80.3 eV (W) and 31.0 eV (Si) as
  recorded in the result metadata, equal to $28.816\sqrt{\rho Z/A}$ eV.
  Suppressing $dN/dk\propto1/k$ by $k^2/(k^2+(\gamma\hbar\omega_p)^2)$
  predicts deficits in the count above 1 keV of 25 % (W, 100 MeV), 17 %
  (Si, 100 MeV) and 7.6 % (W, 10 MeV). The observed PyRITE/Geant4 deficits
  are 24 %, 15.5 % and 6.6 %.

Units, limits and conventions pass.

## 3. Audit of the scripts

`compare.py::observables` computes exactly the estimators of §1:
`_mean_se` (with `ddof=1`), the delta-method rms error `se_sq / (2 rms)`,
`np.hypot` for the combined σ, and the gate `max(0.05 |b|, 3 sigma)`.
Photon sums use `np.bincount(..., minlength=n)`, so primaries without photons
count as zero. `tests/montecarlo/transport/test_mev_full_track.py::_gate` is the same
expression.

Sameness of the two sides:

| Item | Geant4 (`make_macros.py`, `patch_tracking.py`, emission patch) | PyRITE (`pyrite_bench.py`) | Same? |
|---|---|---|---|
| thickness, density | 100 µm W 19.3; 1 mm Si 2.33 g/cm³ | $10^6$ Å W 19.3; $10^7$ Å Si 2.3292 g/cm³ (metadata) | yes (Si −0.04 %) |
| footprint | `setAbsYZ 1 cm` | `crystal_width_mm=crystal_height_mm=10` (full transverse size) | yes |
| stop | `lowestElectronEnergy 10 keV` | `E_cut_keV=10` | yes |
| fluctuations | `fluct false` | `straggling=False` | yes |
| delta rays | 1 km e⁻ cut | none created | yes |
| secondaries | `killSecondaries 2`, no deposit | not created | yes |
| photon floor | 990 eV (1 nm γ cut) | $k_c$ = 1 keV | below every gated $k_0$ |
| exit direction | `ux` at track end (TestEm5 beam is $+x$) | `v_hat` of the terminal row (pre-event, i.e. exit direction) · `initial_v_hat` | yes |
| stopped primary | `A` with $E=0$; deposit gets residual | `EVENT_CUTOFF` → $E_{\rm exit}=0$ | yes |
| side exit | `A` with $E>0$ → fate 3 | `EVENT_EXIT_SIDE` → fate 3 | yes, see note |
| photons | `EBREM` lines from track 1, `eBrem` creator | `EVENT_HARD_RADIATIVE` rows, `hard_radiative_k_eV` | yes |
| track-length bins | h63, 1000 bins on $[0,E_0]$, step mean energy with discrete brem added back | 1000 bins on $[0,E_0]$, row mean (hard rows end pre-emission) | yes (reported only) |

Side-exit note: TestEm5 classifies a fate at the end of the track, which is
the world boundary ($1.2\times$ the absorber). A side-exiting primary that
reaches the world side face at $x\ge x_{\rm end}$ would be labelled `T`. The
committed fates show 0 side exits in W on both sides and at most 24 in
100,000 in Si, so the effect is below every gate. `_geant4_arrays`
deduplicates `(event, k)` pairs. With 17-digit printing this is harmless.

## 4. Recomputation from committed data

`recompute.py` (scratchpad) reads `reference/*.json` and `results/*.json`
directly and does not use `compare.py`. Production run: **78 of 80** gated
observables pass. The two failures are θ_rms for W 50 MeV
(0.928, $z=-5.5$) and W 100 MeV (0.926, $z=-3.2$). Point-nucleus control:
**16 of 16** pass (θ_rms 0.981, $z=-1.3$, and 0.988, $z=-0.5$; θ_mean 0.998
and 0.999). Every ratio quoted in the ledger and README agrees to the printed
precision; the W 3 MeV transmitted loss is 1.0265, which rounds to 1.026 or
1.027. Running `pyrite-dev test --slow tests/montecarlo/transport/test_mev_full_track.py`
(capped at 8 GB) gave 14 passed in 43 s: the 10 committed-reduction cases,
2 control cases, and the 2 slow 300-primary reruns at 22 s each.

Observables that pass only on the 5 % floor while being statistically
resolved:

| Case | Observable | Ratio | $z$ |
|---|---|---:|---:|
| W 3 MeV | deposit | 1.035 | +11.8 |
| W 3 MeV | transmitted loss | 1.026 | +6.6 |
| W 10 MeV | deposit | 1.013 | +9.4 |
| W 30 MeV | θ_rms | 0.964 | −4.1 |
| Si 100 MeV | deposit | 0.999 | −39.4 |

The deposit is nearly deterministic without fluctuations, so its σ is tiny
and only the 5 % floor applies. This is what the stated gate allows, but the
3.5 % W 3 MeV deposit offset is a real systematic difference.

## 5. Defect: the claim text gates a count the scripts do not gate

The ledger **Claim** says the gate covers "primary bremsstrahlung energy and
count above 10 and 100 keV". `compare.py::GATED`, the test's `GATED`, the
README and the `compare.py` docstring gate the count above 100 keV only.
They list the count above 10 keV as reported, not gated. The 80-observable
total (8 × 10) confirms that. The count above 10 keV is not a formality:

| Case | $N_{k\ge10\ \rm keV}$ PyRITE/Geant4 | $z$ | Gate |
|---|---:|---:|---|
| W 100 MeV | 0.947 | −6.9 | **fails** |
| W 50 MeV | 0.990 | −1.2 | passes |
| others | 1.000–1.039 | | pass |

The physics of the failure is expected. At 100 MeV, $\gamma\hbar\omega_p$ is
15.8 keV in W. PyRITE applies Ter-Mikaelian suppression and PenBrem does
not, and the estimate in §2 gives a 5–7 % deficit above 10 keV. So the
exclusion is justified. But the claim as written is contradicted by committed
data, and the ledger **Notes** "Not covered" lists only counts above 1 keV.

## 6. W 50/100 MeV θ_rms exclusion

`tail.py` bins the committed $\log_{10}\theta$ histograms (80 bins). Binned
rms reproduces the unbinned value within 0.3 %.

| Case | rms ratio, θ < 0.5 rad | $N(\theta>0.3)$ ratio | $N(\theta>0.5)$ ratio | $N(\theta>1)$ ratio |
|---|---:|---:|---:|---:|
| W 50, Fermi | 1.006 | 0.88 ± 0.04 | 0.70 ± 0.06 | 0.32 ± 0.08 |
| W 50, point | 1.002 | 1.00 ± 0.05 | 0.97 ± 0.07 | 0.77 ± 0.14 |
| W 100, Fermi | 0.995 | 0.77 ± 0.07 | 0.60 ± 0.10 | 0.56 ± 0.22 |
| W 100, point | 1.004 | 0.99 ± 0.09 | 0.94 ± 0.13 | 0.83 ± 0.29 |

Errors are Poisson on both counts. Above 0.3 rad the tail holds 39–41 % of
Geant4's $\langle\theta^2\rangle$, so a small population sets θ_rms. The
control changes only the ELSEPA nuclear model (`MNUCL 1` vs `3`; the bench
checks `production_key`, and `point_nucleus_table.py` first reproduces the
production Fermi table to $10^{-6}$). Changing it restores the tail counts to
agreement. With the point nucleus every other observable stays within 0.1 %
of its production value, and the fates change only within Poisson error.
This isolates the cause to the nuclear charge distribution.

An independent check of the magnitude: for a uniform sphere with
$r_{\rm rms}$ = 5.37 fm, the squared form factor
$F^2(q)$, $F(x)=3(\sin x-x\cos x)/x^3$, $x=qR/\hbar c$, $q=2p\sin(\theta/2)$
gives, at 50 MeV and 1 rad, 0.55 in Born approximation and 0.26 with the
Coulomb effective-momentum correction
$q\to q[1+\tfrac32Z\alpha\hbar c/(RE)]$. At 100 MeV and 0.5 rad it gives 0.53
and 0.37. The ELSEPA ratios in the ledger (0.37 and 0.54) fall inside these
brackets, and so do the observed tail ratios 0.32 ± 0.08 and 0.60 ± 0.10.
The finite nucleus is the physical model. The exclusion is justified.

Two corrections to the quoted numbers:

- The ledger's "0.60 ± 0.08" uses PyRITE's Poisson error only. With both
  counts the error is ±0.10. The comparison is unaffected.
- The README's "15–45 % fewer" above ~0.3 rad is 12 % at W 50 MeV above
  0.3 rad; 15–45 % holds above 0.5 rad.

**Unexplained residual (not a gate failure).** Si 50 MeV θ_rms is 0.949. It
lies outside 5 % and passes only on 3σ ($z=-2.9$). Its tail is
$N(\theta>0.3)$ = 0.71 ± 0.08 and $N(\theta>0.5)$ = 0.56 ± 0.11. For Si
($r_{\rm rms}$ = 3.12 fm) $F^2$ is 0.93–0.98 over 0.3–0.5 rad at 50 MeV, so
the Fermi/point mechanism explains at most about 5 %, against a 30–45 %
deficit (3.2–3.4σ). A candidate is Geant4's GS Moliere parameters weighting
by $Z(Z+1)$: atomic electrons add a large-angle tail that is kinematically
forbidden above about $\sqrt{2m_ec^2/E}\approx0.14$ rad. That would account
for $1/Z\approx7\,\%$ in Si. This was not verified against the Geant4 source,
and it still leaves about 2.4σ. Si 100 MeV (0.65 ± 0.16) and W 30 MeV above
1 rad (0.54 ± 0.08, where $F^2$ is 0.50–0.81) point the same way at lower
significance. The ledger attributes the tail to the nuclear model only for
W. This Si residual should be recorded, not generalized into
"reference-model difference".

## 7. Not verifiable here

- "Gates fixed before any PyRITE result was inspected": the scripts and
  results arrive in a single commit (`50ea7871`), so git history cannot show
  the order.
- The ELSEPA Fermi/point ratios themselves (no ELSEPA run here). Only the
  form-factor bracket in §6 checks them.
- The lab-box raw logs and shards. The audit starts from the committed
  reductions.

## 8. Verdict

The estimators, σ, gates and cuts match the independent specification. The
78/80 and 16/16 counts and all quoted ratios reproduce. The W 50/100 MeV θ_rms
exclusion is supported by the point-nucleus control and by an independent
form-factor estimate. The ledger **Claim** sentence still states a gate (count
above 10 keV) that the benchmark does not apply and that W 100 MeV fails.
Verdict: `discrepancy`, limited to the claim wording. After the edit below the
evidence supports `rederived`.

Suggested ledger edit:

- **Claim:** replace "primary bremsstrahlung energy and count above 10 and
  100 keV" with "primary bremsstrahlung energy above 10 and 100 keV and count
  above 100 keV". Append "except θ_rms for W at 50 and 100 MeV (Geant4
  point-nucleus Mott correction; see Notes)".
- **Notes / Not covered:** "photon counts above 1 and 10 keV (W 100 MeV
  ≥ 10 keV: 0.947, $z=-6.9$; Ter-Mikaelian, absent from PenBrem); Si
  ≥ 1 keV counts are 7–15 % low at 30–100 MeV for the same reason".
- **Notes:** change the tail error to 0.60 ± 0.10. Record the unexplained Si
  50 MeV tail deficit (θ_rms 0.949, passes on 3σ; $N(\theta>0.3)$
  0.71 ± 0.08), which nuclear size cannot explain.

## 9. Re-check after the ledger edit (2026-10-23)

I re-read the corrected row. The **Claim** now gates the count above 100 keV
only and excepts θ_rms for W at 50 and 100 MeV. **Not covered** lists the
counts above 1 and 10 keV with values that match §5 and §4. The tail error is
now ±0.10, and the Si 50 MeV residual is recorded. The row is consistent
with the committed reductions and with this write-up.

One non-blocking imprecision remains. The Checks phrase "below 0.2–0.5 rad
the rms agrees within 0.5 %" is true below 0.2–0.3 rad. Below 0.5 rad,
W 50 MeV gives 1.006 in the binned estimate of §6, so "within 0.6 %" would be
exact.

Verdict: `rederived`.
