# `energy-step-convergence`

Ledger row: [`energy-step-convergence`](../physics-validation-ledger.md).
Measurement script: `checks/energy_step_convergence_matrix.py`. No production
code changes; this row records measured convergence behaviour of the existing
propagation rules and radiation kernels, and fixes the tolerances that slices
F--H are built against.

## Claim

Three separate convergence questions, measured over a low-Z/high-Z x thin/thick
x 5--300 keV matrix.

1. **Transport observables are insensitive to the propagation rule, with one
   resolved exception.** Replacing the frozen (left-endpoint) rule with the
   slice-C midpoint rule shifts no exit fraction, retained energy, or transit
   clock by more than 1.6 combined Monte Carlo standard errors at 4000
   electrons across 14 cases. The single resolved difference is the mean path
   length of the 5 keV carbon stopping case, where the frozen rule overstates
   the CSDA range by 1.2%.

2. **The 2%/1%/0.5% fractional-loss ladder named in the checklist does not
   exercise these cases.** The elastic mean free path already holds the
   per-flight fractional energy loss below 2% for ~99% of flights *matrix-wide*,
   so the 2% and 1% rungs subdivide almost nothing. This is not case-universal:
   individual cases carry a per-flight p99 above 2% (C 5 keV thin 6.2%, C 5 keV
   thick 9.9%, W 5 keV thin/thick 3.6%/5.0%, C 25 keV thick 3.0%, W 25 keV
   thick 2.3%), and two of the five Part B radiation cases are not bit-for-bit
   no-ops at the 2% rung. The conclusion is unaffected: where the ladder does
   refine, the frozen rule still misses the 0.1 rad tolerance by two to three
   orders of magnitude. A fractional-loss cap is the wrong control variable.

3. **For coherent CXR the binding constraint is the absolute emission phase,
   and it is met by the propagation rule rather than by the substep count.**
   A coherent kernel weights each row by `e^{i omega t_abs}`; the accumulated
   clock error under the frozen rule is O(10--10^3 rad) at every rung of the
   ladder, while the midpoint rule reaches 1e-3--0.2 rad at one row per
   flight, three to four orders better, at equal cost.

Bremsstrahlung, which carries no phase, converges cleanly under the ladder in
every case.

## Governing equations and where they come from

No new physical law enters. The propagation rules are the ledgered pair from
`transport-midpoint-stopping`: the frozen rule

```
E_end = E_start + (dE/ds)(E_start) · s,        Dt = s / beta(E_start)
```

and the midpoint predictor-corrector

```
E_pred = E_start + (dE/ds)(E_start) · s
E_end  = E_start + (dE/ds)((E_start + E_pred)/2) · s
Dt     = s / beta((E_start + E_end)/2)                    [Ang, c = 1]
```

with the Joy--Luo CSDA `dE/ds` ledgered under `electron-transport`. The check
script re-implements exactly this pair in `_advance` so that both rules can be
applied to one fixed set of physical flights; the lockstep core is the
authority and `tests/montecarlo/test_transport_energy_model.py` pins the two
against each other.

The convergence criterion for the coherent kernel comes from the kernel's own
weight. `mc_spectrum(coherent=True)` forms

```
d2N/dEdOmega = (alpha omega / 4 pi^2 hbar c) * | sum_j A_j Q_j e^{i phi_j} |^2
```

with `phi_j = (omega/hbar c)(t_abs,j - n.r_j)` and
`t_abs = t_ang + L/(2 beta) + t0_ang`. An error `dt` in a row's age therefore
enters as a phase error

```
dphi = omega dt / (hbar c) = (E_res / HBARC_EV_ANG) dt
```

which is the quantity tabulated below. `E_res` is the kernel's own resonance
condition `E_res = hbar c beta (v.g) / (1 - beta (v.n))` evaluated on the
incident direction, so the radian figures are representative of the line the
case actually radiates.

The exactness of pure row splitting follows from the kernel's finite-interaction
factor `Q = t_L sinc(P t_L / pi)`, `P = (1 - beta v.n)(omega - omega_res)/2`.
Splitting a constant-velocity, constant-amplitude flight of duration `t_L` into
`n` substeps of duration `t_L/n`, the substep midpoints sit at
`s_k = (k + 1/2) t_L / n` and, writing `u = P t_L / n`,

```
sum_{k=0}^{n-1} (t_L/n) sinc(u/pi) e^{2 i P s_k}
  = (sin u / P) * e^{i u n} sin(u n) / sin u
  = e^{i P t_L} sin(P t_L) / P
  = [ t_L sinc(P t_L / pi) ] * e^{2 i P (t_L/2)},
```

which is exactly the parent row's amplitude carried at the parent's own
midpoint phase. This Dirichlet-kernel composition identity means that at
frozen energy and clock the coherent sum is invariant to subdivision, so any
measured residual there is the kernel's own per-row approximation. That is what
the `floor` rows report.

## Assumptions and limits of validity

- Part A is a **statistical** comparison. The two rules consume the same seed
  but diverge trajectory by trajectory after the first scatter, so only shifts
  large against the combined Monte Carlo error are meaningful. Fourteen cases
  x seven metrics is 98 comparisons, so isolated 2--3 sigma entries are
  expected; the one shift reported as real is replicated across four
  independent seeds.
- Parts B--D **fix the physical flights** from one frozen transport run and
  refine only the numerical sampling of the emission integral along each
  flight. This deliberately removes trajectory divergence so the emission
  quadrature can be measured on its own; it cannot and does not measure how a
  different rule would change the flights themselves. That is Part A's job.
- The substep chain re-anchors each flight to the parent run's `t_ang`, so a
  rung's per-flight clock error does not accumulate on its own. The `cum`
  columns sum the per-flight error along each electron's own flight sequence,
  which is the correct proxy for the same rule applied inside transport.
- Substep energies are floored at 0.1 keV (Joy--Luo diverges as `1/E`) and
  substep counts capped at 512; a rung that hits the cap is reported so it is
  never read as converged.
- CXR columns require the transported material to BE the emitting crystal, so
  they are carbon/hopg only; tungsten contributes bremsstrahlung alone.
- `cxrI` is a **control, not a convergent quantity**. The row-incoherent
  reduction accumulates `|A_j|^2 |Q_j|^2` per row, so splitting one flight into
  `n` substeps replaces one emitter by `n` independent ones with `1/n` the
  coherent time each. Its magnitude measures the artificial decoherence that
  slice G has to remove.
- All CXR numbers use the kernel's default 119 degree take-off. Near-grazing
  geometries are dominated by a different, unrelated defect; see Part D.

### The clock this row converges is the mean-stopping reference clock

Everything below measures convergence to the continuous-slowing-down reference
clock `t = integral ds / beta(s)`. The tolerance set here bounds numerical
error against that mean-stopping reference. Energy-loss straggling has since
been implemented and measured separately by `energy-loss-straggling`; the
distinction between numerical convergence and physical phase fidelity remains.

- **The microphysics is discrete.** Electrons lose energy in stochastic events
  -- plasmons (most probable loss 25--33 eV in graphite), shell ionization,
  Moller transfers with a `1/T^2` tail -- and travel at constant velocity
  between events. CSDA is the condensed-history mean of that process. The
  usual many-event defense does **not** apply per flight: at 25 keV the current
  transport inputs imply about 1.026 inelastic events per flight in carbon and
  about 0.029 in tungsten. The earlier `0.11` tungsten figure was an arithmetic
  error; it would require a mean excitation energy of 193 eV rather than the
  transport table's 727 eV, or a 3.76-times longer elastic flight.
- **Straggling phase jitter is now modeled.** The selected unrestricted Urban
  compound-Poisson law is normalized to the existing mean stopping power. The
  earlier order-300 eV spread and approximately 0.3 rad Jensen-bias estimate
  describe a plasmon-only Poisson model: at 25 keV over 1 um in HOPG the mean
  loss is 2.247 keV, so a 25 eV quantum gives
  $\sigma_E=\sqrt{\Delta E\epsilon_p}=0.237$ keV and 0.317 rad. It omits shell
  ionization and the full Moller tail; it was correctly computed for that
  narrower model, not a bound on the implemented Urban fluctuation.
- **The measured physical effect is larger.** At the same fixed 1 um material
  path and 1 keV photon-energy clock, paired-seed runs measure a mean phase
  shift of $+13.607 \pm 2.401$ rad and a phase standard deviation of
  $378.628 \pm 5.683$ rad with straggling, against
  $0.560 \pm 0.032$ rad without it. The fixed-path shift lies inside the
  predeclared 9.8--19 rad Urban target. This is the free clock term
  $E_\gamma t/(\hbar c)$, not the coherent kernel's complete phase or a
  Debye--Waller exponent.
- **Why a 0.1 rad tolerance is still the right target.** The frozen rule's
  error is **systematic**: it evaluates `beta` at the flight's start, always
  overestimates speed, and the timing error is one-signed and accumulates
  coherently across every electron in the ensemble. Straggling jitter is
  **random** and changes the distribution of the coherent sum. A tight bound on
  a numerical bias remains meaningful because it controls a different error
  channel and is measured against the same mean-stopping reference.

By Jensen's inequality
`<1/beta(E)> != 1/beta(<E>)` (curvature
`d^2(1/beta)/dE^2 = 3 gamma / ((beta gamma)^5 (mc^2)^2)`), so straggling biases
the mean arrival time, not just its variance. That channel is now sampled; the
paired fixed-path result above replaces the plasmon-only estimate as the
relevant evidence. In the zero-bunch-offset coherent HOPG (002) pure-geometry
limit, the same check measures integrated line yield down
$12.52\% \pm 2.54\%$ and peak height down $19.50\% \pm 2.12\%$. Those are not
angle- or bunch-averaged experimental observables, and residual uncertainty in
the *mean* stopping power remains outside both this row and the straggling
claim.

Phase sensitivity to any energy error carries a `(beta gamma)^-3` prefactor,

```
delta_phi = omega delta_t = (omega/c) integral delta(1/beta) ds,
delta(1/beta) = delta_E / ((beta gamma)^3 m c^2),
```

with `(beta gamma)^3 m c^2` = 16.2 keV at 25 keV and 144 keV at 100 keV, rising
to GeV scale at the tens-of-MeV energies of the PXR literature. That is why the
published coherent-radiation literature does not treat energy-loss straggling
  as a decoherence channel. The present low-energy calculation cannot inherit
  that approximation; `energy-loss-straggling` measures the channel directly.

## Limiting cases

- **Lossless flight.** Every ladder entry and every phase error is identically
  zero: the two propagation rules coincide, and subdivision is the exact
  Dirichlet identity above.
- **Pure row splitting at frozen energy and clock.** The coherent sum is
  algebraically invariant by the identity above, so the measured residual is
  entirely the kernel's remaining per-row approximation: the single
  Beer--Lambert escape factor evaluated at the row midpoint. This is verified
  rather than asserted. For one synthetic flight held well inside a 1e4 Ang
  slab (depth 5000 Ang, `v_z = 0.95`, 25 keV, frozen energy and clock), the
  grid L1 against the unsplit row is

  | flight length L (Ang) | n=2 | n=4 | n=16 | n=64 |
  |---:|---:|---:|---:|---:|
  | 50 | 6.42e-07 | 7.96e-07 | 8.40e-07 | 8.42e-07 |
  | 200 | 1.04e-05 | 1.30e-05 | 1.37e-05 | 1.38e-05 |
  | 800 | 1.67e-04 | 2.08e-04 | 2.21e-04 | 2.22e-04 |
  | 3200 | 2.67e-03 | 3.34e-03 | 3.54e-03 | 3.56e-03 |

  Two signatures identify the term. The residual **saturates** in `n` (6.4e-07
  to 8.4e-07 as `n` goes 2 to 64) rather than growing, so subdivision converges
  to a fixed limit and the Dirichlet identity contributes nothing. And it
  scales as **L^2** -- x16.1, x16.0, x16.0 for each x4 in length -- the
  second-order error of a one-point midpoint rule applied to a smooth weight.
  A flight must not cross the surface for this to hold: a flight whose depth
  goes negative sees a discontinuity, not a quadrature error.

  On the real case matrix the same term measures 6.26e-3--9.27e-3 grid L1 at
  the default take-off. It is a genuine correction that refinement makes, not an
  artifact, but it is not an energy-step effect, so an energy-step claim in the
  same column must exceed it. That is what the `floor` rows report.
- **Small step.** Bremsstrahlung, a plain left-endpoint rectangle rule per row,
  falls first order in the rung: halving the substep length halves the error
  (1.87e-3, 1.40e-3, 8.14e-4 down the ladder as the split fraction grows).

## Part A -- transport observables, frozen versus midpoint

4000 electrons, seed 7, lockstep core, `E_cut = 1 keV`. Exit fractions carry
binomial errors, means carry sample standard errors; `shift` is
`(frozen - midpoint)` in units of the combined error.

| case | model | trans | back | side | stop | path/e | E_ret | clock | loss p99 | clk p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| C    5 keV thin | frozen | 0.9245+-0.0042 | 0.0462+-0.0033 | 0.0000+-0.0000 | 0.0293+-0.0027 | 1291 | 3.882 | 1.002e+04 | 6.20e-02 | 1.55e-02 |
| &nbsp; | midpoint | 0.9267+-0.0041 | 0.0390+-0.0031 | 0.0000+-0.0000 | 0.0343+-0.0029 | 1295 | 3.864 | 1.013e+04 | 6.31e-02 | 1.58e-02 |
| &nbsp; | shift/sig | -0.4 | +1.6 | -- | -1.3 | -0.3 | +1.2 | -0.9 | &nbsp; | &nbsp; |
| C    5 keV thick | frozen | 0.0000+-0.0000 | 0.0595+-0.0037 | 0.0000+-0.0000 | 0.9405+-0.0037 | 3460 | 1.113 | 3.171e+04 | 9.89e-02 | 2.49e-02 |
| &nbsp; | midpoint | 0.0000+-0.0000 | 0.0530+-0.0035 | 0.0000+-0.0000 | 0.9470+-0.0035 | 3424 | 1.104 | 3.165e+04 | 9.89e-02 | 2.49e-02 |
| &nbsp; | shift/sig | -- | +1.3 | -- | -1.3 | +4.3 | +0.8 | +0.7 | &nbsp; | &nbsp; |
| C   25 keV thin | frozen | 0.9980+-0.0007 | 0.0020+-0.0007 | 0.0000+-0.0000 | 0.0000+-0.0000 | 2050 | 24.57 | 6814 | 1.13e-02 | 2.63e-03 |
| &nbsp; | midpoint | 0.9980+-0.0007 | 0.0020+-0.0007 | 0.0000+-0.0000 | 0.0000+-0.0000 | 2050 | 24.56 | 6822 | 1.13e-02 | 2.63e-03 |
| &nbsp; | shift/sig | +0.0 | +0.0 | -- | -- | +0.0 | +0.4 | -0.2 | &nbsp; | &nbsp; |
| C   25 keV thick | frozen | 0.9065+-0.0046 | 0.0475+-0.0034 | 0.0000+-0.0000 | 0.0460+-0.0033 | 2.671e+04 | 18.24 | 9.754e+04 | 3.00e-02 | 7.44e-03 |
| &nbsp; | midpoint | 0.9050+-0.0046 | 0.0478+-0.0034 | 0.0000+-0.0000 | 0.0473+-0.0034 | 2.661e+04 | 18.26 | 9.726e+04 | 3.01e-02 | 7.43e-03 |
| &nbsp; | shift/sig | +0.2 | -0.1 | -- | -0.3 | +0.4 | -0.1 | +0.2 | &nbsp; | &nbsp; |
| C  100 keV thin | frozen | 0.9980+-0.0007 | 0.0020+-0.0007 | 0.0000+-0.0000 | 0.0000+-0.0000 | 2.039e+04 | 98.67 | 3.727e+04 | 3.87e-03 | 7.40e-04 |
| &nbsp; | midpoint | 0.9980+-0.0007 | 0.0020+-0.0007 | 0.0000+-0.0000 | 0.0000+-0.0000 | 2.039e+04 | 98.67 | 3.729e+04 | 3.87e-03 | 7.40e-04 |
| &nbsp; | shift/sig | +0.0 | +0.0 | -- | -- | -0.0 | +0.2 | -0.1 | &nbsp; | &nbsp; |
| C  100 keV thick | frozen | 0.9435+-0.0037 | 0.0348+-0.0029 | 0.0000+-0.0000 | 0.0217+-0.0023 | 2.574e+05 | 80.7 | 4.987e+05 | 9.97e-03 | 2.38e-03 |
| &nbsp; | midpoint | 0.9450+-0.0036 | 0.0348+-0.0029 | 0.0000+-0.0000 | 0.0203+-0.0022 | 2.572e+05 | 80.71 | 4.984e+05 | 9.45e-03 | 2.24e-03 |
| &nbsp; | shift/sig | -0.3 | +0.0 | -- | +0.5 | +0.1 | -0.0 | +0.0 | &nbsp; | &nbsp; |
| C  300 keV thick | frozen | 0.9772+-0.0024 | 0.0208+-0.0023 | 0.0000+-0.0000 | 0.0020+-0.0007 | 1.192e+06 | 268.3 | 1.563e+06 | 1.93e-03 | 2.68e-04 |
| &nbsp; | midpoint | 0.9758+-0.0024 | 0.0215+-0.0023 | 0.0000+-0.0000 | 0.0027+-0.0008 | 1.198e+06 | 268 | 1.573e+06 | 2.01e-03 | 2.89e-04 |
| &nbsp; | shift/sig | +0.4 | -0.2 | -- | -0.7 | -0.5 | +0.6 | -0.6 | &nbsp; | &nbsp; |
| W    5 keV thin | frozen | 0.4363+-0.0078 | 0.4765+-0.0079 | 0.0000+-0.0000 | 0.0872+-0.0045 | 462.9 | 3.56 | 3808 | 3.62e-02 | 9.04e-03 |
| &nbsp; | midpoint | 0.4185+-0.0078 | 0.4895+-0.0079 | 0.0000+-0.0000 | 0.0920+-0.0046 | 465.4 | 3.545 | 3848 | 3.67e-02 | 9.15e-03 |
| &nbsp; | shift/sig | +1.6 | -1.2 | -- | -0.7 | -0.4 | +0.6 | -0.6 | &nbsp; | &nbsp; |
| W    5 keV thick | frozen | 0.0000+-0.0000 | 0.5125+-0.0079 | 0.0000+-0.0000 | 0.4875+-0.0079 | 743 | 2.45 | 6715 | 5.00e-02 | 1.25e-02 |
| &nbsp; | midpoint | 0.0000+-0.0000 | 0.5018+-0.0079 | 0.0000+-0.0000 | 0.4983+-0.0079 | 746.8 | 2.423 | 6786 | 5.01e-02 | 1.25e-02 |
| &nbsp; | shift/sig | -- | +1.0 | -- | -1.0 | -0.4 | +0.8 | -0.8 | &nbsp; | &nbsp; |
| W   25 keV thin | frozen | 0.8482+-0.0057 | 0.1517+-0.0057 | 0.0000+-0.0000 | 0.0000+-0.0000 | 824.3 | 24.24 | 2759 | 3.68e-03 | 8.59e-04 |
| &nbsp; | midpoint | 0.8492+-0.0057 | 0.1507+-0.0057 | 0.0000+-0.0000 | 0.0000+-0.0000 | 824.5 | 24.24 | 2761 | 3.68e-03 | 8.58e-04 |
| &nbsp; | shift/sig | -0.1 | +0.1 | -- | -- | -0.0 | +0.1 | -0.0 | &nbsp; | &nbsp; |
| W   25 keV thick | frozen | 0.0405+-0.0031 | 0.5683+-0.0078 | 0.0000+-0.0000 | 0.3912+-0.0077 | 9413 | 12.41 | 4.035e+04 | 2.33e-02 | 5.81e-03 |
| &nbsp; | midpoint | 0.0445+-0.0033 | 0.5557+-0.0079 | 0.0000+-0.0000 | 0.3997+-0.0077 | 9481 | 12.26 | 4.076e+04 | 2.32e-02 | 5.79e-03 |
| &nbsp; | shift/sig | -0.9 | +1.1 | -- | -0.8 | -0.5 | +0.7 | -0.6 | &nbsp; | &nbsp; |
| W  100 keV thin | frozen | 0.8217+-0.0061 | 0.1782+-0.0061 | 0.0000+-0.0000 | 0.0000+-0.0000 | 8936 | 97.19 | 1.643e+04 | 8.94e-04 | 1.71e-04 |
| &nbsp; | midpoint | 0.8160+-0.0061 | 0.1840+-0.0061 | 0.0000+-0.0000 | 0.0000+-0.0000 | 8895 | 97.2 | 1.635e+04 | 8.94e-04 | 1.71e-04 |
| &nbsp; | shift/sig | +0.7 | -0.7 | -- | -- | +0.3 | -0.3 | +0.3 | &nbsp; | &nbsp; |
| W  100 keV thick | frozen | 0.0333+-0.0028 | 0.5982+-0.0078 | 0.0000+-0.0000 | 0.3685+-0.0076 | 9.743e+04 | 52.67 | 2.214e+05 | 9.63e-03 | 2.38e-03 |
| &nbsp; | midpoint | 0.0355+-0.0029 | 0.6002+-0.0077 | 0.0000+-0.0000 | 0.3643+-0.0076 | 9.795e+04 | 52.53 | 2.223e+05 | 9.51e-03 | 2.35e-03 |
| &nbsp; | shift/sig | -0.6 | -0.2 | -- | +0.4 | -0.3 | +0.2 | -0.2 | &nbsp; | &nbsp; |
| W  300 keV thick | frozen | 0.2200+-0.0065 | 0.6162+-0.0077 | 0.0000+-0.0000 | 0.1638+-0.0059 | 5.278e+05 | 206 | 7.592e+05 | 2.79e-03 | 6.69e-04 |
| &nbsp; | midpoint | 0.2135+-0.0065 | 0.6230+-0.0077 | 0.0000+-0.0000 | 0.1635+-0.0058 | 5.31e+05 | 205.3 | 7.64e+05 | 2.77e-03 | 6.63e-04 |
| &nbsp; | shift/sig | +0.7 | -0.6 | -- | +0.0 | -0.3 | +0.3 | -0.3 | &nbsp; | &nbsp; |

Every entry is within 1.6 sigma except the mean path length of `C 5 keV thick`,
at +4.3 sigma (3460 versus 3424 Ang). Replicated at 2e4 Ang, where the target
is thick enough that mean path length is the CSDA range:

| seed | frozen path (Ang) | midpoint path (Ang) | difference | shift/sigma |
|---|---:|---:|---:|---:|
| 7 | 3459.9 | 3423.6 | +36.2 | +4.3 |
| 101 | 3465.1 | 3424.6 | +40.5 | +4.9 |
| 2024 | 3463.1 | 3423.1 | +40.0 | +4.6 |
| 31337 | 3466.9 | 3425.0 | +42.0 | +4.9 |

The sign and the location are both predicted. Joy--Luo `|dE/ds|` grows as `E`
falls, so the left-endpoint rule understates the loss over a flight, needs more
path length to reach `E_cut`, and overstates the CSDA range. The effect scales
with per-flight fractional loss, which peaks in exactly this case (p99 = 9.9%,
the matrix maximum) and falls below 1% for every case at or above 100 keV,
where all shifts are within 0.6 sigma.

## Part B -- radiation refinement ladder at fixed physical flights

200 electrons, seed 7. Substeps integrated with the midpoint rule so the
reference rung is not itself mis-phased; `cxrCfz` repeats the ladder with the
frozen rule against the same reference, isolating the propagation rule from the
substep count.

| case | f | rows | brem L1 | brem max | cxrC L1 | cxrC max | cxrCfz L1 | cxrCfz max | cxrI L1 | cxrI max |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C   25 keV thin | flight | 1411 | 1.53e-03 | 1.67e-03 | 3.57e-01 | 4.24e-01 | 3.57e-01 | 4.24e-01 | 2.60e-01 | 1.60e+00 |
| &nbsp; | 2.000% | 1411 | 1.53e-03 | 1.67e-03 | 3.57e-01 | 4.24e-01 | 3.57e-01 | 4.24e-01 | 2.60e-01 | 1.60e+00 |
| &nbsp; | 1.000% | 1433 | 1.33e-03 | 1.44e-03 | 3.36e-01 | 3.45e-01 | 3.24e-01 | 3.72e-01 | 2.52e-01 | 1.24e+00 |
| &nbsp; | 0.500% | 1615 | 8.20e-04 | 8.66e-04 | 1.18e-01 | 1.16e-01 | 2.32e-01 | 2.87e-01 | 2.23e-01 | 7.49e-01 |
| &nbsp; | floor | &nbsp; | 1.48e-05 | 1.65e-04 | 8.94e-03 | 5.89e-03 | 8.94e-03 | 5.89e-03 | 2.53e-01 | 6.16e-01 |
| C   25 keV thick | flight | 18944 | 2.08e-03 | 2.50e-03 | 4.17e-01 | 5.56e-01 | 4.17e-01 | 5.56e-01 | 1.42e-01 | 1.02e+00 |
| &nbsp; | 2.000% | 19620 | 2.02e-03 | 2.25e-03 | 4.16e-01 | 5.58e-01 | 4.17e-01 | 5.57e-01 | 1.39e-01 | 1.02e+00 |
| &nbsp; | 1.000% | 21706 | 1.54e-03 | 1.65e-03 | 3.57e-01 | 4.29e-01 | 4.07e-01 | 4.72e-01 | 1.19e-01 | 8.21e-01 |
| &nbsp; | 0.500% | 28399 | 8.62e-04 | 9.15e-04 | 1.20e-01 | 1.11e-01 | 2.65e-01 | 2.55e-01 | 9.88e-02 | 5.34e-01 |
| &nbsp; | floor | &nbsp; | 8.94e-06 | 1.35e-04 | 9.27e-03 | 5.21e-03 | 9.27e-03 | 5.21e-03 | 1.42e-01 | 5.05e-01 |
| C  100 keV thick | flight | 42389 | 4.85e-04 | 6.42e-04 | 4.66e-01 | 5.40e-01 | 4.66e-01 | 5.40e-01 | 1.03e-01 | 6.68e-01 |
| &nbsp; | 2.000% | 42562 | 4.90e-04 | 6.42e-04 | 4.66e-01 | 5.40e-01 | 4.66e-01 | 5.40e-01 | 1.03e-01 | 6.68e-01 |
| &nbsp; | 1.000% | 42943 | 4.87e-04 | 6.42e-04 | 4.66e-01 | 5.40e-01 | 4.66e-01 | 5.40e-01 | 1.03e-01 | 6.68e-01 |
| &nbsp; | 0.500% | 44246 | 4.27e-04 | 5.95e-04 | 4.71e-01 | 5.41e-01 | 4.69e-01 | 5.40e-01 | 9.38e-02 | 6.11e-01 |
| &nbsp; | floor | &nbsp; | 2.39e-05 | 5.57e-04 | 6.26e-03 | 3.87e-03 | 6.26e-03 | 3.87e-03 | 9.36e-02 | 3.81e-01 |
| W   25 keV thick | flight | 120242 | 3.51e-04 | 9.04e-04 | -- | -- | -- | -- | -- | -- |
| &nbsp; | 2.000% | 122173 | 4.20e-04 | 7.30e-04 | -- | -- | -- | -- | -- | -- |
| &nbsp; | 1.000% | 127776 | 4.19e-04 | 6.08e-04 | -- | -- | -- | -- | -- | -- |
| &nbsp; | 0.500% | 144250 | 3.49e-04 | 4.51e-04 | -- | -- | -- | -- | -- | -- |
| &nbsp; | floor | &nbsp; | 8.37e-06 | 4.19e-05 | -- | -- | -- | -- | -- | -- |
| W  100 keV thick | flight | 422738 | 2.86e-05 | 9.04e-05 | -- | -- | -- | -- | -- | -- |
| &nbsp; | 2.000% | 424403 | 2.87e-05 | 7.49e-05 | -- | -- | -- | -- | -- | -- |
| &nbsp; | 1.000% | 429110 | 2.81e-05 | 6.41e-05 | -- | -- | -- | -- | -- | -- |
| &nbsp; | 0.500% | 442556 | 2.61e-05 | 5.40e-05 | -- | -- | -- | -- | -- | -- |
| &nbsp; | floor | &nbsp; | 1.47e-06 | 1.91e-05 | -- | -- | -- | -- | -- | -- |

## Part B -- phase criterion

The operative result. Per-flight and trajectory-accumulated clock error of each
rung against the midpoint rule at `f = 0.125%`, converted to radians at the
case's hopg (0,0,2) resonance.

| case | rule | f | dt p50 | dt p99 | cum dt p99 | cum dt max | dphi p99 | cum dphi p99 | cum dphi max |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C   25 keV thin | frozen | flight | 2.74e-01 | 9.85e+00 | 1.53e+01 | 3.67e+01 | 4.86e+00 | 7.56e+00 | 1.81e+01 |
| &nbsp; | &nbsp; | 2.000% | 2.74e-01 | 9.85e+00 | 1.53e+01 | 3.67e+01 | 4.86e+00 | 7.56e+00 | 1.81e+01 |
| &nbsp; | &nbsp; | 1.000% | 2.74e-01 | 7.90e+00 | 1.11e+01 | 2.05e+01 | 3.89e+00 | 5.49e+00 | 1.01e+01 |
| &nbsp; | &nbsp; | 0.500% | 2.74e-01 | 4.31e+00 | 6.21e+00 | 1.23e+01 | 2.13e+00 | 3.06e+00 | 6.04e+00 |
| &nbsp; | midpoint | flight | 3.71e-06 | 8.97e-04 | 1.57e-03 | 4.67e-03 | 4.42e-04 | 7.75e-04 | 2.30e-03 |
| &nbsp; | &nbsp; | 2.000% | 3.71e-06 | 8.97e-04 | 1.57e-03 | 4.67e-03 | 4.42e-04 | 7.75e-04 | 2.30e-03 |
| &nbsp; | &nbsp; | 1.000% | 3.71e-06 | 6.48e-04 | 7.78e-04 | 1.24e-03 | 3.20e-04 | 3.84e-04 | 6.09e-04 |
| &nbsp; | &nbsp; | 0.500% | 3.71e-06 | 1.61e-04 | 2.33e-04 | 3.77e-04 | 7.95e-05 | 1.15e-04 | 1.86e-04 |
| C   25 keV thick | frozen | flight | 4.24e-01 | 2.01e+01 | 8.77e+02 | 1.12e+03 | 9.91e+00 | 4.33e+02 | 5.50e+02 |
| &nbsp; | &nbsp; | 2.000% | 4.24e-01 | 1.63e+01 | 6.26e+02 | 7.16e+02 | 8.04e+00 | 3.08e+02 | 3.53e+02 |
| &nbsp; | &nbsp; | 1.000% | 4.16e-01 | 8.52e+00 | 4.37e+02 | 4.80e+02 | 4.20e+00 | 2.15e+02 | 2.37e+02 |
| &nbsp; | &nbsp; | 0.500% | 3.90e-01 | 5.06e+00 | 2.64e+02 | 2.79e+02 | 2.49e+00 | 1.30e+02 | 1.37e+02 |
| &nbsp; | midpoint | flight | 7.73e-06 | 3.32e-03 | 1.05e-01 | 1.80e+00 | 1.63e-03 | 5.15e-02 | 8.87e-01 |
| &nbsp; | &nbsp; | 2.000% | 7.69e-06 | 1.79e-03 | 3.27e-02 | 3.88e-02 | 8.83e-04 | 1.61e-02 | 1.91e-02 |
| &nbsp; | &nbsp; | 1.000% | 7.45e-06 | 6.14e-04 | 1.69e-02 | 1.75e-02 | 3.03e-04 | 8.32e-03 | 8.64e-03 |
| &nbsp; | &nbsp; | 0.500% | 6.44e-06 | 1.64e-04 | 5.46e-03 | 5.58e-03 | 8.09e-05 | 2.69e-03 | 2.75e-03 |
| C  100 keV thick | frozen | flight | 2.33e-01 | 1.06e+01 | 1.48e+03 | 2.02e+03 | 8.61e+00 | 1.20e+03 | 1.64e+03 |
| &nbsp; | &nbsp; | 2.000% | 2.33e-01 | 1.02e+01 | 1.43e+03 | 1.68e+03 | 8.28e+00 | 1.16e+03 | 1.37e+03 |
| &nbsp; | &nbsp; | 1.000% | 2.32e-01 | 9.90e+00 | 1.30e+03 | 1.44e+03 | 8.03e+00 | 1.05e+03 | 1.17e+03 |
| &nbsp; | &nbsp; | 0.500% | 2.32e-01 | 8.21e+00 | 1.01e+03 | 1.09e+03 | 6.66e+00 | 8.21e+02 | 8.81e+02 |
| &nbsp; | midpoint | flight | 0.00e+00 | 4.62e-04 | 6.70e-02 | 4.74e-01 | 3.75e-04 | 5.43e-02 | 3.84e-01 |
| &nbsp; | &nbsp; | 2.000% | 0.00e+00 | 4.24e-04 | 6.36e-02 | 7.42e-02 | 3.43e-04 | 5.16e-02 | 6.01e-02 |
| &nbsp; | &nbsp; | 1.000% | 0.00e+00 | 3.72e-04 | 4.80e-02 | 5.09e-02 | 3.02e-04 | 3.89e-02 | 4.13e-02 |
| &nbsp; | &nbsp; | 0.500% | 0.00e+00 | 2.41e-04 | 2.40e-02 | 2.49e-02 | 1.95e-04 | 1.95e-02 | 2.02e-02 |
| W   25 keV thick | frozen | flight | 1.19e-02 | 9.45e-01 | 8.83e+01 | 1.21e+02 | 4.66e-01 | 4.35e+01 | 5.96e+01 |
| &nbsp; | &nbsp; | 2.000% | 1.19e-02 | 7.15e-01 | 7.60e+01 | 8.58e+01 | 3.53e-01 | 3.75e+01 | 4.23e+01 |
| &nbsp; | &nbsp; | 1.000% | 1.19e-02 | 4.88e-01 | 6.25e+01 | 6.87e+01 | 2.40e-01 | 3.08e+01 | 3.39e+01 |
| &nbsp; | &nbsp; | 0.500% | 1.19e-02 | 2.97e-01 | 4.62e+01 | 4.98e+01 | 1.47e-01 | 2.28e+01 | 2.45e+01 |
| &nbsp; | midpoint | flight | 0.00e+00 | 7.68e-04 | 5.22e-02 | 5.08e-01 | 3.79e-04 | 2.58e-02 | 2.51e-01 |
| &nbsp; | &nbsp; | 2.000% | 0.00e+00 | 3.80e-04 | 1.94e-02 | 2.79e-02 | 1.87e-04 | 9.54e-03 | 1.37e-02 |
| &nbsp; | &nbsp; | 1.000% | 0.00e+00 | 1.13e-04 | 8.20e-03 | 1.02e-02 | 5.59e-05 | 4.04e-03 | 5.02e-03 |
| &nbsp; | &nbsp; | 0.500% | 0.00e+00 | 3.13e-05 | 2.87e-03 | 3.41e-03 | 1.55e-05 | 1.42e-03 | 1.68e-03 |
| W  100 keV thick | frozen | flight | 3.57e-03 | 3.42e-01 | 1.23e+02 | 1.77e+02 | 2.77e-01 | 9.99e+01 | 1.44e+02 |
| &nbsp; | &nbsp; | 2.000% | 3.57e-03 | 3.27e-01 | 1.19e+02 | 1.41e+02 | 2.65e-01 | 9.65e+01 | 1.14e+02 |
| &nbsp; | &nbsp; | 1.000% | 3.57e-03 | 2.69e-01 | 1.10e+02 | 1.23e+02 | 2.18e-01 | 8.91e+01 | 9.99e+01 |
| &nbsp; | &nbsp; | 0.500% | 3.57e-03 | 1.98e-01 | 9.61e+01 | 1.04e+02 | 1.61e-01 | 7.80e+01 | 8.40e+01 |
| &nbsp; | midpoint | flight | 0.00e+00 | 7.24e-05 | 1.78e-02 | 3.66e-01 | 5.87e-05 | 1.44e-02 | 2.97e-01 |
| &nbsp; | &nbsp; | 2.000% | 0.00e+00 | 7.24e-05 | 1.16e-02 | 2.85e-02 | 5.87e-05 | 9.39e-03 | 2.31e-02 |
| &nbsp; | &nbsp; | 1.000% | 0.00e+00 | 4.17e-05 | 5.80e-03 | 1.05e-02 | 3.38e-05 | 4.70e-03 | 8.53e-03 |
| &nbsp; | &nbsp; | 0.500% | 0.00e+00 | 1.26e-05 | 2.29e-03 | 3.61e-03 | 1.02e-05 | 1.85e-03 | 2.93e-03 |
| exit=0 | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

## Part C -- physical-flight-incoherent CXR

| case | f | rows | flights | cxrF L1 | cxrF max | cxrC L1 | cxrC max |
|:---|---:|---:|---:|---:|---:|---:|---:|
| C   25 keV thin | flight | 409 | 409 | 1.64e-01 | 2.72e-01 | 4.50e-01 | 5.85e-01 |
| &nbsp; | 2.000% | 409 | 409 | 1.64e-01 | 2.72e-01 | 4.50e-01 | 5.85e-01 |
| &nbsp; | 1.000% | 421 | 409 | 1.64e-01 | 2.72e-01 | 3.53e-01 | 3.53e-01 |
| &nbsp; | 0.500% | 474 | 409 | 5.27e-02 | 3.57e-02 | 1.31e-01 | 1.18e-01 |

## Part D -- row-splitting floor versus take-off geometry

| geometry | n_z | E_res eV | f=1% L1 | f=0.125% L1 |
|:---|---:|---:|---:|---:|
| default 119 deg | -0.485 | 973 | 2.41e-03 | 8.94e-03 |
| grazing (1, 0, 0.01) | +0.010 | 1118 | 8.71e-02 | 3.38e-01 |
| exit=0 | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

## What this row does not claim

- No production behaviour changes. This is a measurement row; the check script
  imports the kernels and transport unchanged.
- It does not claim the coherent CXR spectra of this repository are correct,
  only how they respond to numerical refinement of the emission rows.
- It does not select a final propagator. Optical-depth handling and the
  physical-flight/substep identity are slice F; the invariant reductions are
  slice G.

## Independent verification

Fresh-context rederivation (2026-08-18, verifier context separate from the
implementation, from the ledger row and this write-up's Claim section only).
This row is a measurement + step-control decision, not a new equation, so the
unit of trust is the measurement's validity and the honesty of its
conclusions.

**Filters.** Units `pass`: accumulated phase is reported in rad
($\Delta\phi=\omega\Delta t=(E_{\rm res}/\hbar c)\Delta t$, with
$E_{\rm res}/\hbar c$ in $\mathrm{\mathring A}^{-1}$ and $\Delta t$ in
$\mathrm{\mathring A}$ under $c=1$, matching the ledgered clock convention of
`transport-midpoint-stopping`); fractional loss and the L1/shift ratios are
dimensionless. Limits `pass`: a lossless flight makes `dE/ds=0`, so the two
propagation rules coincide exactly and every tabulated entry is identically
zero, as claimed. Pure row splitting at frozen energy and clock was checked
by an independent re-derivation of the Dirichlet composition identity before
reading the script: writing $u=Pt_L/n$, $s_k=(k+\tfrac12)t_L/n$, and using
$\operatorname{sinc}(u/\pi)=\sin(u)/u$,

$$
\sum_{k=0}^{n-1}\frac{t_L}{n}\operatorname{sinc}(u/\pi)\,e^{2iPs_k}
=\frac{t_L}{n}\frac{\sin u}{u}\,e^{iu}\sum_{k=0}^{n-1}e^{i2uk}
=\frac{t_L}{n}\frac{\sin u}{u}\,e^{iu}\,e^{iu(n-1)}\frac{\sin(nu)}{\sin u}
=\frac{\sin(Pt_L)}{P}e^{iPt_L}
=t_L\operatorname{sinc}\!\left(\frac{Pt_L}{\pi}\right)e^{2iP(t_L/2)},
$$

using $nu=Pt_L$ and the geometric-series identity
$\sum_{k=0}^{n-1}e^{i2uk}=e^{iu(n-1)}\sin(nu)/\sin(u)$. This reproduces the
write-up's stated identity term for term, so subdivision at frozen energy and
clock is exactly the parent row's own amplitude and midpoint phase, and any
measured residual there is the kernel's own per-row escape-factor
approximation, not an energy-step effect. Signs/conventions `pass`: Joy–Luo
$dE/ds<0$ makes the frozen (left-endpoint) rule evaluate $\beta$ too high and
$\Delta t=s/\beta$ too low on every row, a one-signed bias, matching the
"systematic, not random" characterization used later in the write-up.

**Step-control reasoning, derived before reading the check script.** A
coherent kernel's row weight is $e^{i\omega t_{\rm abs}}$ with
$t_{\rm abs}$ the running sum of per-flight $\Delta t$ along one electron's
whole trajectory. Because Joy–Luo's sign is fixed, the frozen rule's
per-flight clock error is not a random, cancelling perturbation but a
same-sign local truncation term; summed over the $M\sim10$–$10^2$ flights
that make up a $10^3$–$10^4\,\mathrm{\mathring A}$ trajectory, the
accumulated error grows roughly linearly in $M$ rather than being suppressed
by any $\sqrt M$ averaging. A per-flight fractional-loss cap $|dE|/E<f$
bounds only the *local* truncation term of the flight it is applied to; it is
blind to $M$ and to the fact that refining within the same left-endpoint
substep rule leaves the sign unchanged, so it does not bound the
*accumulated* quantity that actually enters the exponent. An absolute,
trajectory-accumulated phase tolerance is therefore the dimensionally and
causally correct control variable, and switching propagation rule (frozen to
midpoint) attacks the sign of the local term directly, which a smaller $f$
cannot. This reasoning is self-consistent and is exactly what the Part B
phase-criterion table subsequently shows: `cum dphi p99` under the frozen
rule falls only slowly down the $f$ ladder (e.g. C 25 keV thick:
433, 308, 215, 130 rad at flight/2%/1%/0.5%) while the midpoint rule already
clears 0.1 rad at zero refinement (`5.15e-02` rad, same case, `flight` rung).

**Re-derivation vs. the write-up and script.** Read after the above:
`checks/energy_step_convergence_matrix.py` and Parts A–D.

- *Part A path-length bias.* Correctly treated as a detected bias, not a
  null: the write-up explicitly separates the "98 comparisons, isolated 2–3σ
  expected" look-elsewhere caveat from the one entry (`C 5 keV thick` path
  length, +4.3σ) that is replicated at +4.3/+4.9/+4.6/+4.9σ across four
  independent seeds, with the sign and case predicted in advance by
  Joy–Luo's `1/E` growth. `shift/sig` uses
  $(\text{value}_{\rm frozen}-\text{value}_{\rm midpoint})/\sqrt{\sigma_f^2+\sigma_m^2}$
  (`checks/energy_step_convergence_matrix.py:565`), and because both rules
  share one seed and the same first scatter, the true correlated-difference
  variance is at most this independent-sum value, so if anything the quoted
  σ is conservative (understates significance) rather than inflated — a
  favourable direction for a "detected bias" claim. Exit-fraction SEs use the
  binomial $\sqrt{p(1-p)/N_e}$ and continuous observables use
  `std(ddof=1)/sqrt(Ne)`, both correct sample-SE forms.
- *2%/1% no-op claim.* Supported but imprecisely scoped. Several individual
  cases have per-flight fractional-loss p99 above 2% (`C 5 keV thin` 6.2%,
  `C 5 keV thick` 9.9%, `C 25 keV thick` 3.0%, `W 5 keV thin/thick` 3.6%/5.0%,
  `W 25 keV thick` 2.3%), and Part B shows the 2% rung is *not* a bit-for-bit
  no-op for two of the five measured radiation cases (`C 25 keV thick`:
  18944→19620 rows; `W 25 keV thick`: 120242→122173 rows). The "~99% of
  flights" figure is true as a matrix-wide statement but reads as
  case-universal in the Claim section; this is a minor overstatement of
  precision, not of substance — the ladder's own phase table shows that even
  where it does refine (down to the 0.5% rung), the frozen rule still misses
  the 0.1 rad tolerance by two to three orders of magnitude, so the
  conclusion "the ladder is the wrong control variable" survives regardless.
- *`coherent=True` ill-conditioning defect.* Correctly scoped as not an
  energy-step effect: `part_d`'s docstring and the Dirichlet identity above
  both establish that subdivision at frozen energy/clock is exact, so a
  large, non-vanishing residual under refinement is a statement about the
  reduction's own conditioning (a small residual of a large complex-valued
  cancellation), independent of any propagation-rule choice. "Ill-conditioned"
  is the right diagnosis. The specific "54% residual at 100 keV thick" figure
  quoted in the ledger row's Notes, however, is not traceable to any printed
  number in this write-up; the closest analogue is the Part B `cxrC`/`cxrCfz`
  L1 residual for `C 100 keV thick`, which sits at 46.6–47.1% across the
  unrefined-to-0.5% rungs (same order of magnitude, same qualitative
  "residual does not shrink under refinement" signature, but not an exact
  match to "54%"). This is a numeric-provenance gap between the ledger prose
  and the linked record, not a physics error.
- *Near-grazing floor independent of `energy_model`.* Verified by
  construction: `part_d` never passes `energy_model` to `simulate_trajectories`
  and holds energy and clock frozen while only subdividing rows, so the
  measured floor cannot depend on the propagation rule by the same Dirichlet
  argument used above. Numerically, however, the write-up's own Part D table
  (`8.71e-02`, `3.38e-01` for `f=1%`/`f=0.125%` at the grazing geometry) does
  not match the ledger row's quoted `1.7e-1–3.5e-1`; a `--quick` (`Ne=25`)
  rerun of `--part d` here reproduced `1.66e-01`/`3.50e-01`, i.e. the ledger
  prose matches a smaller-`Ne` run rather than the `Ne=200` numbers tabulated
  in this file. Both runs land in the same qualitative regime (tens of
  percent, roughly one to two orders above the ~1% default-take-off floor),
  so the substantive conclusion is unaffected, but the exact figures in the
  ledger Notes should be re-pulled from a single, stated run.
- *Systematic vs. random argument for the 0.1 rad tolerance.* Still sound as a
  separation of error channels: bounding the numerical bias tightly remains
  meaningful even though the now-modeled random loss produces a much wider
  clock distribution. The former approximately 0.3 rad open-gap estimate was
  a plasmon-only model, not a full-loss bound; `energy-loss-straggling` replaces
  it with the paired Urban measurement and keeps the result explicitly scoped
  to the free clock and zero-bunch-offset coherent limit.

**Other findings.** The ledger row's Notes attribute this claim with a
statement that `elec_id` is a row mask, not a grouping key, in `lines.py`.
That is correct — `lines.py:1003` computes `line_electron = seg_elec_id < Ne`,
matching the check script's own comment
(`checks/energy_step_convergence_matrix.py:28-29`) — but neither `elec_id`
nor `lines.py` appears anywhere in this write-up's body; the supporting
evidence lives only in the script docstring and the ledger prose. No other
stale wording was found beyond the grooved/per-electron/CUDA note already
flagged as known-stale.

**Verdict.** `rederived`: the Dirichlet identity, the phase-tolerance
reasoning, and the statistical methods (SE formulas, paired-seed shift,
seed-replication of the one significant bias) all check out on independent
re-derivation, and every headline conclusion (systematic frozen-rule bias,
ineffectiveness of the fractional-loss ladder as a control variable, exact
row-splitting invariance, correct scoping of the two open defects as
non-energy-step effects, near-grazing-independent-of-`energy_model` floor,
and the systematic/random tolerance argument) is honestly and correctly
characterized. Outstanding items for the ledger editor: (1) the "54%" and
"1.7e-1–3.5e-1" figures in the ledger Notes should be re-pulled from one
specific run and matched to numbers actually printed in this file; (2) the
"~99% of flights below 2%" framing in the Claim section should note it is a
matrix-wide, not per-case, statistic, since two of five Part B cases are not
in fact no-ops at the 2% rung. Neither item changes the physics verdict.
`signed-off` remains a human decision.
