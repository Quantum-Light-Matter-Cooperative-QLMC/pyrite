# `mixed-elastic-hinge`

## Scope and independence

Validation: `mixed-elastic-hinge`. Fresh-context verification on 2026-11-05, same state as [`mixed-elastic-split`](mixed-elastic-split.md). Source: PENELOPE-2024 Eqs. 4.20–4.23, 4.28–4.32, 4.85, 4.95–4.98, §§4.3.3, 4.4 and 4.4.1. Derived before reading the implementation.

Verdict: **`discrepancy`**, in the realized step length. The hinge law, its moments, the units and the state machine all match.

## Independent derivation

### Moments and two-box law

With $e_\ell=\exp(-s/\lambda^{(s)}_{\rm el,\ell})$, $\langle P_1\rangle=1-2\langle\mu\rangle=e_1$ and $\langle P_2\rangle=1-6\langle\mu\rangle+6\langle\mu^2\rangle=e_2$ give Eqs. 4.28–4.29.

For $F_a=aU_{0,\mu_0}+(1-a)U_{\mu_0,1}$, the mean gives $2\langle\mu\rangle=\mu_0+1-a$, and the second moment gives $3\langle\mu^2\rangle=\mu_0^2+(1-a)(1+\mu_0)$. Eliminating $a$ gives Eq. 4.32. Substituting the moments gives a closed form with no cancellation:

$$
\mu_0=\tfrac12\left(1-\frac{e_2}{e_1}\right),\qquad
a=e_1+\mu_0,\qquad
1-a=\tfrac12\left(1-2e_1+\frac{e_2}{e_1}\right).
$$

Reachable range, with $x=s/\lambda^{(s)}_{\rm el,1}$ and $y=s/\lambda^{(s)}_{\rm el,2}$:

- $\mu_0\ge0$ if and only if $y\ge x$, that is $T_2\le\tfrac23T_1$.
- $6\mu(1-\mu)\le6\mu$ gives $y\le3x$, so $e^{x-y}\ge e^{-2x}\ge2e^{-x}-1$ and $a\le1$ always.
- With $t=e^{-x}$, the smallest second moment, at $y=3x$, is $\langle\mu^2\rangle=(1-t)^2(t+2)/6\ge0$, so it is non-negative in exact arithmetic.

### Hinge position and truncation

If a step $s$ is cut at $t<s$ independently of $\tau\sim U(0,s)$, the hinge is reached with probability $t/s$ and contributes $s/\lambda$, so the mean deflection is $t/\lambda$ (Eq. 4.23). Dropping a pending hinge is therefore correct to first order, provided the cut is independent of $\tau$ and the moments use the planned $s$.

At fixed energy the renewal process has the exact decay rate: with hard rate $r$, soft rate $\alpha$ and mean hard deflection $h$, the pole of the Laplace-transformed $\langle\cos\chi\rangle$ lies at $-(\alpha+rh)$.

## Source-to-code comparison

| Item | Source | Code | Result |
|---|---|---|---|
| Moments | Eqs. 4.28–4.29 | `_hinge_moments`, using `expm1` | matches |
| Two-box law | Eqs. 4.30–4.32 | `_sample_hinge_mu` | matches, including both inverse-CDF branches |
| Position | $\tau=\xi s$ | `cores.py:509` | matches |
| Energy and length | hinge's energy, full $s$ (§4.3.3, step viii) | `cores.py:1053`: `mix_length[e]`, `E_keV[e]` after the row | matches |
| Step cap | $4\lambda^{(h)}_T$ (§4.4) | `cores.py:505` | matches, $1/\lambda^{(h)}_T$ is elastic + inelastic + radiative |
| Interface | Eq. 4.23, step vi | `cores.py:1066`, boundary precedence at `cores.py:796` | matches |

### Units

| Line | Expression | Check |
|---|---|---|
| `cores.py:446` | `1e8 / total_rate` | [1/cm] to Angstrom |
| `cores.py:463`, `476` | `tau / mu_hard`, `tau / mu_rad` against `step_j` | both are 1/Angstrom |
| `cores.py:489`, `491` | `mu_hard * 1e8`, `mu_rad * 1e8` added to `hard_total` | 1/Angstrom to 1/cm |
| `cores.py:505` | `4 * 1e8 / hard_total` | Angstrom |
| `cores.py:501`, `506` | `mix_stopping -= rad_soft * 1e-3`; `c2 * E / (-mix_stopping)` | `dEds` is negative in keV/Angstrom, `rad_soft` is eV/Angstrom, so radiative loss adds to the stopping and the length is positive |
| `_hinge_direction` | `step_ang * 1e-8` | Angstrom to cm against rates in 1/cm |
| `layer_step_limits_ang` | `max_step_um * 1e4` | um to Angstrom |

### State machine

Traced on `cores.py:478-515`, `942-956` and `1019-1072`:

- Optical depths survive hinge, delta and substep rows, because those paths `continue` before the reset at `cores.py:1068`.
- A boundary or a hard event resets all optical depths and the step state.
- A cutoff is terminal.
- A substep keeps `mix_left` and `mix_hinge`, decremented by the row. A pending hinge therefore survives `max_dE_frac`, and hinge row precedence is correct because `limited_j` outranks it at `cores.py:784`.
- A delta row can only follow a fired hinge, since $\tau\le s$.
- Row events stay unique: every hinge or delta advances `flight_id`, and draws advance by 2 per hinge.

### Numerics and clamps

Over released tables $x\le4\times0.796\times0.2=0.64$ and $T_2/T_1\le0.0103$, against the $\tfrac23$ needed to trigger the clamp. Therefore:

- The `mu_0 < 0` clamp is unreachable on released tables. It preserves the first moment and would lose the second.
- The `a > 1` and `second < 0` clamps fire only at rounding level, as derived above. They cannot bias a physical case.
- Division by $1-2\langle\mu\rangle$ raises `ZeroDivisionError` once $\langle\mu\rangle$ rounds to $\tfrac12$, at $x\gtrsim37$. This was reproduced at $x=40$. It is unreachable with the $4\lambda^{(h)}_T$ cap.

Cancellation in $\langle\mu^2\rangle$ is benign. The absolute error is about $\epsilon\langle\mu\rangle$ with leading value $\langle\mu^2\rangle\simeq A'+2A^2$, where $A=s\Sigma T_1$ and $A'=s\Sigma T_2$. Direct quadrature of the sampler reproduces both moments to $10^{-12}$ for $x\ge10^{-4}$ and to about 1 % on the second moment at $x=10^{-8}$. Below $x\approx10^{-12}$, $a$ rounds to 1 and the upper box vanishes: the second moment is lost, but the mean is carried by the lower box and the lost term has probability below $\epsilon$. No observable can see this.

## Discrepancy

First divergent convention: `cores.py:1053` uses the planned length `mix_length[e]` for the soft moments, while the hard event at `cores.py:463`, `476` and the elastic row can end the step at a realized $t<s$ after the hinge has fired. PENELOPE realizes exactly the sampled $s$ (step iii and step xii, Eq. 4.121), because it thins with $\Sigma_{h,\max}$ (Eqs. 4.96–4.98).

Here the first row of a step always ends at the hinge when `max_dE_frac = 0`, because $\tau<s\le$ `step_j`. The hinge therefore fires with probability 1, not $t/s$, and the Eq. 4.23 argument does not apply:

- Rates that rise after the hinge overcount the soft moment by $(s-t)/s$.
- The ledger's "hinge overtaken by a hard event is dropped" happens only after a `max_dE_frac` substep, and that case is correct to first order, by Eq. 4.23.
- So the row's stated effect is the benign one, and the unlisted effect is the biased one.

### Measurement

Si, 30 to 24 keV, summed hinge $1-\cos$ against $\int\lambda^{-1}_{s,1}ds$. Because $\langle1-\cos\rangle$ per hinge is $1-e^{-x}$, a correct hinge gives a ratio of about $1-\bar x$ (0.995 at $C_1=0.05$, 0.942 at $C_1=0.2$).

| Run | Soft first moment / exact integral | Soft share of first moment |
|---|---|---|
| $C_1=C_2=0.05$ | 1.0049 | 0.106 |
| $C_1=C_2=0.05$, `max_dE_frac` = 0.002 | 0.9955 | 0.106 |
| $C_1=C_2=0.2$ | 0.9772 | 0.290 |
| $C_1=C_2=0.2$, `max_dE_frac` = 0.002 | 0.9501 | 0.290 |

The excess is about $+1$ % at $C_1=0.05$ and $+3$ % at $C_1=0.2$, on the soft part only. Substeps remove it. It has the opposite sign to the larger hard-rate deficit recorded in [`mixed-elastic-split`](mixed-elastic-split.md), so the two partly compensate in $\langle\cos\chi\rangle$. Output-level agreement can therefore hide both. Thinning resolves both.

## Deviations judged

**$C_2$ truncates the step instead of raising the hard rate (Eq. 4.85).** Acceptable. Delta rows keep the optical depth, each truncated step has its own hinge with the correct moments, and a shorter step is closer to the exact law. Reducing $C_2$ moves the Lewis deviation toward zero: Si 30 keV, $C_1=0.2$, from $-3.29$ % at $C_2=0.2$ to $-1.05$ % at $C_2=0.02$. The cost is that the hard elastic rate stays at $1/(C_1\lambda_{\rm el,1})$ where PENELOPE would raise it, so tails above $\mu_c$ are not better resolved in the $C_2$-limited regime, where only two moments are preserved. Fig. 4.3 places that regime at about 10 MeV and above. In shell mode `mix_stopping` is the soft stopping only, whereas $S$ in Eq. 4.85 is the total. That is the right control for a limit on soft loss along a step, and it matches the parameter's docstring, but it is a further difference from Eq. 4.85.

**Default step limit.** Acceptable. PENELOPE's user documentation gives one tenth of the thickness of a thin body, and the `pencyl` default is `DSMAX` = 0.1 times the cylinder thickness. §4.4.1 states one tenth of the expected track length. The in-code comment cites only §4.4.1. For a thick layer the limit is inert and the $4\lambda^{(h)}_T$ and $C_2$ caps govern, which keeps more than about ten hinges per track.

**$4\lambda^{(h)}_T$.** Matches §4.4. The planned value uses the rate at the step's start, as $\Sigma_{h,\max}$ does in spirit but without its upper-bound property. That is the root of the discrepancy.

**Soft inelastic deflection (§4.3.2) omitted.** PENELOPE's hinge uses $\lambda^{(s)}_{\rm comb}$, including soft inelastic deflection. Here the hinge carries only elastic soft moments, plus any `atomic_electron_deflection` rate scaling. This matches the detailed core's physics and the ledger's scope.

## Limits of this check

- Not measured in the Lewis test: compounds, shell or radiative cores, straggling, energies above 1 MeV, and the $C_2$-limited regime above 10 MeV.
- No comparison against PENELOPE itself.
- The exact-Lewis harness monkeypatches the entry position, so it does not exercise surfaces. Interface behaviour was checked only by reading the code against Eq. 4.23.
