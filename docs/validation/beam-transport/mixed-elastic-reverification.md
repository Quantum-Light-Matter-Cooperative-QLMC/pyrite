# Mixed elastic thinning re-verification

## Scope and independence

Claims: `mixed-elastic-split` and `mixed-elastic-hinge`, at HEAD `e25dede8b`, branch `issue-894-mixed-elastic-transport`. Fresh context that did not write the code.

Source: PENELOPE-2024, NEA/MBDAV/R(2024)1, read from the local Zotero PDF (`CQ6Q2F87`): §4.3.1 Eqs. 4.85-4.98, §4.4 steps (i)-(xiii), Eqs. 4.120-4.121, §4.4.1.

Order of work: the two earlier verification pages and the ledger rows were read first, as the task required, so the derivation knew the earlier failure and the named fix. The derivation below was then recorded before `mixed_elastic.py`, `cores.py` or the design note's body was opened. Only the Source URL line of the note had been read.

References: all rates, moments, integrals and bounds in the harness come from raw released ELSEPA tables through `scratch/reverify/lib.py`, which does not call the mixed tables or packers. Node-level agreement with the implementation is to about $10^{-15}$. Inelastic and radiative true rates are the core's own tables (`hard_rate_per_ang`, `radiative_layer_moments_scalar`), so those rates themselves are not independently checked here.

The harness and logs were scratch files and are not committed. Paths under `scratch/reverify/` below name them for the record only.

## Independent derivation (recorded before reading the implementation or the design note)

Source read: PENELOPE-2024, NEA/MBDAV/R(2024)1, local Zotero PDF, §4.3.1 Eqs. 4.86-4.98, §4.4 steps (i)-(xiii), Eq. 4.120-4.121.

Let $E(t)$ be the track energy at path $t$, $\mathcal F_t$ the history, $\Sigma_j(E)$ the hard rate of process $j$.

**R1, thinning theorem.** Let $M_j(t)$ be $\mathcal F_{t^-}$-predictable and proposals Poisson with intensity $M_j$. Accepting with probability $\min(1,\Sigma_j(E(t))/M_j(t))$ gives an accepted intensity $\min(\Sigma_j,M_j)$. So the accepted process has intensity $\Sigma_j$ iff $M_j\ge\Sigma_j(E(t))$ along the track (Eq. 4.97), and otherwise it loses exactly $\int(\Sigma_j-M_j)_+dt$ events. The count satisfies $\mathbb E[N_j]=\mathbb E\int\Sigma_j(E(t))dt$ up to that deficit.

**R2, clocks.** Write the unit-exponential clock $\tau_j$ in optical depth, consumed at $\dot\tau_j=-M_j(t)$. Then a bound may change at any stopping time (any step start), provided the clock is stored in optical depth, or in a distance rescaled by $M_{\rm old}/M_{\rm new}$. A surviving *distance* carried across a bound change, or a clock not decremented on a non-firing step, is wrong. By memorylessness, a redraw at a stopping time has the same law, so "rejected proposals restart only their own clock" is required only as a bookkeeping statement: resetting others at an event is statistically harmless, but losing consumption or conditioning a redraw on the future is not. Only a proposal at $\tau_j=0$ restarts clock $j$.

**R3, one energy.** Steps (iii-1) and (xii) evaluate acceptance at the energy after soft loss on the step. The partition argument of Eqs. 4.35-4.37 closes only if the rate $\Sigma_h$, the element choice $\Sigma_{h,i}/\Sigma_h$, and the quantile $f_{h,i}$ come from the same $E$ and panel. The same holds for inelastic channel and radiative quantities, because Eq. 4.121 draws from one $E$. Acceptance at $E_a$ with a quantile at $E_q$ in another panel breaks closure at the node.

**R4, bound and window.** Eq. 4.97 needs a true sup on $(E_0-\omega_{\max},E_0)$, with $\omega_{\max}$ a real upper bound on the soft loss (Eq. 4.96: "provided only that an upper bound exists for the energy lost"). For a frozen-panel $f_h$, $\Sigma(E)f_{h,k}$ is discontinuous at nodes, so the sup must include both one-sided limits at a node, $\Sigma(E_k)f_{h,k}$ and $\Sigma(E_k)f_{h,k-1}$, for every node in the window, plus the window ends, with the *same* interpolation as the rate. For a compound, the sum of elemental piecewise functions needs all elemental knots. Under CSDA a step capped at $s\le C_2E/S(E)$ loses $\int_0^sS\,ds'$, which is below $1.5C_2E$ iff the average of $S$ on the step is below $1.5S(E)$. Under an untruncated stochastic straggling law there is no upper bound, so Eq. 4.97 is violated in the tail. The size of the effect follows R1: the deficit is $\int(\Sigma-M)_+$ conditional on the tail. The radiative rate $\Sigma_{\rm rad}(E)$ vanishes at $W_{cr}$ and has an interior maximum (a $1/\beta^2$ decline against a rising threshold term), so an endpoint-only bound fails when a window straddles that maximum.

**R5, step and hinge.** PENELOPE step: $s=\min(-\ln\xi/\Sigma_{h,\max},s_{\max})$ fixed at start, $\tau=\xi s$, moments at $E(\tau)$ with full $s$ (steps iii, v, viii). To first order $\mathbb E_\tau[s/\lambda(E(\tau))]=\int_0^s dt/\lambda(E(t))$ exactly, which justifies the hinge energy. So the requirement is: realized length equals planned $s$ whenever no surface or cutoff intervenes. Each hinge has mean $1-\cos=1-e^{-x}$, $x=s/\lambda_{s,1}(E_\tau)$. For $x$ exponential with mean $\bar x$ and no caps, the ratio $\sum(1-\cos)/\int\lambda_{s,1}^{-1}ds$ is $1/(1+\bar x)$, in general $\mathbb E[1-e^{-x}]/\mathbb E[x]$. The correct test is $\sum_{\rm hinges} x_i$ against $\int\lambda_{s,1}^{-1}$ (ratio 1), with $\sum(1-\cos)$ against $\sum(1-e^{-x_i})$ (ratio 1).

**R6, Lewis target.** Soft and hard both exact at fixed $E$ give $\langle P_\ell\rangle(S)=\exp[-\int_0^S dt/\lambda_{{\rm el},\ell}(E(t))]$ for a pure elastic + CSDA medium, with residual only from (i) the last truncated step: at path $t<s$ the hinge gives $1-(t/s)(1-e^{-x})$ against $e^{-xt/s}$, an $O(x^2)$ local error, one per track; and (ii) moments frozen at hinge energy, $O((s\,d\ln\lambda/ds)^2)$. Both are second order in the step and independent of node crossings. A node crossing must leave no first-order feature.

**R7, $C_1=0$.** $f_h=1$, no soft rates, no draws beyond detailed; acceptance must be skipped or use no draws, since an extra accept/reject draw breaks the stream. No thinning proposals may appear as rows. Non-mixed paths must be bitwise equal to the pre-move code.

## Verdict

| Claim | Verdict | First divergent term in the claim's equations |
|---|---|---|
| `mixed-elastic-split` | **`rederived`**, with the scope limit below | none |
| `mixed-elastic-hinge` | **`rederived`**, with the scope limit below | none |

Both earlier causes are gone. Hard-event count against its exact integral moved from 0.98 and 0.94 to 1 within 0.05%. Realized step equals planned step in every traced step.

The scope limit is a judgement call the lead may want to review. Under Urban straggling, $\omega$ has no upper bound, so $E_{\rm low}=E(1-\tfrac32C_2)$ is not the $\omega_{\max}$ of Eqs. 4.96-4.97. That departure is disclosed in the row's Assumptions and has now been measured: it lowers the hard elastic count by 0.07-0.32%. Separately, mixed track path under Urban grows with row length, by +0.3% to +1.3%. I did not call either a discrepancy in these claims, because the first is the ledgered clamp and the second lies in the energy-loss rows, not the split or the hinge. The combination `transport_mode="mixed"` with Urban straggling should not be treated as validated until they are resolved or accepted. CSDA and shell soft-sampler modes at $C_2\ge0.05$ have neither effect at this precision.

## Source-to-code comparison

| Requirement | Code | Result |
|---|---|---|
| R1 proposal rate $M_j$, accept $\Sigma_j(E)/M_j$ | `cores.py` plan block, `_mixed_hard_element`, the `target >= mu_event` branches | matches |
| R2 clocks in optical depth, consumed at $M_j$ | `tau_left -= step_j/lam_ang`, `tau_h -= step_j*mu_hard`, `tau_rad -= step_j*mu_rad`, with `lam_ang`, `mu_hard`, `mu_rad` overwritten by the step's bounds | matches, so a bound change is safe |
| R2 restart | delta with `mix_end` restarts only that clock; limit delta and hinge restart none; hard event and surface restart all (memoryless, so harmless) | matches |
| R3 one energy | elastic: acceptance, element, quantile all at `E_end_j`. Inelastic: `il_lo, il_f` recomputed at `E_end_j`, channel and transfer from it. Radiative: `mu_event` and `rad_scratch_rates` at `E_end_j`, photon from them | matches |
| R4 elastic sup | window ends plus $\Sigma(E_k)\max(f_{h,k-1},f_{h,k})$ at interior nodes; element bounds add | matches, exact on a single element |
| R4 inelastic sup | window ends plus knots, linear in $\ln E$ like the rate | matches, exact |
| R4 radiative sup | $\max(\mu(E_j),\mu(E_{\rm low}))$ | not a sup; size measured below |
| R5 step | $s=\min(\tau_j/M_j,\ \text{limit},\ C_2E/S,\ 4/\sum M_j)$ fixed at start; $\tau=\xi s$; hinge at row-end energy with `mix_length` | matches §4.4 (iii), (v), (viii) |
| R5 surface | hinge dropped, all clocks reset, new step in the new layer | matches (vi), Eq. 4.23 |
| R7 $C_1=0$ | $f_h=1$, soft rate 0, plan block never entered | matches, no extra draws |

Units traced: elastic bound [1/cm] converted through `1e8` to Angstrom; inelastic and radiative bounds [1/Angstrom]; `hard_total` [1/cm]; cap [Angstrom]; `rad_soft*1e-3` eV to keV. The margin comment checks out: $1-\sqrt{1-2C_2}=0.225$ at $C_2=0.2$. The $4\lambda_T^{(h)}$ cap uses bounds, where §4.4 uses $\lambda_T^{(h)}$. That is conservative, and it is the quantity Eq. 4.120 samples.

## Discrepancies

None in the claimed equations. Measured limits and other findings follow.

### Finding 1: clamp under Urban straggling (Question 5)

Urban's tail reaches $T_{\max}=E/2$, so a row can leave the window, and the true rate then exceeds the bound. To isolate the clamp, I reran the same seeds in a scratch copy of HEAD with `MIXED_STEP_LOSS_MARGIN = 1e6`, which opens the window to the cutoff.

| Case, Urban | Steps outside window | Count ratio, HEAD | Count ratio, open window | Clamp effect |
|---|---|---|---|---|
| Si 30 to 24 keV, $C=0.05$, $N=10^6$, seed 6001 | not recorded | $0.99812\pm0.00022$ | $0.99877\pm0.00022$ | $-0.065$% |
| Si 30 to 20 keV, $C=0.05$, $N=6\times10^5$, seed 6002 | 1.42% | $0.99817\pm0.00022$ | $0.99942\pm0.00022$ | $-0.125$% |
| Si 30 to 20 keV, $C=0.05$, $N=2\times10^5$, seed 245 | 1.42% | $0.9981\pm0.0004$ | $0.9997\pm0.0004$ | $-0.16$% |
| Si 10 to 2 MeV, $C=0.2$, $N=3\times10^5$, seed 244 | 1.95% | $0.9969\pm0.0007$ | $1.0001\pm0.0007$ | $-0.32$% |

The time-rescaled hazard agrees: at 10 MeV $H(2)=1.991$ ($z=-5.2$) on HEAD and $2.003$ ($z=+1.7$) with the window open. Rows that overrun the window lose up to $0.50E$. The open-window run carries about 2.7 times the delta rows at 10 MeV, so a wider fixed window costs speed.

With the window open, about 0.02 events per track remain missing at both path lengths (0.025 and 0.021). That scales as a per-track end effect, not a rate error. My hypothesis is that a row crossing the cutoff drops the proposal at its end while its truncated path still counts in the reference. The "no final row" estimator I tried is biased high and does not test this, so the hypothesis is unproven.

Angular effect under Urban, Si 30 to 24 keV, against $\mathbb E[e^{-\int ds/\lambda_\ell}]$ per track: $1-\langle\cos\chi\rangle$ gives $+0.079\pm0.085$% and $-0.047\pm0.086$%; $1-\langle P_2\rangle$ gives $+0.023\pm0.055$% and $-0.159\pm0.055$% (seeds 6001 and 5006). At 30 to 20 keV: $-0.184\pm0.092$% and $-0.105\pm0.062$%. The clamp is not resolved in these angular moments.

### Finding 2: mixed path grows with row length under Urban (new, outside the listed questions)

Mean path to the cutoff, mixed against detailed:

| Mode | Case | $C=0.05$ | $C=0.2$ |
|---|---|---|---|
| CSDA | Si 30 to 24 keV | $+0.003$% | $+0.029$% |
| CSDA | Si 200 to 100 keV | $+0.003$% | $+0.025$% |
| Urban | Si 30 to 24 keV | $+0.117\pm0.166$%; separate higher-statistics run $+0.32\pm0.07$% ($N=10^6$ mixed, seed 6001; $2\times10^5$ detailed, seed 6003) | $+0.855\pm0.165$% |
| Urban | Si 200 to 100 keV | $+0.600\pm0.244$% | $+1.269\pm0.244$% |
| Urban, `max_dE_frac` = 0.002 | Si 30 to 24 keV | $+0.057\pm0.167$% | $+0.005\pm0.167$% |
| Urban, `max_dE_frac` = 0.002 | Si 200 to 100 keV | $+0.412\pm0.245$% | $+0.263\pm0.245$% |
| Shell + soft sampler | Si 30 to 24 keV | $-0.063\pm0.173$% | $+0.157\pm0.173$% |
| Shell + soft sampler | Si 200 to 100 keV | $+0.224\pm0.248$% | $+0.194\pm0.248$% |

Seeds 7100 (detailed) and 7205, 7220 (mixed). $N=40000$ detailed and $200000$ mixed at 30 keV; $8000$ and $40000$ at 200 keV. The effect grows with $C$ and shrinks with substeps. My leading hypothesis is that straggled rows use the stopping frozen at the row start, without the midpoint correction or PENELOPE's Eq. 4.65 energy dependence, and that mixed rows are 3 to 45 times longer. I did not isolate the mechanism. Shell mode shows nothing at this $\pm0.2$% precision, but that does not exclude an effect below it.

### Finding 3: when the bound fails, selection truncates instead of rescaling

Above the bound, `target` is uniform on $(0,M)$ while the cumulative sum runs to $\Sigma>M$. The clamp therefore also removes the end of the element list in `_mixed_hard_element` and the end of the channel list in `_sample_hard_collision`, and likewise the element in `_sample_hard_photon`. So element and channel fractions are distorted in list order, by the fraction $(\Sigma-M)/\Sigma$ on those events. This comes from reading the code; I did not measure it. It affects only clamped events, which do not occur in shell or CSDA modes at defaults. Dividing `target` by $\max(1,\Sigma/M)$ before selection would remove it.

### Smaller findings

- **Inherited lag in fully detailed panels.** A step starting where $f_h=1$ everywhere has no soft part and takes the unthinned row-start hazard, exactly like detailed transport. Au 5 to 1 keV, $C=0.2$, $N=50000$, seed 301: total $0.9970\pm0.0005$, and 1 to 1.5 keV $0.988\pm0.0012$; detailed gives $0.9920\pm0.0004$ total and $0.9857\pm0.0012$ in the same bin (seed 341). `max_dE_frac` = 0.002 removes it in both. It is not specific to mixed transport, but "thinned" does not describe these rows.
- **Corner with low statistics.** Shell mode at $C_2=0.005$ and $W_{cc}\ge1$ keV: 7-27% of steps leave the window; counts $0.9921\pm0.0037$, $0.9958\pm0.0038$, $0.9914\pm0.0047$, which pool to about $-0.66\pm0.23$%. That exceeds the integrated clamp estimate of 0.12-0.28%. Not explained; not a default configuration.
- **Possible small count deficit at $C=0.2$.** Si counts at $C_1=0.2$ below 30 keV across six runs pool to $0.99952\pm0.00019$, about $-2.5\sigma$. SiO2 ($0.99993\pm0.00045$) and other energies ($1.0009\pm0.0011$ at 30 to 20 keV, $0.9993\pm0.0005$ at 1 MeV, $1.0022\pm0.0015$ for W at 1 MeV) do not show a common sign. I could not resolve this further; it is below the precision asked for.
- **Earlier page.** `mixed-elastic-split.md` gives "about 15.4 events" for Si 30 to 24 keV. The same core and harness replayed give 18.41, and the exact integral is 18.73. Its measured ratios reproduce, so this looks like a wrong figure on the page.
- **Stale docstring.** `mixed_elastic_tables` still says the lower node's first moment is the larger one; the earlier page showed that is false for heavy elements below about 1 keV.
- **Ledger.** `_plan_mixed_step` carries `Validation: mixed-elastic-hinge` but is not in the row's `Code` list. The coverage test does not catch this direction.
- **Identity marker.** `MIXED_ELASTIC_MODEL` stayed `penelope-random-hinge.v1` although mixed output changed. `_mixed_elastic_policy.py` is not on `main`, so no released identity is affected; the rule applies if results were already recorded under it.
- **Urban against CSDA path.** Detailed Urban path is 10% above CSDA on the same tables at 30 to 24 keV (35025 against 31842 Angstrom). Non-final rows average 1.58 keV/um against 1.88. This is expected if a wide loss tail is capped by the remaining energy, and mixed transport plays no part. I did not confirm it beyond that.

## Measurements

Real per-electron core, released ELSEPA tables, Si 0.04996 and W 0.06322 atoms/Angstrom$^3$. Tracks start mid-slab through `host_pass.initial_beam_positions`, in a $10^9$ Angstrom slab, unless a surface is the subject. Reference integrals use each row's realized path with $E$ linear in the row and the panel jump resolved exactly. Uncertainties are 1 sigma.

### Question 1: hard-event count against the exact integral

| Si 30 to 24 keV | Earlier page (row-start) | Old core, this harness | HEAD |
|---|---|---|---|
| $C_1=C_2=0.05$ | $0.9809\pm0.0007$ | $0.98263\pm0.00052$ | $0.99967\pm0.00052$ |
| $C_2=0.005$ | $0.9947\pm0.0007$ | not rerun | $1.00154\pm0.00073$ |
| `max_dE_frac` = 0.002 | $0.9956\pm0.0007$ | not rerun | $0.99998\pm0.00073$ |
| $C_1=C_2=0.2$ | $0.9376\pm0.0015$ | $0.93800\pm0.00074$ | $0.99935\pm0.00073$ |
| $C=0.2$, `max_dE_frac` = 0.002 | not on page | not rerun | $0.99965\pm0.00103$ |

$N$ and seed: $2\times10^5$/101, $10^5$/121, $10^5$/131, $4\times10^5$/141, $2\times10^5$/151. Replaying the old core (`aaa66fe4a`) reproduces the page, so the harness can see the old fault. The $C_2=0.005$ row is $+2.1\sigma$.

Path $0.3\lambda_1$: W 28 keV $0.99958\pm0.00028$; W 25.2 keV $1.00016\pm0.00027$ (old core $0.97827\pm0.00028$); W 30 keV $1.00007\pm0.00015$; Si 30 keV $0.99996\pm0.00012$ (old core $0.98091\pm0.00027$).

The time-rescaling test, which is stronger than a count, gives Nelson-Aalen $H(\Lambda)=\Lambda$ with $|z|\le2.2$ at $\Lambda=0.25$-3 in CSDA Si and W from 30 keV to 10 MeV. Au 5 to 1 keV fails this test ($z=-6.5$ at $\Lambda=3$) because of the inherited lag under Smaller findings.

Other modes at $C_2\ge0.05$:

| Mode | Case | Elastic | Inelastic | Radiative |
|---|---|---|---|---|
| Shell + soft | Si 30 to 10 keV, $C=0.05$, $N=10^4$ | $0.9993\pm0.0010$ | $0.9998\pm0.0014$ | |
| Shell + soft | Si 30 to 10 keV, $C=0.2$, $N=10^4$ | $1.0014\pm0.0021$ | $0.9987\pm0.0014$ | |
| Shell + soft | Si 200 to 100 keV, $C=0.2$, $N=3000$ | $1.0031\pm0.0051$ | $0.9990\pm0.0012$ | |
| Shell + soft | W 100 to 30 keV, $C=0.2$, $N=2000$ | $0.9989\pm0.0019$ | $0.9993\pm0.0013$ | |
| Radiative | W 5 to 0.5 MeV, $C=0.2$, $N=2000$ | $0.9973\pm0.0021$ | | $0.9890\pm0.0102$ |
| Radiative + shell + soft | W 1 to 0.1 MeV, $C=0.2$, $N=500$ | $0.9961\pm0.0030$ | $1.0005\pm0.0008$ | $0.9909\pm0.0407$ |

Absorbing hard collisions are recorded as `EVENT_CUTOFF` with a nonzero transfer and are counted here.

### Question 2: Lewis moments at path $0.3\lambda_1$

Reference is $\exp[-\int ds/\lambda_\ell(E(s))]$. "Scheme" is `scratch/reverify/s8_toy.py`, which contains no core code: it samples only $s$ and $\tau$ from the algorithm and takes the expectation over acceptance and the hinge law, so it predicts what a correct implementation should give.

| Case | Earlier page $1-\langle\cos\rangle$; $1-\langle P_2\rangle$ (%) | HEAD $1-\langle\cos\rangle$ (%) | HEAD $1-\langle P_2\rangle$ (%) | Scheme (%) |
|---|---|---|---|---|
| W 28 keV, 0.05 | $-0.30$; $-0.16$ | $+0.159\pm0.107$ | $+0.072\pm0.071$ | $+0.050$; $+0.051$ |
| W 25.2 keV, 0.05 | $-1.91$; $-1.61$ | $-0.114\pm0.109$ | $-0.092\pm0.072$ | $+0.035$; $+0.028$ |
| W 30 keV, 0.05 | $-1.38$; $-1.16$ | $+0.054\pm0.054$ | $+0.019\pm0.035$ | $+0.022$; $+0.021$ |
| Si 30 keV, 0.05 | $-1.34$; $-0.97$ | $+0.047\pm0.036$ | $+0.019\pm0.024$ | $+0.038$; $+0.028$ |
| Si 30 keV, 0.2 | $-3.29$; $-3.41$ | $-0.351\pm0.051$ | $-1.014\pm0.034$ | $-0.298\pm0.022$; $-0.937\pm0.017$ |
| W 30 keV, 0.2 | not on page | $+0.149\pm0.107$ | $-0.101\pm0.071$ | $+0.060$; $-0.044$ |
| Si, $C_1=0.2$, `max_step_um` = 0.13 | | $-0.020\pm0.061$ | $-0.021\pm0.041$ | $+0.007$; $+0.007$ |
| Si, $C_1=0.2$, $C_2=0.02$ | | $-0.022\pm0.061$ | $-0.015\pm0.041$ | |
| SiO2 30 to 27 keV, 0.05 | | $+0.068\pm0.070$ | $+0.050\pm0.047$ | |
| SiO2 30 to 27 keV, 0.2 | | $-0.300\pm0.071$ | $-1.022\pm0.047$ | |

$N=2\times10^6$ per run, plus $10^7$ (Si 0.05), $6\times10^6$ (W 30, 0.05) and $4\times10^6$ (Si 0.2) pooled where marked by the smaller error. Seeds 1257-1320 and 5001-5008. Cutoffs chosen so $\int ds/\lambda_1=0.300$: W 27.541, 24.779, 29.512; Si 27.718 keV.

Detailed controls, $N=2\times10^6$, seed 3030: Si $-0.076\pm0.086$, $-0.066\pm0.058$; W $+0.186\pm0.107$, $+0.112\pm0.071$. SiO2, $N=10^6$: $-0.161\pm0.121$, $-0.088\pm0.082$.

So at $C=0.05$ the moments match to about 0.04-0.1%, better than the 0.2% asked for, and the node-crossing signature is gone: W 25.2 keV differs from W 28 keV by $-0.27\pm0.15$% on $1-\langle\cos\chi\rangle$, where the earlier page had $-1.61$%.

At $C_1=C_2=0.2$ the $-1.0$% on $1-\langle P_2\rangle$ is real and exceeds 0.2%, but it belongs to the scheme, not the code. This path holds only 1.7 hard events and few hinges. Each track ends inside a step, where the hinge law gives $1-(t/s)(1-e^{-y})$ against the exact $e^{-yt/s}$, an error $-\tfrac12y^2(t/s)(1-t/s)$ that is second order in the step. Three things support this: the code-free scheme predicts it ($-0.94$%, core within $2.1\sigma$); using the exact law on the last partial step in the scheme moves it to $+0.21\pm0.05$%; and limiting the step so the path holds about 10 hinges removes it in the core ($-0.021\pm0.041$%), which is PENELOPE's own §4.4 recipe. At 30 to 24 keV it shrinks to $-0.212\pm0.085$% (scheme $-0.102$%). A check at a surface would need the same recipe, and `layer_step_limits_ang` supplies it there.

### Question 3: realized step and soft first moment

Pure-Python trace of the core's own locals, 7 runs, 51041 rows, 21084 steps, covering elastic, substeps, thin slab, shell with straggling, and radiative:

| Check | Pass | Fail |
|---|---|---|
| Plan equals my $\min(\tau_j/M_j,\text{caps})$ | 21084 | 0 |
| Elastic bound equals my knot sup | 21084 | 0 |
| Row consumption of all three clocks at the step's bounds | 51041 | 0 |
| Realized length equals planned, step ending in a proposal or limit | 20615 | 0 |

The other 469 steps were truncated by the cutoff (390) or a surface (79).

| Soft first moment / exact integral | Earlier page | Old core replay | HEAD |
|---|---|---|---|
| $C=0.05$ | 1.0049 | $1.0035\pm0.0007$ | $0.99556\pm0.00074$ |
| $C=0.05$, `max_dE_frac` = 0.002 | 0.9955 | not rerun | $0.99682\pm0.00113$ |
| $C=0.2$ | 0.9772 | $0.9774\pm0.0010$ | $0.96587\pm0.00088$ |
| $C=0.2$, `max_dE_frac` = 0.002 | 0.9501 | not rerun | $0.96585\pm0.00125$ |

The ratio no longer depends on row length, which was the signature of the fault. Its target is $\mathbb E[1-e^{-x}]/\mathbb E[x]$, not the earlier page's rough $1-\bar x$ (0.942). On complete steps the direct tests give:

| Quantity | $C=0.05$ | $C=0.2$ | $C=0.2$, substeps |
|---|---|---|---|
| $\sum x_i/\int\lambda_{s,1}^{-1}ds$ | $0.99998\pm0.00002$ | $0.99977\pm0.00004$ | $0.99997\pm0.00006$ |
| $\sum(1-\cos\theta)/\sum(1-e^{-x_i})$ | $0.9998\pm0.0008$ | $1.0009\pm0.0009$ | $1.0009\pm0.0013$ |
| $\langle\tau/s\rangle$ | $0.5000\pm0.0001$ | $0.4999\pm0.0002$ | $0.5001\pm0.0003$ |

The 0.02% at $C=0.2$ is second order and disappears with substeps. The whole-track ratio sits 0.003 under the complete-step expectation at $C=0.2$, which fits a length-biased final step (my estimate, not separately tested). The old core shows $\langle\tau/s\rangle=0.5121$ on the same test, because a hard event could cut the step.

### Question 4: clocks and one energy

Same trace. Transition from a row's end to the next row's start:

| Transition | Rows | Own clock | Others | Fail |
|---|---|---|---|---|
| Hinge or substep | 29957 | kept | kept | 0 |
| Limit delta | 656 | none restarted | kept | 0 |
| Rejected elastic | 2197 | restarted | kept | 0 |
| Rejected inelastic | 647 | restarted | kept | 0 |
| Rejected radiative | 1 | restarted | kept | 0 |
| Accepted hard event (13040 elastic, 4048 inelastic, 26 radiative) | 17114 | all restarted | | 0 |

One energy: 15237 elastic proposals recomputed with the stream's own uniform and my rate at `E_end_j` agree on accept and reject (13040 accepted, 2197 rejected), with element and quantile at the same energy. All 4072 inelastic events read `il_lo, il_f` at $\ln$`E_end_j`, and `mu_event` equals the rate there. Radiative: 27 accepted events, with `target < mu_event`. Element interval widths in `_mixed_hard_element` match my rate ratios on SiO2 at 50 energies, to the $2\times10^{-4}$ grid.

Radiative rejection is thin evidence: one traced rejection, plus reading the code.

### Question 5: are the bounds true upper bounds?

| Process | Result |
|---|---|
| Elastic, 6 materials including SiO2 and WC, $C_1=0.01,0.05,0.2$, 1500 random windows each | maximum scan/bound $-1$ is $4\times10^{-15}$; matches my knot sup to $4\times10^{-15}$. Tight except WC, slack 5.5% from adding element sups |
| Inelastic, Si and W, $W_{cc}=50$ eV, 1 keV, 2000 windows each | $2\times10^{-16}$ |
| Radiative, Si, W, Au, $W_{cr}=10$ eV-10 keV, exact rate | exceeds the endpoint bound in up to 1.9% of windows at $C_2=0.05$, by at most $8.9\times10^{-4}$; up to 4.3% at $C_2=0.2$, by at most 1.02% (Si, $W_{cr}=1$ keV, $E=4.5$ keV) |
| Radiative, table mode | adds a table-against-exact excess of about $10^{-5}$-$3\times10^{-5}$ in 72-100% of windows, within the table's stated $2\times10^{-4}$ tolerance. Si at $W_{cr}=10$ eV has no exact-rate excess |

The largest radiative exceedances sit at $E\approx$ 0.7-54 keV, about 3.5-7 $W_{cr}$ for $W_{cr}\ge1$ keV. On 5 radiative track runs no step-end rate exceeded the endpoint bound. The effect is at most 1% on an already small rate near threshold. Negligible.

Leaving the energy window:

| Mode | Case | Steps outside | Worst overrun | Above bound |
|---|---|---|---|---|
| CSDA | Si, W 30 keV-10 MeV | 0 | | 0 |
| CSDA | Au 5 to 1 keV, $C=0.2$ | $1.25\times10^{-6}$ | $0.056E$ | 7 steps, deficit $3\times10^{-8}$ |
| Shell + soft | all $C_2\ge0.05$ runs, including $W_{cc}$ = 1, 5 keV | 0 | | 0 |
| Shell + soft | $C_2=0.005$, $W_{cc}\ge1$ keV | 7-27% | up to $0.22E$ | see Smaller findings |
| Radiative | all 5 runs | 0 | | 0 |
| Urban | Si 30 keV, $C=0.2$ | 0.098% | $0.033E$ | deficit $3\times10^{-6}$ |
| Urban | Si 1 MeV, $C=0.2$ | 0.50% | $0.40E$ | deficit about $10^{-3}$ |
| Urban | Si 30 keV, $C=0.05$ | 1.42% | $0.26E$ | see Finding 1 |
| Urban | Si 10 MeV, $C=0.2$ | 1.95% | $0.50E$ | see Finding 1 |
| Urban | Au 2 to 0.5 keV, $C=0.2$ | 2.4% | $0.45E$ | count $0.9868\pm0.0007$, including the inherited lag; the split between them was not measured |

The shell soft sampler stays inside the window at defaults because its Gaussian case truncates at $3\sigma$, which bounds $\omega$ as Eq. 4.96 asks, and $\sigma$ is small with $W_{cc}$ = 50 eV. Its case III top $3(\sigma^2+\langle\omega\rangle^2)/(2\langle\omega\rangle)$ does not scale with the step, which is why small $C_2$ with large $W_{cc}$ leaves it.

### Question 6: $C_1=0$ and the moved samplers

SHA-256 over all result arrays, pre-fix `aaa66fe4a` against HEAD, 400 electrons, seed 77, Si 30 to 5 keV:

| Specializations | Result |
|---|---|
| Per-electron: ELSEPA, SR, Urban, shell + soft, radiative table, radiative exact, shell + radiative | identical across trees |
| Lockstep: ELSEPA, SR, shell + radiative | identical across trees |
| $C_1=0$ against detailed, 6 per-electron specializations | identical, on both trees |

### Surfaces

Thin slabs against detailed: the earlier harness did not cover these.

| Slab | $C$ | $N$ each | $T$ ($z$) | $B$ ($z$) | $\langle\cos\mid T\rangle$ ($z$) | $\langle E\mid T\rangle$ ($z$) |
|---|---|---|---|---|---|---|
| Si 3 um | 0.05 | $10^6$ | $-1.2$ | $-0.5$ | $-0.5$ | $-0.9$ |
| Si 3 um | 0.2 | $10^6$ | $-0.9$ | $-1.1$ | $+0.2$ | $-0.2$ |
| Si 6 um | 0.05 | $6\times10^5$ | $-1.1$ | $-0.5$ | $0.0$ | $-0.7$ |
| W 0.15 um | 0.05 | $2\times10^5$ | $+1.6$ | $-1.6$ | $+0.2$ | $+0.1$ |
| W 0.15 um | 0.2 | $2\times10^5$ | $+1.5$ | $-1.4$ | $-1.1$ | $-0.4$ |
| W 0.6 um | 0.05 | $10^5$ | $+0.6$ | $-0.3$ | $+0.1$ | $+0.8$ |
| W 0.6 um | 0.2 | $10^5$ | $-0.9$ | $-0.4$ | $-0.9$ | $+0.8$ |

A $z=-2.4$ on $\langle\cos\mid T\rangle$ in an earlier, smaller Si 3 um run ($10^5$ detailed) did not survive the larger sample. Si 6 um at $C=0.2$ did not finish before its timeout, and Si 1 um was run only at $3\times10^5$/$10^5$ ($|z|\le1.6$).

## Deviations judged

- **Elastic bound over a window, not PENELOPE's single $\Sigma_{h,\max}$ with $s_{\max}$.** Acceptable. Step-start bounds are predictable, and clocks in optical depth survive their change.
- **Separate clocks per process, against Eq. 4.121's one clock with point probabilities.** Equivalent. Superposed independent thinned processes have the same law.
- **Hard event at step end, elastic acceptance and element from one uniform.** Matches Eq. 4.121 in spirit and the claim as written.
- **Bound window not a true $\omega_{\max}$ under Urban.** Departs from Eqs. 4.96-4.97; disclosed; measured in Finding 1. Accept only with a stated limit.
- **Earlier deviations (per-element cutoff, frozen panel, $C_2$ truncating the step, default limit, soft inelastic deflection).** Not reopened; the earlier judgements stand. The frozen panel's price, a hazard discontinuity, is now resolved by the bound.

## Filter checklist

| Item | Result |
|---|---|
| Notation across ledger, docstrings, note | pass; ledger `Code` omits `_plan_mixed_step` |
| Units at each boundary | pass |
| Reference data | Eqs. 4.96-4.98, 4.120-4.121, §4.4 $4\lambda_T^{(h)}$ (about 2% of steps exceed, $e^{-4}=1.8$%) checked against the PDF |
| Tolerances | ledger anchors use 4 sigma; not tuned as far as I can see |
| Cited evidence | earlier page ratios reproduce on the old core, except the "15.4 events" figure |
| Tests | `tests/dev/test_validation_ledger.py` and `tests/montecarlo/collisions/test_mixed_elastic.py`: 73 passed |

## Limits of this check

- No comparison against PENELOPE itself, and its Fortran was not read.
- Inelastic and radiative true rates are the core's tables; only the thinning around them is tested. Radiative counts are low power (1-8% error), and only one radiative rejection was traced.
- Not tested: multilayer steps across real layer changes beyond the thin-slab exits, compounds with straggling, positrons, secondaries, energies above 10 MeV for counts, GPU (the core is CPU only). Bitwise digests do not cover Mott (no NIST table here), grooved or LUT cores: `transport_core="auto"` with SR matched lockstep, so the LUT was not exercised.
- Under Urban, $E(t)$ inside a row is not defined. The reference assumes $E$ linear in the row, so small integrals there carry that ambiguity; the paired open-window comparison is the reliable measure.
- Pure-Python traces are small samples by design, and they stub `stabilize_closure_cache` in the harness because the core does not import under `NUMBA_DISABLE_JIT=1` otherwise. That mode cannot see JIT-only differences; the JIT runs above cover the same paths statistically.
- Finding 2's mechanism and the 0.02 events per track residual are hypotheses, not demonstrated.
- Two harness errors were found and fixed during the work and their output discarded: zero-loss rows were first given zero path in the reference, which gave nonsense ratios under straggling; and the scheme first treated capped steps as proposals. The first `pgrep`-based waits also returned early, with no effect on results.
- Harness and logs: `scratch/reverify/`. The old-core and open-window trees and the extra environment were deleted after use; their output is in `old.log`, `bit_old.json`, `margin.log`, `margin_s1.log`, `urban_open.log`.
