# Line-scoring roulette

```{admonition} Validation status
:class: note
Ledger claim [`line-scoring-roulette`](../ledger-transport-background.md#line-scoring-roulette). This page derives the estimator; the ledger holds its status and evidence.
```

Opt-in roulette scores the incoherent CXR lines of only some primary histories and reweights them. Transport, the photon grids, characteristic lines and the continuum do not change. The aim is more histories per unit time at MeV, where a few histories carry almost all the line yield.

## Estimator

Let $f_i(E)$ be the incoherent line spectrum of primary $i$ on a fixed grid, conditional on the transported histories, and let the analog estimator be $S(E) = N^{-1}\sum_i f_i(E)$ with $N$ the incident count. A screen pass sums each primary's whole-line mass

```{math}
:label: eq-roulette-mass

m_i = \sum_{\ell \in i} \frac{w_\ell \pi}{a_\ell},
```

over the same surviving lines the spectrum uses, where $w_\ell$ is the line weight and $a_\ell$ the width constant of its $\mathrm{sinc}^2$ profile. It runs no grid reduction. The first $n_p$ primaries are the pilot. They are always scored, and they fix the threshold $s_0$ as the $1-q$ quantile of $\{m_i\}_{i<n_p}$. Every later primary takes

```{math}
:label: eq-roulette-probability

p_i = \begin{cases} 1 & m_i \ge s_0, \\ p_{\min} & 0 < m_i < s_0, \end{cases}
```

and a primary with $m_i = 0$ is skipped. With $B_i \sim \mathrm{Bernoulli}(p_i)$ drawn from a uniform keyed only on the case seed and primary ID, the estimator is

```{math}
:label: eq-roulette-estimator

\hat S(E) = \frac{1}{N}\sum_i \frac{B_i}{p_i} f_i(E).
```

## Unbiasedness and variance

Histories are independent, and $s_0$ depends only on pilot histories, which are scored with $p_i = 1$. So for a later primary $p_i$ is fixed given its own history and the pilot, and $E[B_i/p_i \mid \text{history}, s_0] = 1$. Therefore

```{math}
:label: eq-roulette-unbiased

E[\hat S(E) \mid \text{transport}] = S(E),
```

for every bin and any linear functional of the spectrum. A skipped primary loses nothing: every surviving line has $w_\ell > 0$ and finite $a_\ell$, so $m_i = 0$ means it has no surviving line and $f_i \equiv 0$. This needs the audited per-row route, so flight-grouped substeps are refused: that route reduces whole flights and fills no per-row audit. Among primaries with $m_i > 0$ the screen only sets $p_i$, so a poor screen costs variance and never bias, as long as $p_{\min} > 0$. The proof also needs each $U_i$ independent of the histories. The draw hashes only the seed, a fixed salt and the primary ID, and consumes nothing from the transport streams.

The roulette draws add, conditional on transport,

```{math}
:label: eq-roulette-variance

\mathrm{Var}[\hat S(E) \mid \text{transport}] = \frac{1}{N^2}\sum_i \left(\frac{1}{p_i} - 1\right) f_i(E)^2 .
```

The total variance adds this to the analog population term. One primary stays one sample: $X_i = (B_i/p_i)\, g(f_i)$ for a linear functional $g$, and the sample variance of $X_i$ estimates both terms together.

For integrated yield with $g(f_i) = m_i$, a culled primary has $m_i < s_0$, so its weighted sample is below $s_0/p_{\min}$ and every heavier primary is kept exactly. This bound is what the exact screen adds over a predicted one: a heavy history cannot be assigned the floor.

Resolved spectra are different. Equation {eq}`eq-roulette-variance` is per bin, and a bin fed only by low-mass histories can lose much more than the integrated yield does.

## Why the coherent route is refused

For amplitudes $a_i$ the coherent spectrum squares a sum, and $E[(B_i/p_i)^2] = 1/p_i$ while the off-diagonal factors average to one. So

```{math}
:label: eq-roulette-coherent-bias

E\left|\sum_i \frac{B_i}{p_i} a_i\right|^2 = \left|\sum_i a_i\right|^2 + \sum_i \left(\frac{1}{p_i} - 1\right) |a_i|^2 ,
```

which is biased high. Cases with coherent emission, temporal profiles, channeling, two-beam line scoring or flight-grouped substeps (`max_dE_frac`) are refused when the case is built and again when scoring starts.

## Limiting cases

- $p_{\min} = 1$ gives the analog spectrum, up to floating-point summation order.
- $N \le n_p$ never resolves a threshold and is analog.
- All-zero pilot masses give $s_0 = 0$, so every later primary with positive mass is kept surely.
- With no `line_roulette` record the case payload, its identity and the scoring path are unchanged.

## Block invariance

A draw depends only on the seed and primary ID, and a primary's probability only on its own mass and $s_0$. Blocks hold whole primaries in ascending order, so every pilot mass is known before any later primary is selected, whichever block holds the boundary. Adaptive stopping reads analog masses and then runs the fixed-count case at the realized count.

The selection is reproducible only up to floating-point rounding of the masses. Block layout changes their summation order, and on CUDA the device `bincount` is not bit-stable even on identical input: repeated passes over one population differed by up to $6\times10^{-16}$ relative, while the line spectra stayed bit-identical. `checks/line_roulette_mass_jitter_879.py` measures this; at `684ec7a13` on an RTX 5080 (CuPy, FP64, hopg 30 keV, 40 primaries) raw `bincount` gave 20 distinct results in 20, masses 29 distinct vectors in 30, and spectra 1 in 10. So $s_0$, `line_mass_analog` and the weighted estimate can differ in their last bits between runs. A spectrum can differ only if a later primary's mass lies within that rounding of $s_0$, which moves it between strata. The estimator stays unbiased in that event, since each outcome is a valid fixed threshold, but the two runs are no longer bit-equal.

## Statistics and audits

The screen pass fills the truncation audit, so line mass, out-of-axis bounds and per-primary line-yield statistics stay analog, with no roulette noise and no weights. The adaptive monitor and bandwidth attribution call the scorer without a screen. So the reported yield standard error and the adaptive `target_rse` describe the analog yield and exclude Eq. {eq}`eq-roulette-variance`: they understate the uncertainty of the reweighted spectrum.

Run provenance records the policy, threshold, retained counts, `line_mass_analog`, `line_mass_estimate` and `added_mass_variance`. These masses are unnormalized sums over the population. They count the full profile of each kept line, including any part beyond the grid ends, so they are not the integral of the spectrum over the axis; the two differed by $5\times10^{-4}$ on a 6759-node grid.

## Independent verification

Fresh-context verification at commit `b53e83fb9`, repeated at the fix commit `0ce844613`, scoped to selection and weighting. The per-history spectra are the production scorers and are not rederived here. The derivation in this section was written from the ledger row and the page above before the implementation bodies were read.

### Rederivation

Condition on the transported histories $T$, which fix every $f_i$ and $m_i$. Let $p_i = p_i(T)$ with $p_i \in (0, 1]$ on every primary that is not skipped, and let $B_i = \mathbb{1}[U_i < p_i]$ with $U_i \sim \mathrm{Uniform}[0,1)$.

The proof needs one premise: **the family $\{U_i\}$ is independent of $T$ and mutually independent.** A key of seed and primary ID does not imply this. If $U_i$ were correlated with the transport draws of history $i$, then $E[B_i \mid T] \ne p_i$ and the estimator would be biased even with a correct $p_i$. Given the premise,

$$
E\left[\frac{B_i}{p_i} \,\middle|\, T\right] = 1, \qquad
\mathrm{Var}\left[\frac{B_i}{p_i} \,\middle|\, T\right] = \frac{p_i(1-p_i)}{p_i^2} = \frac{1}{p_i} - 1 ,
$$

so $E[\hat S \mid T] = S$ and $\mathrm{Var}[\hat S \mid T] = N^{-2}\sum_i (1/p_i - 1) f_i^2$, with covariances vanishing by mutual independence. Both match Eqs. {eq}`eq-roulette-unbiased` and {eq}`eq-roulette-variance`.

Conditional on $T$, $p_i$ may depend on anything in $T$ without bias, including later primaries. The pilot restriction is therefore needed for three other things: causal streaming, block invariance, and the unconditional independence of the summands $X_i$ for $i \ge n_p$ given the pilot.

The zero-mass skip assigns $p_i = 0$ and needs $E[(B_i/p_i) f_i \mid T] = f_i$, which holds only if $f_i \equiv 0$. Writing $\mathcal{L}^{\mathrm{screen}}_i$ and $\mathcal{L}^{\mathrm{score}}_i$ for the lines each pass keeps, the requirement is

$$
m_i = 0 \implies \sum_{\ell \in \mathcal{L}^{\mathrm{score}}_i} f_\ell \equiv 0 ,
$$

which holds when $\mathcal{L}^{\mathrm{score}}_i \subseteq \mathcal{L}^{\mathrm{screen}}_i$ and every screen weight is positive. Equality of the sets is not required for unbiasedness, but exact mass is. A nonzero $p_i$ never biases, so the screen may err upward only: a line kept only in the scoring pass, on a primary with no screened line, is the one lethal case.

For the coherent square with independent $B_i$,

$$
E\left[\frac{B_i B_j}{p_i p_j}\right] = 1 \quad (i \ne j), \qquad
E\left[\frac{B_i^2}{p_i^2}\right] = \frac{1}{p_i},
$$

which gives Eq. {eq}`eq-roulette-coherent-bias`.

Block invariance requires three things of the code: $s_0$ read only from the $n_p$ lowest primary IDs, $U_i$ a pure function of seed and ID, and a block's $B_i$ evaluated on masses that do not depend on the screen grid or on which other primaries share the block. A retried block must recompute the same $B_i$ and must not add its screen audit twice.

### Comparison with the implementation

The estimator, draw and threshold in `runner/line_roulette.py` match the rederivation on every route that can reach them. Probes used a 30 keV HOPG film of 1 µm on the CPU backend, with per-primary spectra from the production scorer as the reference. The statistical rows ran at `b53e83fb9`. The fix changed only the refusals and the record's key and docstring, so they were not repeated. The other rows ran at `0ce844613` unless marked.

| Check | Result |
| --- | --- |
| Threshold and $p_i$ recomputed outside the code, $N = 300$, $p_{\min} = 0.25$ (`b53e83fb9`) | Identical |
| Integrated mean over 600 seeds, fixed transport, divided by $S$ (`b53e83fb9`) | $1.00025$, $z = 0.08$ |
| Observed variance over Eq. {eq}`eq-roulette-variance`, integrated (`b53e83fb9`) | $1.04$ |
| Same ratio per bin, 20 largest bins: min, median, max (`b53e83fb9`) | $0.86$, $0.998$, $1.06$ |
| $\mathrm{corr}(U_i, \ln m_i)$, $n = 300$ (`b53e83fb9`) | $0.09$, below the $0.11$ two-sided 5% level |
| $U_i$ against the transport stream keys of the same seed, $n = 4000$ | No equal draw, correlation $0.02$ |
| Screen mass over full-grid audit mass, multilayer stack (`b53e83fb9`) | Relative difference 0 |
| Inverse-probability reconstruction, multilayer stack (`b53e83fb9`) | Relative difference $4 \times 10^{-16}$ |
| Screen mass and $p_{\min} = 1$ spectrum against analog, grooved escape by direct scoring | Relative difference 0 |
| $p_{\min} = 1$ against analog: default, narrow and off-centre axes, non-uniform axis, bin-mean, `sinc_cutoff`, mosaic | Relative difference 0 |
| $p_{\min} = 1$ against analog: `E_cut_lines_keV` of 29.8 and 29.95, secondary cascade (`b53e83fb9`) | Relative difference 0 |
| Same, on a narrow axis with 39 of 40 primaries at zero mass | Relative difference 0 |
| Line-phase OOM injected after the audit merge, rerun on the same phase cache | Spectrum, record and audit line mass unchanged |
| Artifact replay, `max_segments` of 1, 200 and $10^6$, against joined scoring | Same counts and threshold, spectra within $1.5 \times 10^{-15}$ |

The screen window equals the scoring window. The keep-window pad and the tabulation grid read only the first and last grid nodes, so the 2-node grid reproduces them, and no surviving-line mask reads an interior node.

The `id(E_grid)` table key is safe for the screen. `LineScreen` keeps its grid array alive, so the identity cannot be reused, and it holds its own cache separate from the real grid's.

The audit merge does not double count. The joined path builds a new audit on each attempt, the electron-block retry restores a snapshot whose arrays are replaced rather than mutated, live streaming has no audit because it refuses `line_grid_policy`, and artifact replay has no retry.

The device mass jitter cannot affect the skip: a primary with no audited line still sums to exactly zero, and a positive sum stays positive. It only moves masses near $s_0$, and it is independent of $U_i$, so each outcome is a valid threshold.

### Discrepancy

None open at `0ce844613`.

**Resolved: the zero-mass premise failed on the flight-grouped route.** At `b53e83fb9`, a case with `max_dE_frac > 0` took the grouped reduction, which returns before `_accumulate_edge_truncation`. Every primary then read $m_i = 0$ and was skipped, so the $p_{\min} = 1$ spectrum peaked at $0$ against an analog peak of $3.5976 \times 10^{-9}$, with no error.

The current code refuses this at three points. Each was run and raises:

| Input | Refused by |
| --- | --- |
| `Case`, raw `run_case` and raw `run_case_streamed` with `max_dE_frac` | `validate_case` |
| Substepped rows under a case without `max_dE_frac`, joined and through `BlockSpectrumScorer` | The guard in `_mc_spectrum` |
| A written artifact holding substepped rows under such a case, joined and streamed | The guard in `_mc_spectrum` |

The guard is sufficient for the strata. A stratum's rows are a subset of the screened rows, so if the screen is not grouped, no stratum is grouped either.

**Empty-audit routes.** Scoring can produce lines with an empty audit on two routes only, and both now raise. The reduction routes are the audited per-reflection branch and the audited batched branch; their remaining early returns have no surviving line. The two exceptions are the grouped reduction and the two-beam return in `_cxr_for_segments`. A raw two-beam mapping raises in `LineScreen.__init__`, as do raw temporal, coherent and channeling mappings. A raw temporal mapping can raise earlier from its own time-grid cap.

### Block-invariance assertions

Commit `e65acccd8` replaced exact threshold and record equality with exact counts and a relative tolerance of $10^{-12}$ on the mass-derived floats. Injected faults show that this does not materially weaken the check. Each row ran at block sizes 3, 7, 8, 20, 52 and 1000 against an unmodified joined reference:

| Injected fault | Counts | Helper | Spectrum, rtol $10^{-12}$ | Old exact threshold |
| --- | --- | --- | --- | --- |
| None | Pass | Pass | Pass | Pass |
| Threshold from all masses seen so far | Fail | Fail | Fail | Fail |
| Draws keyed on stratum position | Fail | Fail | Fail | Pass |
| One kept and culled swap per block | Fail | Fail | Fail | Pass |
| Threshold scaled by $1 + 10^{-13}$ | Pass | Pass | Pass | Fail |

The only fault the helper now admits changes neither the selection nor the spectrum. The streamed spectrum comparison and the adaptive bitwise spectrum comparison are unchanged, and `primaries_zero_mass` is a new exact check. The tolerance is about 1700 times the quoted jitter, which is looser than needed but harmless, because the spectra are compared separately.

### Other findings

- **Stale record name.** The block-invariance paragraph above still says `line_mass_exact`; the key is now `line_mass_analog`.
- **Jitter evidence has no artifact.** Neither the page, commit `e65acccd8` nor issue #879 records a job or log for the CUDA figures of $6 \times 10^{-16}$ and 20 of 20 distinct sums. The estimator's unbiasedness does not depend on their size.
- **Duck-typed two-beam record.** `validate_case` tests `isinstance(transition, Mapping)`, while `_cxr_for_segments` calls `.get`. A non-mapping object with `.get` passes the refusal. No production path builds such a record.
- **Masses are unnormalized.** `added_mass_variance` carries no $N^{-2}$, unlike Eq. {eq}`eq-roulette-variance`, as the page and docstring now state.

### Not verified

- CUDA execution, including `_fused.line_bracket` and `_queue_unit_lines`, because no device was used.
- Grooved transport end to end. The grooved escape was scored directly on slab rows, because the coupled radiative default refuses grooved targets.
- MeV behavior and analog agreement, which remain pending for `anchored`.
- Replay of artifacts that hold a subset of histories.
