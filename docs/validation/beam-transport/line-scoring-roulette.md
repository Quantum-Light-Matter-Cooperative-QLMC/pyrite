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

The selection is reproducible only up to floating-point rounding of the masses. Block layout changes their summation order, and on CUDA the device `bincount` is not bit-stable even on identical input: repeated passes over one population differed by up to $6\times10^{-16}$ relative, while the line spectra stayed bit-identical. So $s_0$, `line_mass_exact` and the weighted estimate can differ in their last bits between runs. A spectrum can differ only if a later primary's mass lies within that rounding of $s_0$, which moves it between strata. The estimator stays unbiased in that event, since each outcome is a valid fixed threshold, but the two runs are no longer bit-equal.

## Statistics and audits

The screen pass fills the truncation audit, so line mass, out-of-axis bounds and per-primary line-yield statistics stay analog, with no roulette noise and no weights. The adaptive monitor and bandwidth attribution call the scorer without a screen. So the reported yield standard error and the adaptive `target_rse` describe the analog yield and exclude Eq. {eq}`eq-roulette-variance`: they understate the uncertainty of the reweighted spectrum.

Run provenance records the policy, threshold, retained counts, `line_mass_analog`, `line_mass_estimate` and `added_mass_variance`. These masses are unnormalized sums over the population. They count the full profile of each kept line, including any part beyond the grid ends, so they are not the integral of the spectrum over the axis; the two differed by $5\times10^{-4}$ on a 6759-node grid.

## Independent verification

Fresh-context verification at commit `b53e83fb9`, scoped to selection and weighting. The per-history spectra are the production scorers and are not rederived here. The derivation in this section was written from the ledger row and the page above before the implementation bodies were read.

### Rederivation

Condition on the transported histories $T$, which fix every $f_i$ and $m_i$. Let $p_i = p_i(T)$ with $p_i \in (0, 1]$ on every primary that is not skipped, and let $B_i = \mathbb{1}[U_i < p_i]$ with $U_i \sim \mathrm{Uniform}[0,1)$.

The proof needs one premise that the page states only implicitly: **the family $\{U_i\}$ is independent of $T$ and mutually independent.** A key of seed and primary ID does not imply this. If $U_i$ were correlated with the transport draws of history $i$, then $E[B_i \mid T] \ne p_i$ and the estimator would be biased even with a correct $p_i$. Given the premise,

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

The estimator, draw and threshold in `runner/line_roulette.py` match the rederivation on every route except the flight-grouped one. Probes used a 30 keV HOPG film of 1 µm on the CPU backend, with per-primary spectra from the production scorer as the reference.

| Check | Result |
| --- | --- |
| Threshold and $p_i$ recomputed outside the code, $N = 300$, $p_{\min} = 0.25$ | Identical |
| Integrated mean over 600 seeds, fixed transport, divided by $S$ | $1.00025$, $z = 0.08$ |
| Observed variance over Eq. {eq}`eq-roulette-variance`, integrated | $1.04$ |
| Same ratio per bin, 20 largest bins (min, median, max) | $0.86$, $0.998$, $1.06$ |
| $\mathrm{corr}(U_i, \ln m_i)$, $n = 300$ | $0.09$, below the $0.11$ two-sided 5% level |
| Screen mass over full-grid audit mass, multilayer stack | Relative difference 0 |
| Inverse-probability reconstruction, multilayer stack | Relative difference $4 \times 10^{-16}$ |
| $p_{\min} = 1$ against analog: default, narrow and off-centre axes, non-uniform axis, `E_cut_lines_keV` of 29.8 and 29.95, bin-mean, `sinc_cutoff`, mosaic, secondary cascade | Relative difference 0 |
| Same, on a narrow axis with 39 of 40 primaries at zero mass | Relative difference 0 |
| Line-phase OOM injected after the audit merge, rerun on the same phase cache | Spectrum, record and audit line mass unchanged |

The screen window equals the scoring window on these routes. The keep-window pad and the tabulation grid read only the first and last grid nodes, so the 2-node grid reproduces them, and no surviving-line mask reads an interior node.

The `id(E_grid)` table key is safe for the screen. `LineScreen` keeps its grid array alive, so the identity cannot be reused, and it holds its own cache separate from the real grid's.

The audit merge does not double count. The joined path builds a new audit on each attempt, the electron-block retry restores a snapshot whose arrays are replaced rather than mutated, live streaming has no audit because it refuses `line_grid_policy`, and artifact replay has no retry.

### Discrepancy

**The zero-mass premise fails on the flight-grouped route.** With `max_dE_frac > 0`, flights split into substeps and `_setup` selects the grouped reduction. That branch of `_accumulate_reflection` reduces and returns before it reaches `_accumulate_edge_truncation`, so the audit stays empty. `LineScreen.score` then sets $m_i = 0$ for every primary, and every primary is skipped. The first divergent statement is the implication $m_i = 0 \implies f_i \equiv 0$ above; here $m_i = 0$ while $f_i \ne 0$.

| Quantity, $N = 40$, `max_dE_frac` $= 0.002$, $p_{\min} = 1$ | Analog | Roulette |
| --- | --- | --- |
| Peak of `spec` | $3.5976 \times 10^{-9}$ | $0$ |
| Primaries scored | | 0 |
| Primaries at zero mass | | 40 |
| `line_mass_exact` | | 0 |

`Case(...)` accepts this combination, and `run_case` runs it without a warning. On a device backend the grouped route raises, so the failure is silent only on the host.

The sentence "a poor screen costs variance and never bias" needs the same exception: it holds for $p_i > 0$, and the skip is $p_i = 0$.

### Other findings

- **Refusals are enforced only by `Case`.** `line_screen_for` does not call `validate_case`, and `checks/line_roulette_fom_879.py` sets the policy on a raw dict. A raw dict with `temporal_profile` reached temporal setup without the roulette refusal. By reading, a raw two-beam dict would return from `case_dtr_spectrum` with an empty audit and give the same zero spectrum.
- **`line_mass_exact` is a conditional, unnormalized, whole-profile sum.** It is exact in having no roulette noise. It is not the yield on the axis: it is $N$ times the integral of the untruncated profiles of the lines inside the padded window. On the 6759-node resonance grid, $N \int \hat S \, dE$ over `line_mass_estimate` was $0.9995$. `added_mass_variance` has the same normalization, so it carries no $N^{-2}$, unlike Eq. {eq}`eq-roulette-variance`.
- **Block invariance is exact for draws and holds to rounding for masses.** The test helper cites this page for a $6 \times 10^{-16}$ repeatability that the page does not state. A primary within that distance of $s_0$ could change stratum, which leaves the estimator unbiased.
- **Reported statistics describe the analog yield.** The yield standard error and the adaptive `target_rse` exclude Eq. {eq}`eq-roulette-variance`, so they understate the uncertainty of the reweighted spectrum.

### Not verified

- CUDA execution, including `_fused.line_bracket` and `_queue_unit_lines`, because no device was used.
- The grooved route by execution. It shares the audited per-reflection branch with the multilayer route, which was run.
- MeV behavior and analog agreement, which remain pending for `anchored`.
- Replay of artifacts that hold a subset of histories.
