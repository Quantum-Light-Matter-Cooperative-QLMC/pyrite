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

for every bin and any linear functional of the spectrum. A skipped primary loses nothing: every surviving line has $w_\ell > 0$ and finite $a_\ell$, so $m_i = 0$ means it has no surviving line and $f_i \equiv 0$. The screen only sets $p_i$, so a poor screen costs variance and never bias, as long as $p_{\min} > 0$.

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

which is biased high. Cases with coherent emission, temporal profiles, channeling or two-beam line scoring are refused.

## Limiting cases

- $p_{\min} = 1$ gives the analog spectrum, up to floating-point summation order.
- $N \le n_p$ never resolves a threshold and is analog.
- All-zero pilot masses give $s_0 = 0$, so every later primary with positive mass is kept surely.
- With no `line_roulette` record the case payload, its identity and the scoring path are unchanged.

## Block invariance

A draw depends only on the seed and primary ID, and a primary's probability only on its own mass and $s_0$. Blocks hold whole primaries in ascending order, so every pilot mass is known before any later primary is selected, whichever block holds the boundary. Adaptive stopping reads analog masses and then runs the fixed-count case at the realized count.

The selection is reproducible only up to floating-point rounding of the masses. Block layout changes their summation order, and on CUDA the device `bincount` is not bit-stable even on identical input: repeated passes over one population differed by up to $6\times10^{-16}$ relative, while the line spectra stayed bit-identical. So $s_0$, `line_mass_exact` and the weighted estimate can differ in their last bits between runs. A spectrum can differ only if a later primary's mass lies within that rounding of $s_0$, which moves it between strata. The estimator stays unbiased in that event, since each outcome is a valid fixed threshold, but the two runs are no longer bit-equal.

## Statistics and audits

The screen pass fills the truncation audit, so line mass, out-of-axis bounds and per-primary line-yield statistics stay analog and exact, and need no weights. The adaptive monitor and bandwidth attribution call the scorer without a screen. Run provenance records the policy, threshold, retained counts, exact mass, weighted estimate and added mass variance.

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
