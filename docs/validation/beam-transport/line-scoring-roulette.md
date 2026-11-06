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

## Statistics and audits

The screen pass fills the truncation audit, so line mass, out-of-axis bounds and per-primary line-yield statistics stay analog and exact, and need no weights. The adaptive monitor and bandwidth attribution call the scorer without a screen. Run provenance records the policy, threshold, retained counts, exact mass, weighted estimate and added mass variance.
