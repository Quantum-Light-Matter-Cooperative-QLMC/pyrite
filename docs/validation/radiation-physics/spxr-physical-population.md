# Physical population in the classical SPXR field sum

**Validation:** `spxr-physical-population`  
**Verdict:** `rederived`, conditional on the fixed-population, independent-field assumptions below. Research normalization only; no production estimator or absolute SPXR yield is certified. Human sign-off remains pending.

## Claim and independence

The target is the proposed-only row in the [core coherent-physics ledger](../ledger-core-coherent-physics.md#spxr-physical-population): ensemble spectral yield from a fixed physical population of identically distributed, independent complex single-electron channel amplitudes.

The fresh verifier did not implement the claim. On 11 November 2026, the verifier read the target ledger record and the original published source, recorded the algebra and filters in scratch, and only then inspected the [implementation-context research note](../../research/physics/superradiant-pxr.md#physical-charge-must-be-distinct-from-sample-count). The initial ledger extraction also displayed adjacent unrelated records; the target research derivation and production implementation were not read before the independent algebra. No production implementation was inspected or assessed afterward.

The evidence is [Feranchuk, San and Skoromnik, *Superradiant parametric x-ray emission*, Phys. Rev. Accel. Beams **25**, 120702 (2022)](https://doi.org/10.1103/PhysRevAccelBeams.25.120702). A local published PDF was inspected as text and visually at pp. 120702-7 and 120702-8; this is the published version, not an assumed equation match to the preprint.

## Source equation and normalization

Equation (36), p. 120702-7, is a double sum of each electron's trajectory amplitude times the conjugate amplitude of another electron. With the observation frequency, direction and polarization fixed, write its trajectory integral as

$$
A_a=\int dt\,\mathbf E^{(+)}_{\mathbf k s}(\mathbf r_a(t),\omega)
\cdot\mathbf v_a\,e^{i\omega t}.
$$

The source photon spectral-angular yield has the structure

$$
Y=C\sum_{a=1}^{N}\sum_{b=1}^{N}A_a A_b^*
=C\left\lvert\sum_{a=1}^{N}A_a\right\rvert^2,
\qquad C=\frac{e_0^2\omega}{4\pi^2}
$$

in the source's conventions. The prefactor is transcribed to identify the source equation; its electromagnetic normalization and conversion to project units are outside this population-counting claim. Define $\mathcal E_a=\sqrt C A_a$, or use any consistently normalized channel amplitude whose squared modulus is the specified single-electron spectral yield. The research note's $\mathcal E_a$ is this channel amplitude, not a raw electric field measured in volts per metre.

Equation (45), p. 120702-8, selects equal-index terms and exhibits the linear population factor. Equations (49)-(50), on the same page, present a coherent term with $N^2$ after replacing the discrete sum by a distribution average. The continuum replacement is stated in Eq. (22), p. 120702-5. Those formulas support the spontaneous/coherent interpretation and large-population scaling. The exact finite-population $N(N-1)$ coefficient below follows directly from counting unequal-index terms in Eq. (36); it is not the coefficient literally printed in Eq. (49).

The charge example following Eq. (48), p. 120702-8, associates a positive stated bunch charge of $0.2\,\mathrm{nC}$ with approximately $1.3\times10^9$ electrons. This supports interpreting the quoted charge as a magnitude. It is not an independent absolute-yield benchmark here.

## Independent derivation

Assume a fixed integer $N\geq0$, linear superposition, a common specified channel normalization, and identically distributed complex amplitudes with finite second moment. Define

$$
\mu=\mathbb E[\mathcal E],\qquad
m_2=\mathbb E[\lvert\mathcal E\rvert^2].
$$

For $a\ne b$, independence gives

$$
\mathbb E[\mathcal E_a\mathcal E_b^*]
=\mathbb E[\mathcal E_a]\mathbb E[\mathcal E_b^*]
=\lvert\mu\rvert^2.
$$

There are $N$ equal-index terms and $N(N-1)$ **ordered** unequal-index terms. Consequently,

$$
\boxed{\mathbb E\!\left[\left\lvert\sum_{a=1}^{N}\mathcal E_a\right\rvert^2\right]
=Nm_2+N(N-1)\lvert\mu\rvert^2.}
$$

An equivalent decomposition is

$$
\mathbb E[Y]
=N\bigl(m_2-\lvert\mu\rvert^2\bigr)+N^2\lvert\mu\rvert^2
=N\operatorname{Var}_{\mathbb C}(\mathcal E)+N^2\lvert\mu\rvert^2.
$$

The first expression separates self and pair terms. The second separates fluctuations and the mean field. Calling the linear term simply “incoherent” without specifying which decomposition is used can hide a difference of $N\lvert\mu\rvert^2$. Adding $Nm_2$ to an exact $N^2\lvert\mu\rvert^2$ pair term would overcount the total by that amount. Replacing $N(N-1)$ by $N^2$ in the pair term is a large-$N$ approximation, with a fractional excess $1/(N-1)$ relative to the nonzero exact pair contribution for $N>1$.

Pairwise independence suffices for this second-moment calculation; mutual independence is stronger than required. More generally, factorization of these cross moments suffices. Independence of different coordinates within one electron's phase space is not required: a single-electron amplitude can depend on their full joint law.

## Physical charge and ensemble qualification

Let $e_{\rm abs}=\lvert e\rvert>0$ denote the elementary charge magnitude. For an electron-only bunch with signed total charge $Q_{\rm signed}$,

$$
Q_{\rm signed}=-Ne_{\rm abs},\qquad
N=\frac{\lvert Q_{\rm signed}\rvert}{e_{\rm abs}}.
$$

If $Q$ is explicitly defined as the nonnegative bunch-charge magnitude, the equivalent notation is $N=Q/\lvert e\rvert$. A signed negative charge must not be inserted into that latter expression. The research note already defines $Q$ as a magnitude; the ledger should carry that definition too. Mixed-sign net charge cannot determine an electron population by this formula.

The exact finite sum requires an integer physical count. A measured or nominal charge may infer a noninteger effective count; using that continuously is a modeling approximation, not an exact finite-$N$ identity. Charge uncertainty and random shot populations are outside this fixed-$N$ verdict. In particular, an ensemble with fluctuating population cannot in general use the mean population in both factors of the fixed-count expression.

The number of numerical tracks does not occur in the derivation. Replacing physical $N$ by a simulation sample count changes the modeled bunch. This algebra establishes the required physical moments and population weights; it does not establish an unbiased finite-track estimator or its convergence.

For shared bunch settings $Z$, conditional i.i.d. amplitudes obey the same derivation at fixed $Z$ and fixed $N$:

$$
\mathbb E[Y\mid Z]
=Nm_2(Z)+N(N-1)\lvert\mu(Z)\rvert^2.
$$

A subsequent average over $Z$ contains $\mathbb E_Z[\lvert\mu(Z)\rvert^2]$, generally different from $\lvert\mathbb E_Z[\mu(Z)]\rvert^2$. Shared timing or phase jitter therefore cannot silently be folded into an unconditional i.i.d. law. This is an applicability qualification, not validation of a correlated-bunch model or of a numerical estimator.

## Filters

| Filter | Result | Evidence |
| --- | --- | --- |
| Dimensions and normalization | Pass within the stated channel convention | $N$ is dimensionless; $m_2$ and the squared mean amplitude have the same spectral-yield units. For a yield per angular frequency and solid angle, both terms retain those units. No conversion of a raw field norm into photons is inferred. |
| Empty and one-electron limits | Pass | $N=0$ gives zero; $N=1$ gives $m_2$ with no pair term. |
| Zero mean | Pass | $\mu=0$ gives $Nm_2$. |
| Identical deterministic amplitudes | Pass | $\mathcal E_a=a$ gives $N^2\lvert a\rvert^2$. Deterministic but unequal phases do not imply this limit. |
| Signs and conventions | Pass with charge qualification | Ordered-pair counting gives $N(N-1)$; positivity follows from the variance form. A common phase rotation leaves the yield invariant. |
| Reference values | Pass for the cited structural evidence | Equation and page references checked against the published PDF. No optical constants or absolute-yield values were transcribed as validation data. |
| Tolerance | Exact | The numerical reference below uses rational arithmetic; no fitted threshold or statistical uncertainty. |
| External-code or device evidence | Not claimed | No CPU/GPU parity, external simulator agreement, or production-output validation is used. |

## Independent exact enumeration

A deterministic reference enumerated every ordered outcome for three discrete complex-amplitude laws and each $N=0,\ldots,6$. Each outcome's probability multiplies its directly calculated squared summed amplitude. This sum is compared with the moment formula; the enumeration does not call any project physics routine.

| Law | Amplitudes and probabilities | $m_2$ | Squared mean amplitude | Enumerated yields for $N=0,\ldots,6$ |
| --- | --- | --- | --- | --- |
| Complex, nonzero mean | $1+i$ with $1/2$; $-1+2i$ with $1/3$; $2-i$ with $1/6$ | $7/2$ | $5/4$ | $0,7/2,19/2,18,29,85/2,117/2$ |
| Zero mean | $1,-1,i,-i$, each with $1/4$ | $1$ | $0$ | $0,1,2,3,4,5,6$ |
| Deterministic | $2-i$ with probability $1$ | $5$ | $5$ | $0,5,20,45,80,125,180$ |

The following standalone reference was executed through `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python /tmp/spxr-enumerate.py` in the task worktree. `Fraction` evaluates real and imaginary components separately, so all probabilities, moments and residuals are exact.

```python
from fractions import Fraction as F
from itertools import product

cases = {
    "complex": [(1, 1, F(1, 2)), (-1, 2, F(1, 3)), (2, -1, F(1, 6))],
    "zero_mean": [(1, 0, F(1, 4)), (-1, 0, F(1, 4)), (0, 1, F(1, 4)), (0, -1, F(1, 4))],
    "deterministic": [(2, -1, F(1))],
}
for law in cases.values():
    mu_r = sum(p * r for r, i, p in law)
    mu_i = sum(p * i for r, i, p in law)
    m2 = sum(p * (r * r + i * i) for r, i, p in law)
    mu2 = mu_r * mu_r + mu_i * mu_i
    for n in range(7):
        direct = F(0)
        for outcome in product(law, repeat=n):
            r = sum(a[0] for a in outcome)
            i = sum(a[1] for a in outcome)
            p = F(1)
            for a in outcome:
                p *= a[2]
            direct += p * (r * r + i * i)
        assert direct == n * m2 + n * (n - 1) * mu2
```

**Result:** all 21 comparisons passed, with zero rational residual. This is a bounded algebra check recorded in this research write-up, not a committed regression anchor or justification for `anchored` status.

## Comparison and suggested ledger change

The research note's self/pair expression, charge-magnitude definition, fixed-population averaging, units and three stated limits match the independent derivation. No factor, sign or exponent correction is needed. It already separates finite-sample pair-estimator bias, correlations and fixed-charge convergence as unresolved work.

The remaining qualifications are explicit integer population, finite second moment, pairwise independence being sufficient, conditioning on shared bunch settings, and the distinction between the exact $N(N-1)$ coefficient and the source's continuum $N^2$ expression. These constrain applicability without extending the claim to actual XFEL correlations or production output.

Suggested owner edit: advance this row from `unverified` to `rederived`; state that $Q$ is charge magnitude (or write $N=\lvert Q\rvert/\lvert e\rvert$ for signed charge), fixed $N$ is a nonnegative integer, and channel amplitudes have finite second moment. Cite source Eq. (36), pp. 120702-7 to 120702-8, plus this independent write-up and its exact enumeration. Keep the anchor empty and retain the proposed-only scope and unresolved estimator/correlation/convergence notes. Only a human can grant sign-off.
