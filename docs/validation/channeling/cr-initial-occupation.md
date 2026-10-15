# `cr-initial-occupation` re-derivation

Claim: `materials/channeling/transverse.py::entry_populations`. Source: Andersen et al. (1983); Azadegan (2013), Eq. 17.

## Independent derivation (before reading the body)

An electron of momentum $p=\gamma\beta mc$ entering at small angle $\psi$ to the plane family has transverse wave $\phi(x)=e^{ik_xx}$ with

$$
k_x=\frac{p\psi}{\hbar}=\frac{\gamma\beta mc^2}{\hbar c}\,\psi=\frac{\sqrt{T^2+2Tmc^2}}{\hbar c}\,\psi .
$$

Decompose $k_x=k+m_0g$ with $k\in(-g/2,g/2]$ and integer $m_0$, so $e^{ik_xx}=e^{ikx}e^{im_0gx}$ is one plane-wave component of the Bloch space at $k$. Bloch states at that $k$ are $\psi_{n}=d^{-1/2}\sum_mc^{(n)}_me^{i(k+mg)x}$. Averaged over the entry point (transverse translation) the interference between different $k$ vanishes, so the population of band $n$ is

$$
P_n=\bigl\lvert\langle\psi_n\vert\phi\rangle_{\rm cell}\bigr\rvert^2=\bigl\lvert c^{(n)}_{m_0}\bigr\rvert^2 .
$$

The coefficient matrix $C_{mn}$ is unitary in a complete basis, so $\sum_n\lvert C_{m_0n}\rvert^2=1$. A Gaussian divergence of RMS $\sigma$ is $\int P_n(\psi_0+\epsilon)\,\mathcal N(\epsilon;\sigma)\,d\epsilon$. Using the substitution $\epsilon=\sqrt2\sigma t$, this is $\pi^{-1/2}\sum_iw_iP_n(\psi_0+\sqrt2\sigma t_i)$ with physicists' Gauss–Hermite nodes. Equivalently, the probabilists' version has nodes $\sigma\,\xi_i$ and weights normalized to one. Surface refraction (mean inner potential) is not applied, by design.

Symmetry: for an even real potential, $\psi\to-\psi$ maps $c_m\to c_{-m}$ up to the parity sign, so $P(\psi)=P(-\psi)$. At $\psi=0$ ($k=0$, $m_0=0$), odd states have $c_0=0$, so only even states are populated.

## Independent numeric check

My own code (physicists' Gauss–Hermite, 24 nodes, own eigensolver), Si(110), 40 MeV, $M=32$. The profile comes from `planar_profile`.

| Quantity | Mine | Reference |
| --- | --- | --- |
| $\sum_n P_n(0)$ | $1.0000000000000007$ | 1 |
| $P_0(0)$ | 0.2383 | Azadegan 2013: about 0.2–0.25 (ok) |
| $P_{1,3,5,7}(0)$ | $<10^{-25}$ | even states only |
| $(P_0,P_2,P_4,P_6)$ at $\psi=0$ | 0.2383, 0.2553, 0.3059, 0.1977 | |
| $\sum_n P_n$, $\sigma=100\ \mu$rad | $1.0$; odd states populated (e.g. $P_1=0.040$) | |
| $\max\lvert P(+\psi)-P(-\psi)\rvert$, $\psi=120\ \mu$rad | $4\times10^{-14}$ | 0 |

## Diff against the implementation

The code sets $p/\hbar=\gamma\beta m_e/\hbar c$ in Å$^{-1}$, so $k_x=p\psi/\hbar$ as derived. It takes `m0=round(k_x/g)` and calls `transverse_states` at $k_x$, which folds to $k$. It then re-derives the matched order from the folded $k$, `m_match=round((k_x-k)/g)`, which handles the $-g/2\to+g/2$ edge shift correctly, and uses $\lvert C_{m_{\rm match},n}\rvert^2$. The Gaussian average uses `hermegauss` nodes times $\sigma$ with weights normalized to sum 1, which is the same rule as the physicists' version. Orders with $\lvert m_0\rvert>M$ are skipped (population is lost, so the sum is $<1$ there).

Numeric diff against my solver: zero incidence $0.0$; with $\sigma=100\ \mu$rad $1.8\times10^{-14}$; tilted $\psi=120\ \mu$rad $1.0\times10^{-14}$. At the fold edge, $k_x=g/2$: both give $\sum_nP_n=1$ and identical populations (0.2272, 0.0318, ...).

## Verdict

`rederived`. Findings, neither a discrepancy: at $\lvert m_0\rvert>M$ the function silently returns zero rows, so $\sum_nP_n<1$ without a warning; and the populations are per band index, which for shallow bands near the barrier at $k\ne0$ is not a fixed physical state. Both are documented in the docstring.
