# Bulk dielectric loss candidate

Validation: `dielectric-bulk-loss`. Status: filtered; fresh-context source-to-code
validation and human sign-off are pending. This host-only function is not a
transport mode.

## Source and equation

[Tougaard and Yubero, *Surface and Interface Analysis* 54 (2022), Eqs. 9–12](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/sia.7095)
give the bulk differential inverse inelastic mean free path $K_E(W)$ for an
isotropic homogeneous medium. With $Q=\hbar^2k^2/(2m_e)$ in eV and the
oscillator center $W_i(Q)=W_i(0)+\alpha_iQ$,

$$
L(Q,W)=\Theta(W-E_g)\sum_i
\frac{A_i\gamma_i W}{[W_i(Q)^2-W^2]^2+(\gamma_iW)^2},
\qquad
K_E(W)=\frac{1}{\pi E a_0}\int_{k_-}^{k_+}\frac{dk}{k}L(Q,W).
$$

The nonrelativistic electron limits are
$Q_\pm=(\sqrt E\pm\sqrt{E-W})^2$. Since $dk/k=dQ/(2Q)$, the code integrates
$L$ over $\log Q$ with Gauss–Legendre quadrature. $A_i$ has eV$^2$,
$W_i,\gamma_i,E_g,Q$ have eV, and $\alpha_i$ is dimensionless. Thus $L$ is
dimensionless and $K_E$ is Å$^{-1}$ eV$^{-1}$ when $a_0$ is in Å. At $W\leq
E_g$ or $W=E$, the returned rate is zero. The source assumes nonrelativistic
incident electrons and losses small compared with their energy.

Conditional on a positive-rate loss $W$, the same bulk response gives a
momentum-transfer density proportional to $L(Q,W)\,d\log Q$. The host recoil
sampler forms a cumulative trapezoidal integral on a uniform $\log Q$ grid
between the same kinematic limits and inverts it with one uniform variate.
The outgoing primary momentum triangle then gives

$$
\cos\theta=\frac{E+(E-W)-Q}{2\sqrt{E(E-W)}}.
$$

Here $E$, $W$, and $Q$ are all expressed as nonrelativistic kinetic-energy
equivalents of squared momentum. At $Q_-$ the cosine is $+1$; at $Q_+$ it is
$-1$. This is a polar angle only: azimuth, secondary state, and trajectory
rotation remain outside the host candidate.

On a caller-supplied, fixed loss grid, the host candidate linearly interpolates
$K_E(W)$ within each bin. It integrates the zeroth and first moments exactly
under that interpolation. A transfer threshold $W_c$ clips the same bins after
the grid and rates are fixed. The exact band-gap energy is inserted as a grid
node before interpolation so no positive rate leaks below it. This gives

$$
S_{\mathrm{soft}}=\int_0^{W_c}WK_E(W)\,dW,\qquad
S_{\mathrm{hard}}=\int_{W_c}^{W_{\max}}WK_E(W)\,dW,\qquad
\lambda_{\mathrm{hard}}^{-1}=\int_{W_c}^{W_{\max}}K_E(W)\,dW.
$$

The moment integrals have units eV Å$^{-1}$; the hard rate has Å$^{-1}$.
The hard-loss sampler inverts the exact quadratic bin CDF. Moving $W_c$
therefore changes only the partition, not the total rate or first moment.
$W_c\geq W_{\max}$ gives zero hard rate. The grid endpoint $W_{\max}<E$ is
chosen by the caller: these are partial valence moments, not the total
material IMFP or corrected stopping power. Grid convergence and the core
contribution are needed before using them in transport.

The three Si valence oscillators used in the check come from
[Pauly, Yubero and Tougaard's `ELF_Si.txt` dataset](https://zenodo.org/records/6024064),
which attributes them to Yubero et al., *Surface and Interface Analysis* 20
(1993), 719. Their $(W_i,A_i,\gamma_i,\alpha_i)$ values are
$(10,6.08458,5,0.5)$, $(14,30.4229,5,0.5)$ and
$(16.8,212.9603,3.8,0.5)$; $E_g=1.12$ eV. These are fitted material
parameters, not values inferred from SBETHE's atomic OOS.

## Checks and limits

At 1 and 3 keV, the candidate's 14–20 eV fraction is about 0.50 and 0.52
when its 0–50 eV spectrum is normalized to unity; the normalized peak exceeds
0.08 eV$^{-1}$. Both clear the conservative figure-read bounds from
[Werner's Si bulk REELS Fig. 3(b)](https://arxiv.org/abs/cond-mat/0503470).
This is a useful shape cross-check, although the Yubero oscillator fit and
Werner retrieval both arise from REELS and therefore are not wholly
independent measurements. The 64- and 96-point momentum quadratures agree
within $10^{-4}$ relative at selected losses. An independent direct
$\log k$ integral at 1 keV and 16 eV agrees with the code's $\log Q$
quadrature within $10^{-7}$ relative, checking the factor of two in the
variable change. Gap and kinematic-endpoint limits are covered by tests.
For 1 keV incident energy and 16 eV loss, four recoil quantiles agree with an
independent direct $\log k$ integral to $3\times10^{-4}$ absolute CDF
probability. A fixed-seed sampling check covers the median and invalid-input
guards; the momentum triangle is checked at each quantile.
On a fixed 0–100 eV grid at 1 keV, four cutoff choices preserve the integrated
valence rate and first moment; soft and hard moments add without a gap. Five
hard-loss quantiles agree with direct integration of the stored linear bins to
$10^{-12}$ CDF probability, and fixed-seed samples pass a median check.
Adaptive loss integration from the 1.12 eV gap to 100 eV gives partial zeroth
and first moments of 0.0410461 Å$^{-1}$ and 0.867818 eV Å$^{-1}$ at 1 keV,
and 0.0161553 Å$^{-1}$ and 0.330387 eV Å$^{-1}$ at 3 keV. With 0.5, 0.25,
and 0.125 eV linear grids, both relative errors decrease monotonically; the
largest error on the finest grid is $4.7\times10^{-6}$. This checks numerical
loss-grid convergence for the stated 0–100 eV *partial* valence interval,
not convergence of the high-loss endpoint or a total material rate.

The fit represents valence losses only. Integrating the 1 keV candidate from
0 to 100 eV gives a *partial* inverse mean free path corresponding to 24.4 Å;
it omits deep-core interactions and cannot be treated as a full IMFP. It also
has no secondary state, surface response, relativistic correction or
connection to SBETHE's corrected stopping table. Its fitted 1993 Si response
cannot simply be transferred to SiO2 or MoS2. A full positive transfer
spectrum must handle the core tail without double counting, recover the
corrected stopping first moment, and pass total-rate and differential checks
before CPU or CUDA transport integration.

## High-loss endpoint and core diagnostic

An independent optical-limit check sharpens this gate. The [Si ELF table from
Yang et al. (2019)](https://micro.ustc.edu.cn/database/ELF/Si.html), derived
from high-precision REELS analysis, gives $L(0,16\,\mathrm{eV})=3.453856004$
and $L(0,110\,\mathrm{eV})=0.047536284$. Evaluating the three-oscillator
Yubero valence fit at the same zero-momentum points gives 3.211891 and
0.000777612, respectively. The 16 eV fit is within 7% of the later optical
table, but its 110 eV value is about 61 times smaller. The latter is an
expected failure for a valence-only fit; it independently confirms that the
fit's analytic high-loss tail cannot stand in for the Si L-edge response.
These optical data do not determine the finite-momentum core rate or a
valence/core overlap prescription.

The 0–100 eV loss-grid check above establishes numerical convergence only on
that interval. At a fixed 0.25 eV grid spacing, extending the integration
endpoint changes the *partial* Si valence moments as follows. Rates are in
Å$^{-1}$ and first moments in eV Å$^{-1}$:

| Incident energy | Endpoint | Valence rate | Valence first moment |
| ---: | ---: | ---: | ---: |
| 1 keV | 102.2154 eV | 0.041082 | 0.871797 |
| 1 keV | 200 eV | 0.041788 | 0.969078 |
| 3 keV | 102.2154 eV | 0.016166 | 0.331567 |
| 3 keV | 200 eV | 0.016441 | 0.370364 |

The 102.2154 eV endpoint is the first positive shell-edge jump in the shipped
Si OOS table; it is a diagnostic split, not a demonstrated upper bound on
valence losses. Extending to 200 eV changes the first moment by 11.2% at
1 keV and 11.7% at 3 keV, while changing the total event rate by 1.7%.
Thus a rate-converged endpoint cannot establish stopping closure.

For a unit comparison with the raw [Si OOS core diagnostic](gos-core-edge.md),
multiply its per-atom cross sections by the shipped Si atom density
$n=\rho N_A/M$ and divide by $10^8$ Å/cm. At 1 and 3 keV, respectively, its
raw first moments are 0.918234 and 0.519890 eV Å$^{-1}$; its raw event rates
are 0.004751 and 0.002262 Å$^{-1}$. Adding these to the valence moments
truncated at the shell edge gives 1.790031 and 0.851456 eV Å$^{-1}$. The
corrected SBETHE `stp.dat` targets are 1.692945 and 0.891196 eV Å$^{-1}$:
the raw sums are 5.7% high and 4.5% low. Both components remain uncalibrated
in this comparison. A single multiplier could force either mean but would not
establish a valid differential spectrum or absolute event rate. The dielectric
tail can also represent valence transfers above the OOS core edge, so clipping
it there is not a validated non-overlap rule. A sourced high-loss response and
independent core/rate evidence remain necessary before composition or
transport activation.
