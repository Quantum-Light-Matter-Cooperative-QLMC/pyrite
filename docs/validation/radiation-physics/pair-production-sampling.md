# Pair-production energy and angle sampling

`Validation: pair-production-sampling` — independent re-derivation for the
[ledger claim](../ledger-transport-background.md#pair-production-sampling).

## Source, scope, and assumptions

Source: [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§2.4, Eqs. 2.74–2.99 and Table 2.2 (manual pp. 67–74). The equations were
read from the rendered PDF pages, not only from a text extraction, for
Eqs. 2.80, 2.85–2.88 and 2.90–2.96.

The intended quantity is one pair event for a photon of energy
$E=\kappa m_ec^2>2m_ec^2$ converting in the field of an atom of atomic number
$Z$ (1–99): the electron reduced energy
$\epsilon=(E_-+m_ec^2)/E$, the kinetic energies $E_\mp$ in eV, and the two
unit directions. Only the shape of the DCS is used; the total pair cross
section comes from EPDL2025. Assumptions taken from the source: Bethe–Heitler
DCS with exponential (Wentzel) screening, symmetric in
$\epsilon\leftrightarrow1-\epsilon$; triplet events simulated as pairs; polar
angles from the leading high-energy term, azimuths independent and uniform.

## Cheap filters

| check | result |
| --- | --- |
| units | $\epsilon$, $\kappa$, $b$, $R m_ec/\hbar$, $\phi_{1,2}$, $F_0$, $f_C$ and $\cos\theta$ are dimensionless; $E_\mp$ in eV; pass |
| energy conservation | $E_-+E_++2m_ec^2=\epsilon E+(1-\epsilon)E=E$ exactly; pass |
| support | $E_\mp\ge0\iff\epsilon\in[\kappa^{-1},1-\kappa^{-1}]$; pass |
| symmetry | $b(\epsilon)=b(1-\epsilon)$ and $(\tfrac12-\epsilon)^2$ is even about $\tfrac12$, so $p(\epsilon)=p(1-\epsilon)$; pass |
| complete screening | $b\to0$: $g_1\to\tfrac73$, $g_2\to\tfrac{11}6$ (the Tsai complete-screening constants after removing $4\ln(Rm_ec/\hbar)$); pass |
| angular limit | $\beta\to0$: $\cos\theta=2\xi-1$, isotropic; $\beta\to1$: $\cos\theta\to1$; pass |

## Independent construction before code inspection

### DCS shape

Write $L=\ln(1+b^2)$, $A=\arctan(b^{-1})$, $B=4-4bA-3\ln(1+b^{-2})$ and
$\Lambda=\ln(Rm_ec/\hbar)$. From Eq. 2.78,

$$
\Phi_1=2-2L-4bA+4\Lambda,\qquad
\Phi_2=\tfrac43-2L+2b^2B+4\Lambda .
$$

Set $y=(\tfrac12-\epsilon)^2$, so $\epsilon(1-\epsilon)=\tfrac14-y$ and
$\epsilon^2+(1-\epsilon)^2=\tfrac12+2y$. With $\Phi_i'=\Phi_i-4f_C$, the
bracket of Eq. 2.74 becomes

$$
\left(\tfrac12+2y\right)\Phi_1'+\tfrac23\left(\tfrac14-y\right)\Phi_2'
=\frac23\left[2y\,\frac{3\Phi_1'-\Phi_2'}{2}+\frac{3\Phi_1'+\Phi_2'}{4}\right].
$$

Expanding,

$$
\tfrac12(3\Phi_1-\Phi_2)-4\Lambda=\tfrac73-2L-6bA-b^2B=g_1(b),\qquad
\tfrac14(3\Phi_1+\Phi_2)-4\Lambda=\tfrac{11}6-2L-3bA+\tfrac12b^2B=g_2(b),
$$

which reproduces Eq. 2.87 term by term, with $\phi_i=g_i(b)+g_0(\kappa)$ and

$$
g_0=4\ln(Rm_ec/\hbar)-4f_C(Z)+F_0(\kappa,Z),\qquad
b=\frac{Rm_ec}{\hbar}\,\frac{1}{2\kappa\epsilon(1-\epsilon)} .
$$

The Coulomb correction (Eq. 2.80, rendered PDF: $a^2$ multiplies the whole
bracket) is

$$
f_C=a^2\left[(1+a^2)^{-1}+0.202059-0.03693a^2+0.00835a^4-0.00201a^6+0.00049a^8-0.00012a^{10}+0.00003a^{12}\right],
\quad a=\alpha Z .
$$

Eq. 2.88 with $x=2/\kappa$:

$$
F_0=(-1.774-12.10a+11.18a^2)x^{1/2}+(8.523+73.26a-44.41a^2)x
-(13.52+121.1a-96.41a^2)x^{3/2}+(8.946+62.05a-63.41a^2)x^2 .
$$

$\phi_{1,2}$ are set to zero where negative (manual, after Eq. 2.88). The
unnormalized PDF on $(\kappa^{-1},1-\kappa^{-1})$ is Eq. 2.90,

$$
p(\epsilon)=2\left(\tfrac12-\epsilon\right)^2\phi_1(\epsilon)+\phi_2(\epsilon).
$$

Numerically, $g_1$ and $g_2$ are strictly decreasing on
$b\in[10^{-3},10^3]$ (2000 log-spaced points), and $b$ is minimal at
$\epsilon=\tfrac12$, so the clipped $\phi_i$ peak at $\tfrac12$. That is the
condition for the rejection functions below to be bounded by one.

### Composition and rejection

Let $h=\tfrac12-\kappa^{-1}$ and $t=\epsilon-\tfrac12\in(-h,h)$. The normalized
pieces are $\pi_1=\tfrac{3}{2h^3}t^2$ (since $\int_{-h}^{h}t^2dt=\tfrac23h^3$)
and $\pi_2=\tfrac1{2h}$. Requiring
$u_1U_1\pi_1+u_2U_2\pi_2=C\,p(\epsilon)$ with $U_i=\phi_i/\phi_i(\tfrac12)$ gives
$u_1=\tfrac43Ch^3\phi_1(\tfrac12)$ and $u_2=2Ch\,\phi_2(\tfrac12)$. Taking
$C=1/(2h)$,

$$
u_1=\tfrac23h^2\phi_1(\tfrac12),\qquad u_2=\phi_2(\tfrac12),
$$

which is Eq. 2.92. The inverse transforms are

$$
F_1(t)=\frac{t^3+h^3}{2h^3}=\xi\ \Rightarrow\ \epsilon=\tfrac12+h\,(2\xi-1)^{1/3},
\qquad
\epsilon=\kappa^{-1}+2h\xi=\tfrac12+h(2\xi-1),
$$

with $(2\xi-1)^{1/3}$ the real (sign-preserving) cube root, as the manual
warns. Branch $i$ is chosen with probability $u_i/(u_1+u_2)$; accept if a fresh
$\xi\le U_i(\epsilon)$.

The manual does not cover the case $\phi_1(\tfrac12)=\phi_2(\tfrac12)=0$, where
the DCS is zero everywhere. With the reference implementation below this
happens only for $Z\ge85$, below $1.031$ MeV ($Z=85$) to $1.087$ MeV
($Z=99$). Any sampling rule there is a convention, not PENELOPE physics.

### Kinematics and angles

$E_-=\epsilon E-m_ec^2$ and $E_+=E-E_--2m_ec^2=(1-\epsilon)E-m_ec^2$. For
$p(c)\propto(1-\beta c)^{-2}$ on $[-1,1]$,

$$
\int_{-1}^{c}\frac{dc'}{(1-\beta c')^2}=\frac{1+c}{(1+\beta)(1-\beta c)},\qquad
\text{total}=\frac{2}{1-\beta^2},
$$

so $\xi=\dfrac{(1-\beta)(1+c)}{2(1-\beta c)}$, which inverts to

$$
\cos\theta_\pm=\frac{2\xi-1+\beta_\pm}{(2\xi-1)\beta_\pm+1},\qquad
\beta_\pm=\frac{\sqrt{E_\pm(E_\pm+2m_ec^2)}}{E_\pm+m_ec^2},
$$

which is Eqs. 2.98–2.99. Each direction is
$\hat{\mathbf d}_\pm=\cos\theta_\pm\hat{\mathbf k}+\sin\theta_\pm(\cos\varphi_\pm\hat{\mathbf e}_1+\sin\varphi_\pm\hat{\mathbf e}_2)$
for any orthonormal $\hat{\mathbf e}_{1,2}\perp\hat{\mathbf k}$, with independent
$\varphi_\pm$ uniform on $[0,2\pi)$.

## Comparison with the implementation

| item | independent | `pair_production.py` | result |
| --- | --- | --- | --- |
| $g_1$, $g_2$, $b$ | above | lines 127–132 | identical |
| $g_0$ | $4\Lambda-4f_C+F_0$ | line 133 | identical |
| clipping | $\max(\phi_i,0)$ | line 134 | identical |
| $f_C$ | Eq. 2.80 | lines 96–108 | identical; max relative difference $4\times10^{-16}$, $Z=1$–99 |
| $F_0$ | Eq. 2.88, $-1.774$ | lines 111–120 | identical; max absolute difference $1.4\times10^{-14}$ |
| Table 2.2 radii | parsed from the extracted manual table | lines 71–87 | all 99 identical (max difference $0$) |
| $p(\epsilon)$ and support | Eq. 2.90, open interval | lines 137–149 | max relative difference $1.1\times10^{-13}$ (8 $Z$ × 7 energies × 2001 points) |
| $u_1$, $u_2$, $p(1)$ | Eq. 2.92, 2.95 | lines 165–169 | identical |
| branch 1 | $\tfrac12+h\,\mathrm{cbrt}(2\xi-1)$ | line 174, `np.cbrt` | identical, negative argument handled |
| branch 2 | $\tfrac12+h(2\xi-1)$ | line 177 | identical to Eq. 2.96 |
| rejection | $\xi\le\phi_i(\epsilon)/\phi_i(\tfrac12)$ | lines 175, 178–179 | identical; a zero $u_i$ gives that branch probability zero, so no division by zero |
| empty DCS | convention | lines 167–168: uniform on $(\kappa^{-1},1-\kappa^{-1})$ | not in the manual; documented in the docstring and ledger assumptions |
| $E_-$, $E_+$ | $\epsilon E-m_ec^2$, $E-E_--2m_ec^2$ | lines 227–228 | identical; `max(..., 0)` only guards rounding |
| $\beta$, Eq. 2.99 | above | lines 190–192 | identical, bit-for-bit on 200 000 draws at each of five energies |
| rotation, azimuths | orthonormal frame about $\hat{\mathbf k}$ | lines 195–203, 230–233 | $\lvert\hat{\mathbf d}\cdot\hat{\mathbf k}-\cos\theta\rvert\le2.2\times10^{-16}$ |

`_ELECTRON_REST_EV = 510998.95` matches CODATA $m_ec^2$, and
`_ALPHA = 1/137.035999084` matches CODATA $\alpha$.

### $F_0$ leading coefficient against independent data

Eq. 2.85 with $\eta=0$ and $C_r=1.0093$ was integrated with the independent
reference above, $\sigma=r_e^2\alpha Z^2C_r\tfrac23\int p\,d\epsilon$, and
compared with EPDL2025 `pair_nuclear` from
`photon_cross_sections_ang2`. Relative deviation
$\sigma_{\rm Eq.\,2.85}/\sigma_{\rm EPDL}-1$:

| $E$ (MeV) | C, $-1.774$ | C, $-0.1774$ | Si, $-1.774$ | Mo, $-1.774$ | Pb, $-1.774$ | Pb, $-0.1774$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1.2 | $+0.87$ | $+20.9$ | $+1.30$ | $+0.91$ | $+0.20$ | $+8.4$ |
| 1.5 | $-0.115$ | $+3.65$ | $-0.062$ | $-0.099$ | $-0.077$ | $+1.73$ |
| 2 | $-0.051$ | $+1.23$ | $-0.024$ | $-0.032$ | $-0.039$ | $+0.77$ |
| 3 | $+0.009$ | $+0.52$ | $+0.026$ | $+0.023$ | $+0.009$ | $+0.43$ |
| 5 | $+0.018$ | $+0.24$ | $+0.027$ | $+0.023$ | $+0.010$ | $+0.24$ |
| 10 | $-0.003$ | $+0.09$ | $+0.001$ | $-0.001$ | $-0.005$ | $+0.10$ |
| $10^5$ | $-0.000$ | $-0.000$ | $-0.001$ | $-0.001$ | $-0.001$ | $-0.001$ |

The printed $-1.774$ is correct. It reproduces the manual's statement (4 %
at 3 MeV, about 2 % above 6 MeV, large near threshold). The $10^5$ MeV row
agrees to 0.1 %, which independently confirms the Table 2.2 radii, $f_C$ and
$C_r$ where $F_0\to0$. With $-0.1774$ the total exceeds EPDL by a factor of
1.8–2.2 at 2 MeV, 2.7–4.6 at 1.5 MeV and 9–22 at 1.2 MeV.

### Sampling

The code sampler was compared with 200 000 draws per case, independent seeds,
and 40 equal bins on $(\kappa^{-1},1-\kappa^{-1})$. Expected counts came from
quadrature of the independent $p(\epsilon)$:

| $Z$ | 1.1 MeV | 1.5 MeV | 3 MeV | 10 MeV | 1 GeV |
| --- | --- | --- | --- | --- | --- |
| 6 | $\chi^2=43.7$, $p=0.28$ | $28.2$, $0.90$ | $46.9$, $0.18$ | $27.6$, $0.91$ | $29.8$, $0.86$ |
| 82 | $17.3$, $1.00$ | $39.3$, $0.46$ | $49.1$, $0.13$ | $41.6$, $0.36$ | $44.4$, $0.25$ |

(39 degrees of freedom.) A KS test of `sample_pair_polar_cosine` against the
analytic CDF $(1-\beta)(1+c)/[2(1-\beta c)]$ gives $p=0.38$ at each of five
kinetic energies from 0 to 50 MeV. Over 3000 `sample_pair` events,
$\lvert E_-+E_++2m_ec^2-E\rvert/E\le2.2\times10^{-16}$, no energy is negative,
and $\lvert\lVert\hat{\mathbf d}\rVert-1\rvert\le2.2\times10^{-16}$.
`tests/montecarlo/test_pair_production.py`: 22 passed.

## Findings

No discrepancy against PENELOPE-2024 §2.4. The following are not
discrepancies:

1. **Uniform fallback** (`pair_production.py:167-168`). This is a convention
   for the zero-DCS window, which occurs only for $Z\ge85$ below 1.031–1.087 MeV.
   It keeps energy conserved and is disclosed. The ledger should record the
   affected window.
2. **Input direction not normalized** (`pair_production.py:195-203`).
   `_rotate` builds the frame from `direction` without normalizing it. The
   output is renormalized, but if the input is not a unit vector, its
   polar angle about $\hat{\mathbf k}$ is not $\theta$. Callers pass unit
   photon directions, so this does not change the physics.
3. **$Z=100$** (`pair_production.py:90-93`). Table 2.2 stops at $Z=99$ and
   EPDL extends to $Z=100$, so a fermium-bearing layer would raise a
   `ValueError`. No catalog material is affected.
4. **Ledger wording.** "$-0.1774$ overestimates them two- to fourfold near
   threshold" understates the size. The factor is about 1.8–2.2 at 2 MeV, 2.7–4.6 at
   1.5 MeV and 9–22 at 1.2 MeV.

## Verdict

`rederived`. Every term, coefficient, table entry, sampling formula, and
kinematic and angular relation matches an independent derivation from
PENELOPE-2024 §2.4. The printed $F_0$ coefficient $-1.774$ is independently
confirmed against EPDL2025.

## Resolution of findings

Addressed after this validation on branch `issue-275-pair-daughters`: `_rotate`
normalizes its input direction; the docstring of
`sample_pair_reduced_energy` and the ledger state the uniform-$\epsilon$
fallback window ($Z\ge85$ below 1.031–1.087 MeV) as a PyRITE convention; the
ledger quotes the measured $-0.1774$ overestimates. Table 2.2 stops at
$Z=99$, so a $Z=100$ layer raises; no catalog material is affected.

## Anchoring (2026-10-02)

The ledger row moved from `rederived` to `anchored` after the verdict above.
`tests/montecarlo/test_pair_production.py` now also pins:

- `sample_pair` end to end at 1.5, 4 and 10 MeV for Z = 6 and 82. Energy
  closes to two ulp. Each particle's cosine follows Eq. 2.99 at its own
  $\beta$ (probability-integral transform, KS), and the azimuth is uniform.
- The binned energy share and both polar cosines, compared with the committed
  Geant4 `empenelope` references in `checks/pair_production_geant4/reference/`
  (C and Pb, 2 and 5 MeV, two-sample chi-square).
