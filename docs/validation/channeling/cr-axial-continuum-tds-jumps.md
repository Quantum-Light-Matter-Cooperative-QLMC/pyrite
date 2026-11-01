# `cr-axial-continuum-tds-jumps`: independent-string TDS jumps on carrier fibres

**Status:** task-owner derivation (2026-10-31), awaiting fresh-context verification.
Code: `src/pyrite/montecarlo/channeling/axial_tds_continuum.py`. Integration
into the fibre unraveling: `axial_fiber_unraveling._jump` (claim
`cr-axial-fiber-unraveling`).

## Why

The one-cell lattice TDS model, `checks/axial_matching_study.py::_kraus` at
zero shift, makes every string vibrate in lockstep with its periodic images.
Their scattered amplitudes interfere, which confines transfers to reciprocal
lattice vectors. TDS then never moves the Bloch carrier, and each scattered
wave is spread over all images of the struck string. An $N\times N$
supercell only pushes the same correlation out to images $N$ cells apart.
Real thermal displacements of different strings are independent.

## Derivation

**Single-string jump.** Take a string at $\mathbf r_s$ with in-plane
displacement $u\sim N(0,a^2I)$ and linear density $\lambda$. Its phase-shift
fluctuation has Fourier coefficients $c(\mathbf t)w_{\mathbf t}(u)$, where

$$
c(\mathbf t)=\varphi(|\mathbf t|)e^{-i\mathbf t\cdot\mathbf r_s}/A,\qquad
w_{\mathbf t}(u)=e^{-i\mathbf t\cdot u}-e^{-a^2t^2/2}.
$$

This is the dissipator's `tds_phase`: the displaced phase minus its thermal
mean. The jump term is $\lambda\,\mathbb E_u\,J\rho J^\dagger$, with $J$
acting as multiplication by that fluctuation.

**Sum over independent images.** Image $\mathbf R$ carries
$e^{-i\mathbf t\cdot\mathbf R}$ and its own displacement. Using
$\sum_{\mathbf R}e^{-i(\mathbf t-\mathbf t')\cdot\mathbf R}=\frac{(2\pi)^2}{A}\sum_{\mathbf G}\delta(\mathbf t-\mathbf t'-\mathbf G)$
gives

$$
\sum_{\mathbf R}\lambda\,\mathbb E_u\,J_{\mathbf R,u}\rho J_{\mathbf R,u}^\dagger
=\int_{\mathbb R^2}\frac{d^2Q}{(2\pi)^2}\;\text{(output at }Q\text{)}
=\int_{\rm BZ}\frac{d^2\Delta}{(2\pi)^2}\,A\,\mathbb E_u\,J_{\Delta,u}\rho J_{\Delta,u}^\dagger .
$$

Here $J_{\Delta,u}=e^{i\Delta\cdot\mathbf r}\times$ (a cell-periodic
multiplication whose order-$\mathbf n$ harmonic is
$\sqrt\lambda\,c(\Delta+\mathbf g_{\mathbf n})w_{\Delta+\mathbf g_{\mathbf n}}(u)$).
It moves the carrier by $\Delta$, and $\Delta$ is continuous. Each pair
$(\Delta,u)$ is one Kraus operator under the measure
$A\,d^2\Delta/(2\pi)^2\,N(u)\,d^2u$, so the displacement average is itself
the factorization and no eigendecomposition is needed. Restricting $\Delta$
to a $c\times c$ grid recovers the $c\times c$-supercell model of
`bloch_tds_rates`.

**Sampling density.** For a window state $\chi$ with orders
$\mathbf g_p$ relative to the carrier, write the output momentum $Q$
relative to the carrier. The joint density of $(Q,u)$ is

$$
f(Q,u)\propto\Big|\sum_p\chi_p\,c(Q-\mathbf g_p)\,w_{Q-\mathbf g_p}(u)\Big|^2N(u),
$$

and its marginal over $Q\equiv\Delta$ (mod $\mathbf G$) is the rate
$A\,\mathbb E_u\lVert J_{\Delta,u}\chi\rVert^2$.

**Exact rejection.** Pointwise in $u$, the weighted Cauchy–Schwarz
inequality gives
$|\sum_p\chi_pc_pw_p|^2\le(\sum|\chi|)\sum_p|\chi_p||c_p|^2|w_p|^2$.
Take a radial envelope $\hat h(q)\ge|\varphi(q)|^2(1-e^{-a^2q^2})/A^2$:
piecewise constant on geometric bins, using $|\varphi(q_i)|$ at the left
edge, which needs $|\varphi|$ nonincreasing (checked). Then
$|c|^2|w|^2\le\hat h|w|^2/(1-e^{-a^2q^2})$. The proposal is:

- pick $p\propto|\chi_p|$;
- draw $q=Q-\mathbf g_p$ from $\hat h$: a bin, then uniform in area, then a uniform angle;
- draw $u\mid q$ from $|w_q(u)|^2N(u)/(1-e^{-a^2q^2})$.

That proposal has density proportional to the bound, so accepting with
probability target/bound samples $f$ exactly.

For the displacement draw, write $s=\mathbf q\cdot u\sim N(0,\sigma^2)$ with
$\sigma=a|q|$ and $\chi_q=e^{-\sigma^2/2}$. Then
$|w|^2=(1-\chi_q)^2+4\chi_q\sin^2(s/2)$, a positive mixture with weights
$(1-\chi_q)/(1+\chi_q)$ (draw $s$ from $N$) and $2\chi_q/(1+\chi_q)$ (draw
$s\propto\sin^2(s/2)N$). The second component is sampled by rejection,
from $s^2N$ (a signed $\chi_3$) when $\sigma<2$, and from $N$ otherwise.
The perpendicular component of $u$ stays $N(0,a^2)$.

**Unresolved by where $Q$ lands.** Given $(\Delta,u)$,
$P(Q\in\text{box})=\lVert\mathcal BJ_{\Delta,u}\chi\rVert^2/\lVert J_{\Delta,u}\chi\rVert^2$.
So "unresolved" (output momentum outside the outgoing box) is decided
exactly by whether the sampled $Q$ lands outside the box, with no tail sum.
The box membership of $Q$ is its fold shift. Otherwise the box output
$\mathcal BJ_{\Delta,u}\chi$ is the linear convolution of $\chi$ with the
column above at $\delta=\kappa'-\kappa$. It is then split into retained
$P(\kappa')$ and outgoing $\mathcal B-P$. This needs no transfer
truncation, no rank cutoff and no $D\ge B+M$ condition.

**Loss.** $\int f$ gives $\chi^\dagger\Gamma\chi$, with

$$
\Gamma_{ab}=\frac{\lambda}{A(2\pi)^2}e^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r_s}D(|\mathbf g_a-\mathbf g_b|),\quad
D(G)=\int d^2k\,\varphi(|k-\tfrac G2|)\varphi(|k+\tfrac G2|)\big[e^{-a^2G^2/2}-e^{-a^2(k^2+G^2/4)}\big].
$$

$D$ is the production $D(G)$ of the escape research note's continuum
closure. It is evaluated by midpoint-centred polar Gauss–Legendre
quadrature (32 radial nodes per geometric panel, 192 angles), with factors
beyond $q_{\max}=10^4$ Å$^{-1}$ set to zero to match the sampler.
Independent strings add: $\Gamma=\sum_s\Gamma_s$, and the group weights in
`_jump` are $\chi^\dagger\Gamma_s\chi$.

**Limits.**
- $a^2=0$ or $\lambda=0$ gives zero rate.
- A $c\times c$ grid in $\Delta$ is the supercell model, and the midpoint
  rule in $\Delta$ is spectrally accurate because $f$ is smooth and periodic
  in $\Delta$.
- A plane-wave input gives the single-atom incoherent rate
  $\lambda D(0)/(A(2\pi)^2)$: 6.428949e-5 Å$^{-1}$ for Si [001], 5 MeV,
  against 6.428955e-5 from the research note's independent closure.

## Evidence

All of it is in `tests/montecarlo/quantum/test_channeling_axial_tds_continuum.py`.

- **$D(G)$** matches the Gaussian closed form to 1e-10.
- **$\Gamma$ equals the zone integral** of the independent shifted
  factorization `_kraus(shift=Δ)`, $A\sum_j\lVert S_\Delta\chi\rVert^2$, to
  1e-9. This fixes the normalization and the string-phase sign.
- **Single-jump outcomes.** 4,000 jumps from a random 9-order state, with an
  off-centre string, match the factorized reference integrated over the
  zone: the unresolved fraction, and every entry of the normalized
  box-output density matrix, within 5σ.
- **Displacement law.** $s=\mathbf q\cdot u$ follows the target law (KS
  test) on both branches.
- **Supercell equivalence.** A free-particle crystal written as a 1×1 cell
  with 1 string and as a 2×2 supercell with 4 strings gives the same
  retained weight versus depth and the same mean TDS jump count (both are
  unraveling invariants), within 4.5σ over 1,500 histories each.
- **Refusals:** a nonmonotone phase, negative inputs, and sampling a zero rate.
- **Injected faults, all detected:**
  - string-phase sign in $\Gamma$;
  - string phase dropped from the column, or from the sampler target;
  - column evaluated at zero carrier shift;
  - mean subtracted from $w$ dropped;
  - parallel displacement halved.

Material cross-checks (Si [001], 5 MeV):

- $D(G)$ agrees with production `operators._tds_transform` to 1e-6 for
  $G\le13$ Å$^{-1}$. At $G=26$–39 Å$^{-1}$ the production 24-angle rule is
  off by 1–3e-4 relative. Doubling the angular nodes (32×192 → 32×384) changes
  $D$ by ≤1e-8. An independent adaptive quadrature differs by up to 1.5e-6
  at $G=26$ Å$^{-1}$ (radial error), well below the production discrepancy.
- Over the 77 retained states, the one-cell lattice model's mean TDS rate is
  1.7% below the continuum (2.3% for the entry state). That is the
  midpoint-rule error of a single $\Delta=0$ node.

## Limitations

- Isotropic Gaussian displacement, uncorrelated between strings.
- Transfers capped at $q_{\max}$ (tail beyond ~$10^{-10}$ relative).
- $|\varphi|$ must be nonincreasing for the envelope; this is checked on
  the sampling grid. The production potential is log-linear between table
  nodes, so this holds.
- The absorbing no-return model of the fibre unraveling.

This claim does not supply matching, return flux, stopping, a clock, or #624
integration.

## Fresh-context verification (2026-10-31)

**Verdict: `rederived`.** Also covers the integration into
`cr-axial-fiber-unraveling` (`_Engine.__init__`, `_jump`, `_trajectory`).
The derivation below was written from the ledger row, the module docstrings,
the research note's continuum closure and the conventions of
`checks/axial_matching_study.py`. Only then were the implementation bodies
and the task-owner derivation above read. Commit `8d6de11ea`.

- **Claim**: `cr-axial-continuum-tds-jumps`, `montecarlo/channeling/axial_tds_continuum.py::{TdsString,overlap_transform,StringChannel,StringChannel.sample,StringChannel.column,string_jump}`. Source: Castin–Dalibard–Mølmer Eqs. (2), (4), (8), plus the single-string fluctuation of `cr-axial-realspace-dissipator`.
- **Filters**: units pass ($\Gamma$ in Å$^{-1}$: $\lambda$ [Å$^{-1}$] $\times$ $D$ [Å$^{2}$] / ($A$ [Å$^2$])); limits pass ($a^2=0$ or $\lambda=0$ gives zero; a plane wave gives $\lambda D(0)/(A(2\pi)^2)$; the supercell midpoint rule converges to the continuum); signs and conventions pass (string phase, $D$ symmetric in $\pm\mathbf G$, Hermitian Toeplitz $\Gamma$).
- **Re-derivation**: `matches`.

### Independent derivation

Work in a periodic box of area $L^2$ with $N=L^2/A$ cells and plane waves
$|\mathbf k\rangle=e^{i\mathbf k\cdot\mathbf x}/L$. The string at
$\mathbf R+\mathbf r$ with displacement $u$ has

$$
\langle\mathbf k'|J_{\mathbf R,u}|\mathbf k\rangle
=\frac{1}{L^2}\varphi(|\mathbf q|)e^{-i\mathbf q\cdot(\mathbf R+\mathbf r)}w_{\mathbf q}(u),
\qquad \mathbf q=\mathbf k'-\mathbf k .
$$

**Gain.** Take an input in fibre $\kappa$ with
$\rho=\sum\rho_{pp'}|\kappa+\mathbf g_p\rangle\langle\kappa+\mathbf g_{p'}|$.
The bra and ket transfers $\mathbf q,\mathbf q'$ enter through
$\sum_{\mathbf R}e^{-i(\mathbf q-\mathbf q')\cdot\mathbf R}=N\,\delta_{\mathbf q-\mathbf q'\in\{\mathbf G\}}$.
Every output pair therefore lies in one fibre $\kappa+\Delta$. Writing the
outputs as $\kappa+\Delta+\mathbf g_n$ gives $\mathbf t_{np}=\Delta+\mathbf g_{n-p}$.
The sum over the discrete $\Delta$ becomes
$\sum_\Delta\to L^2\int d^2\Delta/(2\pi)^2$, and the $L$ factors cancel:

$$
\sum_{\mathbf R}\lambda\,\mathbb E_u J_{\mathbf R,u}\rho J_{\mathbf R,u}^\dagger
=\int_{\rm BZ}\frac{A\,d^2\Delta}{(2\pi)^2}\,\mathbb E_u\,J_{\Delta,u}\rho J_{\Delta,u}^\dagger,
\qquad
\langle\kappa+\Delta+\mathbf g_n|J_{\Delta,u}|\kappa+\mathbf g_p\rangle
=\sqrt\lambda\,c(\mathbf t_{np})\,w_{\mathbf t_{np}}(u).
$$

Here $c(\mathbf t)=\varphi(|\mathbf t|)e^{-i\mathbf t\cdot\mathbf r}/A$. The
amplitude squared, $A\lambda|\varphi|^2/A^2=\lambda|\varphi|^2/A$, is what
the box calculation gives, so the factor $A$ and the $(2\pi)^2$ are fixed.
On an $M\times M$ supercell with independent images, $\Delta$ is restricted
to the $\mathbf g/M$ grid with weight $1/M^2$ per node: the midpoint rule.
Equivalently, the image-summed squared amplitudes equal one string with
effective area $MA$. This is the `area = cells * area_ang2` of
`bloch_tds_rates`, whose `cells` counts per axis. For $M=1$ the single node
$\Delta=0$ is `_kraus` at zero shift.

**Loss.**

$$
\langle\kappa+\mathbf g_a|\Gamma|\kappa+\mathbf g_b\rangle
=\sum_{\mathbf R}\lambda\,\mathbb E_u\sum_{\mathbf k'}
\langle\kappa+\mathbf g_a|J^\dagger|\mathbf k'\rangle\langle\mathbf k'|J|\kappa+\mathbf g_b\rangle .
$$

The phases combine to $e^{i(\mathbf q_a-\mathbf q_b)\cdot(\mathbf R+\mathbf r)}$
with $\mathbf q_a-\mathbf q_b=\mathbf g_b-\mathbf g_a$. Also
$\mathbb E_u[w_{\mathbf q_b}w_{\mathbf q_a}^*]=e^{-a^2|\mathbf g_a-\mathbf g_b|^2/2}-\chi_{q_a}\chi_{q_b}$.
Substituting the midpoint $\mathbf k=\mathbf k'-\kappa-(\mathbf g_a+\mathbf g_b)/2$ gives

$$
\Gamma_{ab}=\frac{\lambda}{A(2\pi)^2}\,e^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r}\,D(|\mathbf g_a-\mathbf g_b|),
$$

with $D(G)$ as stated. This holds for real isotropic $\varphi$. The sign
agrees with `loss_operator`, whose $\langle p|\Gamma|p'\rangle\propto c_nc^*_{n-G}\propto e^{-i(\mathbf g_p-\mathbf g_{p'})\cdot\mathbf r}$.
It also agrees with the production harmonic
$\lambda A^{-1}e^{-i\mathbf G\cdot\mathbf R}T(G)/(2\pi\hbar v)^2$.

**Sampler.** With $Q=\Delta+\mathbf g_n\in\mathbb R^2$ the joint rate density
is $\frac{A}{(2\pi)^2}\lambda\,|\sum_p\chi_pc(Q-\mathbf g_p)w_{Q-\mathbf g_p}(u)|^2N(u)$.

1. Weighted Cauchy–Schwarz,
   $|\sum\chi_px_p|^2\le(\sum|\chi|)\sum|\chi_p||x_p|^2$, is pointwise in $(Q,u)$.
2. $|\varphi(t)|^2(1-e^{-a^2t^2})\le|\varphi(e_i)|^2(1-e^{-a^2e_{i+1}^2})$ on
   $[e_i,e_{i+1})$ for nonincreasing $|\varphi|$. The factor $|w|^2/(1-\chi_t^2)$
   is common to the bound and the proposal, so dominance holds pointwise in $u$.
3. The proposal picks $p\propto|\chi_p|$, a bin $\propto$ bound $\times$ annulus
   area, $t$ uniform in area, a uniform angle, and $u\mid t$ from
   $|w_t|^2N/(1-\chi_t^2)$. Its density is
   $\propto\sum_p|\chi_p|\,\hat h(t_p)|w_{t_p}|^2N/(1-\chi_{t_p}^2)$, which is the
   bound up to a constant. The marginal mass of each $p$ term is the same
   because $\int|w_t|^2N=1-\chi_t^2$.
4. For $u\mid t$:
   $|w|^2=1-2\chi\cos s+\chi^2=(1-\chi)^2+4\chi\sin^2(s/2)$, with
   $s=\mathbf t\cdot u\sim N(0,a^2t^2)$ and $\chi=e^{-a^2t^2/2}$. Using
   $\mathbb E\sin^2(s/2)=(1-\chi)/2$, the branch weights are $(1-\chi)/(1+\chi)$
   and $2\chi/(1+\chi)$. The $\sin^2$ branch by rejection is exact from the
   signed $\chi_3$ proposal $\propto s^2N$ (accept $\sin^2(s/2)/(s/2)^2\le1$) and
   from $N$ (accept $\sin^2(s/2)\le1$). The perpendicular component stays $N(0,a^2)$.

**Unresolved and box output.** Sampling $(Q,u)$ jointly is equivalent to
sampling $(\Delta,u)$ with weight $\lVert J_{\Delta,u}\chi\rVert^2$ and then
the order $n$ with probability $|\psi_n|^2/\lVert J\chi\rVert^2$. So
$P(Q\in\mathcal B\mid\Delta,u)=\lVert\mathcal BJ_{\Delta,u}\chi\rVert^2/\lVert J_{\Delta,u}\chi\rVert^2$.
The post-decision state $\mathcal BJ\chi/\lVert\mathcal BJ\chi\rVert$ does
not depend on $n$. The box consists of orders $|o|\le B$ about the folded
carrier $\kappa'$. In absolute momentum it is a fixed set of
$(2B+1)^2$ cells about the first zone, so $Q$ lies in the box exactly when
its fold shift $s$ satisfies $|s|_\infty\le B$, with $\kappa'=\kappa+Q-s\mathbf g$.
The box amplitude at order $o$ is
$\psi_o=\sum_p\chi_p\,\mathrm{col}[o-p]$ with
$\mathrm{col}[n]=c\,w$ at $\Delta+\mathbf g_n$ and $\Delta=\kappa'-\kappa$.
The orders satisfy $|o-p|\le B+M$, so linear convolution with FFT size
$(2M+1)+2(B+M)$ has no wrap. Order $o$ sits at index $o+M+(B+M)$.

### Comparison with the implementation

All are matches (`src/pyrite/montecarlo/channeling/axial_tds_continuum.py` unless noted):

- $\Gamma$: lines 118–126. `phase = exp(-1j*(g_ab@r))` with `g_ab = g_a - g_b` is $e^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r}$, with coefficient $\lambda/(A(2\pi)^2)$ and $A=4\pi^2/(g_xg_y)$.
- $D(G)$: lines 63–87. Half-circle polar quadrature doubled. The truncation to $|k|\le q_{\max}$ is exact because both factors nonzero imply $|k|^2+G^2/4\le q_{\max}^2$.
- Envelope: lines 109–117 and 140–143 (left-edge $|\varphi|$, right-edge $1-e^{-a^2e^2}$). Proposal: lines 180–188. Acceptance: line 192. Violation guard: lines 190–191. Any proposal with target above the bound raises, so a non-monotone $\varphi$ between grid nodes cannot bias draws silently.
- Displacement: lines 146–168. `one_minus` $=1-\chi$; branch test `rng*(1+chi) < 1-chi`.
- Fold, box and convolution: lines 220–236. Fold: `axial_fiber_unraveling.py:162-166`.

**Integration** (`src/pyrite/montecarlo/channeling/axial_fiber_unraveling.py`):

- Window orders are built `ij` / C order, matching `chi.reshape(width, width)` (lines 209–211).
- Each $\Gamma_s$ is added to the no-jump loss (212–213).
- `_jump` group weights are $\chi^\dagger\Gamma_{\rm lat}\chi$, $\chi^\dagger\Gamma_s\chi$ and $\lVert S_i\chi\rVert^2$. Their sum is $\chi^\dagger\Gamma_{\rm tot}\chi$, matching the waiting law (323–330).
- A string output is exact on the box, so `_split` uses `outside = 0`. The new fibre is built at $\kappa'$ before splitting (336–350).
- With no strings the rate vector, draw order and kick/lattice branches are unchanged. `np.maximum` is a no-op on the kick Parseval sums.
- `_trajectory` counts string jumps in `lattice_jumps` and terminates on `choice is None` (412–416).

### Independent numerical evidence

Own scripts, `val_cont_*.py` in the session scratchpad. Parameters differ from
the tests: non-square $\mathbf g=(0.83,1.71)$ Å$^{-1}$, off-centre
$\mathbf r=(-0.61,0.94)$ Å, $a^2=0.21$ Å$^2$, $\lambda=1.37$ Å$^{-1}$, and a
non-Gaussian Lorentzian-type phase $\varphi=2.3/(1+0.8q^2)^{3/2}$.

| Check | Result |
|---|---|
| $\Gamma$ (3×3 window) vs own adaptive Cartesian `dblquad` of $D$ with own phase factor | max rel. diff 2.4e-13 |
| Rate vs direct image sum on $M\times M$ supercells, independent strings, $\mathbf g/M$ outputs, off-zero carrier | $M=4$: −1.0e-5; $M=6$, wide output box: −1.0e-8; Gaussian-$u$ MC of the same sum ($M=4$, 400 draws): 0.3σ |
| $Q$ law of `sample` (64 bins + outside) vs own $u$-averaged density quadrature, which closes on the rate to 5e-7 | 3×3 window, 60k: $\chi^2=72.9$/64; 5×5 window, 40k: $55.7$/64. A first 30k run had $\chi^2=75.7$/64 and an outside-bin pull of −3.2σ; two further 100k runs gave −0.8σ and +0.6σ, and the 60k run gives +0.1σ |
| Joint $(Q,u)$: $\mathbb E[1_{Q\in\text{region}}\cos/\sin(\mathbf k\cdot u)]$ vs own analytic Gaussian moments, two $\mathbf k$, square and half-plane regions | all $\lvert z\rvert\le1.3$ (60k and 40k runs) |
| `_displacement` at $\sigma=0.18,1.83,2.11,5.96$ (both rejection regimes) | KS $p\ge0.44$ along $\mathbf t$, $p\ge0.32$ perpendicular, no correlation |
| `string_jump` box output vs own direct sum over $(o,p)$ from the same $(Q,u)$, $B=3$, off-centre carrier | 3754 outputs, max rel. diff 1.5e-15; own fold matches $\kappa'$ and every unresolved decision |
| Unresolved fraction vs own quadrature of the density outside the absolute box | 0.0615 vs 0.0604 ($z=+0.28$, 4000 jumps) |
| No-string unraveling vs pre-change engine (`HEAD~1` module), lattice + kicks, $B=2>M=1$, uniform and discrete azimuths | 2×300 events bitwise identical |
| Si [001] $D(G)$ vs own clustered-angle quadrature (converged ≤3e-7) | code within 3.5e-8 at $G\le10$ and 1.5e-6 at 26.2 Å$^{-1}$; production `_tds_transform` −1.2e-4 at 26.2 and −3.2e-4 at 39.3 Å$^{-1}$ |
| Si plane-wave rate | 6.428950e-5 Å$^{-1}$ ([001]), the same for [110] (4 strings): $\sum_s\lambda_s/A$ equal; $\Gamma_s$ PSD (min eigenvalue > 0, $M=6$) |
| Anchors: `pyrite-dev test` on the continuum, fibre-unraveling and ledger tests | 44 passed |

The material path `checks/axial_outgoing_maps.py::material_fiber_channels(tds="strings")`
creates one `TdsString` per `dissipator.tds_sources` entry. Each takes the
cell-frame position, the per-element linear density, the per-axis $a^2$ and
$\varphi=-V/\hbar v$; the sign of $\varphi$ is a global phase. Lattice columns
and loss are zero in that mode. The [001], [110] and [111] profiles are
rectangular, with 1, 4 and 2 sources, as the cell assumption requires.

### Findings (none blocking)

1. The ledger row's material line is confirmed: production `_tds_transform`
   is 1–3e-4 low at $G\gtrsim26$ Å$^{-1}$. At $G=26.2$ the code and my
   quadrature differ by 1.5e-6, against my own convergence of 3e-7. That is
   looser than the "≤1e-8" in the evidence above, and immaterial.
2. Monotonicity of $|\varphi|$ is checked only on edges and midpoints
   (lines 110–113). Exactness is still protected by the runtime envelope
   guard (lines 190–191).
3. `_jump` clips the string rates at zero (`axial_fiber_unraveling.py:326`),
   but the no-jump loss uses the unclipped $\Gamma_s$. This is moot here
   because the quadrature $\Gamma_s$ is PSD for Si (min eigenvalue
   5e-7 Å$^{-1}$ [001], 2e-11 Å$^{-1}$ [110]).
4. Efficiency: the incoherent bound can be loose by up to the window size
   $K$, and the 1e6-trial budget raises instead of biasing. Each string jump
   lands on a fresh continuous carrier, so every such jump misses the fibre
   cache. Both are performance issues only.

## Implementation change: per-axis window and box (2026-11-01, rederived)

`string_jump` and `StringChannel.column` take per-axis extents $(M_x,M_y)$ and $(B_x,B_y)$ (an integer still means square), for the rectangular windows of #880. The box is a product of intervals, so the unresolved decision is $\lvert\mathrm{fold}(\mathbf Q)_i\rvert>B_i$ on either axis. The column half-width is $B_i+M_i$, the FFT size is $2M_i+1+2(B_i+M_i)$, and output order $o_i$ sits at index $o_i+M_i+(B_i+M_i)$, per axis. The sampler and $\Gamma$ already took arbitrary order lists and are unchanged. Square inputs are bitwise unchanged (`string_jump` outputs and string unravelings vs `a259d557e`).

Evidence: `test_single_jump_outcomes_match_factorized_reference[rectangular]` (window $(1,0)$, box $(2,1)$, $g=(1.3,0.9)$: unresolved fraction and full box density matrix within 5σ; swapping the window extents in the crop offset fails it). `test_loss_is_the_zone_integral_of_shifted_lattice_factorizations[rectangular]` (window $(2,1)$, 1e-9). `test_rectangular_window_supercell_reproduces_the_cell_retained_law` ($g=(1.1,2.2)$, window $(2,1)$, box $(3,2)$ vs a 2×1 supercell with window $(4,1)$, box $(6,2)$).

## Fresh-context re-verification (2026-11-01, #880)

Fresh verification context, commit `f5108dccc` (change `a259d557e..f5108dccc`).
The per-axis box test, half-width and crop were derived before the changed
bodies were read; the derivation is (c) of the
[fibre re-verification](cr-axial-fiber-unraveling.md#fresh-context-re-verification-2026-11-01-880).

**Verdict: `rederived`.** The status stays `anchored`. Human sign-off remains in #277.

- **Claim**: `cr-axial-continuum-tds-jumps`, `montecarlo/channeling/axial_tds_continuum.py::{StringChannel.column,string_jump}` (the per-axis extents; `order_extents` parses them).
- **Filters**: units pass (unchanged); limits pass (integer or equal-pair extents are bitwise the square path); signs and conventions pass (C order with $x$ outer; fold shift per axis).
- **Re-derivation**: `matches`. The box is $\prod_i[-B_i,B_i]$ about the folded carrier, so the jump is unresolved iff $\lvert s_i\rvert>B_i$ on either axis. Box outputs need transfers $\lvert o_i-p_i\rvert\le B_i+M_i$, so the column half-width is $B_i+M_i$. The wrap-free FFT size is $2M_i+1+2(B_i+M_i)$, output $o_i$ sits at $o_i+M_i+(B_i+M_i)$, and the crop starts at $2M_i$. Code: `np.any(np.abs(shift) > b)`, `half = b + m`, `size = width + 2 * half`, `lo = m + half - b`. Match.

### Independent numerical evidence

Scripts: `val_rect_string.py` and `val_rect_string_direct.py` in the session
scratchpad. The model is my own: $\mathbf g=(1.05,0.7)$ Å$^{-1}$, string at
$(0.37,-0.52)$ Å, $a^2=0.15$ Å$^2$, $\lambda=0.9$ Å$^{-1}$, Gaussian phase
$\varphi=1.8e^{-q^2/(2\cdot1.2^2)}$ and carrier $(0.21,-0.11)$ Å$^{-1}$.

| Check | Result |
|---|---|
| `string_jump` at fixed $(\mathbf Q,u)$ vs own direct sum $\psi_{\mathbf o}=\sum_{\mathbf p}\chi_{\mathbf p}\,\varphi\,e^{-i\mathbf t\cdot\mathbf r}w_{\mathbf t}(u)$, $\mathbf t=\Delta+\mathbf g_{\mathbf o-\mathbf p}$; $(M,B)\in\{((2,1),(2,3)),((1,3),(4,3)),((0,2),(3,2)),((3,0),(3,4))\}$, 1,200 draws | every unresolved decision and $\boldsymbol\kappa'$ agree; 540 box outputs, max rel. diff $1.5\times10^{-15}$ |
| Own reference: $40\times40$ midpoint grid over $\boldsymbol\kappa'$ in the first zone, analytic $\mathbb E_u$ of $w_{\mathbf t}\overline{w_{\mathbf t'}}$, wide order sum for the total | total rate equals `StringChannel.rate` to $4\times10^{-13}$ |
| $M=(2,1)$, $B=(2,3)$ ($B_x=M_x$), 20k jumps | unresolved 0.27685 vs 0.27374 ($z=+0.99$); normalized box density, 1,225 entries: max $\lvert z\rvert$ 3.35 (real) and 2.60 (imag); mean $\boldsymbol\kappa'$ $z=(-1.77,+0.90)$ |
| same, 2k jumps, other seed | unresolved $z=-0.17$; max $\lvert z\rvert$ 3.77 and 2.59; mean $\boldsymbol\kappa'$ $z=(+2.25,+0.07)$ |
| $M=(1,3)$, $B=(4,3)$ ($B_y=M_y$), 20k | unresolved $z=-0.77$; 3,969 entries: max $\lvert z\rvert$ 3.02 and 3.17; mean $\boldsymbol\kappa'$ $z=(-0.20,-0.57)$ |
| Square `string_jump` (400 draws) and `column` vs `a259d557e`; string unravelings (strings + kicks, free strings) | byte-identical (see the fibre write-up) |
| Anchors (both test modules + ledger test) | 60 passed |

The largest entry pulls are consistent with the maximum of about $10^3$
correlated normal deviates ($\approx3.3$). The fixed-$(\mathbf Q,u)$ identity
is exact, so the per-axis crop is pinned independently of sampling noise.

### Findings

None blocking. `StringChannel.column` now accepts a pair, and `order_extents`
refuses negative or non-integer halves. That validation is new and
behaviour-neutral for valid inputs.
