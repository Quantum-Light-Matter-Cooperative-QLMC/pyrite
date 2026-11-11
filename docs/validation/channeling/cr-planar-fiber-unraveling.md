# `cr-planar-fiber-unraveling`: planar carrier-fibred absorbing unraveling

**Status:** task-owner derivation (2026-11-10, #1056), awaiting fresh-context
verification.
Code: `src/pyrite/montecarlo/channeling/planar_fiber.py`
(`planar_fiber_inputs`, `plane_sites`, `root_density_harmonics`, `_strings`,
`_kicks`). The sampling engine is the anchored axial fibre unraveling
(`cr-axial-fiber-unraveling`, `cr-axial-continuum-tds-jumps`,
`cr-axial-tds-tail`), used unchanged.

## Why

The planar density-matrix solver (`cr-density-matrix-split-operator`,
`cr-dechanneling-collapse`) holds the Bloch carrier on a discrete grid
$\kappa_r=rg/N_c$ and the momentum transfer on bins of width $g/N_c$ inside
a finite order band. #522 found that the occupation lengths, overflow and
handoff angles it implies depend on that band and on the binning
([dechanneling boundary note](../../research/physics/dechanneling-boundary.md)).
The axial fibre engine has none of those approximations: a continuous
classical carrier, exact continuous TDS transfers and an exact out-of-box
tail. This claim maps the planar Lindblad model onto that engine so the
same instruments (paired $P=R$ arm, tails, per-case ε) become available for
planes.

## Derivation

**Planar model.** `cr-lindblad-jump-operators` gives, per unit depth,
$\dot\rho=-\tfrac{i}{\hbar v}[H_\perp,\rho]+\int C(x,x')\rho(x,x')-\ldots$
with two incoherent mechanisms, each already a sum over transverse
momentum transfers $(q_x,q_y)$ in which the $y$ motion has been traced:

* TDS (Einstein, isotropic $S=a^2I$):
  $C(x,x')=\frac{1}{(\hbar v)^2}\sum_G n_Ge^{iGx}\int\frac{dp\,dq_y}{(2\pi)^2}
  v(q)v(q')e^{ip(x-x')}[e^{-a^2G^2/2}-e^{-a^2(q^2+q'^2)/2}]$,
  $q=(p+G,q_y)$, $q'=(p,q_y)$, with
  $n_G=\sum_s n_se^{-iGx_s}$ the static site density (planes $x_s$,
  volume density $n_s=1/V_{\rm cell}$ per basis atom).
* electronic:
  $C_{\rm el}(x,x')=\int d^2q\,w(q)\sqrt{\rho_q(x)\rho_q(x')}e^{iq_x(x-x')}$.

**Axial model on a $y$-continuum cell.** Take the rectangular cell
$(d,L_y)$, reciprocal $(g,g_y=2\pi/L_y)$. Put at every plane site $x_s$ a
row of strings at $(x_s,jL_y)$ with linear density
$\lambda_s=n_s\,d\,L_y$, so $\lambda_s/A=n_s$ ($A=dL_y$). The
`cr-axial-continuum-tds-jumps` identity sums the per-string Kraus maps over
all images into carrier-moving jumps $J_{Q,u}$ whose harmonic at transfer
$\mathbf t=\boldsymbol\delta+\mathbf g_{\mathbf n}$ is
$\sqrt{\lambda_s}\,c(\mathbf t)w_{\mathbf t}(u)$,
$c(\mathbf t)=\varphi(|\mathbf t|)e^{-i\mathbf t\cdot\mathbf r_s}/A$,
$\varphi=-v/\hbar v$ (the per-atom phase transform, Å²). Its loss is

$$
\Gamma_{ab}=\frac{\lambda_s}{A(2\pi)^2}e^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r_s}D(|\mathbf g_a-\mathbf g_b|),\quad
D(G)=\int d^2k\,\varphi(|k-G/2|)\varphi(|k+G/2|)[e^{-a^2G^2/2}-e^{-a^2(k^2+G^2/4)}].
$$

For window orders with $n_y=0$, $\mathbf g_a-\mathbf g_b=(G,0)$, and with
$k=(p+G/2,q_y)$ this is $n_se^{-iGx_s}$ times the planar $(p,q_y)$ integral
at $G=g_a-g_b$: summed over sites it is exactly the planar full-transfer
Toeplitz $C_{ab}$ above, independently of $L_y$.

**No $y$ folding.** Every transfer satisfies $|\mathbf t|\le q_{\max}$
(strings) or $|\mathbf q|=q_i\le q_{\max}$ (kicks), $q_{\max}=10^4$ Å⁻¹, the
same cut as the planar kernels. With $g_y/2>q_{\max}+|\kappa_y|$ (the code
uses $g_y=10^9$ Å⁻¹) no output folds in $y$, so:
(i) the outgoing box needs only $n_y=0$, $(2B+1)\times1$;
(ii) harmonics with $n_y\neq0$ have $|t_y|\ge g_y-q_{\max}>q_{\max}$ and
vanish identically; (iii) $H(\kappa)$ has only $n_y=0$ orders because the
planar potential and electron density are $y$-independent.
Then $\kappa_y$ is the accumulated out-of-plane momentum. Different
$\kappa_y$ never interfere (carrier block diagonality, `cr-axial-fiber-unraveling`)
and $\hbar^2\kappa_y^2/2\gamma m$ is a constant within a fibre, a global phase
that changes no norm, rate or projection. Hence
$H(\kappa)=H_x(\kappa_x)$ (`bloch._hamiltonian`) and the retained cut uses
$E_\perp$ of the $x$ motion only, as the density-matrix solver does. The
planar reduction traces $y$; the fibre keeps it as a label. Expectations of
$x$-observables agree because tracing a label commutes with the dynamics.

**Kicks.** $C_{\rm el}$ is a superposition over $d^2q$ of rank-one Kraus
maps $S_{\mathbf q}=\sqrt{w(q)d^2q}\,\sqrt{\rho_q(x)}e^{i\mathbf q\cdot\mathbf r}$.
On the quadrature $\int d^2q\,w f\approx\sum_ic_if(q_i)$ of
`operators._electronic_nodes` (radial $2\pi q\,dq$ distant panels and close
loss panels, the same nodes the planar diagonal uses) this is the axial
fibre kick with column $\sqrt{c_i}\,\widehat{\sqrt{\rho_{q_i}}}_n$, magnitude
$q_i$ and uniform azimuth. `root_density_harmonics` evaluates
$\rho_q$ with the planar smoothing $\rho_G e^{-G^2/2q^2}$ exactly at $q_i$
(`operators._root_density` interpolates a log-$q$ table instead).
Truncating harmonics at $|n|\le H$ is consistent: loss and draws both use
the truncated columns (`_autocorrelation_loss`).

**Exactness, window and absorption.** The rest is the anchored axial engine
with window $(M,0)$, box $(B,0)$, cut $E_\perp<U_{\max}+\text{margin}$:
no-jump evolution under $P(\kappa)[H-i\hbar v\Gamma/2]P(\kappa)$, exact
continuous string draws, retained/outgoing/unresolved split without
wrapping, out-of-box string outputs as tails.

**Window convention.** The fibre window is the symmetric
$|\kappa_x+ng|\le(M+\tfrac12)g$ around zero momentum. The density-matrix
solver holds $\kappa_r\in[0,g)$ with orders $|h|\le M$, i.e. momenta
$[-Mg,(M+1)g)$. The two absorbing models therefore differ in which outputs
overflow. At small $M$ this is visible (Si(110), 5 MeV, $M=6$, 30 eV margin,
4000 Å: overflow 0.067 vs 0.052, free 0.082 vs 0.092, retained trace within
0.005); it vanishes as the window grows ($M=16$: all three within
$|z|\le1.6$ at 80k incidents). The window needed is plane dependent: Si(110) agrees at $M=16$; Si(111) (smaller $g$) still shows a ±2σ free/overflow split at $M=16$ and agrees at $M=24$ (fresh-context finding 1).

## Assumptions

First Born per atom; Einstein isotropic Gaussian displacements (anisotropic
or $S_{xy}\ne0$ sites refused); scalar local-density electronic response
(directional valence not supported); fixed energy (no stopping); paraxial
$y$-free motion; transfer cut $10^4$ Å⁻¹ as the planar kernels; absorbing
cut of the planar solver. Same caller obligations as the axial engine:
no matching surface, rechanneling or admission.

## Limiting cases

* Zero mechanisms: every incident retained (engine).
* One plane at $x_s=0$ (Si(110), Si(100) after merging coincident planes):
  real Toeplitz loss.
* $L_y$ independence: only $\lambda_s/A=n_s$ enters loss and draws.
* $M\to\infty$: window convention irrelevant; fibre and solver agree.

## Checks (task owner)

`tests/montecarlo/quantum/test_channeling_planar_fiber.py`:

* plane sites reproduce `operators._site_densities` (Si(110), Si(100),
  diamond(111)) to 1e-14;
* fibre loss equals the solver's full-transfer `C(x, x)` Toeplitz, TDS
  (2e-7 relative, polar vs radial quadrature) and electronic (1e-15), Si(110)
  and Si(100);
* single-jump TDS output momentum law (24k draws from a coherent
  three-state superposition) vs `overflow_spectrum` gain per bin (χ²);
* kick output law, exact azimuth arcsine binning, vs the planar electronic
  kernel gain: total equal to 1e-6, total-variation distance < 0.02;
* ensemble retained trace, in-window free flux and overflow vs
  `DensityMatrixSolver.propagate` (Si(110) 5 MeV, M=16, 30 eV margin,
  3000 Å, TDS 5000 incidents; TDS + electronic 20k incidents, slow), all
  within 4.5σ at every recorded depth;
* tails keep $(k_x,q_y)$, carriers stay in the zone, refusals.

Injected faults detected: plane shifted by 0.15 Å (loss, TDS law, ensemble);
kick smoothing at $2/q$ (loss, kick law).

## Verification request

Fresh context: re-derive (1) the planar TDS and electronic dissipators as
the $y$-continuum limit of the axial string/kick model, including
$\lambda_s=n_sdL_y$, the $L_y$ independence and the no-fold condition;
(2) that $\kappa_y$ and the dropped $\hbar^2\kappa_y^2/2\gamma m$ change no
$x$ expectation; (3) `plane_sites` phase convention against
`_site_densities`; (4) kick columns against $C_{\rm el}$. Then compare the
code and run independent numerical checks.

## Fresh-context verification (2026-11-10)

Verifier: fresh context (did not write `planar_fiber.py`). Inputs read
before the derivation: this ledger row, the module/function docstrings of
`planar_fiber.py`, and the ledger rows of `cr-lindblad-jump-operators`,
`cr-axial-continuum-tds-jumps` and `cr-axial-fiber-unraveling` (taken as
anchored; not re-verified). The derivation below was written before any
implementation body was opened.

### Independent derivation

**Planar targets.** From `cr-lindblad-jump-operators` the dissipator is the
Hadamard form $C(x,x')\rho(x,x')-\tfrac12[C(x,x)+C(x',x')]\rho(x,x')$, so the
no-jump loss operator is the *multiplication* by $C(x,x)$. In the plane-wave
basis $\lvert\kappa+g_a\rangle$ of period $d$ ($g=2\pi/d$) a multiplication
by $f(x)$ has matrix element

$$
\langle\kappa+g_a\rvert f\lvert\kappa+g_b\rangle
=\frac1d\int_0^d f(x)\,e^{-i(g_a-g_b)x}\,dx=f_{G},\qquad G=g_a-g_b .
$$

For TDS, $C(x,x)=(\hbar v)^{-2}\sum_G n_Ge^{iGx}\int\frac{dp\,dq_y}{(2\pi)^2}
v(\mathbf q)v(\mathbf q')[e^{-a^2G^2/2}-e^{-a^2(\lvert\mathbf q\rvert^2+\lvert\mathbf q'\rvert^2)/2}]$
with $\mathbf q=(p+G,q_y)$, $\mathbf q'=(p,q_y)$. Shifting
$\mathbf k=(p+G/2,q_y)$ gives $\mathbf q=\mathbf k+\mathbf G/2$,
$\mathbf q'=\mathbf k-\mathbf G/2$ and
$\lvert\mathbf q\rvert^2+\lvert\mathbf q'\rvert^2=2k^2+G^2/2$, hence

$$
\Gamma^{\rm planar}_{ab}=\frac{n_{G}}{(2\pi)^2}D(\lvert G\rvert),\qquad
D(G)=\int d^2k\,\varphi(\lvert\mathbf k-\tfrac{\mathbf G}2\rvert)\varphi(\lvert\mathbf k+\tfrac{\mathbf G}2\rvert)
\big[e^{-a^2G^2/2}-e^{-a^2(k^2+G^2/4)}\big],
$$

$\varphi=-v/\hbar v$ [Å²], $n_G=\sum_sn_se^{-iGx_s}$, i.e. per site
$n_se^{-i(g_a-g_b)x_s}=n_se^{i(g_b-g_a)x_s}$. Units: $n_s$ [Å⁻³] $\times$
$D$ [Å²] = Å⁻¹, a rate per unit depth.

**Axial string row.** `cr-axial-continuum-tds-jumps` gives, per string at
$\mathbf r_s$ with linear density $\lambda_s$ in a cell of area $A$,
$\Gamma_{ab}=\lambda_se^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r_s}
D(\lvert\mathbf g_a-\mathbf g_b\rvert)/(A(2\pi)^2)$. On the cell
$(d,L_y)$ with window orders $(n,0)$ only, $\mathbf g_a-\mathbf g_b=(G,0)$
and $(\mathbf g_b-\mathbf g_a)\cdot\mathbf r_s=(g_b-g_a)x_s$ for any $y_s$.
With $\lambda_s=n_s\,d\,L_y$, $\lambda_s/A=n_s$ and the axial loss equals
$\Gamma^{\rm planar}_{ab}$ term by term, for any $L_y$. The same holds for the
jumps: the axial measure $A\,d^2\Delta/(2\pi)^2$ times the harmonic weight
$\lambda_s\lvert c\rvert^2=\lambda_s\varphi^2/A^2$ gives a transfer density
$(\lambda_s/A)\varphi(\lvert\mathbf t\rvert)^2\lvert w_{\mathbf t}(u)\rvert^2/(2\pi)^2$
per $d^2t$, again only through $n_s$; this is the planar gain
$n_s\varphi\varphi'\ldots/(2\pi)^2$ per $dp\,dq_y$ with $t_y=q_y$. The
displacement model must be per-atom independent with transverse variance
$a^2$ per axis in both models (planar Einstein isotropic: the beam-axis
component does not change the projected potential).

**No fold.** The fibre carrier after a transfer is
$\kappa_y+t_y$; it stays in the zone $[-g_y/2,g_y/2)$, so that the order
label stays $n_y=0$ and no $y$ umklapp occurs, iff
$\lvert\kappa_y+t_y\rvert<g_y/2$, guaranteed by $g_y/2>q_{\max}+\lvert\kappa_y\rvert$.
Harmonics at $n_y\ne0$ involve $\lvert t_y\rvert\ge g_y-q_{\max}>q_{\max}$
and are zero under the transfer cut. With $g_y=10^9$ Å⁻¹ and
$q_{\max}=10^4$ Å⁻¹, $\lvert\kappa_y\rvert$ would need $\gtrsim5\times10^4$
maximal transfers to violate it; physically transfers are $\lesssim10$ Å⁻¹.
In float64, $\kappa_y\sim10^{4}$ against $g_y=10^9$ loses nothing relevant
(relative spacing $\sim10^{-7}$ Å⁻¹ at $10^9$, but only $\kappa_y$ itself is
carried, not $\kappa_y \bmod g_y$ arithmetic near $g_y$).

**$\kappa_y$ as a label.** The fibre state is
$\psi(x)e^{i\kappa_yy}$; $H=\hat p_x^2/2\gamma m+\hat p_y^2/2\gamma m+U(x)$
acts as $H_x(\kappa_x)+\hbar^2\kappa_y^2/2\gamma m$. Inside one fibre the
second term is a scalar: the no-jump propagator acquires a global phase,
so norms, the waiting law $\lVert e^{-iH_{\rm eff}z}\chi\rVert^2$, the jump
probabilities $\langle\chi\rvert\Gamma\lvert\chi\rangle$ and the output
columns are unchanged. Neither $\Gamma_{ab}$ (depends on $g_a-g_b$) nor the
kick columns depend on $\kappa$. The retained projector $P(\kappa)$ is a
spectral projector of $H(\kappa)$; a constant shift leaves eigenvectors
unchanged and moves the cut only if the cut is on total transverse
energy. The planar solver's cut is on $E_\perp$ of the $x$ motion, so the
fibre must cut on $H_x(\kappa_x)$ to reproduce it (it does by dropping the
constant). Reduced $x$ observables are $\rho_x=\int d\kappa_y\,\rho_{\kappa_y}$;
because every map (no-jump, jump, cut) acts on $x$ independently of
$\kappa_y$, tracing the label commutes with the dynamics and reproduces the
$q_y$-integrated planar dissipator. Distinct $\kappa_y$ never interfere
(fibre block diagonality, anchored).

**Plane sites.** For basis atom $s$ at fractional $\mathbf r_s$ and
coprime normal $(hkl)$, the phase of reciprocal vector $m\,\mathbf G_{hkl}$
is $e^{-2\pi i m(hkl)\cdot\mathbf r_s}$. Writing
$e^{-iGx_s}$ with $G=m g_{\rm coprime}$ requires
$x_s=2\pi(hkl)\cdot\mathbf r_s/g_{\rm coprime}=d_{\rm coprime}\,(hkl)\cdot\mathbf r_s$.
If the profile period is $d=d_{\rm coprime}/j$ (coincident planes), only
$G\in jg_{\rm coprime}\mathbb Z$ appear and folding $x_s$ mod $d$ is
harmless. Site density $n_s=1/V_{\rm cell}$ per basis atom (per element
weight via its own $\varphi$).

**Kicks.** $C_{\rm el}$ is the Kraus sum of
$S_{\mathbf q}=\sqrt{w(q)\,d^2q}\,\sqrt{\rho_q(x)}\,e^{i\mathbf q\cdot\mathbf r}$:
$S_{\mathbf q}\rho S_{\mathbf q}^\dagger(x,x')=w\,d^2q\sqrt{\rho_q(x)\rho_q(x')}e^{iq_x(x-x')}\rho(x,x')$
and $\sum S^\dagger S=\int d^2q\,w\,\rho_q(x)=C_{\rm el}(x,x)$. A radial
quadrature $\int d^2q\,w f\approx\sum_ic_i\langle f\rangle_\theta$ with
uniform azimuth makes each node a fibre kick of magnitude $q_i$, column
$\sqrt{c_i}\,(\sqrt{\rho_{q_i}})_n$, $(\sqrt{\rho_q})_n=\frac1d\int\sqrt{\rho_q}e^{-ingx}dx$;
the output harmonic is the convolution $\sum_n(\sqrt{\rho_q})_{m-n}\psi_n$ at
carrier $\kappa+q(\cos\theta,\sin\theta)$. With $\rho_q=\rho_e*\mathcal N(0,1/q^2)$,
$(\rho_q)_G=\rho_Ge^{-G^2/2q^2}$. The kick loss is then
$\sum_ic_i(\rho_{q_i})_{G}$ (Parseval of the truncated column
autocorrelation), equal to the solver's $C_{\rm el}(x,x)$ harmonics if the
same nodes, the same smoothing and enough harmonics are used.

**Cheap filters.** Units: $\lambda_s$ [Å⁻¹] $=n_s$ [Å⁻³] $d\,L_y$ [Å²] pass;
the kick loss $c_i\rho$ must be Å⁻¹, i.e. $c_i=w\,d^2q$ per electron
in Å² times $\rho$ in Å⁻³, so a column $\sqrt{c_i}\sqrt{\rho}$ is
Å⁻¹ᐟ², as `FiberChannels` documents; pass.
Limits: zero mechanisms retain all; one site at $x_s=0$ gives a real
symmetric Toeplitz; $L_y$ cancels. Signs: the planar phase
$e^{-i(g_a-g_b)x_s}$ and the axial $e^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r_s}$
agree. All pass.

### Code comparison

Read after the derivation: `planar_fiber.py` (all of it), `operators.py`
(`_potential`, `_site_densities`, `_root_density`, `_tds_transform`,
`_tds_diagonal`, `tds_kernel`, `_electronic_nodes`,
`_electronic_diagonal`), `density.DensityMatrixSolver._toeplitz`, and the
engine interfaces `FiberChannels`, `_Engine.__init__`,
`_autocorrelation_loss`, `_fold`, `StringChannel.__init__`, `string_jump`.

| Derived item | Code | Agreement |
|---|---|---|
| $\lambda_s=n_s\,d\,L_y$, $L_y=2\pi/g_y$ | `_strings`: `density_inv_ang3 * period_ang * 2π/Y_CONTINUUM_INV_ANG` | exact |
| $\varphi=-v/\hbar v$, cut $q_{\max}=10^4$ Å⁻¹ | `-_potential(element, q) / hbar_v`, `_Q_MAX_INV_ANG`; the closure binds `element` per site | exact |
| displacement $a^2$ per axis, isotropic only | `covariances[element][0]` ($S_{xx}$); refuses $S_{xy}\ne0$ or $S_{xx}\ne S_{yy}$ | exact |
| $x_s=d_{\rm coprime}(hkl)\cdot\mathbf r_s \bmod d$ | `fraction = (hkl·r)·g/g_coprime`, folded, snapped to 12 digits, coincident planes merged | equivalent to `_site_densities` (which uses `step = round(g/g_coprime)`; equal when the ratio is integral, as for every catalog plane) |
| window/box $(M,0)$, $(B,0)$; $g_y=10^9$ Å⁻¹ | `FiberChannels(order_max=(M,0), outgoing_order_max=(B,0), reciprocal_inv_ang=(g, 1e9))` | exact |
| $H(\kappa)=H_x(\kappa_x)$, cut on $x$ energy | `spectrum` calls `_hamiltonian(profile, gamma, kappa[0], orders)`; `cut` is the constant `barrier + margin` | exact; matches `DensityMatrixSolver.cutoff_eV` |
| kick column $\sqrt{c_i}(\sqrt{\rho_{q_i}})_n$, $(\rho_q)_G=\rho_Ge^{-G^2/2q^2}$, $\frac1d\int\cdot\,e^{-ingx}$ | `_kicks` with `_electronic_nodes` (`c > 0`), `root_density_harmonics` (same density reconstruction as `_root_density`, `fft/n_x`) | exact; evaluated at $q_i$ rather than interpolated |
| loss orientation $\Gamma_{ab}=f_{g_a-g_b}$ | engine `C(p-p')`; solver `_toeplitz` $\langle h\rvert f\lvert h'\rangle=f_{h-h'}$ | exact (numerically confirmed on non-centrosymmetric planes below) |
| entrance $k_x=k\psi$ folded | `order = floor(k_x/g + 1/2)`, carrier `k_x - order g` | exact |

No `Validation:` marker is missing: the module, `plane_sites`, `_strings`,
`root_density_harmonics`, `_kicks` and `planar_fiber_inputs` carry
`cr-planar-fiber-unraveling`. `PlaneSite`/`PlanarFiberInputs` are data
containers.

### Independent numerical evidence

Scripts: `scratch/verify-1056/check_loss.py`, `check_ensemble.py`,
`check_jump_law.py`, `check_faults.py` (worktree-local, not committed).
References are the verifier's own: site sums from the catalog basis, an own
24-point × 96-angle midpoint-polar $D(G)$ quadrature with panels refined at
$r=G/2$, and $\sum_ic_i\rho_Ge^{-G^2/2q_i^2}$ for the electronic loss. Planes
chosen to differ from the task owner's: Si(111) and diamond(111) (two planes
at $x=0,\,3d/4$, so complex Toeplitz losses). GaAs and Ge are refused by
`planar_profile` (no thermal rows), so no two-element case was possible.

* **Plane sites** vs own basis sums, $\lvert m\rvert\le8$: Si(111)
  8.1e-16, diamond(111) 2.9e-15 Å⁻³.
* **TDS loss** ($M=5$, 5 MeV) vs own $D(G)$ Toeplitz: Si(111) 3.4e-8,
  diamond(111) 2.4e-7 relative; $\Gamma_{00}$ = 6.428950e-5 Å⁻¹ (Si) and
  2.366184e-5 Å⁻¹ (C). The transposed matrix misses by 0.95 and 0.91.
* **Electronic loss** with `kick_half=64` vs own sum: 2.6e-16 (Si(111)),
  1.8e-16 (diamond(111)); transposed misses by 0.11 / 0.063. At the
  default `kick_half = B + M` the truncation error is 7.1e-6 / 7.4e-7.
* **$L_y$ independence.** Rebuilding the channels at $g_y=10^6$ and
  $10^{12}$ Å⁻¹ with $\lambda_s$ rescaled: loss identical (0 for Si(111),
  1e-17 for diamond(111)); 300/300 incidents give identical terminal kind
  and depth with the same seed. Fault: $g_y=3$ Å⁻¹ (folding allowed) moves
  overflow at 3000 Å by $+53\sigma$ (Si(111) TDS) and $+156\sigma$
  (diamond(111) TDS + electronic).
* **Single-jump TDS output law** (Si(111), site at $x=3d/4$, $\chi=(0.8,0.6i)$
  on orders 0, 1): engine $\langle\chi\rvert\Gamma_s\lvert\chi\rangle$ =
  6.13249789e-5 vs own 2D integral of the planar gain density 6.13249892e-5
  (1.7e-7). Own $p_x$ marginal vs 40k `StringChannel.sample` draws:
  $\chi^2=23.0/28$ ($p=0.73$); with the site phase sign flipped in the
  reference $\chi^2=5\times10^5$.
* **Ensemble vs `DensityMatrixSolver.propagate`** (5 MeV, 30 eV margin,
  3000 Å, binomial $z$ at every 1000 Å record):
  Si(111) TDS, $M=16$, 60k incidents: trace $\lvert z\rvert\le0.9$, free
  $+2.0$, overflow $-2.2$; $M=24$, 60k: all $\lvert z\rvert\le1.5$.
  Diamond(111) TDS + electronic, $M=16$, 30k: all $\lvert z\rvert\le1.0$.
  Entrance weight equals $1-$`entry_flux` (0.999975, 0.999842).
* **Injected faults** (monkeypatched, source untouched) on Si(111):
  mirrored sites → TDS loss off by 0.95; kick smoothing at $2/q$ →
  electronic loss off by 0.043; $\lambda_s$ without $d$ → off by 0.68.
* Anchor file `tests/montecarlo/quantum/test_channeling_planar_fiber.py`:
  12 passed, 1 slow skipped.

### Findings

No blocking findings. Minor, non-blocking:

1. **Window convention at $M=16$ is plane dependent.** For Si(111)
   ($g=2.00$ Å⁻¹, below Si(110)'s 3.27) the paired free/overflow tallies
   still differ by about $\pm2\sigma$ at 60k incidents
   (free 0.0665 vs 0.0645 at 3000 Å), and agree at $M=24$. This is the
   documented symmetric-vs-$[-Mg,(M+1)g)$ convention, not a model error;
   the statement "comparisons with the solver use $M=16$" holds for Si(110),
   not as a general sufficiency rule.
2. **No-fold condition is not guarded at run time.** Nothing refuses
   $\lvert\kappa_y\rvert+q_{\max}\ge g_y/2$; violation needs
   $\gtrsim5\times10^4$ maximal transfers, so it is unreachable in practice.
3. **Default kick truncation.** `kick_half = B + M` makes the model's
   electronic loss differ from the untruncated planar $C_{\rm el}(x,x)$ by
   ≤7e-6 relative here; the loss and draws remain mutually exact (shared
   truncated columns), as stated.
4. `plane_sites` uses the float ratio $g/g_{\rm coprime}$ and
   `_site_densities` its rounded integer; they agree for catalog planes, but a
   non-integral ratio would silently diverge rather than be refused.

### Verdict

**`rederived`.** The $y$-continuum string-row construction with
$\lambda_s=n_s\,d\,L_y$, the $L_y$ cancellation, the no-fold condition, the
$\kappa_y$-label argument with the $x$-only cut, the `plane_sites` phase
convention and the kick columns re-derive independently and match the code.
Own numerical references agree on two new non-centrosymmetric planes.

- **Claim**: `cr-planar-fiber-unraveling` —
  `montecarlo/channeling/planar_fiber.py::{planar_fiber_inputs,plane_sites,root_density_harmonics,_strings,_kicks}`
  — `cr-lindblad-jump-operators` kernels as the $y$-continuum limit of
  `cr-axial-continuum-tds-jumps` / `cr-axial-fiber-unraveling`
- **Filters**: units pass; limits pass; signs/conventions pass
- **Re-derivation**: matches
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/channeling/cr-planar-fiber-unraveling.md`
- **Suggested ledger change**: Status `rederived` (task owner applies);
  append to Notes "2026-11-10 fresh-context verdict `rederived`: own $D(G)$
  and electronic Toeplitz on Si(111)/diamond(111) (3.4e-8/2.4e-7; 2.6e-16),
  $L_y$ invariance ($g_y=10^6,10^{12}$ bitwise events), single-jump $p_x$
  law $\chi^2$ 23.0/28, solver ensembles $\lvert z\rvert\le1.5$ (Si(111) M=24,
  60k) and $\le1.0$ (diamond(111) TDS+el, 30k); minor: Si(111) needs
  $M>16$ for the window convention to vanish at 60k".
