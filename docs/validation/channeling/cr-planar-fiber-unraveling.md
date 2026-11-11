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
$|z|\le1.6$ at 80k incidents). Comparisons with the solver use $M=16$.

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
