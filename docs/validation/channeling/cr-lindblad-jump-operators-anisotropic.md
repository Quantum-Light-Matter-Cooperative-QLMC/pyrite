# `cr-lindblad-jump-operators` — anisotropic TDS sites (#535)

Verifier: fresh context. Scope: only the "Anisotropic sites (#535)" part of the claim. The isotropic kernel, the electronic kernel and the Bloch-residue representation are covered by [the main write-up](cr-lindblad-jump-operators.md). Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/montecarlo/channeling/operators.py::{_transverse_covariances,_min_eigenvalue,_tds_strength,_tds_diagonal,tds_kernel,decoherence_rate}` and `src/pyrite/materials/channeling/thermal.py::UniaxialThermalSite.covariance`. Commit `81fef1da` on branch `issue-535-layered-thermal-tensors`.

Procedure: I read the ledger row and the module docstring first, derived §1–§4 below, and only then read the implementation bodies (§5). The numeric checks in §6 use my own geometry, my own Mott–Bethe evaluation and brute-force quadratures. They use no implementation helper except catalog data: lattice constants, basis, the Schönfeld $u$ values, and `cromer_mann_f0`.

## Claim (as ledgered)

For a site with Cartesian displacement covariance $U$, take the transverse frame $\hat{\mathbf x}$ = plane normal and $\hat{\mathbf y}=\hat{\mathbf x}\times\hat{\mathbf b}_\parallel$, where $\hat{\mathbf b}_\parallel$ is the beam projected into the plane. The TDS kernel then uses the $(x,y)$ block $S$ of $U$. Its bracket is $e^{-S_{xx}G^2/2}-e^{-(\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q')/2}$ with $\mathbf q=(p+G,q_y)$ and $\mathbf q'=(p,q_y)$. The $S_{xy}$ term is odd in $q_y$, and the half-line $q_y$ quadrature averages both signs. The full-loss diagonal uses $e^{-2S_{xx}(G/2)^2}-e^{-\mathbf k\cdot S\mathbf k-S_{xx}(G/2)^2}$. $\Gamma_{\rm TDS}$ replaces $e^{-a^2q^2}J_0(q\Delta)$ by $\sum_{n\ge0}\epsilon_n e^{-\bar s q^2}I_n(\rho q^2)\cos(n\varphi)J_{2n}(q\Delta)$. `beam_uvw` is required unless $\hat{\mathbf x}$ is an eigenvector of every $U$ and the in-plane block is isotropic.

## 1. Phase covariance with a general covariance $U$

A slab $dz$ multiplies the transverse wave by $e^{i\phi}$, where

$$
\phi(\mathbf r_\perp)=-\frac{1}{\hbar v}\sum_j V_{\rm proj}(\mathbf r_\perp-\mathbf R_{j\perp}-\mathbf u_{j\perp}).
$$

The projection runs along the beam. To channeling accuracy the beam is $\hat{\mathbf b}_\parallel$; its out-of-plane tilt $\psi$ is a few mrad, which changes $S$ only at $O(\psi)$. $\{\hat{\mathbf x},\hat{\mathbf y},\hat{\mathbf b}_\parallel\}$ is orthonormal. The component $\mathbf u\cdot\hat{\mathbf b}_\parallel$ slides the atom along the projection line and leaves $V_{\rm proj}$ unchanged for an infinite projection. Only $\mathbf u_\perp=(\mathbf u\cdot\hat{\mathbf x},\mathbf u\cdot\hat{\mathbf y})$ enters, and for a Gaussian $\mathbf u$ it is Gaussian with covariance

$$
S=PUP^{\mathsf T},\qquad P=\begin{pmatrix}\hat{\mathbf x}^{\mathsf T}\\ \hat{\mathbf y}^{\mathsf T}\end{pmatrix},\qquad
S_{xx}=\hat{\mathbf x}\cdot U\hat{\mathbf x},\ S_{xy}=\hat{\mathbf x}\cdot U\hat{\mathbf y},\ S_{yy}=\hat{\mathbf y}\cdot U\hat{\mathbf y}.
$$

With $V_{\rm proj}(\mathbf r)=\int\frac{d^2q}{(2\pi)^2}\tilde v(q)e^{i\mathbf q\cdot\mathbf r}$, where $\tilde v$ is real and isotropic, and $\langle e^{-i\mathbf k\cdot\mathbf u_\perp}\rangle=e^{-\mathbf k\cdot S\mathbf k/2}$, one atom at $\mathbf R$ contributes

$$
{\rm Cov}[\phi(\mathbf r),\phi(\mathbf r')]=\frac{1}{(\hbar v)^2}\iint\frac{d^2q\,d^2q'}{(2\pi)^4}\tilde v(q)\tilde v(q')
e^{i\mathbf q\cdot(\mathbf r-\mathbf R)-i\mathbf q'\cdot(\mathbf r'-\mathbf R)}
\Bigl[e^{-(\mathbf q-\mathbf q')\cdot S(\mathbf q-\mathbf q')/2}-e^{-(\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q')/2}\Bigr].
$$

Uncorrelated (Einstein) sites add. Summing over atoms per unit depth, with uniform $R_y$ and $x$-periodic site density $n(R_x)=\sum_Gn_Ge^{iGR_x}$, produces $2\pi\delta(q_y-q'_y)$ and $\sum_G n_G\,2\pi\delta(q_x-q'_x-G)$. Then $\mathbf q-\mathbf q'=(G,0)$ and $(\mathbf q-\mathbf q')\cdot S(\mathbf q-\mathbf q')=S_{xx}G^2$. Evaluate at $y=y'$, which is the $x$-reduced kernel of a $y$-translation-invariant beam, and set $\mathbf q'=(p,q_y)$:

$$
\boxed{C(x,x')=\frac{1}{(\hbar v)^2}\sum_G n_Ge^{iGx}\int\frac{dp\,dq_y}{(2\pi)^2}\tilde v(\mathbf q)\tilde v(\mathbf q')e^{ip(x-x')}
\Bigl[e^{-S_{xx}G^2/2}-e^{-(\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q')/2}\Bigr],\quad \mathbf q=(p+G,q_y).}
$$

This matches the ledger bracket. Expanding the exponent:

$$
\tfrac12(\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q')
=\tfrac12S_{xx}\bigl[(p+G)^2+p^2\bigr]+S_{yy}q_y^2+S_{xy}(2p+G)q_y .
$$

$\tilde v(|\mathbf q|)\tilde v(|\mathbf q'|)$ is even in $q_y$. Only the $S_{xy}$ term is odd, so

$$
\int_{-\infty}^{\infty}dq_y\,F(q_y)=2\int_0^\infty dq_y\,\tfrac12\bigl[F(q_y)+F(-q_y)\bigr],
$$

which is the ledgered half-line quadrature averaging $\pm S_{xy}$. The average turns $e^{-S_{xy}(2p+G)q_y}$ into $\cosh\bigl(S_{xy}(2p+G)q_y\bigr)$. Every factor is real, so each $G$ coefficient is real. The momentum matrix is Hermitian because swapping $p\leftrightarrow p+G$, which corresponds to $G\to-G$, leaves the bracket invariant and $n_{-G}=n_G^*$.

Isotropic limit: with $S=a^2I$ the exponent becomes $a^2\bigl[(p+G)^2+p^2+2q_y^2\bigr]/2$, which is the prior isotropic ledger formula exactly.

PSD: $C$ is a covariance of the random field $\phi$, so every finite matrix $C(x_i,x_j)$ and every principal momentum sub-block is positive semidefinite for any PSD $S$.

## 2. Full-loss diagonal

At $x=x'$, substitute $p=k_x-G/2$. Then $\mathbf q=\mathbf k+\tfrac G2\hat{\mathbf x}$ and $\mathbf q'=\mathbf k-\tfrac G2\hat{\mathbf x}$, with $\mathbf k=(p+G/2,q_y)$. The two $\pm$ cross terms in $\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q'$ cancel:

$$
\tfrac12(\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q')=\mathbf k\cdot S\mathbf k+S_{xx}(G/2)^2,\qquad
e^{-S_{xx}G^2/2}=e^{-2S_{xx}(G/2)^2},
$$

so

$$
C(x,x)=\sum_G n_Ge^{iGx}D_G,\qquad
D_G=\frac{1}{(\hbar v)^2}\int\frac{d^2k}{(2\pi)^2}\tilde v\bigl(|\mathbf k+\tfrac G2\hat{\mathbf x}|\bigr)\tilde v\bigl(|\mathbf k-\tfrac G2\hat{\mathbf x}|\bigr)
\Bigl[e^{-2S_{xx}(G/2)^2}-e^{-\mathbf k\cdot S\mathbf k-S_{xx}(G/2)^2}\Bigr],
$$

which matches the ledger. In polar form $\mathbf k=\kappa(\cos t,\sin t)$ we have $|\mathbf k\pm\tfrac G2\hat{\mathbf x}|^2=\kappa^2+G^2/4\pm\kappa G\cos t$ and $\mathbf k\cdot S\mathbf k=\kappa^2(S_{xx}\cos^2t+S_{yy}\sin^2t)+2S_{xy}\kappa^2\cos t\sin t$. The potential product is invariant under $t\to-t$ and under $t\to t+\pi$, which swaps the two factors. The cross term is odd under $t\to-t$. Hence

$$
\int_0^{2\pi}dt\,(\cdots)=2\int_0^{\pi}dt\,\tfrac12\bigl[(\cdots)_{+S_{xy}}+(\cdots)_{-S_{xy}}\bigr].
$$

The half circle $[0,\pi]$ is a fundamental domain of $t\to t+\pi$, so the $\pm$ average is in fact redundant there, but it is exact and harmless. $D_{-G}=D_G$ because $\mathbf k\to-\mathbf k$ maps one integrand onto the other.

For $G=0$, $D_0=(\hbar v)^{-2}\int\frac{d^2q}{(2\pi)^2}\tilde v^2(1-e^{-\mathbf q\cdot S\mathbf q})$ is rotation invariant. It depends only on the eigenvalues of $S$, which is the ledger's "mean loss depends only on eigenvalues".

## 3. Cell-averaged $\Gamma_{\rm TDS}(\Delta)$ for anisotropic $S$

The cell average over $x$ kills every $e^{iGx}$ with $G\ne0$, both in $C(x,x)$ and in $C(x,x+\Delta)$, whose $p$-phase is $e^{-ip\Delta}$. The $G=0$ integrand is even under $\mathbf q\to-\mathbf q$, so

$$
\Gamma_{\rm TDS}(\Delta)=\sum_s\frac{n_{s,0}}{(\hbar v)^2}\int\frac{d^2q}{(2\pi)^2}\tilde v_s^2\,\bigl(1-e^{-\mathbf q\cdot S_s\mathbf q}\bigr)\bigl(1-\cos q_x\Delta\bigr)
=\sum_s\frac{n_{s,0}}{2\pi(\hbar v)^2}\int_0^\infty q\,dq\,\tilde v_s^2\Bigl[1-\langle e^{-\mathbf q S\mathbf q}\rangle_\theta-J_0(q\Delta)+\langle e^{-\mathbf qS\mathbf q}\cos q_x\Delta\rangle_\theta\Bigr].
$$

Both factors in the first form are non-negative, so $\Gamma\ge0$ and $\Gamma(0)=0$. Write $\mathbf q=q(\cos\theta,\sin\theta)$:

$$
\mathbf q\cdot S\mathbf q=q^2\Bigl[\bar s+\tfrac{S_{xx}-S_{yy}}{2}\cos2\theta+S_{xy}\sin2\theta\Bigr]=q^2\bigl[\bar s+\rho\cos(2\theta-\varphi)\bigr],
$$

with $\bar s=(S_{xx}+S_{yy})/2$, $S_{xx}-S_{yy}=2\rho\cos\varphi$, $S_{xy}=\rho\sin\varphi$ and $\rho\ge0$. The eigenvalues are $\bar s\pm\rho$. Use the two generating functions

$$
e^{-z\cos\alpha}=\sum_{n\ge0}\epsilon_n(-1)^nI_n(z)\cos n\alpha,\qquad
\cos(q\Delta\cos\theta)=\sum_{m\ge0}\epsilon_m(-1)^mJ_{2m}(q\Delta)\cos2m\theta,
$$

with $\epsilon_0=1$ and $\epsilon_{n\ge1}=2$. The angular mean of $\cos n(2\theta-\varphi)\cos2m\theta$ is $\delta_{nm}\cos(n\varphi)/\epsilon_n$, and the signs combine as $(-1)^{2n}=1$:

$$
\boxed{\langle e^{-\mathbf q\cdot S\mathbf q}\cos q_x\Delta\rangle_\theta=e^{-\bar sq^2}\sum_{n\ge0}\epsilon_nI_n(\rho q^2)\cos(n\varphi)J_{2n}(q\Delta),\qquad
\langle e^{-\mathbf q\cdot S\mathbf q}\rangle_\theta=e^{-\bar sq^2}I_0(\rho q^2).}
$$

This matches the ledger in sign, factor and $\epsilon_n$. As a sign check, take the $O(\Delta^2)$ term. Directly, $-\tfrac{\Delta^2}{2}\langle q_x^2e^{-\mathbf qS\mathbf q}\rangle=-\tfrac{q^2\Delta^2}{4}e^{-\bar sq^2}(I_0-I_1)$ at $\varphi=0$. The series gives $I_0(1-q^2\Delta^2/4)+2I_1\,q^2\Delta^2/8$, the same. A larger spread along $\Delta$ ($S_{xx}>S_{yy}$) therefore reduces the overlap loss, as expected.

Isotropic limit: $\rho=0$ leaves only $n=0$, $I_0(0)=1$, which gives $e^{-a^2q^2}J_0(q\Delta)$. This is the prior formula.

Truncation. Write $e^{-\bar sq^2}I_n(\rho q^2)=e^{-\lambda_{\min}q^2}\,{\rm ive}(n,\rho q^2)$ with $\lambda_{\min}=\bar s-\rho$. In the Gaussian tail, ${\rm ive}(n,z)\lesssim{\rm ive}(0,z)e^{-n^2/2z}$. A term falls below about $10^{-16}$ once either $\lambda_{\min}q^2>37$, or $n^2>70z$ with $z=\rho q^2<37\rho/\lambda_{\min}$. So $N=\lceil\sqrt{70\cdot37\,\rho/\lambda_{\min}}\rceil$ orders suffice uniformly in $q$ and $\Delta$, since $|J_{2n}|\le1$. The correction $n^4/24z^3$ to the Gaussian approximation is covered by the code's $+4$ margin; see §6.

## 4. Uniaxial covariance and the beam-azimuth rule

$U={\rm diag}(u_b,u_b,u_c)$ in a frame with $\hat{\mathbf c}=\hat{\mathbf e}_3$ equals $u_bI+(u_c-u_b)\hat{\mathbf c}\hat{\mathbf c}^{\mathsf T}$, which is frame-free. Then

$$
S=u_bI_2+(u_c-u_b)\,\mathbf c_\perp\mathbf c_\perp^{\mathsf T},\qquad \mathbf c_\perp=(\hat{\mathbf c}\cdot\hat{\mathbf x},\hat{\mathbf c}\cdot\hat{\mathbf y}).
$$

For general $U$, $\hat{\mathbf y}$ ranges over the unit circle in the plane as the beam azimuth turns. $S_{xx}=\hat{\mathbf x}\cdot U\hat{\mathbf x}$ is azimuth-free. $S_{xy}=(U\hat{\mathbf x})_\parallel\cdot\hat{\mathbf y}$ has constant modulus only if $(U\hat{\mathbf x})_\parallel=0$, that is, $\hat{\mathbf x}$ is an eigenvector of $U$; only $|S_{xy}|$ matters, by the $\pm q_y$ symmetry. $S_{yy}$ is constant on the circle only if the in-plane block is a multiple of the in-plane identity. The ledgered rule, "`beam_uvw` required unless $\hat{\mathbf x}$ is an eigenvector of every $U$ with isotropic in-plane block", is therefore necessary and sufficient. For a uniaxial site with $u_c\neq u_b$ it holds exactly for basal planes ($\hat{\mathbf x}\parallel\hat{\mathbf c}$, giving $S={\rm diag}(u_c,u_b)$). Prism planes need the beam: $S_{yy}$ runs from $u_c$ for a beam in the basal plane to $u_b$ for a beam along $\mathbf c$. Isotropic sites never need it.

## 5. Comparison with the implementation (read after §1–§4)

| Item | Derived | Code | Result |
| --- | --- | --- | --- |
| $S$ block | $S_{xx}=\hat{\mathbf x}U\hat{\mathbf x}$, $S_{xy}=\hat{\mathbf x}U\hat{\mathbf y}$, $S_{yy}=\hat{\mathbf y}U\hat{\mathbf y}$, $\hat{\mathbf y}=\hat{\mathbf x}\times\hat{\mathbf b}_\parallel$ | `_transverse_covariances`: `n @ u @ n`, `n @ u @ y`, `y @ u @ y`, `y = np.cross(n, along / np.linalg.norm(along))`, `along = plane @ beam` | match |
| frames | $\hat{\mathbf x}$, beam, $\hat{\mathbf c}$ in one Cartesian frame | normal $=\mathbf g/\lvert\mathbf g\rvert$ from `_reciprocal_basis`; beam $=$ `uvw @ _direct_lattice_vectors`; `c_axis = _direct_lattice_vectors(lattice)[2]` | consistent |
| azimuth rule | $\hat{\mathbf x}$ eigenvector and in-plane block $\propto I$ | `norm(plane @ u @ n) > tol` or block $\ne\tfrac12{\rm tr}(\text{block})\cdot$`plane` raises | match |
| PD guard | $S_{xx}>0$, $\det S>0$ | same | match |
| $\lambda_{\min}$ | $\bar s-\rho$ | `_min_eigenvalue`: `0.5*(sxx+syy) - hypot(0.5*(sxx-syy), sxy)` | match |
| kernel bracket | §1 boxed, half line $\times2$, $\pm S_{xy}$ average | `_tds_strength`: `quad = sxx*(ket²+bra²) + 2*syy*qy²`, `cross = 2*sxy*qy*(ket+bra)`, `damp` = mean of $e^{-(quad\pm cross)/2}$, bracket $e^{-S_{xx}G^2/2}-$`damp`, factor 2 | match |
| diagonal | §2, half circle $\times2$, $\pm$ average | `_tds_diagonal`: $t\in[0,\pi]$, `k_s_k`, `k_cross = 2 sxy r² cos sin`, exponent $+S_{xx}(G/2)^2$, bracket $e^{-2S_{xx}(G/2)^2}-$`damp`, factor 2, $1/(2\pi\hbar v)^2$ | match |
| harmonic cut | momentum bracket $\le e^{-\lambda_{\min}G^2/4}$, since $\lvert\mathbf q\rvert^2+\lvert\mathbf q'\rvert^2\ge G^2/2$; diagonal $\le e^{-S_{xx}G^2/4}$ | `tds_kernel`: `max_harmonic = ceil(sqrt(4*27.7/a2_min)/g)` with `a2_min` the smallest eigenvalue over species, so $e^{-27.7}\approx10^{-12}$ | match, conservative (using $S_{xx}$ alone would be wrong for the off-diagonal kernel; $\lambda_{\min}$ is correct) |
| $\Gamma$ series | §3 boxed | `tds_density`: `ive(n, rho q²)·exp(-lam q²)` $=e^{-\bar sq^2}I_n$; weight $\epsilon_n\cos n\varphi$, `phi = arctan2(sxy, (sxx-syy)/2)`; factor `j0 - Σ series·jv(2n, qΔ)`; mean `1 - series[:,0]` | match |
| truncation | $N=\lceil\sqrt{70\cdot37\rho/\lambda_{\min}}\rceil$ | same, $+4$ | match |
| uniaxial $U$ | $u_bI+(u_c-u_b)\hat{\mathbf c}\hat{\mathbf c}^{\mathsf T}$, $\hat{\mathbf c}$ normalized | `UniaxialThermalSite.covariance` | match |

When no beam is given and the rule passes, the code takes $\hat{\mathbf y}$ as an arbitrary in-plane unit vector (an SVD column of the projector). That is legitimate because, by §4, $S$ is then azimuth-free.

## 6. Numerical spot checks (independent)

Case: `mos2`, plane $(1,0,1)$, `beam_uvw` $(0,1,0)$, 295 K, 17 MeV kinetic energy. I built my own hexagonal frame ($\mathbf a_1=(a,0,0)$, $\mathbf a_2=(-a/2,a\sqrt3/2,0)$, $\mathbf a_3=(0,0,c)$, $\hat{\mathbf c}=\hat{\mathbf e}_3$); $S$ is frame-invariant. The Mott–Bethe $\tilde v$ used the same ledgered model choices (rescaled $f_0$, $s^{-4}$ tail) through my own code. Script: `check535.py`, about 80 s.

$S$ [Å²]: mine and code agree to all printed digits.

| element | $S_{xx}$ | $S_{xy}$ | $S_{yy}$ | eigenvalues |
| --- | --- | --- | --- | --- |
| Mo | 0.0049369 | 0.0006151 | 0.0075631 | 0.0048, 0.0077 $=(u_b,u_c)$ |
| S | 0.0038519 | 0.0002333 | 0.0048481 | 0.0038, 0.0049 $=(u_b,u_c)$ |

$\Gamma_{\rm TDS}(\Delta)$: my version is a brute 2D quadrature with a periodic trapezoid in $\theta$ ($2q\Delta+256$ nodes) and radial Gauss–Legendre panels $\le\pi/2\Delta$. It applies no Bessel series and no $\pm$ averaging, and takes $q\to\infty$ with an analytic $1/q^3$ tail. The code is `decoherence_rate(...).tds`.

| $\Delta$ [Å] | mine [Å⁻¹] | code [Å⁻¹] | rel. diff |
| --- | --- | --- | --- |
| 0.3 | 1.941616e-4 | 1.941623e-4 | 3.5e-6 |
| 1.0 | 2.151853e-4 | 2.151860e-4 | 3.2e-6 |
| 2.671 ($=d$) | 2.139811e-4 | 2.139818e-4 | 3.2e-6 |
| 5.0 | 2.139239e-4 | 2.139246e-4 | 3.2e-6 |
| 20.0 | 2.139238e-4 | 2.139245e-4 | 3.2e-6 |

The uniform 3e-6 offset is the $\Delta$-independent mean-loss piece, i.e. the shared model's $q$ cut and tail treatment, not the series.

Diagonal Fourier coefficients $n_GD_G$ (summed over species). My version uses a full circle about $\mathbf k$ with the cross term kept as is, no $\pm$ average, 512 angles, and $q\le2000$ Å⁻¹ with no tail. The code is `tds_kernel(..., n_cells=8, max_order=4).diagonal_coefficients`. The relative differences are $D_0$ 3.7e-5, $D_1$ 3.9e-5 and $D_2$ 4.0e-5. The offset is my $q\le2000$ cut: adding the analytic tail moves $D_0$ to 2.13924e-4, which equals $\Gamma(\infty)$ above and the code's 2.139245e-4. $D_3$ is extinct, $n_G=0$, and both values are about 1e-19.

Momentum-kernel elements: my version is the full $q_y$ line with no $\pm$ average and 12-node Gauss–Legendre over the bin $[(l-\tfrac12),(l+\tfrac12)]g/n_{\rm cells}$. $K[r{=}0;0,0]$, $K[1;1,0]$ and $K[5;-2,2]$ agree to 2e-6, 1.7e-6 and 3.8e-6. The extinct $3g$ element is about 1e-21 in both. Every residue block is PSD, with smallest eigenvalue $4.2\times10^{-7}$ Å⁻¹.

Azimuth rule: `_transverse_covariances` without a beam succeeds for $(0,0,1)$, giving $S={\rm diag}(0.0077,0.0048)$ for Mo, and raises for $(1,0,0)$, $(1,0,1)$ and $(1,1,0)$. For $(0,0,1)$, six beams give identical $S$. For the prism $(1,0,0)$, a $[001]$ beam gives $S_{yy}=u_b$ and basal beams give $u_c$, as in §4. A beam parallel to the normal (hexagonal $[110]\parallel(110)$ normal) raises.

Series truncation (script `check535b.py`): for $(\lambda_{\min},\rho)=(0.004,0.0015)$, $(0.001,0.02)$ and $(0.0005,0.05)$ Å², i.e. anisotropy ratios up to 100, the truncated series with the code's $N$ matches a brute angular average to $\le9\times10^{-16}$ over $q\le\sqrt{60/\lambda_{\min}}$ and $\Delta=0.5,3,20$ Å.

## Verdict

Every anisotropic formula re-derives exactly: the bracket, the $\pm q_y$ average, the diagonal with its half-circle symmetrization, the Jacobi–Anger series with $\epsilon_n$, $\cos n\varphi$ and $J_{2n}$, the truncation, the harmonic cut on $\lambda_{\min}$, the uniaxial $U$, and the azimuth rule. Independent quadratures agree to 4e-6, or 4e-5 explained by my own $q$ cut. No divergent factor. **`rederived`** for the #535 anisotropic extension.

Stated approximations, not defects: the projection uses $\hat{\mathbf b}_\parallel$ in place of the slightly tilted beam, an $O(\psi)$ change in $S$ at channeling angles $\psi\sim$ mrad. Sites are Einstein (uncorrelated) Gaussian. The $x$-kernel is the $y=y'$ trace of a $y$-translation-invariant beam.
