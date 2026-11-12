# `cr-planar-exit-out-of-plane` re-derivation

Claim: `montecarlo/channeling/exit_out_of_plane.py::{eta_nodes,out_of_plane_kernel,propagate_batch,exit_cut,out_of_plane_exit_cut,local_contrast,ExitAngularCut}` and the `out_of_plane_ang` paths of `montecarlo/channeling/operators.py::{tds_kernel,electronic_kernel}`. Sources: the $\kappa_y$ label of `cr-planar-fiber-unraveling` (Castin, Dalibard & Mølmer, arXiv:0805.4002), the kernels of `cr-lindblad-jump-operators`, and the master equation of `cr-density-matrix-split-operator`. Gridnev, Khlabutin, Rozum & Vorobiev, phys. stat. sol. (b) 140, K13 (1987) (Zotero `JH2VIWAJ`) supplies only the measured spot counts; it was not reopened here, because the claim under test is the transform, not the measurement.

Fresh-context verification, 2026-11-11 (issue #1116). The derivation below was written from the ledger row and the module/function docstrings before any implementation body was read.

## Independent derivation (before reading the body)

**State and label.** Take $x$ along the plane normal, $y$ in the plane and $z$ along the beam. The planar model averages the crystal over $y$, so the static potential and the $y$-translation-averaged kernels do not depend on $y$. Write the transverse state in the mixed representation $\lvert\phi\rangle\otimes\lvert\kappa_y\rangle$. A beam that is incoherent (translation-averaged) in $y$ has a density matrix diagonal in $\kappa_y$:

$$
\hat\rho=\int d\kappa_y\,\rho_{\kappa_y}\otimes\lvert\kappa_y\rangle\langle\kappa_y\rvert .
$$

The transverse Hamiltonian is $H_x+\hbar^2\kappa_y^2/2\gamma m$. Its second term is a c-number on each $\kappa_y$ sector and cancels in the commutator with a $\kappa_y$-diagonal $\hat\rho$, so the coherent part of each sector evolves under the same $H_\perp=H_x$.

**Jumps.** Each TDS or electronic jump carries a factor $e^{i(q_xx+q_yy)}$. Under the no-folding assumption ($g_y/2$ above every transfer, as in `cr-planar-fiber-unraveling`) $y$ is a continuum, and the factor $e^{iq_yy}$ maps $\lvert\kappa_y\rangle\to\lvert\kappa_y+q_y\rangle$. It therefore preserves $\kappa_y$-diagonality. Resolve the kernel of `cr-lindblad-jump-operators` in $q_y$:

$$
C(x,x')=\int dq_y\,c(x,x';q_y),\qquad
\mathcal J_{q_y}(\rho)(x,x')=c(x,x';q_y)\,\rho(x,x').
$$

Then

$$
\frac{d\rho_{\kappa}}{dz}=\mathcal L_0\rho_{\kappa}+\int dq_y\,\mathcal J_{q_y}\bigl(\rho_{\kappa-q_y}\bigr),
\qquad
\mathcal L_0\rho=-\frac{i}{\hbar v}\bigl(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger\bigr),\quad
H_{\rm eff}=H_\perp-\tfrac{i\hbar v}{2}C(x,x).
$$

The loss is the total jump rate out of a sector, $\int dq_y\,c(x,x;q_y)=C(x,x)$, which is the same for every $\kappa$. Integrating over $\kappa$ gives back the planar equation of `cr-density-matrix-split-operator`.

**Fourier transform.** Define $\tilde\rho(\eta)=\int d\kappa\,e^{i\kappa\eta}\rho_\kappa$. Then

$$
\int d\kappa\,e^{i\kappa\eta}\int dq_y\,\mathcal J_{q_y}(\rho_{\kappa-q_y})
=\int dq_y\,e^{iq_y\eta}\,\mathcal J_{q_y}\bigl(\tilde\rho(\eta)\bigr),
$$

so

$$
\boxed{\frac{d\tilde\rho(\eta)}{dz}=\mathcal L_0\tilde\rho(\eta)+C_\eta(x,x')\,\tilde\rho(\eta)(x,x'),\qquad
C_\eta=\int dq_y\,c(x,x';q_y)\,e^{iq_y\eta}}
$$

The equation is closed for each $\eta$, because $\mathcal L_0$ does not depend on $\kappa$. The opposite sign convention, $e^{-i\kappa\eta}$, gives $C_{-\eta}$. If $c(x,x';-q_y)=c(x,x';q_y)$, then $C_\eta=\int dq_y\,c\cos(q_y\eta)=C_{-\eta}$ is real-weighted and Hermitian in $(x,x')$, the sign convention drops out, and $\tilde\rho(\eta)$ stays Hermitian. Without that symmetry,

$$
C_\eta=\int dq_y\,c_{\rm even}\cos(q_y\eta)+i\int dq_y\,c_{\rm odd}\sin(q_y\eta).
$$

**Parity of the kernels.** In the TDS kernel, $v(\lvert\mathbf q\rvert)$ depends on $q_y^2$. The Debye–Waller bracket $e^{-(\mathbf q\cdot S\mathbf q+\mathbf q'\cdot S\mathbf q')/2}$ contains $2S_{xy}q_xq_y$, which is odd in $q_y$ unless $S_{xy}=0$. The electronic LDA kernel depends only on $\lvert\mathbf q_\perp\rvert$ (through $w$ and the smoothing width $1/\lvert\mathbf q_\perp\rvert$). A uniaxial valence response about one of the frame axes depends on $q_x^2$ and $q_y^2$ separately, so it is also even. The cosine-only transform therefore requires exactly $S_{xy}=0$, as claimed, and also an electronic response with no $x$–$y$ cross term.

**Loss unchanged; $\eta=0$ limit.** Only the gain carries $e^{iq_y\eta}$. The loss $C(x,x)$ sits in $\mathcal L_0$ and is the same for every $\eta$. At $\eta=0$, $C_0=C$, so $\tilde\rho(0)=\int d\kappa\,\rho_\kappa$ is the $\theta_y$-integrated planar state. Because $\lvert\cos\rvert\le1$, $\lvert\operatorname{tr}C_\eta\rvert\le\operatorname{tr}C$.

**Gaussian divergence and inversion.** Take a separable entry, $\rho_\kappa(0)=\rho_x(0)\,f(\kappa)$, with $f$ a normalized Gaussian of width $\sigma_\kappa=p\sigma_y$. Here $p$ is the longitudinal wavenumber $p/\hbar$ in Å$^{-1}$, and $\theta_y=\kappa_y/p$ paraxially. Then $\tilde\rho(\eta,0)=\rho_x(0)\,e^{-(p\sigma_y\eta)^2/2}$. By linearity, $\tilde\rho(\eta,L)=e^{-(p\sigma_y\eta)^2/2}\,\Phi_\eta(L)[\rho_x(0)]$, where $\Phi_\eta$ is the $\eta$-modified planar evolution. Let $\tilde P(\theta_x,\eta)$ be the plane-wave diagonal of $\Phi_\eta(L)[\rho_x(0)]$. It is the Fourier transform of the nonnegative $\kappa_y$ distribution produced by a $\delta(\kappa_y)$ entry, so it is real, even in $\eta$, and $\lvert\tilde P(\theta_x,\eta)\rvert\le\tilde P(\theta_x,0)$. Inverting,

$$
\frac{dP}{d\kappa_y}=\frac1{2\pi}\int_{-\infty}^{\infty}d\eta\,e^{-i\kappa_y\eta}e^{-(p\sigma_y\eta)^2/2}\tilde P
=\frac1\pi\int_0^\infty d\eta\,\cos(\kappa_y\eta)\,e^{-(p\sigma_y\eta)^2/2}\tilde P,
$$

and with $dP/d\theta_y=p\,dP/d\kappa_y$,

$$
\boxed{\frac{dP}{d\theta_y}(\theta_x,\theta_y)=\frac p\pi\int_0^\infty d\eta\,\cos(p\theta_y\eta)\,e^{-(p\sigma_y\eta)^2/2}\,\tilde P(\theta_x,\eta)}
$$

This has units of rad$^{-1}$ per $\theta_x$ component. It matches the ledger, including the factor $p/\pi$ (not $p/2\pi$), which comes from folding the even integrand onto the half-line. **Zero kernel.** $\tilde P$ is independent of $\eta$, and $\int_0^\infty\cos(a\eta)e^{-b^2\eta^2/2}d\eta=\sqrt{\pi/2}\,e^{-a^2/2b^2}/b$ gives $dP/d\theta_y=P_{\rm marg}(\theta_x)\,e^{-\theta_y^2/2\sigma_y^2}/(\sqrt{2\pi}\sigma_y)$. **Normalization.** $\int d\theta_y$ of the boxed form gives $(p/\pi)(\pi/p)\tilde P(\theta_x,0)=P_{\rm marg}$.

**Truncation at $\eta_{\max}=5/(p\sigma_y)$.** Using $\lvert\tilde P(\eta)\rvert\le\tilde P(0)$ and $\lvert\cos\rvert\le1$,

$$
\Bigl\lvert\frac p\pi\int_{\eta_{\max}}^\infty\cdots\Bigr\rvert
\le\frac{P_{\rm marg}}{\sqrt{2\pi}\sigma_y}\,\operatorname{erfc}\bigl(5/\sqrt2\bigr)
=5.7\times10^{-7}\times\bigl(\text{zero-kernel peak of that component}\bigr).
$$

That $\tilde P$ also decays in $\eta$ only tightens the bound. As $\eta\to\infty$, $\tilde P$ tends to the weight that never scattered (or scattered with $q_y\to0$), not to zero, so the Gaussian factor, not the decay of $\tilde P$, is what makes the truncation safe. The truncation is not the limiting error. Quadrature resolution of $\cos(p\theta_y\eta)$ is: with $4\,\text{Å}$ panels of 4 Gauss nodes beyond $\eta=4\,\text{Å}$, the cut is accurate only while $4p\theta_y$ is a few radians, which holds near $\theta_y=0$ (see Findings).

## Cheap filters

| Filter | Result |
| --- | --- |
| Units | $q_y\eta$ is dimensionless (Å$^{-1}\cdot$Å). $p$ is $p/\hbar$ in Å$^{-1}$. $p\,d\eta$ is dimensionless, so $dP/d\theta_y$ is per rad. The kernel keeps the units of $C$ (Å$^{-1}$ per unit depth on the diagonal), because $\cos$ is dimensionless. Pass. |
| Limits | $\eta=0$ gives the default kernel and the marginal. A zero kernel gives marginal $\times$ Gaussian. $\int d\theta_y$ gives the marginal. $\lvert C_\eta\rvert$ is bounded by $C$. Pass. |
| Signs/conventions | The Fourier sign drops out for $q_y$-even kernels. $p/\pi$ on the half-line is correct. The sine term is needed iff $S_{xy}\ne0$, and the code refuses that case. Pass. |

## Diff against the implementation

- `tds_kernel(..., out_of_plane_ang=η)` (`operators.py:754–767`) refuses $\lvert S_{xy}\rvert>10^{-12}\max(S_{xx},S_{yy})$ when $\eta\ne0$. It then takes $q_y$ nodes from `_out_of_plane_panels`, which returns the plain composite rule at $\eta=0$ and `_oscillatory_panels` weights times $\cos(q_y\eta)$ otherwise. `_tds_order_strengths` integrates the half-line with a factor 2, using `weights` (which carry the cosine) for both terms of the bracket $e^{-S_{xx}G^2/2}\,\Sigma - e^{-S_{xx}(p^2+p'^2)/2}\Sigma_{e^{-S_{yy}q_y^2}}$. That is $\int dq_y\,c\cos(q_y\eta)$ over the gain. `diagonal_coefficients` comes from `_tds_diagonal`, which has no $\eta$. Matches.
- `electronic_kernel(..., out_of_plane_ang=η)` (`operators.py:1022–1035`) uses the same `_out_of_plane_panels`. The Abel weight is `2 q_yw · weight_pq(p_x, q_y)` with $q=\sqrt{p_x^2+q_y^2}$, so the cosine multiplies the gain. `_electronic_diagonal` has no $\eta$. Matches.
- `_oscillatory_panels` caps panels at half a $\cos$ period up to $Q_1=2\pi\cdot400/\eta$ and tapers the tail with $\cos^2$ beyond $0.75Q_1$. For $\eta\ge0.014$ Å, $Q_1\le1.8\times10^5$ Å$^{-1}$. For $\eta\gtrsim0.25$ Å, $Q_1<10^4$ Å$^{-1}$ and the taper is active. The dropped part is the oscillatory tail of a smooth integrand. Acceptable. It is not tested independently here except through check (c).
- `eta_nodes`: composite 4-point Gauss–Legendre on $[0,1]$ (panel $0.2$ Å), $[1,4]$ ($0.8$ Å) and $[4,\eta_{\max}]$ (at most $4$ Å, and at most $1/(p\sigma_y)$). $\eta_{\max}=\max(5/(p\sigma_y),4)$. Node 0 is $\eta=0$ with weight 0. Matches the docstring.
- `exit_cut`: `density = p/π · Σ w_i cos(p θ_y η_i) exp(−(p σ_y η_i)²/2) W_i`, with `p` from the solver's kinetic energy ($\gamma\beta mc^2/\hbar c$). `marginal` is `weights[0]`, with unresolved $1-\sum$. This is exactly the boxed inversion. Matches.
- `propagate_batch`: the same Strang step as `DensityMatrixSolver.propagate` (half no-jump with the base solver's loss, jump map $1+dz\,\mathcal J+dz^2\mathcal J^2/2$, half no-jump), with one cell kernel per batch member. The per-$\eta$ equation keeps the $\eta=0$ loss, as derived. It adds a Hermitian symmetrization after each step, which is harmless because $\tilde\rho(\eta)$ is Hermitian for even kernels. Matches.
- `out_of_plane_exit_cut`: $\eta=0$ uses `solver._cell_kernel0` (the default kernel), and $\eta>0$ uses `out_of_plane_kernel(η).cell_kernel`. The entry is `solver.entry_state(0, divergence_rad)` (the in-plane Gaussian on the Bloch grid), and $\sigma_y$ defaults to the in-plane divergence. Matches.
- `local_contrast`: the grid value nearest $\theta_x=ms$, divided by the mean of grid values with $0.15<\lvert\theta_x/s-m\rvert<0.35$. It raises if the ring is empty. Matches the ledger text (assessment below).

## Independent numeric checks

The scripts are in `scratch/` of the verification worktree (not committed). They use Si(110) at 293.15 K, total 1.87 MeV ($T=1.359$ MeV, $p=911.7$ Å$^{-1}$, $p\sigma_y=0.1596$ Å$^{-1}$ at $\sigma_y=0.175$ mrad), with TDS plus electronic default kernels.

**(a) Zero-kernel Gaussian limit** (`verify_oop_ab.py`). The $\eta$ rule `eta_nodes` integrates $(p/\pi)\cos(p\theta_y\eta)e^{-(p\sigma_y\eta)^2/2}$. The table compares it with the analytic $e^{-\theta_y^2/2\sigma_y^2}/(\sqrt{2\pi}\sigma_y)$:

| $\sigma_y$ (mrad) | nodes | $\eta_{\max}$ (Å) | max error / peak, $\theta_y\in[0,4\sigma_y]$ |
| --- | --- | --- | --- |
| 0.10 | 89 | 54.6 | $\le5.0\times10^{-7}$ |
| 0.175 | 65 | 31.1 | $5.73\times10^{-7}$ (81-point scan) |
| 0.25 | 57 | 21.7 | $\le7.5\times10^{-7}$ |
| 1.0 | 45 | 5.4 | $\le5.7\times10^{-7}$ |

The residual at $\theta_y=0$ is $-5.73\times10^{-7}$, which equals $\operatorname{erfc}(5/\sqrt2)=5.733\times10^{-7}$, the analytic truncation bound. `exit_cut` fed random $\eta$-independent weights ($n_{\rm cells}=8$, $H=4$) returns marginal $\times$ Gaussian to $5.7\times10^{-7}$ of the peak at $\theta_y=0$, $\sigma_y$ and $2.5\sigma_y$.

**(b) $\eta=0$ marginal** (`verify_oop_ab.py`). With $n_{\rm cells}=8$, $H=4$, $L=2000$ Å, $dz=20$ Å and a plane-wave entry, `out_of_plane_exit_cut(...).marginal.weights` equals `open_exit_distribution` on a solver with the same default kernel to $1.1\times10^{-16}$, with identical angles. The unresolved weight is 0.075628 against 0.075606. The difference, $2.2\times10^{-5}$, is the integrator's balance defect: `exit_cut` uses $1-\sum$, while `open_exit_distribution` uses the solver's flux tally, as documented. Integrating the cut over $\theta_y\in[-8\sigma_y,8\sigma_y]$ (321 points) recovers the marginal except for up to 0.011 of a maximum component of 0.18. That deficit is the wide out-of-plane halo: in (c), about 10–16% of the weight sits at $\lvert\kappa_y\rvert>8p\sigma_y$. The cut is positive everywhere on the scan.

**(c) Brute-force $\kappa_y$-resolved propagation** (`verify_oop_bruteforce.py`).

- **Partition.** I split the default kernel into $\lvert q_y\rvert$ bins of width $\Delta$, by masking the $\eta=0$ $q_y$ quadrature weights inside the builders. The bins sum back to the default `momentum_kernel` to $\le7\times10^{-16}$. The weight beyond $K_{\max}$ is dropped from the window.
- **Jump.** I wrote my own plane-wave jump superoperator, $\rho(\kappa_s+o_1g,\kappa_s+o_2g)\to K[r,i,j]$ shifted into block $(s+r)\bmod n_{\rm cells}$ with carry. It equals `DensityMatrixSolver.jump` to $5\times10^{-16}$ on a random Hermitian $\rho$.
- **Propagation.** I put an explicit $\kappa_y$ grid on $[-K_{\max},K_{\max}]$, with each bin-$j$ jump adding half its output at $\kappa\pm j\Delta$. My own Strang loop uses the solver's no-jump half-step. The entry is $\delta(\kappa_y)$ times the solver's entry, and I convolve the final $P(\theta_x,\kappa_y)$ with the Gaussian $p\,\mathcal N(p\theta_y-\kappa;p\sigma_y)$ myself.
- **Reference.** The Fourier route is `propagate_batch` plus `exit_cut` at the same $n_{\rm cells}=4$, $H=3$, $L=2000$ Å, $dz=20$ Å and $\sigma_y=0.175$ mrad.

| $\Delta$ (Å$^{-1}$) | $K_{\max}$ (Å$^{-1}$) | bins | cut $\theta_y=0$: max diff / peak | $\theta_y=\sigma_y$ | $\theta_y=2\sigma_y$ | order 0 brute / code (rad$^{-1}$) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 3 | 43 | $1.7\times10^{-3}$ | $7.9\times10^{-4}$ | $3.7\times10^{-3}$ | 367.53 / 368.15 |
| 0.025 | 3 | 67 | $1.5\times10^{-3}$ | $1.2\times10^{-3}$ | $1.2\times10^{-3}$ | 367.61 / 368.15 |
| 0.025 | 6 | 87 | $1.4\times10^{-3}$ | $1.1\times10^{-3}$ | $1.3\times10^{-3}$ | 367.64 / 368.15 |
| 0.0125 | 3 | 87 | $8.1\times10^{-5}$ | $1.8\times10^{-4}$ | $6.2\times10^{-4}$ | 368.14 / 368.15 |
| 0.00625 | 3 | 107 | $1.7\times10^{-4}$ | $1.1\times10^{-4}$ | $1.0\times10^{-3}$ | 368.20 / 368.15 |

Order 1 at $\Delta=0.0125$ is 354.258 (brute) against 354.253 (code). At $\theta_y\le\sigma_y$ the residual falls from about $1.5\times10^{-3}$ to a floor of $1$–$2\times10^{-4}$ of the peak as $\Delta$ shrinks, and it is insensitive to $K_{\max}$. That is the $q_y$-rounding error of the brute force plus the quadrature floor of both routes. At $\theta_y=2\sigma_y$, where the cut is 20% of its axial peak, a residual of about $10^{-3}$ of that cut's peak (median 0.3–0.5% on strong components) does not decrease with $\Delta$ or $K_{\max}$. I attribute it to the code's tapered oscillatory $q_y$ panels at $\eta>0$ or to the $\eta$ rule acting on the non-Gaussian $\tilde P$. It is far below the 2% contrast systematics budget and well below any spot-visibility margin, so it is recorded but not pursued. The transform $\tilde P(\theta_x,\eta)=\sum_\kappa P\cos(\kappa\eta)$ of the brute-force distribution matches the code's per-$\eta$ diagonals to $\le3.5\times10^{-3}$ of $\max\tilde P\approx0.19$ for $0.27\le\eta\le3.9$ Å. That residual is dominated by the transfers dropped beyond $K_{\max}$, which still have $\cos(q_y\eta)\approx1$ at small $\eta$. It changes when $K_{\max}$ goes from 3 to 6 Å$^{-1}$ (e.g. at $\eta=0.59$ Å, from $3.5\times10^{-3}$ to $5\times10^{-5}$), and at $\eta=0$ it is the 1.3% (0.8%) of weight lost from the window. The film cut near $\theta_y=0$ is insensitive to this.

## Is `local_contrast` a fair film-visibility proxy?

The proxy is the $\theta_y=0$ cut at $\theta_x=ms$, divided by the mean of the same cut over $0.15<\lvert\theta_x/s-m\rvert<0.35$. It is a reasonable *relative* indicator: it measures how far the spot stands above the diffuse background next to it, which is what a person counting spots sees. The marginal hid exactly this, by adding the wide $\theta_y$ halo under every order. Its limits:

1. **No intensity scale; nonlinear film.** The ratio has no exposure, fog or density–log-exposure response attached. A threshold of 1.2 is a calibration chosen against the four photographs (three of four match). It is not a property derived from film physics, so the visible/invisible verdict near the threshold (Si(110) 4.15 MeV order $\pm2$: 1.04–1.13) is not discriminating.
2. **Ring fixed in units of $s$, not $\sigma$.** The ring spans $0.15s$–$0.35s$. At Si(110) 1.87 MeV ($s=3.59$ mrad) that is $3.1\sigma$–$7.2\sigma$, so the spot's own Gaussian is gone from the ring. At 4.15 MeV Si(110), 5.70 MeV Si(100) and 2.45 MeV Si(111), $s=1.57$–$1.65$ mrad and $\sigma/s=0.106$–$0.112$, so the ring starts at $1.34\sigma$–$1.41\sigma$, where an isolated Gaussian spot is still about 40% of its peak. A background-free Gaussian spot of width $\sigma$ then has ring contrast 8.1–9.6, against $1.6\times10^3$ at Si(110) 1.87 MeV (`scratch/ring_gauss.py`). The same physical spot-to-background ratio therefore maps to different contrasts in different cases, and the common threshold compares unlike quantities. The ring also contains part of the spot when the spot is blurred by in-plane divergence, beam size or film resolution.
3. **1D cut, not a 2D neighbourhood.** The ring lies along $\theta_x$ at $\theta_y=0$. The eye compares a spot with its 2D surround, and in $\theta_y$ the background falls on the halo scale. An annulus contrast would generally differ. The numbers below compare the two for one case.
4. **Single point; no spot integration.** The numerator is one grid value at the exact order centre. A film integrates over the spot's area and over the finite beam spot and grain. The ratio depends on $\sigma_y$ through the peak height, which is why the study's 4.15 MeV contrast moves from 0.99–1.03 at $\sigma=0.25$ mrad to 1.15–1.37 at 0.10 mrad.
5. **Neighbouring orders and diffuse structure.** The outer ring edge sits $0.65s$ from the neighbouring order. This is harmless while $\sigma\ll s$, but once the halo or rainbow structure between orders is peaked, the ring mean rises and the contrast falls even if a spot is visible. Merged orders ($\sigma\gtrsim0.2s$) would be counted as no spot. The ring averages the $m-1$ and $m+1$ sides, so an asymmetric background is not seen.
6. **Model scope.** Every electron counts, regardless of energy loss (no energy-loss feedback in the kernels). The dechanneled (unresolved) flux has no angle and is absent from both numerator and denominator, although on film it adds a broad background that lowers contrast. The proxy therefore overstates visibility by roughly the unresolved fraction (0.08–0.14 here) spread over the field.

**Numbers** (`verify_oop_contrast.py`; Si(110) total 1.87 MeV, 2 µm, $\sigma=\sigma_y=0.175$ mrad, $n_{\rm cells}=32$, $H=8$, $dz=20$ Å). The 2D annulus contrast takes the same centre value over the mean of the 2D density on $0.15<\sqrt{(\theta_x/s-m)^2+(\theta_y/s)^2}<0.35$, sampled with 57 $\theta_y$ cuts.

| order $m$ | cut ring (`local_contrast`) | own ring re-implementation | 2D annulus | marginal ring |
| --- | --- | --- | --- | --- |
| 0 | 1.876 | 1.876 | 2.076 | 1.026 |
| 1 | 1.914 | 1.914 | 2.000 | 1.095 |
| 2 | 0.988 | 0.988 | 1.027 | 0.994 |
| 3 | 0.992 | 0.992 | 1.001 | 0.998 |

These reproduce the study record's $n_{\rm cells}=32$ row exactly (1.8756/1.9141/0.9881; marginal 1.0256/1.0949). The annulus and the 1D ring agree on which orders stand out, and on this case they agree with the photograph (3 spots) at any threshold between 1.03 and 1.87. The annulus is 4–11% higher for the visible orders. The proxy is therefore fair for *ranking* spots and for clear-cut cases, but a threshold-level verdict (contrast within about 0.1 of 1.2) is not robust to the choice of neighbourhood.

**Grid convergence of the contrast.** The study's own rows at this setting give order-0 contrast 1.876 / 1.687 / 1.657 and order-1 1.914 / 1.991 / 2.008 at $n_{\rm cells}=32/64/128$. Between 64 and 128 cells the change is below 2%, but between 32 and 128 it is 13% (order 0) and 5% (order 1). At 32 cells the in-plane divergence is sampled at $s/32=0.112$ mrad $=0.64\sigma$, and the single-point centre value is sensitive to that.

## Findings

None of these is a discrepancy in the transform or its implementation.

1. **Grid convergence wording (ledger).** The ledger says the cut contrasts converge to below 2% across $n_{\rm cells}=32/64/128$. The study record shows that only for 64→128. From 32 to 128, the Si(110) 1.87 MeV order-0 contrast moves by 13% (1.876→1.657) and order 1 by 5% (1.914→2.008). The slow anchor runs at $n_{\rm cells}=32$ and asserts contrast $>1.6$ at orders 0/±1. It passes at 32 and would still pass at 128, but order 0 at 128 cells (1.657) has only a 4% margin.
2. **$\theta_y$ range of `exit_cut`.** The $\eta$ rule has no node beyond $\eta_{\max}=5/(p\sigma_y)$, and its outer panels are up to $1/(p\sigma_y)$ wide. It therefore resolves $\cos(p\theta_y\eta)$ only for moderate $\theta_y$. With a zero kernel the absolute error over the peak is $\le2\times10^{-5}$ up to $8\sigma_y$, $3.9\times10^{-4}$ at $10\sigma_y$, $1.9\times10^{-3}$ at $15\sigma_y$, and 6% at $20\sigma_y$ (`scratch/thy_range.py`). A halo-like component $\tilde P\propto e^{-b\eta}$ ($b=1$ or $3$ Å$^{-1}$) is integrated to $\le1.4\times10^{-4}$ up to $15\sigma_y$. Every PyRITE use is $\theta_y=0$. The docstring should state the validity range ($\lvert\theta_y\rvert\lesssim10\sigma_y$), since `theta_y_rad` is a public argument.
3. **Halo outside the cut window.** About 10–16% of the weight in check (c) sits at $\lvert\kappa_y\rvert>8p\sigma_y$. The `marginal` therefore contains a broad component that no near-axis cut shows, so $\int d\theta_y$ over a few $\sigma_y$ falls short of the marginal by about 6% of a component. This is physical and consistent, but worth saying, because the marginal and the cut are not normalized "to the same field".
4. **Unresolved weight.** `ExitAngularCut.marginal.unresolved` is $1-\sum$ (it includes the balance defect), while `open_exit_distribution` uses the flux tally. They differ by $2.2\times10^{-5}$ in check (b). This is documented and harmless.
5. **Parity scope.** The cosine-only transform needs every kernel to be even in $q_y$. TDS enforces this through the $S_{xy}$ refusal. The electronic kernel is even for the scalar LDA and for a directional valence whose `c_axis` is restricted to `'x'`, `'y'` or `'z'` (`directional.py:112`), so no refusal is needed today. A future off-axis valence tensor would need the same guard as TDS.
6. **`local_contrast` as a visibility proxy.** It is fair for ranking and for clear-cut cases. Its ring is fixed in units of $s$ rather than $\sigma$ (background-free contrast 8–10 at $s\approx1.6$ mrad against $1.6\times10^3$ at 3.6 mrad). It is a 1D single-point ratio (a 2D annulus is 4–11% higher), it has no film response, and it omits the angle-less dechanneled flux. The 1.2 threshold is a calibration, so verdicts within about 0.1 of it, including the Si(110) 4.15 MeV order ±2 at 1.04–1.13, are not discriminating. The ledger already says "the margin at the threshold is thin". The quantified 4.15 MeV discrepancy should stay framed as model plus proxy, not as a kernel error.

## Verdict

**`rederived`.**

- **Transform.** The $\kappa_y$-label master equation, its per-$\eta$ closure, the $\cos(q_y\eta)$ gain weight with the loss unchanged, and the $S_{xy}=0$ restriction (a sine term otherwise) match an independent derivation.
- **Inversion.** The $(p/\pi)\int_0^\infty$ inversion with the Gaussian factor also matches, with $p$ the wavenumber in Å$^{-1}$ and the result per rad.
- **Truncation.** Truncating at $5/(p\sigma_y)$ is bounded by $\operatorname{erfc}(5/\sqrt2)=5.7\times10^{-7}$ of the zero-kernel peak.
- **Brute force.** An own brute-force $\kappa_y$-resolved propagation agrees with the Fourier route to $\le2\times10^{-4}$ of the peak at $\theta_y\le\sigma_y$ (converged in the $q_y$ bin width) and to $10^{-3}$ at $2\sigma_y$.
- **Visibility proxy.** `local_contrast` is implemented as specified. It is a reasonable but threshold-sensitive visibility proxy (Findings 1 and 6).

Human sign-off remains in #277.
