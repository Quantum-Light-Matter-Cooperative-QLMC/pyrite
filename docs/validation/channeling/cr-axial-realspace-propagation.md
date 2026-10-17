# `cr-axial-realspace-propagation`: real-space split-step propagation of axial transverse states

Fresh-context verification for #508 slice 2, at worktree HEAD `797a0159`. The derivation below was written from the ledger row and the module docstring, before the implementation body of `montecarlo/channeling/realspace.py` was read. Verdict: **rederived**. The implementation matches every term. The scope notes at the end describe accuracy limits, not discrepancies.

## 1. Paraxial transverse equation

Take an electron of energy $E=\gamma mc^2$ and momentum $p=\gamma m v$ along the axis $z$, in a static potential energy $U(\mathbf r)$ with $\mathbf r=(x,y)$. The Klein–Gordon dispersion relation $(E-U)^2=c^2(p_z^2+p_\perp^2)+m^2c^4$, linearized in $U\ll E$ and $p_\perp\ll p$, gives

$$
p_z\simeq p-\frac{p_\perp^2}{2p}-\frac{E}{pc^2}\,U=p-\frac{p_\perp^2}{2p}-\frac{U}{v}.
$$

Write $\Psi=e^{ipz/\hbar}\psi(\mathbf r,z)$ and $-i\hbar\partial_z\Psi=p_z\Psi$. Then $-i\hbar\partial_z\psi=(p_z-p)\psi$. Multiplying by $-v$ and using $vp_\perp^2/2p=p_\perp^2/2\gamma m$ with $p_\perp^2=-\hbar^2\nabla_\perp^2$ gives

$$
i\hbar v\,\partial_z\psi=\Big[-\frac{\hbar^2}{2\gamma m}\nabla_\perp^2+U(\mathbf r)-iW(\mathbf r)\Big]\psi .
$$

Here $W$ is added as the anti-Hermitian (no-jump) part. The equation is equivalent to $i\partial_z\psi+\frac{1}{2k}\nabla_\perp^2\psi=\frac{U-iW}{\hbar v}\psi$ with $k=p/\hbar$, because $\hbar^2/(2\gamma m\,\hbar v)=1/(2k)$.

**Norm.** $\partial_z|\psi|^2=-\frac{2W}{\hbar v}|\psi|^2+(\text{a divergence})$, so $N(z)=N_0e^{-2Wz/\hbar v}$ for uniform $W$ and $dN/dz=-\frac{2}{\hbar v}\int W|\psi|^2$ in general.

## 2. Strang splitting with the carrier factored out

Write $\psi=e^{i\mathbf k_t\cdot\mathbf r}\phi$. Then $\nabla_\perp^2\psi=e^{i\mathbf k_t\cdot\mathbf r}(\nabla_\perp+i\mathbf k_t)^2\phi$, and the plane-wave component $e^{i\mathbf k\cdot\mathbf r}$ of $\phi$ has kinetic eigenvalue $\hbar^2\lvert\mathbf k+\mathbf k_t\rvert^2/2\gamma m$. The potential is a multiplication operator, so it commutes with $e^{i\mathbf k_t\cdot\mathbf r}$. The equation for $\phi$ is therefore the same equation with $\mathbf k\to\mathbf k+\mathbf k_t$ in the kinetic symbol. This is an identity, so it holds for any $\mathbf k_t$. If $\psi$ is a Bloch function of quasi-momentum $\mathbf k_t$ on the supercell, $\phi$ is periodic. A mean tilt $\boldsymbol\theta$ gives $\mathbf k_t=k\boldsymbol\theta$, with $k=p/\hbar$.

The two exact sub-flows over a step $\Delta z$ are

$$
K(\Delta z)=\exp\!\Big[-\frac{i\hbar\lvert\mathbf k+\mathbf k_t\rvert^2\Delta z}{2\gamma m v}\Big]\ (\text{diagonal in }\mathbf k),\qquad
V(\Delta z)=\exp\!\Big[-\frac{i(U-iW)\Delta z}{\hbar v}\Big]\ (\text{diagonal in }\mathbf r).
$$

The symmetric product $K(\Delta z/2)V(\Delta z)K(\Delta z/2)$ matches $e^{-i(T+V)\Delta z/\hbar v}$ through second order in $\Delta z$, by the BCH expansion; the leading error is a double-commutator term of $O(\Delta z^3)$ (Strang 1968). The global error is $O(\Delta z^2)$. For $W=0$ both factors are unitary. Also $\lvert V\rvert^2=e^{-2W\Delta z/\hbar v}$, which is exact for uniform $W$ because a constant $W$ commutes with $K$.

The drift follows from a Galilean boost. If $u$ solves the free equation, then $\psi=e^{i\mathbf k_t\cdot\mathbf r-ik_t^2z/2k}\,u(\mathbf r-\boldsymbol\theta z,z)$ also solves it. The envelope drifts at $d\mathbf r/dz=\mathbf k_t/k=\boldsymbol\theta$.

## 3. Periodization of an isolated entry state

Use the convention $\tilde\psi(\mathbf k)=\int\psi\,e^{-i\mathbf k\cdot\mathbf r}d^2r$. Poisson summation gives

$$
\phi(\mathbf r)=\sum_{\mathbf R}\psi(\mathbf r+\mathbf R)=\frac1A\sum_{\mathbf G}\tilde\psi(\mathbf G)\,e^{i\mathbf G\cdot\mathbf r},\qquad
\mathbf G=2\pi\Big(\frac{m}{L_x},\frac{n}{L_y}\Big),\ A=L_xL_y .
$$

Parseval gives $\int_{\rm cell}|\phi|^2=A^{-1}\sum_{\mathbf G}|\tilde\psi(\mathbf G)|^2$. When the state fits in the cell, this Riemann sum approximates $(2\pi)^{-2}\int|\tilde\psi|^2d^2k=1$. On the $N_x\times N_y$ grid $\mathbf r_j$, with the inverse DFT defined as $\mathrm{ifft2}(a)_j=N^{-1}\sum_{\mathbf G}a_{\mathbf G}e^{i\mathbf G\cdot\mathbf r_j}$ and $N=N_xN_y$, the samples are

$$
\phi(\mathbf r_j)=\frac{N}{A}\,\mathrm{ifft2}\big[\tilde\psi(\mathbf G)\big]_j ,
$$

with the momenta in FFT order and truncated at Nyquist. The discrete norm is $\sum_j|\phi_j|^2A/N$. For the kinetic step, the forward FFT $\hat\phi_{\mathbf k}=\sum_j\phi_je^{-i\mathbf k\cdot\mathbf r_j}$ labels the component $e^{+i\mathbf k\cdot\mathbf r}$, so $K$ multiplies $\hat\phi_{\mathbf k}$ by the symbol at $\mathbf k+\mathbf k_t$.

## 4. Closed-form free-space references

- **Finite-energy Airy.** The equation $i\partial_z\psi+\frac1{2k}\partial_x^2\psi=0$ with $s=x/x_0$ becomes $i\partial_\xi\psi+\frac12\partial_s^2\psi=0$, where $\xi=z/(kx_0^2)$. Siviloglou & Christodoulides (2007, eq. 4) give $\psi=\mathrm{Ai}(s-\xi^2/4+ia\xi)\exp(as-a\xi^2/2-i\xi^3/12+ia^2\xi/2+is\xi/2)$.
- **Gaussian / Laguerre–Gauss.** The function $u=q^{-1}\exp(ik r^2/2q)$ with $q=z-iz_R$ and $z_R=kw^2/2$ solves the free equation, which can be checked by substitution. Its amplitude phase is $\arg(z+iz_R)=\pi/2-\arctan(z/z_R)$. With the $e^{+ipz/\hbar}$ carrier of §1, the Gouy phase of $LG_p^\ell$ is therefore $-(2p+\lvert\ell\rvert+1)\arctan(z/z_R)$, and the curvature phase is $+kr^2z/2(z^2+z_R^2)$.

## 5. Planar limit $\propto\theta^{-2}$

Tilt a plane wave by $\theta$ along $y$, within the plane, and split $U(x,y)=\bar U(x)+\sum_{g_y\neq0}U_{g_y}(x)e^{ig_yy}$. The $g_y\neq0$ components couple states whose kinetic energies differ by $\Delta_\pm=\frac{\hbar^2}{2\gamma m}(g_y^2\pm2g_yk_t)$. First-order amplitudes scale as $1/\Delta\propto\theta^{-1}$. Their cross terms with the main component vanish after integration over $y$. The second-order effective potential $\sum|U_g|^2/\Delta$ pairs $\pm g_y$: the sum $\frac1{g^2+2gk_t}+\frac1{g^2-2gk_t}\simeq-\frac1{2k_t^2}$ cancels the $\theta^{-1}$ terms. The $y$-integrated density therefore approaches the planar evolution as $\theta^{-2}$. If the potential is purely planar, $\phi$ stays independent of $y$. Its only $k_y$ is $0$, so the tilt within the plane gives the global phase $\hbar k_t^2z/2\gamma mv$, and the planar density is unchanged exactly.

## 6. Comparison with the implementation

| Derived | Code (`realspace.py`) | Agreement |
| --- | --- | --- |
| $k=p/\hbar=\gamma\beta mc/\hbar c$ | `momentum_inv_ang` (`states.py:61-64`) | exact |
| $\hbar v=\hbar c\,\beta$ | `self.hbar_v = HBARC_EV_ANG * beta` (L153) | exact |
| $\mathbf k_t=k\boldsymbol\theta$ | `self.tilt_inv_ang = (p*θx, p*θy)` (L155) | exact |
| $\hbar^2\lvert\mathbf k+\mathbf k_t\rvert^2/2\gamma m$ | `HBAR2_2M_EV_ANG2 / gamma * (kx**2+ky**2)` with the tilt added to the FFT momenta (L156-158) | exact; sign matches the forward-FFT label (§3) |
| $K(\Delta z/2)$ | `exp(-0.5j*step*kinetic/hbar_v)` (L171) | exact |
| $V(\Delta z)=e^{-i(U-iW)\Delta z/\hbar v}$ | `exp(-1j*step*(u - 1j*w)/hbar_v)` (L165, L172) | exact; decays for $W\ge0$ (validated L163) |
| $KVK$ order | L198-200 | exact |
| $\phi_j=(N/A)\,\mathrm{ifft2}[\tilde\psi(\mathbf G)]$ | L109-112 | exact; $\tilde\psi$ uses the $e^{-i\mathbf k\cdot\mathbf r}$ convention (`states.py:103,202`) |
| norm $\sum\lvert\phi\rvert^2A/N$ | `RealSpaceGrid.norm` (L95-97) | exact |

The constant $\hbar^2/2m=\hbar c^2/2mc^2$ with $\hbar c=1973.269804$ eV Å and $mc^2=510998.95$ eV is $3.80998$ eV Å². The ratio of the coded kinetic coefficient to $1/(2k)$ is $1$ to $2\times10^{-16}$ at 10 MeV.

## 7. Independent numerical checks

The scripts live in the session scratchpad (`validate-splitstep/{check,bigtilt,planar}.py`). Their references are closed forms or a dense matrix exponential, never the propagator itself.

| Check | Result |
| --- | --- |
| `sample` of a Gaussian $e^{-r^2/w^2}$ centred at $(7,-5)$ Å on a $40\times30$ Å, $128\times96$ grid, against the analytic periodized $\sqrt{2/\pi}\,w^{-1}e^{-r^2/w^2}$ | norm $1-9\times10^{-16}$; max error $6\times10^{-16}$ of peak (normalization and FFT sign) |
| Tilted Gaussian in free space, 200 keV, $\boldsymbol\theta=(2,-3)$ mrad, $z=0.7z_R$ and $3z_R$, against the boosted closed form of §2 | $5\times10^{-16}$ and $9\times10^{-16}$ of peak |
| Same, $\boldsymbol\theta=(100,50)$ mrad ($k_t=25$ Å⁻¹, against a grid Nyquist of $6.7$ Å⁻¹) | $1.1\times10^{-13}$: the carrier is exact far beyond the grid band |
| Strang with a tilt of $(4,-2.5)$ mrad, a non-separable $U$, and non-uniform $W$ on a $16^2$ grid, against $\exp(-iHz/\hbar v)$ (dense `expm`) at $z=1000$ Å | $L^2$ error $8.5\times10^{-4}$, $2.1\times10^{-4}$, $5.2\times10^{-5}$, $1.3\times10^{-5}$ for $\Delta z=10,5,2.5,1.25$ Å (ratios 4.04, 4.01, 4.00) |
| $W=0$ norm after 5000 Å | $1+1.3\times10^{-13}$ |
| $dN/dz$ for non-uniform $W$, against $-\frac2{\hbar v}\int W\lvert\phi\rvert^2$ | $-3.137876\times10^{-4}$ against $-3.137876\times10^{-4}$ Å⁻¹ |
| Ledger anchor suite (`pyrite-dev test --slow tests/montecarlo/test_channeling_axial.py`) | 17 passed (35.5 s) |

The anchor tests independently reproduce the §4 closed forms. The Airy test uses $\xi=z/kx_0^2$ and eq. 4 as derived above. The LG test uses the curvature $+kr^2z/2(z^2+z_R^2)$ and the Gouy phase $-(2p+\lvert\ell\rvert+1)\arctan(z/z_R)$. Both are single free-space steps. Since $V=1$ there, they exercise only the kinetic symbol at $\mathbf k_t=0$. The tilted-Gaussian check above covers $\mathbf k_t\neq0$.

## 8. Scope notes (not discrepancies)

1. **"Any tilt is exact" refers to the representation, not to the splitting error.** The Strang commutator grows with $k_t$, because the envelope translates by $\theta\Delta z$ per step across the potential's features. The table gives the $L^2$ error against `expm` on Si ⟨001⟩ ($24^2$ grid, 10 MeV, $z=2000$ Å):

   | $\theta_y$ | $\Delta z=4$ Å | $\Delta z=2$ Å | $\Delta z=1$ Å |
   | --- | --- | --- | --- |
   | 0 | $1.2\times10^{-3}$ | $3.1\times10^{-4}$ | $7.7\times10^{-5}$ |
   | 20 mrad | $7.8\times10^{-4}$ | $1.9\times10^{-4}$ | $4.8\times10^{-5}$ |
   | 100 mrad | $5.0\times10^{-2}$ | $2.4\times10^{-3}$ | $1.7\times10^{-4}$ |

   The step must satisfy $\theta\,\Delta z\ll$ (string feature size).

2. **The planar-limit numbers mix in a splitting error at the largest tilt.** The ledger quotes 9.6, 1.6 and 0.42 % at $\Delta z=2$ Å. At $\Delta z=0.5$ Å the errors are 9.62, 1.57 and 0.387 % at 20, 50 and 100 mrad, plus 0.098 % at 200 mrad. Successive ratios are 6.13, 4.06 and 3.94, so the converged ratio matches $\theta^{-2}$ more closely. At $\Delta z=2$ Å the 100 mrad point carries about 9 % splitting contamination, and the 200 mrad point stalls at 0.37 %. The test tolerance (`rel=0.25`) covers this.
3. **The model is paraxial in the axis frame.** The model uses $\mathbf k_t=k\boldsymbol\theta$ rather than $k\sin\boldsymbol\theta$, and drops the quartic dispersion term $p_\perp^4/8p^3$. Relative to the exact dispersion, the transverse group velocity has an $O(\theta^3)$ correction and the transverse mass has an $O(\theta^2)$ correction, about $1.5\,\theta^2$, or 1.5 % at 100 mrad. This is the stated model, and it is unchanged by the splitting.
4. **The "tilt within the plane leaves planar evolution unchanged" check is structurally trivial.** For a $y$-uniform $\phi$ under a $y$-independent $U$, $k_t$ enters only as a global phase, so the test cannot detect a wrong sign or magnitude of $k_t$. The tilted-Gaussian drift check in §7 covers both.
