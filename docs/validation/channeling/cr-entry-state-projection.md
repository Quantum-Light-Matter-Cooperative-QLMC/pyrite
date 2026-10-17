# `cr-entry-state-projection`: user entry states on the planar Bloch basis

Implementation-context specification for #508 slice 1. Fresh-context verification is pending.

## Representation

A transverse entry state is defined by its normalized momentum amplitude

$$
\tilde\psi(\mathbf k)=\int\psi(\mathbf r)\,e^{-i\mathbf k\cdot\mathbf r}\,d^d r,\qquad
\int|\tilde\psi|^2\,\frac{d^dk}{(2\pi)^d}=1,
$$

with $x$ along the plane normal and $y$ in the plane. A mean tilt $\psi$ shifts $k_x\to k_x+p\psi$ with $p=\gamma\beta m c/\hbar$. Mixed states are $\rho=\sum_i w_i|\psi_i\rangle\langle\psi_i|$.

## Projection

The planar Hamiltonian $H_\perp$ and the Lindblad dissipator of `cr-lindblad-jump-operators` are invariant under the lattice translation $T_d$. For any lattice-periodic observable $O$ (band populations, line operators $d_c^\dagger d_c$), $\operatorname{tr}(O\,\mathcal E_z[\rho])=\operatorname{tr}(O\,\mathcal E_z[\bar\rho])$, where $\bar\rho$ is the average of $T_d^j\rho T_d^{-j}$ over $j$ and $\mathcal E_z$ is the evolution map, which commutes with $T_d$ conjugation. On the solver's supercell of $n$ periods, momenta are $k=\kappa_r+\nu g$ with $\kappa_r=rg/n$. $T_d$ multiplies $|k\rangle$ by $e^{-ikd}$, so $\bar\rho$ keeps exactly the elements with equal $\kappa$:

$$
\bar\rho_{r}[\nu,\nu']=\int_{\text{bin } r}\frac{d\delta}{2\pi}\;
\tilde\psi(\kappa_r+\nu g+\delta)\,\tilde\psi^*(\kappa_r+\nu' g+\delta),
\qquad |\delta|<\tfrac{g}{2n}.
$$

The continuous momentum $\delta$ inside a bin is assigned to $\kappa_r$, the same discretization the solver's divergence entry uses. A probe's position modulo $d$ relative to the planes therefore matters (`impact="fixed"`). If the entry point is instead uniform over one period, the additional average $\frac1d\int_0^d e^{-i(\nu-\nu')ga}\,da=\delta_{\nu\nu'}$ removes the inter-order coherences (`impact="averaged"`), leaving $|\tilde\psi(k)|^2$ binned onto the grid. A plane wave gives `entry_state(ψ)`, and a Gaussian of momentum RMS $p\sigma$ gives `entry_state(0, σ)`.

For a 2D state the $y$ degree of freedom is traced out: $\rho(k_x,k_x')=\int\frac{dk_y}{2\pi}\tilde\psi(k_x,k_y)\tilde\psi^*(k_x',k_y)$. The two vortices $\ell$ and $-\ell$ are related by $y\to-y$ and so give the same planar density; planar channeling carries no OAM. The out-of-band weight is $1-\operatorname{tr}\bar\rho$. A trace above $1+10^{-3}$ is refused, because it signals an under-resolved quadrature.

## Presets

All presets are normalized analytically, and each check below is a regression test.

| Preset | $\tilde\psi$ | Source |
| --- | --- | --- |
| Hermite–Gauss $HG_m$, waist $w$ | $\sqrt{2\pi}(-i)^m\,HG_m(k;\,2/w)$ | Fourier eigenmodes, Siegman §16 |
| Laguerre–Gauss $LG_p^\ell$ | $2\pi(-1)^p(-i)^{\lvert\ell\rvert}\,LG_p^\ell(k;\,2/w)$ | Hankel transform, Siegman §17 |
| Bessel–Gauss $J_\ell(k_r r)e^{-r^2/w^2}e^{i\ell\phi}$ | $\propto e^{-(k-k_r)^2w^2/4}\,\mathrm{ive}(\ell,kk_rw^2/2)\,e^{i\ell\phi_k}$ | Weber's second exponential integral; Gori et al. 1987 |
| Finite-energy Airy $\mathrm{Ai}(x/x_0)e^{ax/x_0}$ | $x_0\exp[(a-ikx_0)^3/3]$, norm $e^{2a^3/3}/\sqrt{8\pi a}$ | Laplace transform of Ai; Siviloglou & Christodoulides 2007 |
| Aperture probe | $\sqrt{4\pi/(k_o^2-k_i^2)}\,\mathbb 1_{k_i\le k\le k_o}\,e^{-i\chi(\mathbf k)}e^{i\ell\phi_k}e^{-if(\mathbf k)}e^{-i\mathbf k\cdot\mathbf r_0}$ | Kirkland §3; polar $C_{nm}$ notation (Krivanek 1999), as in abTEM |

with $\chi=p\sum_{nm}C_{nm}\alpha^{n+1}\cos(m(\phi-\phi_{nm}))/(n+1)$, $\alpha=k/p$, $C_{10}=-\Delta f$, $C_{30}=C_s$. $f$ is an arbitrary user phase plate. `from_array` wraps any uniform sampled array (rectangle-rule transform); `from_abtem` imports abTEM `Waves`, with ensemble axes as an equally weighted mixture.

## Limiting cases

- **Wide Gaussian:** a waist of $10^4$ Å is a plane wave, reproducing `entry_state(ψ)` to $10^{-6}$ in both impact modes.
- **Divergence entry:** an averaged Gaussian of waist $1/(p\sigma)$ reproduces `entry_state(0, σ)`. The $3\times10^{-3}$ residual is the point-weight versus bin-integral difference, $O(\Delta^2/\sigma_k^2)$.
- **Translations:** averaged mode is invariant under any translation; fixed mode only under $T_d$.
- **Airy disk:** an aberration-free circular aperture gives $\psi(r)=\psi(0)\,2J_1(k_or)/(k_or)$.
- **abTEM agreement:** abTEM 1.0.10 `Probe` intensities agree within 3% of peak for $C_{10},C_{30},C_{12}/\phi_{12}$. Reversing the $C_{10}$ sign gives over 6%, which fixes the sign convention.

## Assumptions

- The planar solver drops coherence between Bloch blocks. This is exact for lattice-periodic observables, as derived above; position-resolved observables need the real-space backend of the axial slices.
- Sampled arrays carry no momenta beyond $\pi/\Delta x$.
- The tilt is applied at projection, and refraction is the caller's `incidence_rad` (`cr-plane-incidence`).

## Independent verification (fresh context, 2026-10-15)

The verifier did not write the implementation. It derived the results below from the ledger row and the cited sources first, then read `states.py`, `DensityMatrixSolver.entry_state`/`entry_from_plane`, `MCWFSolver._entry_mixture`/`_entry_from_mixture`/`run`, and the `emission.py` plumbing. Code reviewed: untracked `states.py`, SHA-256 `464450d5…03ce`. A concurrent session added docstring markers during the review; the numerics did not change. Scratch scripts sit outside the repository.

**Verdict: `discrepancy` (numerical).** Every equation in the claim re-derives, and the code implements each one exactly. The discrepancy is in how the bin integral $\int_{\rm bin}d\delta/2\pi$ is evaluated: for two documented classes of input, a narrow-momentum state and a `superpose` of separated terms, the result is wrong without warning. Details are in Findings.

### Independent derivation

**Translation average.** Take a supercell of $n$ periods with periodic length $L=nd$. Its momenta are $k=2\pi m/L=\kappa_r+\nu g$, where $m=r+n\nu$, $\kappa_r=rg/n$ and $g=2\pi/d$. In the solver convention $\langle x|k\rangle\propto e^{ikx}$, the translation $T_d\psi(x)=\psi(x-d)$ gives $T_d|k\rangle=e^{-ikd}|k\rangle=e^{-2\pi i r/n}|k\rangle$. Conjugation therefore multiplies $\rho(k,k')$ by $e^{-2\pi i(r-r')/n}$. Averaging over $j=0,\dots,n-1$ keeps only $r=r'$, with all $(\nu,\nu')$ inside the block.

In the infinite crystal, the average over $j\in\mathbb Z$ keeps $k-k'\in g\mathbb Z$, which gives

$$
\bar\rho_\kappa[\nu,\nu']\,\frac{d\kappa}{2\pi}=\tilde\psi(\kappa+\nu g)\,\tilde\psi^*(\kappa+\nu'g)\,\frac{d\kappa}{2\pi}.
$$

Summing this over each bin gives the row's formula. Its trace is $\sum_{r,\nu}\int_{\rm bin}|\tilde\psi|^2d\delta/2\pi=\int|\tilde\psi|^2dk/2\pi=1$, less the out-of-band weight.

**Generator symmetry.** $H_\perp$ is periodic, so $T_dH_\perp T_d^{-1}=H_\perp$. The dissipator is $J(\rho)(x,x')=C(x,x')\rho(x,x')$, together with the $-\tfrac12C(x,x)$ terms. It commutes with $T_d$ conjugation if and only if $C(x+d,x'+d)=C(x,x')$. In momentum space, that means $\tilde C(q,q')\neq0$ only for $q-q'\in g\mathbb Z$.

Both kernels of `cr-lindblad-jump-operators` satisfy this condition. The TDS kernel has the form $\sum_G n_Ge^{iGx}f(x-x')$ with $e^{iGd}=1$. The electronic kernel has the form $\sqrt{\rho_q(x)\rho_q(x')}\,e^{iq_x(x-x')}$ with $\rho_q$ periodic. The solver's storage as per-residue blocks $C(\kappa_r+h_ig,\kappa_r+h_jg)$ has the same structure.

Hence the class $(k-k')\bmod g$ is conserved:

$$
J(\rho)(k,k')=\int\tilde C(q,q')\,\rho(k-q,k'-q').
$$

Here $(k-q)-(k'-q')\equiv k-k'\pmod g$. Lattice-periodic observables (band populations, $d_c^\dagger d_c$, bound population) have support only on $k-k'\in g\mathbb Z$. For these observables, $\operatorname{tr}(O\,\mathcal E_z[\rho])=\operatorname{tr}(O\,\mathcal E_z[\bar\rho])$ holds exactly. The cutoff projector $P_S$, the reprojection under energy loss, and $P_SH_{\rm eff}P_S$ act within each block, so they preserve the symmetry.

Assigning the continuous $\delta$ to $\kappa_r$ is a separate step. It is the solver's Bloch-grid discretization, the same one the kernel's $p$ binning and `entry_lattice` use, with error $O(g/n_{\rm cells})$. It is not part of the exactness statement.

**Impact average.** A translation by $a$ multiplies $\tilde\psi(k)$ by $e^{-ika}$. Each bin element then picks up $e^{-i(\nu-\nu')ga}$, because the $e^{-i\delta a}$ factors cancel. Averaging over one period gives $\frac1d\int_0^de^{-i(\nu-\nu')ga}da=\delta_{\nu\nu'}$, which is exact.

**Bin assignment.** The bins are centred, $\lvert\delta\rvert<g/2n$, around $k=lg/n$ with $l=r+n\nu$. This is the same residue/order split as `entry_lattice`, and nearest-grid snapping maps a plane wave to the same bin.

**2D reduction.** With $\int d^2k/(2\pi)^2\lvert\tilde\psi\rvert^2=1$, the reduced density is $\rho(k_x,k_x')=\int\frac{dk_y}{2\pi}\tilde\psi(k_x,k_y)\tilde\psi^*(k_x',k_y)$. A $y$-offset $y_0$ cancels from it. For $\tilde\psi_\ell=A(\lvert\mathbf k\rvert)e^{i\ell\phi_k}e^{-i\mathbf k\cdot\mathbf r_0}$, the mirror image is $\tilde\psi_{-\ell}(k_x,k_y)=\tilde\psi_\ell(k_x,-k_y)$, up to the cancelled $y_0$ phase. So $\ell$ and $-\ell$ give the same planar density. The argument needs the rest of the state to be $y$-mirror symmetric. It is not a general OAM statement: an aberration such as $C_{12}$ with $\phi_{12}\notin\{0,\pi/2\}$ breaks the mirror.

**Presets.** These use $\tilde\psi=\int\psi e^{-ikx}$, with Parseval $\int\lvert\tilde\psi\rvert^2dk/2\pi=\int\lvert\psi\rvert^2$.

- **Hermite–Gauss.** The Hermite functions satisfy $\int h_m(u)e^{-iqu}du=\sqrt{2\pi}(-i)^mh_m(q)$. With $u=\sqrt2x/w$, this gives $\tilde\psi=\sqrt{2\pi}(-i)^m\,\mathrm{HG}_m(k;2/w)$, where the waist maps as $e^{-x^2/w^2}\to e^{-k^2w^2/4}=e^{-k^2/(2/w)^2}$.
- **Laguerre–Gauss.** The angular integral is $\int e^{i\ell\phi}e^{-ikr\cos(\phi-\phi_k)}d\phi=2\pi(-i)^{\lvert\ell\rvert}e^{i\ell\phi_k}J_{\lvert\ell\rvert}(kr)$, and $(-i)^{\ell}(-1)^{\ell}=(-i)^{\lvert\ell\rvert}$ for $\ell<0$. The Hankel eigenvalue of $\mathrm{LG}_p^{\lvert\ell\rvert}$ is $(-1)^p$. Together these give $2\pi(-1)^p(-i)^{\lvert\ell\rvert}\mathrm{LG}_p^\ell(k;2/w)$, normalized to $\int\lvert\mathrm{LG}\rvert^2d^2k=1$.
- **Bessel–Gauss.** Weber's second exponential integral is $\int_0^\infty e^{-r^2/w^2}J_\ell(ar)J_\ell(br)\,r\,dr=\tfrac{w^2}{2}e^{-(a^2+b^2)w^2/4}I_\ell(abw^2/2)$. Hence $\tilde\psi=2\pi(-i)^{\lvert\ell\rvert}\tfrac{w^2}{2}e^{-(k-k_r)^2w^2/4}\,\mathrm{ive}(\ell,kk_rw^2/2)\,e^{i\ell\phi_k}$. The norm is $\int\lvert\psi\rvert^2=2\pi\tfrac{w^2}{4}\mathrm{ive}(\ell,k_r^2w^2/4)$, from the same integral with $w^2\to w^2/2$ and $a=b=k_r$.
- **Finite-energy Airy.** Substituting $s=x/x_0$ gives $\tilde\psi=x_0\int\mathrm{Ai}(s)e^{(a-ikx_0)s}ds=x_0e^{(a-ikx_0)^3/3}$. The Laplace transform is valid for $\operatorname{Re}b=a>0$: Ai decays superexponentially as $s\to+\infty$, and $e^{as}$ suppresses the $\lvert s\rvert^{-1/4}$ tail as $s\to-\infty$. With $\operatorname{Re}(a-iq)^3=a^3-3aq^2$, Parseval gives $\int\lvert\psi\rvert^2=x_0e^{2a^3/3}/\sqrt{8\pi a}$. The spec's "norm $e^{2a^3/3}/\sqrt{8\pi a}$" is $\int\mathrm{Ai}^2e^{2as}ds$; the code includes the factor $x_0$, which is correct.
- **Aperture probe.** The condition $N^2\pi(k_o^2-k_i^2)/(2\pi)^2=1$ gives $N=\sqrt{4\pi/(k_o^2-k_i^2)}$. The inverse transform is $\psi(r)=\frac{N}{2\pi}\int_0^{k_o}J_0(kr)k\,dk=\frac{Nk_o^2}{4\pi}\,\frac{2J_1(k_or)}{k_or}$.
- **Aberration function.** The polar form is $\chi=\frac{2\pi}{\lambda}\sum\frac{C_{nm}}{n+1}\alpha^{n+1}\cos m(\phi-\phi_{nm})$, with $\phi=\operatorname{atan2}(k_y,k_x)$, $\alpha=k/p$, $C_{10}=-\Delta f$ and the probe written as $A\,e^{-i\chi}$.

**MCWF entry.** Write $\rho_r=\sum_s\lambda_sv_sv_s^\dagger$. Draw $(r,s)$ with probability $\lambda_s$, and draw "out of band" with probability $1-\operatorname{tr}\rho$. The existing collapse keeps $P_Sv/\lVert P_Sv\rVert$ with probability $\lVert P_Sv\rVert^2$. Otherwise it books the electron as free at $E_n$ with probability $\lvert\langle n\vert v\rangle\rvert^2/(1-\lVert P_Sv\rVert^2)$. The ensemble means are then $E[\lvert\hat\psi\rangle\langle\hat\psi\rvert\mathbb 1_{\rm surv}]=P_S\rho P_S$ and $E[\text{free at }n]=\langle n\vert\rho\vert n\rangle$, which match `entry_from_plane`.

### Implementation diff

| Item | Independent | Code | Result |
| --- | --- | --- | --- |
| Bin integral, $\kappa_r$, centring | $\kappa_r=rg/n$, $\lvert\delta\rvert<g/2n$, $k=\kappa_r+\nu g+\delta$ | `planar_entry_density`: `offsets` centred midpoints, `np.arange(n_cells)*width + orders*g` | match |
| Basis convention | $e^{+ikx}$ | `_waves = exp(+i(κ+hg)x)`, `⟨h\|f\|h'⟩ = f_{h-h'}` | match |
| Tilt | $\tilde\psi(k-p\psi)$ | `flat = k - shift` | match |
| Impact average | diagonal of fixed | `einsum("rii->ri")` | match |
| 2D trace | $\int dk_y/2\pi$ | `wy/(2π)` | match |
| HG, LG, BG, Airy, aperture amplitudes and norms | as derived | `_hg_factor`, `laguerre_gauss`, `bessel_gauss`, `_airy_factor`, `ApertureProbe.amplitude` | match |
| $\chi$ | as derived | `aberration_phase` | match |
| MCWF | eigen-draw plus out-of-band, then `_collapse` | `_entry_mixture`, `_entry_from_mixture` | match (weights renormalized by $\operatorname{tr}+$ outside; differs from the density solver only when $1<\operatorname{tr}\le1.001$) |

### Independent numerical checks

The checks use Si(110) at 17 MeV with `n_cells = 4`, `max_order = 8` (as in the tests) unless stated.

- **Presets (phase-sensitive).** Each amplitude was compared with a brute-force transform of real-space formulas written and normalized independently by `scipy.integrate.quad`. HG $m=0$–$4$ agrees to $1\times10^{-12}$. LG $(\ell,p)=(0,0),(1,0),(-2,1),(3,2),(-1,2)$ agrees to $1.3\times10^{-13}$. BG $\ell=0,1,-2$ agrees to $1.4\times10^{-13}$, with zero global phase. A Gaussian centred at $\mathbf r_0$ and `translate` both agree to $1.1\times10^{-13}$. The Airy norm agrees with quad to 8 digits for $(x_0,a)=(0.5,0.1),(1,0.3),(0.3,0.05)$, and its amplitude to $4\times10^{-9}$. The Laplace transform at $b=a-0.8i$ matches $e^{b^3/3}$ to $10^{-6}$.
- **abTEM 1.0.10, momentum space and phase-sensitive.** The comparison used `Aberrations._evaluate_from_angular_grid` against $e^{-i\chi}$ for each of $C_{10},C_{30},C_{12}/\phi_{12},C_{21}/\phi_{21},C_{23}/\phi_{23},C_{32},C_{34},C_{45},C_{50}$; the largest difference is $5\times10^{-8}$. The FFT of an abTEM `Probe` placed at $(7,12)$ Å, with $C_{10},C_{30},C_{12},C_{23}$, divided by `amplitude` over 725 points inside $0.95k_o$, is constant to $4\times10^{-7}$ in modulus and $1.4\times10^{-5}$ rad in phase. Flipping $\phi_{12}$ gives a phase spread of 2.2 rad. This confirms the position sign, the $\phi$ orientation and $C_{10}=-\Delta f$, and is far tighter than the 3% intensity test.
- **Periodic observables against real space.** This check pins the fixed-impact phase sign. For a Gaussian with $w=0.5$, $x_0=0.37$, $\langle e^{iMgx}\rangle=\sum_{r,\nu}\bar\rho_r[\nu,\nu+M]$ equals $e^{iMgx_0}e^{-M^2g^2w^2/8}$ to $10^{-16}$ for $M=1,2,3$, and $\langle k\,e^{igx}\rangle=(g/2)\langle e^{igx}\rangle$ to $4\times10^{-16}$. For an off-centre 2D $\mathrm{LG}^{2}_{1}$, $\langle e^{iMgx}\rangle$ equals the 2D real-space integral to $10^{-6}$.
- **Impact average.** The mean of `fixed` over 64 uniform impact points equals `averaged` to $8\times10^{-17}$.
- **Bin assignment.** Wide Gaussians ($3\times10^4$ Å) at tilts of $0,0.49,0.51,1.3,-2.7$ bins match `entry_state(ψ)` to $2\times10^{-6}$.
- **Divergence entry.** At $\sigma=1$ mrad the residual against `entry_state(0, σ)` is $9\times10^{-6}$ ($2\times10^{-4}$ at 0.3 mrad), not $3\times10^{-3}$. The bin weights match exact erf bin integrals to $8\times10^{-7}$.
- **2D reduction.** A separable 2D Gaussian with a 5 Å $y$-offset equals the 1D result to $5\times10^{-16}$. Off-centre $\mathrm{LG}_1^{\pm2}$ agree to $8\times10^{-17}$. An aperture with $\ell=\pm1$ agrees to $3\times10^{-17}$ without aberrations, but differs by $2.3\times10^{-3}$ with $C_{12}=30$ Å, $\phi_{12}=0.5$; this is mirror breaking, as expected.
- **MCWF.** The test drew $2\times10^5$ samples from `_entry_from_mixture`, using a mixture of a fixed Gaussian and an Airy state with margin 5 eV. In fixed mode, the bound population is 0.8807 against 0.8797 ($\sigma\approx0.0011$), free 0.0753 against 0.0764 ($1.9\sigma$), $\max\lvert E[\rho]-\rho_S\rvert=6\times10^{-4}$, and per-state free weights agree within $5\times10^{-4}$. In averaged mode the corresponding figures are 0.7186 against 0.7181 and free 0.0604 against 0.0606.
- **Hard edges.** The projected trace of a full-band aperture is 0.99966, 0.99981 and 0.99989 at 0.1, 0.3 and 1 mrad, and the `wavefunction()` Airy-disk error is $5\times10^{-4}$ of the peak.
- **Test file.** `tests/montecarlo/test_channeling_states.py` gives 26 passed, abTEM test included.

### Findings

1. **Narrow-momentum states silently lose weight. This is the first divergent term.** `states.py:701` clips the per-bin midpoint count at `max_subsamples = 20000`. When the momentum width $1/w$ falls below about $(g/n)/20000$, the midpoint rule samples a spike. The trace guard at `states.py:728` is one-sided: it refuses an overcount but books an undercount as out-of-band weight.

   Results for 1D Gaussians at 0.3 mrad:

   | Waist | Result |
   | --- | --- |
   | $3\times10^4$ Å | correct, $2\times10^{-6}$ |
   | $5\times10^4$ Å | refused, or trace 0.993 depending on the tilt's sub-bin offset |
   | $10^5$ Å | refused, or trace 0.743 with outside 0.257 |
   | $10^6$ Å | refused (trace 4.0), or trace 0 with outside 1.0 |

   In the last case the whole beam becomes entry flux and nothing is raised. Real beam waists of 10–100 µm ($10^5$–$10^6$ Å) are in this regime. The threshold scales with `n_cells`. The row's "a $10^4$ Å Gaussian reproduces `entry_state(ψ)`" is true, but nothing tells the user that wider states fail.
2. **`superpose` of spatially separated terms is mis-normalized.** `Translated` passes on the unshifted state's `k_resolution` (`states.py:278`), so the cross-term fringes $e^{-ik(x_1-x_2)}$ are not resolved. The quadrature normalization in `superpose` (`states.py:629`–`634`) then aliases, and so does the fixed-mode bin integral. For two $w=2$ Å Gaussians separated by $D$:

   | $D$ | Norm (exact $\sqrt2=1.4142$) | Projected trace |
   | --- | --- | --- |
   | 20 Å | correct | 1.0000 |
   | 60 Å | — | 1.0077, refused |
   | 100 Å | 1.902 | 0.553, so outside is a spurious 0.447 |
   | 200 Å | 1.690 | 0.700 |

   With a 150 Å separation in $y$, the norm is 1.80, but the planar result is still right because the state is separable. The test `test_presets_are_normalized_in_momentum_space[superposition]` measures the norm with the same quadrature that set it, so it cannot catch this.
3. **Overstated or loose wording (no physics error).**
   - The row's "to 3e-3; the residual is point versus bin weights" (ledger-channeling.md:231; spec line 47; test lines 127–129): the actual residual is $9\times10^{-6}$, so the tolerance is about 300 times looser than needed.
   - "This is exact for every lattice-periodic observable": exact for the $T_d$ average, while the $\delta\to\kappa_r$ assignment is the $O(g/n)$ Bloch-grid discretization. The spec says this; the row does not.
   - "`ℓ = ±1` give identical planar densities" / "planar cannot carry OAM": holds for $y$-mirror-symmetric states; aberrated vortex probes differ by $2\times10^{-3}$.
   - "Every preset is normalized" for the aperture: true analytically; hard-edge quadrature gives a trace $10^{-4}$–$3\times10^{-4}$ low.
   - "Sampled arrays carry no momenta beyond $\pi/\Delta x$": the rectangle-rule amplitude is periodic in $k$ with period $2\pi/\Delta x$ and is not truncated. Only the overcount guard catches images inside the band.
   - `wavefunction()` caps nodes at 4097 (`states.py:106`). At $C_{10}=-2\times10^4$ Å the error is $4.5\times10^{-3}$ of the peak, without warning.
   - Cost: a 2D state's amplitude array has $n_{\rm cells}\times\text{count}\times N\times n_{k_y}$ elements. A $5\times10^3$ Å 2D Gaussian used 3.5 GB RSS on the 4 × 17 test grid, and production grids (16 × 49) are about 11 times larger.
4. **MCWF and `emission.py` plumbing.** Both are correct. `_depth_history` books `entry = free.sum() + outside` as before, and `divergence_rad` is refused alongside `entry_state`.

### Recommendation

Once findings 1 and 2 are fixed, or the input domain is bounded explicitly with a two-sided guard, the claim qualifies for `rederived` without a new derivation. The symmetry reduction, every preset amplitude and phase, the abTEM/Krivanek convention and the MCWF sampling all re-derive and match to round-off. Possible fixes:

- refine the bin quadrature adaptively, or integrate narrow analytic states exactly, and raise whenever the per-bin step exceeds the state's `k_resolution`;
- give `Translated` (and `Superposition`) a resolution of $\min(\text{res},\,\pi/(2\lvert\text{offset}\rvert))$, or base it on the spread of term centres.

Human sign-off remains separate.


## Implementation response to the fresh-context review

Both numerical findings are fixed in `states.py::planar_entry_density`, `Superposition` and `SampledState`. Each fix is pinned by a regression test in `tests/montecarlo/test_channeling_states.py`.

1. **Narrow-momentum states.**
   - The per-bin sub-interval count `M = ⌈4Δ/k_resolution⌉` has no cap, and only sub-intervals inside the state's momentum support are evaluated.
   - Gaussians of waist 3e4, 5e4, 1e5 and 1e6 Å, at in-bin tilts of 0, 0.49, 0.51 and −2.7 bins, reproduce `entry_state(ψ)` to 1e-6 (`test_narrow_momentum_beams_keep_their_weight`).
   - The trace guard is two-sided when the support lies inside the band.
   - A 2D Gaussian of waist 5e3 Å now projects in milliseconds, with 260 MB peak process RSS on a 16-cell, `H = 16` grid.
2. **Separated superpositions.**
   - `Superposition.k_resolution` includes `π/(2·spread)` of the term centres (every state exposes `centre`).
   - Two w = 2 Å Gaussians 100 and 200 Å apart normalize to √2 to 1e-6, and project with out-of-band weight below 1e-3 (`test_separated_superposition_resolves_its_fringes`).

**Further changes made in response:**
- `SampledState` amplitudes are zero beyond `π/Δx` (band-limited).
- Quadratures that would need more than `MAX_NODES` nodes per axis, or a tensor above `MAX_TENSOR_NODES` (2²⁴) in 2D, raise instead of degrading silently or exhausting memory. The tensor bound was added after the re-check, which found that the per-axis limit alone let a C10 = −2e4 Å probe request a 31 GiB grid.
- The divergence-entry tolerance is tightened to 3e-5.
- The ledger row now qualifies the exactness (the `T_d` average is exact; κ binning is the solver's discretization) and limits the helicity statement to y-mirror-symmetric states.

### Re-check (fresh context, 2026-10-16)

This re-check reviewed commit `381de2ce` (`states.py` SHA-256 `7da87ba6…f07c`). It re-ran the verifier's own scripts, which use independent references; no implementation helpers served as references.

**Verdict: `rederived`.** Both numerical findings are resolved. No formula, factor, sign or convention diverges. One robustness gap remains (below); it is not a physics discrepancy.

- **Narrow-momentum beams.**
  - 1D Gaussians of waist $3\times10^4$, $5\times10^4$, $10^5$, $10^6$ and $10^7$ Å, in both impact modes, at tilts of 0.3, 0.3004 and 0.30075 mrad and 0.4999 bin, all give trace $1$ to $10^{-10}$. They match `entry_state(ψ)` to $\le2\times10^{-12}$ wherever the spike lies inside a bin; before the fix, $10^6$ Å gave trace 0 with outside 1.0.
  - Near a bin edge, the remaining difference ($3\times10^{-3}$ at $3\times10^4$ Å, 0.4999 bin) is the physical spill into the neighbouring bin, which the point snap cannot represent.
  - A spike exactly on a bin edge splits $0.5/0.5$.
  - 2D Gaussians of waist $5\times10^3$–$10^5$ Å match to $2\times10^{-12}$.
- **Memory.** A 2D Gaussian of waist $5\times10^3$ Å now peaks at 0.26 GB RSS on the 4 × 17 grid, down from 3.5 GB. On a production grid (`n_cells = 16`, `H = 24`), 2D Gaussians of waist $10^4$ and $10^6$ Å, and an aberrated 0.3 mrad aperture, project in $\le0.6$ s at $\le0.58$ GB.
- **Separated pairs.** For two $w=2$ Å Gaussians at $D=20$, 60, 100, 140, 200, 500 and 2000 Å, `superpose` normalizes to $\sqrt2$ to $10^{-6}$ and the trace is $1.000000$. The bin-diagonal weights match an adaptive-quadrature integral of $\lvert\tilde\psi\rvert^2$ to $4.6\times10^{-6}$ for $D\ge100$ Å, and to $3.6\times10^{-4}$ at $D=20$ Å, the midpoint error on the fringes. The fixed-mode coherence $(r=1;\,\nu=0,1)$ matches quad to $\le3\times10^{-9}$. A $y$-separation of 150 or 1000 Å reproduces the 1D density to $4\times10^{-15}$.
- **Sampled aliasing.** Amplitudes beyond $\pi/\Delta x$ are now exactly zero for $\Delta x=0.12$, 0.25 and 0.3 Å. A near-Nyquist test function ($\cos(0.9\pi x/\Delta x)$ envelope) projects with trace $1\pm10^{-4}$.
- **Regression.** The earlier preset, abTEM, periodic-observable, impact-average, MCWF and divergence scripts reproduce their previous numbers. Bin-assignment agreement improves from $2\times10^{-6}$ to $3\times10^{-13}$. The divergence residual stays at $9\times10^{-6}$ (1 mrad), within the new $3\times10^{-5}$ tolerance; at 0.3 mrad it is $2\times10^{-4}$, which is not tested. `test_channeling_states.py`, `test_channeling_density.py` and `test_channeling_mcwf.py` give 46 passed.

**Remaining gap.** The response states that `MAX_NODES` "covers the large-defocus `wavefunction()` case". It does not for 2D states. The limit is per axis (`MAX_NODES = 65537`), while the 2D quadrature is a tensor grid. For an aperture probe with $C_{10}=-2\times10^4$ Å at 200 keV, `wavefunction()` requests 64459 nodes per axis, which is under the limit. `ApertureProbe.amplitude` then fails with `MemoryError: Unable to allocate 31.0 GiB` (a 64459 × 64459 meshgrid) instead of the intended `ValueError`. At $C_{10}=-2000$ Å, `wavefunction()` is accurate to $4.3\times10^{-4}$. The same per-axis bound applies to the $k_y$ nodes of a 2D `planar_entry_density`. A product bound on the number of nodes, or a cap of about $\sqrt{\texttt{MAX\_NODES}}$ per axis in 2D, would close the gap.
