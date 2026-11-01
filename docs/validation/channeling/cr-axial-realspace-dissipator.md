# `cr-axial-realspace-dissipator` — real-space Lindblad dissipator and MCWF unravelling for the axial tier

Verifier: fresh context. Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/montecarlo/channeling/axial_dissipator.py::{axial_dissipator,AxialDissipator}` and `src/pyrite/montecarlo/channeling/axial_mcwf.py::AxialMCWF` at `797a0159` (issue #508). Parent claim: `cr-lindblad-jump-operators` ([write-up](cr-lindblad-jump-operators.md)). Source: Dalibard, Castin & Mølmer, PRL 68, 580 (1992).

Independence: the derivation below was written from the ledger row, the module and function docstrings, and the parent write-up, before the implementation bodies were read. The numerical checks use my own band-limited phase, my own quadratures and my own target distributions. The planar helpers `_potential` (the Mott–Bethe $\tilde v$) and `_tds_transform` belong to the parent claim and are used as inputs; `_tds_transform` is re-checked here against my own quadrature.

## Claim

On a periodic 2D supercell grid, the planar kernels become pointwise jump operators.

- **TDS.** An atom of species $s$ on a string at $\mathbf R$ with in-plane displacement $\mathbf u\sim\mathcal N(0,a^2I)$ adds the phase $\phi_{\mathbf u}(\mathbf r)=-V_s(\mathbf r-\mathbf R-\mathbf u)/\hbar v$. The jumps are $L_{s,\mathbf R,\mathbf u}=\phi_{\mathbf u}-\langle\phi\rangle$ with rate measure $\lambda_s\,p(\mathbf u)\,d^2u$. On the grid $\phi$ is band-limited, giving $C_{\rm grid}$. The full diagonal is $C(\mathbf r,\mathbf r)=\sum_{\mathbf G}n_{s,\mathbf G}T_s(\lvert\mathbf G\rvert)e^{i\mathbf G\cdot\mathbf r}/(2\pi\hbar v)^2$, and $\max(C-C_{\rm grid},0)$ is a sink.
- **Electronic.** $L_{\mathbf q}=\sqrt{w(q)\rho_q(\mathbf r)}\,e^{i\mathbf q\cdot\mathbf r}$ on the planar quadrature nodes, with the kick added to the carrier tilt.
- **MCWF.** Waiting-time jumps, mechanism chosen by $\int\lvert\psi\rvert^2C$, exact rejection sampling of $\mathbf u$ with a Lipschitz annulus envelope, sink jumps remove the electron.

## Independent derivation

### (1) TDS: covariance, unravelling, full diagonal

Per unit depth a string carries $\lambda_s$ atoms, each with an independent displacement. As in the parent write-up (a), the second-order cumulant of the transmitted phase gives the dissipator kernel

$$
C(\mathbf r,\mathbf r')=\sum_{\rm strings}\lambda_s\,{\rm Cov}_{\mathbf u}\bigl[\phi_{\mathbf u}(\mathbf r),\phi_{\mathbf u}(\mathbf r')\bigr],
$$

while the mean phase is the continuum potential. A Lindblad family $\{L_\alpha\}$ with measure $d\mu(\alpha)$ reproduces $C$ when $\int d\mu\,L_\alpha(\mathbf r)L_\alpha^*(\mathbf r')=C(\mathbf r,\mathbf r')$. With $L_{\mathbf u}=\phi_{\mathbf u}-\mathbb E_{\mathbf u}\phi$ and $d\mu=\lambda_s\,p(\mathbf u)\,d^2u$,

$$
\lambda_s\int d^2u\,p(\mathbf u)\bigl(\phi_{\mathbf u}(\mathbf r)-\langle\phi\rangle(\mathbf r)\bigr)\bigl(\phi_{\mathbf u}(\mathbf r')-\langle\phi\rangle(\mathbf r')\bigr)=\lambda_s\,{\rm Cov}_{\mathbf u}[\phi(\mathbf r),\phi(\mathbf r')].
$$

Subtracting the mean is required: $\langle\phi\rangle(\mathbf r)$ is a function, not a constant, so dropping it would add the non-dissipative rank-one term $\lambda_s\langle\phi\rangle(\mathbf r)\langle\phi\rangle(\mathbf r')$ that the cumulant expansion has already moved into the Hamiltonian.

**Band-limited grid phase.** On a supercell of area $A$ whose images move together, $\phi_{\mathbf u}(\mathbf r)=\sum_{\mathbf k}\tilde\phi_{\mathbf k}e^{i\mathbf k\cdot(\mathbf r-\mathbf R-\mathbf u)}$ with $\tilde\phi_{\mathbf k}=-\tilde v_s(k)/(\hbar vA)$, restricted to the grid band $\lvert k_x\rvert<\pi n_x/L_x$, $\lvert k_y\rvert<\pi n_y/L_y$ (the Nyquist bin is dropped so that $\tilde\phi_{-\mathbf k}=\tilde\phi_{\mathbf k}^*$ and $\phi$ is real). Since $\mathbb E\,e^{-i\mathbf k\cdot\mathbf u}=e^{-a^2k^2/2}$,

$$
\langle\phi\rangle(\mathbf r)=\sum_{\mathbf k}\tilde\phi_{\mathbf k}e^{-a^2k^2/2}e^{i\mathbf k\cdot(\mathbf r-\mathbf R)},\qquad
\mathbb E_{\mathbf u}\phi_{\mathbf u}^2(\mathbf r)=\bigl(\phi_{\mathbf 0}^2*\mathcal N(0,a^2I)\bigr)(\mathbf r-\mathbf R).
$$

$\phi_{\mathbf 0}^2$ is a trigonometric polynomial with twice the band, so it is represented exactly on a $2n_x\times2n_y$ grid, and the Gaussian convolution multiplies each of its harmonics by $e^{-a^2K^2/2}$. Then $C_{\rm grid}(\mathbf r,\mathbf r)=\sum\lambda_s\bigl[\mathbb E\phi^2-\langle\phi\rangle^2\bigr]$ is exact on the grid nodes.

**Full diagonal.** For independent atoms the single-atom variance $f(\mathbf x)={\rm Var}_{\mathbf u}\phi_{\mathbf u}(\mathbf x)$ is summed over string positions with the volume site density $n_s(\mathbf r)=\sum_{\mathbf G}n_{s,\mathbf G}e^{i\mathbf G\cdot\mathbf r}$, where $n_{s,\mathbf G}=A_{\rm cell}^{-1}\sum_{\rm strings}\lambda_s e^{-i\mathbf G\cdot\mathbf R}$. Then $C(\mathbf r,\mathbf r)=\sum_{\mathbf G}n_{s,\mathbf G}\tilde f(\mathbf G)e^{i\mathbf G\cdot\mathbf r}$ with

$$
\tilde f(\mathbf G)=\frac{1}{(\hbar v)^2}\int\frac{d^2q}{(2\pi)^2}\,\tilde v(q)\,\tilde v(\lvert\mathbf G-\mathbf q\rvert)\Bigl[e^{-a^2G^2/2}-e^{-a^2(q^2+\lvert\mathbf G-\mathbf q\rvert^2)/2}\Bigr]=\frac{T_s(\lvert\mathbf G\rvert)}{(2\pi\hbar v)^2}.
$$

The first term is the Gaussian-smeared transform of $V^2$, the second the transform of $(\mathcal N*V)^2$. $T_s$ depends on $\lvert\mathbf G\rvert$ only when the in-plane displacement is isotropic, which is the stated assumption. Shifting $\mathbf q\to\mathbf q+\mathbf G/2$ gives the parent's symmetric form with bracket $e^{-a^2G^2/2}-e^{-a^2(q^2+G^2/4)}$. For large $G$, $T_s\lesssim e^{-a^2G^2/4}$, so the $\mathbf G$ sum can stop at $e^{-a^2G_{\rm cut}^2/4}=\epsilon$.

**Sink.** $C-C_{\rm grid}$ is the variance carried by momenta outside the grid band. It is not pointwise non-negative: ${\rm Var}(\phi_{\rm band}+\phi_{\rm out})={\rm Var}\,\phi_{\rm band}+{\rm Var}\,\phi_{\rm out}+2{\rm Cov}$, the cross covariance can be negative, and correlated supercell images add further terms to $C_{\rm grid}$. Clamping at zero keeps the sink a valid (non-negative) rate. The no-jump loss is then $C_{\rm grid}+\max(C-C_{\rm grid},0)=\max(C,C_{\rm grid})\ge C$, with equality only where the clamp is inactive.

**$y$ average.** Averaging over $y$ keeps $G_y=0$. For the zone plane normal to $x$, $n_{s,(G_x,0)}$ is the planar volume density harmonic, and $T_s(\lvert G_x\rvert)$ is the planar transform, so the average equals the planar `ScatteringKernel.diagonal`.

### (2) Electronic

From the parent, $C_{\rm el}(\mathbf r,\mathbf r')=\int d^2q\,w(q)\sqrt{\rho_q(\mathbf r)\rho_q(\mathbf r')}\,e^{i\mathbf q\cdot(\mathbf r-\mathbf r')}$. The jumps $L_{\mathbf q}=\sqrt{w\,d^2q\,\rho_q}\,e^{i\mathbf q\cdot\mathbf r}$ give $\sum L L^\dagger=C_{\rm el}$. With $w$ depending only on $\lvert\mathbf q\rvert$, $d^2q=q\,dq\,d\theta$ factorizes: a radial node with weight $c_q=\int_{\rm node}2\pi q\,w\,dq$ and a uniform azimuth. The diagonal is

$$
C_{\rm el}(\mathbf r,\mathbf r)=\sum_q c_q\rho_q(\mathbf r)=\sum_{\mathbf G}\rho_{\mathbf G}\,e^{i\mathbf G\cdot\mathbf r}\sum_qc_q\,e^{-G^2/2q^2},
$$

using the 2D Gaussian smoothing at $\sigma=1/q$. Writing $\psi=e^{i\mathbf k_t\cdot\mathbf r}\varphi$ with $\varphi$ periodic, $L_{\mathbf q}\psi=e^{i(\mathbf k_t+\mathbf q)\cdot\mathbf r}\sqrt{\rho_q}\,\varphi$, so the kick is exact for any $\mathbf q$ if it is added to $\mathbf k_t$ and the kinetic phase uses $(\mathbf k_t+\mathbf k)^2$.

### (3) MCWF

With jumps $\{L_\alpha\}$ and $H_{\rm eff}=H-\tfrac{i\hbar v}{2}\sum_\alpha L_\alpha^\dagger L_\alpha$, the norm decays as $\partial_z\lVert\psi\rVert^2=-\langle\psi\rvert\sum L^\dagger L\lvert\psi\rangle=-\int\lvert\psi\rvert^2C(\mathbf r,\mathbf r)$, since every $L_\alpha$ is diagonal in $\mathbf r$. For trace preservation of the kept channels, the absorptive potential must be $W=\tfrac{\hbar v}{2}C(\mathbf r,\mathbf r)$ including the sink. A jump occurs when $\lVert\psi\rVert^2=\xi\sim U(0,1)$. Channel $\alpha$ is chosen with probability $\lVert L_\alpha\hat\psi\rVert^2/\int\lvert\hat\psi\rvert^2C$. Hierarchically:

- mechanism $m\in\{{\rm tds},{\rm sink},{\rm el}\}$ with weight $\int\lvert\hat\psi\rvert^2C_m$;
- TDS source with weight $\lambda_s\int\lvert\hat\psi\rvert^2{\rm Var}_{\mathbf u}\phi_{\rm band}$, which sums to the TDS mechanism weight;
- displacement from $p(\mathbf u)g(\mathbf u)$ with $g(\mathbf u)=\int\lvert\hat\psi\rvert^2L_{\mathbf u}^2\ge0$, whose $p$-integral is the source weight;
- electronic node with weight $c_q\langle\rho_q\rangle$, which sums to the electronic mechanism weight only if the jump density equals the diagonal's density.

The sink channel maps the state out of the channeled zone; recording such an electron as zero makes the ensemble average $\mathbb E\,\lvert\hat\psi\rangle\langle\hat\psi\rvert$ equal the Lindblad $\rho$ with ${\rm tr}\,\rho$ decaying at $\int\rho(\mathbf r,\mathbf r)\,C_{\rm sink}$.

**Exact rejection.** Expand $L_{\mathbf u}(\mathbf r_j)=\sum_{\mathbf k}S_{\mathbf k}e^{i\mathbf k\cdot(\mathbf r_j-\mathbf R)}(e^{-i\mathbf k\cdot\mathbf u}-m_{\mathbf k})$, $m_{\mathbf k}=e^{-a^2k^2/2}$. Then

$$
g(\mathbf u)=\sum_{\mathbf K}e^{-i\mathbf K\cdot\mathbf u}e^{-i\mathbf K\cdot\mathbf R}(S*S)_{\mathbf K}\,\widehat D(\mathbf K)
-2\sum_{\mathbf k}e^{-i\mathbf k\cdot\mathbf u}e^{-i\mathbf k\cdot\mathbf R}S_{\mathbf k}\,\widehat{DM}(\mathbf k)+\sum_jD_jM_j^2\,\Delta A,
$$

with $D_j=\lvert\psi_j\rvert^2$, $M=\langle\phi\rangle$, $\widehat D(\mathbf K)=\Delta A\sum_jD_je^{i\mathbf K\cdot\mathbf r_j}$ (periodic in $\mathbf K$ over the coarse grid). This is a finite trigonometric polynomial $g=\mathrm{Re}\sum_{\mathbf K}w_{\mathbf K}e^{-i\mathbf K\cdot\mathbf u}$, so $\lvert\nabla g\rvert\le\sum\lvert w_{\mathbf K}\rvert\lvert\mathbf K\rvert$. Every point is within $\delta/\sqrt2$ of a node of a square lattice of spacing $\delta$, so the maximum over lattice nodes within $\delta/\sqrt2$ of an annulus, plus $\delta/\sqrt2\cdot\sum\lvert w\rvert\lvert K\rvert$, bounds $g$ on that annulus. With annulus envelopes $M_i$, the exact scheme proposes annulus $i$ with probability $\propto P_i M_i$ ($P_i$ the Rayleigh mass), $\mathbf u\sim p$ restricted to the annulus, and accepts with $g/M_i$. The accepted density is $\propto P_iM_i\cdot\frac{p}{P_i}\cdot\frac{g}{M_i}=p\,g$. A per-annulus envelope used with an unweighted Gaussian proposal would be biased; the weighting by $M_i$ is required.

### (4) Units and limits

| check | result |
| --- | --- |
| $\tilde\phi=-\tilde v/(\hbar vA)$ | $({\rm eV\,\AA^3})/({\rm eV\,\AA}\cdot{\rm \AA^2})$, dimensionless ✓ |
| $C_{\rm grid}=\lambda\,{\rm Var}\,\phi$ | ${\rm \AA^{-1}}$ ✓ |
| $n_{\mathbf G}T/(2\pi\hbar v)^2$ | ${\rm \AA^{-3}}\cdot{\rm eV^2\AA^6}/({\rm eV\,\AA})^2={\rm \AA^{-1}}$ ✓ |
| $c_q\rho_q$ | ${\rm \AA^2}\cdot{\rm \AA^{-3}}={\rm \AA^{-1}}$ ✓ |
| $a\to0$ | $L_{\mathbf u}\to0$, $T\to0$ ✓ |
| velocity | $C\propto(\hbar v)^{-2}$ ✓ |
| sign | $\phi$ sign irrelevant for a covariance; $C\ge0$ so $W\ge0$ ✓ |
| uniform $\lvert\psi\rvert^2$ | electronic selection weight $=\bar\rho\sum c_q$ ✓ |

## Implementation diff

| item | code | verdict |
| --- | --- | --- |
| phase spectrum | `spectra = where(band, -v/hbar_v/grid.area_ang2, 0)`, band $\lvert m\rvert<n/2$ (axial_dissipator.py:317–325) | matches (1); Nyquist dropped |
| $L_{\mathbf u}$ | `base * (moved - mean)`, `ifft2 * size` (axial_dissipator.py:141–161) | matches $\phi_{\mathbf u}-\langle\phi\rangle$ |
| $C_{\rm grid}$ | $\phi_0^2$ on the 2× grid, damp $e^{-a^2K^2/2}$, shift, subsample `[::2, ::2]`, minus `mean**2`, times $\lambda$ (axial_dissipator.py:358–377) | matches; no aliasing since $2(n/2-1)<n$ |
| full $C$ | `lam / area * exp(-iG.R) * T / (2π ħv)^2`, `_fold` point synthesis (axial_dissipator.py:327–348, 275–281, 380) | matches; cut `g_cut = sqrt(4 ln(1/1e-12)/a²)` matches the $e^{-a^2G^2/4}$ tail |
| sink | `maximum(C - grid_rate, 0)` (axial_dissipator.py:380) | matches the claim; see finding 1 |
| electronic diagonal | `q_weights @ exp(-G²/2q²)` times $\rho_{\mathbf G}$, folded (axial_dissipator.py:394–409) | matches (2) |
| electronic selection | `electronic_expectations`: $\sum_{\mathbf G}\rho_{\mathbf G}e^{-G^2/2q^2}\int\lvert\psi\rvert^2e^{i\mathbf G\cdot\mathbf r}$ over in-band $\mathbf G$ (axial_dissipator.py:244–249) | matches $\langle\rho_q\rangle$ |
| electronic jump | `sqrt(smoothed_density(q)) * phi`, tilt `+ q(cos θ, sin θ)`, new kinetic phase (axial_mcwf.py:170–177, 254–256) | matches; see finding 2 for the amplitude |
| $g(\mathbf u)$ coefficients | self-convolution on the 2× grid times periodic extension of `conj(fft2(density))`, minus `2 * cross`, plus constant (axial_dissipator.py:163–202) | matches the three-term expansion in (3) |
| envelope | lattice $\delta=$ `edges[-1]/64`, nodes within `reach = δ/√2` of the annulus, plus `slack = Σ|w||K| δ/√2` (axial_dissipator.py:212–235) | matches; nodes up to $65\delta$ cover the outer reach |
| rejection | shells $\propto$ `shell_p * envelope`, radius by inverse Rayleigh CDF in the shell, accept `u·M_i < g` (axial_mcwf.py:147–167) | matches the exact scheme; $\lvert\mathbf u\rvert>8\sigma$ (prior mass $e^{-32}$) is not drawn, as documented |
| waiting time | jump when `norms <= thresholds` at step end, state normalized at step end, new threshold after jump (axial_mcwf.py:234–259) | matches; jump depth resolved to one step, as documented |
| absorptive potential | `U - i * absorptive_eV`, `absorptive_eV = ħv/2 * (grid + sink + el)` (axial_mcwf.py:119; axial_dissipator.py:130–137) | matches (3) |
| sink | `live = False`, `lagged = 0`, survivors summed and divided by `n_electrons` (axial_mcwf.py:246–250, 260–271) | matches the Lindblad trace loss |

## Independent numerical checks

Si ⟨001⟩ (one string per $1.920\times1.920$ Å cell, $\lambda=0.18413$ Å⁻¹, $a^2=0.0061$ Å² from the profile). Scripts: session scratchpad `validate-dissipator/check1.py`–`check6.py`.

**1. Planar transform.** My own polar quadrature of $T(\lvert\mathbf G\rvert)$ about $\mathbf q=0$ (not the symmetric form), $a^2=0.0076$ Å²:

| $G$ [Å⁻¹] | mine | `_tds_transform` |
| --- | --- | --- |
| 0 | $2.225504\times10^{5}$ | $2.225506\times10^{5}$ |
| 1.6 | $2.136787\times10^{5}$ | $2.136788\times10^{5}$ |
| 6.5 | $1.381266\times10^{5}$ | $1.381266\times10^{5}$ |
| 20 | $7.018056\times10^{3}$ | $7.018091\times10^{3}$ |

**2. In-grid variance.** My own band-limited $\phi$ (explicit harmonics $\lvert m\rvert<n/2$, $\tilde\phi=-\tilde v/\hbar vA$) and a 40×40 Gauss–Hermite average over $\mathbf u$ reproduce `tds_grid_rate` to $1.4\times10^{-14}$ (1×1, 16²), $2.1\times10^{-14}$ (2×2, 32²) and $4.1\times10^{-14}$ (3×3, 64²), 10 MeV.

**3. Full diagonal and $y$ average.** My own synthesis $\sum n_{\mathbf G}T/(2\pi\hbar v)^2$ with $\lvert\mathbf G\rvert<60$ Å⁻¹ equals `tds_grid_rate + tds_sink_rate` wherever the sink is unclamped, to $3.2\times10^{-4}$ of the maximum (my truncation). The $y$ average of the full TDS diagonal (1×1, 64², 10 MeV) equals planar Si(110) `tds_kernel(...).diagonal` to $1.8\times10^{-11}$ of its maximum; mean rate $6.389\times10^{-5}$ Å⁻¹.

**4. Electronic consistency.** For normalized densities (Gaussian, waist 0.3 Å, on a string and mid-channel; and uniform), $\sum_qc_q\langle\rho_q\rangle$ (the node selection weight) divided by $\int\lvert\psi\rvert^2C_{\rm el}$ (the loss) is $1-2\times10^{-16}$ (1×1, 32²), $1-1.3\times10^{-8}$ (3×3, 64²) and $1-3.1\times10^{-4}$ (1×1, 12², where in-band $\rho_{\mathbf G}$ stop at $\lvert m\rvert\le5$). The applied jump $\sum_qc_q\int\lvert\psi\rvert^2\rho_q^{\rm clip}$ agrees to the same digits after the factor of finding 2, so clipping at zero is negligible here.

**5. Rejection sampler.** 2×2 cells, 24², 1 MeV, source 0, density a Gaussian of waist 0.25 Å at $(0.15,0.05)$ Å. The envelope bounds $g$ at $2\times10^{5}$ random $\mathbf u$ with max $g/M_i=0.921$. 20 000 draws through `AxialMCWF._tds_jump`, against target moments of $p\,g$ from an 80×80 Gauss–Hermite quadrature of $g$ (prior $\mathbb E\lvert\mathbf u\rvert^2=0.01220$ Å², target $0.01934$ Å²):

| moment | sample | target | $z$ |
| --- | --- | --- | --- |
| $u_x$ | $2.293\times10^{-2}$ | $2.412\times10^{-2}$ | $-1.67$ |
| $u_y$ | $8.640\times10^{-3}$ | $8.575\times10^{-3}$ | $+0.10$ |
| $u_x^2$ | $1.0671\times10^{-2}$ | $1.0746\times10^{-2}$ | $-0.91$ |
| $\lvert\mathbf u\rvert^2$ | $1.9331\times10^{-2}$ | $1.9338\times10^{-2}$ | $-0.06$ |
| $u_xu_y$ | $6.249\times10^{-4}$ | $6.725\times10^{-4}$ | $-0.79$ |
| $\lvert\mathbf u\rvert^4$ | $6.018\times10^{-4}$ | $6.041\times10^{-4}$ | $-0.32$ |

The sampler follows $p\,g$, not the prior $p$. Cost: about 25 ms per TDS jump on this 24² grid.

**6. Anchor tests.** `pyrite-dev test --slow tests/montecarlo/quantum/test_channeling_axial_dissipator.py`: 9 passed (31 s), including the MCWF-against-Strang-Lindblad comparison (TDS only, 12²).

## Findings

1. **Clamp excess (model limit, not a formula error).** The clamp is active at many grid points: 15% (1×1, 16²), 67% (2×2, 32²), 84% (3×3, 64²). There the no-jump loss is $C_{\rm grid}>C$, so the total TDS loss exceeds the physical diagonal. Integrated over a uniform density the excess is $4.3\%$ (1×1, 12², the MCWF anchor grid), $0$ (1×1, 32²), $0.55\%$ (3×3, 64²), $0.17\%$ (3×3, 96²) and $9\times10^{-5}$ (2×2, 128²), the same at 1 and 10 MeV. Pointwise it reaches $1.0\times$ the local $C$ (12²) and $0.40\times$ (3×3, 64²), at low-$C$ points between strings. The unravelling remains consistent (the excess is carried by real in-grid jumps), but "$C_{\rm grid}$ + clamped sink $=C$" holds only where the clamp is inactive; the ledger statement should say the loss is $\max(C,C_{\rm grid})$.
2. **`smoothed_density` amplitude (documentation/units).** `AxialDissipator.electron_density` stores $\rho_{\mathbf G}/(n_xn_y)$ and `smoothed_density` returns $\rho_q/(n_xn_y)$, not $\rho_q$ in Å⁻³ as both docstrings state (ratio exactly $1/1024$ on 32², $1/4096$ on 64², $1/144$ on 12²; mean $\bar\rho=0.6992$ Å⁻³). Reproduction: `axial_dissipator(axial_profile("silicon",(0,0,1)), 10e6, (1,1), (32,32), mechanisms=("electronic",)).smoothed_density(5.0).mean() / 0.6992` gives $9.766\times10^{-4}$. The MCWF renormalizes the state after the jump and `electronic_expectations` multiplies the stored array back by $n_xn_y$, so no physical output changes. Any external caller using `smoothed_density` as a density would be off by $n_xn_y$.
3. **Electronic MCWF has no ensemble anchor.** The Lindblad comparison test is TDS-only. The electronic unravelling is verified here per jump (check 4) and by the parent's kernel; the kick changes the carrier tilt, so it cannot be compared with a fixed-grid density matrix directly.

## Verdict

`rederived`. Every formula in the claim re-derives: the TDS unravelling and its in-grid variance (to $4\times10^{-14}$), the full diagonal and its $y$ average (to $2\times10^{-11}$ against the planar diagonal), the electronic jumps and their selection weights (to $2\times10^{-16}$ on a resolved grid), and the exact rejection scheme (moments within $1.7\sigma$). Findings 1 and 2 are wording/documentation corrections; neither changes a physical output.

## Electronic ensemble regression follow-up

The implementation follow-up adds
`test_electronic_mcwf_matches_homogeneous_lindblad` in
`tests/montecarlo/quantum/test_channeling_axial_dissipator.py`. This is an analytic
regression oracle, not an extension of the fresh-context verdict above.

Keep the silicon electronic transfer nodes and weights, but set the electron
density to its spatial mean $n_e$ and the coherent potential to zero. The jump
operator is then $L_{q,\alpha}=\sqrt{n_e w_q}\exp(i\mathbf q_\alpha\cdot\mathbf r)$,
with uniform azimuth $\alpha$. The homogeneous Lindblad equation in momentum
space is the compound-Poisson gain/loss equation. Its characteristic function
for an initially zero-momentum plane wave is

$$
C(b,z)=\left\langle e^{ibk_x}\right\rangle
=\exp\left[z n_e\sum_qw_q\{J_0(qb)-1\}\right].
$$

The free kinetic term leaves momentum probabilities unchanged. The test compares
both Cartesian cosine moments and the vanishing sine moment at mean jump counts
one and three, for 1024 electrons with seed 508. Sampling errors use the exact
second moments: $\langle\cos^2(bk_x)\rangle=(1+C(2b,z))/2$ and
$\langle\sin^2(bk_x)\rangle=(1-C(2b,z))/2$. Four-standard-error bounds include
an absolute 0.015 allowance for the single-jump time discretization
($\Gamma\,dz=0.01$). The total jump count and unit surviving trace are also
checked. The momentum observable uses the explicit carrier, so kicks beyond the
spatial-grid Nyquist are retained.

This supplies an electronic **ensemble** anchor in the homogeneous limit,
superseding the blanket absence noted in Finding 3. It does not test the
inhomogeneous density's localization during propagation; that part still has
per-jump checks only. The ledger status remains `rederived`.
