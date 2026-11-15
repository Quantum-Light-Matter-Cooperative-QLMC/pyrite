# `cr-axial-exit-angular` re-derivation

Claim: `montecarlo/channeling/axial_exit_map.py::{coherent_axial_exit_map,axial_incidence_nodes,AxialExitMap,_deposit}`, plus the batched tilt-stack follow-up (#1216) of `cr-axial-realspace-propagation` in `montecarlo/channeling/realspace.py::SplitStepPropagator`. Sources: Bloch/Floquet theorem for a lattice-periodic Hamiltonian; Shul'ga & Shulga, PLB 769, 141 (2017), Eqs. 13–14 (exit distribution equals the transverse-momentum distribution; used here in its 2D form, not re-opened, since the planar write-up [`cr-planar-exit-angular`](cr-planar-exit-angular.md) already derives the 1D statement from NIMB 402 Eqs. 6–9); Strang splitting as rederived in [`cr-axial-realspace-propagation`](cr-axial-realspace-propagation.md). The potentials $U$ and $W$ are inputs owned by `cr-axial-2d-eigenproblem` and `cr-axial-realspace-dissipator` and are not re-checked here.

Fresh-context verification, 2026-11-15. Order followed: ledger row and module docstring, then the derivation below, then the implementation body.

## Independent derivation (before reading the body)

**Transverse equation.** The paraxial axial equation is

$$
i\hbar v\,\partial_z\psi=H\psi,
\qquad
H=-\frac{\hbar^2}{2\gamma m}\nabla_\perp^2+U(\mathbf r)-iW(\mathbf r),
$$

with $U$ and $W\ge 0$ periodic on the rectangular projected cell $L_x\times L_y$ (reciprocal lattice $\mathbf G=(2\pi m/L_x,\,2\pi n/L_y)$). Let $T_{\mathbf R}$ translate by a cell vector $\mathbf R$. Then $[H,T_{\mathbf R}]=0$, so $e^{-iHz/\hbar v}$ commutes with $T_{\mathbf R}$ and preserves its eigenvalues.

**Bloch reduction.** A plane-wave entry $\psi(\mathbf r,0)=A^{-1/2}e^{i\mathbf k_t\cdot\mathbf r}$, $\mathbf k_t=K\boldsymbol\theta$ with $K=p/\hbar$, satisfies $T_{\mathbf R}\psi=e^{i\mathbf k_t\cdot\mathbf R}\psi$. This eigenvalue is conserved, so $\psi(\mathbf r,z)=e^{i\mathbf k_t\cdot\mathbf r}\varphi(\mathbf r,z)$ with $\varphi$ periodic on **one** cell, and

$$
\varphi(\mathbf r,z)=\sum_{\mathbf G}c_{\mathbf G}(z)\,e^{i\mathbf G\cdot\mathbf r},
\qquad
\psi=\sum_{\mathbf G}c_{\mathbf G}(z)\,e^{i(\mathbf k_t+\mathbf G)\cdot\mathbf r}.
$$

The envelope obeys the same equation with $-i\nabla\to -i\nabla+\mathbf k_t$ in the kinetic term, i.e. the kinetic eigenvalue of order $\mathbf G$ is $\hbar^2\lvert\mathbf k_t+\mathbf G\rvert^2/2\gamma m$.

**Exit comb.** Beyond the exit face $U=W=0$; each component $e^{i(\mathbf k_t+\mathbf G)\cdot\mathbf r}$ is a free plane wave whose transverse momentum is conserved, so in the small-angle limit it travels at

$$
\boxed{\boldsymbol\theta_{\mathbf G}=\frac{\mathbf k_t+\mathbf G}{K}=\boldsymbol\theta+\frac{\mathbf G}{K}}
$$

with probability per incident electron $P_{\mathbf G}=A\lvert c_{\mathbf G}\rvert^2$ (normalization below). The ledger's "$\mathbf G/p$" means $\mathbf G/(p/\hbar)$; the code's `momentum_inv_ang` is $p/\hbar$ in Å⁻¹, so the units close.

**Survival.** Per cell, $\int_{\rm cell}\lvert\psi\rvert^2d^2r=A\sum_{\mathbf G}\lvert c_{\mathbf G}\rvert^2$ (Parseval), and

$$
\frac{d}{dz}\int_{\rm cell}\lvert\psi\rvert^2=-\frac{2}{\hbar v}\int_{\rm cell}W\lvert\psi\rvert^2\le 0 .
$$

So the coherent survival is $S(z)=\sum_{\mathbf G}P_{\mathbf G}(z)$, equal to 1 for $W=0$ and monotonically non-increasing otherwise. The flux removed by $W$ has no angle in this model and cannot appear in the map.

**Supercell.** On a $c_x\times c_y$ supercell the grid reciprocal lattice is $(2\pi m'/c_xL_x,\,2\pi n'/c_yL_y)$. Since $\varphi$ is one-cell periodic, its supercell Fourier coefficient vanishes unless $c_x\mid m'$ and $c_y\mid n'$; populated coefficients coincide with the one-cell $c_{\mathbf G}$ (both are cell averages of $\varphi e^{-i\mathbf G\cdot\mathbf r}$). With the supercell plane wave normalized to unit norm over the supercell area, $P_{\mathbf G}$ is unchanged. The supercell's finer angular pitch $2\pi/(c_xL_xK)$ therefore adds only empty orders, and the map pixel is free.

**Discrete normalization (Parseval).** Sample $\varphi_j$ on $N=n_xn_y$ grid points of the cell. The rectangle rule gives

$$
c_{\mathbf G}=\frac1A\int_{\rm cell}\varphi\,e^{-i\mathbf G\cdot\mathbf r}d^2r\simeq\frac1N\sum_j\varphi_je^{-i\mathbf G\cdot\mathbf r_j}=\frac{\mathrm{FFT}(\varphi)_{\mathbf G}}{N},
\qquad
\boxed{P_{\mathbf G}=\frac{A}{N^2}\lvert\mathrm{FFT}(\varphi)_{\mathbf G}\rvert^2}.
$$

Unit norm per cell means $(A/N)\sum_j\lvert\varphi_j\rvert^2=1$, i.e. $\varphi_j=A^{-1/2}$ for the entry plane wave; then $\mathrm{FFT}_0=N/\sqrt A$ and $P_0=1$. Discrete Parseval, $\sum_{\mathbf G}\lvert\mathrm{FFT}_{\mathbf G}\rvert^2=N\sum_j\lvert\varphi_j\rvert^2$, gives $\sum_{\mathbf G}P_{\mathbf G}=(A/N)\sum_j\lvert\varphi_j\rvert^2$, the grid norm. The FFT index order is that of `fftfreq`, so the angle of each bin is $\theta+2\pi\,\mathrm{fftfreq}/K$; the Nyquist bin's sign is a grid convention (truncation), not physics.

**Divergence.** Independent electrons with a Gaussian spread form an incoherent mixture of plane waves, so

$$
\frac{dP}{d^2\theta}=\int d^2\theta_0\,\frac{e^{-\lvert\boldsymbol\theta_0-\boldsymbol\theta_c\rvert^2/2\sigma^2}}{2\pi\sigma^2}\sum_{\mathbf G}P_{\mathbf G}(\boldsymbol\theta_0)\,\delta^2\bigl(\boldsymbol\theta-\boldsymbol\theta_0-\mathbf G/K\bigr).
$$

The stated quadrature is midpoint nodes $x_i=-4\sigma+(i+\tfrac12)h$, $h=8\sigma/n$, per axis (identical 1D nodes to the planar `incidence_nodes` with span $4\sigma$), restricted to $x^2+y^2\le(4\sigma)^2$, with weights $w\propto e^{-(x^2+y^2)/2\sigma^2}$ normalized to 1. Symmetry about the centre gives zero mean offset. The kept node count is $\approx\pi(n/2)^2$. The truncated-disc 2D Gaussian has, with $a=R^2/2\sigma^2=8$,

$$
\langle x^2\rangle=\sigma^2\,\frac{1-(1+a)e^{-a}}{1-e^{-a}}=0.997315\,\sigma^2,
$$

so a per-axis variance deficit of $2.7\times10^{-3}$ is the truncation bias, not a midpoint-rule error. The planar convention truncates a 1D Gaussian at $\pm4\sigma$ (deficit $\approx1.1\times10^{-3}$); the disc is the stated 2D analogue, not a tensor product of planar nodes. For $\sigma=0$ the mixture is the single node $\boldsymbol\theta_c$ with weight 1.

**Bilinear deposition.** For a point at fractional pixel index $u=(\theta-\theta_{\min})/\Delta$, $i=\lfloor u\rfloor$, $f=u-i$, deposit $(1-f)$ on $i$ and $f$ on $i+1$ per axis (product in 2D). Then $\sum=1$ and $\sum(\text{index}\times\text{weight})=(1-f)i+f(i+1)=u$: weight and first moment are preserved exactly when all four neighbours are in the window. A neighbour outside the window loses its share, so the captured total is $\le S$, and the first moment is not preserved for orders straddling the edge.

**Batched Strang step.** For a stack of tilts $\{\mathbf k_{t,b}\}$ the Hamiltonians differ only in the diagonal kinetic factor. With $\Phi_b$ in momentum space,

$$
\Phi_b\leftarrow K_b\,\mathcal F\,V\,\mathcal F^{-1}K_b\,\Phi_b,
\qquad
K_b=\exp\!\Bigl(-\frac{i\hbar\lvert\mathbf k+\mathbf k_{t,b}\rvert^2dz}{4\gamma mv}\Bigr),
\qquad
V=\exp\!\Bigl(-\frac{i(U-iW)dz}{\hbar v}\Bigr),
$$

is $n$ independent scalar Strang steps provided $K_b$ multiplies only state $b$ and $V$ is shared (it does not depend on the tilt). The half-step exponent follows from $\frac{\hbar^2\lvert\mathbf k+\mathbf k_t\rvert^2}{2\gamma m}\cdot\frac{dz/2}{\hbar v}$. $\lvert V\rvert=e^{-W\,dz/\hbar v}$ gives the norm decay $e^{-2W\,dz/\hbar v}$; $W=0$ makes every factor unimodular, hence unitary.

## Cheap filters

| Filter | Result |
|---|---|
| Units | $\mathbf k_t=K\theta$ and $\mathbf G$ in Å⁻¹, $K$ in Å⁻¹, angle dimensionless; $P_{\mathbf G}=A\lvert\mathrm{FFT}\rvert^2/N^2$ with $\varphi$ in Å⁻¹ is dimensionless; kinetic phase $\hbar^2k^2/2\gamma m\cdot dz/\hbar v$ dimensionless. Pass. |
| $W=0$ | Unitary, survival 1, every order captured in a wide window. Pass (numeric below). |
| $\sigma=0$ | Single node, weight 1. Pass. |
| $z=0$ | $P_{\mathbf G}=\delta_{\mathbf G0}$ at $\boldsymbol\theta$. Pass by construction. |
| Sign of $W$ | $V$ has modulus $e^{-W dz/\hbar v}<1$ for $W>0$: loss, not gain. Pass. |
| Supercell | Off-cell-lattice orders vanish by translation symmetry. Pass (numeric below). |

## Diff against the implementation

Read after the derivation above.

| Derived | Code | Agreement |
|---|---|---|
| $\varphi_j=A^{-1/2}$ entry | `plane_wave = 1/sqrt(grid.area_ang2)`, grid = `profile.period_ang` (one rectangular cell) | exact |
| $P_{\mathbf G}=A\lvert\mathrm{FFT}\rvert^2/N^2$ | `scale = grid.area_ang2 / count**2`; `w * scale * abs(fft2(phi))**2` | exact |
| $\boldsymbol\theta_{\mathbf G}=\boldsymbol\theta+\mathbf G/K$ | `chunk + grid.kx/p`, `grid.kx = 2π fftfreq(n, L/n)`, `p = momentum_inv_ang` | exact, FFT order consistent |
| $S=\sum_{\mathbf G}P_{\mathbf G}$ before windowing | `surviving[i] += power.sum()` | exact |
| Disc-truncated midpoint nodes, normalized Gaussian weights | `axial_incidence_nodes`: span `DIVERGENCE_SPAN = 4.0`, `keep = r2 <= half**2` | exact |
| Bilinear deposit, per-neighbour drop | `_deposit`: `floor`, `(1-t, t)` products, `inside` mask per neighbour, `bincount` | exact |
| Pixel centres $k\Delta$, $\lvert k\Delta\rvert\le$ half range | `n_half = floor(half/pixel + 1e-9)`, origin `centres[0]` | exact |
| Batched $K_b$, shared $V$ | `kinetic[b] = ħ²/2mγ · ((kx+carrier_b,x)² + (ky+carrier_b,y)²)`; half-step `exp(-0.5j·step·kinetic/ħv)`; `V = exp(-1j·step·(u-1j·w)/ħv)` shared | exact |
| Single-tilt path unchanged | `kinetic[0]` when `tilts.ndim == 1`; 3D state with one tilt shares its phase | exact |

The `"cuda"` path reuses the same expressions with `xp = cupy`; the factors are built in NumPy then copied to the device. CUDA parity was not exercised (no device locally; the anchor test skipped).

## Independent numeric checks

Script `scratch/val_axial_exit.py` (ignored, not committed): own constants ($\hbar c=1973.2698$ eV Å, $mc^2=510998.95$ eV), own grid Hamiltonian $H=\mathcal F^{-1}\mathrm{diag}(T)\mathcal F+\mathrm{diag}(U-iW)$ in the periodic-envelope representation propagated with `scipy.linalg.expm` (no splitting), own bilinear binning, own 2×3 supercell split-step. The library supplies only the inputs $U$ (`profile.on_grid`) and $W$ (`axial_dissipator`, TDS only) for Si⟨111⟩ at 1 MeV, one-cell grid 16×12 ($L_x\times L_y=3.8402\times2.2172$ Å, $K=720.616$ Å⁻¹, mean $W=0.0743$ eV).

**Exact expm vs library map, tilt $(1.7,-0.6)$ mrad, pixel 0.2 mrad, window ±30 mrad.** Exact survival 0.9071566355 (300 Å), 0.8162266012 (800 Å).

| `step_ang` | $\Delta S$ at 300 Å | $\Delta S$ at 800 Å | max map diff at 800 Å (peak 0.252) |
|---|---|---|---|
| 5.0 | $+2.28\times10^{-4}$ | $+3.99\times10^{-4}$ | $1.59\times10^{-3}$ |
| 2.5 | $+5.67\times10^{-5}$ | $+9.94\times10^{-5}$ | $3.95\times10^{-4}$ |
| 1.25 | $+1.42\times10^{-5}$ | $+2.48\times10^{-5}$ | $9.87\times10^{-5}$ |

Ratios 4.02 / 4.01 per halving: the only difference from the exact solution is the documented $O(dz^2)$ Strang error. Normalization, angle assignment and binning carry no residual at the $10^{-14}$ level beyond it.

**$W=0$.** Exact survival $1-5\times10^{-15}$; library survival and captured weight 1.0 at both depths.

**Own 2×3 supercell split-step** ($dz=1$ Å, $\varphi=1/\sqrt{A_{\rm super}}$). Off-cell-lattice power / total: $2.8\times10^{-29}$ (300 Å), $1.2\times10^{-28}$ (800 Å). Survival 0.90716570 / 0.81624249, equal to all printed digits to the library one-cell map at `step_ang=1.0` (0.9071657, 0.81624249); each populated supercell order equals the one-cell exact order to $\le8\times10^{-5}$, the $dz=1$ splitting error.

**Quadrature** ($\sigma=0.2$ mrad, $n=30$). 716 nodes ($\pi\cdot15^2=706.9$); weights sum to 1 to $10^{-16}$; mean offset $\sim10^{-20}$ rad; per-axis variance $0.997560\,\sigma^2$, converging with $n$ to the analytic truncated-disc value $0.997315\,\sigma^2$ ($n=60$: 0.997322; $n=120$: 0.997307). The per-axis node set equals planar `incidence_nodes(0, σ, 30)`.

**Deposit.** 400 random points spanning beyond a 21-pixel window: library `_deposit` equals my own loop to $1.6\times10^{-14}$ (edge straddlers included).

**Divergent beam** (centre $(1,0)$ mrad, $\sigma=0.3$ mrad, $n=4$ → 12 nodes, batch 3, `step_ang=1.25`) against my own weighted sum of exact per-tilt expm maps: survival 0.89154099 / 0.78966702 vs 0.89152453 / 0.78964590 (splitting error), max map difference $1.3\times10^{-5}$ on peak $3.2\times10^{-2}$.

**Anchor tests.** `pyrite-dev test tests/montecarlo/quantum/test_channeling_axial_exit_map.py`: 8 passed, 1 skipped (`test_exit_map_cuda_matches_cpu`, no CUDA device).

## Findings (none a discrepancy in the code)

1. **Notation.** The ledger row writes $\mathbf k_t=p\theta$ and $\theta_{\mathbf G}=\theta+\mathbf G/p$; with $p$ a momentum this needs $\hbar$. The code and module docstring use $p/\hbar$ in Å⁻¹ consistently. Suggest "$k_t=p\theta/\hbar$" or "$p$ in Å⁻¹ ($p/\hbar$)".
2. **Variance tolerance.** The `3e-3` tolerance on the per-axis variance absorbs the $2.7\times10^{-3}$ disc-truncation bias (analytic), not quadrature error; the test comment "the midpoint rule is ~exact" is right, but the residual is the truncation. The ledger's "variance σ² to 3e-3" is accurate as a bound; it could say "truncated-disc variance $0.9973\sigma^2$".
3. **"`incidence_nodes` convention".** The per-axis nodes are identical to the planar ones; the 2D domain is a disc, not the planar square product. The ledger and docstring state the disc explicitly, so this is consistent.
4. **First-moment preservation** holds only for orders whose four neighbours are in the window; the ledger's "weight and first moment preserved inside the window" is accurate.
5. **Not verified here.** The MoLab A6 cross-check (8 µm, survival 0.01320 / 2.46e-4) exceeds the 1 µm depth budget of this verification; CPU/CUDA parity needs a device. $U$ and $W$ themselves belong to other claims.

## Verdict

`rederived`. Bloch reduction, comb angles $\boldsymbol\theta+\mathbf G/K$, Parseval normalization $A\lvert\mathrm{FFT}\rvert^2/N^2$, survival, supercell emptiness, disc-truncated Gaussian quadrature, bilinear deposition and the batched Strang step all match the implementation; the only numerical residual against an exact (expm) solution is the second-order splitting error.
