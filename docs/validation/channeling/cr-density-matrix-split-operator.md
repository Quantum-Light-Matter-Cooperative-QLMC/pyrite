# `cr-density-matrix-split-operator` — paraxial Lindblad solver for planar channeling

Verifier: fresh context. Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/montecarlo/channeling/density.py::{DensityMatrixSolver,DensityMatrixSolver.jump,DensityMatrixSolver.rate_matrix,DensityMatrixSolver.propagate}`. Input kernel: `cr-lindblad-jump-operators` (`ScatteringKernel`, taken as validated). Context: issue #504.

Independence: before reading the `density.py` body, I read the ledger row, the module and method docstrings, and the `ScatteringKernel` field definitions, which are the interface of the validated input claim. The derivation below and the dense reference integrator (`scratch/dense_ref.py`, gitignored) were written first. The finite-margin reference reproduces the sink model as stated in the claim. It was finished after the first full-band comparison but before the body was read.

## Claim

$$
\partial_z\rho=-\frac{i}{\hbar v}\bigl(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger\bigr)+J(\rho),\qquad
H_{\rm eff}=H_\perp-\frac{i\hbar v}{2}C(x,x),\qquad J(\rho)(x,x')=C(x,x')\rho(x,x').
$$

$\rho$ is block-diagonal in Bloch $\kappa_r=rg/n_{\rm cells}$. It is held per block in the eigenbasis of $H_\perp(\kappa_r)$ over orders $\lvert h\rvert\le H$. $J$ is applied in position space on $n_{\rm grid}\ge3H+2$ points per cell. The update is a Strang step: half no-jump step, jump map $1+dz\,J+dz^2J^2/2$, half no-jump step. In the dephasing-dominated limit, $dP/dz=WP-\Gamma P$, with $W_{n'n}=J(\lvert n\rangle\langle n\rvert)_{n'n'}$ and $\Gamma_n=\langle n\rvert C(x,x)\lvert n\rangle$.

## Independent derivation (before reading the body)

### (a) Lindblad form and $H_{\rm eff}$

With $z$ as time, $i\hbar v\,\partial_z\psi=H_\perp\psi$. Multiplicative jump operators $L_k(x)$ give $C(x,x')=\sum_kL_k(x)L_k^*(x')$ and $\sum_kL_k^\dagger L_k=C(x,x)$, a multiplication operator. The anticommutator term is

$$
-\tfrac12\{C(x,x),\rho\}=-\frac{i}{\hbar v}\Bigl(-\frac{i\hbar v}{2}C\rho-\rho\,\frac{i\hbar v}{2}C\Bigr),
$$

so $H_{\rm eff}=H_\perp-\tfrac{i\hbar v}{2}C(x,x)$, as claimed. The trace is conserved: $\operatorname{tr}J(\rho)=\int C(x,x)\rho(x,x)\,dx=\operatorname{tr}(C\rho)$. Units: $C$ is in Å$^{-1}$, so $\hbar vC$ is in eV, like $H_\perp$.

### (b) Bloch blocks

Averaging over the entry point makes $\rho$ commute with translation $T_d$. Both $H_\perp$ and $C$ are invariant under the common shift ($C(x+d,x'+d)=C(x,x')$). The equation therefore preserves $[\rho,T_d]=0$. On a supercell of $n_{\rm cells}$ periods with periodic boundaries, the momenta are $k_l=lg/n_{\rm cells}$, and $T_d$-invariance couples only $l\equiv l'\pmod{n_{\rm cells}}$. Hence $\rho=\oplus_r\rho_r$ with $l=r+hn_{\rm cells}$.

In the kernel's representation $C(x,x')=\sum_{r,i,j}K_r[i,j]e^{ik_ix}e^{-ik_jx'}$, with $k_i=\kappa_r+h_ig$. With unit-normalized supercell plane waves,

$$
\langle l\rvert J(\rho)\lvert l'\rangle=\sum_{t,t'}\hat C[t,t']\,\rho[l-t,\,l'-t'],\qquad \hat C[r+h_in,\,r+h_jn]=K_r[i,j].
$$

Kernel residue $r$ moves block $b$ to $b+r$. When $b+r\ge n_{\rm cells}$, a carry adds one to the order: $\kappa_b+\kappa_r=\kappa_{b+r-n}+g$. The full loss is the Toeplitz matrix $\langle l\rvert C(x,x)\lvert l'\rangle=D_{(l-l')/n}$, which is non-zero only for $l\equiv l'$.

### (c) Position-space product and the alias-free grid

Write block $b$ in position space as $\rho_b(x_a,x_c)=\sum_{m,m'}\rho_b[m,m']e^{iq_mx_a}e^{-iq_{m'}x_c}$, with $q_m=\kappa_b+h_mg$. The cell-offset form is $R[j](a,c)=\rho(x_a,x_c+jd)=\sum_be^{-2\pi ibj/n}\rho_b(x_a,x_c)$, a forward DFT over $b$. The product $R[j]\,K_{\rm cell}[j]$ is the exact pointwise $J$. The inverse DFT over $j$ returns block positions. Removing $e^{i\kappa_b(x_a-x_c)}$ and taking a 2D DFT over the cell grid gives the coefficients. Each of the two DFTs contributes $1/n_{\rm grid}$, so the normalization is $1/n_{\rm grid}^2$.

The product momenta in block $b$ have orders $h_i+h_m+c$, with $c\in\{0,1\}$. They span $[-2H,\,2H+1]$. To read $h\in[-H,H]$ without aliasing, every alias $h\pm n_{\rm grid}$ must fall outside that span. This requires $n_{\rm grid}>2H+1+H$ and $n_{\rm grid}>3H$, so

$$
\boxed{n_{\rm grid}\ge3H+2},
$$

the condition claimed. Product components outside $\lvert h\rvert\le H$ are discarded. Together with the transfers the kernel does not keep, they are the overflow: $\operatorname{tr}(D\rho)-\operatorname{tr}_{\rm band}J(\rho)\ge0$.

### (d) Strang step and order

Let $A$ be the no-jump generator and $B=J$. Then $e^{A\,dz/2}e^{B\,dz}e^{A\,dz/2}$ has local error $O(dz^3)$. Replacing $e^{dzJ}$ by $1+dzJ+dz^2J^2/2$ adds $O(dz^3)$ locally, so the scheme stays second order globally.

### (e) Flux bookkeeping, and why $\operatorname{tr}+{\rm flux}-1=O(dz^2)$

For the projected model, $\partial_z\rho_S=-\tfrac{i}{\hbar v}(H_{{\rm eff},S}\rho_S-\rho_SH_{{\rm eff},S}^\dagger)+P_SJ(\rho_S)P_S$, with fluxes

$$
\dot F_{\rm free}=\operatorname{tr}P_FJ(\rho_S),\qquad \dot F_{\rm over}=\operatorname{tr}(D\rho_S)-\operatorname{tr}_{\rm band}J(\rho_S).
$$

For $\rho_S$ supported on $S$, $\operatorname{tr}(P_SDP_S\rho_S)=\operatorname{tr}(D\rho_S)$. The total derivative is

$$
\partial_z(\operatorname{tr}\rho_S+F)=-\operatorname{tr}D\rho_S+\operatorname{tr}P_SJ+\operatorname{tr}P_FJ+\operatorname{tr}D\rho_S-\operatorname{tr}_{\rm band}J=0
$$

exactly. Discretely, the no-jump half steps remove $\int_0^{dz}\operatorname{tr}(D\rho)\,dz'$. Booking $dz\,\operatorname{tr}D\rho_{1/2}+\tfrac{dz^2}{2}\operatorname{tr}D\,J_S\rho_{1/2}$ at the midpoint matches this to $O(dz^3)$ per step. The midpoint-rule error from the two halves cancels at $O(dz^2)$. The jump-map terms balance exactly at orders $dz$ and $dz^2$. The defect is therefore $O(dz^3)$ per step and $O(dz^2)$ globally. It measures convergence and is not an identity.

### (f) Pauli limit

If off-diagonal $\rho_{nm}$ dephase fast compared with scattering, $\partial_zP_{n'}=\sum_n\langle n'\rvert J(\lvert n\rangle\langle n\rvert)\lvert n'\rangle P_n-\langle n'\rvert D\lvert n'\rangle P_{n'}$. The anticommutator gives $-\operatorname{Re}\sum_mD_{n'm}\rho_{mn'}\to-D_{n'n'}P_{n'}$. The gain term equals $\sum_k\lvert\langle n'\rvert L_k\lvert n\rangle\rvert^2$, the golden-rule rate. Blocks never cohere, so the degeneracy across $\kappa$ does not spoil the secular limit.

## Implementation diff

| Item | Code | Verdict |
| --- | --- | --- |
| No-jump step | `_propagators` (density.py:212–222): `expm` of $-\tfrac{i\,dz}{2\hbar v}\bigl(\operatorname{diag}E-\tfrac{i\hbar v}{2}V^\dagger DV\bigr)$ on the $S$ rows and columns of each block | matches, with $H_{\rm eff}$ restricted to $S$ |
| Jump | `jump` (density.py:239–249): $V\rho V^\dagger$ to plane waves, then `_waves` to the grid; `fft` over $r$ times `cell_kernel` (`fft` of the same sign); `ifft`; back-projection divided by $n_{\rm grid}^2$ | matches (c), including the sign of the DFT over offsets |
| Grid | `minimum = 3*max_order + 2`, rounded up to even (density.py:170–173) | matches; the even rounding is not needed |
| Strang/Taylor | `rho + dz*first + 0.5*dz**2*second`, where `second = J(first)` and `first` is the $S$ projection (density.py:420–429) | matches $(1+dzJ_S+dz^2J_S^2/2)$ |
| Flux | free $=dz\,{\rm free}_1+\tfrac{dz^2}{2}{\rm free}_2$; overflow $=dz({\rm loss}_1-{\rm band}_1)+\tfrac{dz^2}{2}({\rm loss}_2-{\rm band}_2)$ | matches (e) |
| Loss | `loss_rate`: $\operatorname{tr}(D_{\rm Toeplitz}V\rho V^\dagger)$ with the full `diagonal_coefficients` | matches |
| Pauli | `rate_matrix`: $S$-diagonal of `jump` on $\lvert rn\rangle\langle rn\rvert$; `loss_rate` of the same | matches (f) |
| Energy loss | kernel $\times(\beta_0/\beta)^2$; $\hbar v$ updated in $H_{\rm eff}$ | $C\propto v^{-2}$, which agrees with `cr-lindblad-jump-operators` |

## Independent numerical checks

Reference: dense plane-wave Lindbladian on the full band $l\in[-Hn,(H+1)n-1]$, built from (b). $H_\perp$ uses my own constants ($\hbar c=1973.269804$ eV Å, $mc^2=510998.95$ eV) and the profile's $U_m$. The input is the kernel's `momentum_kernel` and `diagonal_coefficients`. Integration is exact (`scipy.linalg.expm`) or RK4 at 5 Å. Si(110), 17 MeV, 293 K.

**1. Full band (all states in $S$), TDS kernel, $n_{\rm cells}=3$, $H=5$, entry $\lvert l=0\rangle$, $z=1\ \mu$m.** The reference trace is 0.7830720 and the overflow 0.2169280.

| $dz$ [Å] | $\max\lvert P_{\rm sol}-P_{\rm ref}\rvert$ | $\max\bigl\lvert\lvert\rho_{\rm sol}\rvert-\lvert\rho_{\rm ref}\rvert\bigr\rvert$ | overflow | defect |
| --- | --- | --- | --- | --- |
| 400 | $1.0\times10^{-4}$ | $1.5\times10^{-3}$ | 0.217533 | $5.0\times10^{-4}$ |
| 200 | $1.8\times10^{-5}$ | $2.0\times10^{-4}$ | 0.216941 | $-3.2\times10^{-5}$ |
| 100 | $4.3\times10^{-6}$ | $4.3\times10^{-6}$ | 0.216929 | $-1.1\times10^{-5}$ |
| 50 | $1.07\times10^{-6}$ | $1.07\times10^{-6}$ | 0.216928 | $-3.0\times10^{-6}$ |

The error is second order, 4.0× per halving. All three Bloch blocks are populated: 0.585, 0.099 and 0.099.

**2. Carries and k-folding, synthetic kernel.** $n_{\rm cells}=4$, $H=4$, and a random rank-3 PSD $K_r$ in every residue. The diagonal equals the kept diagonal, optionally plus $3\times10^{-5}$ Å$^{-1}$ of pure overflow. Entries were $l_0=0$, $3$ and $-5$, the last being block 3, $h=-2$, over 2 µm. Block populations agree to $10^{-5}$. $\max\bigl\lvert\lvert\rho\rvert-\lvert\rho_{\rm ref}\rvert\bigr\rvert$ is $1.7\times10^{-5}$ at $dz=200$ and $3.6\times10^{-6}$ at $dz=100$. Overflow agrees to $10^{-5}$ in every case (0.17584, 0.18421, 0.19351, and with extra loss 0.54769, 0.55228, 0.55739). Every $b+r\ge n$ carry path is exercised.

**3. Finite margin (5 eV), TDS plus electronic, $n_{\rm cells}=3$, $H=5$, 1 µm.** The reference is the projected model of (e), RK4 at 5 Å. Its balance is $\lvert{\rm tr}+F-1\rvert<2\times10^{-15}$. There are 5, 6 and 6 $S$ states per block.

| entry | $dz$ | $\max\bigl\lvert\lvert\rho\rvert-\lvert\rho_{\rm ref}\rvert\bigr\rvert$ | trace sol/ref | free sol/ref | overflow sol/ref | entry sol/ref | defect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $l_0=0$ | 200 | $4.3\times10^{-4}$ | 0.65236/0.65689 | 0.11955/0.12021 | 0.21987/0.22115 | 0.001750/0.001750 | $-6.5\times10^{-3}$ |
| $l_0=0$ | 100 | $1.1\times10^{-4}$ | 0.65571/0.65689 | 0.12004/0.12021 | 0.22081/0.22115 | same | $-1.7\times10^{-3}$ |
| $l_0=0$ | 50 | $2.8\times10^{-5}$ | 0.65659/0.65689 | 0.12017/0.12021 | 0.22106/0.22115 | same | $-4.3\times10^{-4}$ |
| $l_0=2$ | 50 | $2.8\times10^{-5}$ | 0.66043/0.66073 | 0.13923/0.13927 | 0.19975/0.19983 | 0.000165/0.000165 | $-4.3\times10^{-4}$ |

The solver converges at second order to an independently integrated sink model. The defect falls 3.8–3.9× per halving.

**4. Aliasing guard.** `n_grid = 3H+1` raises. $3H+2$ and larger are accepted, and the results above use the default even size.

**5. Pauli rates.** `rate_matrix` agrees with $W$ and $\Gamma$ from my dense $J$ and $D$ on the 17 $S$ states to $2.6\times10^{-15}$ relative. Every column satisfies $\sum_{n'}W_{n'n}\le\Gamma_n$; the minimum slack is 3.8%, which is overflow plus free. For the dephasing-limit trend I integrated $dP/dz=(W-\Gamma)P$ with my own matrices from the diagonal of a $\sigma=1$ mrad entry. I compared it with the solver at a fixed scattering dose ($z=5000\ {\rm \AA}/s$). The maximum population error relative to $\max P$ is $2.4\times10^{-3}$, $3.0\times10^{-4}$ and $4.8\times10^{-5}$ for kernel scales $s=1$, $0.1$ and $0.03$. It vanishes as the scattering weakens relative to dephasing.

**6. Repository tests.** `tests/montecarlo/test_channeling_density.py` and `test_channeling_mcwf.py`: 13 passed.

## Findings

1. **None blocking.** Every term of the claim re-derives, and the solver converges at second order to an independent dense Lindblad integration. This holds in the full band, with random kernels across all carry paths, and with the dechanneling sink.
2. **Model statement (wording).** `_propagators` (density.py:216–221) builds the no-jump propagator from $P_SH_{\rm eff}P_S$. The loss operator's $S$–$F$ matrix elements are dropped along with the $S$–$F$ coherences. This is the consistent projected model, and the trace balance in (e) is exact for it. The claim should say so explicitly. As written, it says only that "$S$–$F$ coherences are dropped".
3. **Representation note.** The band per residue is $\kappa_r+hg$ with $\kappa_r\in[0,g)$ and $\lvert h\rvert\le H$. It covers momenta $[-Hg,(H+1)g)$, one order asymmetric about $k_x=0$. This is a valid truncation and identical across the code, the kernel and my reference. Truncation sensitivity is therefore not symmetric in tilt sign at small $H$.
4. **Coefficient of the defect.** At 5 eV margin, the balance defect over 1 µm is about 13× larger than in the full band at the same $dz$: $-6.5\times10^{-3}$ against $5\times10^{-4}$ at 200 Å. It is still $O(dz^2)$, and the ledger's "below $5\times10^{-3}$ at $dz=100$ Å" holds ($1.7\times10^{-3}$).

## Verdict

`rederived`. The ledger Status stays with the task owner. The row cannot reach `anchored` before the abTEM slice, as its Notes already say.

## Resolution (lead, after the verdict)

The claim texts of this row and `cr-dechanneling-collapse` now state that the no-jump propagator uses $P_SH_{\rm eff}P_S$. The `free_flux_by_energy` docstring now says that each deposit is binned at the basis energies current when it happened, and that out-of-range weight is clipped into the end bins.
