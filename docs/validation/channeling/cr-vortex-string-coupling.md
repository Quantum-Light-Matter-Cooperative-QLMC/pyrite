# `cr-vortex-string-coupling`: cylindrical-harmonic (OAM) projection about an atomic string

Fresh-context verification for #508 (vortex slice), at worktree HEAD `c2d62377`. Scope: the projection mathematics of `montecarlo/channeling/vortex.py::{CylindricalProjection,angular_momentum}` and the stated symmetry claims. The physics anchors of the issue (Xin & Zheng on-column 2p selectivity; Lubk loss of selectivity off-centre) are recorded as open in the ledger Notes and are not assessed here.

**Independence caveat.** The implementation body was read before the derivation below was written. The derivation is standard, and every numerical comparison uses an independent evaluator (a closed-form Jacobi–Anger/Bessel sum written for this check), not implementation helpers. A reviewer who wants strict ordering should treat this as a strong `filtered` plus an independent numeric oracle.

Verdict: **rederived**, with two accuracy caveats (Section 7). Neither contradicts the ledger claim, but both are worth stating in the row.

## 1. Cylindrical-harmonic decomposition

Let $\psi$ be square-integrable on the disk $D=\{\lvert\mathbf r-\mathbf c\rvert\le R\}$. Write $\mathbf r=\mathbf c+\rho\,\hat{\mathbf e}_\theta$ with $\hat{\mathbf e}_\theta=(\cos\theta,\sin\theta)$. For each $\rho$, $\{e^{i\ell\theta}/\sqrt{2\pi}\}_{\ell\in\mathbb Z}$ is an orthonormal basis of $L^2(0,2\pi)$, so

$$
\psi(\mathbf c+\rho\hat{\mathbf e}_\theta)=\sum_\ell a_\ell(\rho)\frac{e^{i\ell\theta}}{\sqrt{2\pi}},\qquad
a_\ell(\rho)=\frac{1}{\sqrt{2\pi}}\int_0^{2\pi}\psi(\mathbf c+\rho\hat{\mathbf e}_\theta)\,e^{-i\ell\theta}\,d\theta .
$$

Parseval on each circle, then $dA=\rho\,d\rho\,d\theta$, gives

$$
\int_D\lvert\psi\rvert^2dA=\sum_\ell P_\ell,\qquad P_\ell=\int_0^R\lvert a_\ell(\rho)\rvert^2\rho\,d\rho .
$$

$L_z=-i\hbar\,\partial_\theta$ about $\mathbf c$ acts as $\hbar\ell$ on the $\ell$-th harmonic, so $\langle L_z\rangle_D=\int_D\psi^*L_z\psi\,dA=\hbar\sum_\ell\ell P_\ell$. In units of $\hbar$, this is `angular_momentum` $=\sum_\ell\ell P_\ell$ over the reported $\ell$; harmonics with $\lvert\ell\rvert>\ell_{\max}$ are truncated from both sums.

**Projector and linearity.** $P_\ell=\langle\psi\rvert\Pi_\ell\lvert\psi\rangle$ with $\Pi_\ell=\chi_D\,\Pi^{\rm ang}_\ell$. Here $\chi_D$ is radial and $\Pi^{\rm ang}_\ell$ projects onto angular dependence $e^{i\ell\theta}$, so they commute and $\Pi_\ell$ is an orthogonal projector. Then $P_\ell=\mathrm{Tr}(\varrho\,\Pi_\ell)$ is linear in $\varrho$. The `AxialMCWF.run` loop evaluates the observable on each surviving trajectory, normalized on the grid with that trajectory's current tilt, and divides the sum by `n_electrons`; removed electrons contribute zero. The ensemble mean is therefore $\mathrm{Tr}(\bar\varrho\,\Pi_\ell)$ for the trace-decreasing Lindblad $\bar\varrho$. The claim holds.

## 2. Discretization

**Azimuthal.** With $\theta_j=2\pi j/n_\theta$, the trapezoid rule gives

$$
a_\ell(\rho)\approx\frac{1}{\sqrt{2\pi}}\frac{2\pi}{n_\theta}\sum_j\psi_j e^{-i\ell\theta_j}=\frac{\sqrt{2\pi}}{n_\theta}\,\mathrm{DFT}[\psi]_\ell .
$$

This is exact up to aliasing $\ell\to\ell+mn_\theta$. The code computes `np.fft.fft(values, axis=1) * (sqrt(2 pi) / n_theta)` and reads column $\ell\bmod n_\theta$, so it matches.

**Radial.** Gauss–Legendre on $[0,R]$ with $\rho_j=\tfrac R2(x_j+1)$ gives weights $\tfrac R2w_j\rho_j$, including the Jacobian $\rho$. This matches `_rho_weight`. The integrand $\lvert a_\ell\rvert^2\rho$ is entire but oscillates on the scale $1/k$, so the rule is spectrally accurate only when $n_{\rm radial}\gtrsim kR$ for the occupied $k$ (Section 7b).

## 3. Off-grid evaluation of $\psi$

On the periodic grid, the validated convention (`cr-axial-realspace-propagation`) is $\psi=e^{i\mathbf k_t\cdot\mathbf r}\phi$, with $\phi(\mathbf r)=N^{-1}\sum_{\mathbf k}\tilde\phi_{\mathbf k}e^{i\mathbf k\cdot\mathbf r}$ over the `fftfreq` set, where $\tilde\phi=\mathrm{FFT}(\phi)$. This is the band-limited function that `RealSpaceGrid.sample` produces and the propagator's kinetic symbol $\lvert\mathbf k+\mathbf k_t\rvert^2$ acts on. The code evaluates exactly this sum at the polar nodes (separable `einsum`) and multiplies by $e^{i\mathbf k_t\cdot\mathbf r}$ in absolute coordinates. Relative to a centre-referenced tilt, that adds the constant phase $e^{i\mathbf k_t\cdot\mathbf c}$, which drops out of $\lvert a_\ell\rvert^2$. **Tilt handling matches.**

**Even-$N$ Nyquist.** `fftfreq` puts the Nyquist row at $-\pi/\Delta x$ only. On grid nodes, $e^{\mp i\pi n}$ agree, so the grid dynamics are unaffected. Off grid, the continuation is convention-dependent:

- real $\phi$ gives a complex interpolant (imaginary/modulus $\approx0.12$ at a random off-grid point for white noise);
- the interpolant is not covariant under the grid's $90^\circ$ index rotation, because the $\mathbf k$ set $\{-N/2,\dots,N/2-1\}$ is not $C_4$-closed.

It matters only to the extent that the Nyquist rows/columns carry weight. See Section 6 for the measured effect.

## 4. Alias-free $n_\theta$

For a single plane wave, the Jacobi–Anger expansion

$$
e^{i\mathbf q\cdot(\mathbf c+\rho\hat{\mathbf e}_\theta)}=e^{i\mathbf q\cdot\mathbf c}\sum_\ell i^\ell J_\ell(q\rho)\,e^{i\ell(\theta-\alpha_{\mathbf q})}
$$

gives $a_\ell(\rho)=\sqrt{2\pi}\,e^{i\mathbf q\cdot\mathbf c}\,i^\ell J_\ell(q\rho)e^{-i\ell\alpha_{\mathbf q}}$. The harmonics are not strictly band-limited. For $\lvert\ell\rvert$ beyond $qR$, $J_\ell(q\rho)$ decays after an Airy transition of width $\sim(qR)^{1/3}$, and then super-exponentially, as $(eqR/2\lvert\ell\rvert)^{\lvert\ell\rvert}$. A reported $\lvert\ell\rvert\le\ell_{\max}$ is aliased only from $\lvert\ell'\rvert\ge n_\theta-\ell_{\max}$.

The code uses `need = 2 (ceil(k_max R) + max_ell) + 16`, rounded up to a power of two. Here `k_max = hypot(max|kx|+|t_x|, max|ky|+|t_y|)` is at least $\max\lvert\mathbf k+\mathbf k_t\rvert$, so the bound is conservative. Then $\lvert\ell'\rvert\ge2k_{\max}R+\ell_{\max}+16$, roughly twice the cutoff, where $J$ is negligible. The bound in the ledger, $n_\theta>2(\lvert\mathbf k+\mathbf k_t\rvert_{\max}R+\ell_{\max})$, is satisfied.

For `disk_weight`, which sums all bins and so is the trapezoid rule for $\lvert\psi\rvert^2$, the cross-alias error involves $\lvert J_{n_\theta/2}\rvert^2$ with $n_\theta/2\ge k_{\max}R+8$. That is only an Airy-tail margin, but the power-of-two rounding and the corner $k_{\max}$ make it negligible in practice (Section 6).

## 5. Symmetry claims

**Plane wave, any tilt.** From Section 4, $\lvert a_\ell\rvert^2=2\pi\lvert c\rvert^2J_\ell(q\rho)^2$, and $J_{-\ell}=(-1)^\ell J_\ell$, so $P_\ell=P_{-\ell}$ and $\langle L_z\rangle_D=0$ about every centre. The total is $2\pi\lvert c\rvert^2\int_0^R\rho\,d\rho\sum_\ell J_\ell^2=\pi R^2\lvert c\rvert^2$. This is correct as stated: a *single* plane wave, so not a superposition of tilts. The discrete DFT preserves $\ell\leftrightarrow-\ell$ symmetry, because the aliases are symmetric too.

**$C_m$ string lattice.** Suppose the Hamiltonian commutes with the rotation $R_m$ by $2\pi/m$ about $\mathbf c$, and the initial state is an eigenstate of $R_m$, for example a centred $\ell$ vortex with eigenvalue $e^{-2\pi i\ell/m}$. Then the state stays in that eigenspace, which is spanned by harmonics $\ell'\equiv\ell\pmod m$. The claim requires:

1. $C_m$ symmetry of the projected potential about $\mathbf c$;
2. zero transverse tilt, because $\lvert\mathbf k+\mathbf k_t\rvert^2$ is not $C_m$-invariant for $\mathbf k_t\ne0$;
3. on the grid, a $C_m$-closed node set (only $m=2,4$ on a square grid with $\mathbf c$ on a node).

Si ⟨001⟩ meets condition 1: the $\bar4$ site symmetry of diamond about [001] projects to $C_4$. The test meets conditions 2 and 3: `step=1.0`, no tilt, $64\times64$ on $4a_p\times4a_p$ with $a_p=1.92011$ Å, and $\mathbf c$ at node $(32,32)$. The measured $C_4$ residual of `profile.on_grid` under index rotation about node 32 is $5.6\times10^{-16}$. The DFT kinetic operator commutes with that index rotation, because the Nyquist index is self-conjugate mod $N$ and $k^2$ is symmetric. The ledger wording "an $m$-fold string lattice conserves $\ell\bmod m$ for a centred state" omits conditions 2 and 3.

## 6. Numerical evidence

Scratch scripts are in the session scratchpad (`validate-vortex/check{1,2,3}.py`).

| check | result |
| --- | --- |
| `pyrite-dev test tests/montecarlo/test_channeling_vortex.py` | 9 passed (3.1 s) |
| Independent Bessel oracle versus `project`: random complex $\phi$ on $16\times20$ ($10\times12$ Å), $\mathbf c=(4.3,6.1)$, $R=3$, tilt $(0.7,-1.1)$, $\lvert\ell\rvert\le10$, $k_{\max}R=25.6$ | max $\lvert\Delta P_\ell\rvert/\sum P=4.5\times10^{-15}$ |
| $n_\theta$ doubled ($512\to1024$), white noise on $64^2$ / 20 Å, $R=9.5$, $\ell_{\max}=8$ | $9\times10^{-17}$ relative |
| $n_\theta=128$, below the bound | $1.3\times10^{-3}$ relative; the bound is load-bearing |
| `disk_weight` at $n_\theta=256$ (just below the bound $2\lceil k_{\max}R\rceil+16=284$) versus default 512 | $3\times10^{-5}$ |
| $C_4$-eigen white-noise state ($\ell\equiv1$) on $32^2$, $\mathbf c$ on a node: forbidden-class weight | $1.4\times10^{-3}$ with Nyquist content; $1.5\times10^{-30}$ with Nyquist rows/columns zeroed |
| Same, $\ell\equiv3$ eigenstate: allowed-class weight | $1.0$; leakage appears only in the other class at the $10^{-3}$ level |
| Nyquist weight fraction of the test's centred LG (waist 0.5 Å, $\Delta x=0.120$ Å) | $1.8\times10^{-33}$ |
| Radial convergence, white noise on $64^2$ / 20 Å, $R=9.5$ ($k_{\rm axis}R\approx95$): `disk_weight` at $n_{\rm radial}=48/96/160$ | 559.725 / 559.4167 / 559.4167 ($5.5\times10^{-4}$ error at the default 48); $P_0$ error $3\times10^{-3}$ |

## 7. Adjudication

All formulas match: harmonic normalization ($\sqrt{2\pi}/n_\theta$), the Gauss–Legendre weights including $\rho$, Parseval/$L_z$ identities, linearity in $\varrho$, tilt handling, Jacobi–Anger symmetry, and the $n_\theta$ bound.

**(a) Nyquist convention.** "Evaluated exactly" is exact for the `fftfreq` band-limited continuation that `sample` and the propagator define. That continuation is one of several consistent with the grid samples, and it is not $C_4$-covariant or reality-preserving off grid. $P_\ell$ therefore carries a convention error bounded by the Nyquist row/column weight. This is $10^{-3}$ for white noise and negligible ($10^{-33}$) for the test states. A symmetric split (half the Nyquist coefficient on $\pm N/2$, that is, $\cos$) or zeroing would remove it; this is a recommendation, not a discrepancy.

**(b) Radial resolution.** $n_\theta$ adapts to $k_{\max}R$, but `n_radial` (default 48) does not. For states with content near the grid band edge and $kR\gtrsim n_{\rm radial}$, $\sum P_\ell$ and $P_\ell$ carry quadrature error ($\sim10^{-3}$ at $kR\approx95$). Smooth, resolved states, which covers all tests, are exact to $10^{-12}$. Recommend documenting $n_{\rm radial}\gtrsim k_{\rm occ}R$, or choosing it like $n_\theta$.

**Minor.** The class docstring says the callable returns $P_\ell$ "of the state normalized on the grid", but `__call__` does not normalize. It relies on `AxialMCWF` passing normalized states, which it does.

**Scope framing.** The ledger Notes correctly mark the Xin & Zheng and Lubk physics anchors as open, and the row claims only projection and symmetry. The bound-state energies quoted in the Notes (Si ⟨001⟩ 2p at 100–300 keV, −18.0 eV at 10 MeV) were not checked here.

## 8. Decay-observable review (2026-10-17)

Scope: the subsequent W 300 keV population follow-up at base `c4ab4179`.
This verifier did not implement it. The population test was inspected before
this derivation; this is therefore an adversarial observable check, not a
new independent re-derivation of the TDS or electronic scattering kernel.
The expression below was recorded before inspecting the MCWF implementation.

### Physical observable and carrier convention

For orthonormal fixed-string 2p orbitals $u_a$, the physical population is
$p_{2p}=\operatorname{Tr}(\rho\Pi_{2p})$, with
$\Pi_{2p}=\sum_a|u_a\rangle\langle u_a|$.
In the carrier representation $\psi(\mathbf r)=e^{i\mathbf t\cdot\mathbf r}\phi(\mathbf r)$,

$$
p_{2p}(\phi,\mathbf t)=\sum_a\left|
\int d^2r\,u_a^*(\mathbf r)e^{i\mathbf t\cdot\mathbf r}\phi(\mathbf r)
\right|^2.
$$

It is dimensionless, non-negative and at most one for a normalized state
and an orthogonal projector. At zero tilt it reduces to the ordinary
overlap. Under a change of carrier representation
$\phi\mapsto e^{-i\mathbf q\cdot\mathbf r}\phi$,
$\mathbf t\mapsto\mathbf t+\mathbf q$, it is invariant. Omitting the
carrier phase instead computes occupation of orbitals tilted along with
the trajectory, a different observable. An electronic momentum-transfer
jump $L_{\mathbf q}\propto\sqrt{\rho_q(\mathbf r)}e^{i\mathbf q\cdot\mathbf r}$
must therefore affect the fixed-string overlap even when its phase is
stored entirely in $\mathbf t$.

For a normalized eigenstate, the total jump length is
$\lambda_{\rm jump}^{-1}=\langle u|\sum_\alpha L_\alpha^\dagger L_\alpha|u\rangle$,
with units Å$^{-1}$. Its population derivative is instead

$$
\frac{d p_{2p}}{dz}=
-\frac{i}{\hbar v}\operatorname{Tr}(\Pi_{2p}[H,\rho])+
\sum_\alpha\operatorname{Tr}\left[
\left(L_\alpha^\dagger\Pi_{2p}L_\alpha
-\tfrac12\{\Pi_{2p},L_\alpha^\dagger L_\alpha\}\right)\rho\right].
$$

The positive recycling term prevents identification of jump length with
population lifetime. The artificial grid sink contributes loss to the
represented ensemble; it does not establish physical escape. Coherent
eigenstate stationarity likewise establishes no scattering lifetime.

### Comparison and numerical counterexample

`AxialMCWF._electronic_jump` returns
`sqrt(smoothed_density(q)) * phi` and adds the sampled momentum transfer to
`tilt`. `AxialMCWF.run` passes that envelope and tilt to the observable.
The test's `two_p(phi, tilt)` ignores `tilt` and contracts its orbitals
directly with `phi`. The first divergent term is therefore the missing
$e^{i\mathbf t\cdot\mathbf r}$ in the overlap integral, not the MCWF
waiting time or the electronic jump rate.

A deterministic counterexample uses exactly the test's W $\langle001\rangle$
300 keV orbitals (`max_order=12`), three-cell 72² grid, central-cell cut
and discrete normalization. Their Gram matrix has unit diagonal and
off-diagonal magnitude below $3\times10^{-16}$. Set
$\phi=e^{-iqx}u_1$ and $\mathbf t=(q,0)$: every row represents the same
physical state $\psi=u_1$, with true 2p population one.

| $q$ [Å$^{-1}$] | carrier-aware population | current test's overlap |
| --- | --- | --- |
| 0 | 1.000000 | 1.000000 |
| 1 | 1.000000 | 0.948243 |
| 3 | 1.000000 | 0.630300 |
| 5 | 1.000000 | 0.312356 |

This is a representation-invariance test with an exact known answer,
not a stochastic decay fit. It proves the electronic population regression
measures the wrong observable; it does not quantify the correction to any
specific ensemble curve. Both over- and underestimates are possible for
general states. TDS-only runs at the test's zero initial tilt keep zero
carrier tilt and are unaffected by this particular error.

The minimal correction is to project
`exp(1j * (tilt[0] * grid.x[:, None] + tilt[1] * grid.y[None, :])) * phi`
against the fixed orbitals. Pin carrier invariance with the construction
above before regenerating the seeded electronic reference and its
24-versus-32-samples-per-cell convergence evidence. No change to the jump
operator follows from this finding.

The recorded grid-convergence curves remain historical model results.
The TDS-plus-electronic values must not be treated as validated fixed-string
populations until the observable is corrected and the curves rerun. A
passing seeded regression against those same values would not resolve the
discrepancy. The published coherent phase/selectivity anchors cannot
establish the dissipative population lifetime.

- **Claim:** `cr-vortex-string-coupling` — population follow-up in
  `tests/montecarlo/test_channeling_axial_dissipator.py::test_tungsten_two_p_population_decay`.
- **Filters:** units pass; zero-tilt limit passes; carrier-representation
  invariance fails when electronic kicks create nonzero tilt.
- **Re-derivation:** differs by the missing carrier phase in the projector.
- **Verdict:** `discrepancy` for the TDS-plus-electronic population
  extension; no new verdict on the parent scattering kernels or the
  previously checked cylindrical-harmonic projector.
- **Suggested ledger change:** flag the electronic population follow-up as
  a discrepancy and withdraw its numeric values as physical population
  evidence. Preserve the established projection/symmetry derivation and
  distinguish the unaffected zero-tilt TDS-only evidence. Do not promote
  the combined vortex acceptance to `anchored` from the current curves.

The implementation and ledger were not changed by this verifier, following
the [independent-verifier contract](../methodology.md#independent-verifier-contract).
