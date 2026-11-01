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
| `pyrite-dev test tests/montecarlo/quantum/test_channeling_vortex.py` | 9 passed (3.1 s) |
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
  `tests/montecarlo/quantum/test_channeling_axial_dissipator.py::test_tungsten_two_p_population_decay`.
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

The implementation and ledger were not changed during this verification, following
the [independent-verifier contract](../methodology.md#independent-verifier-contract).

## 9. Implementation follow-up (2026-10-17)

This is an implementation-owner response to Section 8, not an independent
validation of the corrected decay curves. The production `LevelDensity`
observable already includes the carrier phase; the error was confined to
the population regression's local observable. The scattering kernels,
propagator, RNG streams and production observables are unchanged.

The regression now uses `_orbital_population`, which includes the full
carrier phase before contracting with the fixed normalized orbitals.
Four analytic carrier-invariance cases use an orthonormal pair of grid
Fourier modes and a normalized coherent superposition whose true subspace
population is one. A fifth test applies an actual reciprocal-grid
momentum kick to a constant envelope: its overlap with the original
constant orbital must become zero. This distinguishes a physical kick
from a mere change of carrier representation. Before correction, the
three nonzero-carrier cases and the kick case failed; all five pass after
correction, each in less than 5 ms. Their absolute tolerance is
$10^{-12}$ for these small, well-conditioned discrete inner products.

With 256 electrons, seed 11, a three-cell 72² grid and 1 Å steps, the
corrected TDS-plus-electronic 2p populations at 0, 10, 20, 50 nm are
0.9427, 0.2980, 0.0913, 0.0035. The TDS-only regression still passes
against its original 0.9427, 0.3392, 0.1194, 0.0065 reference. The
population tolerance remains 0.01; it was not widened to absorb the
change. These are seeded model regressions, not measured lifetimes.

### Corrected grid comparison

SLURM array 1569 on `amd-node` reran both mechanisms on three-cell 72²
and 96² grids (24 and 32 samples per cell), with 1024 electrons per case,
seed 11 and 1 Å steps. Entry, normalized cut 2p orbitals (`max_order=12`)
and the corrected observable match the regression. The entry population
is 0.9427 on either grid.

| mechanisms | grid | 10 nm | 20 nm | 50 nm |
| --- | --- | --- | --- | --- |
| TDS | 72² | 0.3553 | 0.1343 | 0.0071 |
| TDS | 96² | 0.3624 | 0.1402 | 0.0081 |
| TDS + electronic | 72² | 0.3048 | 0.1108 | 0.0041 |
| TDS + electronic | 96² | 0.3123 | 0.1146 | 0.0049 |

For a trajectory population bounded by zero and one, the estimated
standard-error scale is at most $\sqrt{p(1-p)/1024}$: approximately
0.015, 0.010 and 0.002–0.003 at these depths. The population differences
between the two grids lie below those scales. This agreement does not
establish convergence of the entire density matrix or scattering kernel.
At 50 nm the represented trace is 0.6934/0.8125 (TDS, 72²/96²) and
0.6729/0.7900 (TDS + electronic). Sink counts are 314/192 and 335/215,
respectively, and electronic jump counts are 727/836. Those quantities
remain resolution-dependent. A grid sink is not physical escape.

The isolated snapshot was created by `pyrite remote sync`, without
changing the shared checkout used by other jobs. Its recorded code
digest is `ff10c6d36f9bb59e3159b4622e988b5fe26fc2bdb7e44713efe0a6f2785e2dd5`
(base `634c3c13`, dirty test edits). The uploaded regression module's
SHA-256 is `0c61dcac08c751a3fdd75d00d665efc1a8ee918cfc40c04827bec7a82f469a16`.
Raw outputs are `scratch/decay-{0,1,2,3}.json` under
`/home/user/pyrite-508-carrier` on `cluster-head`. The jobs explicitly selected
that snapshot's bundled catalog rather than the remote user's default.
These results supersede the envelope-only electronic references; they
still require fresh-context physics validation before the vortex claim
can advance to `anchored`.

## 10. Independent corrected-observable verification (2026-10-18)

This fresh verifier did not implement the correction. The ledger and earlier
write-ups were read first; the following derivation was recorded before reading
the corrected helper, regression bodies or MCWF implementation. Thus the
derivation is independent of the implementation, but not blind to the earlier
observable diagnosis. Scope is the fixed-string observable and the evidence
supporting it, not a fresh validation of either scattering kernel.

### Derivation before implementation inspection

Let $u_a$ be fixed physical string orbitals, orthonormal in the transverse
inner product. The Born-rule projector is

$$
\Pi=\sum_a |u_a\rangle\langle u_a|,\qquad
p(\phi,\mathbf k_c)=\sum_a\left|
\int u_a^*(\mathbf r)e^{i\mathbf k_c\cdot\mathbf r}
\phi(\mathbf r)\,d^2r\right|^2.
$$

For sampled normalized orbitals the amplitude uses the area element
$\Delta A=\Delta x\Delta y$ inside the sum, hence population contains
$\Delta A^2$. Transverse wavefunctions have units of inverse length;
amplitudes and populations are dimensionless. If cut orbitals are not
orthonormal, normalizing each separately is insufficient: with columns $U$
and Gram matrix $G=U^\dagger U$, the subspace projector is
$UG^{-1}U^\dagger$ (with the grid inner product understood).

The transformation $\phi\mapsto e^{-i\mathbf q\cdot\mathbf r}\phi$,
$\mathbf k_c\mapsto\mathbf k_c+\mathbf q$ leaves the integrand unchanged.
A physical kick changes only one of those quantities and need not preserve
population. At zero carrier this reduces to the ordinary orbital overlap;
for a normalized state in the subspace it gives one and for an orthogonal
state zero. Positivity and the upper bound one follow from $\Pi^2=\Pi$.

For $N$ equal-weight MCWF histories, with surviving normalized states indexed
by $S$, the represented density operator and population are

$$
\bar\rho=\frac1N\sum_{j\in S}|\psi_j\rangle\langle\psi_j|,
\qquad \bar p=\frac1N\sum_{j\in S}p_j.
$$

Dividing by the surviving count would instead measure a conditional
population. Removed histories contribute zero, so
$0\le\bar p\le\operatorname{Tr}\bar\rho\le1$.
The disk angular-momentum moment is
$\langle L_z\rangle_D=\hbar\sum_\ell\ell P_\ell$; its reported numerical
value in units of $\hbar$ is dimensionless. Neither this moment nor the
two-state population is automatically a conditional helicity measurement.

For a length-parametrized Lindblad generator, the population derivative is

$$
\frac{dp}{dz}=-\frac{i}{\hbar v}\operatorname{Tr}(\Pi[H,\rho])
+\sum_\alpha\operatorname{Tr}\!\left[
\left(L_\alpha^\dagger\Pi L_\alpha
-\tfrac12\{L_\alpha^\dagger L_\alpha,\Pi\}\right)\rho\right].
$$

The jump rate $\langle\sum_\alpha L_\alpha^\dagger L_\alpha\rangle$
has units inverse length. Its reciprocal is a waiting length, whereas
population decay includes the recycling term and possible repopulation.
For a coherent spectral projector commuting with $H$, population is constant.
A central-cell cut of an isolated-string eigenfunction need not be an exact
eigenfunction of the finite lattice Hamiltonian, so observed stationarity
is also a numerical test. Neither coherent stationarity nor agreement of
two finite grids establishes a scattering lifetime. Loss to an artificial
momentum/grid sink establishes loss from the represented Hilbert space,
not escape from a physical crystal.

For completeness, before inspecting the cylindrical implementation, angular
orthonormality independently gives
$a_\ell(\rho)=(2\pi)^{-1/2}\int\psi(\mathbf c+\rho\hat{\mathbf e}_\theta)e^{-i\ell\theta}d\theta$.
Parseval followed by the polar Jacobian gives
$P_\ell=\int_0^R|a_\ell|^2\rho d\rho$ and the disk sum rule.
On $n_\theta$ uniform angles the coefficient is
$\sqrt{2\pi}\operatorname{DFT}(\psi)_\ell/n_\theta$.
Gauss–Legendre mapping to $[0,R]$ needs weights
$(R/2)w_j\rho_j$. A plane wave has coefficients proportional to
$i^\ell J_\ell(q\rho)e^{-i\ell\alpha_q}$, so opposite harmonics have
equal weight. Commutation of the zero-tilt Hamiltonian with rotation through
$2\pi/m$ preserves its rotation eigenspaces, equivalently $\ell$ modulo $m$.
These arguments require the stated symmetry of both potential and discrete
grid. A finite angular sample count is an accuracy bound, not literal exact
alias freedom: Bessel harmonics have infinite support even for a band-limited
plane wave. The ledger's word “alias-free” means negligible tail error in
the tested regime, not a mathematical zero-error theorem.

### Comparison, provenance audit and adjudication

The corrected `_orbital_population` multiplies by the positive carrier
phase before contracting with conjugate orbitals and multiplying by the
grid area. It matches the derived expression, including the squared area
factor in population. The analytic tests correctly distinguish a carrier
change from a physical reciprocal-grid kick. `AxialMCWF.run` normalizes
each surviving state, passes its current carrier, and divides accumulated
observables by `n_electrons`. Its removed histories contribute zero as
required. The five carrier tests and eight documentation checks passed in
this context (13 passed, 13 deselected).

`CylindricalProjection` matches the independently derived Fourier factor,
positive carrier phase, radial Jacobian and harmonic selection;
`angular_momentum` reports the moment in units of $\hbar$. The pre-existing
tests provide analytic LG, plane-wave, derivative-operator and symmetry
anchors. The owner separately reran all three slow coherent W checks
successfully during this review. These are coherent coupling/selectivity
checks, consistent with the pre-existing Decisions in issue #508.

An independent evaluation of the coherent fixture's orbital Gram matrix
gave both diagonal elements 0.999864715 and off-diagonal magnitude below
$2\times10^{-16}$. Thus its unrenormalized cut-orbital overlap has a small
normalization error of $1.35\times10^{-4}$, negligible compared with its
population thresholds (0.9 and 0.02). The dissipative regression explicitly
renormalizes its cut orbitals. The independently evaluated coherent W
energies were $-277.8468$, $-27.75385$, $-27.75385$, and $7.42314$ eV;
the 2p pair is degenerate and lies below the stated zero-energy reference.
This verifies the fixture assumptions and qualitative selectivity, not
quantitative reproduction of the source's Co/Dy examples.

Read-only inspection of `cluster-head` confirmed every population, surviving
fraction and jump count in all four `scratch/decay-*.json` files in Section
9. The uploaded module is actually at
`scratch/test_channeling_axial_dissipator.py`; its SHA-256 matches the
recorded value. The `.pyrite-sync` digest, revision and dirty flag also
match. The submitted script loads `_orbital_population` from that exact
module, builds the stated normalized central-cell orbitals, and uses 1024
histories, seed 11 and 1 Å steps. Its batch script explicitly sets the
snapshot's bundled catalog. Array 1569 logs contain the matching outputs.
No additional heavy computation was run by this verifier.

The differences between 72² and 96² populations are below the reported
individual finite-sampling scales. For bounded continuous trajectory
populations, $\operatorname{Var}(X)\le p(1-p)$, so the binomial expression
is an upper-bound error scale, not an exact standard deviation. The same
seed across grids also prevents interpreting it as an independent
two-sample significance test. The trace/sink differences remain substantial;
the data support stability of this particular small population observable
over those two grids, not convergence of the whole transport model.

Source metadata were checked against the primary arXiv records:
[Xin & Zheng, arXiv:1204.0021](https://arxiv.org/abs/1204.0021) discusses
on-column 2p coupling and coherent persistence; its tabulated elements
are Co and Dy. [arXiv:1111.6050](https://arxiv.org/abs/1111.6050) is
Löffler & Schattschneider, *Electron vortices in crystals*, and describes
position- and depth-dependent angular components in Fe. Earlier “Lubk et
al.” attributions of that identifier are incorrect. Neither source is a
measured W dissipative lifetime reference.

- **Claim:** `cr-vortex-string-coupling` —
  `montecarlo/channeling/vortex.py::{CylindricalProjection,angular_momentum}`
  and the corrected fixed-string population regression — angular
  orthonormality/Parseval and Born-rule projectors; source context above.
- **Filters:** units pass; zero-carrier, subspace and orthogonal-kick limits
  pass; carrier/sign/symmetry conventions pass.
- **Re-derivation:** matches; the former missing positive carrier phase is
  present. Finite angular sampling remains a controlled approximation.
- **Verdict:** `rederived` for the corrected observable and parent
  projection identities within the stated numerical accuracy.
- **Write-up:** `docs/validation/channeling/cr-vortex-string-coupling.md`.
- **Suggested ledger change:** the task owner may advance the parent row
  to `anchored` for projection/symmetry and the issue's explicitly coherent
  W coupling/selectivity scope, citing the analytic and coherent tests.
  Include the five analytic carrier/kick tests and seeded decay regression
  as model anchors; retain sink, finite-grid and source-regime limitations.
  This review resolves the corrected-observable discrepancy and audits the
  recorded decay follow-up. It does not independently validate the TDS or
  electronic kernels, establish a measured lifetime, or certify physical
  scattering survival beyond 50 nm. Those are not the coherent acceptance
  defined in the issue's existing Decisions. Human sign-off remains separate.

There is no additional vortex acceptance blocker under that existing coherent
scope. Other issue claims (notably the Watson eigenproblem review) are outside
this verdict. Only this validation document was edited by the verifier.
