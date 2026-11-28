# `cr-submev-axial-coherence-width` re-derivation

Claim: `checks/channeling_submev_w001_linewidth.py::{coherence_decay,tds_gain,electronic_gain,loss_rates}`.
Recorded in the {ref}`W ⟨001⟩ linewidth section <sec-submev-w001-linewidth>`.
Research record for #585; no production axial line shape exists.

Fresh context, 2026-11-16, commit `f58a36929`. The derivation below was
written from the ledger row, the `cr-line-shape` write-up and the
`axial_dissipator` module docstring, before the check's function bodies
were read.

## Secular coherence equation with local jumps

Every jump of `cr-axial-realspace-dissipator` is a multiplication operator
$L_k(\mathbf r)$ with rate measure $d\mu_k$. The large-angle sink $S(\mathbf r)$
only removes. Per unit depth,

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}[H,\rho]
+\int d\mu_k\Bigl(L_k\rho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\rho\}\Bigr)
-\tfrac12\{S,\rho\}.
$$

In the position basis the kernel is
$C(\mathbf r,\mathbf r')=\int d\mu_k\,L_k(\mathbf r)L_k^*(\mathbf r')$, and the
no-jump part is $-\tfrac12[C(\mathbf r,\mathbf r)+S(\mathbf r)+(\mathbf r\to\mathbf r')]$.
Take eigenstates $\lvert i\rangle$ (upper, 2p) and $\lvert f\rangle$ (lower, 1s)
and $\sigma=\langle f\vert\rho\vert i\rangle$. The term $L\rho L^\dagger$ gives
$\sum_{mn}\langle f\vert L\vert m\rangle\rho_{mn}\langle i\vert L\vert n\rangle^*$.
Keeping only couplings of equal frequency (secular) and only the pair
$(m,n)=(f,i)$ itself (justified below),

$$
\frac{d\sigma}{dz}=\Bigl[i\frac{\Omega_{if}}{v}-\frac{\Gamma_i+\Gamma_f}{2}
+\int d\mu_k\,\langle L_k\rangle_f\langle L_k\rangle_i^*\Bigr]\sigma,
\qquad
\Gamma_n=\Bigl\langle n\Bigl\vert \int d\mu_k L_k^\dagger L_k+S\Bigr\vert n\Bigr\rangle .
$$

Because $L_k$ and $S$ are multiplicative,
$\Gamma_n=\int\lvert w_n\rvert^2[C(\mathbf r,\mathbf r)+S(\mathbf r)]\,d^2r$,
i.e. the full no-jump rate (in-grid TDS, sink, electronic) of the
`total_rate` property, and $\langle L\rangle_n=\int\lvert w_n\rvert^2L\,d^2r$.
The decay rate is therefore

$$
\boxed{\kappa=\frac{\Gamma_i+\Gamma_f}{2}-\mathrm{Re}\,J,\qquad
J=\int d\mu_k\,\langle L_k\rangle_f\langle L_k\rangle_i^*}
$$

and $\mathrm{Im}\,J$ is a line shift. The sink has no $L\rho L^\dagger$ term, so
it enters only $\Gamma_n$. Conjugating $J$ (the ledger writes
$A_iA_f^*$) changes only the sign of the shift, not $\kappa$.

**Which density.** For a flat band, every Bloch block $k$ carries the same
frequency, so all inter-block transfers are secular (`cr-line-shape`). With
an on-site dipole $d_0$, the radiated amplitude is
$\sum_k a^{(k)*}\sigma^{(k)}\propto\sum_k\langle f_k\vert\rho\vert i_k\rangle
=\sum_{\mathbf R}\langle f,\mathbf R\vert\rho\vert i,\mathbf R\rangle$, the
on-site Wannier coherence. Flat band means the Wannier states are
eigenstates. In the string-localized limit
$\langle n,\mathbf S\vert L\vert n,\mathbf R\rangle\approx\delta_{\mathbf{SR}}
\int\lvert w_n(\mathbf r-\mathbf R)\rvert^2L$. The on-site coherence then
closes on itself with the single-site densities $\lvert w_n\rvert^2$. That is
the ledger's "block-to-block transfer sums to the single-site Wannier
overlap product". A Bloch-state expectation $\langle L\rangle$ in one block
would instead scale as $1/N$ per source, and $J\to0$. That is the
per-block error recorded in `cr-line-shape`.

**The 2p doublet.** $\sigma_{2p_x,1s}$ and $\sigma_{2p_y,1s}$ are degenerate,
so the secular equation also couples them through
$\int d\mu\,\langle 2p_y\vert L\vert 2p_x\rangle\langle L\rangle_{1s}^*$ and
$\langle 2p_y\vert C+S\vert 2p_x\rangle$. The string site is $C_{4v}$, the TDS
displacement is isotropic and the electronic azimuth is uniform. So the
ensemble-averaged generator commutes with $C_{4v}$ on the 2D irrep $E$, and by
Schur's lemma it is a multiple of the identity there. The off-diagonal
couplings vanish, and any orthonormal basis of the doublet gives the same
$\kappa$. The record's identical $2p_x$ and $2p_y$ rows (to $10^{-15}$)
confirm this.

## The two gains

**TDS.** $L_{s,\mathbf R,\mathbf u}=\phi_{\mathbf u}-\langle\phi\rangle$ is real,
with $d\mu=\lambda_s\,p(\mathbf u)\,d^2u$ and
$p=\mathcal N(0,a_s^2\mathbf 1)$, $a_s^2$ per axis. So

$$
J_\mathrm{TDS}=\sum_s\lambda_s\,E_{\mathbf u}\bigl[\langle L_{\mathbf u}\rangle_i\langle L_{\mathbf u}\rangle_f\bigr]
$$

is real. The sum runs over every string, not just the home string. Both
densities are centrosymmetric about their string. So the first-order term
$-\mathbf u\cdot\nabla\phi$ of $L$ averages to zero over each density, and
for the home string $J_\mathrm{TDS}=O(\lambda a^4)$, while
$\Gamma=O(\lambda a^2)$. A near-zero TDS gain is expected: the displaced
atom's dipolar kick changes parity, so it moves amplitude out of the pair
rather than carrying the pair coherence.

**Electronic.** $L_{\mathbf q}=\sqrt{w\rho_q}\,e^{i\mathbf q\cdot\mathbf r}$ with
$d\mu=w(q)\,d^2q$. Put the site density at $\mathbf R$. Then
$\langle L_{\mathbf q}\rangle_n=\sqrt{w}\,e^{i\mathbf q\cdot\mathbf R}A_n(\mathbf q)$, with
$A_n=\int\lvert w_n\rvert^2\sqrt{\rho_q}\,e^{i\mathbf q\cdot(\mathbf r-\mathbf R)}$.
The factor $e^{i\mathbf q\cdot\mathbf R}$ cancels in
$\langle L\rangle_f\langle L\rangle_i^*$. The phase convention is irrelevant,
provided both densities share the same $\mathbf R$. With nodes $q_j$ and
weights $W_j\approx\int_{\rm panel}2\pi q\,dq\,w$ (so $\int d^2q\,w$),
$\int d^2q=\int q\,dq\int d\varphi$ gives

$$
J_\mathrm{el}=\sum_j W_j\,\overline{A_i(q_j,\varphi)A_f^*(q_j,\varphi)}^{\,\varphi},
\qquad
\Gamma_n^\mathrm{el}=\sum_j W_j\int\lvert w_n\rvert^2\rho_{q_j}.
$$

There is no extra $2\pi$: the azimuthal $2\pi$ is in $W_j$, and the average
divides by it. $q\to0$ gives $A_n\to\langle\sqrt{\rho_0}\rangle_n$. That
cancels the loss only when $\rho$ is uniform over both densities, which is the
test's construction.

**Bounds.** By Cauchy–Schwarz in $\mu$ and then in $\lvert w_n\rvert^2$
($\lvert\langle L\rangle_n\rvert^2\le\langle\lvert L\rvert^2\rangle_n$),

$$
\lvert J_\mathrm{mech}\rvert\le\sqrt{\Gamma_i^\mathrm{mech}\Gamma_f^\mathrm{mech}},
\qquad
\kappa\ge\tfrac12\bigl(\sqrt{\Gamma_i}-\sqrt{\Gamma_f}\bigr)^2\ge0 .
$$

For TDS the sharp bound uses the in-grid part only, because the jumps are
the band-limited phases. Including the sink, as the code does, is looser but
still valid.

**Widths.** $\lvert\sigma\rvert\propto e^{-\kappa z}=e^{-\kappa vt}$, so the
transverse Lorentzian HWHM is $\hbar v\kappa$. The emitted phase advances as
$\Omega t(1-\beta\cos\theta)$, so the forward ($\theta=0$) photon HWHM is
$\hbar v\kappa/(1-\beta)$ (`cr-line-shape`).

**Units.** $L$ is a phase, dimensionless. $\lambda_s$ is in Å⁻¹ and $W_j\rho_q$
in Å⁻¹. $\kappa$ is in Å⁻¹ and $\hbar v\kappa$ in eV.

## Comparison with the implementation

| Item | Derivation | Code | Result |
| --- | --- | --- | --- |
| $\kappa$ | $\tfrac12(\Gamma_i+\Gamma_f)-J_\mathrm{TDS}-\mathrm{Re}J_\mathrm{el}$ | `coherence_decay`: `half_loss - j_tds - j_el`, `total = sum(losses)` | match |
| $\Gamma_n$ | $\int\lvert w_n\rvert^2(C_\mathrm{grid}+S+C_\mathrm{el})$ | `loss_rates`: einsum with `cell_area_ang2`, densities normalized to $\sum\cdot\Delta A=1$ | match |
| sink | loss only | absent from both gains | match |
| TDS measure | $\mathcal N(0,a^2\mathbf 1)$, $a^2$ per axis | `hermegauss` weights (sum $\sqrt{2\pi}$) outer product $/2\pi$; nodes $\times\sqrt{a^2}$; `tds_phase` mean $e^{-a^2k^2/2}$ is the same per-axis variance | match |
| TDS sources | every string and image | loop over all `tds_sources` | match |
| $J_\mathrm{TDS}$ | $\lambda E_u[\langle L\rangle_i\langle L\rangle_f]$ | `src.linear_density * einsum("p,pm,pn->mn", w2, means, means)` | match |
| $W_j$ | $\int d^2q\,w$ per node | `_electronic_nodes`: radial $2\pi q\,dq$ panels; `electronic_rate` $=\sum_jW_j\rho_{q_j}$ (`axial_dissipator`, same nodes) | match |
| $J_\mathrm{el}$ | $\sum_jW_j\overline{A_iA_f^*}$ | `weight / n_azimuth * outer(amp, amp.conj())`, midpoint azimuth, wave centred on `centre` | match |
| densities | single-site $\lvert w_n\rvert^2$ about one string | `AxialLineCatalog.orbitals`: $k=0$ Bloch functions cut to the string's Wigner–Seitz cell, renormalized | match (Wannier approximation) |
| widths | $\hbar v\kappa$; $/(1-\beta)$ | `hbar_v * kappa`; `2 * hwhm / (1 - beta)` | match |

## Limiting-case tests

`tests/check/test_channeling_submev_w001_linewidth.py` uses a fake
dissipator.

- **Rigid kick.** $L=u_x$ on every point gives
  $J=\lambda E[u_x^2]=\lambda a^2$, equal to the loss. This tests the
  Gauss–Hermite normalization and the $\sqrt{a^2}$ scaling exactly, as well
  as the cancellation. Valid.
- **Loss only.** A non-uniform sink and zero phase give
  $\kappa=\tfrac12(\Gamma_i+\Gamma_f)$ and zero gain. This tests that the sink
  carries no gain. Valid.
- **Soft electronic kick.** $q=10^{-6}$ with uniform $\rho$ gives
  $A_n=\sqrt\rho$ and $J=W\rho=\Gamma$. Valid.
- **Hard electronic kick.** $q=40$ Å⁻¹ lies above the test grid's Nyquist
  ($12.6$ Å⁻¹), so $e^{iqx}$ aliases to about 10 Å⁻¹. That still suppresses
  $A_n$ for these densities, so the test passes, but it does not test a
  resolved hard kick (minor).

## Numerical spot check (300 keV)

I reran 300 keV with my own probe (`/tmp`, not committed), importing the
check's functions. The production values reproduce the record
`w001-linewidth.json` exactly: $\kappa=0.044052$ Å⁻¹,
$\Gamma_{1s}=0.07859$ Å⁻¹, $\Gamma_{2p}=0.01416$ Å⁻¹.

| Test | Value | Note |
| --- | --- | --- |
| $J_\mathrm{TDS}$, Gauss–Hermite 20 vs 30 | $-1.23963\times10^{-4}$ both | converged |
| $J_\mathrm{TDS}$, independent Monte Carlo $\mathbf u$ (4000 per source) | $-1.215\times10^{-4}$ | agrees within MC noise |
| $J_\mathrm{el}$, azimuth 16 vs 32 | $2.44491$ vs $2.44550\times10^{-3}$ | $2\times10^{-4}$ relative |
| $J_\mathrm{el}$ without nodes above grid Nyquist (35 Å⁻¹; 0.23 % of weight) | $2.44524\times10^{-3}$ | aliasing negligible |

The table and bullets of `sec-submev-w001-linewidth` check against the
record. FWHM is 239, 157 and 135 eV. The forward width is 356, 347 and
604 eV. The saturation estimate is 158, 132 and 122 eV; I recomputed the
30 keV value by hand (158.36 eV). $\Gamma_{1s}$ is 85–91 % of the total, the
sink is 30–33 % of $\Gamma_{1s}$, the electronic gain is 4.8–5.3 % of
$\kappa$, and electronic-only FWHM is 4.7–10.9 eV.

## Caveats

- **Secular and flat-band.** These are stated honestly, including that the
  secular approximation is marginal when $\hbar v\kappa$ is comparable to the
  gap (FWHM/gap 0.54–1.05). The neglected
  $\langle n,\mathbf S\vert L\vert n,\mathbf R\rangle$ inter-site transfers are of
  tunnelling size for these strongly bound orbitals.
- **Sink as pure loss.** Its true gain needs $\lvert w_n\rvert^2$ Fourier content
  above the grid band (35 Å⁻¹), which is negligible. Treating it as loss is
  sound.
- **Saturation estimate.** It scales only the losses and leaves the gains
  linearized. This is self-consistent only because $J_\mathrm{TDS}\approx0$.
  It is labelled a sensitivity estimate, not adopted, and the conclusion
  does not rest on it. Acceptable.
- **Label mismatch (wording).** The doc (line 825) and the ledger Notes say
  the "per-atom TDS phase variance seen by 1s" is 1.0 rad² at 30 keV
  (0.40, 0.23). Those numbers are the record's `per_atom_tds_loss["1s"]`:
  all TDS loss, in-grid plus sink, from every string, divided by $\lambda$.
  The record field that the check docstring (line 33) calls the phase
  variance, `per_atom_tds_phase_variance_1s_rad2`, is 0.70, 0.27 and 0.15:
  the home string's in-grid $\mathrm{Var}_u\phi/\lambda$. The inference
  (the kernel is outside its small-phase range at 30 keV) holds either way.
  The quoted quantity should be named correctly.
- **Doc reason for $J_\mathrm{TDS}\approx0$** (line 822: "1s and 2p see the
  displaced atom's phase differently"). The sharper statement is parity: the
  linear displacement term averages to zero over each centrosymmetric
  density, leaving $O(a^4)$.
- **Conclusion.** Under the MeV-calibrated kernels, the 2p→1s line is
  unresolved at 30–300 keV: a Lorentzian FWHM of 0.5–1.05 of the gap, and
  0.49–0.70 even with the saturation scaling. The conclusion is dominated by
  $\Gamma_{1s}$, which is grid-independent (the full diagonal). The
  conclusion is conditional on the kernels, as the doc states. It is not a
  validated sub-MeV linewidth.

## Verdict

`rederived`. The boxed $\kappa$, $J$ definitions, signs, measure
normalization, $2\pi$ placement, density, phase convention, bounds and width
conversions match. The findings are wording only. No implementation change
is needed.
