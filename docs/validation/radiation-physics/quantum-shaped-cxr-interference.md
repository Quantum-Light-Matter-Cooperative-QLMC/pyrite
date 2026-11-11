# Shaped-electron CXR interference (Shi 2024, #1102 / #527 option C)

Validation: `quantum-shaped-cxr-interference`.

## Scope and independence

This is a fresh-context verification at `40e4df9fc` on branch
`issue-1102-shaped-electron-pxr`. The order of reading was:

1. The ledger row and the source: Shi, Wong, Huang, Wong & Kaminer,
   Nat. Commun. 15, 7803 (2024). From the main text: Eq. (3), Methods Eqs.
   (4)–(7), the section "Coherently shaped CXR by tailoring electron
   wavefunctions" and the Fig. 5 caption. From the SI: Eqs. (S3), (S9),
   (S12)–(S17), §S6 and §S7.
2. §1–§6 below, derived before opening the implementation. The numbers in §6
   come from the validator's own script, which builds the Yukawa $U_{\mathbf G}$,
   the Pendellösung distance, the geometry and the Ewald excitation errors from
   Shi's SI Eq. (S3) parameters without importing PyRITE.
3. Only after that: `checks/shaped_electron_1102.py`, which was run in 45 s,
   with rc 0. Then the note `docs/research/physics/shaped-electron-cxr-1102.md`
   §1–§7, and the docstrings of the two reused functions
   `materials/recoil.py::recoil_resonance_eV` and
   `quantum/_longitudinal.py::slab_formation_amplitude`.

The claim is measurement-only, so no production code is under review. The
validator edited nothing except this file. Its scratch scripts sit outside the
repository.

## 1. Rate with an explicit trace over final electrons

**Setup.** The crystal is static, laterally periodic and occupies
$0<z<L$. Its interaction is $\sum_{\mathbf g}V_{\mathbf g}e^{i\mathbf g\cdot\mathbf r}$
inside the slab. The electron state is a density matrix
$\rho=\sum_{ij}\rho_{ij}\lvert\mathbf q_i\rangle\langle\mathbf q_j\rvert$ over
on-shell plane waves of a single energy $E$.

**Conservation laws.**

- The static potential conserves energy. A final electron therefore has
  $E_f=E-\hbar\omega$.
- Lateral periodicity conserves transverse momentum modulo $\mathbf g_\perp$:
  $\mathbf q_{f\perp}=\mathbf q_{i\perp}+\mathbf g_\perp-\mathbf k_\perp$.
  This is Shi's sign, $\mathbf q_i+\mathbf g=\mathbf q_f+\mathbf k$ (Eq. (3),
  Methods Eq. (6)).
- The slab breaks $z$ translation. The $z$ Kronecker delta of Shi Eq. (3)
  becomes the overlap integral

$$
F(\Delta)=\int_0^L e^{i\Delta z}\,dz=L\,e^{i\Delta L/2}\operatorname{sinc}\!\frac{\Delta L}{2},
\qquad \Delta_{i\mathbf g}=q_{iz}+g_z-k_z-q_{fz}.
$$

**Final-state label.** Fix the photon $(\mathbf k,\sigma)$. The final electron
is then fully labelled by $\mathbf Q_f\equiv\mathbf q_{i\perp}+\mathbf g_\perp$
(that is, $\mathbf q_{f\perp}+\mathbf k_\perp$). Its $q_{fz}$ is the forward
root of $E_f(\mathbf q_{f\perp},q_{fz})=E-\hbar\omega$. The backward root has
$\lvert\Delta\rvert\sim2k$ and is negligible.

**Amplitude.** First-order PXR plus second-order CB, as in Shi SI (S14)–(S15)
with $u_f=u_i$, gives the amplitude of component $i$ into the final state $f$:

$$
A_\sigma(f,i)=\sum_{\mathbf g:\ \mathbf q_{i\perp}+\mathbf g_\perp=\mathbf Q_f}
M_\sigma(\mathbf q_i,\mathbf g)\,F(\Delta_{i\mathbf g}).
$$

**Rate.** The unobserved final electron is traced out. A pure state
$\sum_ic_i\lvert\mathbf q_i\rangle$ gives
$\sum_{f,\sigma}\lvert\sum_ic_iA_\sigma(f,i)\rvert^2$. Linearity in
$c_ic_j^*\to\rho_{ij}$ then gives

$$
\frac{d^2N}{d\omega\,d\Omega}\propto\omega\sum_\sigma\sum_f\sum_{ij}A_\sigma(f,i)\,\rho_{ij}\,A^*_\sigma(f,j)
=\omega\sum_\sigma\operatorname{Tr}\!\left(A_\sigma\rho A_\sigma^\dagger\right).
$$

**Prefactor.** Shi (S17) carries $\omega^2/(c^3(2\pi)^3)$, and each $M'$
carries $\sqrt{\hbar/(2\varepsilon_0\omega V)}$ from SI Eq. (S9). That leaves
the net $\omega$ in the ledger row.

**Final-state density of states.** Converting the $q_{fz}$ sum with the energy
delta adds a weight $1/(\hbar v_{fz})$ per final state $f$. The ledger form
omits it. The weight is positive and common to every $ij$ term of a given $f$,
so neither the selection rule nor the Cauchy–Schwarz bound depends on it. I
inserted $\sqrt{v/v_{fz}}$ into the amplitudes. This moved the Fig. 5 peak
ratios from 1.95946 / 0.05941 to 1.95940 / 0.05944, which is negligible.

## 2. Lattice selection rule

A cross term $i\ne j$ needs a common final state $f$. Final electrons with
different energies are orthogonal, so $E_i=E_j$ is required. A common $f$ also
needs $\mathbf q_{i\perp}+\mathbf g_\perp=\mathbf q_{j\perp}+\mathbf h_\perp$,
which means

$$
\mathbf q_{i\perp}-\mathbf q_{j\perp}\in\Lambda^*_\perp
=\{\mathbf g_\perp:\ \mathbf g\ \text{an allowed reflection}\}.
$$

In a laterally infinite crystal this is a Kronecker condition. If it fails,
$A(f,i)A^*(f,j)=0$ for every $f$, and the cross term vanishes identically, not
merely on average.

For Si [001], allowed $(hkl)$ are all-odd, or all-even with
$h+k+l\equiv0\pmod 4$. The projection therefore contains every $(h,k)$ with
$h\equiv k\pmod 2$ (for even pairs, choose $l$ to fix the sum mod 4). Shi's
offset $\mathbf q_1-\mathbf q_2=(0,4,0)\,2\pi/a$ belongs to this set.

## 3. Plane-wave limit and the finite slab

For $\rho=\lvert\mathbf q_0\rangle\langle\mathbf q_0\rvert$, the trace gives

$$
R=\omega\sum_\sigma\sum_{\mathbf g_\perp}\Bigl\lvert\sum_{g_z}M_\sigma F(\Delta_{\mathbf g})\Bigr\rvert^2 .
$$

**Diagonal part.** This is the #527 incoherent per-reflection sum
$\omega\sum_{\sigma,\mathbf g}\lvert MF\rvert^2$. Each line peaks where
$\Delta_{\mathbf g}=0$, that is
$\lvert\mathbf q_0+\mathbf g-\mathbf k\rvert$ on the $E-\hbar\omega$ shell. This
is Shi's exact Methods Eq. (6).

**Cross terms.** Same $\mathbf g_\perp$, different $g_z$, at a common $q_{fz}$
gives $\Delta_{\mathbf g}-\Delta_{\mathbf g'}=g_z-g'_z$. On the Si [001]
lattice, the $g_z$ steps at fixed $\mathbf g_\perp$ are $2\cdot2\pi/a$ (odd
$\mathbf g_\perp$) or $4\cdot2\pi/a$ (even). So
$\lvert\Delta g_z\rvert\ge4\pi/a=2.31$ Å⁻¹. Where line $\mathbf g$ peaks,
$\lvert F(\Delta_{\mathbf g'})\rvert\le2/\lvert\Delta g_z\rvert$, and

$$
\frac{\lvert\text{cross}\rvert}{\text{line}}\le
\frac{4}{L}\sum_{g'_z}\frac{\lvert M_{\mathbf g'}\rvert}{\lvert M_{\mathbf g}\rvert\,\lvert\Delta g_z\rvert}.
$$

This is $O(1/(\lvert\Delta g_z\rvert L))$, with a coefficient set by the
relative reflection strengths. The $\operatorname{sinc}$ tail oscillates as
$\sin(\Delta g_zL/2)$, so this is an envelope bound, not a monotone decrease.

**Thickness scan.** I evaluated $\text{exact}/\text{incoherent}-1$ at the root
of every line stronger than 20 % of the strongest, for normal incidence,
30 keV, θ = 110° and a 1850–2100 eV window. The maximum is 1.9 % at
$L=30$ nm, 3.6 % at 60 nm, 2.6 % at 120 nm, 1.3 % at 240 nm and 0.9 % at
480 nm. The worst line is $(\bar1\bar1\bar3)$ from 60 nm on. These values sit
under an envelope of about $45\ \text{Å}/L$, which is about
$100/(\lvert\Delta g_z\rvert L)$. The coefficient is large because weak lines
have strong same-$\mathbf g_\perp$ partners.

The ledger's "≤ 1.9 % at peaks, 30 keV, 30 nm" is therefore a correct
measurement for that thickness. Thicker films are not uniformly closer to the
incoherent sum until $L\gtrsim200$ nm.

**Line centres.** I wrote an independent root finder for Shi Eq. (6),
$\mathbf q_f=\mathbf q+\mathbf g-\mathbf k$, with $q=(1,2,q_z)\,2\pi/a$. It
agrees with `recoil_resonance_eV(T, u, -g, n)` to $5\times10^{-11}$ eV for
$(00\bar4)$, $(11\bar3)$, $(\bar1\bar1\bar3)$, $(20\bar2)$ and $(31\bar5)$.
The $-\mathbf g$ sign conversion in the check is correct. For $(0\bar40)$,
PyRITE returns a root of $1.3\times10^{-14}$ eV, which is the
degenerate-Bragg $\omega\to0$ root. My bracket search starts at 1 eV and
reports none. This is consistent, and the root lies outside every window.

## 4. Mixture, two components, phase law and bound

**Mixture.** For a diagonal $\rho$,
$\operatorname{Tr}(A\rho A^\dagger)=\sum_i\rho_{ii}\sum_f\lvert A(f,i)\rvert^2$,
so the rate is $\sum_i\rho_{ii}R_i$.

**Two components.** Take $c=(1,e^{i\phi})/\sqrt2$, so that
$\rho_{12}=c_1c_2^*=e^{-i\phi}/2$. The off-diagonal pair gives
$2\operatorname{Re}(\rho_{12}\sum A_{f1}A_{f2}^*)$, hence

$$
R(\phi)=\tfrac12(R_1+R_2)+\operatorname{Re}\!\left(e^{-i\phi}C\right),\qquad
C=\omega\sum_{\sigma,f}A_\sigma(f,1)A_\sigma^*(f,2).
$$

**Bound.** Cauchy–Schwarz on the $(\sigma,f)$ sum, at a fixed photon
$(\omega,\hat{\mathbf n})$, gives $\lvert C\rvert\le\sqrt{R_1R_2}\le(R_1+R_2)/2$.
Pointwise, $R(\phi)/R_{\rm mix}\in[0,2]$.

**Phase average.** The $\phi$ average removes the cross term, leaving the
mixture.

**Consistency of the peak values.** Pointwise,
$R(0)/R_{\rm mix}+R(\pi)/R_{\rm mix}=2$ exactly. The ledger's 1.96 and 0.06
sum to 2.02. This is not a contradiction, because each is the spectral
**maximum** of its own curve over the window, and the maxima need not sit at
the same photon energy. The window-integrated values $\pm0.95$ are linear and
sum to zero, as they must.

## 5. Registration law and Gaussian jitter

Translating the electron by $\mathbf d$, $\psi(\mathbf r)\to\psi(\mathbf r-\mathbf d)$,
multiplies $c_i$ by $e^{-i\mathbf q_i\cdot\mathbf d}$. Then
$\rho_{12}\to\rho_{12}e^{-i(\mathbf q_1-\mathbf q_2)\cdot\mathbf d}$. The
components share $q_z$, so only the transverse part enters, and
$\phi\to\phi+\mathbf G_\perp\cdot\mathbf d$ with
$\mathbf G_\perp=\mathbf q_{1\perp}-\mathbf q_{2\perp}$.

Let $\mathbf d$ be Gaussian with variance $\sigma^2$ per Cartesian axis. Then
$\langle e^{-i\mathbf G_\perp\cdot\mathbf d}\rangle=\exp(-\lvert\mathbf G_\perp\rvert^2\sigma^2/2)$.
If $\sigma$ were instead the 2D radial rms, the exponent would be
$\lvert\mathbf G_\perp\rvert^2\sigma^2/4$. The note states "per axis" and the
check samples each axis with `normal(0, sigma)`, so the two are consistent.

With $\lvert\mathbf G_\perp\rvert=4\cdot2\pi/a=4.628$ Å⁻¹, the factor is
0.898 at 0.1 Å, 0.652 at 0.2 Å and 0.069 at 0.5 Å. The fringe period is
$2\pi/\lvert\mathbf G_\perp\rvert=a/4=1.358$ Å.

## 6. Electron-Bragg status and the two-beam claims

**Shi's pair is an exact Bragg pair.** Shi's Fig. 5 / SI §S7 components are
$\mathbf q_{1,2}=(1,\pm2,q_z)\,2\pi/a$. They have equal $\lvert\mathbf q\rvert$
and $\mathbf q_1-\mathbf q_2=(040)$, an allowed reflection with $S=8$ (real at
the atom origin). Therefore $\lvert\mathbf q_2+\mathbf G_{040}\rvert=\lvert\mathbf q_2\rvert$:
the pair is an exact electron Bragg pair, and each component alone is at the
$(0\bar40)$ / $(040)$ condition.

**The CB pole.** In Shi's CB vertex (S15), the denominator
$E_{\mathbf q_i}-E_{\mathbf q_i+\mathbf g}$ vanishes for $\mathbf g=\pm(040)$.
Shi does not discuss this pole.

**Independent numbers.** These use Shi's Yukawa parameters,
$U_{\mathbf G}=-\frac{4\pi e^2}{4\pi\varepsilon_0}\frac{Z}{a^3}\sum_kC_k/((\mu_k/a_0)^2+G^2)\,S_{\mathbf G}$,
with no Debye–Waller factor.

| Quantity (30 keV) | Validator | Script / note |
|---|---|---|
| $k$ | 90.029 Å⁻¹, $\beta=0.32838$ | — |
| $\lvert U_{040}\rvert$ | 2.780 eV | 2.78 eV |
| $\xi_{040}=\pi\hbar v_z/\lvert U\rvert$ | 73.19 nm ($L/\xi=0.41$) | 73.2 nm, 0.41 |
| $\lvert U_{220}\rvert$, $\xi_{220}$ | 4.411 eV, 46.1 nm | 4.41 eV, 46 nm |
| tilt of each component | 1.647° = 28.7 mrad | — |
| angle between components | 2.945° = 51.4 mrad | 51 mrad |
| same at 300 keV ($k=319.1$ Å⁻¹) | 14.5 mrad | 14.5 mrad |
| STEM semi-angle $\lvert\mathbf G_\perp\rvert/(2k)$ | 25.7 / 7.25 mrad | 26 / 7 mrad |
| single-entry transfer $\sin^2(\pi L/\xi)$ | 0.92 | 0.92 |

**Two-beam derivation.** Fixed energy, beams $\{\mathbf q_1,\mathbf q_2\}$, and
$q_{1z}=q_{2z}$. The relativistic $z$-propagation is
$i\,d\mathbf c/dz=[q_zI-H/(\hbar v_z)]\mathbf c$ with
$H=\begin{pmatrix}0&U_{\mathbf G}\\U^*_{\mathbf G}&0\end{pmatrix}$, after
dropping the common $U_0$. From $(E-U)^2=\hbar^2c^2k^2+m^2c^4$ one gets
$\delta k_z=-U/(\hbar v_z)$.

- **Pendellösung distance.** Eigenvalues $q_z\pm\lvert U\rvert/(\hbar v_z)$
  give the beat period $\xi=\pi\hbar v_z/\lvert U_{\mathbf G}\rvert$.
- **Preserved coherence.** Real $U_{040}=-2.78$ eV makes the generator
  proportional to $\sigma_x$. This rotates the Bloch vector about $x$, which
  preserves $\langle\sigma_x\rangle=2\operatorname{Re}\rho_{12}$ and rotates
  $(\operatorname{Im}\rho_{12},\rho_{11}-\rho_{22})$.
- **Eigenstates.** $\phi=0$ and $\phi=\pi$ are the eigenstates
  $(\lvert1\rangle\pm\lvert2\rangle)/\sqrt2$. With $U<0$, the symmetric state
  ($\phi=0$) has density $\cos^2(2\pi y\cdot2/a)$, maximal on the (040) atom
  planes at $y\in\{0,a/4,\dots\}$. It has the larger $k_z$, which is the more
  bound branch.
- **Bloch populations.** For a general $\phi$, the Bloch-mode populations are
  $(1\pm\cos\phi)/2$. They are set by $\operatorname{Re}\rho_{12}$ alone, and
  are therefore conserved.
- **Mixture.** $\rho=I/2$ is invariant under any unitary propagation. The
  mixture spectrum can still change, because the two branches emit at mismatches
  shifted by $\mp\lvert U\rvert/(\hbar v_z)$. Here
  $2\kappa L=2\pi L/\xi=2.58$ rad is below $2\pi$, so the change is small.
- **Single-component entry.** With $\mathbf c(0)=(1,0)$,
  $\mathbf c(z)=(\cos\alpha,-i\sin\alpha)$ for $\alpha=\lvert U\rvert z/(\hbar v_z)$.
  So $\rho_{12}(z)=i\cos\alpha\sin\alpha$ is purely imaginary and
  $\operatorname{Re}\rho_{12}=0$ is preserved. With no prepared phase there is
  no $\cos\phi$ term at all.
- **Effect of Im ρ12.** The imaginary coherence contributes
  $2\operatorname{Re}(\rho_{12}C)=-2\operatorname{Im}\rho_{12}\operatorname{Im}C$,
  which vanishes only where $C$ is real.
- **Why $C$ is real at the measured azimuth.** At $\phi_{\rm det}=0$ (photon in
  the $xz$ plane), the operation $y\to-y$ followed by the translation
  $(\tfrac14,\tfrac14,\tfrac14)a$ maps diamond onto itself (it swaps the
  sublattices). It maps $\mathbf q_1\leftrightarrow\mathbf q_2$ and leaves the
  detection geometry invariant up to a polarization sign. The phase it
  generates, $e^{-i\mathbf G_{040}\cdot\mathbf t}=e^{-2\pi i}=1$, is trivial. So
  $R(\phi)=R(-\phi)$ and $\operatorname{Im}C=0$. Numerically,
  $\max\lvert\operatorname{Im}C\rvert/\lvert C\rvert=5.7\times10^{-5}$ over the
  window, and $\arg C=-0.007°$ at $\phi_{\rm det}=30°$, 1950 eV.

**Neighbouring beams.** The nearest other zeroth-Laue-zone beams of $\mathbf q_2$
are $(220)$ and $(\bar220)$, landing at $(3,0)$ and $(-1,0)$. They have
$s=\pm0.0297$ Å⁻¹ against a coupling $\lvert U\rvert/(\hbar v)=0.0068$ Å⁻¹, so
they are excited at the ~10–20 % amplitude level. Their second-order (Bethe)
corrections to the effective $U_{040}$ come with opposite signs of $s$ and
cancel at this symmetric setting. The two-beam reduction is reasonable, and the
ledger excludes many-beam effects.

## 7. Comparison with the implementation

| Item | Independent | `checks/shaped_electron_1102.py` | Agreement |
|---|---|---|---|
| Rate | $\omega\sum_\sigma\operatorname{Tr}(A\rho A^\dagger)$ | `w * einsum("sfi,ij,sfj->", m, rho, m.conj())` | yes; omits $1/v_{fz}$ (§1, effect $6\times10^{-5}$) |
| Final label | $\mathbf q_{i\perp}+\mathbf g_\perp$ at $E-\hbar\omega$ | `labels = round((q[:2] + g[:, :2]) / KEY)` | yes |
| Mismatch | $q_{iz}+g_z-k_z-q_{fz}$, forward root | `delta = q[2] + kz_shift + g[:, 2] - kvec[2] - qfz` | yes |
| $F$ | $\int_0^Le^{i\Delta z}dz$ | `slab_formation_amplitude` docstring and body | yes |
| PXR vertex | $\chi_{-\mathbf g}^*\,\boldsymbol\beta\cdot[k^2I-(\mathbf k-\mathbf g)(\mathbf k-\mathbf g)]\cdot\mathbf e/(\lvert\mathbf k-\mathbf g\rvert^2-k^2)$ | `chi * pxr / denom` with `chi_g(hkl)` | yes ($\chi_{-\mathbf g}^*=\chi_{\mathbf g}$ for real $\chi(\mathbf r)$) |
| CB vertex | $U_{\mathbf g}\,\boldsymbol\beta\cdot\mathbf e\,[1/(E_i-E_{\mathbf q+\mathbf g})+1/(E_f-E_{\mathbf q-\mathbf k})]$, same prefactor $ice T/\hbar$ as PXR | `pot * (beta @ e) * (d1 + d2)` | yes, including the relative sign |
| $\chi_{\mathbf g}$, $U_{\mathbf g}$ | Shi (S3), $e<0$, $U=e\varphi$ attractive | `chi_g`, `potential_g` | yes |
| Shi sign | $\mathbf q_f=\mathbf q_i+\mathbf g-\mathbf k$; PyRITE takes $-\mathbf g$ | `recoil_resonance_eV(T30, u, -HKL * B0, N110)` | yes |
| CB pole | belongs to the electron state | `exclude=((0,4,0),(0,-4,0))` | yes |
| Two-beam $K$ | $q_zI-H/(\hbar v_z)$, $\hbar v_z=\hbar^2c^2q_z/E$ | `TwoBeam`, `hv = HBARC**2 * qz / E` | yes; branches with `kz_shift` |
| Jitter | $\exp(-\lvert\mathbf G_\perp\rvert^2\sigma^2/2)$, $\sigma$ per axis | `rng.normal(0, sigma, (N, 2))` | yes |

**Number reconciliation.** I compared every number in the ledger row and in
note §3–§4 with the run output. All of them agree:

- the 1.3 % / 5 % / 26 lines and 1.9 % / 32 lines rows, with
  $\lvert\Delta\rvert\le1.8\times10^{-13}$ Å⁻¹;
- the mixture and phase-average residuals $2.8$ and
  $3.1\times10^{-15}$;
- the phase table 1.96 / 1.48 / 1.00 / 0.52 / 0.06 and $\pm0.95$ / $\pm0.48$;
- visibility 0.91–0.98;
- two-beam 1.99 / 1.49 / 1.00 / 0.51 / 0.06, and the mixture at
  0.99× kinematic;
- the offset-scan table, including the tilt column $d\cdot2\pi/(ak)$ = 0.64 /
  1.29 / 2.57 / 5.14 / 10.28 mrad and the "up to 0.26" difference;
- zero shared final states off the lattice;
- jitter 0.974 / 0.898 / 0.652 / 0.381 / 0.069;
- the $\xi$ values 73 / 122 / 155 / 173 nm and 46 / 77 / 98 / 109 nm.

Two cosmetic points. The script defines the `# --- parts ---` constants block
twice. Note §7 gives a run time of "about 25 s"; this run took 45 s.

## 8. Wording review

### Ledger row

- **W1, Notes.** "Shi's 'almost twice' (SI §S7) is reproduced" overstates the
  match. SI §S7 compares coherent and incoherent **angular maps at 2.6 keV**.
  The check computes peak spectral ratios at θ = 110°, $\phi_{\rm det}=0$,
  1850–2100 eV, which is Fig. 5c geometry. A ratio near 2 also follows
  generically from near-saturated Cauchy–Schwarz (visibility 0.91–0.98). The
  correct statement is "consistent with", not "reproduced".
- **Shi's own inconsistency.** Shi is itself inconsistent: SI §S7 says
  "Fig. 4b" and 2.6 keV, while main-text Fig. 5 is at 2.0 keV. This is worth a
  clause.

### Note §5, "Preparation requirements and experiments"

- **W2, misattribution.** "Biprism or two-hole condenser aperture (Shi's
  suggestion)". Shi's text is "can be created by a bi-prism or a phase plate in
  electron microscopy". The two-hole condenser aperture is not Shi's
  suggestion. Note §1 quotes Shi correctly, and §5 contradicts that quote.
- **W3, unflagged disagreement and an overclaim.** The phase-plate bullet
  rejects the second preparation Shi names, without saying that it disagrees
  with Shi.
  - The biprism bullet says the biprism "gives the right kind of state" and
    that the "open requirement" is registration. The state also needs a
    specimen-plane crossing angle equal to the 51 mrad component separation
    (14.5 mrad at 300 keV). No source is cited for a biprism reaching that
    angle at the specimen, so that is a second open requirement, not a given.
  - A two-hole condenser aperture with a focused beam is a two-beam
    convergent probe. Its registration is the probe position, exactly as in
    the STEM bullet. So the "needs active drift correction" contrast applies
    to the biprism, not to the aperture.
- **W4, wrong reason for PINEM.** "Laser or nanofabricated phase plates
  (Kapitza–Dirac, PINEM, holographic gratings). Their momentum offsets are
  $10^{-3}$–$10^{-2}$ Å⁻¹ … They are not lattice vectors". PINEM sidebands
  are **energy** offsets ($n\hbar\omega$), with a longitudinal, not
  transverse, momentum change. They are excluded by the equal-energy condition
  (§2; `longitudinal-energy-diagonal-emission`), not by the transverse
  selection rule. The conclusion (no cross term) stands; the reason given is
  wrong for PINEM.
- **W5, unsupported judgement.** "This is the most practical route" (STEM
  probe) is an unsupported judgement. "A candidate route" would be accurate.

### Other note wording (non-blocking)

- **W6, Findings bullet.** "An unshaped beam tilted onto the same Bragg
  condition … shows no signature" is too broad. Its spectrum does change
  dynamically (92 % Pendellösung transfer, branch splitting). What it lacks is
  the $\cos\phi$ / registration signature, which is how §3 and the ledger
  phrase it.
- **W7, note §3.** "Here $C$ is real" needs a qualifier: real by the
  $y$-glide symmetry at $\phi_{\rm det}=0$, and numerically
  ($\lvert\operatorname{Im}C\rvert/\lvert C\rvert<10^{-4}$). The note gives
  no reason for it.
- **W8, note §3.** "The mixture spectrum is unchanged: 0.99× the kinematic
  one" is self-contradictory. The mixture **density matrix** is invariant;
  the spectrum changes slightly through the branch shifts.
- **W9, note §2.2.** "$\lvert\Delta g_z\rvert\ge2\pi/a$" is true but loose.
  The tight lattice bound is $4\pi/a$. Also, the $O(1/(\lvert\Delta g_z\rvert L))$
  residual is an oscillating envelope with a large coefficient (§3:
  3.6 % at 60 nm), so "≤ 1.9 %" should stay tied to 30 nm.
- **W10, note §7.** "prints all numbers in §3–§4" is accurate. The §5
  numbers (51 / 14.5 mrad, 26 / 7 mrad, 1.36 Å) are hand-derived; they are
  verified in §6 here.

## 9. Verdict

- **Claim**: `quantum-shaped-cxr-interference` — measurement only,
  `checks/shaped_electron_1102.py` against
  `materials/recoil.py::recoil_resonance_eV` and
  `quantum/_longitudinal.py::slab_formation_amplitude` — Shi et al. (2024)
  Eq. (3), Fig. 5, SI (S3), (S12)–(S17), §S6–§S7.
- **Filters**:
  - units: pass. The rate carries net $\omega$, PXR and CB vertices are
    dimensionless relative to each other, and $\xi$ is in nm.
  - limits: pass. A plane wave gives the incoherent per-reflection sum, a
    mixture gives $\sum\rho_{ii}R_i$, the phase average gives the mixture, and
    large $\sigma$ and off-lattice offsets give no interference.
  - signs/conventions: pass. Shi $+\mathbf g$ is converted to PyRITE
    $-\mathbf g$, verified against an independent Eq. (6) solver to
    $5\times10^{-11}$ eV; $\rho_{12}=e^{-i\phi}/2$; the translation law gives
    $\phi\to\phi+\mathbf G_\perp\cdot\mathbf d$; $U_{040}<0$ is real.
  - reference wording: **failure**, items W1 and W2.
- **Re-derivation**: `matches` for all the physics. This covers the
  final-state trace, the selection rule, the plane-wave reduction with its
  measured 1.9 % residual, the mixture, the phase law and the bound, the
  registration and jitter, the (040) Bragg status, $\xi=73$ nm, preservation of
  $\operatorname{Re}\rho_{12}$ and the absence of a $\cos\phi$ term for a
  single-component entry. Every number in the row and the note reproduces. The
  only divergence is the omitted $1/v_{fz}$ weight, which has an effect of
  $6\times10^{-5}$ and changes no claim.
- **Verdict**: `discrepancy`, confined to wording. First divergent item: ledger
  *Notes* "Shi's 'almost twice' (SI §S7) is reproduced". SI §S7 is a 2.6 keV
  angular-map comparison, and the check computes a different observable. Then
  note §5 "(Shi's suggestion)" for the two-hole condenser aperture, where Shi
  wrote "bi-prism or a phase plate". After the text fixes below, the expected
  verdict is `rederived`.
- **Write-up**: `docs/validation/radiation-physics/quantum-shaped-cxr-interference.md`
- **Suggested ledger change** (the owner applies it; it is not a sign-off):
  - *Notes*: replace "Shi's 'almost twice' (SI §S7) is reproduced, not
    digitized" with "consistent with Shi's 'almost twice' (SI §S7, 2.6 keV
    angular maps; Fig. 5 is at 2.0 keV), which is not the same observable and
    not digitized".
  - Link this write-up in *Notes*. Add `quantum-shaped-cxr-interference` to
    the toctree in `docs/validation/radiation-physics/index.md`.
  - Optionally add to *Checks*: "rate omits the $1/v_{fz}$ final density of
    states (effect $6\times10^{-5}$ on the Fig. 5 ratios)".
  - Status: `unverified` until the wording is fixed, then `rederived`.
- **Note fixes**:
  - **§5 first bullet:** "Biprism (Shi) or two-hole condenser aperture." Add
    the 51 mrad specimen-plane crossing angle as a requirement. State that the
    two-hole aperture's registration is the probe position.
  - **§5 phase-plate bullet:** say that it disagrees with Shi's "phase plate"
    suggestion, and that PINEM is excluded by the energy condition.
  - **§5 STEM bullet:** "most practical" → "a candidate".
  - **Findings bullet:** "no signature" → "no $\cos\phi$ (registration)
    signature".
  - **§3:** add the $C$-real qualifier, and change "spectrum unchanged" to
    "density matrix unchanged; spectrum 0.99×".
  - **§2.2:** $4\pi/a$.
  - **§7:** run time.

### Re-check after owner corrections (2026-11-11)

The owner applied the §9 corrections without committing them. The same fresh
context re-checked only the corrected text (`git diff HEAD`) and edited only
this write-up.

- **Ledger row, *Notes*.**
  - The row now says "consistent with" Shi's "almost twice". It records that
    SI §S7 shows 2.6 keV angular maps while Fig. 5 is at 2.0 keV, so the
    observable differs. This resolves W1.
  - It records the $1/v_{fz}$ omission ($6\times10^{-5}$), the 30 nm scope of
    the 1.9 % residual (3.6 % at 60 nm), and a link to this write-up.
  - One bookkeeping point remains. *Notes* says "the status stays
    `rederived`" while the *Status* field still reads `unverified`. The owner
    should set *Status* to `rederived` together with this verdict.
- **Note, Findings.**
  - "Size" now calls the values spectral maxima at different photon energies.
    It explains why they need not sum to 2, and states the observable
    difference with SI §S7.
  - The Bragg bullet now reads "no $\cos\phi$ (registration) signature" (W6).
- **Note §2.2.** The bound is now $\lvert\Delta g_z\rvert\ge4\pi/a$. The
  thickness residuals (1.9 / 3.6 / 2.6 / 0.9 % at 30 / 60 / 120 / 480 nm) and
  the $1/(\hbar v_{fz})$ omission ($6\times10^{-5}$) match §1 and §3 here
  (W9).
- **Note §3.**
  - "Density matrix unchanged; spectrum 0.99×" (W8).
  - $C$ real is now attributed to the $y$-mirror plus
    $(\tfrac14,\tfrac14,\tfrac14)a$ glide at $\phi_{\rm det}=0$, with
    $\lvert\operatorname{Im}C\rvert/\lvert C\rvert\le6\times10^{-5}$ (W7).
- **Note §5.**
  - **Biprism bullet.** It is now attributed to Shi, the aperture is not, and
    it lists both open requirements: the 51 / 14.5 mrad specimen crossing and
    the sub-0.2 Å registration. It identifies the focused two-hole aperture as
    a probe-registered two-beam probe. This resolves W2 and W3.
  - **STEM bullet.** It now reads "a candidate route" (W5).
  - **Phase-plate bullet.** It flags the disagreement with Shi's "a phase
    plate" and excludes PINEM by the equal-energy condition (W3, W4).
- **Note §7.** The run time now reads "25–45 s".
- **Script.** The duplicate constants block is removed. A rerun gives rc 0,
  with output byte-identical to the first run (§7).
- **Toctree.** `quantum-shaped-cxr-interference` is listed in
  `docs/validation/radiation-physics/index.md`.

**Final verdict: `rederived`.** The derivation, the source reads and the
corrected claim, note and check agree. The owner applies the status; human
sign-off remains pending (#277).
