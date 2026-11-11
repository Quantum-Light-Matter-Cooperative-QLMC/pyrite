# Shaped-electron CXR interference: #527 option C

**Status:** Research note for [#1102](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1102) (option C, deferred from [#527](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/527)), 11 November 2026. It changes no production model. The reference calculation is `checks/shaped_electron_1102.py`, ledgered as `quantum-shaped-cxr-interference`. It follows the [unshaped transverse-recoil note](transverse-recoil-527.md).

**Decision: defer.** The prototype stays in `checks/` and no production work is opened. The signature is real and large in the model, but it needs a transverse electron state whose density fringes are registered to the crystal lattice to about 0.1–0.2 Å. No PyRITE beam model represents such a state, and no shaped-electron CXR measurement exists to anchor one. Unshaped beams, mixtures and randomly registered beams reduce exactly to the incoherent per-reflection sum that [#1099](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1099) implements.

Findings:

- **Selection rule.** After the trace over final electrons, two components $\mathbf q_1,\mathbf q_2$ interfere only if they have the same energy and $\mathbf q_{1\perp}-\mathbf q_{2\perp}$ is a transverse reciprocal vector. Otherwise they share no final state, and the cross term is exactly zero.
- **Size.** For Shi's Fig. 5 state (30 keV, 30 nm Si, θ = 110°) the coherent peak is $1.96\times$ the mixture at relative phase $\phi=0$ and $0.06\times$ at $\phi=\pi$. The window-integrated change is $\pm0.95$. This reproduces Shi's "almost twice" (SI §S7) and is close to the Cauchy–Schwarz bound of 2.
- **Phase means position.** $\phi$ is the lateral position of the electron's density fringes, period $2\pi/\lvert\mathbf G_\perp\rvert$ = 1.36 Å, relative to the (040) atom planes. Averaging over registration returns the mixture exactly. Gaussian jitter $\sigma$ multiplies the interference by $\exp(-\lvert\mathbf G_\perp\rvert^2\sigma^2/2)$: 0.90 at 0.1 Å, 0.65 at 0.2 Å and 0.07 at 0.5 Å.
- **Electron Bragg coupling.** Equal energy plus a reciprocal-vector offset is the electron Bragg condition. Shi's two components are exact (040) Bragg partners, and so is each component alone. At 30 keV the Pendellösung distance is 73 nm, so a 30 nm film is not kinematic for the electron. Even so, a two-beam Bloch calculation changes the phase table by ≤ 0.03. At this symmetric condition the coupling preserves the $\cos\phi$ part of the coherence, and the two extremes $\phi=0,\pi$ are the two Bloch eigenstates. An unshaped beam tilted onto the same Bragg condition acquires only an imaginary coherence, so it shows no signature.

## 1. Source

Shi, Wong, Huang, Wong & Kaminer, *Transverse recoil imprinted on free-electron radiation*, Nat. Commun. 15, 7803 (2024), DOI `10.1038/s41467-024-52050-w`. Zotero `Q4I3JISU`; the main-text PDF and the SI were read. The note uses:

- main-text Eq. (3), the coherent sum over the initial amplitude $\varphi(\mathbf q)$;
- the section "Coherently shaped CXR by tailoring electron wavefunctions" and Fig. 5. Their state is $(\lvert\mathbf q_1\rangle+e^{i\phi_{12}}\lvert\mathbf q_2\rangle)/\sqrt2$ with $\mathbf q_1=(2\pi/a,4\pi/a,q_z)$ and $\mathbf q_2=(2\pi/a,-4\pi/a,q_z)$, at 30 keV, observed at θ = 110°, φ = 0. The text says it "can be created by a bi-prism or a phase plate";
- SI Eqs. (S3), (S12)–(S17): Salvat–Yukawa susceptibility and potential, and the PXR and CB elements;
- SI §S6 (finite transverse coherence) and §S7 (coherent against incoherent: peaks "almost twice").

Shi's paper does not discuss how $\phi_{12}$ depends on registration, electron diffraction of the prepared pair, or a tolerance on the momentum offset.

## 2. Shaped-state rate with the final-state trace

Use Shi's reciprocal sign, $\mathbf q_f=\mathbf q_i+\mathbf g-\mathbf k$; PyRITE's `recoil_resonance_eV` takes $-\mathbf g$. The crystal is a static, laterally periodic Si [001] slab of thickness $L$. It conserves energy and transverse momentum exactly; longitudinal momentum is conserved only through the formation integral. Energy coherences between components do not change photon counts (`longitudinal-energy-diagonal-emission`), so all components share the energy $E$.

Label a final electron by its transverse wavevector $\mathbf Q_f$, written before the photon recoil, at energy $E-\hbar\omega$. The amplitude for component $i$ to reach $f$ together with photon $(\mathbf k,\sigma)$ is

$$
A_\sigma(f,i)=\sum_{\mathbf g:\ \mathbf q_{i\perp}+\mathbf g_\perp=\mathbf Q_f}
M_\sigma(\mathbf q_i,\mathbf g)\,F(\Delta_{i\mathbf g}),\qquad
F(\Delta)=\int_0^L e^{i\Delta z}\,dz,
$$

with $\Delta_{i\mathbf g}=q_{iz}+g_z-k_z-q_{fz}$ and $q_{fz}$ on the forward final mass shell. $M_\sigma$ is Shi's spinless PXR + CB element, SI Eqs. (S14)–(S15) with $u_f=u_i$. Replacing $\varphi(\mathbf q_i)\varphi^*(\mathbf q_j)$ in Shi Eq. (3) by a density matrix $\rho_{ij}$, and summing over the unobserved final electron, gives

$$
\frac{d^2N}{d\omega\,d\Omega}\propto\omega\sum_\sigma\operatorname{Tr}\!\left(A_\sigma\,\rho\,A_\sigma^\dagger\right).
$$

Consequences:

1. **Selection rule.** The $i\neq j$ terms need some $f$ reached from both components, $\mathbf q_{i\perp}+\mathbf g_\perp=\mathbf q_{j\perp}+\mathbf h_\perp$, so $\mathbf q_{i\perp}-\mathbf q_{j\perp}$ must lie in the projected reciprocal lattice. Offsets off the lattice share no final state, and the cross term vanishes identically.
2. **Plane-wave limit.** For $\rho=\lvert\mathbf q_0\rangle\langle\mathbf q_0\rvert$ the rate is $\sum_\sigma\sum_{\mathbf g_\perp}\lvert\sum_{g_z}M_\sigma F\rvert^2$. Reflections with the same $\mathbf g_\perp$ and different $g_z$ do reach the same slab final state, but their mismatches differ by $\lvert\Delta g_z\rvert\geq2\pi/a$. Their cross terms are therefore $O(1/(\lvert\Delta g_z\rvert L))$ relative to the lines. Dropping them gives the #527 incoherent per-reflection sum $\sum_{\mathbf g}\lvert M F\rvert^2$. Each line sits at $\Delta=0$, which is the `line-recoil-resonance` root. Shi's infinite-crystal Kronecker delta in $z$ removes these cross terms exactly.
3. **Mixture.** A diagonal $\rho$ gives $\sum_i\rho_{ii}R_i$.
4. **Two components.** For $(\lvert\mathbf q_1\rangle+e^{i\phi}\lvert\mathbf q_2\rangle)/\sqrt2$,

   $$
   R(\phi)=\tfrac12(R_1+R_2)+\operatorname{Re}\!\left(e^{-i\phi}C\right),\qquad
   C=\sum_\sigma\sum_f A_\sigma(f,1)A_\sigma^*(f,2).
   $$

   Cauchy–Schwarz gives $\lvert C\rvert\leq\sqrt{R_1R_2}$, so the coherent rate is at most twice the mixture. The average over $\phi$ is the mixture.
5. **Registration.** Translating the electron by $\mathbf d$ multiplies $\varphi(\mathbf q)$ by $e^{-i\mathbf q\cdot\mathbf d}$, so $\phi\to\phi+\mathbf G_\perp\!\cdot\mathbf d$ with $\mathbf G=\mathbf q_1-\mathbf q_2$. Equivalently, $\lvert\psi\rvert^2\propto1+\cos(\mathbf G_\perp\!\cdot\mathbf r_\perp+\phi)$ is a density grating commensurate with the lattice, and $\phi$ fixes where it sits. Averaging $\mathbf d$ over a cell, or over a beam that is not registered, removes the cross term. Gaussian jitter of width $\sigma$ per axis multiplies it by $\exp(-\lvert\mathbf G_\perp\rvert^2\sigma^2/2)$.

## 3. Electron Bragg coupling

Two components with equal energy and $\mathbf q_1-\mathbf q_2=\mathbf G$ satisfy the electron Bragg condition for $\mathbf G$. Their coincident lines then overlap exactly, since $\Delta_{1\mathbf g}=\Delta_{2\mathbf h}$, and this is also where the crystal diffracts the electron. In the kinematic CB vertex this shows up as the pole $1/(E_{\mathbf q}-E_{\mathbf q+\mathbf G})$. Shi's Fig. 5 components are each on the exact (040) condition. The check drops that one CB term, which belongs to the electron state rather than to the emission vertex, and treats the pair as two-beam Bloch waves:

$$
i\frac{d\psi}{dz}=K\psi,\qquad
K=\begin{pmatrix}q_{1z}&0\\0&q_{2z}\end{pmatrix}
-\frac{1}{\hbar v_z}\begin{pmatrix}0&U_{\mathbf G}\\U_{\mathbf G}^*&0\end{pmatrix},
$$

with the entrance condition $\psi(0)=\psi_{\rm in}$. Each branch emits with its own wavevector $\kappa_b$. The Pendellösung distance is $\xi=\pi\hbar v_z/\lvert U_{\mathbf G}\rvert$. With Shi's Yukawa potential and no Debye–Waller factor, $\lvert U_{040}\rvert=2.78$ eV, so $\xi$ is 73 nm at 30 keV and 173 nm at 300 keV. For $\lvert U_{220}\rvert=4.41$ eV, $\xi$ is 46 nm and 109 nm.

- A mixture $\rho=I/2$ is invariant under the unitary propagation, so the mixture spectrum is unchanged: 0.99× the kinematic one.
- At the symmetric condition ($q_{1z}=q_{2z}$, real $U_{040}<0$) the generator is proportional to $\sigma_x$. It keeps $\operatorname{Re}\rho_{12}$ and rotates $\operatorname{Im}\rho_{12}$ into a population difference. Here $C$ is real, so the $\cos\phi$ interference survives. The extremes $\phi=0$ (lower branch, density on the atom planes) and $\phi=\pi$ (upper branch) are stationary.
- A single plane wave entering at the Bragg tilt starts with $\rho_{12}=0$ and only acquires an imaginary $\rho_{12}$. It therefore produces no $\cos\phi$ signature. Shaping is required; tilting an ordinary beam onto the Bragg condition does not substitute for it.

## 4. Prototype results

All runs use 30 keV, a 30 nm Si [001] slab, θ = 110°, φ = 0 and a 1850–2100 eV window. Reflections cover $\lvert h\rvert,\lvert k\rvert\leq7$ and $\lvert l\rvert\leq6$; only ratios are reported.

| Check | Result |
|---|---|
| Plane wave, normal incidence: exact slab against the #527 per-$\mathbf g$ sum | ≤ 1.3 % at peaks above 20 % of the maximum (≤ 5 % above 5 %); 26 lines, each with $\lvert\Delta\rvert\leq2\times10^{-13}$ Å⁻¹ at its `recoil_resonance_eV` root |
| Plane wave at $\mathbf q_\perp=(1,2)\,2\pi/a$ | ≤ 1.9 % at peaks; 32 lines on their roots |
| Mixture against $(R_1+R_2)/2$ | $3\times10^{-15}$ |
| Coherent at $\phi$ = 0 / 60 / 90 / 120 / 180° (peak, coh/mix) | 1.96 / 1.48 / 1.00 / 0.52 / 0.06 |
| Window-integrated $(R-R_{\rm mix})/R_{\rm mix}$ at the same phases | +0.95 / +0.48 / 0.00 / −0.48 / −0.95 |
| Phase average against the mixture | $3\times10^{-15}$ |
| Visibility $\lvert C\rvert/\tfrac12(R_1+R_2)$ where the mixture exceeds 20 % of its peak | 0.91–0.98 |
| Two-beam Bloch against kinematic, same phases | 1.99 / 1.49 / 1.00 / 0.51 / 0.06 |
| Component offset off the lattice ($\epsilon$ = 0.001 or 0.05 × 2π/a) | 0 shared final states |

**Momentum-offset scan.** Both components are shifted by $d$ along $y$. The difference stays a lattice vector, but the excitation error $s=q_{1z}-q_{2z}$ grows. The table gives the largest $\lvert R(\phi)-R_{\rm mix}\rvert$ over phase, divided by the mixture peak.

| $d$ (2π/a) | common tilt (mrad) | $\lvert s\rvert L$ | $\lvert s\rvert\xi/\pi$ | two-beam | kinematic |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.99 | 0.96 |
| 0.05 | 0.6 | 0.9 | 0.7 | 0.92 | 0.94 |
| 0.1 | 1.3 | 1.8 | 1.4 | 0.75 | 0.86 |
| 0.2 | 2.6 | 3.6 | 2.8 | 0.70 | 0.44 |
| 0.4 | 5.1 | 7.1 | 5.5 | 0.71 | 0.80 |
| 0.8 | 10.3 | 14.3 | 11.1 | 0.16 | 0.15 |

The kinematic column is valid only for $\lvert s\rvert\gg\pi/\xi$; between $\lvert s\rvert\xi/\pi\approx1$ and 3 it differs from the two-beam result by up to 0.26. The interference survives a common tilt of a few mrad at 30 nm. It is suppressed once the coincident lines separate by more than their thickness width, at $\lvert s\rvert L\gtrsim10$.

## 5. Preparation requirements and experiments

A two-component state that shows the signature needs all of the following:

- **Lattice-matched offset.** $\mathbf q_{1\perp}-\mathbf q_{2\perp}=\mathbf G_\perp$ exactly (the selection rule) and equal energies. For Si (040) that means a beam separation of $\lvert\mathbf G_\perp\rvert/k$ = 51 mrad at 30 keV and 14.5 mrad at 300 keV.
- **Registration.** The fringe pattern, period 1.36 Å for (040), must be held against the crystal to $\sigma\lesssim0.1$–0.2 Å for the whole acquisition, keeping 0.9–0.65 of the interference. Unregistered or drifting beams give the mixture.
- **Common tilt.** Within a few mrad at 30 nm, as in the scan above.
- **Transverse coherence.** At least a unit cell and several fringe periods (Shi SI §S6). A sub-cell probe is a many-component state and is not covered by this two-component reference.

Candidate preparations:

- **Biprism or two-hole condenser aperture** (Shi's suggestion). It gives the right kind of state, with coherent tilted plane waves of one energy. The open requirement is sub-0.2 Å registration of the fringe pattern to the specimen over the counting time. That needs active drift correction against the lattice, which Shi does not discuss.
- **Convergent STEM probe.** A probe with semi-angle at least $\lvert\mathbf G_\perp\rvert/(2k)$ (26 mrad at 30 keV, 7 mrad at 300 keV) contains lattice-matched pairs. Its registration is the probe position, which atomic-resolution STEM already controls. The observable would be position-resolved CXR, meaning a CXR spectrum that changes as the probe moves within the unit cell. This is the most practical route, but it needs a many-beam Bloch calculation, which is out of scope here.
- **Laser or nanofabricated phase plates** (Kapitza–Dirac, PINEM, holographic gratings). Their momentum offsets are $10^{-3}$–$10^{-2}$ Å⁻¹, far below $\lvert\mathbf G_\perp\rvert\approx3$–5 Å⁻¹. They are not lattice vectors, so the cross term is exactly zero after the trace.
- **An upstream crystal.** It produces lattice-matched pairs, but their phase is registered to that crystal, not to the target, unless both are one piece. With a single piece the setup is ordinary dynamical diffraction, and §3 shows that an unshaped entry gives no $\cos\phi$ signature.

**Experiments.** No shaped-electron CXR measurement exists; this was the #527 decision record. The existing PXR/CBS line-shift data (Huang 2023; Shentcis 2020) use unshaped beams with EDS. Nothing in the literature PyRITE tracks combines a registered biprism state, or a registered probe, with resolved CXR. Such an experiment is possible in principle in an aberration-corrected STEM with a large-solid-angle detector, but it is not planned anywhere we know of. The binding constraints would be registration and the bremsstrahlung and characteristic backgrounds (Si Kα and the K edge lie near the 30 keV comb), not counting statistics.

## 6. Recommendation

**Defer.** Keep `checks/shaped_electron_1102.py` as the reference; do not open implementation work.

- PyRITE has no beam model for transverse states registered to the lattice, and transport would destroy such states after the first elastic scattering. At 30 keV the elastic mean free path is about the film thickness (#527 note §3.1).
- No data exist to anchor a shaped-electron prediction, and the sub-MeV per-reflection intensities that set $\lvert C\rvert$ are still unvalidated (#527 note §5.1).
- If the topic is reopened, the right formalism is Bloch-wave (many-beam) emission from a registered probe, extending the #508 entry states. A biprism-specific plane-wave model is the wrong starting point. Triggers to reopen: a position-resolved CXR proposal or measurement, or Bloch-state CXR work that needs the vertex in §2.

## 7. Reproduce

```bash
uv run python checks/shaped_electron_1102.py
```

The script runs in about 25 s on one CPU core and prints all numbers in §3–§4. Its amplitude model is Shi's: spinless, $u_f=u_i$, Salvat–Yukawa Si without Debye–Waller, and no X-ray dynamical diffraction. Absolute rates are not computed.
