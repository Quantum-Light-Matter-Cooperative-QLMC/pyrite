# Quantum-coherent CXR emission from shaped and entangled transverse states

**Status:** Research and design note for [#501](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/501), 15 October 2026, at `8f988e5e`. No production model, persisted channel or default-output change. It builds on the #56 channeling engine ([ADR-0018](../../adr/0018-quantum-channeling-zone.md), [channeling walkthrough](channeling-radiation-physics.md)). Entry states come from the #508 interface ([Interface requirements](#interface-requirements)). Classical microbunched superradiant PXR stays in [superradiant PXR](superradiant-pxr.md).

The question is whether shaped or entangled free-electron transverse states change crystal X-ray emission (CXR: PXR and CBS, plus channeling radiation, CR) in ways a classical-trajectory kernel cannot represent, and in which regimes. In short:

- A single electron's spontaneous rate cannot depend on its wavefunction shape unless a guard-rail mechanism is present. Every candidate effect below names one.
- Transverse coherence over $\Delta x \approx 1$ Å survives only about **0.07 µm at 30 keV** and **0.4–0.7 µm at ≥ 300 keV** in Si(110). Over even the shortest PXR/CBS formation length ($1/\lvert g\rvert$, 0.2–0.3 Å) the decoherence is negligible ($\lesssim 10^{-3}$). The constraint that matters is how much path a coherent effect needs.
- In bulk Si(110), propagation erodes a shaped state's enhancement of local incoherent emission to factors of order 1–3 within a micron. The planar overlap factor can never exceed 9.8. Nothing here supports a bulk $10^3$ bremsstrahlung gain.
- **Go (research only):** transverse-recoil-resolved CXR in thin films. **No-go:** entangled-pair super/subradiance without an engineered source; Dicke CR at keV energies in planar Si/diamond; the Shatnev phase-locked harmonic gain.

Every quantitative statement below carries its source, assumptions, units and a limiting case. The three research claims are ledgered: `quantum-decoherence-budget` and `channeling-modulated-incoherent-emission` in the [channeling ledger](../../validation/ledger-channeling.md), and `quantum-pair-population-normalization` in the [core coherent-physics ledger](../../validation/ledger-core-coherent-physics.md).

Sources were inspected in this order: the Zotero library (Wong 2024, Karnieli 2021 PRL and Genz 1996 records), Europe PMC open-access full text (Wong 2024; Shi 2024), and arXiv abstract and HTML pages (Shatnev; Zhang et al.; Karnieli 2021 Sci. Adv.). Items marked *(verify against Zotero)* are not yet in the library. Only their abstracts or quoted equations were read.

## 1. Guard-rail

**Statement.** In first-order perturbation theory with an initially empty photon field, a single free electron's spontaneous photon emission rate into any channel is independent of the shape of its wavefunction, provided the interaction is translation-invariant and the final electron states are traced over. Zhang, Ianconescu, Friedman, Scheuer, Tokman, Pan and Gover (arXiv:2401.05978, abstract) state: "the spontaneous emission rate of photons by free electrons, unlike stimulated emission, is independent of the shape or modulation of the quantum electron wavefunction". The same paper finds that the emitted photon *state* (off-diagonal photon density matrix) does depend on the shape. Karnieli et al. (Sci. Adv. 7, eabf8096, 2021; arXiv:2011.00623) tie the light's *coherence* (for example the Cherenkov shockwave duration) to the electron's coherent momentum uncertainty. Their result concerns photon correlations, not the rate. *(Both: verify against Zotero.)*

**Why, in operator form.** Let $\hat M$ be the first-order emission operator of one photon channel ($\mathbf k$, polarization, frequency), acting on the electron. The emitted photon number is $n = \operatorname{Tr}(\rho\,\hat M^\dagger\hat M)$. It is linear in $\rho$. In a translation-invariant medium $\hat M$ maps momentum $\mathbf p$ to $\mathbf p-\hbar\mathbf k$, so $\hat M^\dagger\hat M$ is diagonal in momentum. Then $n=\sum_{\mathbf p}\rho(\mathbf p,\mathbf p)\lvert m(\mathbf p)\rvert^2$ depends only on the momentum distribution, not on phases. A shaped state with the same momentum spectrum emits identically.

Assumptions: one photon, first order in $e$; vacuum initial field; final electron states summed. Units: $n$ is dimensionless per mode, or carries the channel's spectral-yield units through $\lvert m\rvert^2$. Limiting case: in the harmonic-ladder test of #505 (`test_harmonic_coherent_state_is_classical_and_has_no_phase_locked_gain`), a coherent and a fully dephased state with the same populations radiate identically. Their ratio is 1 to $10^{-12}$.

**Allowed mechanisms.** Shape dependence of a measured rate needs at least one of:

1. **(M1) position-dependent interaction.** A crystal or atom breaks translation invariance, so $\hat M^\dagger\hat M$ acquires momentum off-diagonal parts (lattice momenta $\hbar\mathbf g$). Examples: overlap with atomic planes, channeling, ALCHEMI-type emission.
2. **(M2) recoil-resolved or postselected final states.** The final electron states are not traced over, or are resolved: transverse-recoil pathways, coincidence detection.
3. **(M3) multi-particle correlations.** The pair term $\langle\hat M_a^\dagger\hat M_b\rangle$ of §4: classical bunching, or entanglement.

**Linearity rule.** Coherent effects enter only through observables linear in $\rho$ (or in the reduced two-body $\rho^{(2)}$). Squaring per-trajectory MCWF field amplitudes and then summing across trajectories is not such an observable. Neither is adding amplitudes that lead to orthogonal final states (§8). Every "enhancement" reported below names its mechanism.

## 2. Operator mapping onto the #56 engine

| Object | Definition | #56 owner |
| --- | --- | --- |
| Transverse state | translation-averaged $\rho$, block-diagonal in Bloch $\kappa_r=rg/n_{\rm cells}$, each block in the $H_\perp(\kappa_r)$ eigenbasis | `DensityMatrixSolver` (`cr-density-matrix-split-operator`) |
| Incoherent jumps | Lindblad kernel $C(x,x')=\sum_k L_k(x)L_k^*(x')$ for TDS and electronic excitation | `tds_kernel`, `electronic_kernel` (`cr-lindblad-jump-operators`) |
| Emission operator | $\langle f\rvert(\mathbf v\cdot\boldsymbol\epsilon^*/c)\,e^{-iqx}\lvert i\rangle$ between Bloch states at $k$ and $k-q$; the dipole limit is $q\to0$ | `transition_operator`, `first_order_transition` (`cr-dipole-rate`) |
| Local incoherent emission | $\operatorname{Tr}(\rho\,T_n)$ with $T_n[h,h']=n_{h-h'}$ a density Toeplitz matrix | check-local prototype (§5) |
| Two-body state | reduced $\rho^{(2)}$ on the product space; the pair term $\operatorname{Tr}[\rho^{(2)}(\hat M^\dagger\otimes\hat M)]$ | not implemented; §4 and §7 |

The Lindblad equation of the engine, per unit depth $z$ in the paraxial limit, is

```{math}
:label: eq-qce-lindblad

\frac{\partial\rho(x,x')}{\partial z}
=-\frac{i}{\hbar v}[H_\perp,\rho](x,x')
+\Bigl[C(x,x')-\tfrac12C(x,x)-\tfrac12C(x',x')\Bigr]\rho(x,x').
```

Here $x$ is the transverse coordinate normal to the planes [Å], $v$ the speed and $C$ [1/Å] the kernel of `cr-lindblad-jump-operators`.

A d-periodic transverse state with fixed registration to the lattice is a single Bloch block ($\kappa=0$) and survives the translation average unchanged. A finite or non-periodic shaped probe must be averaged over its registration and becomes block-diagonal. Free (above-barrier) Bloch bands are in the same basis up to the subspace cutoff. The planar continuum model, however, averages over the in-plane directions and over $z$. Reciprocal vectors with components along the beam ($g_\parallel$) or in the plane are absent. This gap matters for §6.

## 3. Decoherence budget

### 3.1 Derivation

`Validation: quantum-decoherence-budget` — research claim; status in the ledger (see the [channeling ledger](../../validation/ledger-channeling.md)).

Take a coherence $\rho(x,x+\Delta)$. The dissipator of {eq}`eq-qce-lindblad` multiplies it by the local rate $\tfrac12C(x,x)+\tfrac12C(x+\Delta,x+\Delta)-C(x,x+\Delta)$. The real part of that rate damps the coherence and the imaginary part adds a phase. For a translation-averaged beam, the cell average gives

```{math}
:label: eq-qce-gamma

\Gamma(\Delta)
=\Bigl\langle\tfrac12C(x,x)+\tfrac12C(x+\Delta,x+\Delta)-\operatorname{Re}C(x,x+\Delta)\Bigr\rangle_x
=\tfrac12\sum_k\bigl\langle\lvert L_k(x)-L_k(x+\Delta)\rvert^2\bigr\rangle_x\;\ge0 .
```

The second form follows from $C=\sum_kL_kL_k^*$ and $\lvert a-b\rvert^2=\lvert a\rvert^2+\lvert b\rvert^2-2\operatorname{Re}ab^*$. Only the $G=0$ part of $C$ survives the cell average. With the #504 jump operators this gives closed forms, as implemented in `decoherence_rate`:

- TDS (Einstein model): $\Gamma_{\rm TDS}=\sum_s\frac{n_s}{(\hbar v)^2}\frac1{2\pi}\int q\,dq\,v_s(q)^2\bigl(1-e^{-a_s^2q^2}\bigr)\bigl(1-J_0(q\Delta)\bigr)$;
- electronic (LDA): $\Gamma_{\rm el}=2\pi\int q\,dq\,w(q)\bigl[\bar\rho-J_0(q\Delta)\langle s_q(x)s_q(x+\Delta)\rangle_x\bigr]$.

The symbols are those of `cr-lindblad-jump-operators`: $v_s$ is the Mott–Bethe potential, $a_s^2=B_s/8\pi^2$, $w$ the PENELOPE-2024 $q_\perp$-resolved GOS, and $s_q=\sqrt{\rho_q}$.

Limits:

- $\Gamma(0)=0$.
- $\Gamma(\Delta\to\infty)$ is the mean loss rate $\langle C(x,x)\rangle$ when $C(x,x+\Delta)\to0$.
- Small $\Delta$: $\Gamma\simeq\tfrac12D\Delta^2$, logarithmically corrected by the Coulomb $1/q^4$ tail. $D=d\langle q_x^2\rangle/dz$ [Å⁻³] is the transverse momentum-diffusion rate, so $d\langle\theta_x^2\rangle/dz=D/k^2$ with $k=p/\hbar$.
- The TDS rate scales exactly as $C\propto1/v^2$. The electronic rate does to within 1.4 % between 30 keV and 3 MeV ($\beta^2\Gamma_{\rm el}(1\,\text{Å})$ = 8.53e-5 and 8.65e-5 Å⁻¹). Both are therefore nearly energy independent as $\beta\to1$.

**Budget criterion.** With $H_\perp$ switched off (or for components degenerate in $E_\perp$), each $\lvert\rho(x,x+\Delta;z)\rvert$ decays at its local rate $\Gamma_{\rm loc}(x,\Delta)$, whose cell average is $\Gamma(\Delta)$. By Jensen's inequality the cell-averaged surviving coherence is $\ge e^{-\Gamma(\Delta)z}$. A coherent effect that needs a superposition across $\Delta$ over a path $L$ is suppressed by about $e^{-\Gamma(\Delta)L}$, so it is confined to a length of order $\ell(\Delta)=1/\Gamma(\Delta)$; this is conservative in the no-$H_\perp$ case. With $H_\perp$ on, the coherent dynamics move the separation, and $\ell$ is an order-of-magnitude scale, not a bound. The exact bound-state coherence decay comes from the density solver. Assumptions are those of the #504 kernels: first Born approximation, Einstein phonons, quasi-elastic TDS, LDA for the electronic term, and energy loss not fed back.

### 3.2 Tabulation

`checks/quantum_decoherence_budget.py` evaluates `decoherence_rate` at 293.15 K and runs in a few seconds. Values for Si(110), $d=1.920$ Å, with TDS + electronic; $\ell=1/\Gamma$:

| $T$ | $\Gamma_{\rm TDS}(1\,\text{Å})$ [Å⁻¹] | $\Gamma_{\rm el}(1\,\text{Å})$ | $\ell(d/2)$ | $\ell(1\,\text{Å})$ | $n_b$ at $k=0$ | $L_{\rm CR}=\hbar v/\Delta E_{10}$ | $\Gamma(d/2)L_{\rm CR}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 5.97e-4 | 7.91e-4 | 736 Å | 721 Å | 1 | — | — |
| 300 keV | 1.07e-4 | 1.43e-4 | 0.41 µm | 0.40 µm | 1 | — | — |
| 3 MeV | 6.57e-5 | 8.84e-5 | 0.66 µm | 0.65 µm | 2 | 166 Å | 0.025 |
| 30 MeV | 6.43e-5 | 8.65e-5 | 0.68 µm | 0.66 µm | 5 | 295 Å | 0.044 |

- **Saturation.** $\Gamma_{\rm TDS}$ reaches 90 % of its saturated value by $\Delta\approx0.3$ Å, about four times the 0.077 Å thermal amplitude, and saturates by 1 Å: one TDS event fully decoheres any larger separation. $\Gamma_{\rm el}$ keeps rising to $\Delta\sim10^2$ Å, set by the plasmon localization length $\sim v/\omega_p$. For example, at 3 MeV $\Gamma_{\rm el}$ is 3.88e-4 at 100 Å and 4.20e-4 Å⁻¹ at 1000 Å.
- **PXR/CBS formation length.** The phase-matched line from reciprocal vector $g$ has $\omega(1-\beta\cos\theta)=vg_\parallel$, so its formation (phase-slip) length is $L_f=1/g_\parallel\ge1/\lvert g\rvert$. The shortest case, $1/\lvert g\rvert$, is 0.31 Å for Si(110) and 0.20 Å for diamond(110). There $\Gamma L_f\le1.3\times10^{-3}$ in every cell, and $\le10^{-3}$ for $\Delta\le3$ Å. Geometries with small $g_\parallel$ have proportionally longer $L_f$, scaling $\Gamma L_f$ by $\lvert g\rvert/g_\parallel$. For direct incidence, a single PXR/CBS formation length does not decohere. The relevant length is the coherent path an effect accumulates over, bounded by $\ell$.
- **CR formation length.** Over one CR formation length the coherence decays by only 2.5–4.4 % at MeV energies. No bound-bound CR exists in planar Si(110) or diamond(110) at 30 or 300 keV: there is only one bound state at $k=0$.
- **Multiple-scattering consistency.** The log-sensitive effective $D_{\rm eff}=2\Gamma/\Delta^2$ at $\Delta=0.03$ Å gives $\theta_{\rm rms}(1\,\mu\text{m})=6.8$ mrad at 3 MeV and 0.77 mrad at 30 MeV. The Highland formula (PDG; Si $X_0=9.37$ cm) gives 7.3 and 0.82 mrad. The full-tail $\Gamma''(0)$ with $q_{\max}=10^4$ Å⁻¹ is larger: TDS alone gives 11.8 mrad at 3 MeV. So the −7 % agreement depends on the choice of $\Delta$, and 1 µm is $x/X_0\approx10^{-5}$, below Highland's stated validity range. It is an order-of-magnitude consistency check, not an anchor.

Diamond(110), $d=1.261$ Å: the electronic term now uses the sourced 34 eV diamond valence oscillator{cite:p}`bhattacharya2025`, through the production kernel without an override. The [material survey](channeling-materials.md) records its scope and remaining model limits.

| $T$ | $\Gamma_{\rm TDS}(1\,\text{Å})$ | $\Gamma_{\rm el}(1\,\text{Å})$ | $\ell(d/2)$ | $n_b$ |
| --- | --- | --- | --- | --- |
| 30 keV | 2.21e-4 | 1.52e-3 | 831 Å | 1 |
| 300 keV | 3.96e-5 | 2.75e-4 | 0.46 µm | 1 |
| 3 MeV | 2.44e-5 | 1.69e-4 | 0.75 µm | 1 |
| 30 MeV | 2.39e-5 | 1.66e-4 | 0.76 µm | 4 |

**Order-of-magnitude comparison (not an anchor).** {cite:t}`genz1996` deduce, through their Eq. 19 $\Gamma_{\rm in}=2\gamma^2\hbar c/l_{\rm coh}$, line coherence lengths for diamond(110) 1→0 of 0.53–0.74 µm at 5.2 MeV and 0.68 µm at 9.0 MeV (Table VI). They also find that it "does not change with energy". The diamond-input $\ell(d/2)$ = 0.75–0.76 µm at 3–30 MeV varies little with energy and is of the same order. The observables differ: their quantity is a bound-state line coherence, ours a cell-averaged position coherence. This does not validate linewidths.

**Not computable** (`ChannelingDataError`, no channeling thermal data): the HOPG, hBN, MoS₂ and WSe₂ (001) cells at 30 keV. These crystals do have conduction-band shell data. Graphite's displacements are also strongly anisotropic, so the catalog's basal-only `B_ang2` is not an adequate stand-in for a thermal covariance. No values are given for these cells (follow-up F4).

### 3.3 Reading the budget

At 30 keV, the regime of the Zhai/PyRITE table-top CXR targets, a transverse superposition across about 1 Å survives only $\ell\approx0.07$ µm in Si: 0.13 µm at 0.3 Å, 0.054 µm at 1.92 Å. Shape-dependent effects that need that superposition are confined to the entrance layer, a fraction $\ell/L$ of the emitting path $L$. At ≥ 300 keV, $\ell$ is 0.4–0.7 µm. These are bulk statements: monolayer targets, such as those in Wong 2024, do not face this budget.

## 4. Generalized physical-population pair term

`Validation: quantum-pair-population-normalization` — research claim; status in the ledger (see the [core coherent-physics ledger](../../validation/ledger-core-coherent-physics.md)).

Take $N$ electrons with state $\rho_N$ and one photon channel. The first-order one-photon amplitude operator is $\hat A=\sum_{a=1}^N\hat M^{(a)}$, where $\hat M^{(a)}$ acts on electron $a$. The photon number is $\operatorname{Tr}(\rho_N\hat A^\dagger\hat A)$, which splits into reduced density matrices:

```{math}
:label: eq-qce-pair

\langle n\rangle
=\sum_a\operatorname{Tr}\bigl[\rho^{(1)}_a\hat M^\dagger\hat M\bigr]
+\sum_{a\ne b}\operatorname{Tr}\bigl[\rho^{(2)}_{ab}\bigl(\hat M^\dagger\otimes\hat M\bigr)\bigr].
```

**Separable i.i.d. limit.** If $\rho^{(2)}_{ab}=\rho\otimes\rho$ for every ordered pair, then

```{math}
:label: eq-qce-iid

\langle n\rangle=N\operatorname{Tr}(\rho\hat M^\dagger\hat M)+N(N-1)\bigl\lvert\operatorname{Tr}(\rho\hat M)\bigr\rvert^2 .
```

Here $\rho$ is a classical mixture $\sum_jp_j\lvert j\rangle\langle j\rvert$ of trajectory wavepackets with $\hat M\lvert j\rangle=\mathcal E_j\lvert j'\rangle$. Then $\operatorname{Tr}(\rho\hat M^\dagger\hat M)=\langle\lvert\mathcal E\rvert^2\rangle$ exactly. However, $\operatorname{Tr}(\rho\hat M)=\sum_jp_j\mathcal E_j\langle j\vert j'\rangle$ equals $\langle\mathcal E\rangle$ only when $\langle j\vert j'\rangle\approx1$. That needs negligible recoil and a wavepacket localized below $1/k$ along $\mathbf k$; otherwise the overlap is the single-electron form factor. Within that point-charge idealization, {eq}`eq-qce-iid` becomes `spxr-physical-population`, with $N=Q/\lvert e\rvert$ ([superradiant PXR](superradiant-pxr.md)).

**Quantum content.** $\operatorname{Tr}(\rho\hat M)$ needs coherence of $\rho$ between the states $\hat M$ connects: an initial state and its recoiled final state. For a free electron this is the density modulation (bunching) of the wavefunction at the photon wavevector, the "modulated QEW" route to $N_e^2$ emission of Zhang et al. (arXiv:2401.05978). A single electron's $\operatorname{Tr}(\rho\hat M)$ depends on its shape, but it enters only the pair term. The guard-rail holds for $N=1$.

**Bounds.** By Cauchy–Schwarz on $\operatorname{Tr}(\rho\,\cdot)$, $\lvert\operatorname{Tr}(\rho\hat M)\rvert^2\le\operatorname{Tr}(\rho\hat M^\dagger\hat M)$. The same bound holds for an entangled $\rho^{(2)}$, since $\lvert\operatorname{Tr}[\rho^{(2)}(\hat M^\dagger\otimes\hat M)]\rvert\le\operatorname{Tr}(\rho^{(1)}\hat M^\dagger\hat M)$ for equal marginals. So for any state, $\langle n\rangle\le N^2\operatorname{Tr}(\rho^{(1)}\hat M^\dagger\hat M)$. Unequal marginals give $(\sum_a\sqrt{T_a})^2$ with $T_a=\operatorname{Tr}(\rho^{(1)}_a\hat M^\dagger\hat M)$. The ceiling is per channel: correlations raise (superradiance) or lower (subradiance) a channel's emission up to $N^2$ times the self term, and no further. No cross-channel sum rule is derived here. Reduced matrices are unit-trace labelled marginals over ordered pairs.

**Exchange.** For indistinguishable electrons in a quasi-free (Slater) state with one-body matrix $\gamma$, the exchange term is $-\operatorname{Tr}(\gamma\hat M^\dagger\gamma\hat M)\le0$ (Pauli antibunching). Its size satisfies $0\le\operatorname{Tr}(\gamma\hat M^\dagger\gamma\hat M)\le\delta\operatorname{Tr}(\gamma\hat M^\dagger\hat M)$, with $\delta=\lVert\gamma\rVert_{\rm op}$ the beam degeneracy. It does not vanish merely for orthogonal orbitals: it needs $\hat M$ to have no matrix elements between distinct occupied orbitals, i.e. a phase-space separation larger than the recoil or emission bandwidth. Beam degeneracies are $\ll1$, so the exchange term is dropped, as an assumption bounded by $\delta$ times the linear term.

**Entangled pairs.** A non-product $\rho^{(2)}$ can give a nonzero pair term while each marginal has $\operatorname{Tr}(\rho^{(1)}\hat M)=0$. This is the Bell-state super/subradiance of {cite:t}`karnieli2021prl`, which the abstract says "cannot be accounted for by a classical mixed state".

Units: $\langle n\rangle$ carries the channel's units; $N$ is dimensionless. Limits:

- $N=1$ has no pair term.
- $\operatorname{Tr}(\rho\hat M)=0$ gives $N\operatorname{Tr}(\rho\hat M^\dagger\hat M)$.
- Identical pure states that are eigenvectors of $\hat M$ give $N^2$. Any other pure state gives less than $N^2\lVert\hat M\psi\rVert^2$.
- For a classical mixture, the classical i.i.d. form is recovered.

**Engine evolution of $\rho^{(2)}$.** Two non-interacting electrons see independent jump operators when their paths are farther apart than the phonon and plasmon correlation lengths: then $\rho^{(2)}(z)=(\Phi_z\otimes\Phi_z)[\rho^{(2)}(0)]$, with $\Phi_z$ the single-electron Lindblad map. If the two electrons pass within a phonon period and a few ångström, they see the same frozen displacement. The pair then needs a collective dissipator, which is not modelled. An operator-Schmidt decomposition $\rho^{(2)}(0)=\sum_kw_kA_k\otimes B_k$ reduces $\Phi_z\otimes\Phi_z$ to single-electron propagations of the Hermitian parts of each $A_k$ and $B_k$. $\Phi_z$ is linear and Hermiticity-preserving, so the existing solver suffices once it is documented to accept Hermitian, non-positive initial operators. A dense product space would be infeasible: 528 states per electron on the production grid ($H=16$, 16 cells) means $\sim10^{11}$ matrix elements.

## 5. Crystal as a wavefunction shaper: channeling-modulated incoherent emission

`Validation: channeling-modulated-incoherent-emission` — research prototype, documented negative result for promotion (see the [channeling ledger](../../validation/ledger-channeling.md)).

**Factor.** Incoherent emission whose rate is local in $x$ is weighted by the transverse probability density. That covers nuclear bremsstrahlung and large-$q$ inner-shell ionization. Relative to a uniform (random-orientation) distribution,

```{math}
:label: eq-qce-overlap

F(z)=\frac{\int_{\rm cell}\rho(x,x;z)\,n(x)\,dx}{\bar n\operatorname{tr}\rho}
=\frac{\sum_r\operatorname{Tr}\bigl(T_nP_r\bigr)}{n_0\operatorname{tr}\rho},
\qquad T_n[h,h']=n_{h-h'},
```

with $P_r=V_r\rho_rV_r^\dagger$ the Bloch block in plane waves and $n_m$ the thermally smeared density harmonics. Population that has dechanneled is assigned $F=1$. The depth-averaged yield factor is $Y(L)=L^{-1}\int_0^L[\operatorname{tr}\rho\,F+(1-\operatorname{tr}\rho)]\,dz$.

Assumptions:

- The emission is local. The nuclear density stands in for bremsstrahlung.
- Inner-shell ionization delocalizes over the impact parameter $\sim\hbar v/E_K\approx1$ Å at MeV energies for Si K. Its factor lies between $F_{\rm nuc}$ and 1: a kernel about 1 Å wide, against $d=1.92$ Å, can flatten it below the electron-density factor $F_{\rm el}$.
- The photon is unaffected by channeling.
- Dechanneled population (removed by the collapse rule) emits at the random-orientation rate, $F=1$. States up to the 300 eV margin above the barrier stay in the subspace and carry their own $F$.
- The collapse rule of `cr-dechanneling-collapse` applies.

Units: dimensionless. Limits:

- A plane wave (uniform density) has $F(0)=1$ exactly, up to the interference with the above-cutoff entry weight $w$ that the collapse rule drops ($\lvert F-1\rvert\le2(n_{\max}/\bar n)\sqrt w$).
- A uniform density $n$ gives $F\equiv1$.
- Perfect focusing gives the upper bound $n_{\max}/\bar n=9.81$ for Si(110) at 293 K.

**Prototype.** `checks/channeling_modulated_emission.py` runs Si(110) with TDS + electronic kernels, $H=16$, 16 cells, $dz=100$ Å, sampled every 100 Å. It compares three entry states: an aligned plane wave; a d-periodic Gaussian comb with $\sigma=0.15$ Å centred on the planes; and the same comb between the planes. The comb is a planar analogue of the shaped beams of {cite:t}`wong2024freeelectron`. Its converter is check-local and stands in for the #508 planar projection. It agrees with direct position-space quadrature to better than $10^{-8}$ when no entry weight is lost; the check gates at $10^{-6}$. Convergence: at 3 MeV, $dz=50$ Å, $H=20$ and 24 cells move $Y(1\,\mu\text{m})$ by ≤ 0.6 %. The run takes 1.5–3 min on CPU.

| $T$ | state | $F_{\rm nuc}(0)$ | $Y(0.1\,\mu\text{m})$ | $Y(1\,\mu\text{m})$ | $Y(2\,\mu\text{m})$ |
| --- | --- | --- | --- | --- | --- |
| 300 keV | plane, $\psi=0$ | 1.00 | 1.72 | 1.30 | 1.16 |
| 300 keV | on-plane comb | 4.48 | 1.47 | 1.22 | 1.11 |
| 3 MeV | plane, $\psi=0$ | 1.00 | 2.15 | 1.58 | 1.31 |
| 3 MeV | on-plane comb | 4.53 | 3.19 | 2.07 | 1.62 |
| 3 MeV | between comb | 0.00 | 0.61 | 0.75 | 0.82 |
| 17 MeV | plane, $\psi=0$ | 1.00 | 2.04 | 1.61 | 1.39 |
| 17 MeV | on-plane comb | 4.53 | 5.10 | 3.19 | 2.35 |

Orientation scan, plane-wave entry, $Y(1\,\mu\text{m})$, tilt in units of $\psi_c$:

| $T$ | $-2$ | $-1$ | $-0.5$ | $0$ | $0.5$ | $1$ | $2$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 keV | 0.98 | 0.91 | 1.12 | 1.30 | 1.12 | 0.91 | 0.98 |
| 3 MeV | 0.99 | 0.91 | 0.95 | 1.58 | 0.95 | 0.91 | 0.99 |
| 17 MeV | 0.96 | 0.89 | 1.12 | 1.61 | 1.12 | 0.89 | 0.96 |

**Findings (mechanism M1).**

1. Channeling alone, from an aligned plane wave, raises local incoherent emission by 1.3–1.6× over the first micron. Shoulders near $\pm\psi_c$ are suppressed (≈ 0.9). This matches the qualitative statement of {cite:t}`genz1996` that additional bremsstrahlung "is enhanced under channeling conditions compared to the case of a random direction". No quantitative orientation-dependent yield dataset was readable in this context. Taftø & Spence (Science 218, 49, 1982) and later ALCHEMI measurements are not in the Zotero library *(verify)*, so this is not a comparison against data.
2. A shaped (on-plane) state's entry overlap of about 4.5 decays within the decoherence and Pendellösung lengths. Over 1 µm it adds only 1.3× (3 MeV) to 2× (17 MeV) over the aligned plane wave. At 300 keV, with one bound state, the focused comb does *worse* than the plane wave (1.22 vs 1.30): its above-barrier components spread out.
3. **Wong 2024 in bulk.** The paraxial overlap part of the >10³ bremsstrahlung enhancement predicted by {cite:t}`wong2024freeelectron` is bounded by $n_{\max}/\bar n\approx10$ for a planar geometry. In bulk Si it erodes to the factors above within $\ell$. That paper treats monolayers (graphene, WS₂, MoS₂, WSe₂) with 20 keV and 300 keV electrons holding up to ~300 momentum states at incidence angles up to 15°. It states explicitly that its enhancements "cannot be attributed solely to the spatial overlap", because they need non-paraxial components and substantial recoil. Components at 15° lie far outside the planar critical angle (9 mrad at 300 keV) and the engine's paraxial validity. They are free states in a bulk crystal, and their coherent sum over atoms along $z$ becomes a phase-matched, CBS-like condition. So the engine cannot test the non-paraxial part. What it does show: the paraxial overlap enhancement does not survive a bulk propagation beyond ≲ 1 µm, and is bounded by about 10 in planar geometry.

**Decision: not promoted.** The factor is physical and inexpensive, but no readable quantitative anchor exists yet. Promotion would also be a production-scope decision (it would modulate the incoherent background of #507), so it is left to the maintainer. The row stays `unverified` as a documented negative result for promotion. Follow-up F1 records the conditions for revisiting.

## 6. Quantum PXR/CBS with transverse recoil (Shi et al. 2024)

{cite:t}`shi2024transverse` (Nat. Commun. 15, 7803; Europe PMC full text) treat a joint electron–photon state in which the crystal's transverse reciprocal vectors $\mathbf g_\perp$ deflect the electron during emission. Each transverse-recoil pathway has its own photon frequency, so the classical CXR peak "splits and shifts" and the split peaks are linearly polarized. Their worked cases are 30 keV, 300 keV and 5 MeV electrons through 30 nm Si films at normal incidence. They note that in vdW crystals (WSe₂, graphite) the $\mathbf g_\perp=0$ peaks dominate, which explains why the splits were not seen in those experiments.

- **Mechanism:** M2, recoil-resolved final states, through M1, the lattice. It is a single-electron effect, so it is allowed by the guard-rail.
- **Required basis:** free (above-barrier) continuum transverse states carrying both transverse components of $\mathbf g_\perp$ and the longitudinal $g_\parallel$. Pathways must be resolved in transverse momentum to $\delta k<g_\perp/2$. The #56 planar continuum Bloch basis keeps only $\mathbf g$ normal to one plane family and averages out $g_\parallel$. It has neither PXR's crystal-mediated photon mode $\propto\chi_{-\mathbf g}e^{i(\mathbf k-\mathbf g)\cdot\mathbf r}$ nor the $g_\parallel$ phase matching. `transition_operator` supplies the transverse half of the matrix element, but the photon mode function and longitudinal momentum ledger are missing.
- **Dominant decoherence channel:** jumps with $q\gtrsim\delta k$ scramble the pathway label. Their rate is approximated by $\Gamma(\Delta\approx1/\delta k)$, with $\Delta=1.73$ Å for Si $g_\perp=2\pi/a$. At 30 keV the electronic term dominates (1.3e-3 against TDS 5.9e-4 Å⁻¹ at 1.92 Å). At ≥ 300 keV the two are comparable.

**Go (research), regime bounds for Si.** The pathway survives a film of thickness $L$ with probability $e^{-\Gamma L}$, and the beam divergence must stay below $g_\perp/2k$:

| $T$ | $\Gamma(1.73\,\text{Å})$ [Å⁻¹] (`decoherence_rate`) | $1-e^{-\Gamma\cdot30\,\text{nm}}$ | $L$ for $\Gamma L\le0.3$ | divergence $<g_\perp/2k$ |
| --- | --- | --- | --- | --- |
| 30 keV | 1.80e-3 | 0.42 | 17 nm | 6.4 mrad |
| 300 keV | 3.24e-4 | 0.09 | 93 nm | 1.8 mrad |
| 3 MeV | 2.00e-4 | 0.06 | 150 nm | 0.33 mrad |
| 30 MeV | 1.96e-4 | 0.06 | 153 nm | 0.04 mrad |

Shi's 30 nm films lose 42 % of the pathway coherence at 30 keV and 6–9 % at higher energies. This is an estimate: $\Gamma(1/\delta k)$ is a proxy for the rate of pathway-scrambling jumps. The effect matters most for Si-like crystals with strong $\mathbf g_\perp\ne0$ structure factors. For PyRITE's vdW targets it is weak, by Shi's own statement. Follow-up F2 gives the design.

## 7. Two-electron super/subradiance (Karnieli et al. 2021)

{cite:t}`karnieli2021prl` show that "a pair of coincident path-entangled electrons can demonstrate either super- or subradiant light emission, depending on the two-particle wave function" (abstract). The PRL full text was not readable here, so its quantitative factors are not quoted.

- **Mechanism:** M3 with quantum correlations: the pair term of {eq}`eq-qce-pair` with a non-product $\rho^{(2)}$.
- **Required basis:** the product of two single-electron transverse spaces, evolved by operator-Schmidt decomposition (§4). The path separation is the entangled degree of freedom.
- **Dominant decoherence channel:** for path separations ≫ Å, the full loss rate $\Gamma(\infty)$, applied to each electron. The pair coherence survives as $e^{-2\Gamma(\infty)L}$: in Si, $1/\Gamma(\infty)$ is 272 Å at 30 keV and 0.13 µm at 300 keV.
- **Coincidence:** both electrons' emitted wave trains must overlap. A train emitted over a coherent length $\ell$ lasts $\tau\approx\ell(1/v-\cos\theta/c)$ in the lab. With $\ell=\ell(1\,\text{Å})$ in Si: $\tau\approx$ 0.5 fs forward and 0.7 fs at 90° at 30 keV; 0.4 fs and 1.7 fs at 300 keV; 23 as forward and 2.2 fs at 90° at 3 MeV.

**No-go** for any PyRITE beam model. Random coincidences in a Poisson beam are classical mixtures. They contribute only the separable term of {eq}`eq-qce-iid`, which needs bunching, not entanglement. Entangled pairs need an engineered source with sub-fs coincidence, and no such source is specified. Revisit only with a concrete source model (follow-up F3, conditional).

## 8. Dicke superradiance of CR from prebunched beams

The i.i.d. pair term {eq}`eq-qce-iid` with arrival times $t_a$ gives $N(N-1)\lvert\operatorname{Tr}(\rho\hat M)\rvert^2\lvert b(\omega)\rvert^2$, where $b(\omega)=\langle e^{i\omega t_a}\rangle$ is the bunch form factor at the CR photon frequency. The single-electron factor $\lvert\operatorname{Tr}(\rho\hat M)\rvert^2/\operatorname{Tr}(\rho\hat M^\dagger\hat M)\le1$ is the transverse-coherence fraction. It requires coherences $\rho_{nm}$ between the bands the line connects. These are zero for an aligned plane wave in a symmetric well by parity ($\hat M\propto x$ connects even to odd states), and they decay at about $\Gamma$.

**No-go in the planar keV regime.** Si(110) and diamond(110) have a single bound state at $k=0$ at 30 and 300 keV (§3.2). Only free–bound and band transitions remain, which are broad. Bunching at an EUV line also needs sub-fs modulation (for example ≈ 0.14 fs period for ≈ 30 eV photons) with $N\gg1$ electrons per coherence time $\tau\lesssim1$ fs. At MeV energies, where bound lines exist, the effect is the classical-correlation form-factor case. It belongs with the [superradiant PXR](superradiant-pxr.md) normalization work, gated on a fixed-period bunch source. No implementation issue is proposed.

## 9. Shatnev (arXiv:2603.28827): verdict

Claim, from the abstract and the arXiv HTML (v2; equation numbers as read there, *verify against Zotero*): positrons at 4–14 GeV in diamond(110) at $\theta_{\rm in}=31$ µrad. The harmonic well makes the levels equidistant, and entry produces a Glauber coherent state with $\lvert c_n\rvert^2=e^{-n_0}n_0^n/n!$. The paper then defines the coherent intensity (Eq. 12) as $\sum_j\lvert\sum_nc_nM_{n,n-j}\rvert^2\delta(\omega-\omega_j)$ and the gain (Eq. 13) as $\mathcal G_j=\lvert\sum_nc_nM_{n,n-j}\rvert^2/\sum_n\lvert c_n\rvert^2\lvert M_{n,n-j}\rvert^2$, reporting $\mathcal G=12$–31 with $I_{\rm coh}\propto n_0^2$.

**Verdict: rejected. No PyRITE work.**

1. Eq. 12 adds amplitudes $c_nM_{n,n-j}$ whose final states $\lvert n-j\rangle$ are mutually orthogonal. The photon rate traces over final electron states: $\sum_f\lvert\langle f\rvert\hat M\lvert\psi\rangle\rvert^2=\sum_n\lvert c_n\rvert^2\lvert M_{n,n-j}\rvert^2$, exactly the "incoherent" sum. Eq. 12 is not linear in $\rho$ and violates the linearity rule of §1.
2. The #505 test `test_harmonic_coherent_state_is_classical_and_has_no_phase_locked_gain` evaluates exactly this case: exact ladder, all $n\to n-1$ lines in one cluster, coherent state with $\lvert\alpha\rvert^2=10$. The rate equals $\lvert\alpha\rvert^2$ times the $0\to1$ rate. The coherent-to-dephased ratio is 1 to $10^{-12}$. The rate equals the classical Larmor oscillator of the same energy. The preprint's Eq. 13 ratio evaluated on the same state is 15.6, inside its quoted 12–31 band. The test pins that the engine does not produce it.
3. Correspondence: a coherent state is the quantum image of the classical oscillator, whose intensity scales with amplitude², i.e. $\propto E_\perp\propto\theta_{\rm in}^2$. That is the preprint's "incoherent" scaling (i). The claimed $\theta_{\rm in}^4$ scaling (ii) contradicts the classical limit.
4. The reported agreement with Avakyan et al. covers first-harmonic *peak positions*, which do not depend on $\mathcal G$.
5. Scope: GeV positrons are outside the channeling engine (refused above 100 MeV), so nothing is computable here in any case.

No guard-rail mechanism applies: N = 1, no recoil resolution, and the interaction is the same harmonic well.

## Interface requirements

Entry states come from the #508 interface, which owns sampled $\psi(x)$/$\psi(x,y)$, presets, `from_array`/`from_abtem`, the planar Bloch projection and the `initial=` plumbing through `channeling_spectrum` and `MCWFSolver.run`. This research needs, from that interface:

1. **Registered periodic states.** A d-periodic $\psi(x)$ with fixed registration to the lattice must project into a single Bloch block ($\kappa=0$) without translation averaging (§5), alongside registration-averaged finite probes. The returned entry flux must count above-cutoff weight, as `propagate(initial=(rho, entry_flux))` expects.
2. **Hermitian, non-positive initial operators.** `propagate` should be documented and tested as a linear map on Hermitian operators of any sign, for operator-Schmidt two-body propagation (§4). Its current symmetrization step already preserves Hermitian inputs.
3. **Depth observables.** A depth callback or recorded $\rho$ snapshots, so that $\operatorname{Tr}(\rho T)$ can be scored without chaining solves. The prototype chains 100 Å solves.
4. **Local-density observable.** A generic $\operatorname{Tr}(\rho\,T_f)$ for a periodic function $f(x)$, reusing the solver's existing Toeplitz construction.

Items 2–4 are proposals for the #56 engine, not changes made here. The check-local comb converter is a stand-in to be replaced by the #508 projection once it lands.

## Follow-ups

Proposed only; no issues have been created.

- **F1 — channeling-modulated incoherent background (conditional).** Gate: a readable quantitative orientation-dependent yield dataset (Taftø & Spence 1982 or an MeV characteristic or bremsstrahlung tilt scan), and a maintainer decision to modulate the #507 incoherent background. Scope: the factor {eq}`eq-qce-overlap` with an impact-parameter-delocalized inner-shell kernel, anchored against that dataset. Status target: `anchored`.
- **F2 — recoil-resolved CXR operator (research).** Scope: a free-state transverse basis with both $\mathbf g_\perp$ components and $g_\parallel$; the crystal-mediated photon mode $\chi_{-\mathbf g}$; reproduce one Shi et al. 30 nm Si spectrum at 300 keV; regime bounds as in §6. Depends on #508 entry states and #56 jump operators. No production channel.
- **F3 — entangled-pair emission (conditional on a source model).** Operator-Schmidt two-body propagation (§4) and the pair term. Only once a specific entangled-pair source with sub-fs coincidence is in scope.
- **F4 — channeling thermal data for layered crystals.** Anisotropic $\langle u_x^2\rangle$/$\langle u_z^2\rangle$ for HOPG, hBN, MoS₂ and WSe₂ in `channeling_thermal.toml`, plus diamond conduction-band shell data. Together these let the §3.2 cells be computed.

Documented no-go: Dicke CR at keV in planar Si/diamond (§8); the Shatnev phase-locked gain (§9).
