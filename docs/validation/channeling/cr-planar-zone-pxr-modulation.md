# Channeled-state PXR modulation on planar zone rows

Validation: `cr-planar-zone-pxr-modulation`. Status: **rederived**. #1148 measured the omission and found it fails at Bragg-pole geometries (sections below). #1166 implements the source ([Implementation](#implementation-1166)), and a fresh-context verification re-derived it ([below](#fresh-context-verification-1166)). The production re-measurement at the #1148 centres is pending.

## Question

Native line scoring treats a planar channeling zone row as a straight plane-wave electron at the beam direction ([`cr-planar-zone-row-ownership`](cr-planar-zone-row-ownership.md)). For the reflections that are not on the plane's own reciprocal row, it scores PXR/CBS natively on zone rows. The question is whether the omitted modulation by the tracked transverse states changes the band-integrated intrinsic `spec` at the #1148 detectors by more than the first-pass tolerance of 10%. Those detectors are a source-facing JUNGFRAU square, $2.05\times10^{-4}$ sr at 5 m, polar centres 30–45°, with `channeling_demo` (Si(1-10), 17 MeV, 2 µm). Detector response is out of scope.

## Source and amplitude

Write the PXR amplitude in the reciprocity (photon-mode) form of Feranchuk et al., Eq. (2). For photon $(E,\hat n)$, polarization $e_s$, reflection $g$ (sign with $v\cdot g>0$), $k=E\hat n/\hbar c$ and $Q=k+g$, the diffracted mode component in a kinematic crystal is proportional to $\chi_g F_s/\det$, where

$$
F_s=Q\,(g\cdot e_s)-k^2e_s,\qquad \det=|k+g|^2-k^2=g^2+2k\cdot g .
$$

Inside the zone the electron is in transverse Bloch states $|i\rangle$ (fibre $\kappa$) of the planar continuum potential, times an in-plane plane wave along $\hat v$. The beam lies in the plane, so the transverse momentum is carried only by the states. The transition current to $|f\rangle$ (fibre $\kappa-Q_x$, $Q_x=Q\cdot\hat x$) contracted with $F_s$ is

$$
M^s_{fi}=\beta\,(\hat v\cdot F_s)\,A_{fi}+(F_s\cdot\hat x)\,B_{fi},\qquad
A=\langle f|e^{-iQ_xx}|i\rangle,\quad
B=\langle f|\tfrac12\{v_x/c,e^{-iQ_xx}\}|i\rangle .
$$

Energy conservation gives the line centre $E_{fi}=E_0+(\varepsilon_i-\varepsilon_f)/D$, with $D=1-\beta\hat v\cdot\hat n$ and $E_0$ the exact-recoil free root (`materials/recoil.py::recoil_resonance_eV`).

**Finite slab.** For a Laue slab of path length $L$, the time-reversed photon mode has zero diffracted amplitude at the exit face. The diffracted component is then the particular solution minus a free diffracted wave $k_h=k+g+q\hat s$, with $|k_h|=k$ and $q=-\det/(2Q\cdot\hat s)$ to first order. Along the path $z$,

$$
K_{fi}(E)=\frac{1}{\det(E)}\int_0^L dz\,e^{i\Delta z}\bigl[1-e^{iqs(L-z)}\bigr],\qquad
\Delta=\frac{D\,(E-E_{fi})}{\hbar c\beta},\quad s=\hat s\cdot\hat v .
$$

**Pre-crystal path.** The same Green's function leaves a diffracted wave $\propto(1-e^{iqsL})e^{-ik_h\cdot r}$ outside the entrance face, which the electron crosses before entering the slab. It adds $K_{\rm pre}=(1-e^{iqsL})/(\det\,i(\Delta-qs))$, and

$$
K+K_{\rm pre}=\frac{\varphi(\Delta)}{\det}\,\frac{-qs}{\Delta-qs},\qquad
\varphi(\Delta)=\int_0^Le^{i\Delta z}dz ,
$$

which is the first-order Born result: the electron's vacuum field scattered by the truncated $\chi_g e^{ig\cdot r}$, a single line with no pole. Before entry the electron is a free plane wave, so it has no transverse transition to feed the pre-crystal term. The check uses the Born form for transverse-elastic channels ($|\varepsilon_i-\varepsilon_f|<0.15$ eV, which covers $f=i$ and degenerate pairs) and the in-slab $K$ for transverse transitions. That is exact at $Q_x=0$, where $A_{fi}=\delta_{fi}$; off the plane the term belongs on the $e^{-iQ_xx}$ vertex of every pair, and the split over-states (1 1 −3) by 3–10% per ray (Verdict, item 2).

The band-integrated yield per solid angle is

$$
\frac{dN}{d\Omega}\propto\sum_s\int dE\;E\,\bar T(E)\,|\chi_g|^2
\sum_{f}\Bigl|\sum_i\psi_i\,M^s_{fi}(E)\,K_{fi}(E)\Bigr|^2 ,
$$

with $d^2N/dE\,d\Omega\propto\omega|a|^2$ as in `_cxr_weight_core` and $\bar T$ the uniform-emission Beer–Lambert escape mean. The plane-wave reference is the same expression with $A=1$, $B=0$, $\varepsilon_i=\varepsilon_f$, and the field frozen at $E_0$:

$$
\Bigl(\sum_s\beta^2(\hat v\cdot F_s)^2\Bigr)\frac{2\pi L\hbar c\beta}{D\,\det(E_0)^2}E_0\bar T|\chi_g|^2\operatorname{tr}\rho .
$$

**Two lines per transition.** The first term of $K$ is PXR: in the long-slab limit $\int|{\cdot}|^2dE\to2\pi L\hbar c\beta/(D\det^2)$ at $E_{fi}$. The second term phase-matches at $\Delta=qs$, i.e. $\omega/v-k_h\cdot\hat v=(\varepsilon_i-\varepsilon_f)/\hbar v$. That is exactly the condition for real forward emission of photon $k_h$ by the transverse transition. This term is forward channeling radiation Bragg-diffracted into $\hat n$: **diffracted channeling radiation (DCR)**. It replaces the unbounded kinematic pole $1/\det$ of the frozen-field picture with a finite, thickness-limited diffraction integral. That pole is why the earlier frozen-field diagnostics could not certify a supremum bound (#1148 Progress, 2026-11-12).

## Limiting cases

- **Free electron** ($\varepsilon_i=\varepsilon_f$, $A=1$, $B=0$): the Born form applies, and its band integral reduces to the plane-wave reference wherever $\det$ is slowly varying over the line. `tests/montecarlo/channeling/test_zone_pxr_modulation_check.py` checks this at the (−1 −1 3) minus-in ray for (1 1 −3), 34 eV from its pole: the check agrees with the free Born result and with native to within 1%. The DCR term alone cannot phase-match for a free electron ($\omega/v-k_h\cdot\hat v\ge\omega(1/v-1/c)>0$), but without $K_{\rm pre}$ it is **not** negligible near a pole: the in-slab term alone gave 2.24 at that ray (first verifier, finding 1). Very close to a pole the finite-length line samples a varying $\det$, so even the Born free electron departs from the native frozen-root value; at the single mirror ray that departure is ×1.28, but over the order-3 square it is only 1.005 (`free_born_over_native`, table below). That native finite-line error is in scope for #1166.
- **Frozen field** ($K$ replaced by its PXR term at $E_0$, complete final basis, full coherences): reproduces the independent 2026-11-12 frozen-field diagnostic. (1 1 −3) at the (−1 −1 3) minus-in peak gives 1.0051 here versus 1.005227 there.
- **Complete basis closure:** $\sum_f|A_{fi}|^2=1$ in the finite 33-wave basis, so the scalar part of the frozen sum equals the reference exactly.
- **Thin-crystal reflectivity:** the energy-integrated DCR line equals forward CR times the kinematic Laue reflectivity $R_E=E\,R_\theta/\tan\theta_B$, with $R_\theta=\pi k|\chi_g|^2t/(2\gamma_0\sin2\theta_B)$ (Authier, *Dynamical Theory of X-ray Diffraction*, kinematic limit). The independent estimate below uses this.

## Measurement

Implemented in `checks/channeling_zone_pxr_modulation.py`. Inputs:
- the live depth-averaged retained density of the admitted demo zone: 81 depth records, trace 0.963, identical eigenbasis at every depth;
- 16 Bloch fibres × 33 plane waves;
- every off-own-row reflection whose aperture yield exceeds $10^{-3}$ of the strongest, or of the strongest ≥2 keV reflection;
- a ±400 eV window at 0.1 eV (step-converged in a synthetic test; windowed weight leakage below $10^{-6}$);
- tensor Gauss rules over the square (`validation/channeling_detector.py::square_aperture_rule`).

Record: `docs/validation/check-records/planar-zone/zone-pxr-modulation-1148.json`.

Order-3 Gauss rule over the square (qlmc jobs 12268–12277); order 1 (job 12278) for comparison. *Ratio* = model / native plane-wave reference, band-integrated. *vs Born* divides by a free plane-wave electron in the same Born kernel. *Born/native* isolates the native frozen-root finite-line error.

| centre | polar | ratio, o1 | ratio, o3 | vs Born, o3 | Born/native, o3 | ≥2 keV ratio (share) |
|---|---|---|---|---|---|---|
| (0 0 −4)-az | 30° | 0.9967 | 0.9967 | 0.9972 | 0.9994 | 0.998 (0.14) |
| (0 0 −4)-az | 37.5° | 0.9974 | 0.9974 | 0.9974 | 1.0000 | 0.999 (0.13) |
| (0 0 −4)-az | 42.807° | 0.9978 | 0.9978 | 0.9977 | 1.0001 | 0.999 (0.22) |
| (0 0 −4)-az | 44.479° | 0.9980 | 0.9980 | 0.9979 | 1.0001 | 0.999 (0.27) |
| (0 0 −4)-az | 45° | 0.9981 | 0.9981 | 0.9980 | 1.0001 | 0.999 (0.29) |
| (−1 −1 3)-az | 30° | 0.9962 | 0.9962 | 0.9969 | 0.9993 | 0.998 (0.14) |
| (−1 −1 3)-az | 37.5° | 0.9974 | 0.9975 | 0.9978 | 0.9997 | 0.999 (0.32) |
| (−1 −1 3) minus-in peak | 42.807° | 1.0094 | 1.0527 | 1.0533 | 0.9995 | 1.074 (0.72) |
| (−1 −1 3) mirror | 44.479° | 1.6510 | **1.4372** | 1.4301 | 1.0050 | 1.780 (0.56) |
| (−1 −1 3)-az | 45° | 1.2944 | **1.2045** | 1.2021 | 1.0024 | 1.288 (0.71) |

Every reflection other than (1 1 −3) stays within 2% of the reference at every centre. The excess sits in (1 1 −3), whose pole lies 15–17 eV above the root at the mirror and 45° squares and 28–43 eV at the minus-in peak:

| centre | (1 1 −3) ratio, o3 | band share | pole offset (eV) |
|---|---|---|---|
| mirror | 2.466 | 0.30 | 14.9–16.4 |
| (−1 −1 3)-az 45° | 1.390 | 0.53 | 14.8–17.8 |
| minus-in peak | 1.090 | 0.60 | 27.7–43.4 |

The single mirror ray (order 1) gives 22.8 for (1 1 −3); the square average dilutes it, which is why the order-1 band ratios over-state the order-3 ones. Coherences contribute below $10^{-4}$ relative (`frozen_full` vs `frozen_diag`). Cost: 13–52 min and ≤0.66 GiB per order-3 centre on one CPU.

## Independent DCR estimate

`scratch` script `dcr_check.py` (qlmc job 12279, 49 min, 1.4 GiB) estimates DCR without the transverse-state model. For each ray of the square it takes production forward `spec_channeling` at the Bragg-mirror image direction $\hat k_h$ and the exact vacuum Bragg energy $E_B=\hbar c\,g^2/(2|\hat n\cdot g|)$, multiplies by the kinematic σ integrated reflectivity $R_E$ of Si(113) ($R_E=0.085$ eV at $t=2$ µm, $\theta_B=22.3°$, $|\chi_g|=3.9\times10^{-6}$) and by ½ (mean remaining depth for uniform emission). It compares that with production native PXR of $\pm(1\,1\,\bar3)$ on the same transport (8 electrons, 8–12 keV at 0.5 eV, bin-mean).

| centre | native PXR (ph/e) | DCR estimate (ph/e) | DCR/PXR |
|---|---|---|---|
| mirror, order 3 | $4.22\times10^{-11}$ | $5.99\times10^{-11}$ | 1.42 |
| minus-in peak, order 3 | $1.63\times10^{-10}$ | $4.0\times10^{-12}$ | 0.024 |
| minus-in peak, single ray | $7.96\times10^{-7}$ /sr | $1.40\times10^{-8}$ /sr | 0.018 |

At the mirror the independent $(\mathrm{PXR}+\mathrm{DCR})/\mathrm{PXR}=2.42$ agrees with the check's 2.47 for (1 1 −3), within the estimate's ~20–30% systematics (forward-CR model, ~−10% dynamical $R_E$, the ½ depth factor, no PXR–DCR interference); the close match is partly fortuitous. At the minus-in peak $\hat k_h$ is 24–35 mrad off the beam, where forward CR at $E_B\approx10.8$ keV is about 25× weaker than on axis ($3.3\times10^{-7}$ against $8.6\times10^{-6}$ near 10 keV), and both methods give a few-percent effect. The first verifier's ray-level estimate (CR × $R_E$ × ½ within 10–30% of the check after the $K_{\rm pre}$ fix) is consistent.

This matches the literature: diffracted channeling radiation was predicted by Yabuki et al., *Phys. Rev. B* 63, 174112 (2001), at up to about 10× PXR, and Korotchenko, Pivovarov & Tukhfatullin, *NIM B* 269, 2840 (2011), show that PXR is the transverse-elastic special case of DCR, with DCR maxima up to 100× the PXR maxima for axially channeled electrons. DCR has not yet been observed experimentally.

## Assumptions and omissions

- Kinematic two-beam diffraction: $L=2$ µm against an extinction length of about 9 µm for Si(113) at 10 keV.
- Flat $\chi_g$ across the window.
- $\chi_0$, refraction and the photon's own absorption inside the DCR path are omitted (escape is applied at $E_0$).
- Stationary depth-averaged density.
- Diagonal populations in the frequency-dependent term (coherences enter only through the frozen correction `frozen_full − frozen_diag`).
- Finite state lifetimes and line tails beyond the window are omitted.
- CBS distorted-wave resolvents and PXR–CBS interference are not included.
- The plane-wave reference is evaluated in the same model, so ratios rather than absolute yields are reported.
- Native diffracted bremsstrahlung (DBR), with channeling on or off, is a related omission that this check does not evaluate.

## Verdict

Two fresh-context verifications (2026-11-13), both `concern`; status stays **unverified** and the omission bound **fails**.

1. First pass (`scratch` report, pre-fix kernel): the in-slab kernel dropped the electron's pre-crystal path. That made the free-electron limit fail near poles (2.24 at the minus-in ray) and made the pre-fix minus-in failure (1.753) an artifact. Fixed by the Born form above; the mirror and 45° DCR excess survived.
2. Second pass: kernel, signs, root $q$, phase match, units ($R_E=0.084$ eV) and the Born identity rederived independently. One difference: before entry the electron is $|\mathrm{in}\rangle=\sum_ic_i|i\rangle$, so $K_{\rm pre}(\Delta_0)$ belongs on the $e^{-iQ_xx}$ vertex of *every* pair (and on the velocity vertex only through the entrance velocity), not on an energy-selected elastic subset. At $Q_x=0$ ($A_{fi}=\delta_{fi}$) the split is exact; at off-plane Gauss nodes 7–8% of the $A$ weight sits in inelastic pairs that keep the truncated-path kernel. Per-ray (1 1 −3) inflation: mirror +3%, 45° +4–5%, minus-in +10%. Corrected order-3 estimates: **mirror ≈1.43, (−1 −1 3)-az 45° ≈1.19, minus-in ≈1.02**; production arms unaffected. This also explains why the minus-in rises from order 1 to order 3 and why its (1 1 −3) excess (0.090) is about 4× the independent 0.024.

Robustness: the mirror failure is DCR and robust (the 2.42 vs 2.47 agreement is within the independent estimate's ~20–30% systematics; 30% less DCR still gives ≈1.31). The 45° failure (≈1.19) has no independent DCR estimate and is marginal if DCR were overstated 2×. All passes are robust. The free-electron test's `estimate ≈ free_born` assertion is near-tautological (free states make every pair elastic); `≈ native` is the real check. A vertex-correct $K_{\rm pre}$ and an off-plane closure test with bound states belong to the implementation in #1166.

Never `signed-off` by an agent; human sign-off in #277.

## Implementation (#1166)

`channeling/zone_pxr.py` implements the source and `runner/planar_zone_pxr.py` wires it into `spec_channeling`. Native line scoring drops every source-owned reflection on zone rows ([`cr-planar-zone-row-ownership`](cr-planar-zone-row-ownership.md)). A reflection is source-owned towards $\hat n$ when $v\cdot g>0$, it has a physical exact-recoil root, and both $\hat n\cdot\hat s$ and $\hat Q\cdot\hat s$ exceed $3\sqrt{|\chi_0|}$ (Laue geometry, outside total external reflection). `CHANNELING_MODEL` is `planar-density-matrix-emission-v10`.

**Two-beam mode.** The reciprocity mode enters through the exit face with unit transmitted and zero diffracted amplitude. In the inward depth $u=\zeta_L-\zeta$, with the physical $\chi$ ($\operatorname{Im}\chi_0>0$ absorbs), it obeys

$$
\frac{dP}{du}=iNP,\qquad N=\begin{pmatrix}k\chi_0/2n_s & a\\ b & q+k^2\chi_0/2Q_s\end{pmatrix},\qquad ab=\frac{k^4|\chi_g|^2C_s^2}{4k\,n_sQ_s},
$$

with $q=-\det/(2Q_s)$. Relative to the kinematic particular amplitude $\chi_gF_s/\det$, the diffracted profile is $P(z)=-q\,[e^{i\lambda_1s(L-z)}-e^{i\lambda_2s(L-z)}]/(\lambda_1-\lambda_2)$, where $\lambda_{1,2}$ are the eigenvalues of $N$. Its kinematic limit is the #1148 profile $1-e^{iqs(L-z)}$. The $z$ integral is closed-form,

$$
K_{fi}=\frac{\psi(\Delta_{fi},\lambda_1s)-\psi(\Delta_{fi},\lambda_2s)}{2Q_s(\lambda_1-\lambda_2)},\qquad
\psi(x,\mu)=\int_0^Le^{ixz+i\mu(L-z)}dz=\frac{e^{ixL}-e^{i\mu L}}{i(x-\mu)},
$$

and it stays finite at the pole: $q/\det$ cancels. $\psi$ keeps both exponentials bounded when $\operatorname{Im}\mu>0$; the product form $e^{i\mu L}\varphi(x-\mu)$ overflowed at soft, grazing roots. $\operatorname{Im}\chi_0$ supplies the photon escape absorption, $\operatorname{Re}\chi_0$ the refraction shift, and $ab$ the Pendellösung depletion. For Si(113) at 10 keV and $t=2$ µm, $\sqrt{ab}\,t\approx0.2$, so the dynamical change is $\approx x^2/3\approx1$–2%. The #1148 estimate of about −10% took $\Lambda_0/\pi$ for the Pendellösung period $\Lambda_0$.

**Pre-crystal vertex.** Before entry the electron is the plane wave $|p\rangle$. Both $e^{-iQ_xx}$ and the velocity vertex act on it, so $\sum_i\langle i|p\rangle M_{fi}=\langle f|M|p\rangle$ and

$$
M^{\rm pre}_{fi,s}=\Bigl[\beta\hat v\cdot F_s+(F_s\cdot\hat x)\tfrac{v_x(p-Q_x/2)}{c}\Bigr]A_{fi},\qquad
\Delta_{\rm pre}=\frac{D(E-E_0)-2CpQ_x}{\hbar c\beta}.
$$

This is the vertex-correct form the second verifier asked for. $p=\kappa+\nu G$ is the true momentum of each entering plane wave. The entry density is an incoherent mixture of plane waves, from beam divergence or the source spot, so every entering order carries its own $\Delta_{\rm pre}$ and entrance vertex, and its intensity is weighted by its entry weight. A state $i$ averages over the entering waves with weight $w_\nu|\langle i|\kappa+\nu G\rangle|^2$, normalized per state, so it interferes only with the waves that fed it; this is exact in the free limit. Fibres fed only by scattering keep a pre-crystal term at the folded $\kappa$, the same stationary path-mean approximation. Folding $p$ into $[-G/2,G/2)$, as the first implementation did, was 2–17% off for $\nu=1$–4 within 12 eV of a mirror pole (fresh-context verifier, #1166). Tilted #1055 cases enter at $\nu\approx2$–3. It differs from the #1148 elastic split in two ways: off the plane, and through the velocity vertex at the mirror, where $\hat v\cdot F_s\to0$. At the single mirror ray the source gives 24.2 against the check's 22.8. Line centres carry $+CQ_x^2$, the transverse recoil already in $E_0$, so free states return exactly to the free root.

**CBS.** Native scores $A=A_{\rm PXR}+A_{\rm CBS}$ coherently. The source therefore adds $u_gf_{{\rm cbs},s}A_{fi}K^{\rm cbs}_{fi}$, with $\chi_g$ and $u_g=U_g/mc^2$ taken from native's `emission_coupling_tables` (so the PXR–CBS phase matches native), native's exact $f_{\rm cbs}$ at the beam velocity, and $K^{\rm cbs}$ the transmitted branch of the same mode. CBS shares the momentum transfer $Q=k+g$ and hence $A_{fi}$; it has no pre-crystal term. Without it the source missed the native zone-row band by 6–11% in the test fixture, with ±(20–80)% per reflection from PXR–CBS interference. CBS distorted-wave resolvents stay out of scope.

**Numerics.** Pairs holding the weakest $10^{-7}$ of the frozen weight are dropped. The energy window spans the PXR centres $E_{fi}$ and the DCR phase-match energies (only for inelastic pairs; elastic ones never phase-match) $\pm200$ slab widths, capped at ±1.5 keV. The step is a quarter width, because $|\varphi_L|^2$ is band-limited. Output is bin-mean or node, following the case's `line_quadrature`. The source averages the detector cone and the in-plane divergence (5 Gauss–Hermite nodes, a rotation about the plane normal). Cost is about 3 s per ray and reflection for the demo zone (16 fibres × 33 waves).

| check | result |
|---|---|
| two-beam profile vs `expm` of $N$ | $10^{-8}$ |
| $K$ vs quadrature of $e^{i\Delta z}P$; $K_{\rm pre}$ vs upstream continuation | $10^{-6}$; $10^{-12}$ |
| kinematic elastic pair = Born line | $10^{-9}$ |
| free electron far from a pole vs native PXR and PXR+CBS analytic band | $3\times10^{-3}$ |
| bound-state closure at $Q_x\neq0$ (deep cosine potential, tilted entrance) | $10^{-8}$ |
| 17 MeV Si(1-10) fixture at 35° and 40° Laue directions (poles 3–60 keV away): source band / replaced native zone-row band | 1.007, 1.007 |
| `channeling_demo` mirror / minus-in single rays, (1 1 −3), / plane-wave | 24.2 / 1.009 |

Anchors: `tests/montecarlo/channeling/test_zone_pxr.py`, `tests/montecarlo/channeling/test_planar_zone_pxr.py`.

## Fresh-context verification (#1166)

2026-11-13, two passes by an independent verifier (`physics-validator`) against the uncommitted fixes.

| item | first pass | re-check |
|---|---|---|
| mode matrix $N$, $ab$, $C_s$, $\chi_0$ sign and conversion | pass | pass |
| $K$, $K_{\rm pre}$; $K+K_{\rm pre}=\varphi(\Delta)/(\det+2Q_s\Delta/s)$ | pass (algebraic) | pass |
| pre-crystal vertex: forms and $\Delta_{\rm pre}$ | **discrepancy**: folded $p$ off by 2.2 / 6.7 / 17% at $\nu=1/2/4$, 11.7 eV from the pole | fixed (code/Born = 1.00000 at $\nu=1,2,4,-3$). Mixed entry with entry weight only was 3.9% off; now per-state overlap weights |
| line centres $+CQ_x^2$ | pass | pass |
| absolute normalization ($\int|\varphi_L|^2dE=2\pi L\hbar c\beta/D$) | pass | pass |
| CBS phase | concern: $\chi(h)$ vs native $\overline{\chi(-h)}$, a phase of up to 0.15 rad | fixed (native tables) |
| ownership | concern: decided per node; could double count | fixed (decided once at the centre) |
| window and pruning | pass; ±1.5 keV cap below 200 widths for $D<0.13$ | documented |

Assumption the verifier added: the longitudinal part $\det(e_s\cdot\hat Q)\hat Q$ of $F_s$ is a local response and strictly carries only $\varphi_L$. The source gives it the homogeneous waves too. The effect is zero for σ and at the pole, and 0.2% of a bound-state band 24 eV from the pole. Verdict after the fixes: `rederived`. Probes: `scratch/verify_1166_{probe,probe2,coupling,recheck}.py` in the task worktree (not committed).

