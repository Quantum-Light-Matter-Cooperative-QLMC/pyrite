# Enhanced Parametric X-Ray Radiation Emission from Shaped Electrons

Parametric X-ray radiation (PXR) is produced when the electromagnetic field of a moving electron polarizes a periodic crystal and the resulting polarization currents radiate coherently. In the conventional description, constructive interference is imposed by the reciprocal lattice, giving the familiar PXR phase-matching condition and strong angular and spectral selectivity [3,4]. In a real-space description, the emitted field is the coherent sum of radiation from induced atomic dipoles; at resonance, the longitudinal contribution from ($N$) equivalent crystal planes can scale as ($N^2$), while the transverse interaction is limited by the spatial extent of the electron near field. This **crystal coherence** is already contained in ordinary PXR theory and must be distinguished from quantum coherence of the incident electron.

A shaped electron introduces a quantum interference mechanism. Let the incident electron be prepared as a coherent superposition of transverse momentum states,

$$
|\psi_i\rangle=\sum_m c_m|\mathbf q_m\rangle .

$$

In a crystal, a PXR event may exchange a reciprocal-lattice momentum ($\mathbf g$), so that momentum conservation requires

$$
\mathbf q_m+\mathbf g_m=\mathbf q_f+\mathbf k .

$$

Normally, different ($\mathbf g_m$) correspond to distinguishable final electron states, and their photon-emission probabilities add incoherently. However, the incident electron can be shaped so that two or more initial momenta satisfy

$$
\mathbf q_m-\mathbf q_n = \mathbf g_n-\mathbf g_m .

$$

The different PXR pathways then terminate in the **same final electron-photon state**, and their amplitudes, rather than their probabilities, add. This is the QED interference principle developed for shaped-electron radiation in Ref. [1] and subsequently applied explicitly to bulk-crystal PXR: incident momentum components separated by reciprocal-lattice vectors can coherently combine otherwise distinct electron-recoil pathways.

For the simplest two-state electron,

$$
|\psi_i\rangle = \frac{1}{\sqrt{2}}
\left(
|\mathbf q_1\rangle
+
e^{i\phi}|\mathbf q_2\rangle
\right),

$$

with

$$
\mathbf q_1-\mathbf q_2=\mathbf G_\perp ,

$$

the PXR amplitude into a selected final photon-electron mode is

$$
\mathcal A = \frac{1}{\sqrt{2}}
\left(
M_1+e^{i\phi}M_2
\right).

$$

Allowing for partial transverse coherence through a normalized mutual-coherence factor ($\mu_{12}$), the intensity becomes

$$
I = \frac{1}{2}
\left(
|M_1|^2+|M_2|^2
\right)
+
\operatorname{Re}
\left[
\mu_{12}e^{i\phi}M_2M_1^*
\right].

$$

For approximately equal PXR matrix elements,

$$
|M_1|\simeq |M_2|,

$$

and an optimally chosen relative phase, the enhancement relative to an incoherent mixture of the same two momentum components is approximately

$$
\boxed{
\eta_2
\equiv
\frac{I_{\rm shaped}}{I_{\rm incoh}}
\simeq
1+|\mu_{12}|
}.

$$

Thus a fully coherent two-state electron can give an ideal **factor-of-two enhancement** of the selected PXR mode:

$$
|\mu_{12}|=1
\quad\Longrightarrow\quad
\eta_2=2.

$$

Conversely,

$$
|\mu_{12}|=0
\quad\Longrightarrow\quad
\eta_2=1,

$$

recovering the ordinary incoherent result. This is a **single-electron coherence effect**: coherence between different electrons in the bunch is not required.

The same argument generalizes to ($N_p$) engineered recoil pathways. In density-matrix notation, the radiation probability contains

$$
I_{\rm PXR}
\propto
\sum_{m,n}
\rho_{mn}
M_mM_n^* .

$$

The diagonal terms,

$$
\rho_{mm}|M_m|^2,

$$

represent ordinary incoherent emission, whereas the off-diagonal terms,

$$
\rho_{mn}M_mM_n^*,
\qquad m\neq n,

$$

contain the additional quantum interference produced by the transverse coherence of the electron.

For ($N_p$) approximately equal pathways with optimally aligned phases and average pairwise coherence ($\bar{\mu}$), a useful scaling estimate is

$$
\boxed{
\eta_Q
\simeq
1+
(N_p-1)\bar{\mu}
}.

$$

In the fully coherent limit,

$$
\bar{\mu}\rightarrow1,

$$

so that

$$
\boxed{
\eta_{Q,\max}\simeq N_p
}.

$$

The approximately linear ($N_p$) enhancement follows from normalization of the electron state. For ($N_p$) equally populated components, each amplitude carries a factor ($1/\sqrt{N_p}$). Constructive addition of all ($N_p$) amplitudes therefore increases the net amplitude by ($\sqrt{N_p}$), giving an ($N_p$)-fold increase in intensity relative to the corresponding incoherent mixture.

This should not be confused with the familiar ($N^2$) scaling associated with (N) coherently radiating crystal planes. Conventional PXR already contains that crystal-interference physics. The shaped-electron enhancement is an **additional interference between alternative quantum pathways of the incident electron**.

The transverse coherence length ($\ell_\perp$) determines whether these electron-state interference terms survive. For a simple Gaussian model of transverse mutual coherence, one may write

$$
|\mu(\Delta r_\perp)|
\simeq
\exp
\left[
-\frac{\Delta r_\perp^2}
{2\ell_\perp^2}
\right].

$$

If the relevant transverse spatial scale is taken to be the lattice period corresponding to a transverse reciprocal-lattice vector,

$$
d_\perp = \frac{2\pi} {|\mathbf G_\perp|}

$$

then a simple phenomenological estimate for a two-path enhancement is

$$
\boxed{
\eta_2(\ell_\perp)
\simeq
1+
\exp
\left[
-\frac{d_\perp^2}
{2\ell_\perp^2}
\right]
}.

$$

This gives the expected limits

$$
\ell_\perp\ll d_\perp
\quad\Longrightarrow\quad
\eta_2\rightarrow1,

$$

and

$$
\ell_\perp\gg d_\perp
\quad\Longrightarrow\quad
\eta_2\rightarrow2.

$$

For example,

$$
\ell_\perp=\frac{d_\perp}{2}
\quad\Rightarrow\quad
\eta_2\simeq1.14,

$$

$$
\ell_\perp=d_\perp
\quad\Rightarrow\quad
\eta_2\simeq1.61,

$$

and

$$
\ell_\perp=2d_\perp
\quad\Rightarrow\quad
\eta_2\simeq1.88.

$$

The precise functional dependence is not universal and depends on the actual electron density matrix and preparation method. The Gaussian expression should therefore be treated as a useful scaling model rather than as a fundamental PXR formula.

For a beam whose incoherent rms transverse angular spread is ($\sigma_\theta$), the corresponding transverse coherence length can be estimated from the transverse momentum uncertainty,

$$
\ell_\perp
\simeq
\frac{\hbar}{\sigma_{p_\perp}}.

$$

Using

$$
\sigma_{p_\perp}
\simeq
p,\sigma_\theta,

$$

this becomes

$$
\ell_\perp
\simeq
\frac{\hbar}{p\sigma_\theta} =\frac{\lambda_e}
{2\pi\sigma_\theta},

$$

where ($\lambda_e=h/p$) is the electron de Broglie wavelength.

For example, at approximately ($30~\mathrm{keV}$),

$$
\lambda_e\simeq6.98~\mathrm{pm}.

$$

An incoherent angular width of

$$
\sigma_\theta=1~\mathrm{mrad}

$$

therefore gives

$$
\ell_\perp
\simeq
\frac{6.98~\mathrm{pm}}
{2\pi\times10^{-3}}
\simeq
1.1~\mathrm{nm}.

$$

This is comparable to or larger than typical atomic lattice spacings, suggesting that appreciable transverse quantum coherence can in principle survive for sufficiently low-emittance electron beams.

Importantly, ($\sigma_\theta$) here refers to the **uncontrolled momentum spread around each deliberately prepared momentum component**. It should not be identified with the total angular separation between the components of a shaped electron state. The latter can be deliberately large while the individual components remain mutually coherent.

The resulting shaped-electron PXR spectrum can therefore be represented schematically as

$$
\boxed{
\frac{d^2N_{\rm shaped}}
{d\omega,d\Omega}
\simeq
\eta_Q
\frac{d^2N_{\rm PXR}^{(0)}}
{d\omega,d\Omega}
}.

$$

Here,

$$
\frac{d^2N_{\rm PXR}^{(0)}}
{d\omega,d\Omega}

$$

is the ordinary PXR result, including the crystal susceptibility, structure factor, reciprocal-lattice phase matching, finite interaction length, absorption, extinction, crystal thickness, and ordinary coherent summation over the crystal. The factor

$$
\eta_Q

$$

then represents the **additional quantum enhancement generated by coherent interference between deliberately engineered electron-recoil pathways**.

Transverse coherence alone is therefore not sufficient to enhance PXR. A beam may have a large ($\ell_\perp$), yet behave essentially like an ordinary PXR source if its incident wavefunction contains only a single transverse momentum component. Enhancement requires the electron to be shaped into multiple coherent momentum components whose separations are chosen so that distinct crystal momentum-transfer channels converge on the same final electron-photon state:

$$
\mathbf q_m+\mathbf g_m = \mathbf q_n+\mathbf g_n.

$$

The phases of those components must additionally be arranged for constructive interference in the desired X-ray mode.

For a first experimental implementation, a two-component shaped electron therefore provides the clearest target. The predicted quantum enhancement spans

$$
\boxed{
1
\leq
\eta_2
\leq
2,
}

$$

depending on transverse coherence and phase control. More complex ($N_p$)-component states could in principle extend the enhancement toward

$$
\boxed{
\eta_Q\sim N_p,
}

$$

provided that all of the relevant recoil pathways remain mutually coherent and phase matched.

Very large enhancement factors predicted for free-electron-crystal bremsstrahlung should not be transferred directly to PXR. In those schemes, shaping can cause many lattice-cell scattering amplitudes that would normally contribute incoherently to become mutually coherent. Conventional PXR, in contrast, already derives much of its strength from coherent crystal response. The additional gain available from electron shaping is therefore most naturally understood as the number of otherwise distinct PXR recoil pathways that can be made quantum mechanically indistinguishable.

## References

[1] L. J. Wong, N. Rivera, C. Murdia, T. Christensen, J. D. Joannopoulos, M. Soljačić, and I. Kaminer, “Control of quantum electrodynamical processes by shaping electron wavepackets,” *Nature Communications* **12**, 1700 (2021).

[2] “Free-electron crystals for enhanced X-ray radiation,” *Light: Science & Applications* (2024).

[3] V. G. Baryshevsky, I. D. Feranchuk, and A. P. Ulyanenkov, *Parametric X-Ray Radiation in Crystals: Theory, Experiment and Applications*, Springer Tracts in Modern Physics **213** (Springer, 2005).

[4] A. Balanov, A. Gorlach, and I. Kaminer, “Temporal and spatial design of X-ray pulses based on free-electron–crystal interaction,” *APL Photonics* **6**, 070803 (2021).
