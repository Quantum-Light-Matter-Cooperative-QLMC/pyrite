# `cr-axial-source-partition`: finite-grid transition ownership and on-axis mapping

**Status:** task-owner derivation (2026-11-05), awaiting fresh-context verification.
Code: `src/pyrite/montecarlo/channeling/axial_source_partition.py` and the attenuation
branch of `axial_qrt_source.on_axis_photon_spectrum`.

This claim covers a supplied finite global eigensystem and on-axis observation.
It does not establish a material source, a physical axial dissipator, off-axis
recoil, Hilbert-space convergence or standard-run admission; those stay under
`cr-axial-persisted-channel`.

## Transition ownership

Let $H=\sum_n E_n|n\rangle\langle n|$ and let the positive-frequency current be
$A_\lambda=\sum_{E_i-E_f>\tau}a^\lambda_{fi}|f\rangle\langle i|$. For a declared
boundary $E_b$, let $Q_b$ project on states with $E_n\le E_b$ and
$Q_u=1-Q_b$. Define

$$
A^\mathrm{ax}_\lambda=Q_bA_\lambda,\qquad A^\mathrm{ff}_\lambda=Q_uA_\lambda .
$$

These sum to $A_\lambda$. ADR-0018 assigns bound→bound and free→bound transitions
to channeling radiation and free→free transitions to coherent bremsstrahlung. A
lowering current has $E_f<E_i$, so a free final state implies a free initial state
and $Q_uA_\lambda=Q_uA_\lambda Q_u$: no bound→free block exists. The owner depends
only on the final state. A boundary inside a level with degeneracy tolerance $\tau$
has no basis-independent meaning and is refused.

## Interference

The observed correlation expands into both blocks and a cross term,

$$
G=G^\mathrm{ax,ax}+G^\mathrm{ff,ff}+G^\mathrm{ax,ff}+G^\mathrm{ff,ax}.
$$

For $z\ge z'$,
$G^{ab}_\lambda(z,z')=\mathrm{Tr}[A^a_\lambda e^{\mathcal L(z-z')}(\rho(z')A^{b\dagger}_\lambda)]$.
The operator $X=\rho A^{\mathrm{ff}\dagger}=\rho A^\dagger Q_u$ has its right index in
$Q_u$. Hamiltonian evolution in the eigenbasis only multiplies matrix elements
by phases, so it keeps that index in $Q_u$. Then

$$
\mathrm{Tr}[Q_bA\,e^{-iH\Delta/\hbar v}X\,e^{iH\Delta/\hbar v}]
=\mathrm{Tr}[A\,(\cdots)\,Q_uQ_b]=0
$$

by cyclicity. The other cross term is the conjugate of this one, so the cross
term vanishes identically under unitary evolution, for any incident density and
at every photon energy. A jump term $L_rXL_r^\dagger$ can carry the right index
from $Q_u$ to $Q_b$, so a generator that moves final-state coherence between the
blocks can produce a nonzero signed cross term. The implementation therefore
computes the total from the unsplit current and reports the defect instead of
assuming zero.

## Line shape

The two limits below follow directly from the GKSL generator and the source form
$d^2N/dE\,d\Omega=\alpha E\,(4\pi^2h_v^2)^{-1}\sum_\lambda
\iint dz\,dz'\,e^{iED(z-z')/h_v}G_\lambda(z,z')$, with $h_v=\hbar v$ and
$D=1-\beta$. Let $\delta=(ED-\Delta)/h_v$, with $\Delta$ the Bohr gap.

For excited-state decay at rate $r$ with incident weight in the upper level,
$G=|a|^2e^{-i\Delta(z-z')/h_v}e^{-r(z+z')/2}$. The double integral factorizes:

$$
\left|\frac{1-e^{(i\delta-r/2)\ell}}{r/2-i\delta}\right|^2 .
$$

For pure dephasing of the upper level at rate $r$, $L=\sqrt r|e\rangle\langle e|$,
the population stays constant and
$G=|a|^2e^{-i\Delta(z-z')/h_v}e^{-r|z-z'|/2}$. The result depends on $|z-z'|$ and
is the finite-path Lorentzian of half-width $r/2$ in $\delta$:

$$
\int_0^\ell\!\!\int_0^\ell dz\,dz'\cos[\delta(z-z')]e^{-r|z-z'|/2}.
$$

## Attenuation and on-axis geometry

Beer–Lambert intensity attenuation $e^{-\mu L}$ acts on the photon amplitude as
$e^{-\mu L/2}$, at both depths of the two-time form. On resonance without damping,

$$
\left|\int_0^\ell e^{-\mu(\ell-z)/2}\,dz\right|^2
=\left(\frac{1-e^{-\mu\ell/2}}{\mu/2}\right)^2,
$$

which differs from the intensity-level form $\ell(1-e^{-\mu\ell})/\mu$. Zero
absorption reproduces the unattenuated result exactly.

With inward unit normal $\hat n$, axis $\hat a$ and $\cos\alpha=\hat a\cdot\hat n>0$,
axial depth $z$ lies at normal depth $z\cos\alpha$. A forward photon along
$\hat a$ reaches the exit plane of a slab of thickness $T$ after
$L=(T-z\cos\alpha)/\cos\alpha=T/\cos\alpha-z$. Off-axis rays are refused because
the photon recoil phase changes the Bloch sector. The slab is laterally infinite.

Summing both transverse polarizations is $\sum_\lambda A_\lambda^\dagger A_\lambda$,
which is invariant under a real rotation of the transverse basis. The
unpolarized on-axis yield therefore needs no azimuth mapping.

## Omitted yield of a truncated subspace

At long path, a stationary incident density gives the secular directional rate
per depth

$$
\frac{dN}{dz\,d\Omega}=\frac{\alpha}{2\pi h_vD^2}
\sum_\lambda\sum_{f,i,i'}^{E_i=E_{i'}}
\rho_{ii'}\,a^\lambda_{fi}\,a^{\lambda*}_{fi'}\,(E_i-E_f).
$$

This is the golden-rule form $\alpha E_0|a|^2/(2\pi h_vD)$ with $E_0=(E_i-E_f)/D$.
The omitted rate for a retained set $S$ is the full-grid rate minus the rate with
$i,i',f\in S$. $S$ must hold whole multiplets, or the result depends on the basis
chosen inside a multiplet. The result bounds omission only within the represented
grid. Grid, supercell and Bloch-sector refinement stay outside this claim.

## What native scoring owns

The native CBS kernels carry the denominator $\gamma(\mathbf v\cdot\mathbf g)$, so
they have no resonance at $\mathbf v\cdot\mathbf g=0$. The transverse reciprocal
vectors that form the continuum potential satisfy this condition at exact axial
alignment. The represented free→free block therefore has no native counterpart at
exact alignment, and is an explicit omission of the axial source rather than
something native scoring supplies. A production adapter must report it or bound it.
Zone rows also keep the ADR-0018 convention of passing straight mean segments to
native scorers, so nonzero axial-order $\mathbf g$ and incoherent bremsstrahlung
stay native.
