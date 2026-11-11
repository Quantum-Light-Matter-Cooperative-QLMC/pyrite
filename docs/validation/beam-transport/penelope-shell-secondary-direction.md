# PENELOPE shell secondary emission direction

Validation: `penelope-shell-secondary-direction`. Status: rederived. The
host-side collision sampler returns secondary angles only when its secondary
kinetic energy meets the production threshold. It does not yet enqueue a
secondary track. The shared-frame helper rotates both directions into laboratory
coordinates.

## Source and construction

[PENELOPE-2024](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf)
§3.2.5.4, Eqs. 3.137–3.138, puts the emitted electron along momentum
transfer $\mathbf q$ under the initially stationary target-electron
approximation. Set $M=m_ec^2$ and $p(T)^2=T(T+2M)$ in energy units. For a
longitudinal event, its polar cosine in the incoming-primary frame is

$$
\cos\theta_s=\frac{p(E)^2+p(Q)^2-p(E-W'_k)^2}
{2p(E)p(Q)}.
$$

This is Eq. 3.137 rearranged using $\mathbf q=\mathbf p_0-\mathbf p_1$.
The angular transfer is the modified resonance $W'_k$ for a broadened bound
shell and the resonance $W_{cb}$ for the conduction band, because the
longitudinal $Q$ was sampled with that value. The sampled loss $W$ still
sets the emitted kinetic energy: $W-U_k$ for a substituted inner shell and
$W$ for an outer shell. For a close event, $Q=W$ and Eq. 3.138 reduces to

$$
\cos\theta_s=\sqrt{\frac{W}{E}\frac{E+2M}{W+2M}}.
$$

The secondary azimuth is $(\phi+\pi)\bmod 2\pi$. The primary's azimuth
$\phi$ is uniform. A distant transverse event has no modeled primary
deflection or sampled $Q$. Its secondary cosine is fixed to $0.5$, following
the [Geant4 Penelope ionisation implementation](https://apc.u-paris.fr/~franco/g4doxy/html/G4PenelopeIonisationModel_8cc-source.html#l00740);
this is a model convention rather than a consequence of Eq. 3.137. The
same opposite-azimuth convention supplies its otherwise arbitrary plane.

## Limits and checks

The close cosine is in $[0,1]$ and approaches zero as $W\to0$ at fixed $E$. At the
minimum longitudinal recoil $Q_-$, the transfer is longitudinal and the
cosine approaches one. All returned cosines are clipped only for roundoff.
No direction is returned when the secondary is not emitted.

## Laboratory-frame handoff

`shell_collision_world_directions` rotates the primary and secondary with the
same orthonormal frame about the incoming unit direction. It reuses the
transport core's `_rotate_direction_scalar` convention. If the frame vectors
are $\mathbf u,\mathbf v,\mathbf d$ with $\mathbf d$ the incoming direction,
each returned vector is

$$
\mathbf d'=\cos\theta\,\mathbf d+\sin\theta
(\cos\phi\,\mathbf u+\sin\phi\,\mathbf v).
$$

Using $\phi_s=\phi+\pi$ in this *same* frame makes the two transverse
components antiparallel. The result is a unit vector with
$\mathbf d'\cdot\mathbf d=\cos\theta$; a forward primary retains the flight
direction, and a suppressed secondary has no world direction. The helper
rejects nonunit or nonfinite input directions. The two directions are not yet
consumed by a transport scheduler, so no flight segment or secondary track is
created here.

`tests/montecarlo/collisions/test_shell_partition.py` checks every nonzero fixture
branch at three recoil quantiles, the momentum-triangle expression, close
limit, opposite azimuth, transverse convention, and suppression above the
production threshold. The shell loss sampler's distant-triangle source
conflict remains in `penelope-shell-hard-loss-sampling`; secondary angular
agreement does not resolve it. The independent conditional-angle verification below matches the source;
transport scheduling and CPU/GPU comparison remain outside this verification. The shared-frame
rotation is checked for on-axis and tilted incoming flights, including unit
norms, polar cosines, opposite transverse components, and a suppressed
secondary.

## Independent verification (2026-11-11, issue #1101)

### Source-first derivation recorded before implementation inspection

This verifier read only the selected ledger row, AST-extracted function signatures
and docstrings, and original sources before writing this subsection. The existing
write-up and implementation bodies were not read until after this expression was
recorded. The reader's need is independent evidence; this ledgered page is its
canonical owner.

Sources: [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§3.2.5.4, printed pp. 135–136 (PDF pp. 155–156), Eqs. 3.137–3.138;
and [Geant4 v11.3.2 source](https://github.com/Geant4/geant4/blob/v11.3.2/source/processes/electromagnetic/lowenergy/src/G4PenelopeIonisationModel.cc),
`SampleFinalStateElectron`, lines 893–897 for the transverse convention.
The PDF equations and surrounding paragraphs were visually inspected.

Let $M=m_ec^2$, $P(T)=cp(T)=\sqrt{T(T+2M)}$, and
$R=cq=\sqrt{Q(Q+2M)}$. Here $E$, $W$, $Q$, and $M$ are energies in the
same unit; $Q$ parameterizes momentum transfer and need not equal the energy
lost $W$. For the angular approximation let $w=W'_k$ in a distant bound-shell
event, $w=W_k$ for conduction, and $w=W$ for a close event. The source
explicitly requires the resonance substitution in distant events, including when
binding and sampled energy accounting use a different loss.

From $\mathbf q=\mathbf p-\mathbf p_f$, the momentum triangle gives

$$
P(E-w)^2=P(E)^2+R^2-2P(E)R\cos\theta_s,
\qquad
\boxed{\cos\theta_s=\frac{P(E)^2+R^2-P(E-w)^2}{2P(E)R}}.
$$

Expanding the numerator and using $\beta=P(E)/(E+M)$ gives

$$
\cos\theta_s=
\frac{w}{\beta R}\left[1+\frac{R^2-w^2}{2w(E+M)}\right].
$$

Squaring reproduces Eq. 3.137. The positive root follows from the momentum
triangle in the allowed recoil interval, rather than an arbitrary choice after
squaring. Setting $Q=W$ yields

$$
\boxed{\cos\theta_s=\sqrt{\frac{W(E+2M)}{E(W+2M)}}},
$$

the source's Eq. 3.138. Since the primary transverse momentum and transfer
transverse momentum are opposite, $\phi_s=(\phi+\pi)\bmod 2\pi$.
For distant transverse events, no finite momentum triangle is retained by the
implementation convention: Geant4 assigns primary cosine $1$ and secondary
cosine $1/2$. This is a prescribed convention, not a consequence of Eq. 3.137.

For any orthonormal incoming frame $(\mathbf u,\mathbf v,\mathbf d)$, both
particles must use the same rotation

$$
\mathbf d'(\mu,\phi)=\mu\mathbf d+
\sqrt{1-\mu^2}(\cos\phi\,\mathbf u+\sin\phi\,\mathbf v).
$$

It has unit norm and $\mathbf d'\cdot\mathbf d=\mu$. The secondary must
be rotated about the incoming direction, rather than the outgoing primary.
Uniform azimuth means density $1/(2\pi)$; there is no per-solid-angle density
or extra Jacobian to apply to this deterministic direction mapping.
A suppressed secondary has no physical emitted direction in this API.

Cheap filters before bodies: energy-squared ratios are dimensionless; $Q=Q_-$
means $R=P(E)-P(E-w)$ and gives forward secondary emission; at fixed $E$,
close $W\to0$ gives transverse emission; in the nonrelativistic limit the close
cosine tends to $\sqrt{W/E}$; for $E,W\gg M$ it tends to one. Opposite
azimuth and the common rotation preserve the scattering plane and signed
transverse correlation. These checks pass within the stated angular model,
which omits bound-electron Doppler broadening and treats binding separately.

### Comparison after recording the derivation

Reviewed revision `032ec80ccf762443e4fdc814085204ea817a5e41`. Both named
owners carry the correct validation marker. `sample_shell_hard_collision`
implements the independently derived momentum triangle literally; its distant
bound-shell substitution matches Eq. 3.78 on printed p. 122, including the
energy-dependent branch. Conduction uses its unchanged resonance. The close
branch implements Eq. 3.138 with transfer $W$, without subtracting binding
inside the angle. The distant transverse branch uses exactly the explicitly
identified Geant4 convention. All branches return the opposite azimuth only
when an emitted secondary exists. The delegated loss sampler suppresses
nonpositive or below-threshold secondary energy; an exactly positive threshold
energy is included. This is consistent with the documented API boundary.

`_rotate_direction_scalar` constructs a unit transverse vector from the cross
product of the incoming vector with a selected Cartesian reference, and the
second transverse vector from their cross product. Selection avoids a parallel
reference. Both calls in `shell_collision_world_directions` use the same incoming
components, so they construct the same frame, including for negative axial
incoming directions. Output normalization preserves direction. No independent
rotation about the deflected primary is introduced. The missing-secondary and
missing-azimuth branches are explicit, and malformed incoming vectors are
rejected. The $10^{-10}$ input norm tolerance is numerical input validation,
not a physical angular uncertainty; normalization can change the nominal polar
cosine by that order for inputs at its boundary.

Units and notation agree among the ledger equation, docstrings, and original
construction: electronvolts throughout, $P$ and $R$ in energy units, cosines
unitless, azimuths in radians, and incoming/output vectors unitless. There are
no tabulated reference values or density conversions in this claim. The
conditional angular model is independently verified here; the chosen distant
loss law and channel probabilities remain owned by separate ledger claims.
The fixed transverse cone is not an independently measured angular distribution.

### Independent numeric comparison and tolerance provenance

A scratch reference evaluated the unsquared expanded Eq. 3.137 at 60 decimal
digits using Python `decimal`, independently of implementation angle helpers.
It used returned $Q$ as the conditional input and source Eq. 3.78 for $w$;
therefore it tests angles rather than independently recertifying the recoil
sampler. Both electron and positron partitions were exercised with the silicon
fixture at $E=40,200,500,1000,2000,5000,10000,10^8$ eV, all active shell/channel
pairs, loss quantiles $0.01,0.5,0.99$, and recoil quantiles
$0,0.25,0.75,\operatorname{nextafter}(1,0)$. The mass energy was
$M=510998.9506917532$ eV from the installed SciPy constants; it is a shared
physical constant, not a fitted angular reference.

| Angular branch | Comparisons per projectile | Largest absolute cosine difference |
| --- | ---: | ---: |
| Longitudinal, unchanged resonance | 324 | $2.861\times10^{-10}$ |
| Longitudinal, modified resonance | 48 | $4.441\times10^{-16}$ |
| Distant transverse | 372 | $0$ |
| Close | 396 | $1.111\times10^{-16}$ |

The worst case is the conduction oscillator at $E=10^8$ eV,
$Q=0.000272894110888159$ eV: the reference cosine is
$0.999999999999999$, versus $0.9999999997139418$. Subtraction of nearly equal
squared primary momenta is ill-conditioned here. An initial scratch criterion
of $10^{-10}$ absolute failed; it was replaced by an arithmetic error bound,
not a physical/model uncertainty allowance. With
$A=P(E)^2$, $B=P(E-w)^2$, $C=R^2$, the conservative first-order FP64 bound used
was

$$
\lvert\Delta\mu\rvert\leq
16\epsilon\frac{A+B+C}{2\sqrt{AC}},
\qquad \epsilon=2^{-52}.
$$

The factor 16 bounds the small number of multiplications, additions, square
roots, and division, including input rounding; it is not calibrated to the
observed residual. All longitudinal residuals were below this bound; the
largest ratio to it was $0.052$. Other branches used $16\epsilon$ absolute.
This supports algebraic agreement, while exposing high-energy cancellation
that a future numerical-stability change could reduce. The existing anchor's
$10^{-10}$ absolute tolerance is suitable for its 10 keV fixture; it should
not be read as a uniform high-energy accuracy guarantee.

Independent unit-norm and incoming-dot-product checks at incoming
$(0,0,1)$, $(1,0,0)$, $(-1,0,0)$, and $(1,1,1)/\sqrt3$ gave maximum absolute
error $4.441\times10^{-16}$. A $10^{-14}$ arithmetic tolerance is justified
by the short FP64 vector calculation. Existing anchors additionally check
opposite transverse components, suppression, and invalid inputs.

The focused canonical command
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/collisions/test_shell_partition.py`
passed: **100 tests in 6.95 s**. No source or test was modified. Scratch probes
are session evidence, not a committed regression or CPU/GPU parity artifact.
No CPU/GPU or external runtime comparison is claimed. The Geant4 source bytes
used have SHA-256 `362f393283e2c4d10a348ae14bb561bc2c9c8641ff606aa0ee7b841fa47b297f`;
the original PDF bytes have SHA-256
`06b4bc6eade9f0057f1c6caf8b3d0bd7b2fbc565f1e2db2a583065449810e76a`.

### Verdict and scope

**Re-derived: matches.** Units, limits, signs, normalization, source equations,
and all active angular branches pass within the stated approximation. Suggested
ledger change: `unverified` to `rederived`, record this independent derivation
and the focused test result, and remove the note that independent angular
validation remains open. The owner may separately assess the anchor evidence
for `anchored`; only a human may apply `signed-off`.

Two prose qualifications do not change the verified equations. First, the
historical introduction incorrectly stated that neither direction was rotated;
the owner corrected it to describe the existing shared-frame helper. Track
scheduling remains outside this host-side API. Second, the shorthand close limit
$W/E\to0$ means $W\to0$ at fixed $E$ (or a nonrelativistic ratio limit).
Taking $E\to\infty$ at fixed nonzero $W$ instead gives
$\sqrt{W/(W+2M)}$, not zero. The explicit Eq. 3.138 is correct. The owner qualified the earlier limit statement accordingly.

Documentation tests and rendered-math inspection are delegated to the task
owner's integrated build, as specified in this verifier handoff. This report
makes no claim that the rendered page was checked in the verifier worktree.
