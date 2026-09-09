# Photon escape and in-medium dispersion

Both radiation kernels emit inside the sample, so every photon must be
transported out of it. PyRITE does that with a straight-ray Beer--Lambert model
evaluated from the emitting segment's midpoint, plus an optional bulk refractive
correction to the photon's dispersion relation and propagation phase. Nothing
here feeds back on emission: attenuation is passive.

## Attenuation coefficient

For an element of number density $n$ and Henke/Chantler imaginary scattering
factor $f_2(E)$, the intensity attenuation coefficient follows from
$\beta_{\rm idx}=r_e\lambda^2 n f_2/(2\pi)$ and $\mu=2k\beta_{\rm idx}$:

$$
\mu(E)=2\,r_e\,\lambda\,n\,f_2(E)\quad[\text{\AA}^{-1}],
\qquad I(z)=I(0)e^{-\mu z}.
$$

Compounds add inverse lengths, $\mu=\sum_i \mu_i$ through $\sum_i n_i f_{2,i}$.
The absorber composition defaults to the crystal's own basis at its total atom
density — exact for elemental crystals — and can be given explicitly as
`[(element, n_per_Ang3), ...]`.

The line kernel needs $\mu$ at each segment's own resonance energy, which would
otherwise mean a host round-trip per segment. Instead each element's
$\log\mu_i$ is tabulated and interpolated **linearly in $\log E$**, then summed:
that reproduces the pinned xraydb rule for non-`f1` Chantler data, whereas
interpolating the compound total in either linear or log space does not. The
shared grid is a 1 eV mesh unioned with the native Chantler nodes of every basis
*and* explicitly named absorber element, so edge jumps — tens of percent at, say,
the C K-edge — are resolved rather than smeared. Layered and grooved escape keep
exact per-point $\mu$ instead of the tabulation.

Outside the tabulated range the coefficient is unavailable. The line kernel lets
NaN propagate and drops the segment on its finite mask; the wide bremsstrahlung
grid instead treats an unavailable $\mu$ as zero, i.e. fully transparent. See
[Bremsstrahlung](bremsstrahlung.md) for what that means at the extremes of a wide
grid.

## Escape geometry

The optical depth is $\tau=\mu\,L_{\rm esc}$ and the transmission
$T_{\rm abs}=e^{-\tau}$, applied to the intensity in the incoherent kernels and
as $\sqrt{T_{\rm abs}}$ to the field in the coherent one. $L_{\rm esc}$ is the
straight-line distance from the segment midpoint to the sample boundary along the
fixed observation direction $\hat{\mathbf n}$, in one of four geometries:

| Geometry | Escape path |
|---|---|
| Flat slab (default) | $z_{\rm mid}/\lvert n_z\rvert$ out the entrance face when $n_z<0$, else $(d-z_{\rm mid})/n_z$ out the back |
| Finite footprint | nearest of the rectangular prism's six faces along $\hat{\mathbf n}$ |
| Layered stack | piecewise $\sum_i \mu_i\,\Delta z_i$ across every crossed layer, with $\Delta z_i$ scaled by $1/\lvert n_z\rvert$ |
| Blazed groove | exact periodic ray--plane distance to the nearest working facet |

The layered path is what makes a film-on-substrate sample predictive: a soft line
born in the film is attenuated by the whole stack, and for lines below ~4.5 keV
the substrate is often optically thick enough to dominate the correction. See
[Multilayer materials](../materials/multilayer-materials.md).

For the blazed groove, the working facet is perpendicular to $\hat{\mathbf n}$
and the relief facet is perpendicular to the beam, so a photon leaving through
its first working-facet crossing cannot re-enter material later — the first
crossing *is* the complete material path. This holds only for the exact
working-facet normal $\hat{\mathbf n}=(\cos t_p,0,-\sin t_p)$; other directions,
and any combination with layers, raise rather than silently using a flat path.
Grooving is purely an absorption-path effect in this model: emission amplitudes
and resonance kinematics are untouched, and there is no wave-optical diffraction
off the groove edges. A finite crystal footprint is permitted but only classifies
launch hit/miss in transport; the groove escape treats the sawtooth as laterally
periodic.

## In-medium dispersion

The line kinematics always run on the bulk crystal dielectric response; there
is no vacuum-dispersion setting, and $k=\omega$ is recovered only in the
physical $\chi_0\to0$ (high-energy) limit. The $g=0$ susceptibility

$$
\chi_0(E)=-\frac{r_e\lambda^2}{\pi V_{\rm cell}}\sum_i\bigl[(Z_i+f'_i)+if''_i\bigr]
$$

is $\chi_{\mathbf g}$ at $\mathbf g=0$, where every Debye--Waller factor and basis
phase collapses to unity, and uses the Henke convention $f_1=Z+f'$ so that it
shares one normalization with the absorption model exactly. The Maxwell relation
$k^2=(1+\chi_0)\omega^2$ then gives

$$
n(E)=\sqrt{1+\chi_0(E)}\;\approx\;1-\delta-i\beta,
$$

with the square root taken exactly rather than linearized. In the X-ray regime
$\delta\sim10^{-5}$--$10^{-3}$: negligible per Ångström, but it accumulates over
micron-scale trajectories, which is exactly what this model tracks.

**Only $\mathrm{Re}\,n$ is applied.** $\mathrm{Im}\,n$ is the same absorption
already carried by the Beer--Lambert $\mu(E)$, so folding it in here as well
would double count it.

### Resonance

With $k=n(\omega)\,\omega$ along the observation direction, the resonance
condition becomes implicit:

$$
\omega_{\rm res}=\frac{\mathbf v\cdot\mathbf g}
{1-\mathrm{Re}\,n(\omega_{\rm res})\,(\mathbf v\cdot\hat{\mathbf n})} .
$$

It is solved by fixed-point iteration from the vacuum root. In the X-ray regime
the map's derivative is of order $\delta\sim10^{-5}$, so each pass gains about
five digits and two are already at float64 rounding; three are taken for margin.

**That contraction is conditional, and it is checked rather than assumed.** The
rate above rests on $\mathrm{Re}\,n=1-\delta$, which only holds off-edge in the
X-ray regime. A segment scattered nearly perpendicular to $\mathbf g$ puts the
vacuum root down in the optical/UV, where the tabulations honestly carry
$\mathrm{Re}\,n>1$ (carbon: $6.24$--$285$ eV, peaking at $4.766$). There
$\mathrm{Re}\,n\,(\mathbf v\cdot\hat{\mathbf n})$ can approach unity, the
denominator collapses toward a spurious Cherenkov-like zero, and the map becomes
an expansive 2-cycle rather than a contraction — three passes then return
whichever half of the cycle the last pass landed on. So the last pass must move
the denominator by less than a relative $10^{-3}$; pairs that fail carry NaN out
and drop on the same finite mask as out-of-range tabulation energies. A genuine
contraction moves it by $\sim\delta^3$, five orders inside the tolerance. This
is rejection, not repair: such samples violate the CBS amplitude's own
perturbative validity condition
$\lvert U_{\mathbf g}\rvert g^2/(\gamma mc^2(\mathbf v\cdot\mathbf g)^2)\ll1$, so
there is no correct value to compute for them. At the catalog's 1000 Å
production thickness the guard rejects no pairs at all; it fires only in thick,
fast, many-segment cases (0.233% at $10^6$ Å).

The substitution leaves
every kinematic identity intact — $\mathbf k\cdot\mathbf v=\omega(1-{\rm denom})$
still holds exactly, while $\mathbf k\cdot\mathbf g$ picks up one power of
$\mathrm{Re}\,n$ and the PXR numerator's $k^2$ two. Out-of-range tabulation
energies carry NaN out of $\mathrm{Re}\,n$ and drop the segment on the caller's
finite mask, matching the $\chi$/$U$/$\mu$ convention.

### Propagation phase

Under the coherent policy the same dispersion relation moves the
segment-to-segment propagation phase: each segment's field picks up
$-\delta(E)\,\omega(E)\,L_{{\rm esc},j}$ over its in-crystal escape path, the
real partner of the amplitude factor $\sqrt{T_{\rm abs}}$ applied over that same
path. This is refused for layered absorbers, whose per-layer $\delta$ is not
modelled.

That the phase runs over the *escape* path, and not some other length, is not a
convention. The observation-time phase is
$\omega\,(t_j + n\,L_{{\rm esc},j} + L_{\rm vac},j)$, and to first order the
geometric total $L_{\rm esc}+L_{\rm vac}$ is $R-\hat{\mathbf n}\cdot\mathbf r_j$,
so the vacuum term $\omega d_j$ (with $d_j=t_j-\hat{\mathbf n}\cdot\mathbf r_j$)
picks up exactly the excess

$$
\omega\bigl(\operatorname{Re}n(E)-1\bigr)L_{{\rm esc},j}
=-\delta(E)\,\omega(E)\,L_{{\rm esc},j}.
$$

Equivalently, the escape leg contributes
$e^{\,i n \omega L}=e^{\,i\omega L}\,e^{-i\delta\omega L}\,e^{-\beta\omega L}$,
whose last factor is $\sqrt{e^{-\mu L}}$ — the Beer–Lambert amplitude the
coherent path already applies. Only the two together are one complex $n$.

This is deliberately **not** $k(E)\,\hat{\mathbf n}\cdot\mathbf r_j$, which would
charge the medium's index for the whole flight to the detector. The two agree
only when the photon exits along the face normal, where $L_{\rm esc}$ and
$\hat{\mathbf n}\cdot\mathbf r$ differ by a segment-independent constant — i.e. by
a global phase. The term is tabulated on the **output** grid, because it is a
propagation phase read across the whole spectrum rather than a coupling frozen
at the line energy.

## Assumptions and limits

- straight photon rays: no refraction at interfaces, no Fresnel reflection or
  transmission, no diffraction off groove edges;
- **bulk response only** — grazing observation geometry, where interface optics
  dominate, is out of scope for the refractive model;
- passive attenuation: absorbed photons are gone, with no fluorescence,
  re-emission, or scattering into the detector direction;
- the escape path is taken from the segment **midpoint**, consistent with the
  finite-time factor's constant-velocity segment;
- the in-medium resonance is only reported where its fixed point **converges**;
  segment/reflection pairs whose root lands in the near-Cherenkov regime are
  dropped rather than approximated, because the emission amplitude's own
  perturbative expansion has failed there;
- attenuation does not feed back on emission, and emission does not deplete the
  incident beam;
- interfaces are sharp, static, and perpendicular to $z$;
- the refractive model is bit-for-bit inert when unselected, so vacuum runs are
  unaffected by its presence.

## Validation

Ledger rows: `absorption-length` (`anchored`) for $\mu$ itself,
`line-absorption-tabulation` and `self-absorption` for the interpolated and
layered escape, `multilayer-stack` for the stack, `finite-transverse-crystal` for
the prism escape, `blazed-groove-geometry` (**`unverified`**) for the groove, and
`xray-chi-zero`, `xray-refractive-index`, `xray-in-medium-resonance`,
`xray-in-medium-propagation-phase` for the dispersion model. All are `rederived`
or better except the groove row; none is human `signed-off`. Consult the
[validation ledger](../../validation/physics-validation-ledger.md) before
scientific use.

Implementation owners: `pyrite.materials.crystal.absorption_length_ang`,
`pyrite.materials.crystal.chi_0`, `pyrite.materials.crystal.refractive_index`,
`pyrite.materials.attenuation`, `pyrite.montecarlo.groove.escape_distance_ang`,
and the escape/dispersion helpers in `pyrite.montecarlo.spectrum.lines`.
