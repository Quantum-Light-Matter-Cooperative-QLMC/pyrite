# Photon escape and in-medium dispersion

Line, bremsstrahlung, and characteristic radiation use straight-ray Beer--Lambert attenuation from each point of a segment to the sample boundary, averaged or integrated along the segment rather than taken at its midpoint. The PXR/CBS line model also uses bulk refractive dispersion in its resonance and coherent propagation phase. Attenuation is passive and does not feed back on emission.

## Attenuation coefficient

Escape is a **narrow-beam** problem: a photon counts as escaped only if it reaches the surface unscattered, so every interaction channel removes it, not only the ones that absorb it. The coefficient is therefore the total

$$
\mu(E)=n\bigl[\sigma_{\rm photo}+\sigma_{\rm coh}+\sigma_{\rm incoh}
      +\sigma_{\rm pair,nuc}+\sigma_{\rm pair,el}\bigr](E)
  \quad[\text{\AA}^{-1}],
\qquad I(z)=I(0)e^{-\mu z}.
$$

Every per-atom cross section comes from one evaluated library, EPDL2025 (the EPICS2025 photo-atomic data of D. E. Cullen, `NDS-IAEA-225`){cite:p}`epdl2025,cullen1997epdl`, for Z = 1--100 from 1 eV to 100 GeV. The two pair terms are zero below their thresholds, $2m_ec^2=1.022$ MeV in the nuclear field and $4m_ec^2$ in the electron field. The cross sections are evaluated lin-lin between the library's knots, as the ENDF file declares, and are right-continuous at each photoionization edge. Against NIST XCOM{cite:p}`berger2010xcom` the total agrees within 0.5% for light and mid-Z elements and within 3% for W and Pb, away from edges (Validation: `narrow-beam-total-attenuation`). No build-up factor is applied: scattered photons leave the ray and are not returned to it. Pairs are a removal term only; they are not created or transported.

This $\mu$ is deliberately **not** the photoabsorption of the refractive index. The PXR/CBS couplings, $\beta_{\rm idx}=r_e\lambda^2 n f_2/(2\pi)$ and the Si sensor response keep the Chantler/FFAST $f_2$ (`absorption-length`). The two compilations differ by a few percent in photoabsorption, more within a few eV of an edge, because each places its edges at its own energies (C K: 288 eV in EPDL, 283.8 eV in Chantler). Grid refinement and line-window seeding locate edges in the table each quantity reads.

Compounds add inverse lengths, $\mu=\sum_i \mu_i$. The absorber composition defaults to the crystal's own basis at its total atom density — exact for elemental crystals — and can be given explicitly as `[(element, n_per_Ang3), ...]`.

The line kernel needs $\mu$ at each segment's own resonance energy, which would otherwise mean a host round-trip per segment. Instead each element's $\log\mu_i$ is tabulated and interpolated **linearly in $\log E$**, then summed; interpolating the compound total instead does not reproduce a sum of elemental terms. The shared grid is a 1 eV mesh unioned with the native Chantler nodes of every basis and absorber element, every EPDL knot of each absorber, and a float32-adjacent node pair at each EPDL edge. No mesh interval therefore straddles an EPDL slope change, and the one interval that straddles an edge jump is one ulp wide. The residual against direct evaluation is below $10^{-4}$ relative, and node values are exact. Layered and grooved escape keep exact per-point $\mu$ instead of the tabulation.

Outside 1 eV -- 100 GeV the coefficient is unavailable (NaN). The line kernel drops such segments on its finite mask. The continuum scorers (wide bremsstrahlung grid, hard-photon events, characteristic escape) raise instead of reading NaN as transparency; only sub-eV nodes of a grid that was not floored keep $\mu=0$. See [Bremsstrahlung](bremsstrahlung.md).

## Escape geometry

The optical depth is $\tau=\mu\,L_{\rm esc}$ and the transmission $T_{\rm abs}=e^{-\tau}$. $L_{\rm esc}$ is the straight-line distance from an emission point to the sample boundary along the fixed observation direction $\hat{\mathbf n}$. Each segment is cut into pieces on which $L_{\rm esc}$ is affine (`segment-escape-average`). Incoherent emitters -- characteristic lines, bremsstrahlung and the incoherent PXR/CBS route -- weight their intensity by the segment mean $\langle e^{-\tau}\rangle$. The phased-field line reductions (coherent route, flight-grouped reduction) instead damp the field by $e^{-\tau/2}$ inside each piece's formation integral (`coherent-formation-absorption`), whose integral over the line is the same mean. The four escape geometries are:

| Geometry | Escape path |
|---|---|
| Flat slab (default) | $z_{\rm mid}/\lvert n_z\rvert$ out the entrance face when $n_z<0$, else $(d-z_{\rm mid})/n_z$ out the back |
| Finite footprint | nearest of the rectangular prism's six faces along $\hat{\mathbf n}$ |
| Layered stack | piecewise $\sum_i \mu_i\,\Delta z_i$ across every crossed layer, with $\Delta z_i$ scaled by $1/\lvert n_z\rvert$ |
| Blazed groove | exact periodic ray--plane distance to the nearest working facet |

In a stack, emission is attenuated by every layer crossed on the escape path. See [Multilayer materials](../materials/multilayer-materials.md).

For the blazed groove, the working facet is perpendicular to $\hat{\mathbf n}$ and the relief facet is perpendicular to the beam, so a photon leaving through its first working-facet crossing cannot re-enter material later — the first crossing *is* the complete material path. This holds only for the exact working-facet normal $\hat{\mathbf n}=(\cos t_p,0,-\sin t_p)$; other directions, and any combination with layers, raise rather than silently using a flat path. Grooving is purely an absorption-path effect in this model: emission amplitudes and resonance kinematics are untouched, and there is no wave-optical diffraction off the groove edges. A finite crystal footprint is permitted but only classifies launch hit/miss in transport; the groove escape treats the sawtooth as laterally periodic.

## In-medium dispersion

The line kinematics always run on the bulk crystal dielectric response; there is no vacuum-dispersion setting, and $k=\omega$ is recovered only in the physical $\chi_0\to0$ (high-energy) limit. The $g=0$ susceptibility

$$
\chi_0(E)=-\frac{r_e\lambda^2}{\pi V_{\rm cell}}\sum_i\bigl[(Z_i+f'_i)+if''_i\bigr]
$$

is $\chi_{\mathbf g}$ at $\mathbf g=0$, where every Debye--Waller factor and basis phase collapses to unity, and uses the Henke convention $f_1=Z+f'$ so that it shares one normalization with the absorption model exactly. The Maxwell relation $k^2=(1+\chi_0)\omega^2$ then gives

$$
n(E)=\sqrt{1+\chi_0(E)}\;\approx\;1-\delta-i\beta,
$$

with the square root taken exactly rather than linearized. In the X-ray regime $\delta\sim10^{-5}$--$10^{-3}$: negligible per Ångström, but it accumulates over micron-scale trajectories, which is exactly what this model tracks.

Only $\mathrm{Re}\,n$ enters the resonance and propagation phase. Photoabsorption is already included in the Beer--Lambert factor and must not be applied a second time through a complex wavevector. The optical constants use photoabsorption data; total narrow-beam attenuation also includes scattering.

### Resonance

With $k=n(\omega)\,\omega$ along the observation direction, the resonance condition becomes implicit:

$$
\omega_{\rm res}=\frac{\mathbf v\cdot\mathbf g}
{1-\mathrm{Re}\,n(\omega_{\rm res})\,(\mathbf v\cdot\hat{\mathbf n})} .
$$

The implementation takes three fixed-point iterations from the vacuum root. Away from absorption edges in the X-ray regime, the refractive correction is small and iteration converges rapidly. This is not guaranteed near low-energy roots where the refractive index can differ substantially from unity.

The final denominator update must be below a relative $10^{-3}$. Pairs that fail carry NaN and are dropped by the finite-value mask, as are out-of-range tabulation energies. This guard rejects an unsettled root; it does not repair the resonance or extend the perturbative emission model into that regime. See [Validation: `xray-in-medium-resonance`](../../validation/radiation-physics/xray-in-medium-resonance.md).

The substitution leaves every kinematic identity intact — $\mathbf k\cdot\mathbf v=\omega(1-{\rm denom})$ still holds exactly, while $\mathbf k\cdot\mathbf g$ picks up one power of $\mathrm{Re}\,n$ and the PXR numerator's $k^2$ two. Out-of-range tabulation energies carry NaN out of $\mathrm{Re}\,n$ and drop the segment on the caller's finite mask, matching the $\chi$/$U$/$\mu$ convention.

### Propagation phase

Under the coherent policy the same dispersion relation moves the segment-to-segment propagation phase: each segment's field picks up $-\delta(E)\,\omega(E)\,L_{{\rm esc},j}$ over its in-crystal escape path, the real partner of the amplitude factor $e^{-\tau/2}$ applied over that same path. This is refused for layered absorbers, whose per-layer $\delta$ is not modelled.

Both factors vary along a segment, and both are integrated along each linear escape piece rather than frozen at its midpoint: the piece field is $t_L e^{i\Phi_c}F$ with $F=e^{-\tau_c/2}\sinh w/w$, $w=iv-q$, $q=(\tau_{\rm end}-\tau_{\rm start})/4$ and $v=a_{\rm vac}(E-E_{\rm vac})-\delta\omega\,\Delta L_{\rm esc}/2$ on the vacuum sinc centre and width. Pieces of a straight flight therefore sum to it exactly. The coherent line centre is where this full phase is stationary; for a flat exit face that is the Snell-refracted root, which differs from the bulk resonance above by $O(\delta)$ except at normal exit. The incoherent route keeps the bulk root. See [Validation: `coherent-formation-absorption`](../../validation/radiation-physics/coherent-formation-absorption.md).

That the phase runs over the *escape* path, and not some other length, is not a convention. The observation-time phase is $\omega\,(t_j + n\,L_{{\rm esc},j} + L_{\rm vac},j)$, and to first order the geometric total $L_{\rm esc}+L_{\rm vac}$ is $R-\hat{\mathbf n}\cdot\mathbf r_j$, so the vacuum term $\omega d_j$ (with $d_j=t_j-\hat{\mathbf n}\cdot\mathbf r_j$) picks up exactly the excess

$$
\omega\bigl(\operatorname{Re}n(E)-1\bigr)L_{{\rm esc},j}
=-\delta(E)\,\omega(E)\,L_{{\rm esc},j}.
$$

The implementation keeps real propagation and attenuation separate. The escape factor is $e^{i\,\operatorname{Re}n\,\omega L}\sqrt{T_{\rm abs}} =e^{i\omega L}e^{-i\delta\omega L}e^{-\tau/2}$, at each emission point. Here $\tau$ includes total narrow-beam attenuation, not only the photoabsorption associated with the imaginary optical index. Keeping the factors separate also avoids mixing the sign convention for complex $n$ with the convention for the propagated field.

This is deliberately **not** $k(E)\,\hat{\mathbf n}\cdot\mathbf r_j$, which would charge the medium's index for the whole flight to the detector. The two agree only when the photon exits along the face normal, where $L_{\rm esc}$ and $\hat{\mathbf n}\cdot\mathbf r$ differ by a segment-independent constant — i.e. by a global phase. The term is tabulated on the **output** grid, because it is a propagation phase read across the whole spectrum rather than a coupling frozen at the line energy.

## Assumptions and limits

- straight photon rays: no refraction at interfaces, no Fresnel reflection or transmission, no diffraction off groove edges;
- **bulk response only** — grazing observation geometry, where interface optics dominate, is out of scope for the refractive model;
- passive attenuation: absorbed photons are gone, with no fluorescence, re-emission, or scattering into the detector direction;
- the escape path is **affine on each piece** of a segment, which makes the segment-mean escape (incoherent) and the per-piece formation integral (coherent) exact; emission amplitudes and $\mu$ stay frozen at the segment's line energy;
- the in-medium resonance is only reported where its fixed point **converges**; segment/reflection pairs whose root lands in the near-Cherenkov regime are dropped rather than approximated, because the emission amplitude's own perturbative expansion has failed there;
- attenuation does not feed back on emission, and emission does not deplete the incident beam;
- layer interfaces are sharp, static, and perpendicular to $z$;
- vacuum dispersion is recovered in the physical limit $\chi_0\to0$; there is no runtime switch to disable the bulk response.

## Validation

Ledger rows: `absorption-length` for photoabsorption and `narrow-beam-total-attenuation` for total removal, `line-absorption-tabulation` and `self-absorption` for the interpolated and layered escape, `multilayer-stack` for the stack, `finite-transverse-crystal` for the prism escape, `blazed-groove-geometry` (**`unverified`**) for the groove, and `xray-chi-zero`, `xray-refractive-index`, `xray-in-medium-resonance`, `xray-in-medium-propagation-phase` for the dispersion model; `segment-escape-average` and `coherent-formation-absorption` for the escape along a segment.

The propagation-phase derivation record still contains a complex-index factorization with an inconsistent attenuation sign; the separate real-phase and transmission factors above match the implementation. That record needs correction before sign-off. Consult the [validation ledger](../../validation/physics-validation-ledger.md) before scientific use.

Implementation owners: `pyrite.materials.crystal.absorption_length_ang`, `pyrite.materials.crystal.chi_0`, `pyrite.materials.crystal.refractive_index`, `pyrite.materials.attenuation`, `pyrite.montecarlo.groove.escape_distance_ang`, and the escape/dispersion helpers in `pyrite.montecarlo.spectrum.lines`.
