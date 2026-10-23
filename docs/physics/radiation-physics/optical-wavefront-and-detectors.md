# Optical wavefront propagation and a calibrated pulse detector (preliminary)

These **opt-in, separately staged** numerical building blocks implement limited
portions of [#616](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/616)
and [#617](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/617).
They are **not** a validated end-to-end CTR bunch-length diagnostic. They do
not modify the optical/X-ray continuum floor, the production TR component,
existing detector defaults, or the current planar TR/CTR source (#615).

## Monochromatic vacuum wavefront (#616)

`pyrite.montecarlo.spectrum.optical_wavefront.OpticalWavefront` contains a
complex electric **phasor** $\mathbf E_\parallel(x,y;\omega)=(E_x,E_y)$ in V/m,
with one photon energy in eV and sampling pitches $\Delta x,\Delta y$ in m.
The reference plane is $z=z_0$, normal to the **positive z** propagation
direction. The time convention is $\exp(-i\omega t)$; $E_z$ is determined for
each outgoing plane-wave component by $\mathbf k\cdot\mathbf E=0$.

`propagate_vacuum` applies the vector angular-spectrum propagator independently
to both tangential components:

$$
\widetilde E_{x,y}(k_x,k_y,z+\Delta z)
=\widetilde E_{x,y}(k_x,k_y,z)\exp(i k_z\Delta z),\quad
k_z=\sqrt{(\omega/c)^2-k_x^2-k_y^2},\quad
\operatorname{Im} k_z\geq0.
$$

The propagation is forward-only: negative distances are refused to avoid
unbounded amplification of evanescent waves. The FFT imposes a *periodic*
transverse window, so insufficient zero padding causes wraparound artifacts.
Neither propagation nor the data class supplies an optical source or chooses
the source mesh/window. For high-NA polarization or exact physical metal
apertures, solve the vector boundary conditions first. An inverse transform of
an **intensity-only** far-field TR spectrum cannot recover this input.

`apply_pupil` optionally applies a 2-D **amplitude** transmission in [0,1],
a spatial phase mask in radians and a constant passive 2x2 Jones matrix.
These are paraxial thin-plane operators. They do **not** model induced metal
currents, diffraction radiation from a beam missing an aperture, physical lens
aberrations, mirror geometry or calibrated optics. Jones singular values must
not exceed one. A hard-edged mask requires spatial-grid/padding convergence.

`forward_power_w` integrates the outgoing propagating normal Poynting flux
for a calibrated **monochromatic** phasor, including the longitudinal field:

$$
P_+=\frac{\Delta x\Delta y}{2 Z_0 N_xN_y}
\sum_{k_x^2+k_y^2<k_0^2}\frac{k_z}{k_0}
 \bigl(|\widetilde E_x|^2+|\widetilde E_y|^2+
       |\widetilde E_z|^2\bigr),\qquad
\widetilde E_z=-\frac{k_x\widetilde E_x+k_y\widetilde E_y}{k_z}.
$$

This result is in W and **cannot** be converted to J/pulse or photons/eV
without a specified temporal/spectral-field Fourier convention. Do not confuse
it with #615's outgoing `photons/eV/sr` intensities. Vacuum propagation and
FFT kernels accept an `xp` array backend; NumPy is covered here, and
**device-resident CuPy and remote CUDA parity have not yet been verified**.

## Calibrated optical single-pulse energy channel (#617)

`pyrite.detectors.optical_energy.CalibratedPulseEnergyDetector` accepts measured,
explicitly identified spectral **energy-absorption fraction** $\eta(E)$ in
[0,1] over a strictly bounded photon-energy band, a single-pulse voltage gain
$G$ in V/J, optional electronics RMS noise in V and a positive saturation
voltage. There are **no** inferred optical constants or X-ray detector QE
fallbacks. The input to `predict` must already have been integrated over the
source, instrument acceptance and collection region:

$$
\Phi_{\rm collected}(E)\ \ [\mathrm{photons/(eV\,pulse)}],\qquad
U_{\rm incident}=e\!\int\!E\Phi(E)dE,\qquad
U_{\rm absorbed}=e\!\int\!E\eta(E)\Phi(E)dE,\quad
V_{\rm peak,expected}=G U_{\rm absorbed}.
$$

Here $E$ is numerically in eV and $e=1.602176634\times10^{-19}$ J/eV.
The integrals use trapezoidal quadrature on the **physical energy grid**:
`predict` accepts *spectral density*, not per-bin photon counts, and does not
extrapolate the calibration even for zero-valued samples. The readout separately
reports expected voltage and clipped voltage, an RMS-noise magnitude without
adding random noise, and a SHA-256 fingerprint of the calibration. If a
repetition frequency is given, it reports average incident/absorbed **optical
power** in W without misapplying a pulse V/J gain as a DC V/W responsivity.
No temporal impulse response, AC coupling, integration gate, thermal recovery,
pile-up, shots/exposure, shot noise or instrument polarization is implied.

Minimal *synthetic* use (these values are not a measured calibration):

```python
import numpy as np
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector

detector = CalibratedPulseEnergyDetector(
    energy_eV=np.array([0.2, 0.5, 1.0]),
    efficiency=np.array([0.25, 0.5, 0.7]),
    pulse_gain_v_per_j=1e9,
    calibration_id="synthetic demo: NOT a real detector calibration",
)
energy = np.linspace(0.2, 1.0, 101)
collected_photons_per_eV_per_pulse = np.ones_like(energy) * 1e6
reading = detector.predict(energy, collected_photons_per_eV_per_pulse, repetition_hz=1e6)
print(reading.absorbed_joule_per_pulse, reading.expected_voltage_v)
```

## Boundaries and remaining gates

- **#616:** the PEC disc/annulus **magnetic-equivalent surface-current**
  source and finite-distance Green operator are implemented as an independent
  bounded reference, with oblique straight-charge incidence and transport
  handoff. Real materials, general finite surfaces and exact edge currents
  are not. Measured passive Jones elements and a pitch/window convergence
  diagnostic exist (sections below). Published numerical benchmark,
  measured element data, spectral convergence and remote CPU/CUDA parity
  remain open.
- **#617:** supply a real measured and provenance-pinned detector calibration,
  temporal/electronic response, collected-field-to-spectrum pipeline, polarized
  Martin–Puplett interferogram, uncertainties and finite-band bunch-length
  inference. Published data reproduction and remote CUDA tests remain open.
- **DTR:** the existing atomic Bragg DTR solver (#581) is a separate X-ray
  coherent-boundary model, **not** optical diffraction from a wavelength-scale
  grating or optical aperture DR.

Tests exercise analytically known plane waves, power and phase limits, evanescent
decay, linear pulse-energy integration, saturation, calibration hashing and
strict refusal of unsupported inputs. They are **not** experimental anchors:
`optical-wavefront-propagation` and `optical-instrument-observable` remain
unanchored until independent re-derivation, published regressions and device
parity have been completed.

## Circular PEC source (#616, bounded reference)

The opt-in `pyrite.montecarlo.spectrum.optical_finite_screen` module constructs
**physically normalized, single-electron** backward TR from a perfect-conductor
disc, or aperture diffraction radiation (DR) from a conductor annulus with a
beam passing through the hole. It is a **separate spectral-field representation**
from the steady-monochromatic `OpticalWavefront` above; the two must not be
substituted or combined without an explicit temporal Fourier normalization.

The geometry is the plane $z=0$, metal at inner radius $a\le\rho\le b$,
a straight electron with $\beta_z<0$ and $|\boldsymbol\beta|<1$ arriving
from the positive-$z$ (vacuum) half-space, and observation in positive $z$. Its charge is one signed
elementary charge: *not* a macro-particle carrying the bunch population. This
is an ideal-conductor, local planar boundary-source calculation; there is no
material permittivity or optical Fresnel factor to apply again.

For one **normally incident** electron passing the plane at
$(x_e,y_e,t_e)$, with temporal Fourier convention $\widetilde{\mathbf E}(\omega)=\int\mathbf E(t)
 e^{+i\omega t}dt$, the free-electron tangential incident field is

$$
\widetilde{\mathbf E}_{e,\parallel}(\boldsymbol\rho,\omega)=
\frac{q\kappa}{2\pi\epsilon_0|v_z|}
 K_1(\kappa r)\,\hat{\mathbf r}\,e^{+i\omega t_e},
\quad r=|\boldsymbol\rho-\boldsymbol\rho_e|,
\quad\kappa=\frac{\omega}{\gamma|v_z|}.
$$

The radiating electric boundary field on the metal is
$\widetilde{\mathbf E}_{s,\parallel}=-\widetilde{\mathbf E}_{e,\parallel}$,
and it is zero outside the metal. This is the normal-incidence scalar-medium
specialization of the vector PEC boundary condition in
[Shkvarunets and Fiorito (2008), Eq. (15)](https://arxiv.org/abs/0712.2848).
The source is an outgoing magnetic-equivalent sheet
$2(\hat{\mathbf z}\times\widetilde{\mathbf E}_{s,\parallel})$.
For field points $z>0$, `observe_screen_green` evaluates its retarded vector
Green-curl directly:

$$
\widetilde{\mathbf E}(\mathbf R)=
2\int_{\rm metal} \left[\nabla_{\mathbf R}
\frac{e^{ik_0|\mathbf R-\mathbf r|}}{4\pi|\mathbf R-\mathbf r|}
\right]\times(\hat{\mathbf z}\times
\widetilde{\mathbf E}_{s,\parallel}(\mathbf r))\,d^2r.
$$

This includes the exact $1/R$ and $1/R^2$ terms of the equivalent-current
field, not a ray or far-field detector approximation. `propagate_screen_field`
uses the **same outgoing solution** on a regular FFT angular-spectrum grid,
including decaying evanescent modes, but periodically tiles the source.
Use the direct Green operator for a selected set of physical observation points,
and the FFT operator for spatial-field maps when periodic-image and padding
errors have been converged. Both return complex **V\,s/m** spectral fields,
not V/m phasors.

`far_field_photons_per_eV_sr` evaluates the outgoing (positive-$z$) far-field
photon yield $d^2N/(dE\,d\Omega)$ at propagating FFT directions. With the
stated *two-sided* temporal Fourier convention,
`spectral_energy_per_eV` computes forward $dU/dE$ (joules per eV) from
$(\pi Z_0)^{-1}\sum_{k_\perp<k_0}(k_z/k_0)|\widetilde{\mathbf E}(\mathbf k_\perp)|^2$
(the outgoing normal flux of each propagating mode, including $E_z$; it
reduces to $(\pi Z_0)^{-1}\int|\widetilde{\mathbf E}|^2dA$ only paraxially) and the
$1/\hbar$ conversion; it is **not** average continuous-wave optical power.
Do not multiply these single-charge yields by a macro-charge squared. Use the
physical $N$ and a resolved coherent electron-field estimator as in #615,
retaining arrival phases, using the bounded finite-screen bunch path below.

The zero-screen limit returns zero; a finite aperture emits DR without a
boundary crossing; increasing aperture radius suppresses its yield. Absolute
far-field amplitudes approach the infinite-PEC Ginzburg–Frank formula at
1 eV for $\beta=0.95$ **and** the 30-keV electron $\beta\simeq0.328$.
Tests also compare direct finite-distance Green quadrature against FFT
propagation, check propagation composition/energy conservation, and verify
solid-angle Jacobian integration. **These are synthetic numerical checks,
not independent published finite-target evidence or experimental validation.**

Constraints: circular flat PEC screens only; negative $\beta_z$ (backward
radiation); one charge; source-grid centre sampling of the integrable $1/r$
field; the point source cannot fall exactly on a metallic sample node;
transverse wraparound and insufficient mesh resolution can bias the result.
Oblique beams are supported only under the straight-current ideal-PEC
approximation detailed below. No imperfect metals, finite-conductivity edge
currents, X-ray DTR, optical gratings, realistic apparatus or CUDA parity are
claimed.
The following section adds a bounded coherent-bunch reference estimator,
not yet a production optical transport/instrument integration. Validation claim `optical-finite-boundary-field` remains
**unanchored**, pending published finite-disc/DR benchmarks and hardware tests.

```python
import numpy as np
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
    propagate_screen_field,
    observe_screen_green,
    spectral_energy_per_eV,
    far_field_photons_per_eV_sr,
)

screen = PECCircularScreen(outer_radius_m=4e-6, inner_radius_m=0.5e-6, beta_z=-0.95)
field = circular_pec_transition_field(1.0, screen, nx=256, ny=256, dx_m=50e-9, dy_m=50e-9)
near_field = propagate_screen_field(field, distance_m=0.2e-6)
selected_points = observe_screen_green(field, np.array([[0.2e-6, 0, 0.2e-6]]))
theta, phi, photons_per_eV_sr = far_field_photons_per_eV_sr(field)
print(spectral_energy_per_eV(near_field), selected_points.shape)
```

## Oblique incidence and formation-scale guard (#616 extension)

The single-electron `PECCircularScreen` accepts `beta_x` and `beta_y` in
addition to the previously supported negative `beta_z`. For a charge in
uniform straight-line motion at total speed $v=c|\boldsymbol\beta|$, let
$\hat{\mathbf d}=\boldsymbol\beta/|\boldsymbol\beta|$ and
$\mathbf u=(x-x_e,y-y_e,0)$. Define
$\mathbf p=\mathbf u-\hat{\mathbf d}(\hat{\mathbf d}\cdot\mathbf u)$,
$a=|\mathbf p|/\gamma$, $t_*=\hat{\mathbf d}\cdot\mathbf u/v$,
$\gamma=(1-|\boldsymbol\beta|^2)^{-1/2}$. The exact Fourier transform
of the uniformly moving charge's *vacuum incident field* at the screen is

$$
\widetilde{\mathbf E}_{e}(\mathbf u,\omega)=
\frac{q\omega}{2\pi\epsilon_0\gamma^2v^2}
\exp[i\omega(t_e+t_*)]
\left[\frac{\mathbf p}{a}K_1\!\left(\frac{\omega a}{v}\right)
-i\hat{\mathbf d}K_0\!\left(\frac{\omega a}{v}\right)\right].
$$

The PEC equivalent tangential field on the metal is its negative, using
only the $x,y$ components. The $K_0$ term is important for oblique
incidence: there is generally a transverse field contribution from the
longitudinal velocity. The source phase also varies spatially by
$\exp(i\omega t_*)$. Setting $\beta_x=\beta_y=0$ reduces exactly to the
normal-incidence $K_1$ formula above. This expression has a separate
numerical test against direct time integration of the Liénard–Wiechert
field for a uniformly moving charge. That is an **independent analytical
limit**, not an experimental validation of the finite conductor.

The entrance-handoff rotation is still proper and explicit:
$\boldsymbol\beta_{\mathrm{PEC}}=
(\beta_{x,\mathrm{slab}},-\beta_{y,\mathrm{slab}},-\beta_{z,\mathrm{slab}})$.
Users must explicitly increase `max_angle_rad` to admit a tilted beam;
this prevents quietly broadening previously validated near-normal
applications. Aperture geometry and finite-grid constraints remain in force.

For radiation in the backward +$z$ hemisphere collected to polar angle
$\theta\le\theta_{\max}<\pi/2$, the inverse of the minimum phase-mismatch
rate across the cone is a conservative **1-radian kinematic length along the
trajectory**, not a quantitative finite-track radiation error bound:

$$
L_{f,\mathrm{back}}=
\frac{|\boldsymbol\beta|}{k_0[1-\beta_\perp\sin\theta_{\max}
-\beta_z\cos\theta_{\max}]},
\quad k_0=\frac{E}{\hbar c},\quad
\beta_\perp=\sqrt{\beta_x^2+\beta_y^2}.
$$

`backward_formation_length_ang(E, screen, theta_max_rad=...)` returns this
scale in Å. `min_formation_lengths=m` optionally requires the transported
post-entry straight run to be at least $mL_f$ at the *lowest* photon energy
in a band and its declared acceptance angle. It does not establish the
actual finite-current waveform, PEC foil transmission, deflection, optical
energy budget, or independence from other radiation mechanisms. Do not
promote the finite-boundary validation ledger claim on this basis alone.

## Finite-screen coherent-bunch scoring (#616 follow-up)

`pyrite.montecarlo.spectrum.optical_finite_bunch` connects the existing
`PECCircularScreen` *spectral Fourier* fields to the physical-population
contract in `OpticalBunchSource` (the same contract used by the planar CTR
source, #615). Unlike a far-field intensity multiplied by an arbitrary
``N**2``, the finite-screen scorer first computes the full complex
single-electron field for **each physical sampled history**, including the
actual impact position and arrival-time phase at the common conducting disc or
annulus. Distinct sampled trajectories may have different signed longitudinal
velocities (all must travel toward the screen, as required by the single-
particle model), so their responses need not be identical. A history carries
**one elementary charge**, never a macro-charge.

For sampled histories indexed by $i$, with positive importance weights $w_i$,
complex spectral field $S_i(\mathbf k_\perp)$, and $W=\sum_i w_i$,
$W_2=\sum_i w_i^2$, the estimates at every outgoing angular mode use the
physical vector-field Poynting metric $\|S\|_P^2=|S_x|^2+|S_y|^2+|S_z|^2$:

$$
I_{\mathrm{self}}=\frac{\sum_i w_i\|S_i\|_P^2}{W},\quad
I_{\mathrm{pair}}=\frac{\|\sum_i w_iS_i\|_P^2-
             \sum_i w_i^2\|S_i\|_P^2}{W^2-W_2},\quad
I_{\mathrm{bunch}}=N[I_{\mathrm{self}}+(N-1)|F_t|^2I_{\mathrm{pair}}].
$$

The common angular and spectral normalization is exactly that of
`far_field_photons_per_eV_sr`; *no* extra Fresnel factor, crossing spectrum,
photon energy, or bunch population is applied. $N=Q/e$ is determined solely
by `OpticalBunchSource.charge_pC`, not the number of simulated trajectories.
An `OpticalBunchSource.realization` instead requires exactly $N=M$ unweighted
physical electrons: its result is the exact complex bunch-field intensity
$\|\sum_i S_i\|_P^2$, and
`finite_screen_realization_field` returns the **summed complex temporal Fourier
field** [V s/m] for forward near-field propagation by `propagate_screen_field`
or direct observation with `observe_screen_green`.

The distinct-pair estimate is unbiased for independent weighted sampled
histories, but noisy when the true pair coherence is small. A negative estimate
must not be clipped into a false positive detection; such a result calls for
more sampling or an explicit uncertainty treatment. This reference model does
not calculate that uncertainty or correlations between physical electrons.

For independent **temporal-only Gaussian** or empirical offsets, source
histories must carry identical offset-free entrance positions/times; the
existing source object's `form_factor_squared` supplies $|F_t(E)|^2$. The
transverse covariance/profile **must be identically zero** for such analytic
laws. Unlike an infinite planar radiator, a finite disc or hole breaks
transverse translation invariance: changing the electron impact point changes
the induced boundary field itself. To model transverse position, bunch-front
tilt, arrival-time correlation or microbunching with a finite screen, specify
`OpticalBunchSource.sampled` or `.realization` and sample both the entrance
position and time in the `PECCircularScreen` histories. No analytic transverse
form factor is silently multiplied onto this finite source.

### Far-field angular collection to the calibrated energy channel

```python
import numpy as np
from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource
from pyrite.montecarlo.spectrum.optical_finite_screen import PECCircularScreen
from pyrite.montecarlo.spectrum.optical_finite_bunch import finite_screen_bunch_band
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector

source = OpticalBunchSource.gaussian(
    charge_pC=1e-6,
    description="synthetic temporal Gaussian, no transverse spread",
    rms_t_fs=0.35,
)
screens = [PECCircularScreen(outer_radius_m=3.5e-6)] * 3
band = finite_screen_bunch_band(
    np.linspace(0.8, 1.2, 17),
    screens,
    source=source,
    nx=256,
    ny=256,
    dx_m=50e-9,
    dy_m=50e-9,
    theta_max_rad=np.pi / 4,
)
# These example values are NOT a measured detector calibration.
detector = CalibratedPulseEnergyDetector(
    energy_eV=np.array([0.8, 1.2]),
    efficiency=np.array([0.3, 0.3]),
    pulse_gain_v_per_j=1e9,
    calibration_id="SYNTHETIC EXAMPLE ONLY",
)
reading = detector.predict(band.energy_eV, band.collected_photons_per_eV_per_pulse)
```

`finite_screen_bunch_angular` exposes the single-energy self, pair and
bunch maps, and `collect_cone(theta_min_rad=..., theta_max_rad=...)` uses the
outgoing angular FFT-cell Jacobian. `finite_screen_bunch_band` iterates the
source mesh over a **strictly increasing** photon-energy grid, returning
photons/(eV pulse) integrated over the selected far-zone cone. It does not
include finite-distance optical acceptance, polarization-dependent collection,
window response, or measured detector calibration: add these explicitly before
claiming a *detected* bunch spectrum. The response operator in #617 then
converts such a collected spectrum to joules and an isolated-pulse signal.

This is a CPU reference, not yet production transport/geometry integration.
For realistic bunches, memory is kept to a few FFT planes per energy, but
computational cost scales with sampled histories × photon energies × FFT mesh.
Numerical convergence must be established separately for the source/edge
pixels, periodic FFT window and padding, angular collection and photon-energy
grid. No independent measured TR/DR/CTR comparisons, current-profile
reconstruction, calibrated optics or real-hardware GPU parity is claimed.

## Transported entrance histories as finite PEC diagnostics (#616, transport handoff)

The opt-in handoff in
`montecarlo.spectrum.optical_finite_transport.entrance_pec_screens_from_crossings`
reads the established `BoundaryCrossings` contract. The convenience adapter
`montecarlo.runner.finite_screen_emission.case_finite_screen_band` extracts
entrance crossings from `transport["segs"]` using `boundary_crossings` and
`case_layers`. The companion `case_finite_screen_realization_field` returns a
**complex** spectral source field, but **only** for
`OpticalBunchSource.realization`, whose `Ne_brem` histories are the actual
physical electrons. For sampled macroparticle ensembles there is no unique
physical wavefront; use the self/pair spectrum instead.

This is **not** another transition-radiation term to add on top of planar TR:
it is an **alternative, explicitly idealized PEC radiator** placed at the
transport entrance plane. The simulated target need not be metal, and its
transport was **not** recomputed with that PEC radiator. Therefore, applying
this source to arbitrary transported trajectories is a counterfactual
source-only diagnostic, *not* self-consistent foil transport. It does not
include material response, scattering/recoil in the PEC, or the formation
zone following electron deflection. If the radiator and planar source describe
the same physical interface, summing their intensities would double-count
radiation. No automatic physical validity guarantee is implied.

The accepted geometry is the transport slab's `z=0` entrance, with primary
electrons traveling toward positive slab `z`; observation is upstream `-z`.
The change of frame is the **proper 180-degree rotation about x**,
`(x,y,z)_PEC = (x,-y,-z)_slab`. Consequently the PEC incident electron
has `-1 < beta_z < 0` and the source emits into positive PEC `z`. Photon
angles and complex field coordinates are **in the PEC frame**, not the
transport slab frame. The screen centre is `centre_xy_ang` in slab Å.

Only the first `Ne_brem` sampled histories are selected, indexed by their
`electron_id`; larger-ID electron histories transported for other purposes
are ignored. Precisely **one** vacuum-to-layer-0 entry is required for each
selected primary. A backscattered vacuum exit cannot masquerade as an entry.
No silent tilt correction is made: the default `max_angle_rad=1e-3` refuses
more-oblique histories, while a larger explicit threshold uses the exact
uniform-velocity oblique field (not a normal-incidence approximation).
`min_straight_after_ang` optionally bounds the first straight run **inside**
the transport target, and `min_formation_lengths` requests a multiple of the
kinematic backward-TR formation scale from the lowest energy and largest
collection angle. These are conservative *diagnostics*, not a sufficient
formation-zone bound or model of electron interaction with the PEC.
History clocks are `time_ang` in Å/c and are converted to arrival seconds
with an explicit, shared `time_origin_ang` reference. Source importance
weights remain indexed by the same `electron_id` and are not multiplied into
single-electron charge.

Example, once a case and optical source are available (synthetic PEC geometry;
**not a calibrated metal/detector**):

```python
import numpy as np
from pyrite.montecarlo.runner.finite_screen_emission import case_finite_screen_band
from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource

# case, transport = ...  # a normal-incidence, midpoint PyRITE transport result
source = OpticalBunchSource.sampled(0.01, "entrance-timed sampled electrons")
result = case_finite_screen_band(
    case,
    transport,
    np.linspace(0.8, 1.2, 9),
    source=source,
    outer_radius_m=3e-6,
    inner_radius_m=1e-6,  # annulus = aperture DR
    nx=256,
    ny=256,
    dx_m=6e-8,
    dy_m=6e-8,
    theta_max_rad=0.3,
    # Optional: min_straight_after_ang=<independently chosen formation bound>
)
# photons/(eV pulse) inside a far-field vacuum cone, NOT a detector plane
print(result.collected_photons_per_eV_per_pulse)
```

The incidence angle, screen radius, sampling pitch/window, and
frequency/angular integration must be converged separately. The optical energy
nodes are passed **explicitly**: nothing lowers PyRITE's common X-ray floor,
invents optical constants, or modifies case persistence. As before, the
calibrated energy-integrating detector in #617 may consume this *cone*-integrated
spectrum only after any additional instrumental collection factors are
provided. A finite-distance sensor instead needs an explicit optical field
propagator and transfer/acceptance model.

The transported handoff is covered by synthetic field/phase/kinematics,
history-index, wrong-face, incidence-acceptance, aperture and source-law tests.
**Unresolved gates:** a full-repository transported `Case` acceptance run,
independent finite-boundary publication benchmark, formation-zone energy
budget, GPU/device implementation and experiment remain required before
promoting `optical-finite-boundary-field` beyond *unanchored*.

## Finite-distance, polarization-resolved collection (#616 next slice)

`pyrite.montecarlo.spectrum.optical_finite_collection` provides an **opt-in CPU
reference** for the finite circular PEC screen source implemented in
`optical_finite_screen.py` and the physical-bunch estimates implemented in
`optical_finite_bunch.py`. Unlike the earlier angular-cone scorer, this operator
integrates **a physical area on a plane at a stated finite distance**. It does
not extrapolate the far-field photon intensity back to a close detector.

With the same temporal Fourier convention as the PEC source,

$$
\widetilde{\mathbf E}(\omega) = \int_{-\infty}^{\infty}
  \mathbf E(t)e^{+i\omega t}\,dt,\qquad
\frac{dU_{\mathcal A}}{dE}
  =\frac{1}{\pi\hbar}\int_{\mathcal A}dA\;
     \Re[\widetilde E_x\widetilde H_y^*
        -\widetilde E_y\widetilde H_x^*].
$$

The 2-D angular-spectrum field is propagated in **vacuum**, reconstructing
$E_z=-(k_x E_x+k_y E_y)/k_z$ and the magnetic components using
$\mathbf H=(\mathbf k\times\mathbf E)/(Z_0 k_0)$ on every outgoing FFT mode.
This retains evanescent components until propagation to the collection plane.
The local normal Poynting flux can be negative due to near-field interference;
values are **signed**, never clipped. A negative per-pulse photon-density
estimate cannot be passed to `CalibratedPulseEnergyDetector` as counts.

The explicitly declared `PlaneAcceptance` describes a **rectangle** (exact
pixel-area overlap) or **circle** (8×8 subpixel-area integration) within the
computational FFT cell. The aperture must lie strictly inside the grid window;
its size, mesh spacing, and physical propagation distance must be separately
converged. The grid is periodic, so a calculation can conserve the total
outgoing power while still suffering from periodic-image contamination in a
small aperture. A finite pupil here **collects** existing light; it does not
create diffraction radiation by driving currents on an additional metal edge.

An optional 2×2 passive `jones_sp` operator acts on the **propagating s/p
plane-wave amplitudes**, before spatial collection. This is a transverse
wavevector-dependent basis and **not** a uniform laboratory x/y polarizer.
`[[1,0],[0,0]]` selects s and `[[0,0],[0,1]]` selects p. The operator rejects
net gain (singular value over one), preserves relative phase, and removes
undetected evanescent components. It does not claim calibrated
angle-dependent reflection/transmission for real mirrors, windows or lenses.
Over the whole FFT plane, separately projected s and p powers sum to total
propagating power; on a finite region, their *spatial* interference need not
add to unfiltered local power, so specify the actual analyzer rather than
assuming two intensity maps are additive.

### Single electron or a physical realization

```python
import numpy as np
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
)
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
)

screen = PECCircularScreen(outer_radius_m=2e-6, inner_radius_m=0.5e-6)
field = circular_pec_transition_field(1.0, screen, nx=128, ny=128, dx_m=70e-9, dy_m=70e-9)
acceptance = PlaneAcceptance("circle", radius_m=1.5e-6)
reading = collect_screen_field(
    field,
    plane_z_m=1e-6,
    aperture=acceptance,
    jones_sp=[[0, 0], [0, 1]],  # ideal p-only analyzer
)
print(reading.photons_per_eV)  # photons/(eV * physical electron)
```

For a **physical realization**, supply the coherent source field returned by
`finite_screen_realization_field` instead of the one-electron field. Complex
fields are summed before collection, retaining the correlations and
cancellation that an incoherent sum would erase.

### Sampled physical bunches and calibrated pulse detector

`finite_screen_bunch_plane_band` performs the analogous **weighted self/pair
estimator** for a source sampled with law `sampled`, `gaussian`, or `empirical`,
using exactly the physical population and offset-law restrictions established
for `finite_screen_bunch_angular`. The local bilinear Poynting quantity is
computed from the **coherent sums of E and H** rather than from a scalar
$|E|^2$. The estimator is

$$
\begin{aligned}
S_1 &= \sum_i w_i\,\mathcal P[E_i,H_i],\\
S_2 &= \sum_i w_i^2\,\mathcal P[E_i,H_i],\\
C &= \mathcal P[\sum_i w_iE_i,\sum_i w_iH_i],\\
\frac{dU_{\rm bunch}}{dE}
 &= N\left[\frac{S_1}{W}+(N-1)|F_t(E)|^2
    \frac{C-S_2}{W^2-W_2}\right],
\end{aligned}
$$

where $\mathcal P$ is the area-integrated spectral Poynting bilinear, and
$W=\sum_i w_i$, $W_2=\sum_i w_i^2$. For an actual `realization`, the exact
$C$ is used, not the independent-electron estimate. Only **temporal** analytic
form factors are admitted: transverse translations alter the finite metal
surface currents, and must instead be represented by sampled intercepts.

```python
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    finite_screen_bunch_plane_band,
)
from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector

# Here `screens` is a tuple of actual PEC source histories, for example from
# `entrance_pec_screens_from_crossings`. Optical detector calibration is external.
# Replace `screens` and `detector` with your explicitly configured sources.
source = OpticalBunchSource.sampled(1.0, "sampled physical bunch")
# result = finite_screen_bunch_plane_band(
#     [0.8, 1.0, 1.2], screens, source=source,
#     nx=128, ny=128, dx_m=70e-9, dy_m=70e-9,
#     plane_z_m=1e-6,
#     aperture=PlaneAcceptance("rectangle", width_m=3e-6, height_m=3e-6),
# )
# pulse_reading = result.detect(detector, repetition_hz=1e6)
```

The entry-point `runner.finite_screen_emission.case_finite_screen_plane_band`
connects the same scorer to transported primary entrance crossings. The PEC
screen is still an **independent diagnostic model**, not an extra physical
boundary in electron transport. If the optional straight-run formation check
is enabled, the user must explicitly provide a conservative
`formation_theta_max_rad`: an extended finite source cannot infer that bound
solely from the detector's centre ray. Optical detected joules per pulse and
volts remain separate from the source's photons/eV/pulse. The returned
`configuration` records the grid, aperture, plane, ideal s/p Jones matrix and
Fourier normalization; no optical response is inferred or re-applied.

**Remaining evidence gates:** wavelength-dependent detector measurements and
noise, source/grid/FFT-padding/acceptance convergence studies, comparison with
a published finite-screen near-field observable, and device-resident
CPU/CUDA parity. This slice is **not** an anchored
`optical-instrument-observable` or `optical-finite-boundary-field` validation.

## Polarized two-port Martin–Puplett forward model (#617, new slice)

`pyrite.detectors.optical_interferometer` implements an opt-in **ideal Jones
reference** for a Martin–Puplett-type polarizing Fourier-transform
interferometer. This is a *forward* model, not spectral phase retrieval or
experimental proof of bunch-length inference. A real instrument requires
calibrated wire-grid transmission/reflection, roof-mirror rotations, polarization
axes, propagation/collection, detector impulse response and background ports.
The present model approximates the ideal polarization-retarder transfer derived
for the Martin–Puplett interferometer in [Interferometric polarization
control](https://doi.org/10.1364/AO.45.005107). It explicitly includes the
**input polarizer**; without it, an unpolarized field would not show the same
interferometric modulation. It does *not* claim a two-input differential MPI
or a measured wire-grid transfer.

Input `OpticalSpectralCoherency` requires strictly increasing energy [eV] and a
Hermitian positive-semidefinite $2\times2$ spectral polarization-density matrix
$J(E)$ in **photons/(eV pulse)**. Its diagonal trace is already spatially
collected. Cross polarization phase is specified by off-diagonal entries;
scalar intensity alone cannot determine it. A convenient *explicit* linear
polarization assumption uses `OpticalSpectralCoherency.linearly_polarized`,
with a provenance explanation. No such polarization is silently inferred from
`FinitePlaneBunchBand`, which contains only a signed intensity estimator.

Let $u=(\cos\alpha,\sin\alpha)$ be the input grid's transmitted Jones axis;
let $v_+=(\cos\eta,\sin\eta)$ and $v_-=(-\sin\eta,\cos\eta)$ be the two
output analyzing axes. The calibrated input spectral density passed into the
interferometer is $I_u(E)=u^\dagger J(E)u$. With passive **intensity** arm
transmissions $T_1,T_2$, the ideal arm Jones operator is

$$
D(E,\ell)=\operatorname{diag}(\sqrt{T_1},\sqrt{T_2}\,e^{i\delta}),
\qquad\delta(E,\ell)=\frac{2\pi E\ell}{hc}+\delta_0,\qquad
\ell=\ell_0+2\,\Delta x_{\rm mirror}.
$$

The output spectra are

$$
\Phi_{\pm}(E,\ell)=\eta_{\pm} I_u(E)
\left|v_{\pm}^{\dagger}D(E,\ell)u\right|^2,
$$

where $\eta_\pm\in[0,1]$ are explicitly chosen port *throughput* factors.
For Gaussian RMS *optical-path* jitter $\sigma_\ell$, only the interference
cross term is reduced by $\exp[-(2\pi E\sigma_\ell/hc)^2/2]$. This is not
an electron-bunch arrival-time jitter model: common arrival-time shifts cancel
from the two arms of a single pulse. At ideal $45^\circ$ input and output
grids and no loss, $\Phi_\pm=I_u(1\pm\cos\delta)/2$. The ports' sum equals
$I_u$ for any delay. Note that an initially **unpolarized** source loses half
its power at the *input grid* before either port, rather than producing a
false unit-efficiency signal.

The two explicitly supplied `CalibratedPulseEnergyDetector` objects transform
spectra to absorbed joules per *pulse*, readout voltage, RMS electronic noise
and saturation independently. Mirror displacement [m], actual OPD [m] and
optical delay [fs] are all returned. Repetition rate affects *average optical
power* only. `OpticalInterferogram.save` writes non-pickled NPZ arrays and
JSON calibration/provenance, including both detector response hashes.

```python
import numpy as np
from pyrite.detectors.optical_interferometer import (
    IdealMartinPuplett,
    OpticalSpectralCoherency,
    polarized_interferogram,
)
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector

energy = np.linspace(0.8, 1.2, 401)
# Synthetic example only: do not infer real beam polarization from intensity.
source = OpticalSpectralCoherency.linearly_polarized(
    energy,
    1e6 * np.ones_like(energy),
    angle_rad=np.pi / 4,
    provenance="synthetic input: assumed 45 degree linear polarization",
)
cal = CalibratedPulseEnergyDetector(
    np.array([0.8, 1.2]),
    np.array([0.5, 0.5]),
    pulse_gain_v_per_j=1e9,
    calibration_id="synthetic, unmeasured pulse response",
)
result = polarized_interferogram(
    source,
    np.linspace(-3e-6, 3e-6, 201),
    instrument=IdealMartinPuplett("ideal wire-grid and roof-mirror reference"),
    detectors=(cal, cal),
    repetition_hz=1e6,
)
print(result.optical_delay_fs, result.expected_voltage_v_ports)
```

### Finite complex field → polarimetric spectrum

`pyrite.montecarlo.spectrum.optical_interferometer_handoff`
`.coherency_from_screen_fields` accepts **complex, realized**
`ScreenSpectralField` arrays (one per photon-energy node) from #616, propagates
them to a bounded collecting plane, and constructs their Cartesian $x/y$
spectral coherency by integrating $\widetilde E_i\widetilde E_j^*$ over the
chosen physical aperture. The paraxial normalization is

$$
J_{ij}(E)=\frac{1}{\pi\hbar_{\mathrm{eV\,s}}Z_0E_{\mathrm{eV}}e}
\int_A\widetilde E_i(\mathbf r;E)\widetilde E_j^*(\mathbf r;E)\,dA.
$$

A finite-screen source is not always paraxial: the adapter **refuses** fields
with more than a declared fraction of spatial FFT energy outside a specified
numerical aperture (including evanescent components), or where this paraxial
power differs too much from the existing full-vector Maxwell/Poynting flux.
The Jones axes must approximate globally fixed instrument transverse axes,
so a wide-angle s/p basis that rotates with observation direction is not
silently identified with the instrument's wire-grid axes. This is an
opt-in conservative approximation, not an arbitrary near-field polarimeter.

Important limitations: the current field handoff is for **true complex
realizations**, not a sampled $N\gg M$ bunch's intensity-only self/pair
expectation. Neither the interferometer nor the adapter constructs the missing
complex ensemble coherence. Full optical instrument geometry, background
subtraction, shot-charge variation, uncertainty inference, finite-band
retrieval ambiguities, measured MPI comparisons, and CPU/CUDA parity are
outstanding. Synthetic forward/port-conservation tests do **not** anchor
`optical-instrument-observable`.

### Conditional Gaussian CTR bunch-duration inference (#617)

`pyrite.detectors.optical_ctr_inference` implements a **bounded parametric
forward fit**, *not* a unique electron-bunch profile reconstruction. It uses
exactly the ideal Jones/OPD-jitter transfer of `polarized_interferogram` and
the measured spectral efficiency and isolated-pulse gain of each
`CalibratedPulseEnergyDetector`. The *measured* inputs are calibrated two-port
pulse voltages vs actual mirror displacement and strictly positive, independent
RMS voltage uncertainties. Lock-in signals, exposure-integrated responses or
correlated noise require a separate observation operator; they cannot be
silently treated as pulse voltages.

The `GaussianCTRTemplates` source description separates two **already
collected, per-physical-bunch** photon density templates:

- `incoherent_photons_per_eV_per_pulse`: physical self and other fixed sources;
- `coherent_envelope_photons_per_eV_per_pulse`: physical distinct-pair excess,
  *before* temporal form-factor attenuation, including physical population,
  transverse factors, acceptance and all other fixed source physics.

Neither term should be multiplied by bunch charge, `N`, or `N(N-1)` a second
time. The construction of independent templates from a transport case is not
automatic; the inputs must be independently derived with stated calibration and
source validity. In particular, an unresolved **signed** pair Monte Carlo
estimate cannot simply be clipped and declared a nonnegative physical source.
For an independent Gaussian arrival-time offset with **RMS** duration
$\sigma_t$ and no added longitudinal/transverse correlations,

$$
\Phi(E;\sigma_t)=\Phi_{\rm self}(E)
+\Phi_{\rm pair,0}(E)\exp[-(E\sigma_t/\hbar)^2],
\qquad E_w=\hbar/\sigma_t.
$$

This shape *assumes*, rather than measures, Gaussian longitudinal structure.
The source's linear Jones polarization angle must also be supported by
independent knowledge; a scalar spectrum alone cannot establish it. The fit
optimizes $\sigma_t$ within **user-selected** positive bounds, optionally
profiling an overall nonnegative source scale and independent constant port
voltage offsets. An optional Gaussian relative scale prior can represent
*one common* charge/throughput calibration uncertainty, but is **not** a
substitute for correlated source-, transverse-, and response-model covariance.

For independent Gaussian measurement uncertainties, the reported 95% interval
uses the conditional one-parameter profile criterion
$\chi^2(\sigma_t)-\chi^2_{\min}\le 3.841$. If it reaches a declared fit bound,
that endpoint is **`None` (open)**, not a finite fitted uncertainty; multiple
separated admissible regions are flagged. This interval excludes spectral
calibration errors, fitting the instrument transfer, sampling uncertainty of
the self/pair templates, drifts and correlated noise. A model can have a small
$\chi^2$ while its RMS duration remains **unidentified**. The returned
`is_identified_95` flag refers only to this stated Gaussian/conditional model.
The results expose the fitted $E_w$ and both measured-band edge form-factor
values, and document the **unmeasured continuation** rather than extrapolating
into a missing low-frequency band.

For a 200 fs RMS bunch, $E_w\approx0.00329$ eV (far-IR/THz), so a
0.05–3 eV optical system alone does not provide the low-energy turnover;
a solver must not return a falsely precise 200 fs result based only on a
vanishing visible-band factor. For other durations, the applicable energy
band and mirror travel must be checked independently. Mirror mechanical
travel is half the stated optical path difference.

Illustrative **synthetic** computation (not measured data or validation):

```python
import numpy as np
from pyrite.detectors.optical_ctr_inference import (
    GaussianCTRTemplates,
    OpticalVoltageScan,
    fit_gaussian_ctr,
    gaussian_ctr_forward_voltage,
)
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import IdealMartinPuplett

energy = np.linspace(0.0005, 0.025, 180)  # eV, includes 200 fs turnover
source = GaussianCTRTemplates(
    energy_eV=energy,
    incoherent_photons_per_eV_per_pulse=np.full(energy.size, 1e8),
    coherent_envelope_photons_per_eV_per_pulse=np.full(energy.size, 2e10),
    polarization_angle_rad=np.pi / 4,
    provenance="synthetic per-bunch templates, NOT transport-derived",
)
cal = CalibratedPulseEnergyDetector(
    np.array([0.0001, 0.03]),
    np.array([0.72, 0.72]),
    pulse_gain_v_per_j=1e12,
    calibration_id="synthetic detector, NOT measured",
)
mpi = IdealMartinPuplett("synthetic ideal reference")
mirror_m = np.linspace(-0.0012, 0.0012, 81)
voltage = gaussian_ctr_forward_voltage(
    source,
    mirror_m,
    rms_duration_fs=200.0,
    instrument=mpi,
    detectors=(cal, cal),
)
measurement = OpticalVoltageScan(
    mirror_m,
    voltage,
    np.full(voltage.shape, 1.5e-5),
    provenance="synthetic noise-free expectation with declared RMS error",
)
fit = fit_gaussian_ctr(
    measurement,
    source,
    instrument=mpi,
    detectors=(cal, cal),
    sigma_bounds_fs=(20.0, 3000.0),
)
print(fit.rms_duration_fs, fit.confidence_interval_95_fs, fit.is_identified_95)
fit.save("synthetic_ctr_fit.npz")  # arrays, model and calibration hashes
```

`GaussianCTRFit.save()` persists the predicted port voltages, residuals,
open/closed conditional confidence interval, calibration/source/scan hashes,
analysis bounds and assumptions without pickles. CPU-only focused tests cover
synthetic parameter recovery, detector-response/Jones-forward consistency,
unconstrained high-frequency bands and strict rejection of saturation or
unsupported calibration. These are **software and mathematical checks, not
independent measured-data anchors**. The published experimental comparison,
real CUDA reduction, broader spectral-band support, correlated uncertainty
propagation and nonparametric phase retrieval remain open in #617.

## Window-independent hemisphere energy and convergence (#616)

The k-space normal-flux density of an outgoing angular spectrum is

$$
\frac{dU}{dE\,d^2k}\propto\frac{k_z}{k_0}\Bigl(|\widetilde E_x|^2+|\widetilde E_y|^2
+\frac{|k_x\widetilde E_x+k_y\widetilde E_y|^2}{k_z^2}\Bigr),
$$

whose longitudinal term diverges as $1/k_z$ on the light circle
$k_x^2+k_y^2=k_0^2$. The singularity is integrable, but a Riemann sum on the
periodic FFT grid (`spectral_energy_per_eV`, `collect_cone` near
$\theta\to\pi/2$, and plane collection through FFT propagation) converges
slowly and **non-monotonically** with the window: for a 2 µm disc at 1 eV,
$\beta=0.95$, the FFT hemisphere total varies from 0.84 to 1.13 of the
converged value between 6.4 µm and 51 µm windows, independent of pitch.

`hemisphere_energy_per_eV` evaluates the same sampled source by a *direct*
(non-periodic) Fourier sum at Gauss–Legendre nodes in $u=\cos\theta$ and
uniform $\phi$. Because $d^2k=k_0^2\,u\,du\,d\phi$, the integrand is smooth;
it is exactly window-independent for a compact source and agrees with the
infinite-PEC Ginzburg–Frank hemisphere yield for a 5 µm disc with the charge
on a pixel corner at 0.1 µm pitch: $\sim10^{-5}$ observed, pinned at
$10^{-4}$ in `tests/montecarlo/test_optical_finite_screen.py`. An off-corner
charge adds a pitch²-convergent sampling error (1.6 % at 0.1 µm for a
partial-pixel offset in the independent check).

**Validity range.** The source is the physical-optics boundary field
($-\widetilde{\mathbf E}_{e,\parallel}$ on the metal, zero off it), exact only
for an unbounded PEC. It requires screen and hole radii, and the electron's
distance to every edge, to be much larger than the wavelength; edge
diffraction corrections at the rim are not included.

`finite_screen_collection_convergence` (module `optical_finite_convergence`)
halves the pitch at fixed window and doubles the window at fixed pitch,
separately, and reports both relative changes plus `hemisphere_fft_bias`, the
FFT-vs-quadrature hemisphere ratio of the base source. `converged` requires all
three below `rtol`. It is a numerical diagnostic for one geometry and energy,
not physics validation.

`collect_screen_green` removes the window error from finite-aperture
collection. It sums the detector-plane fields directly from the compact source
with the free-space Green function: $\mathbf E=2\int\nabla G\times\mathbf A\,dA'$
with $\mathbf A=\hat{\mathbf z}\times\widetilde{\mathbf E}_s$, and

$$
\nabla\times\mathbf E=2\int\bigl[(\mathbf A\cdot\nabla)\nabla G+k_0^2G\,\mathbf A\bigr]dA',
\qquad
\mathbf H=\frac{\nabla\times\mathbf E}{ik_0Z_0},
$$

then integrates $\operatorname{Re}(\mathbf E\times\mathbf H^*)_z$ over the
aperture on a `pitch_m` grid. For the 2 µm disc and 2 µm aperture at 0.5 µm,
it agrees with the FFT path only once the FFT window reaches ~100 µm
(2×10⁻⁴); a 6.4 µm window over-reads by 19 %. Its H equals a finite-difference
curl of the direct E to 10⁻⁸. Passing `green_pitch_m` to the convergence
diagnostic adds this reference (`green_relative_difference`).

## Measured passive optical elements (#616)

`pyrite.detectors.optical_elements.CalibratedJonesElement` holds a *measured*
2×2 Jones transfer $T(E)$ on a bounded calibration grid in the source's Jones
basis, a calibrated maximum ray half-angle and provenance; its SHA-256
response hash enters the output provenance. `apply_optical_elements` maps
$J\to TJT^\dagger$ per energy, in beam order. Passivity (largest singular
value ≤ 1) is enforced at every node and preserved by complex-linear
interpolation (a convex combination of contractions is a contraction).
There is no extrapolation, no inferred optical constants, no etalon or
angle-resolved model, and the caller must state its collection half-angle.
`from_intensity_transmission` encodes a polarization-independent
$\sqrt{T(E)}\,\mathbb 1$ assumption. Tests: Malus law, lossless
retarder composition (QWP+QWP = HWP), interpolation and refusals.

## Shot-to-shot charge jitter in CTR templates (#617)

With per-shot population $N$ of mean $\bar N$ and RMS $r\bar N$, the
shot-averaged incoherent term keeps $\langle N\rangle=\bar N$ while the pair
term becomes $\langle N(N-1)\rangle=\bar N^2(1+r^2)-\bar N$, i.e. the
coherent envelope scales by $1+\bar N r^2/(\bar N-1)\to1+r^2$.
`GaussianCTRTemplates.with_charge_jitter(mean_electrons=, charge_rms_fraction=)`
applies exactly this, assuming charge is uncorrelated with bunch shape and the
scan voltages are linear shot averages. Ignoring it biases the fitted duration
(regression in `tests/detectors/test_optical_ctr_inference.py`). Common
arrival-time jitter cancels in the interferogram; per-shot charge
normalization and the extra voltage variance from jitter are not modeled.

