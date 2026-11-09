# PXR/CBS downstream carbon filter: conditional first-interaction check (#411)

**Status:** Research-only geometry, EPDL2025 channel rates and exact
first-interaction bookkeeping. This is a follow-on to
[`cxr-boundary-photon-source.md`](cxr-boundary-photon-source.md), **not** a
physical PXR/CBS boundary source, a photon tracker, or a fluorescence/scatter
prediction. No CXR default, production `Result`, photon tracker admission, or
validation status changes.

## What this check resolves

The proposed h-BN (921 nm), 20 keV electron, 17° tilt, 130° azimuth,
flight-incoherent combined PXR/CBS, 0.25–1.5 keV study uses a **chosen**
0.5 µm carbon filter. Before claiming the detector receives a meaningful
scattered-in or fluorescent component, determine its actual material optical
depths in this energy band. We use EPDL2025 integrated photon interaction
cross sections, already the project's narrow-beam removal data, for a separately
**assumed** amorphous-carbon density of 2.0 g/cm³ (not an experimental
measurement of a particular foil). The geometry check chooses a finite
4 mm-radius plate 10 mm downstream of the sample and the prescribed
2.922 mm-radius ideal disk 20 mm from the sample, both normal to the detector
axis. The latter subtends ~0.066 sr **from the sample center**. The plate
radius/offset are additional numerical study parameters, not published ones.

For carbon atomic number density $n$ in Å$^{-3}$, EPDL cross section
$\sigma_k(E)$ in Å$^2$, and actual geometric path $\ell$ in mm,

$$
 \mu_k(E)\ [{\rm mm}^{-1}] = 10^7 n\sigma_k(E),\qquad
 \tau=\ell\sum_k\mu_k,\quad P_0=e^{-\tau},\qquad
 P_{1,k}=\frac{\mu_k}{\sum_j\mu_j}\left(1-e^{-\tau}\right).
$$

The last quantity is the probability the **first interaction** occurs through
channel $k$ before the ray leaves the plate. The five channels are EPDL
photoelectric, Rayleigh (`coherent`), Compton (`incoherent`), nuclear-field pair
and electron-field pair; the pair terms vanish in the stated soft-X-ray band.
For zero total attenuation, $P_0=1$ and all $P_{1,k}=0$. The partition is
nonnegative and $P_0+\sum_kP_{1,k}=1$ to rounding precision. `-expm1(-tau)`
protects thin-filter numerical accuracy. This treats independent-atom EPDL
cross sections, **not** coherent lattice diffraction.

The finite-cylinder intersection accounts for side exits and tangent rays;
blindly using $0.5\,\mathrm{µm}/|\cos\theta|$ overestimates grazing paths
that leave the plate radially. A front-sided ray/detector-disk intersection
provides the direct vacuum acceptance. The reference optical-depth table
printed by the script uses the sample center as a **geometry-only axis probe**.
It is not a claimed origin distribution for escaped PXR/CBS.

## Three controls and what cannot yet be inferred

Given a **separately derived** validated joint set of escaped boundary cells
$(\mathbf x_j,E_j,\hat{\mathbf n}_j,w_j)$ (weights already photons per
incident electron), the analysis-only `boundary_controls` calculates:

- **Vacuum:** $\sum_{j\in\mathrm{detector\ hit}}w_j$.
- **Plate with collisions disabled:** the **same** geometric sum, not the
  narrower Beer–Lambert transmission. This is a necessary identity control.
- **Plate, uncollided direct:**
  $\sum_{j\in\mathrm{hit}}w_j\exp(-\tau_j)$.

It also tabulates $\sum_jw_j P_{1,k,j}$ across **all source directions** that
intersect the plate, including those outside the detector's direct cone. These
are nonnegative *budgets for first interactions*, not detector yields. The
Rayleigh/Compton first-interaction budgets are loose upper bounds on the number
of singly initiated scattered photons that could *possibly* enter the detector;
they are **not** angular redistribution probabilities. The photoelectric
budget is **not** fluorescence: EADL transition yields, competing Auger decay,
secondary escape, and detector response have not been applied. Neither bound
includes subsequent collisions or electron feedback. The direct uncollided
population is disjoint from first-interacted photons, so no total is counted
twice. The finite plane distance clips interaction events beyond the detector
plane when that plane is ahead of the ray.

This check cannot yet provide physically meaningful absolute numbers for
source-weighted detector counts: `mc_cxr_spectrum` contains only an escaped
energy/angle marginal and **cannot** fill in the missing exit positions,
polarization/coherence, or finite-distance wave propagation. The actual
boundary measure and vacuum source closure remain separate scientific gates;
not even a correct plate calculation supplies them.

## Reproduction and next gate

Run `uv run python checks/cxr_filter_first_collision_411.py` after fetching
the pinned EPDL tables (`uv run pyrite tables fetch epdl`). It prints EPDL
optical depths, uncollided probabilities and first-interaction partitions at
250, 280, 290, 500, 1000 and 1500 eV, straddling the carbon K edge. These
numbers are calculated at runtime from the active pinned table; no
hand-entered attenuation constants or purported physics results are shipped.
Run the CPU-only contract tests with
`uv run pytest -q tests/checks/test_cxr_filter_first_collision_411.py`.
The script requires no CXR photon source and does not launch photon transport.

To move beyond this slice, (1) establish a physically supported joint boundary
measure (or document a no-go for a positive radiance at the demanded spatial
resolution), (2) close its vacuum angular marginal against the combined
PXR/CBS scorer with finite-distance/wave-validity criteria, and **only then**
(3) run position-sensitive, off-cone and shadowed physical scattering studies
with polarization and uncertainty controls, comparing full hits and a
separately validated conditional estimator. Do not infer detector fluorescence
yields from a carbon photoelectric interaction probability.
