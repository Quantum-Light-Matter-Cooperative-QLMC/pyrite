# Sub-MeV planar channeling radiation: static screen

Investigation for [#585](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/585).
This page records where the existing static planar model admits a bound→bound
line below 1 MeV. It is a screening record, not a support envelope: no regime
here is supported, and no validation status changes. Claims stay with
`cr-transverse-eigenproblem`, `cr-dipole-matrix-element`,
`cr-doppler-line-energy` and `cr-initial-occupation`.

## Method

`checks/channeling_submev_screen.py` prints the full JSON record. It solves
`materials.channeling.transverse_states` for Si and diamond (110), (111) and
(100) at 293.15 K, for 30, 50, 100, 200, 300, 500, 750 and 1000 keV. By time
reversal, $E_n(k)=E_n(-k)$. The scan therefore covers $k\in[0,g/2]$ with
trapezoid weights, including the zone centre and edge where 1D band extrema
lie. A band counts as bound where it lies below the planar barrier. A pair
counts as radiating where both bands are bound and the velocity-form dipole
$|x_{fi}|$ exceeds $10^{-6}$ Å. Line energies are vacuum and forward
($\theta=0$), from `line_energy_eV`.

The reference uses $M=32$ harmonics and 64 half-zone intervals. Refining to
$M=48$ changes bound band edges and pair gaps by less than $4\times10^{-11}$ eV.
Halving to 32 intervals leaves the band extrema unchanged. It moves the
pair-availability edges by at most 0.18 eV in gap and by $1/128$ in zone
measure. Every case keeps the same bound bands and pairs under both changes.

Entry populations use a plane-aligned beam with no face refraction and
Gaussian divergence $\sigma\in\{0, 0.5, 1\}\,\psi_c$, where
$\psi_c=\sqrt{2U_0/pv}$ and $U_0$ is the well depth. "Bound" sums the
populations of states below the barrier at each sampled entry $k$.

## Results

Bound-band counts are given as bound somewhere in the zone / bound throughout.

| Plane | Depth (eV) | 30–100 keV | 200–500 keV | 750 keV | 1 MeV |
| --- | --- | --- | --- | --- | --- |
| Si(110) | 21.0 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Si(111) | 23.0 | 2 / 1 | 2 / 2 | 2 / 2 | 3 / 2 |
| Si(100) | 11.7 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |
| diamond(110) | 23.6 | 1 / 0 | 1 / 0 → 1 / 1 at 500 keV | 1 / 1 | 1 / 1 |
| diamond(111) | 28.4 | 1 / 1 | 2 / 1 | 2 / 2 | 2 / 2 |
| diamond(100) | 12.1 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

(110) and (100) in both crystals have at most one bound band throughout
30 keV–1 MeV. They have no bound→bound line in this model. They can still
channel, and free→bound capture is not excluded.

The radiating pairs are:

| Plane | T (keV) | Pair | Zone measure | Forward vacuum line (eV) | $\lvert x_{fi}\rvert$ (Å) |
| --- | --- | --- | --- | --- | --- |
| Si(111) | 30 | 1→0 | 0.51 | 15.5–19.0 | 0.50–0.74 |
| Si(111) | 50 | 1→0 | 0.55 | 17.6–22.1 | 0.47–0.72 |
| Si(111) | 100 | 1→0 | 0.66 | 22.7–29.6 | 0.42–0.67 |
| Si(111) | 200 | 1→0 | 1.00 | 33.0–45.0 | 0.38–0.60 |
| Si(111) | 300 | 1→0 | 1.00 | 43.8–55.9 | 0.38–0.55 |
| Si(111) | 500 | 1→0 | 1.00 | 67.5–78.3 | 0.39–0.49 |
| Si(111) | 750 | 1→0 | 1.00 | 99.6–108.1 | 0.39–0.44 |
| Si(111) | 1000 | 1→0 | 1.00 | 133.2–139.5 | 0.38–0.41 |
| Si(111) | 1000 | 2→1 | 0.16 | 124.6–126.7 | 0.63–0.65 |
| diamond(111) | 200 | 1→0 | 0.21 | 45.8–49.0 | 0.54–0.62 |
| diamond(111) | 300 | 1→0 | 0.37 | 62.0–72.2 | 0.41–0.55 |
| diamond(111) | 500 | 1→0 | 0.60 | 99.2–127.5 | 0.29–0.46 |
| diamond(111) | 750 | 1→0 | 1.00 | 154.4–211.2 | 0.25–0.39 |
| diamond(111) | 1000 | 1→0 | 1.00 | 217.6–271.8 | 0.25–0.35 |

At 1 MeV, Si(111) 2→0 also passes the dipole floor, but only at
$|x_{fi}|\le0.003$ Å. It is nearly parity-forbidden and is omitted above.

Zone measure is the fraction of the Bloch zone where the pair is available. It
is not an occupation probability. Below 200 keV in Si(111), and below 750 keV
in diamond(111), band 1 is bound only over part of the zone. The entry
populations at each $k$ then decide whether the line has any weight. Band 1
gets no population at plane alignment ($\sigma=0$, $k=0$) because it has the
wrong parity. With $\sigma=0.5\,\psi_c$ it holds 0.24–0.35 of the beam. That
figure is a static overlap only. Occupation evolution, dechanneling and capture
have not been run.

## What the screen shows

- In this model the lowest candidates are Si(111) from 30 keV and
  diamond(111) from 200 keV. Si(111) becomes zone-wide at 200 keV and
  diamond(111) at 750 keV. No other low-index plane qualifies.
- Every candidate line lies at 15–272 eV, in the EUV/soft-X-ray band. The
  bulk plasma energies are 31.05 eV (Si) and 38.19 eV (diamond)
  (`plasma_energy_eV`). The ratio of the top line to the plasma energy is 0.6
  for Si(111) at 30 keV and 0.94 at 100 keV. At 30–100 keV, the whole Si(111)
  line therefore sits at or below $\hbar\omega_p$, where the weak-refraction
  optics model fails. The vacuum line
  energy, intrinsic emission and escaped spectrum must be kept separate.
- The kinematic corrections omitted by `line_energy_eV` are small here.
  Photon recoil relative to the total energy is $\le1.8\times10^{-4}$.
  $U_0/pv\le4.9\times10^{-4}$, an upper bound on the longitudinal-speed
  correction. Neither is the binding limitation.
- The well depth does not depend on energy. The bound-band count rises with
  $\gamma m$ only slowly: Si(110) still has one band at 1 MeV, consistent with
  the walkthrough's computed count. Below 1 MeV the line energies scale with
  $2\gamma^2$ together with the dispersion of band 1. The scaling is not
  uniform.

## Not established

These items are open under #585 and none is settled by this screen:

- literature recovery for sub-MeV channeling-radiation photons, as distinct
  from electron diffraction, EELS and cathodoluminescence;
- longitudinal lattice harmonics, or full Bloch or multislice propagation, as
  against plane averaging;
- TDS and electronic scattering validity at these energies, along with
  occupation lengths and linewidths;
- dielectric modification of the emission and of escape through a finite
  slab;
- axial cases beyond the isolated-string flat-band gate. W [001] at 300 keV
  is the existing diagnostic.

[Watson & Koehler (1982)](https://doi.org/10.1103/PhysRevB.25.3079) observed
axial lines at 1–3 MeV. They saw no planar lines above about 1 keV. That
result is consistent with the eV-scale planar candidates above. It does not
show that sub-MeV planar radiation is absent, and it gives no anchor for it.
