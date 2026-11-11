# Sub-MeV channeling radiation: static screens

Investigation for [#585](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/585).
This page records where the existing static planar and axial models admit a
bound→bound line below 1 MeV. It is a screening record, not a support envelope: no regime
here is supported, and no validation status changes. Claims stay with
`cr-transverse-eigenproblem`, `cr-dipole-matrix-element`,
`cr-doppler-line-energy`, `cr-initial-occupation` and
`cr-axial-2d-eigenproblem`. The [axial screen](#axial-screen) follows the
planar one.

## Planar method

`checks/channeling_submev_screen.py` prints the full JSON record. It solves
`montecarlo.channeling.transverse_states` for Si and diamond (110), (111) and
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

## Planar results

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

## What the planar screen shows

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

## Axial screen

`checks/channeling_submev_axial_screen.py` prints the axial JSON record. It
solves `axial_states` for Si and diamond ⟨100⟩, ⟨110⟩ and ⟨111⟩ and W ⟨001⟩
and ⟨111⟩ at 293.15 K, at the same eight energies. The script's docstring
gives the definitions. In brief: the k scan covers the half zone (by time
reversal), states count as bound below the potential maximum $U_{\max}$, and
pairs need a velocity-form dipole above $10^{-6}$ Å and a transverse gap of
at least 1 eV. Below 1 eV, the members of one dispersing level give an
intra-level ratio, not a line. Line energies are vacuum and forward. The
record also gives the level count admitted by `axial_line_catalog`, the
production flat-band gate with 0.1 eV bandwidth.

Numerical checks cover all 64 rows. Raising the plane-wave cutoff from 2 to
3 well momenta at $k=0$ leaves every bound count unchanged. It moves levels
by at most 0.15 eV and adjacent gaps by at most 0.14 eV. Doubling the k grid
from 2 to 4 half-zone intervals, at cutoff 1.5, also keeps every bound count.
It moves band edges by at most 0.49 eV and pair gaps by at most 2.0 eV. The
set of pairs changes in 9 rows, all on multi-string axes: Si⟨110⟩ at 30 and
300–1000 keV, Si⟨111⟩ and diamond⟨110⟩ at 750 keV and above, and W⟨111⟩ at
500 keV. For those axes the pair availability, and the zone measures below,
are not converged at this sampling. Single-string pair sets are stable. The
full run took 71 min on 4 CPU threads, with a peak of 5.9 GiB, most of it in
W⟨111⟩.

Bound states are counted as bound throughout the zone / bound somewhere.
Degenerate members count separately. 50 keV matches 30 keV in every row.

| Axis | Strings per cell | Depth (eV) | 30 keV | 100 keV | 200 keV | 300 keV | 500 keV | 750 keV | 1 MeV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Si⟨100⟩ | 1 | 85.4 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 2 | 1 / 3 |
| diamond⟨100⟩ | 1 | 87.5 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| W⟨001⟩ | 1 | 820 | 3 / 4 | 4 / 4 | 4 / 4 | 4 / 5 | 5 / 7 | 8 / 9 | 8 / 9 |
| Si⟨111⟩ | 2 | 98.4 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 4 | 3 / 5 |
| diamond⟨111⟩ | 2 | 100.7 | 0 / 2 | 0 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| W⟨111⟩ | 2 | 946 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 10 | 16 / 17 | 18 / 18 | 18 / 18 |
| Si⟨110⟩ | 4 | 127.3 | 4 / 6 | 4 / 6 | 6 / 8 | 8 / 10 | 10 / 12 | 13 / 14 | 14 / 16 |
| diamond⟨110⟩ | 4 | 134.5 | 4 / 4 | 4 / 4 | 4 / 4 | 4 / 6 | 5 / 6 | 6 / 8 | 10 / 11 |

A pair needs an excited level on a string, so a bound count no larger than the
number of strings per cell has no on-string line. Those states are the
string ground orbitals, and their splitting is inter-string tunnelling.

**Single-string axes.** These give the cleanest on-string transitions.

- W⟨001⟩: the 2p doublet and 1s are bound throughout the zone from 30 keV. The
  2p→1s forward vacuum line is 333–337 eV at 30 keV, 387–390 eV at 50 keV,
  518–520 eV at 100 keV, 800 eV at 200 keV, 1119–1120 eV at 300 keV,
  1877 eV at 500 keV, 3036 eV at 750 keV and 4413 eV at 1 MeV, with
  $|x_{fi}|$ falling from 0.10 to 0.066 Å. At 300 keV the gap is 250.2 eV,
  which reproduces the existing diagnostic fixture. The 2p band disperses by
  2.6, 2.0, 1.1, 0.27 and 0.06 eV at 30, 50, 100, 200 and 300 keV, so
  `axial_line_catalog` admits the line only from 300 keV. That floor is set by
  the numerical bandwidth gate, not by the band structure. Below 200 keV a
  dispersive near-barrier band adds 14–59 eV lines to the 2p levels. Its line
  to 1s is nearly forbidden ($|x_{fi}|\le0.03$ Å).
- Si⟨100⟩: one bound level through 500 keV, so no line. At 750 keV a second
  band is bound over 0.12 of the zone (214 eV, 0.27 Å). At 1 MeV it is bound
  over 0.94 (342–368 eV, 0.17–0.22 Å).
- diamond⟨100⟩: at most one bound level through 1 MeV, so no line. Below
  500 keV it is bound over only part of the zone.

**Two-string axes.** Si⟨111⟩ and diamond⟨111⟩ have two bound states up to
500 keV and 1 MeV respectively: the ground orbitals of the two strings, with
no pair of gap ≥ 1 eV. Si⟨111⟩ gains lines at 750 keV (276–282 eV over up to
0.75 of the zone, about 0.2 Å) and at 1 MeV (438–464 eV). W⟨111⟩ has eight
bound states from 30 keV, and its lines run from 34 to 447 eV at 30 keV.

**Four-string axes.** Si⟨110⟩ below 200 keV and diamond⟨110⟩ below 300 keV
have four states bound throughout: the ground orbitals of the four strings,
which form two dumbbells. Their pairs are inter-string splittings, with
forward lines of 7–61 eV (Si) and 14–53 eV (diamond). The top line is 0.9–2.0
times $\hbar\omega_p$. They are not on-string transitions. Above that the bound
count grows. The flat catalog admits lines from 200 keV (Si) and at 1 MeV
(diamond). The pair sets are not k-converged from 300 keV (Si) and 750 keV
(diamond). Nothing here classifies which
admitted catalog lines are inter-string splittings.

Entry overlaps use an axis-aligned plane wave without refraction, with
isotropic 2D Gaussian divergence on a 4×4 Gauss–Hermite grid, where
$\psi_c=\sqrt{2U_0/pv}$ and $U_0$ is the full well depth. At $\sigma=0$, 0.29–1.0
of the beam lies below $U_{\max}$. Most of that weight sits in delocalized
near-barrier bands, not in radiating levels. At $\sigma=0.5\,\psi_c$ the
bound fraction falls to 0–0.20. The reason is that $\psi_c$ uses the full
depth, while the cell-averaged potential lies close to $U_{\max}$. Per-band
populations are not recorded. The W 2p doublet is odd, so it gets no weight
at $k=0$. These are static overlaps, not occupations.

### What the axial screen shows

- In this static model, W⟨001⟩ has a zone-wide 2p→1s line from 30 keV, at
  333 eV and above. Its current 300 keV floor comes from the 0.1 eV flat-band
  gate. Si⟨100⟩ has no line below 750 keV and diamond⟨100⟩ none through 1 MeV.
  The ⟨111⟩ light-element axes have no line below 750 keV (Si) and none
  through 1 MeV (diamond).
- Sub-MeV lines in light-element ⟨110⟩ axes are inter-string splittings near
  $\hbar\omega_p$. They are not the on-string transitions of the MeV
  literature.
- The kinematic corrections are larger in W. $U_0/pv$ reaches
  $1.4\times10^{-2}$ (⟨001⟩) and $1.6\times10^{-2}$ (⟨111⟩) at 30 keV, and
  photon recoil over total energy reaches $5.4\times10^{-3}$. In Si and
  diamond they stay at or below $2.3\times10^{-3}$ and $5.1\times10^{-4}$. A
  percent-level longitudinal correction would be comparable to the 2p
  dispersion at 30 keV. The finite-β audit
  ({ref}`sec-submev-w001-finite-beta`) below finds the actual corrections an order of magnitude smaller.
- The W lines at 0.33–4.4 keV lie well above $\hbar\omega_p=80.2$ eV, but
  within W N- and M-shell absorption. Escape needs the dielectric/attenuation
  bound; vacuum line energy is not a detected spectrum.

### Focused W ⟨001⟩ entrance-population audit

`checks/channeling_submev_w001_entry.py` is a bounded follow-up to the broad
axial screen: it resolves the static *incident* probability into the 1s
band (index 0), the combined 2p doublet (indices 1–2), other bound states,
unbound states and incident orders outside the truncated plane-wave basis.
The sum is unity before any conditional renormalization; in particular, a
large total bound overlap does not establish population of the 2p emitter.
At exact axial alignment the 2p projector has zero overlap by symmetry.
The doublet is grouped to avoid arbitrary eigenvector rotations in its
degenerate subspace. Rank labels may cease to identify orbitals at crossings.

Run from the repository root with
`PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_entry.py`
(optionally `--energies-kev ...`). Divergence is an isotropic 2D Gaussian in
the incident tilt, of width $\sigma = f\,\psi_c$. The overlaps vary on the
angular scale $g/p$ of one reciprocal vector, which is 0.19, 0.18 and 0.15
$\psi_c$ at 30, 100 and 300 keV. A tensor Gauss–Hermite grid of a few nodes
cannot resolve that: an earlier 4×4 versus 6×6 run disagreed by up to 0.36 in
the 2p weight and was off by up to a factor of three. The check therefore
writes the incident wavevector as $K = k + G$ with $k$ in the first zone. One
solve at $k$ gives the overlap of every basis order $G$, and the Gaussian
average becomes a midpoint sum over $k$ of $w(k+G)$ times the band projector
of row $G$. The sum over all $G$ is periodic in $k$, so the midpoint rule
converges exponentially once $\sigma$ exceeds the node spacing. The record
compares 16² and 24² zone grids and a 2→3 well-momentum cutoff. It sums the
Gaussian over an untruncated $G$ window as a normalization check. The
`outside_basis` weight is **numerical truncation**, not dechanneling.

The run at 30, 100 and 300 keV took 21 min on 4 CPU threads, with a peak of
0.65 GiB. Zone-grid deltas are at most $1.4\times10^{-3}$ and cutoff deltas at
most $1.3\times10^{-4}$. Every normalization is one to $10^{-6}$, and no weight
falls outside the basis. The 1s band and the 2p doublet are bound across the
whole zone at all three energies. The 2p doublet stays at least 9.3, 16.7 and
35.2 eV below band 3, so the rank labels cross nothing. The 2p dispersion is
2.8, 1.1 and 0.07 eV.

Incident probability at 24² zone nodes, cutoff 2:

| T | $\psi_c$ (mrad) | $f=\sigma/\psi_c$ | 1s | 2p doublet | other bound | unbound |
| --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 167.7 | 0 | 0.084 | 0 | 0.895 | 0.021 |
| | | 0.01 | 0.084 | 0.022 | 0.872 | 0.022 |
| | | 0.02 | 0.084 | 0.064 | 0.824 | 0.028 |
| | | 0.05 | 0.080 | 0.166 | 0.604 | 0.150 |
| | | 0.1 | 0.071 | 0.209 | 0.276 | 0.445 |
| | | 0.25 | 0.040 | 0.109 | 0.061 | 0.790 |
| | | 0.5 | 0.017 | 0.038 | 0.017 | 0.928 |
| 100 keV | 94.5 | 0 | 0.073 | 0 | 0.875 | 0.052 |
| | | 0.01 | 0.072 | 0.008 | 0.864 | 0.055 |
| | | 0.02 | 0.072 | 0.028 | 0.834 | 0.067 |
| | | 0.05 | 0.069 | 0.097 | 0.628 | 0.206 |
| | | 0.1 | 0.061 | 0.147 | 0.296 | 0.496 |
| | | 0.25 | 0.035 | 0.089 | 0.065 | 0.811 |
| | | 0.5 | 0.015 | 0.033 | 0.018 | 0.935 |
| 300 keV | 57.9 | 0 | 0.052 | 0 | 0.691 | 0.257 |
| | | 0.01 | 0.052 | 0.002 | 0.681 | 0.265 |
| | | 0.02 | 0.052 | 0.008 | 0.649 | 0.291 |
| | | 0.05 | 0.050 | 0.036 | 0.472 | 0.442 |
| | | 0.1 | 0.044 | 0.071 | 0.220 | 0.664 |
| | | 0.25 | 0.026 | 0.057 | 0.048 | 0.869 |
| | | 0.5 | 0.011 | 0.023 | 0.014 | 0.952 |

What the table shows, within this static plane-wave model:

- The emitter is thinly populated at entry. The 1s band takes 5–8 % of an
  aligned beam. The 2p doublet takes none at exact alignment, by symmetry.
  Its weight grows roughly as $\sigma^2$ for small $f$, peaks near
  $f\approx0.1$ (0.21, 0.15 and 0.07 at 30, 100 and 300 keV), and falls off as
  the beam leaves the well.
- Most of the aligned bound weight sits in "other bound" bands, near the
  barrier and spread over the cell. The broad screen found their line to 1s
  nearly forbidden. This weight is not 2p→1s emitter population.
- At a fixed fraction of $\psi_c$, the 2p entry weight falls with energy.
  Since $\psi_c$ itself shrinks as $1/\sqrt{pv}$, a fixed angular divergence in
  mrad is a larger fraction of $\psi_c$ at higher energy.
- The 2p→1s line needs 2p population. These entry weights give no
  occupation evolution, coherence, linewidth or yield. They set only the
  scale of the initial 2p population at an ideal surface.

These overlaps omit surface refraction and the entrance potential step,
incoherent scattering, capture, occupation evolution, emission, absorption
and detector response. They give no support classification.

(sec-submev-w001-finite-beta)=
### Finite-β audit of the W ⟨001⟩ line

`checks/channeling_submev_w001_kinematics.py` tests the two approximations
between the transverse eigenvalues and the forward photon energy.

*Transverse Hamiltonian.* For a potential $V$ independent of the axial
coordinate $z$, the spin-free Klein–Gordon equation
$[(E-V)^2-c^2p^2-m^2c^4]\psi=0$ separates exactly. With
$\psi=e^{ip_zz/\hbar}\phi$ and $c^2p_z^2=E^2-m^2c^4-2E\varepsilon$,

$$
\Bigl[\frac{c^2p_\perp^2}{2E}+V-\frac{V^2}{2E}\Bigr]\phi=\varepsilon\phi .
$$

The transverse mass $E/c^2=\gamma m$ is therefore exact, not a paraxial
approximation; low $\beta$ does not by itself invalidate it. The production
Hamiltonian drops $-V^2/2E$. With $V=\bar V+U$, $\bar V$ the mean inner
potential and $U_{00}=0$, the dropped term is a constant plus
$-U\bar V/E-U^2/2E$. The check adds both, using
$\bar V=-35.75$ eV from `mean_inner_potential_eV`
(`cr-mean-inner-potential`) and the exact convolution of the stored $U$
coefficients for $U^2$.

*Line energy.* The production line
$\hbar\omega=\Delta\varepsilon/(1-\beta)$ (forward) is first order in
$\hbar\omega/E$ (`cr-doppler-line-energy`). The check solves the exact forward
conservation of total energy and $p_z$, with $\varepsilon$ re-solved at the
final energy:
$cp_z(E,\varepsilon_{2p}(E))-cp_z(E-\hbar\omega,\varepsilon_{1s}(E-\hbar\omega))=\hbar\omega$.
The photon is on the axis, so the Bloch $k$ is unchanged. A unit test checks
the secant root against an independent bracketed solve.

Both are evaluated at $k=0$ with cutoffs 2 and 3 well momenta. The run took
11 min on 4 CPU threads, with a peak of 1.3 GiB. Relative changes agree
between the cutoffs to $2\times10^{-6}$.

| T | Gap (eV) | KG gap change | First-order line (eV) | Exact kinematics | Both | $\hbar\omega/E$ |
| --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 226.52 | $+7.5\times10^{-4}$ | 337.28 | $-1.09\times10^{-3}$ | $-3.4\times10^{-4}$ | $6.2\times10^{-4}$ |
| 50 keV | 229.08 | $+7.2\times10^{-4}$ | 390.04 | $-1.18\times10^{-3}$ | $-4.5\times10^{-4}$ | $7.0\times10^{-4}$ |
| 100 keV | 234.88 | $+6.7\times10^{-4}$ | 519.90 | $-1.39\times10^{-3}$ | $-7.2\times10^{-4}$ | $8.5\times10^{-4}$ |
| 200 keV | 243.89 | $+5.8\times10^{-4}$ | 800.45 | $-1.79\times10^{-3}$ | $-1.21\times10^{-3}$ | $1.1\times10^{-3}$ |
| 300 keV | 250.21 | $+5.2\times10^{-4}$ | 1119.62 | $-2.17\times10^{-3}$ | $-1.65\times10^{-3}$ | $1.4\times10^{-3}$ |
| 500 keV | 257.39 | $+4.2\times10^{-4}$ | 1876.87 | $-2.87\times10^{-3}$ | $-2.45\times10^{-3}$ | $1.9\times10^{-3}$ |
| 750 keV | 260.41 | $+3.4\times10^{-4}$ | 3035.60 | $-3.68\times10^{-3}$ | $-3.35\times10^{-3}$ | $2.4\times10^{-3}$ |
| 1 MeV | 259.99 | $+2.8\times10^{-4}$ | 4412.57 | $-4.44\times10^{-3}$ | $-4.16\times10^{-3}$ | $2.9\times10^{-3}$ |

Relative changes are against the production gap or line at cutoff 2. "Both"
is the exact root with the KG terms.

- The KG term lowers 1s by 0.11–0.18 eV and 2p by 0.01–0.04 eV, because $U$
  is deepest at the string core. The gap widens by at most $7.5\times10^{-4}$.
  $\bar V$ contributes $-\bar V/E\approx6.6\times10^{-5}$ of that at 30 keV.
  The scale $U_0/pv$ overstates the effect: only the difference of
  $\langle U^2\rangle$ between the two levels enters the gap.
- The exact kinematic root lies below the first-order line by about
  $1.5\,\hbar\omega/E$. The correction grows with energy, since
  $\hbar\omega/E\approx2\gamma\Delta\varepsilon/mc^2$. It is not a
  low-energy breakdown.
- At 30–100 keV the combined shift is 0.11–0.38 eV. That is comparable to
  the basis-cutoff change in the line (0.18–0.24 eV) and well below the 2p
  dispersion. The 2.8 eV dispersion at 30 keV spreads the forward line by
  about 4 eV. At 300 keV–1 MeV the shift is 1.8–18 eV, larger than both. A
  promoted line at these energies would need the exact root, or an explicit
  $O(\hbar\omega/E)$ tolerance.

This audit omits spin. The squared Dirac equation adds a term
$i\hbar c\,\boldsymbol\alpha\cdot\nabla V$ that the Klein–Gordon form
lacks; it is not bounded here. Also omitted are longitudinal lattice
harmonics, the dielectric response, off-axis emission (whose transverse photon
momentum changes the Bloch $k$), and $k\neq0$.

## Not established

These items are open under #585 and none is settled by this screen:

- full-text literature recovery for sub-MeV channeling-radiation photons, as
  distinct from electron diffraction, EELS and cathodoluminescence; the
  cursory search below found no sub-MeV anchor;
- longitudinal lattice harmonics, or full Bloch or multislice propagation, as
  against plane averaging;
- TDS and electronic scattering validity at these energies, along with
  occupation lengths and linewidths;
- dielectric modification of the emission and of escape through a finite
  slab;
- axial occupation evolution, radiating weight and capture; the
  [axial screen](#axial-screen) records zone-wide availability, and the
  focused W ⟨001⟩ audit only static per-band entry overlaps;
- a converged k grid and an inter-string versus on-string classification of
  multi-string axial pairs, and the flat-band catalog's admission of such pairs.
- spin terms of the squared Dirac equation, which the finite-β audit omits.

[Watson & Koehler (1982)](https://doi.org/10.1103/PhysRevB.25.3079) observed
axial lines at 1–3 MeV. They saw no planar lines above about 1 keV. That
result is consistent with the eV-scale planar candidates above. It does not
show that sub-MeV planar radiation is absent, and it gives no anchor for it.

## Literature: cursory search (2026-11-10)

A short web search looked for photon spectra from electrons below 1 MeV in
crystals. It found no report of a measured transverse bound→bound line below
1 MeV. That is not proof of absence. Every item below was read only as an
abstract or bibliographic record; none is full text, and none is a validation
anchor.

- **Coherent bremsstrahlung (CB) at microscope energies.** Spence, Reese,
  Yamamoto and Kurizki reported broad zone-axis peaks in TEM X-ray
  microanalysis spectra, identified by their orientation and beam-energy
  dependence (Phil. Mag. B **48**, L39 (1983),
  [record](https://scilit.com/publications/982ef0699fb3e1908c457a69709f02d5)).
  Reese, Spence and Yamamoto followed with "Coherent bremsstrahlung from
  kilovolt electrons in zone axis orientations", Phil. Mag. A **49**, 697
  (1984), [doi:10.1080/01418618408233297](https://doi.org/10.1080/01418618408233297).
  A secondary source treats CB and channeling radiation as two aspects of one
  process; the paper itself was not read. Later TEM work measured CB in Si
  ⟨110⟩ and ⟨100⟩ at 80–200 keV (Pantel et al., record only). Duval and
  Delacour found the elementary CB formula insufficient to identify such
  peaks ([MMM 1996](https://mmm.edpsciences.org/10.1051/mmm:1996103)). These
  peaks come from the lattice period along the beam, at keV photon energies.
  They are not transverse bound→bound lines. They are a sub-MeV background to
  separate, and evidence that zone-axis photon spectra are measurable in this
  geometry. The 1984 paper is the first full-text target.
- **Transverse bound states without emission.** An on-column 2p state with
  topological charge ±1 has been discussed for vortex-beam STEM
  ([arXiv:1204.0021](https://arxiv.org/abs/1204.0021)), the same 1s/2p
  structure as the W ⟨001⟩ emitter here. Paulauskas and Klie model kV-scale
  bound states as decaying roughly exponentially with depth
  ([arXiv:1811.09138](https://arxiv.org/abs/1811.09138)). That bears on
  occupation lengths, not on emission. STEM s-state (1s) models and
  channelling/ALCHEMI reviews treat propagation only.
- **Lowest-energy CR experiments found.** The Fermilab/ASTA design targets a
  line near 1.2 keV from about 4 MeV electrons
  ([arXiv:1407.6053](https://arxiv.org/abs/1407.6053); Blomberg et al.,
  FEL2013 MOPSO07). No measured spectrum was found. Watson and Koehler (above)
  remain the lowest-energy measured axial lines known here, at 1–3 MeV.
