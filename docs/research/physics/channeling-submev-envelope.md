# Sub-MeV channeling radiation: static screens

Investigation for [#585](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/585).
This page records where the existing static planar and axial models admit a
bound→bound line below 1 MeV. It is a screening record, not a support
envelope. The draft support matrix ({ref}`sec-submev-support-matrix`) classifies
every case for design review, but no regime here is supported, and no
validation status changes. Claims stay with
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
(diamond). The [multi-string
classification](#multi-string-classification) below sorts the admitted
catalog lines.

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
- The refractive index of the medium shifts the line more than either
  correction at 30–300 keV: by −1.9 to −6.6 eV (see
  {ref}`sec-submev-optics`).

*Spin correction.* `checks/channeling_submev_w001_spin.py` adds the first-order
Dirac correction to the spin-free Hamiltonian. It takes the matrix element
of $V$ between normalized positive-energy plane-wave spinors at energy $E$,
each on shell:

$$
V_q\,u^\dagger(\mathbf p')u(\mathbf p)=V_q\,\frac{E+mc^2}{2E}\Bigl[1+\frac{c^2\bigl(\mathbf p'\cdot\mathbf p+i\boldsymbol\sigma\cdot(\mathbf p'\times\mathbf p)\bigr)}{(E+mc^2)^2}\Bigr].
$$

The spin-independent part minus one is a Darwin-type term. The spin part
couples the spin to the longitudinal momentum through
$(\hbar c\beta/2(E+mc^2))\,\boldsymbol\sigma\cdot(\nabla V\times\hat z)$,
which is the Thomas–BMT precession energy, and to the transverse orbital
motion through a $\sigma_z$ term that splits 2p. The problem is solved in the
plane-wave basis ⊗ spin at $k=0$, cutoffs 2 and 3. The spin-free solve
reproduces `axial_states` to $10^{-10}$ eV. The run took 3 min on 4 threads,
with a peak of 3.3 GiB.

| T | Darwin-type gap change | With spin, both 2p sub-levels | 2p fine structure (eV) | Line change (eV) |
| --- | --- | --- | --- | --- |
| 30 keV | $-2.5\times10^{-4}$ | $-3.0$ to $-1.9\times10^{-4}$ | 0.024 | −0.10 to −0.07 |
| 100 keV | $-2.3\times10^{-4}$ | $-2.6$ to $-1.6\times10^{-4}$ | 0.024 | −0.14 to −0.08 |
| 300 keV | $-1.7\times10^{-4}$ | $-1.9$ to $-0.9\times10^{-4}$ | 0.025 | −0.21 to −0.10 |
| 1 MeV | $-0.8\times10^{-4}$ | $-0.9$ to $-0.2\times10^{-4}$ | 0.018 | −0.40 to −0.10 |

The cutoffs agree to better than 1 %. The Darwin-type term raises 1s by
0.02–0.06 eV and barely moves 2p. The spin terms then split 2p by about
0.02 eV. Together they lower the gap by at most $3\times10^{-4}$, against
the $+7.5\times10^{-4}$ Klein–Gordon term, and by much less than the
refractive shift.

The correction is signed. Eliminating the lower Dirac component at fixed
$p_z$ gives exactly

$$
[(E-V)^2-m^2c^4-c^2p^2]\phi=-i\hbar c^2(\boldsymbol\sigma\cdot\nabla V)(E+mc^2-V)^{-1}(\boldsymbol\sigma\cdot\mathbf p)\phi .
$$

Divided by $2E$, the right side is
$c^2[-i\hbar\nabla V\cdot\mathbf p+\hbar\boldsymbol\sigma\cdot(\nabla V\times\mathbf p)]/2E(E+mc^2)$,
up to relative $O(V/(E+mc^2))\lesssim10^{-3}$. The Hermitian part of the
first term is the on-shell Darwin form above,
$+\hbar^2c^2\nabla^2V/4E(E+mc^2)$. Its anti-Hermitian part is a commutator
with the unperturbed Hamiltonian and gives no first-order shift. The
on-shell construction is therefore the consistent first-order correction
beyond Klein–Gordon. Taking one $p_z$ for every component instead leaves
components with transverse momentum off shell. It adds a spurious
$O(V^2/E)$ term, which belongs to the exact $-V^2/2E$ already treated, and
flips the Darwin-type term to $+9\times10^{-5}$ at 30 keV. That variant is
inconsistent. Since Klein–Gordon is exact for the spin-free part, what this
audit omits beyond first order is a relative $O(V/(E+mc^2))$ correction to
the Darwin and spin terms. A fresh-context check (2026-11-15) rederived the
spinor product, the Thomas–BMT coefficient, the Klein–Gordon separation and
this argument.

Still omitted from the audit: longitudinal lattice harmonics, off-axis
emission (whose transverse photon momentum changes the Bloch $k$), and
$k\neq0$. The dielectric response is covered separately in
{ref}`sec-submev-optics`.

### Multi-string classification

`checks/channeling_submev_multistring.py` addresses two open points of the
broad axial screen. Its pair sets were keyed by band rank and changed under k
refinement. It also could not tell an on-string transition from a
tunnelling splitting of string ground orbitals. Run it with
`PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_multistring.py`
(optionally `--cases silicon:110 ...` and `--energies-kev ...`).

*Classifier.* For two Bloch states at the same $k$, the integrand of the
velocity-form matrix element
$\mathbf p_{fi}=\int_\text{cell}\psi_f^*(-i\nabla)\psi_i$ is cell-periodic.
The integral therefore splits over any partition of the cell, with no origin
to choose. Let $D$ be the discs of radius $\rho=f\,d/2$ about every string,
where $d$ is the distance to the nearest distinct string, images included.
The on-string fraction is
$R(f)=\lVert\mathbf p_{fi}|_D\rVert_F/\lVert\mathbf p_{fi}\rVert_F$. The
Frobenius norm runs over both multiplets and both components, so $R$ does not
depend on how degenerate members are rotated.

- An on-string p→s transition carries its momentum flux on the string, so
  $R\to1$.
- For an inter-string splitting of two s orbitals, each disc integral of
  $s_A\nabla s_A$ vanishes by symmetry. The flux lies in the bridge between
  the strings.

The class uses $f=0.5$ ($\rho=d/4$): on-string for $R\ge0.8$, inter-string
for $R\le0.4$, mixed in between. $R$ at $f=0.3$ and $0.7$ is recorded too. A
pair whose upper state lies above the percolation energy is *delocalized*.
That state is not confined to any string or closed string cluster, so $R$
does not classify it. On a real-space grid fine enough for products to be
exact, the full-cell integral reproduces the plane-wave matrix element to
$10^{-11}$.

*Convergence.* Each class records three quantities: the zone measure where a
pair of that class exists, the zone-weighted $\sum|x_{fi}|^2$, and the
weighted mean forward vacuum line. None of these depends on band ranks. They
are compared at 2, 4 and 6 half-zone intervals (cutoff 1.5 well momenta) and
against cutoff 2 at 2 intervals. A class counts as converged at this sampling
when, between two grids:

- its zone measure moves by less than 0.05;
- its weight moves by less than 10 %;
- its mean line moves by less than 2 eV.

A class present on only one grid counts as converged if its zone measure
there is below 0.05.

The full run covers Si and diamond ⟨110⟩ and ⟨111⟩, W ⟨111⟩, and the W ⟨001⟩
control, at the eight screen energies. It took 39 min of compute on 4 CPU
threads under an 8 GiB cap; most of that was W ⟨111⟩ at 500 keV–1 MeV. Peak
memory was not recorded. The grid integral matches the plane-wave matrix
element to $6\times10^{-11}$ in every row.

*Calibration.* The W ⟨001⟩ 2p→1s pair has $R=1.00$ at every energy and
every k node. The Si ⟨110⟩ dumbbell ground-orbital splittings have
$R=0.23$–$0.27$. Pairs weak enough for the outside integral to cancel part
of the disc integral can reach $R>1$; that happens only for on-string or
delocalized pairs.

Classification of the radiating pairs, read from the per-pair record at
6 intervals. "Delocalized" means the upper state lies above the percolation
energy:

| Axis | Below 1 MeV | Converged at this sampling |
| --- | --- | --- |
| Si⟨110⟩ | 30–500 keV: below percolation, every pair is inter-string ($R$ 0.23–0.27), 7–12 eV forward. Above percolation, 28–138 eV pairs are delocalized. 750 keV: catalog pairs at a 31 eV gap are on-string or mixed ($R$ 0.77–1.13). | Inter-string yes. Delocalized not always (weight up to 21 %). |
| diamond⟨110⟩ | 30 keV–1 MeV: every pair below percolation is inter-string ($R$ 0.20–0.31), 12–74 eV. At 1 MeV, the 439–509 eV pairs are delocalized or mixed over 0.08 of the zone. | Inter-string yes from 100 keV; at 30 keV the zone measure moves by up to 0.13. |
| Si⟨111⟩ | No pair with a gap of at least 1 eV through 500 keV. 750 keV: 278–286 eV, $R$ 0.91–0.93, but the upper state lies above percolation. 1 MeV: on-string 440–453 eV over 0.85 of the zone, $R$ 0.95–0.97. | Yes for the on-string line. |
| diamond⟨111⟩ | No pair with a gap of at least 1 eV through 1 MeV. | n/a |
| W⟨111⟩ | 30–300 keV: every pair is on-string. 2p→1s on each string: 405 eV (30 keV), 621 eV (100 keV), 1318 eV (300 keV), $R=1.00$. Upper-level→2p: 34–42, 70–76 and 233–235 eV. 500 keV–1 MeV: many near-degenerate bands; classes mix along the zone. | Yes through 300 keV, except a 0.06-zone mixed arm under the cutoff change. Not from 500 keV. |

The near-barrier percolation energy decides the "delocalized" label. In
Si ⟨111⟩ it lies 1.0 eV below $U_{\max}$, so the 750 keV pair with
$R\approx0.92$ is on-string in character, but its upper state is not
confined to a single cell.

*Flat catalog.* `axial_line_catalog` admits these multi-string pairs:

- Si ⟨110⟩ at 300 and 500 keV: pairs at 1.3–2.5 eV are inter-string
  ($R=0.25$–$0.27$). Splittings of 0.0–0.2 eV between near-degenerate ground
  orbitals are admitted too.
- Si ⟨110⟩ at 750 keV and 1 MeV: pairs at 31–33 eV are on-string
  ($R=0.77$–$1.13$; all at least 0.82 at 1 MeV).
- Si ⟨111⟩ at 750 keV and 1 MeV, and diamond ⟨110⟩ at 1 MeV: the single
  admitted line is a sub-0.2 eV ground-orbital splitting.

The flat-band gate therefore admits tunnelling splittings as lines. In this
static model, no light-element multi-string axis has an on-string line below
750 keV.

(sec-submev-optics)=
### Optics of the candidate lines

`checks/channeling_submev_optics.py` evaluates the medium at each candidate
line recorded above, at both ends of its range. It runs in about 2 s.

*Refractive Doppler factor.* A transverse transition $\Delta\varepsilon$
emits a photon of energy $\hbar\omega$ and, in the medium, momentum
$n\hbar\omega/c$. Conserving energy and $p_z$ to first order, as in the
finite-β audit, gives $\hbar\omega(1-\beta n\cos\theta)=\Delta\varepsilon$.
With $n=1-\delta$, the forward line is
$\Delta\varepsilon/[1-\beta(1-\delta)]$ rather than
$\Delta\varepsilon/(1-\beta)$. The small parameter is $\beta\delta/(1-\beta)$.
This takes the photon's momentum in the medium as $n\hbar\omega/c$ and is
derived here. No source specific to channeling radiation was found in the
project library, and the derivation has not had a fresh-context check.
$\delta$ is taken at the shifted energy by fixed-point iteration. No medium
line is reported where that parameter exceeds 0.3, because the
weak-refraction premise has then failed.

*Data.* The index comes from the Chantler isolated-atom $f_1, f_2$
(`crystal.optical_constants`). The EPDL total attenuation length is an
independent check of the absorptive part.

*Escape.* Mean transmission of a line emitted uniformly through a slab of
thickness $t$ and leaving along the normal:
$T=(1-e^{-t/L})/(t/L)$.

| Line | $f_1/Z$ | $\delta$ at vacuum line | $L_\text{abs}$ (nm), Chantler / EPDL | Medium shift (eV) | $T$ at 100 nm |
| --- | --- | --- | --- | --- | --- |
| W⟨001⟩ 2p→1s, 30 keV, 333 eV | 0.40 | $1.2\times10^{-2}$ | 34 / 31 | −1.9 | 0.32 |
| W⟨001⟩, 100 keV, 518 eV | 0.51 | $6.1\times10^{-3}$ | 54 / 48 | −3.8 | 0.46 |
| W⟨001⟩, 300 keV, 1119 eV | 0.65 | $1.7\times10^{-3}$ | 191 / 175 | −6.6 | 0.78 |
| W⟨001⟩, 500 keV, 1877 eV | 0.33 | $3.0\times10^{-4}$ | 125 / 137 | −3.1 | 0.69 |
| W⟨001⟩, 1 MeV, 4412 eV | 0.94 | $1.6\times10^{-4}$ | 728 / 690 | −11.0 | 0.93 |
| W⟨111⟩ 2p→1s, 30 / 100 / 300 keV | 0.45–0.64 | $8.8\times10^{-3}$–$1.2\times10^{-3}$ | 43–275 | −1.7 / −3.5 / −5.4 | 0.39–0.84 |
| Si⟨111⟩ on-string, 1 MeV, 440–453 eV | 0.95 | $2.3\times10^{-3}$ | 369–395 / 342–366 | −17 | 0.88 |
| Si(111) planar, 1 MeV, 133–140 eV | 0.24–0.29 | $6.4$–$7.1\times10^{-3}$ | 40–42 | −9 to −11 | 0.37 |
| diamond(111) planar, 1 MeV, 218–272 eV | 0.35–0.61 | $9.5$–$3.4\times10^{-3}$ | 702–1157 | −46 to −25 | 0.93–0.96 |
| diamond(111) planar, 750 keV, 154–211 eV | 0.63–0.69 | $2.1$–$1.0\times10^{-2}$ | 321–656 / 268–551 | refused / −28 | 0.86–0.93 |
| Si⟨100⟩ 2→0, 1 MeV, 342–368 eV | 0.92–0.93 | $3.8$–$3.3\times10^{-3}$ | 203–241 / 188–223 | −22 to −20 | 0.79–0.82 |

The shift uses $\delta$ at the medium line, not the tabulated vacuum value.
The two differ most at the W M₄ edge ($\delta=2.6\times10^{-4}$ at 1873 eV)
and for diamond(111) at 1 MeV ($1.7\times10^{-2}$ at 172 eV). "Refused"
means $\beta\delta/(1-\beta)$ exceeds 0.3 during the iteration; this also
refuses the 127.5 eV upper end of diamond(111) at 500 keV. Rows not shown
lie below $3\hbar\omega_p$, where no medium line is given (next
paragraph). The JSON record has every row and both ends of each range.

What the record shows:

- **Refraction is the largest correction to the W lines.** At 30–100 keV, the
  medium lowers the 2p→1s line by 1.9–3.8 eV. That is 10–17 times the combined
  Klein–Gordon and exact-kinematics shift of the finite-β audit, and
  comparable to the 2p dispersion. At 1 MeV the shift is −11 eV against
  −18 eV from kinematics. In light elements at 750 keV–1 MeV,
  $\beta\delta/(1-\beta)$ at the medium line reaches 0.27, close to the 0.3
  cap. The diamond(111) line then moves by
  up to 21 %, so a vacuum line energy is not a prediction there.
- **The W 500 keV line sits on the M₄ edge** (1877 eV against 1874 eV in
  EPDL). There $f_1/Z$ drops to 0.33. Isolated-atom tables do not resolve
  solid-state edge structure, so the escape and line shift at that energy
  have no error bound.
- **Escape is strong but not prohibitive above $3\hbar\omega_p$.** Absorption
  lengths are 34–730 nm for W and 40–1160 nm for the light-element MeV
  lines. A 100 nm slab passes 32–96 %. Thicker targets lose the W lines below
  300 keV quickly: at 30 keV, $T=0.07$ through 500 nm.
- **Below $3\hbar\omega_p$ there is no optics model.** This covers the
  Si(111) planar lines at 30–500 keV (15–78 eV), the diamond(111) lines at
  200–500 keV, and every ⟨110⟩ inter-string line (7–58 eV). In that range
  $\delta$ reaches 0.1–2.7, no medium line is given, and in Si below about
  50 eV Chantler and EPDL disagree by more than a factor of 1.5. Those Si
  lines lie below the Si L edge (99.8 eV), in the EUV window, where the
  tables give $L_\text{abs}\approx0.3$–0.6 µm. The tables cannot bound that
  number, because collective and band-structure response dominate there.

The record has no transition-radiation background, surface or interface
term, angular aperture, crystal-field or excitonic structure, or detector
response. It is a static, homogeneous, isotropic estimate from tabulated
atomic data, not a dielectric theory of channeling radiation.

(sec-submev-w001-zone-line)=
### Zone-resolved W ⟨001⟩ line

`checks/channeling_submev_w001_dispersion.py` resolves the W ⟨001⟩ 2p→1s
line over the Brillouin zone. `axial_line_catalog` uses $k=0$ levels and
refuses a band that spreads by more than 0.1 eV, so it refuses this line below
300 keV. Widening that gate would put the $k=0$ energy on a band that spreads
by up to 2.8 eV.

*Method.*

- A forward photon carries no transverse momentum. Every transition is
  therefore vertical in $k$.
- At each node of a midpoint zone grid, each 2p band $b$ emits to 1s at its
  gap $\Delta_b(k)$. Its velocity-form dipole is $x_{0b}(k)$, and its rate per
  path length comes from `dipole_rate_per_ang` (`cr-dipole-rate`).
- Each band is weighted by its static entry population
  $P_b(k)=\sum_G w(k+G)\,|\langle G|b\rangle|^2$, for an isotropic Gaussian
  divergence of width $f\,\psi_c$. This is the zone integral of the
  focused entrance audit above.
- The record gives the distribution of the forward vacuum line weighted by
  $P_b(k)$ times the rate, the mean shifted into the medium
  ({ref}`sec-submev-optics`), and the total initial rate.
- Coherences between the two 2p bands are dropped. The record measures that
  omission: the full entry state changes the dipole strength by at most
  0.26 % (30 keV, $f=0.05$), and by less than $4\times10^{-4}$ from
  100 keV. The doublet splits by up to 2.75 eV at 30 keV.
- A unit test checks that at $k=0$ the record reproduces the catalog's gap
  and its doublet-summed dipole strength.

Run with
`PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_dispersion.py`.
It took 14 min on 4 CPU threads, with a peak of 0.66 GiB, under an 8 GiB
cap.

*Convergence.* Refining the zone grid from 12² to 16² moves the mean, the
width and the 2p population by less than $10^{-3}$ eV and $2\times10^{-7}$,
and the rate by at most $1.1\times10^{-4}$. Raising the basis cutoff from 2
to 3 well momenta moves the mean by 0.16–0.24 eV and the rate by about 0.1 %,
consistent with the broad screen. The 1s band and the 2p doublet stay bound
across the whole zone.

At $f=0.1$, near the peak of the 2p entry weight:

| T | $k=0$ line (eV) | Weighted mean (eV) | Mean − $k=0$ (eV) | rms width (eV) | 5–95 % (eV) | Medium shift (eV) | 2p population | Initial rate (µm⁻¹) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 337.28 | 335.13 | −2.14 | 1.20 | 333.65–337.27 | −1.90 | 0.209 | $9.2\times10^{-4}$ |
| 50 keV | 390.04 | 388.15 | −1.89 | 1.09 | 386.79–390.07 | −2.57 | 0.188 | $7.3\times10^{-4}$ |
| 100 keV | 519.90 | 518.65 | −1.25 | 0.77 | 517.67–519.92 | −3.83 | 0.147 | $5.4\times10^{-4}$ |
| 200 keV | 800.45 | 800.00 | −0.45 | 0.30 | 799.60–800.47 | −5.64 | 0.098 | $3.9\times10^{-4}$ |
| 300 keV | 1119.62 | 1119.48 | −0.14 | 0.10 | 1119.34–1119.63 | −6.56 | 0.071 | $3.3\times10^{-4}$ |

From $f=0.05$ to $0.25$, mean − $k=0$ runs from −2.42 to −1.74 eV at
30 keV, −1.54 to −1.05 eV at 100 keV, −0.60 to −0.40 eV at 200 keV, and
−0.20 to −0.13 eV at 300 keV. The rms width grows with $f$, from 0.99 to
1.30 eV at 30 keV.

What the record shows:

- **The $k=0$ catalog line sits at the top of the band.** The weighted mean
  lies 1.7–2.4 eV below it at 30 keV and 0.13–0.20 eV below it at 300 keV.
  At 30–100 keV that offset is larger than the cutoff change and the
  finite-β and spin terms. It is comparable to the refractive shift.
- **The static dispersion width is about 1 eV rms at 30–50 keV.** It is
  0.8 eV at 100 keV and 0.1 eV at 300 keV. These widths are from population
  and dispersion only, before any scattering, decay or occupation evolution.
  They are not linewidths.
- **The line exists across the whole zone from 30 keV.** The 0.1 eV gate's
  300 keV floor is a numerical choice. A production record below 300 keV
  needs the zone-weighted mean and width, not the $k=0$ level.
- **The initial rate is $3$–$9\times10^{-4}$ photons per µm per electron.**
  It is highest at 30 keV, where the 2p entry weight is largest. It is a
  static entry rate, not a yield: 2p depopulation, dechanneling and
  absorption are not included.

(sec-submev-w001-occupation)=
### W ⟨001⟩ occupation evolution

`checks/channeling_submev_w001_occupation.py` follows the 1s and 2p
populations into the crystal. It is a diagnostic: the TDS and electronic jump
kernels are the MeV-calibrated ones, with no sub-MeV check.

**Method.**

- **Entry and propagation.** A plane wave enters a $4\times4$ supercell of
  W⟨001⟩ strings, 25 grid points per cell. Its carrier tilt is drawn from the
  same 2D Gaussian divergence $f\psi_c$ as the static entry audit.
  {py:class}`~pyrite.montecarlo.channeling.axial_mcwf.AxialMCWF` propagates
  each tilt with the TDS jumps of `axial_dissipator`, with or without the
  electronic jumps. Each case runs 128 tilts × 16 histories to 1000 Å in
  0.5 Å steps.
- **Observable.** The state $e^{i\mathbf K\cdot\mathbf r}\phi$ is projected
  exactly onto the Bloch bands at the $4^2$ crystal wavevectors
  $\mathbf K+\mathbf g\,\mathbf r/4$ that it occupies. Band 0 counts as 1s and
  bands 1–2 as 2p, if bound.
- **Rate.** The 2p→1s rate at each $\mathbf k$ uses that band's vertical gap
  and velocity-form dipole, and the doublet bands add incoherently. At entry
  the omitted coherence is at most 0.26 %
  ({ref}`sec-submev-w001-zone-line`).
- **Photons per electron.** The running depth integral of that rate.

A first run (array 13190) instead projected onto the $k=0$ orbitals cut to one
string cell. That fails below 1 MeV: the 2p band disperses by 2.6 eV at
30 keV, and a carrier tilt that does not fit the supercell is not periodic on
it. Per tilt, the 2p population was wrong by up to a factor of 3, so that run
is discarded.

**Checks.**

- **Entry.** At depth 0, every case reproduces the static 1s/2p populations
  of the focused entrance-population audit above within 0.003. The
  per-case standard error at entry is 0.001–0.004. Two examples:
  0.080/0.165 against 0.080/0.166 (30 keV, $f=0.05$) and 0.045/0.071 against
  0.044/0.071 (300 keV, $f=0.1$).
- **Convergence at $f=0.1$** (TDS + electronic), against the 0.5 Å, 25-point,
  $4\times4$ default:

  | Change | 30 keV | 100 keV | 300 keV |
  | --- | --- | --- | --- |
  | 0.25 Å step | −1.0 % | −0.7 % | 0.0 % |
  | 1.0 Å step | +4.5 % | +1.8 % | +0.8 % |
  | 32-point grid | +4.0 % | +3.1 % | +1.5 % |
  | $6\times6$ supercell | +0.3 % | | |

  These are changes in photons per electron at 1000 Å; the statistical error
  is about 3 %. The 1/e depths move by at most 4 %.
- **Survival is not converged.** On the 32-point grid, the fraction still in
  the simulation at 1000 Å rises from 0.25 to 0.42 at 30 keV, and the TDS
  jump count from 3.1 to 4.4 per electron. Survival and jump counts are
  therefore not reported. The cause is not established.

**Results.** Photons per electron in the 2p→1s line, TDS + electronic, with
the TDS-only value at 1000 Å for comparison. Standard errors at 1000 Å are
about $2\times10^{-7}$. Record:
`docs/validation/check-records/channeling-submev/w001-occupation-13219.json`.

| T | $f$ | 1s 1/e (Å) | 2p 1/e (Å) | 100 Å | 300 Å | 1000 Å | 1000 Å, TDS only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 0.05 | 7.1 | 79 | $4.6\times10^{-6}$ | $5.9\times10^{-6}$ | $6.4\times10^{-6}$ | $6.3\times10^{-6}$ |
| 30 keV | 0.1 | 7.0 | 64 | $5.1\times10^{-6}$ | $6.2\times10^{-6}$ | $6.9\times10^{-6}$ | $7.1\times10^{-6}$ |
| 30 keV | 0.25 | 7.3 | 60 | $2.4\times10^{-6}$ | $3.3\times10^{-6}$ | $4.1\times10^{-6}$ | $4.1\times10^{-6}$ |
| 100 keV | 0.05 | 11.3 | 133 | $2.9\times10^{-6}$ | $4.6\times10^{-6}$ | $5.8\times10^{-6}$ | $5.2\times10^{-6}$ |
| 100 keV | 0.1 | 10.9 | 104 | $3.9\times10^{-6}$ | $5.9\times10^{-6}$ | $7.3\times10^{-6}$ | $7.2\times10^{-6}$ |
| 100 keV | 0.25 | 10.9 | 108 | $2.2\times10^{-6}$ | $3.6\times10^{-6}$ | $4.9\times10^{-6}$ | $4.9\times10^{-6}$ |
| 300 keV | 0.05 | 16.7 | 249 | $1.4\times10^{-6}$ | $2.8\times10^{-6}$ | $4.4\times10^{-6}$ | $3.7\times10^{-6}$ |
| 300 keV | 0.1 | 16.6 | 110 | $2.4\times10^{-6}$ | $4.1\times10^{-6}$ | $6.1\times10^{-6}$ | $5.5\times10^{-6}$ |
| 300 keV | 0.25 | 17.7 | 122 | $1.9\times10^{-6}$ | $3.3\times10^{-6}$ | $5.4\times10^{-6}$ | $5.1\times10^{-6}$ |

What the record shows, under the MeV-calibrated kernels:

- **The yield is a few $\times10^{-6}$ photons per electron.** It is
  4–7 × 10⁻⁶ by 1000 Å at every energy, highest near $f=0.1$. Dividing by the
  initial rate gives an effective radiating length of about 75 Å at 30 keV,
  135 Å at 100 keV and 195 Å at 300 keV (the last truncated at 1000 Å).
- **1s empties within 7–18 Å, about ten times faster than 2p**
  (60–250 Å). The 1s is the most string-localized state and takes the most
  TDS.
- **At 30 keV the line is a surface effect.** About 73 % of the photons
  come from the first 100 Å. At 300 keV only 39 % do, and the yield is
  still rising at 1000 Å.
- **Electronic scattering matters only at 300 keV.** With electronic jumps,
  the 1000 Å yield changes by ±2 % at 30 keV. At 100 keV it is +12 % at
  $f=0.05$ and within ±2 % otherwise. At 300 keV it is +6 to +19 %, and the
  2p 1/e depth at $f=0.05$ doubles (249 against 117 Å). Small electronic
  kicks appear to return electrons to 2p. This is the term most exposed to
  the unvalidated sub-MeV electronic kernel.
- **Not included:** photon absorption and escape, capture from outside the
  supercell, longitudinal harmonics and stopping. The record is not a
  linewidth or a validated yield.

(sec-submev-w001-linewidth)=
### Scattering width of the W ⟨001⟩ line

`checks/channeling_submev_w001_linewidth.py` estimates the line's scattering
width (claim `cr-submev-axial-coherence-width`, unverified). It applies the
secular equation of `cr-line-shape` to the axial jump operators in the
flat-band limit. The radiating coherence $\langle 1s|\rho|2p\rangle$ decays
at

$$\kappa = \tfrac12(\Gamma_{2p} + \Gamma_{1s}) - J_\mathrm{TDS} - \mathrm{Re}\,J_\mathrm{el}.$$

- $\Gamma_n$ is the full loss rate of level $n$, over its string orbital: the
  in-grid TDS, the large-angle TDS sink and electronic excitation.
- The gains $J$ are the coherence that a jump carries over to the same pair,
  because it acts alike on ket and bra.

The transverse Lorentzian half width is $\hbar v\kappa$. The forward photon
width is that divided by $1-\beta$.

**Checks.** Three limits are tested:

- a spatially uniform phase jump gives no width;
- a loss-only kernel gives the lifetime width $(\Gamma_{2p}+\Gamma_{1s})/2$;
- a $q\to0$ electronic kick gives no width.

On W ⟨001⟩, the displacement quadrature reproduces the dissipator's TDS rate
to $6\times10^{-14}$, and the electronic nodes reproduce its electronic loss
to $5\times10^{-4}$. Record:
`docs/validation/check-records/channeling-submev/w001-linewidth.json`.

| T | $\Gamma_{1s}$ (Å⁻¹) | $\Gamma_{2p}$ (Å⁻¹) | Transverse FWHM (eV) | Gap (eV) | FWHM / gap | Forward photon FWHM / line (eV) | Saturation estimate (eV) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 0.350 | 0.036 | 239 | 226 | 1.05 | 356 / 337 | 158 (0.70 of gap) |
| 100 keV | 0.136 | 0.017 | 157 | 235 | 0.67 | 347 / 519 | 132 (0.56) |
| 300 keV | 0.079 | 0.014 | 135 | 250 | 0.54 | 604 / 1117 | 122 (0.49) |

What the record shows, under the MeV-calibrated kernels:

- **The 2p→1s line is not resolved at 30–300 keV.** Its scattering FWHM is
  half to all of the transition energy. The photon is a broad band
  hundreds of eV wide, not a line. The static dispersion width
  ({ref}`sec-submev-w001-zone-line`) is 0.1–1.2 eV and negligible beside it.
- **1s sets the width.** $\Gamma_{1s}$ is 85–91 % of
  $\Gamma_{1s}+\Gamma_{2p}$, and about a third of it is large-angle TDS. The
  1s population decay lengths, $1/(\Gamma_n - J_{nn})$, are 3.1, 8.0 and
  13.9 Å. The occupation run's 1/e depths are longer (7–17 Å) because it
  includes refilling.
- **The gains do not narrow the line.** The TDS gain is zero within
  $4\times10^{-4}$ Å⁻¹. The linear (dipolar) displacement term averages
  to zero over each centrosymmetric density, which leaves only
  $O(a^4)$. The electronic gain removes about 5 % of $\kappa$. Electronic
  excitation alone would give a transverse FWHM of 5–11 eV.
- **Kernel range.** The phase variance that one atom of its own string
  imprints on 1s, in-grid and density-weighted, is 0.70 rad² at 30 keV,
  0.27 at 100 keV and 0.15 at 300 keV. The jump operators expand
  $e^{i\phi}$ to second order, so at 30 keV the kernel is near the edge of
  its range. The full 1s TDS loss per atom (in-grid plus sink, all
  strings), $V$, is 1.0, 0.40 and 0.23. Scaling each level's TDS loss by the
  Gaussian phase-object saturation $(1-e^{-V})/V$ lowers FWHM/gap to 0.70, 0.56 and 0.49. That is a
  sensitivity estimate, not a kernel. The conclusion does not depend on it.
  The full phase object ({ref}`sec-submev-w001-tds-phase`) gives FWHM/gap
  0.70, 0.55 and 0.49 with the sink, the gap shift and the electronic term.
- **Not included:** the 2p dispersion (2.6 eV at 30 keV), transfer into other
  band pairs (non-secular) and any sub-MeV validation of the TDS kernel. When
  $\kappa\hbar v$ is comparable to the gap, the secular approximation itself
  is marginal. The FWHM is then an order-of-magnitude statement, which is
  enough to rule out a resolved line under these kernels.

(sec-submev-w001-tds-phase)=
### Strong-phase check of the TDS kernel

The TDS jumps expand one displaced atom's transmission $e^{i\delta\phi_u}$ to
second order in the phase fluctuation. At 30 keV the 1s orbital sees a
per-atom phase variance near 1 rad², so `checks/channeling_submev_w001_tds_phase.py`
(claim `cr-submev-tds-phase-object`, unverified) keeps the full phase object
instead. It uses the same projected potential and Einstein displacements.

**Method.** Per atom of string $s$, with
$M_n(u)=\int|w_n|^2e^{i\delta\phi_u}$ and independent atoms:

- the coherent-wave loss is $-2\sum_s\lambda_s\ln|\mathrm E_uM_n|$;
- the population retained in $n$ decays at
  $-\sum_s\lambda_s\ln\mathrm E_u|M_n|^2$;
- the 2p→1s coherence decays at
  $\kappa=-\sum_s\lambda_s\ln|\mathrm E_uM_{1s}M_{2p}^*|$.

To second order these are the dissipator's rates. The phase is band-limited
to the grid, so the record refines a $3\times3$ supercell until the
large-angle sink is about 1 % of the 1s loss (128 points per cell). Changing
the displacement quadrature from 24 to 48 nodes per axis then moves $\kappa$
by 0.1 % or less. At 64 and 96 points the values are not converged. Record:
`docs/validation/check-records/channeling-submev/w001-tds-phase.json`.

**Checks.** Two analytic limits are tested:

- a phase linear in $u$ is Gaussian, so exact and second order agree;
- a uniform quadratic phase follows the χ² characteristic function and
  saturates below the second-order value.

| T | $\kappa_\mathrm{TDS}$, 2nd order (Å⁻¹) | $\kappa_\mathrm{TDS}$, phase object (Å⁻¹) | Ratio | 1s population loss, 2nd order / phase object (Å⁻¹) | TDS FWHM incl. sink (eV) | Gap shift (eV) | + electronic: FWHM / shifted gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 keV | 0.174 | 0.0975 | 0.56 | 0.317 / 0.166 | 129 | −28 | 140 / 199 eV = 0.70 |
| 100 keV | 0.0685 | 0.0510 | 0.74 | 0.122 / 0.088 | 112 | −20 | 119 / 214 eV = 0.55 |
| 300 keV | 0.0421 | 0.0359 | 0.85 | 0.071 / 0.059 | 112 | −14 | 116 / 236 eV = 0.49 |

The TDS FWHM adds half the residual large-angle sink (1 % of the 1s loss
at 128 points) as pure loss. The gap shift is the phase of
$\mathrm E_u[M_{1s}M_{2p}^*]$: a non-Gaussian correction beyond the mean
potential, $\hbar v\sum_s\lambda_s\arg\mathrm E_u[M_{1s}M_{2p}^*]$ with
$\phi=-V/\hbar v$. The gap narrows by 12 %, 9 % and 6 %. The electronic part
is that of {ref}`sec-submev-w001-linewidth`.

What the record shows:

- **The second-order kernel overstates the TDS width.** The overstatement is
  1.8× at 30 keV, 1.3× at 100 keV and 1.2× at 300 keV. The 1s population
  decay length at 30 keV becomes 6.0 Å instead of 3.2 Å. That is closer to
  the occupation run's 7 Å, which also includes refilling.
- **The line is still unresolved.** With the strong-phase TDS, the sink and
  the electronic term, the transverse FWHM is 0.49–0.70 of the shifted gap.
  The forward photon FWHM is about 210 eV on a 296 eV line at 30 keV, and
  520 eV on a 1055 eV line at 300 keV (vacuum, before refraction).
- **The production occupation and yield runs use the second-order kernel.**
  At 30 keV they deplete 1s too fast. The 2p rates change less, because 2p
  sees a per-atom variance below 0.1.
- **Not tested.** The thin-screen (eikonal) assumption for each atom is not
  tested; at 30 keV the Fresnel phase $q^2d/2k$ is about 0.8 rad for
  $q=1/u_\mathrm{rms}$ (per axis) across a 0.3 Å core, and 0.4 rad for the
  2D $u_\mathrm{rms}$, which needs multislice through
  the 3D atomic potential. Also not tested: the projected Mott–Bethe
  potential itself and the Einstein model.

(sec-submev-support-matrix)=
## Draft support matrix

This section collects the screens above into one classification per
material, geometry and energy. It is a draft for design review under #585,
not a support envelope. No row is supported: none has an experimental anchor,
an occupation or linewidth model, a scattering-validity bound or a yield.
The proposed claim `cr-submev-support-envelope` has no ledger row yet.

**Current production behaviour.** No sub-MeV case is in either admission
table. The planar table holds one 17 MeV Si(110) case; the axial table holds
two 3.5 MeV Si⟨001⟩ fibre cases. A standard run therefore refuses every case
in this matrix. With `admission="accept-inaccurate"` it runs unadmitted, with
no error budget. A planar case then passes the general gates, which need only
one bound state. That admits the zero-line and inter-string rows below.

**Classes.**

- **none**: no bound→bound pair in the static model. Channeling and
  free→bound capture are not excluded. This rests on bound-state counts,
  which held under every cutoff and k refinement above.
- **inter-string**: every bound pair is a tunnelling splitting of string
  ground orbitals ($R\le0.4$, or a gap below 1 eV). These are not
  channeling-radiation lines. `axial_line_catalog` labels them
  `inter_string`, but still emits them.
- **vacuum only**: a planar or on-string line exists, but the medium has no
  model at that line. The reasons are a photon energy below $3\hbar\omega_p$,
  a refraction parameter $\beta\delta/(1-\beta)$ above 0.3, a refused
  iteration, or an absorption edge within 3 %. The vacuum energy is not a
  prediction of the emitted or escaping line.
- **unconverged**: the pair set or class moves under k or cutoff refinement.
- **candidate**: an on-string or planar line with numerical and medium
  corrections bounded (below). It is still diagnostic-only.

Zone measures below 1 mean the pair exists over part of the Bloch zone only.

| Geometry | 30–100 keV | 200–300 keV | 500 keV | 750 keV | 1 MeV |
| --- | --- | --- | --- | --- | --- |
| Si, diamond (110), (100) planar | none | none | none | none | none |
| Si(111) planar | vacuum only (15–30 eV, zone 0.51–0.66) | vacuum only (33–56 eV) | vacuum only (68–78 eV) | no optics row (100–108 eV, above $3\hbar\omega_p$) | candidate: 133–140 eV vacuum, −9 to −11 eV in medium |
| diamond(111) planar | none | vacuum only (46–72 eV, zone 0.21–0.37) | vacuum only (zone 0.60; 99 eV below floor, 128 eV refused) | lower end refused; upper end −28 eV | candidate: −25 to −46 eV (up to 21 %) |
| Si⟨100⟩ | none | none | none | 214 eV over 0.12 of zone; no optics row | candidate: 342–368 eV, zone 0.94, −21 eV |
| diamond⟨100⟩ | none | none | none | none | none |
| Si⟨111⟩ | none | none | none | upper state above percolation | candidate: on-string 440–453 eV, −17 eV |
| diamond⟨111⟩ | none | none | none | none | none |
| Si⟨110⟩ | inter-string | inter-string | inter-string | on-string or mixed at 31–33 eV gap; no optics row; not k-converged | as 750 keV |
| diamond⟨110⟩ | inter-string | inter-string | inter-string | inter-string | inter-string; delocalized or mixed over 0.08 of zone |
| W⟨001⟩ | candidate: 333–520 eV | candidate: 800–1120 eV | M₄ edge: escape and shift unbounded | candidate: 3036 eV, kinematic root needed | candidate: 4413 eV, kinematic root needed |
| W⟨111⟩ | candidate: 405–621 eV | candidate: 1318 eV (300 keV) | unconverged | unconverged | unconverged |

Rows marked "no optics row" have a line above $3\hbar\omega_p$ that the
optics record does not evaluate. They need that row before they can be
classified.

**W line budget.** The W⟨001⟩ 2p→1s line at 30–300 keV is the only
sub-MeV case with every static term bounded. All values are line-energy
changes in eV:

| Term | 30 keV | 100 keV | 300 keV | Status |
| --- | --- | --- | --- | --- |
| Basis cutoff 2→3 | 0.18–0.24 | 0.18–0.24 | not recorded | numerical |
| Klein–Gordon + exact kinematics | −0.11 | −0.38 | −1.8 | computable correction |
| First-order Dirac (Darwin + spin) | −0.07 to −0.10 | −0.08 to −0.14 | −0.10 to −0.21 | signed correction, 2p sub-levels |
| Refraction (Chantler $n$) | −1.9 | −3.8 | −6.6 | correction; table error not bounded |
| Zone-weighted mean − $k=0$ line ($f$ 0.05–0.25) | −1.7 to −2.4 | −1.1 to −1.5 | −0.13 to −0.20 | computable correction |
| Static dispersion rms width ($f=0.1$) | 1.20 | 0.77 | 0.10 | line spread, not error |
| Scattering FWHM, transverse (MeV-calibrated kernels) | 239 | 157 | 135 | larger than every other term; line unresolved |
| Scattering FWHM, strong-phase TDS + electronic | 140 | 119 | 116 | 0.49–0.70 of the gap; eikonal not tested |
| Strong-phase gap shift beyond mean field | −28 | −20 | −14 | computable correction; eikonal not tested |
| Escape through 100 nm | $T=0.32$ | 0.46 | 0.78 | from tables |
| Photons per electron to 1000 Å ($f=0.1$, before escape) | $6.9\times10^{-6}$ | $7.3\times10^{-6}$ | $6.1\times10^{-6}$ | diagnostic, MeV-calibrated kernels |

Refraction and the zone weighting dominate the line position. The scattering width
({ref}`sec-submev-w001-linewidth`) is 20–100 times every positional term, so
under the current kernels the line is not resolved. The 2p
dispersion dominates its static width at 30–100 keV
({ref}`sec-submev-w001-zone-line`). That dispersion is why
`axial_line_catalog` admits the line only from 300 keV: its gate is a 0.1 eV
bandwidth, which is a numerical choice, not a physical floor. The 2p entry weight peaks at 0.21, 0.15 and
0.07 of the beam (30, 100, 300 keV), near $\sigma=0.1\,\psi_c$.

**Unbounded in every row.** TDS and electronic scattering at these energies,
occupation evolution and linewidth (W⟨001⟩ has diagnostic records with
MeV-calibrated kernels, {ref}`sec-submev-w001-occupation` and
{ref}`sec-submev-w001-linewidth`), coherence, free→bound capture,
longitudinal lattice harmonics, surface and transition-radiation terms, and
separation from coherent bremsstrahlung and PXR. These terms set yield and
observability. A candidate row is a static line position, not a predicted
spectrum.

**Anchor coverage.** None. No measured sub-MeV bound→bound line was found.
The nearest measured anchors are the 1–3 MeV axial lines of Watson and
Koehler.

**Proposed go/no-go for design review.**

- *Go, diagnostic-only design target:* W⟨001⟩ and W⟨111⟩ at 30–300 keV.
  The next slices are, in order:
  1. a dispersion-aware line record, so that the line below 300 keV is not
     refused by the flat-band gate and is not widened into a false line.
     The research record ({ref}`sec-submev-w001-zone-line`) is done for
     W⟨001⟩; production adoption waits on step 3;
  2. the medium line energy $\Delta\varepsilon/[1-\beta(1-\delta)]$;
  3. TDS and occupation evolution for the 2p and 1s bands, run on the
     remote box. The diagnostic record for W⟨001⟩ is done
     ({ref}`sec-submev-w001-occupation`). It uses the MeV-calibrated
     kernels, so its yields are not validated;
  4. a line width. The flat-band record for W⟨001⟩ is done
     ({ref}`sec-submev-w001-linewidth`). Under the MeV-calibrated kernels
     the FWHM is 0.5–1.05 of the gap, so the line is unresolved.

  *Finding for review:* with the current kernels the W⟨001⟩ 2p→1s emission
  below 300 keV is a broad band, not a line. Before any further W design
  work, the go decision should be revisited. The gating item is now a
  sub-MeV check of the TDS kernel beyond small-phase linearization: an
  exact phase-object or multislice reference for one displaced W string.
  The phase-object check is done ({ref}`sec-submev-w001-tds-phase`). It
  narrows the width by up to 1.8×, but FWHM/gap stays at 0.49–0.70: the
  emission is a broad peak, not a resolved line. Only the eikonal-per-atom
  assumption (multislice) remains untested.
- *No-go:* every light-element case below 750 keV. Each such row is none,
  inter-string or vacuum only.
- *No-go until fixed:* W⟨001⟩ at 500 keV, which needs solid-state M₄-edge
  optics, and W⟨111⟩ from 500 keV, which needs k convergence.
- *Deferred:* the light-element candidates at 750 keV–1 MeV. Their medium
  shift reaches 21 %, and the MeV regime nearby is better served by the
  existing anchors.

## Not established

These items are open under #585 and none is settled by this screen:

- full-text literature recovery for sub-MeV channeling-radiation photons, as
  distinct from electron diffraction, EELS and cathodoluminescence; the
  cursory search below found no sub-MeV anchor;
- longitudinal lattice harmonics, or full Bloch or multislice propagation, as
  against plane averaging;
- TDS and electronic scattering validity at these energies, along with
  occupation lengths and linewidths;
- a dielectric theory of emission below $3\hbar\omega_p$, and solid-state
  optical data at absorption edges. The [optics
  record](#sec-submev-optics) bounds escape and the refractive line shift
  only above $3\hbar\omega_p$ and away from edges, from isolated-atom tables;
- validated axial occupation evolution and capture. The
  [axial screen](#axial-screen) records zone-wide availability. The W ⟨001⟩
  occupation record ({ref}`sec-submev-w001-occupation`) uses unvalidated
  sub-MeV kernels, has no capture from outside the supercell, and its
  survival fraction is not grid-converged;
- a production rule for multi-string axes. The
  [classification](#multi-string-classification) shows that the flat-band
  catalog admits inter-string splittings. Since #1219, `axial_line_catalog`
  labels them (`string_class`, `below_optics_floor`) but still emits them,
  and W ⟨111⟩ from 500 keV is not converged at 6 intervals;

[Watson & Koehler (1982)](https://doi.org/10.1103/PhysRevB.25.3079) observed
axial lines at 1–3 MeV. They saw no planar lines above about 1 keV. That
result is consistent with the eV-scale planar candidates above. It does not
show that sub-MeV planar radiation is absent, and it gives no anchor for it.

## Literature: cursory search (2026-11-10)

A short web search looked for photon spectra from electrons below 1 MeV in
crystals. It found no report of a measured transverse bound→bound line below
1 MeV. That is not proof of absence. Items from that search were read only as
an abstract or bibliographic record. Items marked "full text, Zotero" were
added from the project library on 2026-11-15 and read in full. None is a
validation anchor.

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
- **Zone-axis X-rays from 60–300 keV electrons (full text, Zotero).**
  Shentcis et al. measured tunable X-ray peaks from TEM electrons along the
  stacking axis of van der Waals crystals (WSe₂, MnPS₃, CrPS₄, FePS₃) at
  60–300 keV, at roughly 0.7–2.1 keV photon energy
  ([Nat. Photon. 14, 686 (2020)](https://doi.org/10.1038/s41566-020-0689-7)).
  They attribute the peaks to parametric X-ray radiation and coherent
  bremsstrahlung, which share one dispersion relation set by the lattice
  period along the beam. That is a measured sub-MeV crystal photon spectrum,
  but it is not a transverse bound→bound line. It is a background to separate
  from any sub-MeV CR, and the same caveat applies as to the TEM CB work above.
  Shi et al. treat that radiation in QED with transverse recoil and, for
  30 keV electrons through 30 nm Si, predict extra peaks about 30 eV apart
  ([Nat. Commun. 15, 7803 (2024)](https://doi.org/10.1038/s41467-024-52050-w)).
  That paper is theory; it bears on how transverse quantization imprints on
  CB, not on bound→bound emission.
- **Si ⟨111⟩ bound states at 1–2 MeV (full text, Zotero).** Kaplin et al.
  solved a single-row continuum potential with an unscreened $1/r$ form and
  non-overlapping rows. They give hydrogen-like 2D levels
  $E_N\propto\gamma/(N+\tfrac12)^2$
  ([Phys. Lett. A 54, 447 (1975)](https://doi.org/10.1016/0375-9601(75)90806-3)).
  At 1 MeV they find 1s and a barely bound $N=1$ (2s/2p) level, and they
  relate the 2p state to a ring in measured transmitted-electron patterns.
  They ascribe the six-spot patterns to states bound to several rows. Their
  potential is far deeper near the row than the screened, thermally averaged
  one used here, so their level energies are not comparable. Qualitatively
  they agree with the
  [multi-string classification](#multi-string-classification): Si ⟨111⟩ has
  an on-string-like pair from 750 keV, whose upper state lies just below the
  barrier, and an on-string line at 1 MeV. This is an
  electron-transmission observation, not a photon anchor.
- **Diffracted channeling radiation (full text, Zotero).** Yabuki et al.
  ([PRB 63, 174112 (2001)](https://doi.org/10.1103/PhysRevB.63.174112)) and
  Korotchenko et al. (γ = 20,
  [NIM B 269, 2840 (2011)](https://doi.org/10.1016/j.nimb.2011.09.006))
  treat Bragg-diffracted CR from 8–20 MeV electrons. Yabuki et al. place
  observable DCR in the quantum channeling regime, about ≤ 20 MeV planar and
  ≤ 10 MeV axial. Neither paper works below 1 MeV; both are context only.
- **Lowest-energy CR experiments found.** The Fermilab/ASTA design targets a
  line near 1.2 keV from about 4 MeV electrons
  ([arXiv:1407.6053](https://arxiv.org/abs/1407.6053); Blomberg et al.,
  FEL2013 MOPSO07). No measured spectrum was found. Watson and Koehler (above)
  remain the lowest-energy measured axial lines known here, at 1–3 MeV.
