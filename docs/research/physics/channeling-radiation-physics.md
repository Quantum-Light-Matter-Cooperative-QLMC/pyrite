# Channeling radiation: physics implementation walkthrough

**Status:** Proposed physics design; not implemented as a production model.

Companion to [relativistic electron transport](relativistic-electron-transport.md).
That document fixes the architecture (backends, data contracts, validation
gates). This one works through the physics itself, step by step, in the order
an implementation would build it: continuum potential → transverse quantum
states → initial occupation → depth evolution → photon emission → axial
extension. It closes with the physics the design document does not mention and
an explicit map of where Geant4 does and does not intersect.

Conventions used throughout:

- $m$ is the electron rest mass, $\gamma = 1 + T/mc^2$ with $T$ the kinetic
  energy, $\beta c$ the speed, $p = \gamma m \beta c$ the momentum, and
  $pv = \gamma m \beta^2 c^2$ the standard channeling energy scale.
- $z$ is beam-longitudinal, $x$ (planar) or $(x, y)$ (axial) transverse.
- Working units are eV and Å: $\hbar^2 / 2 m_e = 3.810\ \mathrm{eV\,Å^2}$,
  $\hbar c = 1973.3\ \mathrm{eV\,Å}$, $m_e c^2 = 511.0\ \mathrm{keV}$.
- The electron charge is $-e$; "potential" $U$ below always means the
  *potential energy of the electron*, so wells sit **on** atomic planes and
  strings, not between them. This one sign flip drives most of the
  electron-vs-positron phenomenology.

## 0. Regime: why quantum, and what the numbers look like

The transverse motion of a channeled particle is governed by a potential well
of depth $U_0$ and width $\sim d_p$ (the interplanar spacing). A WKB count of
bound levels,

$$
n_b \sim \frac{d_p}{\pi \hbar} \sqrt{2 \gamma m U_0},
$$

grows like $\sqrt{\gamma}$. Classical/quasiclassical treatments (Baier–Katkov,
`G4ChannelingFastSimModel`) require $n_b \gg 1$; at hundreds of MeV that
holds, at 1–10 MeV it does not:

| $T$ | $\gamma$ | $pv$ | $2\gamma^2$ | $\psi_c$ Si(110) | $n_b$ Si(110) | forward $\hbar\omega$ for $\Delta E_\perp = 5{-}20$ eV |
| --- | --- | --- | --- | --- | --- | --- |
| 1 MeV | 2.96 | 1.34 MeV | 17.5 | 5.7 mrad | ~2 | 0.09–0.35 keV |
| 3 MeV | 6.87 | 3.44 MeV | 94.4 | 3.6 mrad | ~4 | 0.5–1.9 keV |
| 10 MeV | 20.6 | 10.5 MeV | 846 | 2.0 mrad | ~7 | 4–17 keV |

using Si(110): $d_p = 1.92$ Å, $U_0 \approx 22$ eV at room temperature, and
the critical angle $\psi_c = \sqrt{2 U_0 / pv}$.

Three consequences frame the whole implementation:

1. **A handful of levels.** Spectra are discrete lines from transitions
   between a few bound states — exactly what Watson observed at 1–3 MeV. The
   states must come from an explicit eigensolve; there is no continuum
   approximation to fall back on.
2. **The photon energies land in or near the PyRITE detector band.** The
   Doppler upshift $2\gamma^2$ maps eV-scale transverse spacings to
   0.1–20 keV photons, overlapping PXR/CBS lines. Channeling radiation is a
   physical foreground/background for the coherent spectra, not a separate
   curiosity.
3. **Electrons dechannel fast.** Because the bound states peak on the atomic
   planes, channeled electrons sit in the region of maximum nuclear and
   electronic density. Occupation lengths are of order a micron at these
   energies (Kephart), so the channeling module governs only the first
   $\lesssim$ few µm of path — the rest is ordinary transport (Geant4's job).

## 1. Step 1 — Planar continuum potential

### 1.1 From crystal structure to $U(x)$

Averaging the crystal potential over the two directions in the plane kills
every Fourier component except those with $\mathbf{g}$ normal to the plane.
For plane $(hkl)$ with spacing $d_p$ and $g_n = 2\pi n / d_p$:

$$
U(x) = \sum_{n \neq 0} U_n\, e^{i g_n x}, \qquad
U_n = -e V_{\mathbf{g}_n},
$$

where $V_\mathbf{g}$ is the electrostatic Fourier coefficient of the full
lattice at the reciprocal vector $\mathbf{g}_n$ normal to the plane. In the
standard electron-microscopy convention,

$$
V_\mathbf{g} = \frac{h^2}{2 \pi m_e e\, \Omega}
  \sum_j f^{e}_j(s_g)\, e^{-B_j s_g^2}\, e^{-i \mathbf{g} \cdot \mathbf{r}_j},
\qquad s_g = \frac{g}{4\pi} = \frac{\sin\theta}{\lambda},
$$

with $\Omega$ the unit-cell volume and the prefactor equal to
$47.878\ \mathrm{V\,Å^3}$ when $f^e$ is in Å and $\Omega$ in Å³. The sum over
the basis is exactly the machinery already in
`materials/crystal.py::structure_factor` — same extinction rules, same
Debye–Waller placement — evaluated with *electron* scattering factors instead
of X-ray ones.

### 1.2 Electron scattering factors via Mott–Bethe

The repository already carries source-backed X-ray form factors
(`materials/atomic.py::atomic_form_factor`, Waasmaier–Kirfel $f_0$). The
electron scattering factor follows from Poisson's equation (Mott–Bethe):

$$
f^{e}(s) = \frac{Z - f_x(s)}{8 \pi^2 a_0 s^2},
\qquad a_0 = 0.5292\ \text{Å},
$$

giving $f^e$ in Å. This reuses validated atomic data instead of introducing a
second tabulation (Doyle–Turner would be an independent cross-check, not the
primary source). Caveats to encode as tests: the formula is for neutral atoms
(fine for our catalog); the $s \to 0$ limit is finite for neutral atoms
because $Z - f_x(s) \to O(s^2)$, but it must be evaluated with the analytic
limit, not numerically at $s = 0$; only the real part $f_0$ enters — the
anomalous X-ray terms $f', f''$ describe photon absorption and do **not**
belong in the electrostatic potential.

### 1.3 Thermal averaging and temperature dependence

Each coefficient carries the Debye–Waller factor
$e^{-B_j s_g^2}$ with $B = 8\pi^2 \langle u_x^2 \rangle$ and
$\langle u_x^2 \rangle$ the one-dimensional projected mean-square thermal
displacement. Equivalently: the plane's atomic density profile is a Gaussian
of width $u_\text{th} = \sqrt{\langle u_x^2 \rangle}$ ($\approx 0.077$ Å for
Si at 293 K, $B \approx 0.47$ Å²), and the potential is the static-lattice
potential convolved with it.

The design document adds `crystal_temperature_K` but no model behind it. The
implementation needs one: a Debye-model $B(T)$,

$$
B(T) = \frac{6 h^2}{m_a k_B \Theta_D}
  \left[ \frac{\phi(x_D)}{x_D} + \frac{1}{4} \right],
\qquad x_D = \frac{\Theta_D}{T},
$$

with $\phi$ the Debye integral and $m_a$ the atomic mass, anchored to the
room-temperature $B$ already in the catalog. Well depth and level spacings
shift by several percent between 100 K and 400 K, which is comparable to the
line-position accuracy the anchors demand — so this is load-bearing, not
cosmetic. Note the catalog's placeholder $B = 0.6$ Å² entries (flagged in the
validation ledger) are *not* adequate for channeling profiles; the design
document's per-material "source-backed thermal vibration amplitude"
requirement is exactly right.

### 1.4 Derived densities

Two more profiles fall out of the same construction and are needed later:

- **Nuclear (plane) density** $n(x)$: Gaussians of width $u_\text{th}$ on each
  plane position — drives dechanneling transition rates.
- **Electron density** $\rho_e(x)$: from Poisson,
  $\rho_e \propto \nabla^2 V$, i.e. multiply Fourier coefficients by $g_n^2$
  — drives electronic scattering rates and the local stopping correction.

Sanity anchors for step 1: Si(110) depth $\approx 22$ eV and diamond(110)
$\approx 25{-}30$ eV at room temperature; symmetric profile for
centrosymmetric planes; $U_n \to 0$ for extinct reflections.

## 2. Step 2 — Relativistic transverse eigenproblem

### 2.1 The equation and where $\gamma m$ comes from

Starting from the Dirac equation in a potential $U(\mathbf{r})$ that depends
only on $x$, squaring, and dropping terms of order $(U/E)^2$ and the spin
coupling (both $\lesssim 10^{-5}$ here — see §9), the longitudinal motion
separates and the transverse wavefunction obeys

$$
\left[ -\frac{\hbar^2}{2 \gamma m} \frac{d^2}{dx^2} + U(x) \right] u_k^{(n)}(x)
  = E_\perp^{(n)}(k)\, u_k^{(n)}(x).
$$

This is the Andersen–Bonderup–Pantell master equation: a Schrödinger problem
with relativistic mass $\gamma m$. All relativistic kinematics enters through
that single $\gamma$ (plus the Doppler map of §4). In working units the
kinetic prefactor is $3.810 / \gamma\ \mathrm{eV\,Å^2}$ — nothing else in the
solve knows about relativity.

### 2.2 Bloch structure

$U(x)$ is periodic with period $d_p$, so eigenstates are Bloch waves
$u_k^{(n)}$ with transverse crystal momentum $k \in (-\pi/d_p, \pi/d_p]$ and
band index $n$. Practical structure:

- **Deeply bound bands are flat** (negligible inter-well tunneling): solve at
  a single $k$ and treat levels as isolated-well states.
- **Near-barrier and above-barrier bands disperse** and must be sampled over
  the Brillouin zone; they carry the quasi-channeled population and the
  bound→free edges.

Numerically: plane-wave diagonalization in the $e^{i g_n x}$ basis (64–128
components converges eV-level spacings to well below linewidths) or central
finite differences with Bloch boundary conditions. Both are cheap in 1D; run
both once as a cross-check, keep one.

Required invariants (fast tests): orthonormality, parity for centrosymmetric
planes, variational monotonicity with basis size, zero potential → free
spectrum $\hbar^2(k+g_n)^2/2\gamma m$ and no bound states, and the
$\sqrt{\gamma}$ growth of $n_b$.

### 2.3 Electron-specific structure (why this is not the positron problem)

Because wells sit on the planes, the low states are localized on the nuclei
with strongly **anharmonic** spacing — nothing like the near-harmonic ladder
positrons see between planes. Line identification therefore requires the
actual eigenvalues; there is no useful harmonic approximation to check
against. The harmonic limit is still a *test* (artificially deep smooth
potential → equally spaced levels), just not physics.

## 3. Step 3 — Initial occupation

An electron entering at angle $\psi$ to the plane is a transverse plane wave
$e^{i k_x x}$ with $k_x = p \psi / \hbar$ and transverse energy
$E_\perp^{inc} = (pv/2)\, \psi^2$. Its decomposition over crystal eigenstates
gives the entry populations:

$$
P_n(\psi) = \sum_k \left| \langle u_k^{(n)} | e^{i k_x x} \rangle \right|^2 ,
$$

which for Bloch states collapses onto the $k$ matching $k_x$ modulo
$2\pi/d_p$. Averaging over the transverse entry point within one period is
equivalent to the Bloch normalization — no separate impact-parameter loop is
needed for an ideal surface. On top of this:

- **Beam divergence:** convolve $P_n(\psi)$ with the beam angular
  distribution (Gaussian $\sigma_\psi$; compare to $\psi_c$ — at 3 MeV,
  $\psi_c \approx 3.6$ mrad, so a 1 mrad beam populates selectively, a
  10 mrad beam mostly feeds the continuum).
- **Surface refraction:** crossing the surface step of the mean inner
  potential $\bar{U} \sim 10{-}15$ eV changes $E_\perp$ by an amount
  comparable to level spacings only when $\psi \lesssim \sqrt{2\bar{U}/pv}$;
  include it as the constant $n=0$ term it is (shift of the energy zero), and
  document it — it slightly reweights populations near $\psi = 0$.

Unitarity: $\sum_n P_n + P_\text{continuum} = 1$ at entry, exactly.

## 4. Step 4 — Photon kinematics

Energy–momentum conservation for emission $i \to f$ with a photon at lab
polar angle $\theta$ gives the resonance condition

$$
\hbar\omega \,(1 - \beta \cos\theta) = \Delta E_\perp
  \left[ 1 + O\!\left( \frac{\hbar\omega}{E} \right) \right],
\qquad \Delta E_\perp = E_\perp^{(i)} - E_\perp^{(f)},
$$

so for small angles

$$
\hbar\omega(\theta) \simeq \frac{2 \gamma^2\, \Delta E_\perp}{1 + \gamma^2 \theta^2}.
$$

The recoil correction is bounded by $\hbar\omega / E \lesssim 2.5 \times
10^{-3}$ at 10 MeV — include it as the first-order multiplicative factor and
stop there.

Two consistency points with existing code:

- The Doppler denominator $1 - \beta\cos\theta \equiv 1 - \mathbf{v} \cdot
  \hat{\mathbf{n}} / c$ is the *same object* as in the ledgered
  `line-energy-dispersion` row, which currently carries an unresolved
  numerator-sign discrepancy. The channeling implementation must adopt one
  convention for $\hat{\mathbf{n}}$, the tilt sign (`docs/physics/geometry/tilt-convention.md`),
  and the frequency sign, and add a cross-check test that the two modules
  agree on the denominator for the same geometry.
- $\beta$ here is the *longitudinal* velocity. It differs from the total
  $\beta$ by $O(E_\perp / pv) \lesssim 10^{-5}$ — ignorable, but say so in
  the derivation docstring.

## 5. Step 5 — Depth evolution

### 5.1 Master equation

Populations evolve along depth $z$ by incoherent scattering:

$$
\frac{dP_n}{dz} = \sum_m \left[ W_{m \to n} P_m - W_{n \to m} P_n \right]
  - W_{n \to \text{free}}\, P_n
  + \sum_{m > n} \frac{A_{m \to n}}{\beta c}\, P_m - \sum_{m < n} \frac{A_{n \to m}}{\beta c}\, P_n .
$$

The last two (radiative cascade) terms are numerically tiny compared with
collisional rates but cost nothing and matter for line-ratio anchors.

### 5.2 Transition rates

Golden-rule rates from the *fluctuating* part of the lattice potential
(thermal displacements) plus electron-gas excitation. The practical,
defensible form is the projected local-density expression:

$$
W_{n \to m} = \int dq_x\, \left[
  n_\text{at} \frac{d\sigma_\text{TDS}}{dq_x}
  + \rho_e \frac{d\sigma_e}{dq_x} \right]_{\text{weighted by } |u|^2 \text{ overlaps}}
  \left| \langle u^{(m)} | e^{i q_x x} | u^{(n)} \rangle \right|^2 ,
$$

i.e. amorphous single-atom differential cross sections, weighted by where the
state actually samples nuclei ($|u_n(x)|^2$ against the thermal plane profile
$n(x)$ from §1.4) and electrons ($\rho_e(x)$). Key qualitative behavior this
must reproduce: states localized on planes (low states, for electrons) have
the *shortest* lifetimes; near-barrier states live longer; total dechanneling
length at a given energy scales roughly like $\gamma$.

Because absolute rates are the least first-principles piece of the chain,
anchor them: the Kephart 17 MeV (and 54 MeV) silicon occupation lengths are a
direct measurement of $W$ totals, and the natural place for a single global
calibration factor if one proves necessary — declared in the profile
metadata, not buried.

### 5.3 What is deliberately diagonal

A plane wave populates Bloch states *coherently*; the density matrix has
off-diagonal elements that oscillate with beat length
$\lambda_{nm} = 2\pi \hbar \beta c / \Delta E_{nm}$ (pendellösung-like, tens
to hundreds of nm here) and produce depth-oscillating yield in thin crystals.
The module propagates only diagonal populations. Justification: collisional
dephasing at electron dechanneling rates kills coherences within a fraction
of a micron. The regression test making this honest: for crystals thinner
than $\sim \lambda_{nm}$, flag results as outside validity rather than
returning silently wrong smooth yields. The design document does not mention
this limit; it should live in the profile validity metadata.

### 5.4 Conservation

At every depth: bound + quasi-free + handed-off + radiatively-shifted
populations sum to 1 within solver tolerance. This is the channeling
analogue of the design document's "conservation across bound, continuum,
stopped, and exited populations" test.

## 6. Step 6 — Radiation emission

### 6.1 Dipole rates

Dipole approximation validity: in the average rest frame the photon has
$\hbar\omega' = \gamma \Delta E_\perp \sim 10^2$ eV, wavelength
$\sim 10^2$ Å, versus a state extent $\lesssim 1$ Å — excellent, and worth a
one-line docstring justification. The rest-frame spontaneous rate for
$i \to f$ is the textbook

$$
A'_{if} = \frac{e^2\, \omega'^3\, |x_{if}|^2}{3 \pi \epsilon_0 \hbar c^3},
\qquad x_{if} = \langle u^{(f)} | x | u^{(i)} \rangle ,
$$

with $\omega' = \gamma \Delta E_\perp / \hbar$. Parity gives the selection
rule (odd $\Delta n$ between opposite-parity bands for centrosymmetric
planes); it must *emerge* from the computed $x_{if}$, not be imposed. Lab
rate per unit path: $dN_{if}/dz = A'_{if} / (\gamma \beta c)$ — one factor of
$\gamma$ for time dilation, and this quantity multiplies $P_i(z)$ under the
depth integral.

### 6.2 Angular distribution and polarization

Build the lab distribution by boosting the rest-frame dipole pattern rather
than transcribing a final formula: rest-frame
$dP'/d\Omega' = (3/8\pi) \sin^2\Theta'$ about the dipole axis $\hat{x}$,
then the aberration map

$$
\cos\theta' = \frac{\cos\theta - \beta}{1 - \beta\cos\theta},
\qquad
\frac{d\Omega'}{d\Omega} = \frac{1}{\gamma^2 (1 - \beta\cos\theta)^2},
$$

with $\hbar\omega(\theta)$ from §4 tying angle to energy per line. The result
is the familiar $1/\gamma$ forward cone with an azimuthal asymmetry about the
plane normal. Planar CR is linearly polarized along the projection of
$\hat{x}$ — propagate this as a per-component polarization fraction so it can
feed the same detector machinery as PXR polarization if/when needed. Cross
checks: integrating the boosted pattern over $\Omega$ returns $A'/\gamma$
exactly; the pattern reduces to isotropic-dipole at $\gamma \to 1$.

### 6.3 Line shapes

Physical widths, convolved in this order:

1. **Occupation (dominant for electrons):** Lorentzian with
   $\Gamma = \hbar \beta c\, (L_i^{-1} + L_f^{-1})$ from the collisional
   lifetimes of both states; $L \sim 1$ µm gives $\Gamma \sim 0.2$ eV on
   eV-to-tens-of-eV transitions — percent-level lifetime widths.
2. **Doppler-geometric:** the $\theta$-dependence of $\hbar\omega$ across the
   detector solid angle and beam divergence (reuse the existing
   detector-solid-angle machinery; this usually dominates the *observed*
   width).
3. **Beam energy spread:** $\delta\omega/\omega = 2\,\delta\gamma/\gamma$ —
   note the factor 2 from $\gamma^2$.
4. **Band dispersion** for transitions involving dispersive bands: width from
   the $k$-average of $\Delta E_\perp(k)$.
5. Natural radiative width: negligible; drop after a one-time estimate.

### 6.4 Yield assembly

$$
\frac{dN}{d(\hbar\omega)\, d\Omega}
  = \sum_{i \to f} \int_0^{t} dz\; P_i(z)\,
    \frac{dN_{if}}{dz\, d\Omega}\,
    \mathcal{L}_{if}(\hbar\omega - \hbar\omega_{if}(\theta))\,
    e^{-\mu(\omega)\, \ell_\text{exit}(z, \theta)} ,
$$

normalized to photons/eV/sr/electron at sample exit — the
`radiation_components["channeling"]` contract. Photon attenuation reuses the
existing Beer–Lambert path machinery; per the design document, photons
emitted from a Geant4-transported region must not be attenuated twice.

## 7. Axial channeling (experimental tier)

The same six steps with dimension bumped to 2:

- **Potential:** $U(x, y)$ from summing string potentials (the same
  $V_\mathbf{g}$ construction keeping all $\mathbf{g} \perp$ axis), thermally
  smeared in 2D. Axial wells for electrons are several times deeper than
  planar (order $10^2$ eV) → more states, harder spectra, photon lines a few
  times higher in energy at the same beam energy.
- **Eigenproblem:** 2D plane-wave basis over the transverse reciprocal mesh,
  sparse Lanczos/shift-invert for the lowest ~50 states. For an isolated
  string the potential is nearly cylindrical, so states classify as
  $(n, \ell)$ — 2D-atom-like 1s, 2p, 3d… labels with dipole selection rule
  $\Delta\ell = \pm 1$. The 2D string lattice breaks the symmetry and mixes
  $\ell$; report the cylindrical labels as *approximate* metadata, let the
  matrix elements decide the actual rates.
- **Polarization:** $\Delta\ell = \pm1$ transitions radiate circular
  polarization about the axis; unpolarized beams average to zero net helicity
  but the azimuthal pattern differs from planar — store polarization per
  component rather than assuming linear.
- **Populations/rates/emission:** identical machinery with 2D overlaps.
  Doughnut scattering (azimuthal randomization of quasi-free electrons around
  the axis) is automatically represented by the dense above-barrier spectrum
  only if the continuum sampling is fine enough — this is the main numerical
  risk, and a good reason the axial tier stays `experimental` with
  qualitative peak-order checks only (matching the design document).

Anchor: Watson's 1–3 MeV silicon axial lines (peak energies and ordering).

## 8. Where Geant4 intersects — explicit ownership map

| Physics | Owner | Notes |
| --- | --- | --- |
| Amorphous/misaligned e⁻ transport, 0.1–10 MeV | Geant4 (`G4EmStandardPhysics_option4`) | includes MSC, ionization, bremsstrahlung, secondaries |
| Transverse eigenstates, populations, CR emission | native | this document |
| Incoherent scattering *of channeled electrons* | native (via $W_{nm}$) | Geant4 MSC must be **off** for the channeled segment — the master equation *is* the multiple scattering model there; running both double-counts |
| Bremsstrahlung of channeled electrons | native suppresses, Geant4 excluded on segment | close-collision bremsstrahlung during the ~µm channeled path is small; folded into the validity budget rather than modeled |
| Energy loss on channeled segment | native, as continuous loss | v1: amorphous stopping power over the µm-scale segment (error small because $\Delta T \ll T$); refinement: scale by $\rho_e(x)$ overlap per state |
| δ-ray production on channeled segment | neither (folded into stopping) | explicit limitation; acceptable because channeled path ≪ total path |
| Dechanneled / never-channeled electrons | handed to Geant4 mid-volume | phase-space handoff $(\mathbf{r}, \hat{\mathbf{v}}, T)$ at the depth where the population leaves the bound set |
| Bremsstrahlung + atomic relaxation components | Geant4 | scored into their own `radiation_components` |
| $\gamma(z)$ drift feeding back into line energies | native, optional | negligible over µm occupation lengths; only matters if thick-crystal quasi-channeled re-feeding is ever modeled |
| ≥200 MeV channeling | Geant4 `G4ChannelingFastSimModel`/`G4BaierKatkov` | validation oracle only, never in the 1–10 MeV product path (design doc rule) |

The handoff is one-directional in v1: once dechanneled, an electron does not
re-enter the bound-state module (rechanneling of near-aligned quasi-free
electrons is real but second-order at these dechanneling lengths; record it
as a known omission in profile metadata).

## 9. Physics the design document does not mention

Inventory of effects checked for this walkthrough, with include/exclude
decisions to encode in metadata:

1. **Transition radiation at the entry/exit surfaces.** Energy per boundary
   $\sim \alpha \gamma \hbar\omega_p / 3$ with $\hbar\omega_p \approx 31$ eV
   for Si: ~1.5 eV at 10 MeV, spectrum extending to
   $\sim \gamma \hbar\omega_p$ (0.1–0.6 keV). That is $10^{-3}$–$10^{-2}$
   photons/electron — the *same order as the CR lines* and sitting exactly on
   top of the 1–3 MeV planar band. Decision needed: add as a fifth
   `radiation_components` entry, or compute a bound and document exclusion.
   Silently omitting it risks a real disagreement with soft-X-ray anchors.
2. **Free–free coherent radiation double counting.** Radiative transitions
   between above-barrier (continuum) transverse states *are* coherent
   bremsstrahlung — already owned by the PXR/CBS amplitude machinery. The
   channeling module must therefore emit only bound→bound lines and
   (decision) bound→free edges, and must **not** integrate free→free dipole
   matrix elements, or CBS is counted twice. This boundary deserves its own
   regression test (channeling module with zero bound states emits exactly
   nothing).
3. **Coherent-population (pendellösung) oscillations** in thin crystals —
   §5.3; diagonal-only propagation is fine above ~µm thickness, flagged
   invalid below the beat length.
4. **Temperature model behind `crystal_temperature_K`** — §1.3; Debye $B(T)$
   anchored to catalog values; placeholder $B = 0.6$ Å² materials are
   ineligible for validated channeling profiles.
5. **Beam energy spread** doubles into line width via $\gamma^2$ — §6.3.
6. **Surface refraction** by the mean inner potential — §3; small reweighting
   near exact alignment.
7. **Doppler-denominator convention coupling** to the open
   `line-energy-dispersion` sign discrepancy — §4; add the cross-module
   consistency test *before* anchoring line positions, or the same ambiguity
   infects two subsystems.
8. **Planar validity requires axis avoidance.** A "planar" run aligned near a
   low-index axis is silently axial; validate that the beam direction keeps
   all major axes outside a few times $\psi_c^{axial}$, else refuse/warn.
9. **Spin and quantum-recoil corrections**: spin-flip rates suppressed by
   $(\hbar\omega/E)^2 \lesssim 10^{-5}$; recoil handled at first order —
   both documented as dropped with bounds, satisfying the derivation-docstring
   assumption rule.
10. **PXR/CBS from *channeled* electrons.** Channeled states sample the unit
    cell non-uniformly, which in principle modulates their PXR emission.
    Out of scope v1 (channeled fraction of path is µm-scale); recorded so the
    coherent and channeling modules don't silently claim inconsistent
    electron ensembles.

## 10. Proposed ledger rows

One row per load-bearing equation, `file::symbol` anchors to be filled at
implementation (module split suggestion: `channeling/potential.py`,
`channeling/eigensolver.py`, `channeling/populations.py`,
`channeling/radiation.py`):

| id | claim | source | anchor idea |
| --- | --- | --- | --- |
| `cr-mott-bethe-factor` | $f^e(s) = (Z - f_x)/(8\pi^2 a_0 s^2)$, neutral-atom, analytic $s\to0$ | Mott–Bethe; ICRU/IT-C conventions | cross-check vs Doyle–Turner tabulation |
| `cr-planar-continuum-potential` | plane-projected $U(x)$ from $V_\mathbf{g}$ with DW smearing | ABP 1983 | Si(110) depth ≈ 22 eV @ 293 K |
| `cr-debye-b-of-t` | Debye-model $B(T)$ anchored to catalog RT values | standard Debye–Waller theory | published Si $B(T)$ points |
| `cr-transverse-eigenproblem` | $[-\hbar^2/2\gamma m\, \partial_x^2 + U]u = E_\perp u$, Bloch BCs | ABP 1983; Dirac reduction | free limit; parity; $\sqrt{\gamma}$ level count |
| `cr-initial-occupation` | plane-wave → Bloch decomposition, divergence convolution | ABP 1983 | unitarity; selective vs flooded population regimes |
| `cr-population-master-equation` | diagonal transport with collisional + cascade terms | Kephart 1989 methodology | occupation lengths @ 17 MeV |
| `cr-dechanneling-rates` | overlap-weighted TDS + electronic golden-rule rates | ABP 1983; Uggerhøj 2005 | state-ordering of lifetimes; global calibration declared |
| `cr-doppler-line-energy` | $\hbar\omega = \Delta E_\perp / (1-\beta\cos\theta)$ + recoil bound | kinematics | consistency test vs `line-energy-dispersion` denominator |
| `cr-dipole-rate` | $A' = e^2\omega'^3|x_{if}|^2 / 3\pi\epsilon_0\hbar c^3$, $\omega' = \gamma\Delta E_\perp/\hbar$ | standard QED dipole | emergent parity selection rule |
| `cr-boosted-dipole-angular` | aberration-mapped $\sin^2\Theta'$ pattern + polarization | relativistic aberration | solid-angle integral returns $A'/\gamma$; $\gamma\to1$ limit |
| `cr-line-shape` | occupation Lorentzian ⊗ geometric ⊗ $2\delta\gamma/\gamma$ | §6.3 derivation | width ordering vs Watson linewidths |
| `cr-yield-assembly` | depth-integrated exit yield, photons/eV/sr/e⁻ | §6.4 | Watson 1–3 MeV Si peak energies |
| `cr-axial-2d-eigenproblem` | 2D sparse eigensolve, approximate $(n,\ell)$ labels | ABP 1983; Watson axial data | peak ordering only (experimental tier) |
| `cr-transition-radiation-bound` | TR yield bound at boundaries vs CR yield | Ginzburg–Frank | decision record: include or bounded-exclude |

## 11. End-to-end pipeline

```text
per (material, plane/axis, T):                      [cached profile]
  1. V_g from structure_factor machinery + Mott-Bethe f^e + B(T)
  2. U(x) [or U(x,y)], n(x), rho_e(x)
  3. eigensolve at gamma grid -> {E_perp, u, x_if, W_nm, A_if}

per electron ensemble (beam E, divergence, tilt):
  4. initial populations P_n(0)
  5. integrate master equation over z -> P_n(z), dechanneling flux
  6. dechanneled phase space -> Geant4 handoff (MSC/brem off before handoff)
  7. assemble line spectrum -> radiation_components["channeling"]
  8. Geant4 segments -> "bremsstrahlung", "atomic_relaxation"
  9. PXR/CBS amplitude machinery -> "pxr_cbs"   (free-free excluded from 7)
```

Steps 1–3 are per-profile and cacheable (the $\gamma$ grid is the only beam
dependence); steps 4–7 are cheap per configuration. Nothing in the channeling
chain needs Monte Carlo sampling except through the handoff — the quantum
module is deterministic given the beam distribution, which keeps it testable
to solver tolerance rather than statistical tolerance.
