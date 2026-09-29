# PENELOPE shell oscillators

Validation: `penelope-shell-oscillators`. Status: unverified. The oscillator set
is host-side input for a shell GOS model; no cross section or transport mode
consumes it yet.

## Source and equations

PENELOPE-2024 §3.2.1 (NEA/MBDAV/R(2024)1, pp. 117–118) describes a material
per formula unit ($Z$ electrons) with one conduction-band oscillator and one
oscillator per bound shell $k$. The conduction band has $U_{cb}=0$ and cutoff
$Q_{cb}=W_{cb}$. Its strength $f_{cb}$ and resonance $W_{cb}$ "should be
identified with" the effective plasmon electron count and the plasmon energy
from EELS or optical data. When no data are available, $f_{cb}$ counts
electrons with $U<15$ eV, and

$$
W_{cb}=\sqrt{f_{cb}/Z}\;\Omega_p \qquad \text{(Eq. 3.62)}.
$$

Each bound shell uses Sternheimer's form with a Lorentz–Lorenz term,

$$
W_k=\sqrt{(aU_k)^2+\tfrac{2}{3}\tfrac{f_k}{Z}\Omega_p^2}\qquad \text{(Eq. 3.63)},
$$

and one material factor $a$ satisfies

$$
Z\ln I=f_{cb}\ln W_{cb}+\sum_k f_k\ln W_k \qquad \text{(Eqs. 3.61, 3.64)}.
$$

Compounds use Bragg additivity (Eq. 3.65), matching SBETHE's input $I$.
$\Omega_p$ is the all-electron plasma energy (`plasma_energy_eV`, Eq. 3.51).
Shells and $U_k$ come from the checksum-pinned `pdatconf.p14`
([`sbethe-atomic-shell-inputs`](sbethe-atomic-shell-inputs.md)).

## Measured conduction bands

Following the manual's preferred route, `src/pyrite/data/conduction_band.toml`
supplies measured $W_{cb}$ and the chemical-valence $f_{cb}$ per formula unit:

| material | $f_{cb}$ | $W_{cb}$ (eV) | basis | Eq. 3.62 (eV) | $a$ |
| --- | ---: | ---: | --- | ---: | ---: |
| silicon | 4 | 16.7 | ELF maximum, [Yang et al. 2019](https://doi.org/10.1103/PhysRevB.100.245209) | 16.60 | 2.2024 |
| sio2 | 16 | 22.0 | AR-EELS $q=0$ volume plasmon of amorphous SiO₂, [Saito et al. 2025](https://doi.org/10.1093/jmicro/dfae056) | 22.06 | 3.4701 |
| mos2 | 18 | 23.0 | π+σ bulk plasmon, [Moynihan et al. 2020](https://doi.org/10.1111/jmi.12900) | 21.61 | 1.7797 |
| hopg | 4 | 25.0 | in-plane ($E\perp c$) optical ELF π+σ maximum, [Taft & Philipp 1965](https://doi.org/10.1103/PhysRev.138.A197) | 25.03 | 2.6359 |
| hbn | 8 | 26.4 | in-plane ($q\perp c$) low-$q$ EELS σ+π plasmon, [Tarrio & Schnatterly 1989](https://doi.org/10.1103/PhysRevB.40.7852) | 24.70 | 2.5860 |

The ELF maxima were read from the authors' tabulated 0.1 eV grids
(Si: 3.7585 at 16.7 eV). For SiO₂ the owner selected Saito's amorphous
22 eV value, which matches the catalog's amorphous density; see the caveat
below. The MoS₂ value is the reported thick-terrace peak ("around
23 eV"); its separate 8.6 eV π plasmon has no oscillator here. HOPG and
h-BN likewise keep only the π+σ plasmon (their π plasmons near 7 and
8.5 eV have no oscillator). Each measured value lies within 7% of the
free-electron Eq. 3.62 value with the same $f_{cb}$ (Si 0.6%, MoS₂ 6.4%,
SiO₂ −0.25%, HOPG −0.14%, h-BN 6.9%).

Caveats on these inputs:

- **SiO₂ density and value choice.** The REELS ELF of Da et al. peaks at
  23.6 eV for a 2.65 g/cm³ sample; the catalog `sio2` is amorphous at
  2.20 g/cm³. A $\sqrt{\rho}$ rescale of that peak gives about 21.5 eV.
  The owner selected Saito's amorphous-SiO₂ AR-EELS $q=0$ peak, 22 eV. The
  authors note that surface losses lower their $q=0$ peak; it rises to about
  24 eV for $q>0.4$ Å⁻¹. The 21.5–23.6 eV spread measures the input
  uncertainty.
- **Layered-crystal anisotropy and π plasmon.** HOPG and h-BN are
  c-axis-normal in the catalog, so small-angle distant losses of normally
  incident primaries carry $q$ mostly in the basal plane; both entries use
  the in-plane branch. For $q\parallel c$ the π+σ peak falls to about
  18 eV (graphite; Marinopoulos et al., PRL 89, 076402 (2002),
  doi:10.1103/PhysRevLett.89.076402) and 23 eV (h-BN, Tarrio &
  Schnatterly). Scattered electrons and secondaries sample mixed
  orientations. A π-plasmon oscillator ($W_{cb}\approx7$–8.5 eV) with the
  whole-valence $f_{cb}$ drives $a$ above 24 and the K resonances to
  5–10 keV, so it is not a usable single-oscillator choice. Both
  π and π+σ losses lie below the default 50 eV cutoff, so the choice moves
  the soft-loss spectrum and IMFP, not the hard-event split.
- **HOPG value choice.** Taft & Philipp's ELF maximum (25 eV) sits at the
  26 eV edge of their measured reflectance, so it depends on the
  Kramers–Kronig extrapolation and is likely biased low. In-plane EELS on
  single-crystal graphite gives 28 eV at $q=0.25$ Å⁻¹ (Marinopoulos et al.),
  and the π+σ plasmon disperses upward, so $q\to0$ lies below that; 27 eV
  is often quoted but no primary source was verified. The 25–28 eV spread
  measures the input uncertainty. The catalog density (2.2665 g/cm³) is
  ideal-crystal; real HOPG (2.25–2.26 g/cm³) moves Eq. 3.62 by <0.4%.
- **h-BN value and density.** Tarrio & Schnatterly's samples are described
  only as hexagonal BN of varying purity; whether 26.4 eV is a $q\to0$
  extrapolation or a low-$q$ peak was not verified. Its 6.9% excess over
  Eq. 3.62 is the largest in the table. The catalog density (2.279 g/cm³)
  is ideal-crystal; pyrolytic or turbostratic BN (1.9–2.2 g/cm³) gives a
  lower plasmon. c-BN values (about 30 eV) do not apply.
- **Effective electron count.** $f_{cb}$ is the chemical valence, which the
  manual permits (it cites Sternheimer) but does not prescribe. A partial
  f-sum of the Da ELF reaches only about 8–11 electrons by 80 eV. So 16 is
  an upper-bound count, and O 2s contributes only partly to the SiO₂
  plasmon.
- **Peak versus log-mean.** The ELF peak lies below the valence ELF's
  log-mean energy (Si about 18.6 eV). Re-solving $a$ preserves $I$ and
  stopping, but the choice moves the IMFP.
- **Mo configuration.** `pdatconf.p14` lists Mo as N4 (4 electrons) plus O1
  (2), both at 8.317 eV, rather than $4d^55s^1$. Both are consumed together,
  so the valence total is unaffected.

A measured band consumes whole `pdatconf.p14` shells in increasing $U$ until
it reaches $f_{cb}$. A boundary that splits a shell, or a group of shells
with equal $U$, raises. Si uses M1+M2 (next shell 104 eV); SiO₂ uses Si
M1+M2 and O L1–L3 (next 104 eV); MoS₂ uses Mo N4+O1 and S M1–M3 (next
42 eV); HOPG uses C L1+L2 (next 288 eV); h-BN uses B L1+L2 and N L1–L3
(next 192 eV). Unlike the 15 eV default, this includes O 2s (28.5 eV) and S 3s
(20.2 eV) valence electrons. Materials without a measured entry use the
manual default, labelled in `conduction_source`.

## Checks and limits

- Dipole sum: $f_{cb}+\sum_k f_k=Z$.
- Eq. 3.64 closes to relative $10^{-12}$ for fixtures and all five
  measured materials.
- Limiting case from the manual: a single bound shell with $f=Z$ and no
  conduction band gives $W=I$.
- Si's default rule reproduces $f_{cb}=4$, $W_{cb}=16.60$ eV and
  $a=2.208$.
- The fitted $a$ is pinned to $10^{-4}$: 2.2024 (Si), 3.4701 (SiO₂),
  1.7797 (MoS₂), 2.6359 (HOPG), 2.5860 (h-BN). SiO₂ gave 3.2025 with the earlier 23.6 eV input.
- Split-shell and equal-$U$ boundaries, a formula mismatch, a missing
  element, non-finite or non-positive inputs, $W_{cb}\ge I$, any bound
  $W_k\le U_k$, and an $I$ below the zero-binding limit all raise.

The measured $W_{cb}$ strongly affects inelastic mean free paths, while
stopping is nearly insensitive (manual, Fig. 3.11 discussion). The optical
($q\to0$) plasmon energy is used without dispersion. These parameters do not
validate the GOS spectrum, rates or transport; those checks come with the
shell GOS model. `material.f` is not available locally, so its coded default
is unchecked.

Fresh-context re-derivation: [verification record](penelope-shell-oscillators-verification.md).
