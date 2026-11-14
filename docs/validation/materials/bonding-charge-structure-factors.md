# Bonding-charge structure factors (Si, diamond)

Fresh-context verification of
[`bonding-charge-structure-factors`](../ledger-crystallography-atomic-data.md#bonding-charge-structure-factors)
(issue #1177). Sections 1–3 were written from the ledger row, the module
docstring of `materials/_structure_factors.py` and the sources only, before
any implementation body was read. Section 4 compares the implementation.

## 1. Sources and intended quantity

- Lu, Zunger & Deutsch (LZD), PRB **47**, 9385 (1993). Table III: Si dynamic
  per-atom $F_{\rm expt}$ (Cummings & Hart Pendellösung set, corrected for
  anomalous dispersion and nuclear scattering; $F(222)$ from Alkire, Yelon &
  Schneider), $\sigma$ in me/atom; dynamic theory values use
  $B = 0.4632$ Å$^2$, which LZD take from Spackman (their Ref. 30), not from
  their own Table I model fit ($B_{\rm val}=0.4585$ Å$^2$). Table V: diamond
  TTKS column (Takama et al., $F(222)$ from Weiss & Middleton), with
  $B = 0.1379$ Å$^2$ from the LZD Table I fit to that set.
- Klein et al., PRB **31**, 68 (1985), Sec. VI B: many-beam planar channeling
  energies of diamond (111) with the Göttlicher–Wölfel bonding correction
  (model 111A).

Quantity: the unit-cell structure factor (electrons/cell) used in X-ray
$\chi_{\mathbf g}$, and the Mott–Bethe electron potential Fourier coefficient
$U_{\mathbf g}$ (and planar/axial projections), corrected for valence bonding
charge that the independent-atom model (IAM) misses.

## 2. Independent derivation

### 2.1 Geometric sum of the diamond structure

CIF frame: atoms at $\mathbf 0$ and $(\tfrac14,\tfrac14,\tfrac14)$ plus the fcc
translations, $N=8$. With $\mathbf G\cdot\mathbf r = 2\pi(hx+ky+lz)$ and
unmixed $hkl$,

$$
P(\mathbf G)=\sum_j e^{i\mathbf G\cdot\mathbf R_j}
=4\left[1+e^{i\pi(h+k+l)/2}\right].
$$

Hence $\lvert P\rvert = 4\sqrt2$ for $h+k+l$ odd [(111), (311), (331), (333),
(511)], $8$ for $h+k+l\equiv0 \pmod 4$ [(220), (400), (422), (440)], and $0$
for $h+k+l\equiv2 \pmod 4$ [(222)].

### 2.2 Per-atom convention

LZD tabulate dynamic $F$ in e/atom such that the IAM limit is
$F = f_0(s)e^{-Bs^2}$ (their "dynamic, atoms" column: Si (111)
$10.455$, (222) $0.000$). The cell factor of an allowed reflection is
therefore $F_{\rm cell} = F\,P(\mathbf G)$; check: Si
$4\sqrt2\times10.6025 = 59.98$ e/cell, the familiar $\approx 60$.

For (222) $P=0$, so the per-atom value is defined as
$\lvert F_{\rm cell}\rvert/N$: Alkire's $1.456$ e/cell $/8 = 0.182$ e/atom,
exactly the Table III entry. So $\lvert F_{\rm cell}(222)\rvert = 8F$.

### 2.3 Phase of the IAM-extinct reflection

The inversion centres of the diamond structure are the bond midpoints, e.g.
$\mathbf o = (\tfrac18,\tfrac18,\tfrac18)$. The 16 bond midpoints per cell
are $\pm(\tfrac18,\tfrac18,\tfrac18)$-type points
$(\tfrac18,\tfrac18,\tfrac18)$, $(\tfrac18,-\tfrac18,-\tfrac18)$,
$(-\tfrac18,\tfrac18,-\tfrac18)$, $(-\tfrac18,-\tfrac18,\tfrac18)$ plus fcc
translations. For $\mathbf G=(222)$ every one has phase
$e^{2\pi i\cdot 6/8}=e^{2\pi i(-2)/8}=-i$, and the fcc translations add
$e^{2\pi i\cdot 2}=1$. A point charge $q>0$ (electrons) on each midpoint thus
gives

$$
F_{\rm cell}(222)=16q\,e^{i\mathbf G\cdot\mathbf o},\qquad
e^{i\mathbf G\cdot\mathbf o}=e^{3\pi i/2}=-i .
$$

So $F_{\rm cell}(222) = N F e^{i\mathbf G\cdot\mathbf o}$ with $F>0$ is
exactly "electron excess at the bond centres". This is the physical sign: all
ab initio and multipole treatments of Si and diamond (LZD's own
$F_{\rm calc}(222)=+0.161$ Si, $+0.108$ C, in the centrosymmetric frame) give
positive $F(222)$ from the covalent bond charge; Alkire and Weiss–Middleton
measure only $\lvert F\rvert$, and the sign is taken from theory. (The
$(\tfrac58,\tfrac58,\tfrac58)$-type antibonding sites carry the same (222)
phase, so (222) alone cannot separate them; that does not change the sign.)

For every allowed $\mathbf G$, centrosymmetry about $\mathbf o$ makes
$F_{\rm cell}e^{-i\mathbf G\cdot\mathbf o}$ real, and a per-atom $F$ close to
$f_0e^{-Bs^2}$ keeps the IAM phase: $F_{\rm meas,cell} = F P(\mathbf G)$. All
signed permutations of $hkl$ are equivalent in $m\bar3m$ (Laue class), so each
tabulated entry stands for its family; (333) and (511) share $\lvert\mathbf
G\rvert$ but are distinct families, and both are tabulated.

Friedel: $P(-\mathbf G)=P(\mathbf G)^*$ and
$e^{-i\mathbf G\cdot\mathbf o}=(e^{i\mathbf G\cdot\mathbf o})^*$, with $F$
real, so $\Delta F(-\mathbf G)=\Delta F(\mathbf G)^*$.

### 2.4 Thermal treatment

The source values are dynamic: $F_{\rm meas,cell} = F_{\rm static}\,
e^{-B_{\rm src}s^2}$ in the single-$B$ rigid-atom model (LZD Eq. (1)), with
$s=\sin\theta/\lambda = \lvert\mathbf G\rvert/4\pi$ ($\lvert\mathbf
G\rvert=2\pi/d$, $s = 1/2d$). The static correction relative to the IAM is

$$
\Delta F(\mathbf G)=F_{\rm meas,cell}(\mathbf G)\,e^{+B_{\rm src}s^2}
-f_0(s)\,P(\mathbf G),
$$

and the corrected dynamic sum at the caller's $B$ is
$[f_0P+\Delta F]e^{-Bs^2}$. At $B=B_{\rm src}$ this reproduces the measured
value exactly; for $B\neq B_{\rm src}$ it rigidly re-smears the bond charge
with the atomic $B$ (the rigid-atom approximation LZD themselves use). For Si
the catalog/source difference ($0.46$ vs $0.4632$ Å$^2$) changes
$e^{-Bs^2}$ at (111) ($s=0.160$ Å$^{-1}$) by $8\times10^{-5}$; for diamond
($0.21$ vs $0.1379$) by $0.7\%$ at (111) and $2.9\%$ at (440), applied to
the whole $F$, not just $\Delta F$. Anomalous $f'$, $f''$ stay in the atomic
term because the measured values are dispersion-corrected.

### 2.5 Electron potential

Mott–Bethe per cell:
$U_{\mathbf g}\propto\sum_j(Z_j-f_j(s))e^{i\mathbf g\cdot\mathbf R_j}e^{-W}/g^2$.
More electrons lower the potential seen by the electron, so the bonding term
enters as

$$
U_{\mathbf g}\propto\frac{\left[(Z-f_0)P(\mathbf g)-\Delta F(\mathbf g)\right]
e^{-Bs^2}}{g^2},
$$

i.e. $f_0\to f_0+\Delta F$. A module using the opposite Fourier convention
$\sum_{\mathbf g}U_{\mathbf g}e^{-i\mathbf g\cdot\mathbf r}$ must use
$\Delta F(-\mathbf g)=\Delta F(\mathbf g)^*$ for its coefficient. For the
extinct (222) with phase $-i$, the conjugate is $+i$; using $\Delta F(\mathbf
g)$ instead would put the bond charge at the antisymmetric image, i.e. flip
its sign in the real-space profile.

### 2.6 Limiting cases

No record: $\Delta F\equiv0$. $F=f_0e^{-B_{\rm src}s^2}$ for every entry:
$\Delta F = 0$ for allowed reflections; for (222) $F=0$ (the IAM value)
gives $\Delta F=0$. Both recover the IAM exactly.

## 3. Cheap filters

| Filter | Result |
|---|---|
| Units | $F$ e/atom, $\sigma$ me/atom $\to$ e/atom ($/1000$), $\Delta F$ e/cell, $B$ Å$^2$, $s$ Å$^{-1}$: pass |
| Limits | §2.6: pass |
| Signs/conventions | positive (222) at bond midpoints (§2.3), $-\Delta F$ in $U$ (§2.5), Friedel conjugate in $e^{-i\mathbf g\cdot\mathbf r}$ modules: pass, subject to §4 |

## 4. Comparison with the implementation

Read after §§1–3: `materials/_structure_factors.py::bonding_corrections`,
`materials/crystal.py::{_bonding_table,bonding_charge_correction,structure_factor,U_g,reflection_coupling_tables}`,
`materials/channeling/potential.py::{planar_profile,_add_bonding_charge}`,
`materials/channeling/axial.py::axial_profile`, the two catalog records and
`tests/materials/test_bonding_charge_structure_factors.py`.

### 4.1 Transcription

Every catalog $F$ and $\sigma$ equals LZD Table III $F_{\rm expt}$ (18 Si
reflections, $\sigma$ me/atom $/1000$) and Table V TTKS $F_{\rm expt}$ (10
diamond reflections); the adjacent $F_{\rm calc}$ columns were not taken.
$B_{\rm src}$: Si $0.4632$ Å$^2$, diamond $0.1379$ Å$^2$, as in the sources.

### 4.2 Static correction

The code computes, per family member,

```text
undo = math.exp(record.B_ang2 * s * s)
P = complex(np.exp(2j * math.pi * positions @ h).sum())
if abs(P) > _EXTINCT_ATOL:
    out[hkl] = (reflection.F * undo - float(f0_of(element, g))) * P
else:
    phase = complex(np.exp(2j * math.pi * float(h @ origin)))
    out[hkl] = len(basis) * reflection.F * undo * phase
```

with $s=\lvert\mathbf G\rvert/4\pi$ and `f0_of` the same Waasmaier–Kirfel
`cromer_mann_f0` used by every IAM sum (`atomic_form_factor` adds only
$f'+if''$ to it). This is §2.4 term by term. The catalog CIF frame is origin
choice 1 (atoms at $\mathbf 0$, $(\tfrac14,\tfrac14,\tfrac14)$ + fcc), so
`origin = [0.125, 0.125, 0.125]` is a bond midpoint, as §2.3 requires; the
test builds the 16 midpoints from the basis itself, so a future CIF origin
change would be caught.

Numerical check (independent basis, independent bond-midpoint sum, all 48
signed permutations per family): reconstructed
$F_{\rm cell}(\mathbf G)$ at $B=B_{\rm src}$ minus the LZD value
$F\,P(\mathbf G)$ or $8F e^{i\varphi_{\rm bond}}$ is at most
$1.8\times10^{-14}$ e/cell for both crystals;
$\Delta F(-\mathbf G)=\Delta F(\mathbf G)^*$ to $10^{-12}$. Si
$\lvert F_{\rm cell}(111)\rvert = 59.98$, $8F(222) = 1.456$ (Alkire). Static
(222) $\Delta F$ is $-1.526i$ (Si), $-1.190i$ (C): phase $-i$, the bond-centre
phase. Si $\lvert\chi_{222}\rvert$ at 10 keV, $B=B_{\rm src}$, equals
$r_e\lambda^2\,8(0.182)/(\pi V)=1.2533\times10^{-7}$.

### 4.3 Consumers

| Consumer | Code | §2 | Result |
|---|---|---|---|
| `structure_factor` / `chi_g` | `S + dF * dwf` | $S\to S+\Delta F e^{-Bs^2}$ | match |
| `U_g` | `acc - dF / g**2 * dwf` | $(Z-f_0)P-\Delta F$ | match |
| `reflection_coupling_tables` | `c + delta`, `a - delta`, same `dwf` | as above | match (test: equals per-reflection functions) |
| `planar_profile` | phases $e^{-i n\mathbf g\cdot\mathbf R}$; `_add_bonding_charge` uses `bonding_charge_correction(crystal, -m*hkl_c)` | $\Delta F(-\mathbf G)$ | match |
| `axial_profile` | phases $e^{-i\mathbf G\cdot\mathbf R}$; `bonding_charge_correction(crystal, -row)` with full-integer zone rows | $\Delta F(-\mathbf G)$ | match |

Planar indexing: `_add_bonding_charge` loops over `n[1:]` of the full harmonic
array $n=0..n_{\max}$, so harmonic $m$ is $\mathbf G=m\,\mathbf h_c$ for
coprime $\mathbf h_c$; the extinct-harmonic `step`/`keep` subsampling runs
after the correction, on the corrected coefficients. For diamond (111) the
$m=2$ harmonic is IAM-zero and becomes $+0.1232i$ eV, so `step` stays 1. Both
planar and axial modules smear $\Delta F$ with the site-mean Debye–Waller
factor (single-element cells, so exact at the rigid-atom level).

Real-space sign check (diamond (111), channeling thermal $B=0.144$ Å$^2$):
the corrected-minus-IAM electron density peaks at $x=1.802$ Å, the midpoint
of the close plane pair ($1.544$, $2.059$ Å); the $m=2$ term alone is
$+0.051$ e/Å$^3$ there. The potential well becomes shallower
($28.36\to26.53$ eV peak-to-peak), the sense Klein describes.

### 4.4 Planar anchor (Klein et al. 1985, Sec. VI B)

Klein's text confirms the anchor values: (111A) 30.5 MeV $3\to0$ $73.6$ keV
(obs. $72\pm1$), $4\to1$ $92$ (obs. $89\pm1.5$), $2\to1$ $36.3$ (obs.
$34.5\pm1$, "5% high"); 54.5 MeV $3\to0$ $162$ (obs. $\approx154\pm5$,
"very difficult to extract"); inputs $\Delta f_x = 0.26$ (111), $0.15$ (222)
from Göttlicher–Wölfel powder data. PyRITE, recomputed here:

| Line | Corrected | IAM | Klein (111A) | Obs. |
|---|---|---|---|---|
| 30.5 MeV $3\to0$ | 73.58 | 76.76 | 73.6 | $72\pm1$ |
| 30.5 MeV $4\to1$ | 92.15 | 95.98 | 92 | $89\pm1.5$ |
| 30.5 MeV $2\to1$ | 36.30 | 35.99 | 36.3 | $34.5\pm1$ |
| 54.5 MeV $3\to0$ | 162.34 | 166.89 | 162 | $154\pm5$ |

Agreement with Klein's own calculation is within $0.2\%$, although the inputs
differ: PyRITE's TTKS-based per-atom correction is about $0.245$ (111) and
$0.144$ (222) versus Klein's GW $0.26$/$0.15$, and the atomic potentials and
thermal amplitudes are not Klein's. Part of that close agreement is therefore
coincidental, and the 1% tolerance is wider than the gap but narrower than the
input difference would justify demanding. The observations stay 2–5% below
both calculations, as Klein reported. The anchor therefore pins the
implementation to a published bonding-corrected many-beam calculation and
shows the correction moves the isolated lines in the observed direction. It
does not show that the residual offset to experiment has been removed.

### 4.5 Scope and wording findings (not discrepancies)

1. The Si $B_{\rm src}=0.4632$ Å$^2$ is the single $B$ LZD take from
   Spackman (their Ref. 30), not an LZD fit (Table I: $0.4585$). The ledger
   Notes call both values "LZD single-B fits"; only diamond's is.
2. With diamond's catalog $B=0.21$ Å$^2$, the X-ray path reproduces the TTKS
   values only to within $e^{-(0.21-0.1379)s^2}$ (0.7% at (111), 2.9% at
   (440)). That is the open `debye-waller-catalog-provenance` issue. The
   correction is right.
3. $\Delta F$ absorbs the Waasmaier–Kirfel residual against the measurement
   (e.g. Si (111): static per-atom $\Delta F = 0.191$ e versus LZD's
   dynamic-solid minus dynamic-atom difference $0.148$ e). This is intended
   (ledger Notes).
4. The (222) sign comes from theory, not from the measurement. Alkire and
   Weiss–Middleton determine $\lvert F\rvert$ only.

## 5. Verdict

Re-derivation matches the implementation in every factor, sign, phase and
convention: units, limits and signs/conventions all pass. Verdict:
**rederived**. The anchor test passes (31 passed, 1 skipped: the (100)/(110)
case at an energy with no bound transition), so `anchored` is supportable,
with the scope in §4.4. Human sign-off is pending.
