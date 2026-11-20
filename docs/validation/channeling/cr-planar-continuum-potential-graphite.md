# `cr-planar-continuum-potential` (graphite) — 2H graphite planes (001), (100), (110) and the uniaxial row

Verifier: fresh context (2026-11-19). Ledger: `docs/validation/ledger-channeling.md` (`cr-planar-continuum-potential`, `cr-uniaxial-thermal-projection`). Code: `src/pyrite/materials/channeling/potential.py::{planar_profile,_normal_b_factors}`, `src/pyrite/materials/channeling/thermal.py::UniaxialThermalSite`, `src/pyrite/data/channeling_thermal.toml::hopg`, catalog crystal `hopg` (`src/pyrite/data/cifs/hopg.cif`, `materials/crystal.py::CRYSTALS["hopg"]`). Issue #540. Complements [the Si/diamond write-up](cr-planar-continuum-potential.md) and [the MoS$_2$ projection write-up](cr-uniaxial-thermal-projection.md); it does not repeat their general derivations.

Independence note. The extinction analysis, period reduction, projected $B$ and numerical rebuild below were done from the ledger claim, the Trucano and Chen paper and my own $P6_3/mmc$ basis before I read the bodies of `planar_profile`, `_normal_b_factors` and `UniaxialThermalSite`. I had read the ledger rows, the two earlier write-ups, the TOML row and the CIF (data only) beforehand. The implementation's $f_0$ is `xraydb` (Waasmaier–Kirfel), the same package I used for the primary rebuild, so the WK form factor is not independent. I therefore also repeated the sums with the 1968 Cromer–Mann carbon coefficients ($a=2.31,1.02,1.5886,0.865$; $b=20.8439,10.2075,0.5687,51.6512$; $c=0.2156$), typed in from memory of International Tables Vol. IV and only validated against `xraydb` (agreement in the table below). All numerics are in `scratch/v540g/` (`indep.py`, `indep_run.py`, `cmp.py`, `extras.py`, `sens.py`), not committed.

## 1. Source transcription (Trucano and Chen 1975)

I re-read the PDF (Zotero `NRG2VMUP`, local storage `8FP3EFGZ`), p. 137 Table 2, last column ($P6_3/mmc$, 1/24 sphere, atoms $(0,0,\tfrac14)$, $(\tfrac13,\tfrac23,\tfrac14)$): $\mathrm C_1$ at $(0,0,\tfrac14)$ with $U_{11}=0.0031\ \text{\AA}^2$ (a bare value, i.e. fixed, not refined) and $U_{33}=0.016\pm0.002\ \text{\AA}^2$; $\mathrm C_2$ at $(\tfrac13,\tfrac23,\tfrac14)$ with $U_{11}=0.0031\ \text{\AA}^2$ and $U_{33}=0.017\pm0.002\ \text{\AA}^2$; $R=4.2\%$. The text states the basal $0.0031\ \text{\AA}^2$ was assumed from neutron measurements (Ludsteck 1972), that $\mathrm C_1$ and $\mathrm C_2$ "are not significantly different", and gives Debye temperatures $490\pm40$ and $470\pm40$ K. **No temperature is stated anywhere in the letter.** Table 1 lattice: $a=2.464\pm0.002$, $c=6.711\pm0.004$ Å (catalog $a=2.461$).

Stored row `[hopg.C]`: `u_basal_ang2 = 0.0031` (matches), `u_c_ang2 = 0.0165` $=(0.016+0.017)/2$ (the equal-multiplicity mean, correct arithmetic), `temperature_K = 300`, `tolerance_K = 10`. The transcription matches the paper. The paper's $U_{ij}$ are mean-square amplitudes ($\langle u^2\rangle$, Å$^2$), consistent with the code's `u_*_ang2`; for a hexagonal site the basal value is the same along every basal direction, so the Cartesian $U_{xx}=U_{yy}$ equals the table's $U_{11}$.

## 2. Independent derivation

### 2.1 Lattice and basis

$\mathbf a_1=(a,0,0)$, $\mathbf a_2=(-a/2,\,a\sqrt3/2,\,0)$, $\mathbf a_3=(0,0,c)$, $V=\tfrac{\sqrt3}{2}a^2c=35.1999\ \text{\AA}^3$ for $a=2.461$, $c=6.711$ Å; $\mathbf b_i=2\pi(\mathbf a_j\times\mathbf a_k)/V$. The paper's atoms are $\mathrm C_1$: $(0,0,\tfrac14),(0,0,\tfrac34)$ (Wyckoff $2b$) and $\mathrm C_2$: $(\tfrac13,\tfrac23,\tfrac14),(\tfrac23,\tfrac13,\tfrac34)$ ($2c$). Both sites have site symmetry $\bar6m2$ (a threefold-or-higher axis along $c$), so each individually has the Cartesian covariance $\mathrm{diag}(u_b,u_b,u_c)$, with $u_b$ the same for both and $u_c$ the only inequivalence.

The catalog CIF shifts $z$ by $-\tfrac14$: $(0,0,0),(0,0,\tfrac12)$ and $(\tfrac13,\tfrac23,0),(\tfrac23,\tfrac13,\tfrac12)$. This is a pure translation by $-c/4$ along $\mathbf c$, so for a planar sum along $\hat{\mathbf c}$ it multiplies $U_n$ by a phase and moves the wells; every modulus, period and depth is unchanged. The bilayer stacking (A at $z=0$ holds $(0,0),(\tfrac13,\tfrac23)$; B at $z=\tfrac12$ holds $(0,0),(\tfrac23,\tfrac13)$) is the Bernal ABAB sequence, as the paper's.

### 2.2 Structure factors and extinctions

For harmonic $n$ of the coprime plane $\mathbf h$ the geometric factor is $S_n=\sum_je^{-i2\pi n\,\mathbf h\cdot\mathbf R_j}$ (times $e^{-B_js_n^2}$ with the plane-normal $B$). With the catalog origin:

- $(001)$: $\mathbf h\cdot\mathbf R_j=z_j\in\{0,\tfrac12,0,\tfrac12\}$, so $S_n=2\,(1+(-1)^n)$, which vanishes for odd $n$ ($00l$ with $l$ odd is the $6_3$ extinction) and equals $4$ for even $n$. Allowed $n=2,4,\ldots$; $\gcd=2$; true period $c/2=3.3555$ Å.
- $(100)$: $h\cdot x_j\in\{0,\tfrac13,0,\tfrac23\}$, so $S_n=2+2\cos(2\pi n/3)$, equal to $1$ for $n\not\equiv0\bmod3$ and $4$ for $n\equiv0\bmod 3$. No extinction; $n=1$ is allowed; true period $d_{100}=a\sqrt3/2=2.1313$ Å.
- $(110)$: $\mathbf h\cdot\mathbf R_j=x_j+y_j\in\{0,1,0,1\}$ integers for every atom, so $S_n=4$ for all $n$. No extinction; true period $d_{110}=a/2=1.2305$ Å. (All four atoms sit on the planes, which is why the profile is a single set of wells at $x=0$.)

These agree with the ledger's extinction claims `(00l) odd extinct`, `(100) a√3/2`, `(110) a/2`. Because all atoms are one element and $Z-f_0>0$, the potential coefficients inherit the zeros of $S_n$ (no accidental cancellation), so the period reduction is exact here.

### 2.3 Plane-normal thermal projection

For uniaxial $U=\mathrm{diag}(u_b,u_b,u_c)$, $\langle e^{-i\mathbf g\cdot\mathbf u}\rangle=e^{-\mathbf g\cdot U\mathbf g/2}=e^{-B_ns_n^2}$ with $B_n=8\pi^2[u_b(1-\cos^2\theta)+u_c\cos^2\theta]$ (derivation as in the MoS$_2$ write-up). Here $\hat{\mathbf n}\parallel\mathbf g_{hkl}$:

| plane | $\hat{\mathbf n}$ | $\cos^2\theta$ | $B_n=8\pi^2\hat{\mathbf n}\cdot U\hat{\mathbf n}$ |
| --- | --- | --- | --- |
| $(001)$ | $\hat{\mathbf z}$ | 1 | $8\pi^2u_c=1.30279\ \text{\AA}^2$ |
| $(100)$ | $(\tfrac{\sqrt3}{2},\tfrac12,0)$ | 0 | $8\pi^2u_b=0.244766\ \text{\AA}^2$ |
| $(110)$ | $(\tfrac12,\tfrac{\sqrt3}{2},0)$ | 0 | $8\pi^2u_b=0.244766\ \text{\AA}^2$ |

Consequence for the carbon inequivalence: $(100)$ and $(110)$ have $\theta=\pi/2$, so only $u_b$ (identical for $\mathrm C_1$ and $\mathrm C_2$) enters and the mean-versus-two-class approximation is exact for them. Only $(001)$ and non-prismatic planes see the $u_c$ difference.

### 2.4 Potential coefficients

$U_n=-\dfrac{4\pi e^2}{Vg_n^2}\,(6-f_0(g_n))\,S_n^{\rm th}$ with $S_n^{\rm th}=\sum_je^{-i2\pi n\mathbf h\cdot\mathbf R_j}e^{-B_js_n^2}$, $e^2=14.3996\ \text{eV\,\AA}$ ($e^2/(Vg^2)$: eV·Å/(Å$^3$·Å$^{-2}$) = eV), $U_0=0$, $U(x)=\sum_{n\ne0}U_ne^{ingx}$ with $U_{-n}=U_n^*$. In the catalog origin the three profiles are centrosymmetric about $x=0$ ($U_n$ real), with the wells at $x=0$ (all atoms sit at $\mathbf h\cdot\mathbf R=0$ for $(110)$; the atomic planes of $(001)$ and $(100)$ are at $x=0$ and its images). Independent numbers (primary $f_0$ = `xraydb`, harmonics to $s\le6\ \text{\AA}^{-1}$, $U(x)$ on 8001 points):

| plane | period [Å] | $U_1$ or fundamental [eV] | depth [eV] |
| --- | --- | --- | --- |
| $(001)$ | 3.3555 | $U_2=-9.536467$ (odd $n$: $\lvert U_n\rvert\sim10^{-15}$) | 49.4439 |
| $(100)$ | 2.13129 | $U_1=-1.681041$ ($U_3=-1.065196$) | 11.7295 |
| $(110)$ | 1.2305 | $U_1=-3.087153$ | 13.9135 |

## 3. Comparison with the implementation

Source comparison. `planar_profile` computes `phases = exp(-2j*pi*n*(hkl·R))`, the per-element `site_sum = Σ phases * exp(-B s²)`, accumulates `(Z-f0)/g² * site_sum`, multiplies by $-4\pi e^2/V$ with $n=0$ left at zero, and takes `step = gcd(live)` with `live` defined by $\lvert U_n\rvert>10^{-9}\max\lvert U\rvert$. That is term by term my formula, including the $e^{-B s^2}$ on the whole bracket and the sign of the site phase. `_normal_b_factors` computes $\cos^2\theta=(\mathbf g\cdot\mathbf c)^2/(g^2c^2)$ with $\mathbf c$ the third direct-lattice vector, which is $\hat{\mathbf z}$ for the catalog hexagonal cell (confirmed: the code's normals are $(0,0,1)$, $(0.866,0.5,0)$, $(0.5,0.866,0)$), and `b_factor_along` is $8\pi^2[u_b(1-c^2)+u_cc^2]$. `bonding_dF` is empty for `hopg`, so the bonding-charge branch is inactive (independent atom). `UniaxialThermalSite` enforces $\lvert T-T_{\rm row}\rvert\le$ tolerance.

Numerical comparison, `planar_profile("hopg", hkl, 300.0)` against my rebuild with the same $f_0$:

| plane | period (code / mine) [Å] | $B$ (code / mine) [Å$^2$] | #coefficients | $\max\lvert\Delta U_n\rvert$ [eV] | depth (code / mine) [eV] | $\max\lvert\Delta U(x)\rvert$ [eV] |
| --- | --- | --- | --- | --- | --- | --- |
| $(001)$ | 3.3555 / 3.3555 | 1.302788 / 1.302788 | 40 / 40 | 8.7e-8 | 49.443901 / 49.443901 | 3.3e-7 |
| $(100)$ | 2.131289 / 2.131289 | 0.244766 / 0.244766 | 25 / 25 | 1.5e-8 | 11.729508 / 11.729508 | 7.2e-8 |
| $(110)$ | 1.2305 / 1.2305 | 0.244766 / 0.244766 | 14 / 14 | 2.8e-8 | 13.913453 / 13.913453 | 8.1e-8 |

Minima and maxima agree: $(001)$ $-36.6259$/$+12.8180$ eV, $(100)$ $-7.8644$/$+3.8651$ eV, $(110)$ $-8.8818$/$+5.0316$ eV. Mean densities are $4/V$ atoms and $24/V$ electrons, $U_0=0$.

Additional off-axis planes (to test the projection at intermediate $\theta$ with the graphite lattice): for $(101),(102),(111),(211),(113)$ the code's $B$ equals my $8\pi^2\hat{\mathbf n}\cdot U\hat{\mathbf n}$ to $10^{-16}$ ($\cos^2\theta=0.0916,0.2875,0.0325,0.0142,0.2323$; $B=0.34170,0.54891,0.27918,0.25979,0.49053\ \text{\AA}^2$), and $\max\lvert\Delta U_n\rvert\le2.5\times10^{-8}$ eV with matching periods ($(111)$ and $(113)$ have step 2, odd harmonics extinct, as the $P6_3/mmc$ $6_3$ rule $h-k=3m$, $l$ odd requires).

Independent form factor. Replacing `xraydb` $f_0$ by the 1968 Cromer–Mann carbon fit changes the depth by $-0.0085$ eV for $(001)$ ($-0.017\%$), $+0.0002$ eV for $(100)$ and $-0.005$ eV for $(110)$: parametrization noise, not a formula difference.

## 4. Filters, limits and caveats

- Units: Å$^2$ for $U$ and $B$, dimensionless $B s^2$, eV for $U_n$. Pass.
- Isotropic limit $u_b=u_c$: injecting a `UniaxialThermalSite` with $u_b=u_c=0.0165$ at runtime (no source edit) gives $B=8\pi^2u=1.302788$ for $(001),(100),(110),(101),(111)$ and the same $U_n$ as my isotropic rebuild ($\le8.7\times10^{-8}$ eV). Basal plane uses $u_c$, prismatic uses $u_b$. Pass.
- Harmonic cutoff convergence (code): depth $(001)$ 49.443901 eV at $s_{\max}=3,6,8,12\ \text{\AA}^{-1}$; $(100)$ 11.7074, 11.7295, 11.7295, 11.7295 eV; $(110)$ 13.9034, 13.9135, 13.9135, 13.9135 eV. The default $s_{\max}=6$ is converged to $<10^{-5}$ eV (the $e^{-B s^2}$ factor, $B\ge0.245$, kills harmonics above $s\approx4$), and truncating at $s=3$ costs only 0.02 eV, unlike Si. Pass.
- Period threshold caveat (`cr-planar-continuum-potential`): the gcd keys on $\lvert U_n\rvert>10^{-9}\max\lvert U\rvert$, not on $S_n$. For $(001)$ the odd harmonics sit at $\sim10^{-15}$ eV relative to $9.5$ eV, so $\sim10^{-16}$ against a $10^{-9}$ threshold (seven orders of margin), and $n=2$ (the one that fixes step 2) is $9.5$ eV. $(100)$ and $(110)$ have $n=1$ live. With a single element and $Z-f_0>0$, no accidental zero can occur. The caveat is harmless here but remains a structural weakness for other crystals. Pass for this claim.
- Signs and conventions: structure factor $e^{-i2\pi n\mathbf h\cdot\mathbf R}$ with $e^{+ingx}$ synthesis, $U_{-n}=U_n^*$, $x$ from the cell origin. The code follows the ledger's post-#508 convention. In the catalog origin the $(001)$ wells are at $x=0$ and the barrier at $x=c/4$ (code $x_{\max}=1.677$ Å, mine 1.678 Å). Pass.

### Mean versus two-class carbon ($U_{33}$ 0.016 vs 0.017)

The row stores $u_c=0.0165$. Rebuilding with $\mathrm C_1=0.016$ and $\mathrm C_2=0.017$ separately (my code, two $B$'s, all four atoms):

| plane | $\Delta$depth (two-class $-$ mean) | $\max\lvert\Delta U(x)\rvert$ | worst relative $\Delta U_n$ for $\lvert U_n\rvert>10^{-3}\max$ |
| --- | --- | --- | --- |
| $(001)$ | $+1.9\times10^{-3}$ eV ($+0.004\%$) | $1.9\times10^{-3}$ eV | $3.8\times10^{-3}$ at $n=20$ ($\lvert\Delta U_n\rvert\le1.4\times10^{-4}$ eV) |
| $(100)$ | 0 | 0 | 0 (exact, $\theta=\pi/2$) |
| $(110)$ | 0 | 0 | 0 (exact, $\theta=\pi/2$) |
| $(101)$ | $-5.7\times10^{-3}$ eV | $7.0\times10^{-3}$ eV | $6.6\times10^{-2}$ at $n=10$ (first harmonic $2\times10^{-4}$) |

For $(001)$ both carbon classes share the same phases, so the sum is $\propto e^{-B_1s^2}+e^{-B_2s^2}$ against $2e^{-\bar Bs^2}$ for the mean. With $\delta=8\pi^2(0.0005)s^2$ the relative difference is $\delta^2/2$, which reproduces the numeric values ($3.84\times10^{-7}$ at $n=2$, $2.40\times10^{-4}$ at $n=10$, $3.84\times10^{-3}$ at $n=20$, 0.061 at $n=40$) to better than 1%. It is second order, grows as $s^4$, and only matters at harmonics whose amplitude is already $\ll$ the fundamental. The approximation is bounded: it changes no period, no extinction and no depth by more than $2\times10^{-3}$ eV for the three requested planes, which is two orders below the source's own $\pm0.002\ \text{\AA}^2$ on $U_{33}$.

## 5. Sensitivity to the data (not formula)

Depth change versus the stored $U$ values (my rebuild, mean row):

| variation | $(001)$ | $(100)$ | $(110)$ |
| --- | --- | --- | --- |
| $u_c=0.0145$ ($-2\sigma$) / $0.0185$ ($+2\sigma$) | $+1.57\%$ / $-1.44\%$ | 0 | 0 |
| $u_c=0.0140$ (Chen–Trucano 1978, not used) | $+1.98\%$ | 0 | 0 |
| $u_b\times1.5$ / $\times0.5$ (fixed value, no source uncertainty) | 0 | $-4.3\%$ / $+5.9\%$ | $-4.3\%$ / $+5.8\%$ |
| both $U$ scaled by $\pm3\%$ (a ±10 K proxy) | $\mp0.37\%$ | $\mp0.29\%$ | $\mp0.29\%$ |
| catalog $a=2.461$ versus the paper's 2.464 | $-0.24\%$ | $-0.05\%$ | $-0.02\%$ |

Temperature. $\partial\ln(\text{depth})/\partial\ln U\approx-0.12$ to $-0.14$ for each plane, i.e. depth is only weakly sensitive to $U$. A $\pm3\%$ change in both $U$'s (a generous proxy for a ±10 K shift: an Einstein oscillator at $\Theta\approx480$ K gives $\partial\ln U_{33}/\partial\ln T\approx0.67$, so about $\pm2\%$ for ±10 K) moves the depths by $<0.4\%$. Even a 30% error in $U$ from a mistaken measurement temperature would move the depth by about 4%.

## 6. Findings and data gaps

- No sign, factor, exponent, unit or convention divergence between my derivation and `planar_profile` / `_normal_b_factors` / `UniaxialThermalSite.b_factor_along` for $(001),(100),(110)$ nor for the five off-axis planes tested.
- Extinctions and true periods (derived by hand, then confirmed numerically): $(001)$ odd $n$ extinct, period $c/2=3.3555$ Å; $(100)$ none, period $a\sqrt3/2=2.1313$ Å; $(110)$ none, period $a/2=1.2305$ Å.
- Data transcription: Table 2 values match the paper exactly ($U_{11}=0.0031$, $U_{33}=0.016(2),0.017(2)$); the mean $0.0165$ is correct arithmetic; $U_{11}$ is fixed in the source, so it carries no stated uncertainty (the $(100)$ and $(110)$ depths hinge on that fixed Ludsteck value: $\partial\ln\text{depth}/\partial\ln u_b\approx-0.13$).
- Data gap (temperature): the letter states none. The 300 ± 10 K window is an assumption, correctly flagged in the TOML comment, the `source` string and the ledger. It does not affect the formula, extinctions, periods or the projection, and moves absolute depths by $<0.4\%$ per ±10 K by the estimate above. It affects the verdict only as a provenance caveat on absolute well depths, not on the equations.
- Data gap (mean approximation): the carbon classes are inequivalent in the source; the mean is bounded at $\le2\times10^{-3}$ eV in depth for $(001)$ and exact for prismatic planes, with relative errors growing as $s^4$ at high harmonics. Tracked in #1335.
- Data gap (anchors): no experimental graphite structure-factor or well-depth value was compared; this check is a derivation rebuild, not a measured-data anchor. The $f_0$ cross-check with Cromer–Mann 1968 used coefficients recalled from International Tables, validated only by agreement with `xraydb`.
- Independent-atom potential only: bonding-charge corrections are not modelled for graphite (`bonding_dF` empty); for a $\pi$-bonded layered crystal this affects low-order harmonics at a level this review does not bound.
- Mild lattice difference: catalog $a=2.461$ versus the paper's $a=2.464$ Å (bond length 1.422 Å gives $a=2.463$ Å). Effect $\le0.24\%$ in depth.

## 7. Verdict

Re-derivation matches the implementation in factor, sign, units, phase convention, thermal exponent, plane-normal projection, extinctions and true period, to $\le9\times10^{-8}$ eV in $U_n$ and $\le3.3\times10^{-7}$ eV in $U(x)$ for the three requested planes. Existing tests also pass (10 selected graphite/uniaxial tests, rc 0). Verdict: `rederived` for `cr-planar-continuum-potential` (2H graphite, planes $(001)$, $(100)$, $(110)$) and `cr-uniaxial-thermal-projection` (graphite row). Not applicable: any `signed-off` transition, which remains human-only.

Suggested ledger edit (task owner applies; not `signed-off`): in the `cr-uniaxial-thermal-projection` and `cr-planar-continuum-potential` Notes, replace "have not had a fresh-context `rederived` review" by "2026-11-19 fresh-context verdict `rederived` for 2H graphite `(001)`, `(100)`, `(110)` and the uniaxial row ([write-up](channeling/cr-planar-continuum-potential-graphite.md)): independent hexagonal rebuild matches to <9e-8 eV in `U_n` (depths 49.444, 11.730, 13.913 eV); Trucano–Chen Table 2 re-read and matches; carbon mean-vs-two-class <= 2e-3 eV in depth; 300 K remains an assumption with <0.4% depth effect per 10 K." Keep the temperature assumption and #1335 notes.
