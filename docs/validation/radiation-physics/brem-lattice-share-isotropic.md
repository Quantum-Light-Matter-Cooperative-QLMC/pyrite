# `brem-lattice-share-isotropic`

Fresh-context independent verification for issue #1261, on branch
`issue-1261-eedl-lattice-share` at `00029e0f1`. During the review the branch
was rebased to `e1cfa05fa`. Every file under review (the three source modules,
the two test modules and the ledger part) is byte-identical between the two
commits.

## Claim and scope

- **Claim:** an isotropic continuum (EEDL or Bethe–Heitler $d\sigma_j/dk$,
  emitted as $d\sigma_j/dk/4\pi$ toward every direction) loses the
  lattice-coherent CBS share of the scored reflections by absolute subtraction
  of its solid-angle integral,
  $d\sigma_j/dk\to\max(0,\,d\sigma_j/dk-\Delta\sigma_j)$, with
  $\Delta\sigma_j(T,k)=2\pi\int S_j(T,\theta,k)\sin\theta\,d\theta/n_j$.
  $\Delta\sigma_j$ is the SDCS that the BremsLib route removes from element $j$.
- **Code:** `montecarlo/spectrum/brem_lattice_share.py::{isotropic_share_sigma,IsotropicLatticeShare,isotropic_shares_of,without_isotropic_share}`;
  `montecarlo/spectrum/brem.py::mc_brem_spectrum` (`isotropic_lattice_share`);
  `montecarlo/runner/lattice_share.py::{layer_isotropic_share,lattice_share_record}`;
  `montecarlo/runner/emission.py::_uncoupled_brem_from_segments`.
- **Source:** the parent claim `brem-lattice-coherent-fraction`, which supplies
  the share density $S_j$ and the absolute-subtraction denominator; its
  [write-up](brem-lattice-coherent-fraction.md) and
  [production verification](brem-lattice-coherent-fraction.md#production-verification-2026-11-17).
  Also `bremslib-angular-model`, which supplies the scaled DDCS/SDCS convention
  and `solid_angle_integral`.

**Independence.** The derivation below uses only the ledger row, the parent
write-up and the `bremslib-angular-model` row. It was written before I read the
bodies of the functions under review.

## Independent derivation

### 1. Double count per direction and per unit path

From the parent claim, the orientation-averaged CBS-only line yield of the
scored set $\mathcal G$, per unit electron path, is

$$
S(T,\theta,k)=\sum_{\mathbf g\in\mathcal G}\lvert U_{\mathbf g}\rvert^2 h(\lvert\mathbf g\rvert;T,\theta,k)
\qquad[\text{photons}/(\text{eV sr Å})],
$$

with $\theta$ the angle between the segment direction $\hat{\mathbf v}$ and the
photon direction $\hat{\mathbf n}$. The orientation average makes $S$
azimuthally symmetric about $\hat{\mathbf v}$. The element split
$S=\sum_j S_j$ uses the parent's incoherent weights.

A straight segment of length $L$ in element $j$ (density $n_j$ in atoms/Å$^3$)
contributes the following isotropic continuum per eV per sr:

$$
\frac{d^2N^{\rm iso}_j}{dk\,d\Omega}=\frac{n_j L}{4\pi}\,\frac{d\sigma_j}{dk}.
$$

The line route adds $L\,S_j(\theta)$ in the same direction. The continuum is an
isolated-atom (amorphous) model, so by the parent's sum rule it already
contains the Debye–Waller-weighted coherent part. In the true angular
distribution, the coherent part is $L\,S_j(\theta)$ per sr.

### 2. Angle-integrated consistency fixes the isotropic correction

An isotropic continuum has one degree of freedom per $(T,k)$: its total
$d\sigma_j/dk$. Write the corrected continuum as
$(d\sigma_j/dk-\Delta\sigma_j)/4\pi$. Requiring that lines plus continuum,
integrated over $4\pi$, equal the isotropic model's own angle-integrated yield
gives

$$
n_jL\left(\frac{d\sigma_j}{dk}-\Delta\sigma_j\right)+L\!\int_{4\pi}\!S_j\,d\Omega
=n_jL\,\frac{d\sigma_j}{dk}
\;\Longrightarrow\;
\boxed{\;\Delta\sigma_j(T,k)=\frac{2\pi}{n_j}\int_0^\pi S_j(T,\theta,k)\sin\theta\,d\theta\;}
$$

The azimuthal integral is trivial because $S_j$ does not depend on the azimuth.
The result holds for every segment direction, because $\int d\Omega$ over
$\hat{\mathbf n}$ is the same for every $\hat{\mathbf v}$.

**Units.** $S_j$ is in photons/(eV sr Å), so $\int S_j\,d\Omega$ is in
photons/(eV Å). Dividing by $n_j$ (Å$^{-3}$) gives Å$^2$/eV. Then
$1\ \text{Å}^2=10^{-16}$ cm$^2$ gives cm$^2$/eV per atom, the unit of the
EEDL and Bethe–Heitler $d\sigma/dk$. **pass.**

### 3. Identity with the BremsLib route's removed SDCS

The BremsLib route stores $\chi=(k/Z^2)\,d^2\sigma/dk\,d\Omega$ in mb/sr. At
each node, the parent subtracts

$$
\Delta\chi_j=\frac{k\,S_j}{n_jZ^2}\times10^{11}\ \text{mb/sr}.
$$

It evaluates $S_j$ on the kernel's $\theta$ nodes and interpolates it linearly
to the table's $\theta$ nodes. The table's SDCS is the solid-angle integral of
the DDCS, with the DDCS linear in $\theta$ between nodes. The SDCS ratio
rescaling in `lattice_share_table` preserves that property, because the staged
SDCS equals that integral of the uncorrected DDCS. The removed scaled SDCS is
therefore $\mathcal I_{\rm tab}[\Delta\chi_j]$, where $\mathcal I_\Theta$ is the
exact integral $2\pi\int f\sin\theta\,d\theta$ of the linear interpolant of $f$
on node set $\Theta$. Converting with $d\sigma/dk=\chi Z^2/k$ and
$1\ \text{mb}=10^{-27}$ cm$^2$:

$$
\Delta\!\left(\frac{d\sigma_j}{dk}\right)_{\rm BremsLib}
=10^{-16}\,\frac{\mathcal I_{\rm tab}[S_j^{\rm lin}]}{n_j}\ \text{cm}^2/\text{eV}.
$$

When the table nodes contain the kernel nodes (here both are every 5° from
0° to 180°), the linear interpolant of $S_j$ on the kernel nodes, sampled at the
table nodes and re-interpolated, is the same piecewise-linear function. So
$\mathcal I_{\rm tab}=\mathcal I_{\rm kernel}$ exactly. The identity therefore
holds to rounding, but only under three conditions:

- the same piecewise-linear-in-$\theta$ rule is used on the kernel nodes;
- no DDCS node clamps on the BremsLib side;
- it is evaluated at a BremsLib incident-energy node. Between nodes, each route
  applies its own interpolation in $\ln T$.

The total of a linear-in-$\theta$ rule is a modelling choice. Another rule on
the same nodes differs by the quadrature error (§7, check 4). The identity does
not depend on that choice, because both routes use the same rule.

### 4. Absolute versus fractional subtraction

The approved P4 wording defines a BremsLib-DDCS-weighted average fraction:

$$
\bar r_j=\frac{\int r_j\,\mathrm{DDCS}_j\,d\Omega}{\mathrm{SDCS}_j},
\qquad
r_j=\frac{S_j/n_j}{\mathrm{DDCS}_j}
\;\Longrightarrow\;
\bar r_j=\frac{\int S_j\,d\Omega}{n_j\,\mathrm{SDCS}^{\rm BL}_j}
=\frac{\Delta\sigma_j}{\mathrm{SDCS}^{\rm BL}_j}.
$$

Applying $\bar r_j$ to EEDL removes
$\bar r_j\,\sigma^{\rm EEDL}_j=\Delta\sigma_j\,(\sigma^{\rm EEDL}_j/\sigma^{\rm BL}_j)$.
The two readings differ by $\Delta\sigma_j(\sigma^{\rm EEDL}/\sigma^{\rm BL}-1)$.

- The **absolute** form keeps lines + continuum $=\sigma^{\rm model}$ (§2).
  This is the parent's primary denominator. The line route scores $S$ whatever
  continuum table is used, so the photons counted twice are $S$, not
  $\bar r\,\sigma^{\rm model}$.
- The **fractional** form keeps the coherent *fraction* of the continuum. It
  would be correct only if the EEDL-vs-BremsLib normalization difference sat
  proportionally in the coherent part. It needs BremsLib tables in an EEDL run.
  It also inherits the parent's warning: electron–electron brem, which EEDL
  includes, is not lattice-coherent, so a fraction would have to apply to the
  nuclear part only.
- Magnitude (check 3): $\sigma^{\rm EEDL}/\sigma^{\rm BL}$ for Si over 0.5–10 keV
  is 1.013, 1.014, 1.001, 0.956 and 0.945 at 6, 30, 100, 300 and 1000 keV. The
  removed share is 12–16 % of EEDL in band (Si, $G=30$ Å$^{-1}$). The readings
  therefore differ by at most about $7\times10^{-3}$ of the in-band continuum,
  the same order as the parent's absolute-vs-fraction systematic (0.001–0.008).

**Absolute is the correct reading.** The ledger's note that the fraction
"collapses to $\int S_j\,d\Omega/(n_j\,\mathrm{SDCS}_{\rm BremsLib})$" is
correct.

### 5. What an isotropic correction does and does not do

After correction, the error relative to a hypothetical "true" continuum, per
direction, is

$$
\underbrace{\Big[\tfrac{n_j}{4\pi}\tfrac{d\sigma_j}{dk}-n_j\,\mathrm{DDCS}_j(\theta)\Big]}_{\text{pre-existing isotropic-model error}}
-\underbrace{\Big[\tfrac{n_j\Delta\sigma_j}{4\pi}-S_j(\theta)\Big]}_{\text{residual of the share}} .
$$

The second bracket integrates to zero over $4\pi$ for every segment, so the
double count is removed exactly in the angle-integrated yield. Per direction it
is not: a detector at angle $\theta_d$ to a segment keeps a double count
$L[S_j(\theta_d)-\langle S_j\rangle_{4\pi}]$, which can be positive or negative.
The anisotropy is large (check 5). For Si, $G=30$ Å$^{-1}$, band
0.5–10 keV, $S(\theta)/\langle S\rangle$ is:

| $\theta$ | 0° | 45° | 90° | 135° | 180° |
|---|---|---|---|---|---|
| 30 keV | 2.66 | 1.57 | 0.81 | 0.56 | 0.49 |
| 300 keV | 12.9 | 1.56 | 0.42 | 0.18 | 0.13 |

The BremsLib DDCS has a similar but weaker anisotropy:
$\mathrm{DDCS}/(\mathrm{SDCS}/4\pi)$ is 2.08 / 1.41 / 0.86 / 0.70 / 0.67 at
30 keV and 8.6 / 1.67 / 0.52 / 0.27 / 0.21 at 300 keV. The local coherent
fraction $r(\theta)/\bar r$ therefore varies by about $\times0.6$–$1.5$ at
300 keV, which is much less than $S(\theta)$ itself varies. Relative to the
isotropic model's own angular error (the first bracket), removing the share
isotropically is close to removing the same fraction per direction. In a
directional observable, the per-direction residual of the share is therefore
second order next to the EEDL route's existing isotropy approximation. It is
still not zero. A forward beam with little multiple scattering, viewed at a
fixed angle, keeps a direction-dependent residual of order
$(r(\theta)/\bar r-1)\,\Delta\sigma/\sigma$, a few percent of the in-band
continuum at 300 keV. The ledger's Assumptions state this qualitatively. It
should state the magnitude.

### 6. Limits, floor and range

| Case | Expected | Reason |
|---|---|---|
| $B\to\infty$ | $\Delta\sigma_j=0$ | $\lvert U_{\mathbf g}\rvert^2\propto e^{-2W}\to0$ |
| $G\to0$ (empty $\mathcal G$) | $\Delta\sigma_j=0$ | empty sum |
| $T<E_{\rm cut,lines}$ | $\Delta\sigma_j=0$ | the line route scores no CBS there. The parent's v2 consistency condition, $\Delta\sigma_j(T)=(\cdot)\,\Theta(T-E_{\rm cut,lines})$, applies unchanged to an isotropic continuum. |
| $T$ above the kernel top (1053.4 keV) | $\Delta\sigma_j=0$ | the kernel is undefined there. This is a **known under-subtraction**: the lines are still scored. Same scope as the parent. |
| $k>T$, $k\le0$ | $\Delta\sigma_j$ irrelevant | the clamp gives zero, because EEDL is zero there |
| amorphous layer, channeling, `cxr_source="none"`, disabled | no correction | as in the parent |

**Floor shape.** The consistency condition is a step at
$E_{\rm cut,lines}$. A share table whose first $\ln T$ node *is*
$E_{\rm cut,lines}$, and which is zero below it, gives an exact step. The
BremsLib route instead ramps over one staged interval (parent finding 1). Just
below the cutoff the two routes therefore differ, and the isotropic route is
the more exact of the two.

### 7. Clamping

$\max(0,\,d\sigma_j/dk-\Delta\sigma_j)$ per (segment, $k$) cell
under-subtracts wherever $\Delta\sigma_j>d\sigma_j/dk$, and the
angle-integrated identity then fails by the clipped amount. The BremsLib route
clamps per DDCS node, which differs from clamping the SDCS. The two agree only
when neither clamps. Per-element clamping makes the element split load-bearing,
exactly as in the parent. For Si the share is at most 21 % of EEDL (6 keV,
$k\approx130$ eV), so clamping is inactive in the checked scenes. The record
should count clipped cells, or the ledger should state that clamping is
unmonitored on this route; see flag F3.

### 8. Cheap filters

| Check | Result |
|---|---|
| Units of $\Delta\sigma_j$ | photons/(eV sr Å) · sr / Å$^{-3}$ = Å$^2$/eV → $\times10^{-16}$ cm$^2$/eV. **pass** |
| Same units as the subtrahend | EEDL and Bethe–Heitler $d\sigma/dk$ are in cm$^2$/eV per atom; the scorer multiplies by $n\cdot10^{24}$ cm$^{-3}$ and path in cm. **pass** |
| Sign | subtraction; non-negative after the clamp. **pass** |
| $B\to\infty$, $G\to0$ | zero. **pass** |
| Floor / kernel top | zero below and above. **pass** (above-kernel scope noted) |
| $4\pi$ sum rule | lines + corrected continuum = model total per segment, when no cell clamps. **pass** |
| Absolute vs fraction | absolute is correct (§4). **pass** |

## Implementation comparison

| Item | Code | Result |
|---|---|---|
| $\Delta\sigma_j$ | `brem_lattice_share.py::isotropic_share_sigma`: `lattice_share_density` on the kernel $\theta$ nodes; `solid_angle_integral(deg2rad(theta), S.T)`; `* 1e-16 / n` | **matches** §2. `solid_angle_integral` (`_bremslib_table.py:112–129`) is the exact integral of the linear interpolant; I re-derived its per-interval formula. The kernel $\theta$ nodes span 0–180° every 5°, so the full sphere is covered. |
| BremsLib identity | the same rule and nodes as `lattice_share_table` (`:350–402`) | **matches** §3; reproduced to $1.3\times10^{-14}$ (check 1) |
| $T$ table | `:465–480`: nodes from `max(kernel T0, T_floor_keV)` through the inner kernel nodes to the kernel top, each interval split into 4 in $\ln T$; the first node is the floor itself | **matches**; exact step at the floor (§6) |
| $T$ evaluation | `:518–532`: linear in $\ln T$, `inside` mask zero outside $[T_{\rm lo},T_{\rm hi}]$ | **matches** |
| $k$ grid | `:488–516`: direct when the grid has ≤ 96 points per decade; otherwise 96 log nodes per decade, linear in $\ln k$; $k\le0\to0$ | **matches** the ledger. Interpolation error is covered by the test's between-node check. |
| Subtraction and clamp | `:540–549`, `xp.maximum(dsig - share.sigma(...), 0)` | **matches** §7 (per cell, per element) |
| Scorer hook | `brem.py:1223`: applied to the EEDL or Bethe–Heitler `dsig` (cm$^2$/eV) before `* T_abs` and `_accumulate(n_i*1e24*unit*path_cm, ...)` (`:1226`) | **matches**; absorption acts on the remainder; `unit = 1` on this path |
| Isotropic elements only | `isotropic_shares_of` (`:535–537`) drops elements staged for BremsLib; `brem.py:1046` | **matches**. A staged element's per-segment EEDL fallback is not corrected, and the kernel range lies inside BremsLib's, so this is inert. |
| Fused CUDA kernel | `brem.py:1047`: `_use_jit = ... and not isotropic_shares` | **matches** (chunked path only) |
| Runner route | `emission.py:105–114`: shares only when `"bremslib_tables"` is absent, via `layer_isotropic_share` | **matches**; untabled elements of a BremsLib run stay uncorrected and warned (`lattice_share.py:203–212`) |
| Layer wiring | `lattice_share.py:266–311`: `skip_reason` → same scored set (`scored_reflections`), `B_ang2`, composition and floor `E_cut_lines_keV` as the BremsLib route | **matches** |
| Record | `lattice_share.py:494–525`: isotropic record when `_case_bremslib_tables` is `None` (the model is not `"bremslib"`), `model=lattice_share_model(...)` → `.../isotropic-theta-integrated-v1` | **matches** the routing key (`case["bremsstrahlung_model"]`) |
| Dielectric suppression | `dielectric_scoring_plasma` rejects a plasma frequency for non-BremsLib models | n/a; no ordering issue |

## Numerical checks run for this verification

All checks ran locally, one at a time, under
`systemd-run --user --scope -p MemoryMax=8G`, against the released kernel
(`require_verified("brem-lattice-share")`).

1. **BremsLib identity (independent script).** Si, $B=0.46$ Å$^2$,
   $G=8$ and $30$ Å$^{-1}$: `lattice_share_table`'s removed SDCS
   ($Z^2/k\times10^{-27}$ × scaled SDCS difference) versus
   `isotropic_share_sigma` at the 30/100/300 keV table nodes, over
   0.5–10 keV. The maximum relative difference is $1.3\times10^{-14}$; zero
   cells agree (at $G=8$, 30 keV, above the line endpoint). No nodes clamped.
   The removed fraction of the BremsLib SDCS is 0.067/0.060/0.055 ($G=8$) and
   0.141/0.124/0.112 ($G=30$), which matches the ledger's 0.10–0.13 at
   100–300 keV.
2. **Absolute scale of the $\theta$ integral, independent classical Monte
   Carlo.** I took one shell with $\lvert U\rvert^2=1$ eV$^2$ and the parent's
   classical Liénard–Wiechert yield
   $dN/d\Omega\,dL=\alpha\lvert\hat{\mathbf n}\times[(\hat{\mathbf n}-\boldsymbol\beta)\times\mathbf G]\rvert^2/[2\pi\omega(\gamma mc^2)^2D_0^5\beta]$.
   I sampled $\hat{\mathbf g}$ and $\hat{\mathbf n}$ jointly uniform
   ($2\times10^7$ pairs), histogrammed in $E=\hbar c\,\mathbf g\cdot\mathbf v/D_0$
   and multiplied by $4\pi$. This gives $\int S\,d\Omega$ with no kernel and no
   $\theta$ quadrature. Ratio of `isotropic_share_sigma` × $10^{16}$ to the
   Monte Carlo, two seeds:

   | $T$, $\lvert g\rvert$ | 0.5–10 keV band | per 500 eV bin |
   |---|---|---|
   | 300 keV, 8 Å$^{-1}$ | 1.007 / 1.005 | 0.977–1.027 |
   | 300 keV, 3 Å$^{-1}$ | 1.004 / 1.001 | 0.969–1.010 |
   | 100 keV, 8 Å$^{-1}$ | 0.997 / 0.997 | 0.956–1.007 |

   The 100 keV per-bin deficit rises toward high $k$, as recoil (absent from the
   classical reference) predicts; the parent observed the same at 30 keV.
3. **EEDL vs BremsLib, and clamp margin.** For Si, $G=30$, the band ratio
   $\sigma^{\rm EEDL}/\sigma^{\rm BL}$ and the in-band $\Delta\sigma/\sigma^{\rm EEDL}$
   are 1.013 and 0.164 at 6 keV, 1.014 and 0.140 at 30 keV, 1.001 and 0.125 at
   100 keV, 0.956 and 0.122 at 300 keV, and 0.945 and 0.158 at 1000 keV. The
   maximum $\Delta\sigma/\sigma^{\rm EEDL}$ over 0.1 keV–$T$ is 0.21, so no
   clamp occurs.
4. **Quadrature rule.** On the same $S$ nodes, trapezoid in $\cos\theta$ divided
   by the exact linear-in-$\theta$ rule is 1.0004 / 1.0011 / 1.0035 at
   30/100/300 keV. The quadrature-rule uncertainty of the absolute $\Delta\sigma$
   is ≤ 0.4 %; it does not affect the identity.
5. **Anisotropy** (§5 table).
6. **Anchors.** I ran
   `pyrite-dev test --slow tests/montecarlo/spectrum/test_brem_lattice_share.py tests/montecarlo/runner/test_lattice_share.py`:
   21 passed, including
   `test_isotropic_share_is_the_sdcs_the_bremslib_route_removes`,
   `test_isotropic_share_limits_and_floor`,
   `test_eedl_spectrum_loses_exactly_the_isotropic_share` and
   `test_eedl_crystal_gets_theta_integrated_shares_without_warning`.

## Discrepancies and flags

No term, factor, unit or sign diverges from the re-derivation. Flags:

- **F1. Per-direction residual (scope; ledger wording).** The isotropic
  correction is exact only in the $4\pi$ integral (§5). $S(\theta)/\langle S\rangle$
  spans 0.13–12.9 at 300 keV. Against the isotropic model's own angular error,
  the residual is $(r(\theta)/\bar r-1)\,\Delta\sigma/\sigma$, a few percent of
  the in-band continuum at 300 keV for a forward beam at a fixed detector angle.
  The ledger's Assumptions state the principle but not the magnitude.
- **F2. Identity conditions (ledger wording).** "Exactly the SDCS the BremsLib
  route removes" holds only under three conditions: at BremsLib $T$ nodes, when
  no DDCS node clamps, and when the table's $\theta$ nodes contain the kernel's.
  Between $T$ nodes the two routes interpolate differently in $\ln T$. Just
  below $E_{\rm cut,lines}$ the BremsLib route ramps its share while this
  route steps it exactly. The Checks give the measured figures; the Claim text
  should carry the qualifier.
- **F3. Clamping unrecorded (minor).** `without_isotropic_share`
  (`brem_lattice_share.py::without_isotropic_share`) clamps per cell, but the `continuum="isotropic"`
  record (`lattice_share.py:298–307`) has no clipped-cell count, unlike the
  BremsLib record's `clamped_nodes`. The clamp is inactive in the Si scenes
  checked (maximum 21 % of EEDL). A compound whose element split
  over-weights a light element is the case to watch.
- **F4. Selector inconsistency (documented).** The reflection selector's net
  metric stays BremsLib-only (`SelectionContinuum.applies` requires tables), so
  an EEDL selection uses gross line yield. The Notes state this. It is not a
  physics error in the correction.
- **F5. Above-kernel scope (documented).** For $T>1053.4$ keV the lines are
  scored and nothing is subtracted. The parent Notes say "> 1.06 MeV"; the
  released kernel's top node is 1053.4 keV, and this row's "1.05 MeV" is the
  accurate figure.
- **F6. Docs wiring (for the task owner).** This page is not yet in the
  `docs/validation/radiation-physics/index.md` toctree, and the ledger row does
  not link it. The verifier may write only this file.

## Verdict

- **Claim**: `brem-lattice-share-isotropic` —
  `montecarlo/spectrum/brem_lattice_share.py::{isotropic_share_sigma,IsotropicLatticeShare,without_isotropic_share}`,
  `montecarlo/runner/lattice_share.py::layer_isotropic_share`,
  `montecarlo/spectrum/brem.py::mc_brem_spectrum`,
  `montecarlo/runner/emission.py::_uncoupled_brem_from_segments` — the
  $4\pi$-integral of the parent's Born–DW CBS-only share density $S_j$
  (`brem-lattice-coherent-fraction` §§1–3, absolute denominator), per atom,
  subtracted from the isotropic $d\sigma_j/dk$.
- **Filters**: units pass (photons/(eV sr Å) → Å$^2$/eV → $10^{-16}$ cm$^2$/eV);
  limits pass ($B\to\infty$, $G\to0$, line-cutoff step, kernel top);
  signs and conventions pass (absolute, not fractional; same linear-in-$\theta$
  rule as `solid_angle_integral`).
- **Re-derivation**: `matches`. The identity with the BremsLib-removed SDCS
  reproduced to $1.3\times10^{-14}$ at table nodes; the absolute scale is within
  0.3–0.7 % of an independent classical full-sphere Monte Carlo.
- **Verdict**: `rederived`. The named anchors are green locally (21/21 with
  `--slow`); they support `anchored` once CI is green.
- **Write-up**: `docs/validation/radiation-physics/brem-lattice-share-isotropic.md`
- **Suggested ledger change**: Status `unverified` → `rederived`, or
  `anchored` when CI is green on the named anchors. Never `signed-off`.
  - Claim: after "equals the isotropic model's own", add "(at BremsLib `T`
    nodes, with no DDCS node clamped)".
  - Assumptions: add "per direction the double count is
    `L[S_j(θ) − ⟨S_j⟩_{4π}]`; `S(θ)/⟨S⟩` spans 0.49–2.66 (30 keV) and
    0.13–12.9 (300 keV) for Si `G = 30`, against 0.67–2.08 and 0.21–8.6 for
    the BremsLib DDCS, so the local coherent fraction varies ×0.6–1.5 at
    300 keV: a few percent of the in-band continuum in a fixed-angle
    forward-beam observable".
  - Notes: add "Fresh-context verification 2026-11-21: `rederived`; BremsLib
    identity reproduced to 1.3e-14 (Si, `G = 8`, 30), absolute θ-integral within
    0.3–0.7 % (band) of an independent classical full-sphere Monte Carlo at
    100–300 keV; absolute vs fractional (`r̄_j`) differ by
    `Δσ(σ_EEDL/σ_BL − 1)`, ≤ 7e-3 of the in-band continuum for Si; floor is an
    exact step at `E_cut_lines_keV` (the BremsLib route ramps over one staged
    interval); clamped cells are not recorded. `[Independent
    verification](radiation-physics/brem-lattice-share-isotropic.md)`".
  - Add this page to the `radiation-physics` toctree.
