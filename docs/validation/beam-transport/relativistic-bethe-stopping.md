# Relativistic (Berger–Seltzer) collision stopping — independent re-derivation

- **id**: `relativistic-bethe-stopping`
- **anchor**: `montecarlo/transport.py::_dEds_bs_compound_scalar`,
  `::_dEds_spliced_compound_scalar`, `::_dEds_spliced_packed_scalar`,
  `::_bs_joy_luo_crossover_keV`, `::spliced_stopping_keV_per_ang`,
  `::build_transport_energy_lut`; `montecarlo/transport/_jit_device.py::_dEds_packed`;
  `montecarlo/spectrum/lines.py::_spliced_stopping_magnitude_xp`
- **source**: ICRU Report 37 (1984); Berger & Seltzer, NBSIR 82-2550 (1982);
  PDG *Atomic and Nuclear Properties* (mean excitation energies, Sternheimer
  density-effect coefficients); Sternheimer, Berger & Seltzer, *At. Data Nucl.
  Data Tables* **30**, 261 (1984)
- **verifier**: fresh context (did not write the implementation)

Process note: the ledger row for this claim (`docs/validation/ledger-transport-background.md`)
already carries an unusually detailed prose account of the intended closed
form (prefactor value, `F^-(0)` limit, Joy–Luo constants, crossover values,
Bragg-sum coefficient rewrite) as part of its own in-context filter record.
That prose was read as instructed, before the implementation. §2 below is a
from-scratch reconstruction of the ICRU-37/Berger–Seltzer formula from the
physical constants and the cited papers (electron-shell-corrected Bethe
formula with Møller kinematics), not a transcription of the ledger's prose or
of the code; §5's numeric evaluation is a standalone script
(`indep_bethe.py`, not importing `pyrite`) built from that reconstruction.
Agreement between this derivation and the ledger's own prose record is
therefore evidence the ledger's record is itself correct, in addition to
being evidence the code matches the source.

## 1. Intended quantity

The mean rate of kinetic-energy loss per unit path length of a fast electron
to atomic electronic excitation/ionization (collision stopping power,
excluding radiative loss), for a compound built from `TRANSPORT_ELEMENTS`
data by Bragg additivity, spliced onto the Joy–Luo non-relativistic form
(`electron-transport`) below each element's own crossover energy so that
transport uses the accurate high-energy law above ~1–10 keV while keeping the
already-validated shell-correction-free low-energy branch where
Berger–Seltzer is not applicable (NIST restricts ESTAR collision stopping to
$\geq 10$ keV for the same reason). Units at the physics boundary: $E$ in
keV, $\rho$ in g cm$^{-3}$, $A$ in g mol$^{-1}$, $J$ (mean excitation energy)
in keV, number densities $n_i$ in $\text{\AA}^{-3}$, output $dE/ds$ in
keV $\text{\AA}^{-1}$ (negative).

## 2. Independent derivation

### 2.1 Relativistic collision stopping power (Berger–Seltzer/ICRU-37)

The relativistic Bethe stopping-power formula for electrons (distinguished
from heavy charged particles by Møller, i.e. identical-particle, kinematics
for the maximum energy transfer, and by the resulting $F^-(\tau)$ correction
term) is, per ICRU 37 and Berger & Seltzer,

$$
-\frac{1}{\rho}\frac{dE}{dx}\bigg|_{\rm col}
= \frac{2\pi r_e^2 m_ec^2 N_A}{\beta^2}\,\frac{Z}{A}
\left[\ln\!\frac{\tau^2(\tau+2)}{2\,(I/m_ec^2)^2}
+ F^-(\tau) - \delta\right]
\qquad [\mathrm{MeV\,cm^2\,g^{-1}}],
$$

with $\tau = T/m_ec^2$ the kinetic energy in electron rest-mass units,
$\gamma = 1+\tau$, $\beta^2 = 1-1/\gamma^2$, $I$ the mean excitation energy,
$\delta$ the (density-effect) polarization correction, and

$$
F^-(\tau) = 1 - \beta^2 + \frac{\tau^2/8 - (2\tau+1)\ln 2}{(\tau+1)^2}.
$$

Shell corrections are dropped — this is the same restriction that keeps
NIST ESTAR's tabulated electron collision stopping power above 10 keV, and is
why the low-energy branch of transport stays with Joy–Luo (`electron-transport`).

**Prefactor.** $P \equiv 2\pi r_e^2 m_ec^2 N_A$ with the classical electron
radius $r_e = 2.8179403262\times10^{-13}\,\mathrm{cm}$, $m_ec^2 =
0.51099895\,\mathrm{MeV}$, $N_A = 6.02214076\times10^{23}\,\mathrm{mol^{-1}}$
evaluates to $0.153537\,\mathrm{MeV\,cm^2\,mol^{-1}}$ (this is exactly half
the heavy-particle Bethe constant $K = 4\pi r_e^2 m_ec^2N_A = 0.307075\,
\mathrm{MeV\,cm^2\,mol^{-1}}$, the factor of 2 tracing to the electron/electron
Møller cross section versus the electron/heavy-particle Rutherford one).
Converting to the transport module's working units (keV, $\text{\AA}$):
$1\,\mathrm{MeV\,cm^2\,mol^{-1}} = 10^3\,\mathrm{keV\,cm^2\,mol^{-1}}$, and
$1\,\mathrm{cm^2} = 1\,\mathrm{cm^3/cm} = 10^{-8}\,\mathrm{cm^3\,\text{\AA}^{-1}}$
(since $1\,\mathrm{cm}=10^8\,\text{\AA}$), so

$$
P = 0.153537\ \mathrm{MeV\,cm^2\,mol^{-1}}
= 1.53537\times10^{-6}\ \mathrm{keV\,\text{\AA}^{-1}\ per\ (mol\,cm^{-3})}.
$$

**Compound (Bragg) additivity.** With atomic number density $n$
[$\mathrm{cm^{-3}}$], $\rho = nA/N_A$ so $\rho Z/A = nZ/N_A$; with $n_i$ in
$\text{\AA}^{-3}$ ($n = 10^{24}n_i$),

$$
\frac{\rho Z}{A} = \frac{10^{24}n_i Z}{6.02214076\times10^{23}}
= \frac{n_i Z}{0.602214076},
$$

exactly the rewrite already established for Joy–Luo (`electron-transport`
§2.1), reused unchanged here because Bragg additivity is a property of the
number-density weighting, not of the particular stopping law. Hence, for a
compound with per-element coefficients $c_i \equiv n_i Z_i / 0.602214076$,

$$
\left(\frac{dE}{ds}\right)_{\rm BS}
= -\frac{1.535\times10^{-6}}{\beta^2}
\sum_i c_i\left[\ln\!\frac{\tau^2(\tau+2)}{2(I_i/m_ec^2)^2} + F^-(\tau) - \delta\right]
\quad [\mathrm{keV\,\text{\AA}^{-1}}].
$$

### 2.2 Limiting case: $\tau\to0$ and $F^-(0)$

At $\tau=0$: $\gamma=1$, $\beta^2=0$, so

$$
F^-(0) = 1 - 0 + \frac{0 - 1\cdot\ln2}{1} = 1-\ln2 \approx 0.30685.
$$

### 2.3 Non-relativistic reduction

For $\tau\ll1$: $\tau^2(\tau+2)\to2\tau^2$, so the log argument
$\to \tau^2/(I/m_ec^2)^2 = (E/I)^2$ and the bracket's leading log term
$\to 2\ln(E/I)$; $\beta^2\to2\tau=2E/m_ec^2$; $F^-\to1-\ln2$. So

$$
\left(\frac{dE}{ds}\right)_{\rm BS}
\;\xrightarrow{\tau\to0}\;
-\frac{P\,m_ec^2}{2E}\,\frac{\rho Z}{A}\Big[2\ln(E/I) + 1-\ln2\Big],
$$

the classical non-relativistic Bethe form for electrons (Møller max-transfer
convention), with residual $O(\tau)$ — the dropped $(\tau+2)/2\to1$ and
$\beta^2\to2\tau$ approximations both carry the next order.

### 2.4 Joy–Luo/Berger–Seltzer crossover

Both laws carry the same $\rho Z/A$ (equivalently $c_i$) factor, so their
ratio at any energy is independent of $\rho$ and $A$ — the crossover where
$|dE/ds|_{\rm JL} = |dE/ds|_{\rm BS}$ depends only on $(Z, J)$ per element and
must be solved numerically (the two closed forms differ in the *shape* of the
logarithm, not by a constant), by bisection on

$$
g(E) \equiv \left(\frac{dE}{ds}\right)_{\rm JL}(E;Z,A,J) - \left(\frac{dE}{ds}\right)_{\rm BS}(E;Z,A,J) = 0.
$$

Because stopping is additive over elements, splicing each element at its
*own* crossover keeps every term — and hence the compound sum — continuous
by construction, for any composition, with no per-material tuning; a single
global splice energy cannot be continuous for every element since the
crossover is strongly $Z$/$J$-dependent.

### 2.5 Spliced compound closed form

$$
\left(\frac{dE}{ds}\right)_{\rm spliced}
= -\frac{7.85\times10^{-4}}{E}\!\!\sum_{i:\,E<E_{\rm cross,i}}\!\! c_i
\ln\!\left[\frac{1.166(E+k_iJ_i)}{J_i}\right]
-\frac{1.535\times10^{-6}}{\beta^2}\!\!\sum_{i:\,E\geq E_{\rm cross,i}}\!\! c_i
\left[\ln\!\frac{\tau^2(\tau+2)}{2(I_i/m_ec^2)^2} + F^-(\tau)\right],
$$

with the Joy–Luo prefactor $7.85\times10^{-4}$, $k_i =
0.731+0.0688\log_{10}Z_i$, and $c_i$ as in §2.1, all already established
under `electron-transport`. This is a term-by-term reuse of both closed
forms with each element routed to whichever side of its own $E_{\rm
cross,i}$ the evaluation energy $E$ falls on — the sum stays a single Bragg
sum, just partitioned by element.

**Consistency of the two prefactors.** Both stopping laws must agree at
each element's own crossover by definition, and both reduce to the same
non-relativistic Bethe form as $E\to0$, so the two prefactors should very
nearly coincide once $\beta^2\to2\tau$ is substituted:
$P\,m_ec^2 = 1.53537\times10^{-6}\times 510.99895 = 7.847\times10^{-4}$
versus the Joy–Luo constant $7.85\times10^{-4}$ — a $0.04\%$ difference,
consistent with the Joy–Luo constant being an independently fit empirical
value (Joy & Luo 1989) rather than derived from $2\pi r_e^2m_ec^2N_A$, not a
divergence.

## 3. Comparison with the implementation

| quantity | derived above | implementation | verdict |
| --- | --- | --- | --- |
| BS prefactor | $1.5354\times10^{-6}$ keV/Å per (mol/cm³) | `_BS_PREFACTOR = 1.535e-6` | matches (4-sig-fig rounding of the exact constant) |
| $\tau,\gamma,\beta^2$ | $\tau=E/m_ec^2$, $\gamma=1+\tau$, $\beta^2=1-1/\gamma^2$ | `_dEds_bs_compound_scalar` | identical |
| $F^-(\tau)$ | $1-\beta^2+[\tau^2/8-(2\tau+1)\ln2]/\gamma^2$ | same expression, `f_minus` | identical |
| log term | $\ln[\tau^2(\tau+2)/(2I_{\rm rel}^2)]$ | `np.log(tau*tau*(tau+2.0)/(2.0*I_rel*I_rel))` | identical |
| compound coefficient | $c_i = n_iZ_i/0.602214076$ | `coeff_arr`/`coeff` built the same way (established under `electron-transport`) | identical |
| $\delta$ handling | subtracted inside the bracket | `... + f_minus - delta` | identical (every call site passes `delta=0.0`; a documented, measured-bound approximation, out of scope for this claim's `Code:` list) |
| crossover solve | bisection on $g(E)=0$, $(Z,J)$-only | `_bs_joy_luo_crossover_keV`: bisection from `lo=2J` to `hi=300`, 200 iterations | identical (bisection bracket choice differs from mine but both bracket the same root; see §5) |
| splice routing | per-element, own crossover, additive | `_dEds_spliced_compound_scalar`/`_dEds_spliced_packed_scalar`: `if E_i < E_cross_arr[i]: joy_luo_total += ... else: bs_total += ...`, then `-7.85e-4/E_i*joy_luo_total - _BS_PREFACTOR/beta_sq*bs_total` | identical |
| device/vector twins | same closed form, same routing | `transport/_jit_device.py::_dEds_packed` (device, float64 constants), `spectrum/lines.py::_spliced_stopping_magnitude_xp` (host/device array twin) | identical construction of $\tau,\gamma,\beta^2,F^-$ and the same two-branch sum, checked by inspection |
| `build_transport_energy_lut` | tabulate the closed form on a uniform energy grid, linear-interpolate at runtime | delegates to the same per-element/spliced kernels to fill `dEds`; no new physics, a numerical-method (grid+lerp) layer only | identical (nothing to re-derive beyond §2.1–2.5) |

No divergent factor, sign, exponent, unit, or convention found in any listed
`Code:` symbol.

## 4. Filters

- **Units.** $P\,\rho Z/(A\beta^2)$: MeV cm² mol⁻¹ × (mol cm⁻³, via $\rho Z/A$
  rewritten as $n_iZ_i/N_A$) → MeV cm⁻¹, and the cm→Å conversion gives
  keV Å⁻¹ (§2.1, reproduced independently to 5 significant figures against
  $2\pi r_e^2m_ec^2N_A$). Pass.
- **Limits.** $F^-(0)=1-\ln2$ exact (§2.2); non-relativistic reduction
  converges monotonically toward the classical Bethe form with residual
  $O(\tau)$, confirmed numerically in §5 (ratio $1.0266\to1.0024\to0.9999$ at
  $E=10,1,0.1$ keV); a single-element compound reduces to the elemental form
  by construction of the Bragg sum (same argument as `electron-transport`).
  Pass.
- **Signs/conventions.** $dE/ds<0$ throughout $1$–$300$ keV for every element
  tried (§5); the bracket that Berger–Seltzer needs positive for a physical
  sign has its own zero at $\sim0.86J$, well below the $\sim35J$ crossover
  region, so the splice never evaluates Berger–Seltzer where it would be
  unsigned. Pass.

## 5. Numeric evidence

Standalone script (`indep_bethe.py`, no `pyrite` import — literature
constants and the closed forms of §2.1/§2.5 only), carbon
($Z=6$, $A=12.011$, $J=78$ eV, the ICRU-37/PDG value already used by
`TRANSPORT_ELEMENTS["C"]`, read only to fix the reference material, not the
formula) unless noted:

| check | independent result | reference | verdict |
| --- | --- | --- | --- |
| $2\pi r_e^2m_ec^2N_A$ | $0.153537$ MeV cm²/mol | ledger's cited $0.1535$ | matches |
| $F^-(0)$ | $0.306853$ | $1-\ln2=0.306853$ | matches |
| non-rel. ratio at 10/1/0.1 keV | $1.0266$ / $1.0024$ / $0.9999$ | monotone $\to1$, ledger's "residual $O(\tau)$" | matches |
| carbon crossover | $2.662$ keV | ledger's stated range 2.66 keV (B) – 10.46 keV (Bi); carbon should fall inside | matches (inside range) |
| Bi crossover (approx. literature $I=823$ eV) | $10.42$ keV | ledger's $10.46$ keV | matches to $0.4\%$ (input-$I$-precision limited, not a formula divergence) |
| $P\,m_ec^2$ | $7.8457\times10^{-4}$ | Joy–Luo constant $7.85\times10^{-4}$, ledger's stated $0.08\%$ | matches |
| Joy–Luo/Berger–Seltzer ratio, carbon, 10/25/50/100/200/300 keV | $0.976/0.937/0.879/0.778/0.627/0.521$ | `tests/montecarlo/test_stopping_berger_seltzer.py::test_matches_the_documented_validity_ceiling_table` (`{10:0.98,25:0.94,50:0.88,100:0.78,200:0.63,300:0.52}`, tol $\pm0.005$), also reproduced in `electron-transport.md` §5 | matches within stated tolerance at every point (max deviation $0.0041$, at 10 keV) |

The validity-ceiling row is the one point where an existing repository
anchor (`test_stopping_berger_seltzer.py`) was consulted for its *reference
numbers*, per the methodology's allowance for `checks/`/`tests/` anchors in
step 4 — the numbers being checked were produced by the independent script,
not by importing the anchor's helpers.

## 6. Verdict

- **Claim**: `relativistic-bethe-stopping` —
  `montecarlo/transport.py::_dEds_bs_compound_scalar`,
  `::_dEds_spliced_compound_scalar`, `::_bs_joy_luo_crossover_keV`,
  `::spliced_stopping_keV_per_ang`, `::build_transport_energy_lut` (+ device/
  vector twins) — Berger–Seltzer/ICRU-37 relativistic collision stopping,
  spliced onto Joy–Luo per element at each element's own crossover.
- **Filters**: units **pass**; limits **pass**; signs/conventions **pass**.
- **Re-derivation**: **matches** — every closed form in §2 reproduces the
  cited ICRU-37/Berger–Seltzer and Joy–Luo forms to machine/analytic
  precision (prefactor to 5 significant figures from first principles,
  $F^-(0)$ exact, non-relativistic limit convergent, crossover bisection
  reproduces the ledger's stated range, validity-ceiling ratios within the
  anchor's own tolerance). No divergent factor, sign, exponent, unit, or
  convention found. Density-effect $\delta$ is correctly threaded as a
  parameter but is out of this claim's `Code:` scope (every call site
  currently passes `0.0`; the omission is measured, not assumed, elsewhere
  in the ledger row).
- **Verdict**: `rederived`.
- **Write-up**: `docs/validation/beam-transport/relativistic-bethe-stopping.md`.
- **Suggested ledger change**: status `filtered → rederived` for
  `relativistic-bethe-stopping` in `docs/validation/ledger-transport-background.md`;
  add a note that a fresh-context re-derivation (this write-up) confirms the
  prefactor, $F^-(\tau)$, non-relativistic limit, per-element crossover
  construction, and spliced closed form against ICRU-37/Berger–Seltzer and
  the Joy–Luo constants from first principles, with no divergence found; link
  `[write-up](beam-transport/relativistic-bethe-stopping.md)`. Human applies
  `signed-off`.
