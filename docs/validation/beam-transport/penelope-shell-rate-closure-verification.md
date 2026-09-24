# Verification: `penelope-shell-rate-closure`

Fresh-context independent verification (2026-09-24) of the claim
`penelope-shell-rate-closure`, anchored on
`src/pyrite/montecarlo/transport/shell_rates.py::{close_shell_rates,inner_shell_cutoff_eV,is_inner_shell,eedl_inner_cross_sections,adopted_stopping_cs,catalog_shell_rate_closure}`.
Verified at commit `a85d836b` on branch `issue-93-soft-hard-inelastic-transport`.
The uncommitted working-tree edit to `shell_gos.py` (a concurrent soft/hard
$W_c$ slice) is out of scope.

The derivation below (§1) was written from the PENELOPE-2024 manual
(NEA/MBDAV/R(2024)1), the ledger row, and the function signatures and
docstrings. The implementation body was read only afterwards (§3).

## 1. Independent derivation from the source

### 1.1 Inner/outer split (Eq. 2.112)

Eq. 2.112 defines

$$
E_c=\max\{50\ {\rm eV},\,U_{\max,{\rm out}}(Z_m)\},
$$

where $U_{\max,{\rm out}}(Z_m)$ is the largest ionisation energy of the O, P or Q
subshells of the heaviest element present. The manual states the inner/outer
rule in three places:

| Location | Wording | Inner means |
| --- | --- | --- |
| §2.6, text before Eq. 2.112 | "subshells with ionisation energies $U_i$ larger than $E_c$ as inner subshells. Shells with ionisation energies less than $E_c$, or beyond the N7 subshell, are regarded as outer" | $U_i>E_c$ |
| §7.1, footnote 1 | "Inner shells are K, L, M and N shells that have ionisation energies larger than the cutoff energy $E_c$ … less than $E_c$ are considered as outer shells" | $U_i>E_c$ |
| §3.2.6.1, after Eq. 3.141 | "inner shells [K to N7 shells with binding energies less than the cut-off energy $E_c$ …]" | $U_i<E_c$ (literal) |

The literal §3.2.6.1 reading is contradicted twice. It is also physically
inconsistent. $E_c$ is the lowest x-ray/Auger energy PENELOPE follows, so it is
the shells *above* $E_c$ whose vacancies must be produced. Taken literally, the
reading would substitute DWBA rates for the shallow shells whose relaxation is
never simulated and leave K shells on the GOS rate. The misprint reading is
therefore confirmed:

$$
\text{inner}\iff \text{shell}\in\{{\rm K},\dots,{\rm N7}\}\ \wedge\ U_i>E_c .
$$

The conduction band ($U=0$) and every O/P/Q shell are outer.

For the manual's own examples, the O1 ($5s$) ionisation energy gives
$E_c\approx114$ eV for Au and $\approx329$ eV for U, consistent with the rule.
For Ba, the largest O-shell energy (5s, about 40 eV) gives $E_c=50$ eV, but the
manual quotes 92 eV, which is the Ba 4d (N4/N5) energy. That quoted value does
not follow the stated rule. For the catalog materials ($Z_m\le42$) the heaviest
element is Mo, whose O shells ($5s$) lie far below 50 eV, so $E_c=50$ eV
whichever way the Ba anomaly is resolved.

### 1.2 Rate substitution (Eq. 3.141 and the §3.2.6.1 prose)

Write the raw GOS moments of oscillator $k$ at energy $E$ as

$$
\sigma^{(n)}_k(E;\delta_F)=\sigma^{(n)}_{{\rm dis},l,k}+\sigma^{(n)}_{{\rm dis},t,k}(\delta_F)+\sigma^{(n)}_{{\rm clo},k},\qquad n=0,1,2,
$$

from Eqs. 3.103–3.106. Only the transverse term (Eq. 3.105) contains $\delta_F$,
through the bracket $\ln\gamma^2-\beta^2-\delta_F$. The longitudinal term (Eq. 3.104)
and the close term (Eq. 3.106) do not contain it. So

$$
\sigma^{(0)}_k(E;0)=\sigma^{(0)}_k(E;\delta_F)
+\Big[\sigma^{(0)}_{{\rm dis},t,k}\Big]_{\delta_F\to0}-\sigma^{(0)}_{{\rm dis},t,k}(\delta_F).
$$

The prose after Eq. 3.141 prescribes three steps:

1. The total cross section of each inner shell becomes the DWBA value
   $\sigma^{(\pm)}_{{\rm si},i}$. Here the owner baseline substitutes EEDL.
2. That value is reduced by the density-effect ratio
   $\rho_i=\sigma^{(0)}_i(E;\delta_F)/\sigma^{(0)}_i(E;0)\le1$.
3. The energy-loss and angular PDFs of the GOS model are kept.

Keeping the PDF $p_i(W)=\sigma_i^{-1}d\sigma_i/dW$ fixed while changing the
total from $\sigma^{(0)}_i(\delta_F)$ to $\tilde\sigma_i=\rho_i\sigma_{{\rm si},i}$
means that every moment scales by one factor:

$$
\tilde\sigma^{(n)}_i=s_i\,\sigma^{(n)}_i(\delta_F),\qquad
s_i=\frac{\rho_i\,\sigma_{{\rm si},i}}{\sigma^{(0)}_i(\delta_F)}
=\frac{\sigma_{{\rm si},i}}{\sigma^{(0)}_i(0)} .
$$

The second form shows that $s_i$ does not depend on $\delta_F$ explicitly. It
is the EEDL/GOS ratio with the density effect removed from the GOS.

### 1.3 Stopping closure (Eq. 3.142)

Outer oscillators $j$ are multiplied by one energy-dependent factor
$\mathcal N(E)$. The manual notes this is formally the same as $f_j\to f_j\mathcal N$,
so all three moments scale by $\mathcal N$ and the PDFs are again kept. The
closure condition "adopted stopping power reproduced exactly" is a condition on
$\sigma^{(1)}$:

$$
S_{\rm adopt}(E)=\sum_i s_i\sigma^{(1)}_i+\mathcal N(E)\sum_j\sigma^{(1)}_j
\quad\Longrightarrow\quad
\boxed{\mathcal N(E)=\frac{S_{\rm adopt}(E)-\sum_i s_i\sigma^{(1)}_i}{\sum_j\sigma^{(1)}_j}} .
$$

In PENELOPE, $S_{\rm adopt}$ is the GOS stopping $\sum_k\sigma^{(1)}_k$ itself.
Here it is the corrected SBETHE `stp.dat` collision stopping, converted to
eV cm² per formula unit. This is a documented owner deviation. It changes only
the numerator of $\mathcal N$, not the structure of Eq. 3.142.

### 1.4 Units, sign and limiting cases

- **Units.** $\sigma_{{\rm si},i}$ is in cm² per atom and becomes cm² per
  formula unit after multiplying by the stoichiometric count $n_Z$ of element
  $Z$. $\rho_i$, $s_i$ and $\mathcal N$ are dimensionless. $\sigma^{(0,1,2)}$
  are in cm², eV cm² and eV² cm². $S_{\rm adopt}$ in eV cm² per formula unit is
  $S_{\rm mass}\,M/N_A$ ($S_{\rm mass}$ in eV cm²/g, $M$ the formula mass in
  g/mol). Eq. 3.142 is homogeneous in eV cm².
- **Sign.** $\mathcal N>0$ requires $S_{\rm adopt}>\sum_is_i\sigma^{(1)}_i$, which
  is a physical requirement. A non-positive $\mathcal N$ must fail closed.
  (Before computing, I expected the inner shells to carry only a small
  fraction of the stopping. §4 shows that was wrong: with $E_c=50$ eV the
  Si/S L shells are inner, and the inner shells carry 18–62%.) $s_i\ge0$ requires
  $\sigma_{{\rm si},i}\ge0$.
- **$\delta_F=0$** (below the density-effect onset): $\rho_i=1$ and
  $s_i=\sigma_{{\rm si},i}/\sigma^{(0)}_i$.
- **No inner shells:** $\mathcal N=S_{\rm adopt}/\sum_k\sigma^{(1)}_k$.
- **GOS-consistent inputs** ($\sigma_{{\rm si},i}=\sigma^{(0)}_i(0)$ and
  $S_{\rm adopt}=\sum_k\sigma^{(1)}_k$): $s_i=1$ and $\mathcal N=1$, so the raw
  moments are returned. This is PENELOPE's own case when the inner rates happen
  to equal the GOS rates.
- **Below threshold** ($E\le U_i$): the GOS gives $\sigma^{(n)}_i=0$ and EEDL gives
  $\sigma_{{\rm si},i}=0$, so shell $i$ is closed. Take $s_i=0$ rather than
  $0/0$. EEDL and SBETHE thresholds differ slightly, so there is an energy
  window where one is positive and the other is zero. That window must be
  handled explicitly, either by failing closed or by a documented convention.

### 1.5 What the implementation must therefore do

1. Select inner oscillators as bound, K–N, $U_k>E_c$, with $E_c$ from Eq. 2.112.
2. Build $\sigma_{{\rm si},i}$ per formula unit as $n_Z$ times the EEDL subshell
   sum joined to the SBETHE shell.
3. Compute $\rho_i$ from the transverse term alone, with $\delta_F\to0$ entering
   nowhere else ($W_k$, $a$ and $N$ must keep the true $\Omega_p$).
4. Scale all three moments of $i$ by $s_i=\rho_i\sigma_{{\rm si},i}/\sigma^{(0)}_i(\delta_F)$.
5. Scale all three moments of every other oscillator by $\mathcal N$ from the boxed formula.
6. Require $\mathcal N>0$ and finite.

## 2. Cheap filters

| Filter | Result |
| --- | --- |
| Units | Pass. `inner_cross_sections_cm2` are in cm² per formula unit (`composition[z]` is the atom count per formula unit: Si $\{14{:}1\}$, SiO₂ $\{14{:}1,8{:}2\}$, MoS₂ $\{42{:}1,16{:}2\}$). `stp_cs` (eV cm² per molecule) equals mass stopping $\times10^6\,M_{\rm hdr}/N_A$ to $\le6\times10^{-6}$, which is vendor print precision. $\rho_i$, $s_i$ and $\mathcal N$ are dimensionless. |
| Sign | Pass. $\mathcal N\in[0.759,1.281]$ over Si/SiO₂/MoS₂ at 1 keV–1 MeV. $\mathcal N\le0$, non-finite $\mathcal N$ and negative $\sigma_{\rm si}$ raise. All closed per-shell moments are $\ge0$. |
| $\delta_F=0$ | Pass. MoS₂ has $\delta_F=0$ at 1–10 keV, and there $\rho_i=1$ exactly. |
| No inner shells | Pass. $\mathcal N=S/\sigma^{(1)}_{\rm GOS}$ (test fixture). |
| GOS-consistent inputs | Pass. $s_i=\mathcal N=1$ and the raw moments return (test fixture). |
| $E\le U_i$ | Pass. EEDL and SBETHE $U$ are identical for every catalog inner shell (§4.3), so a closed shell has $\sigma_{\rm si}=\sigma^{(0)}_{\rm GOS}=0$ and $s_i=0$. A positive $\sigma_{\rm si}$ with zero GOS raises. |
| Convention: inner/outer | Pass. $U_i>E_c$, K–N only, the band is never inner; §1.1 confirms the misprint reading. |

## 3. Symbolic comparison with the implementation

The implementation body (`close_shell_rates`, `inner_shell_cutoff_eV`,
`is_inner_shell`, `eedl_inner_cross_sections`, `adopted_stopping_cs` at
`a85d836b`) was read after §1 was written.

| §1 quantity | Code | Agreement |
| --- | --- | --- |
| $E_c=\max\{50,U_{\max,{\rm OPQ}}(Z_m)\}$ | `max([threshold_eV, *outer])` over `label[0] in "OPQ"` of `max(composition)` | exact |
| inner: K–N, $U>E_c$, bound | `atomic_number > 0 and label[0] in "KLMN" and U > cutoff` | exact |
| $\sigma_{{\rm si},i}=n_Z\sum_s\sigma_s$ | `composition[z] * sum(sigma[label] for label in joined[...])` | exact |
| $\rho_i=\sigma^{(0)}_i(\delta_F)/\sigma^{(0)}_i(0)$ | `sigma0[k] / sigma0_bare[k]`, bare = `replace(material, plasma_energy_eV=0.0)` | exact; the runtime guard checks that $\Omega_p=0$ changes only the transverse channel ($\delta_F=0$, longitudinal and close bit-identical) |
| $s_i=\rho_i\sigma_{{\rm si},i}/\sigma^{(0)}_i(\delta_F)$ | `adopted[k] / sigma0[k]` | exact |
| $\mathcal N=(S-\sum_is_i\sigma^{(1)}_i)/\sum_j\sigma^{(1)}_j$ | `(stopping - inner_stopping) / outer_stopping` | exact |
| all three moments scaled | `distant_longitudinal`, `distant_transverse`, `close` $\times$ `scale[:, None]` | exact |
| $S_{\rm adopt}$ | log-log interpolation of `stopping_cs_eV_cm2`, raising outside the grid | exact |

The $\Omega_p=0$ route for $\sigma^{(0)}(0)$ is legitimate. In the committed
`shell_gos_moments`, $\Omega_p$ enters only through `density_effect_correction`:
$W_k$ and $a$ are stored on the oscillators, and $N$ is used only in
`path_moments`. The guard enforces this at run time.

## 4. Numeric comparison

A scratch script reimplemented every channel of Eqs. 3.94–3.106 by adaptive
quadrature (triangle $p_{\rm dis}$, Eq. 3.83 $Q_-$, own $\delta_F$ root from
Eqs. 3.70–3.72, Møller $F^{(-)}(E+U_k,W)$ from $U_k$). It then computed the
closure with $s_i=\sigma_{{\rm si},i}/\sigma^{(0)}_i(0)$, the algebraically
equivalent form from §1.2. $S$ was taken from the mass column with xraydb
molar masses, and $\sigma_{\rm si}$ from lin-lin interpolation of the EEDL
tables with an independent label join. The EEDL file declares ENDF law 2
(lin-lin) and the loader enforces it.

### 4.1 Agreement

| Quantity | Materials, energies | Max relative difference |
| --- | --- | --- |
| raw GOS channels $\sigma^{(0,1,2)}_{l,t,c}$ | Si, SiO₂, MoS₂; 1 keV–1 MeV | $5.9\times10^{-9}$ (quadrature), identical zero pattern |
| $\delta_F$ | same | identical to printed precision |
| inner selection | same | identical sets (Si: K, L1–L3; SiO₂: + O K; MoS₂: Mo K–M5, N1; S K, L1–L3) |
| inner $s_i$ | same | $1.4\times10^{-10}$ |
| $\rho_i$ | same | $4\times10^{-12}$ |
| $S_{\rm adopt}$ | same | $\le6\times10^{-5}$ (xraydb vs SBETHE atomic weights); $\le3\times10^{-6}$ with the header $M$ |
| $\mathcal N$ | same | $\le1.1\times10^{-4}$, entirely from the $S$ difference above |
| closure $\sum\tilde\sigma^{(1)}=S$ | same | $<10^{-13}$ against the code's own $S$ |

### 4.2 Closure diagnostics (independent values)

| material | 1 keV | 2 keV | 5 keV | 10 keV | 20 keV | 50 keV | 100 keV | 1 MeV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Si $\mathcal N$ | 0.965 | 1.060 | 1.117 | 1.141 | 1.112 | 1.108 | 1.095 | 1.056 |
| SiO₂ $\mathcal N$ | 0.759 | 0.790 | 0.831 | 0.856 | 0.844 | 0.832 | 0.819 | 0.791 |
| MoS₂ $\mathcal N$ | 1.037 | 1.032 | 1.107 | 1.187 | 1.227 | 1.270 | 1.281 | 1.270 |
| Si inner $S$ fraction | 0.37 | 0.40 | 0.45 | 0.48 | 0.53 | 0.55 | 0.57 | 0.62 |
| SiO₂ inner $S$ fraction | 0.19 | 0.25 | 0.32 | 0.35 | 0.39 | 0.43 | 0.45 | 0.50 |
| MoS₂ inner $S$ fraction | 0.18 | 0.26 | 0.32 | 0.34 | 0.38 | 0.40 | 0.42 | 0.48 |

These reproduce the implementation derivation's $\mathcal N$ table to its
three printed digits. The closed/raw total $\sigma^{(0)}$ spans −23.2% to
+23.4%, and $\sigma^{(2)}$ spans −33.6% to +13.1%, matching the ledger note.
$\min\rho_i$ is 0.951 (Si), 0.909 (SiO₂) and 0.936 (MoS₂) at 1 MeV.
Just above the Si K (1844.0001 and 1845 eV), S K (2476.5 eV) and Mo L1
(2872.5 eV) thresholds the closure stays finite, non-negative and continuous
($\mathcal N=1.039$, 1.039, 1.050).

### 4.3 Threshold window

For every catalog inner shell the SBETHE $U_k$ equals the EEDL binding energy
exactly (Si K 1844, Si L1 154, Si L2/L3 104, O K 538, S K 2476, S L1–L3
232/170/168, Mo K 20006, Mo L1–L3 2872/2632/2527, Mo M1–M5 511/416/399/237/234,
Mo N1 68 eV). The fail-closed branch "positive $\sigma_{\rm si}$ where the GOS
vanishes" therefore cannot trigger for these materials. For other materials
it would raise, rather than silently mis-scale, in any window where the EEDL
threshold lies below the SBETHE $U_k$.

### 4.4 Manual $E_c$ examples

Using the SBETHE shells, the code gives $E_c=114$ eV for Au (O1) and 329 eV
for U (O1), matching the manual. For Ba it gives 50 eV (largest O shell O1 =
31 eV), while the manual prints 92 eV, which is the Ba N5 energy in the same
data. The code follows the stated rule. This is correct, and the anomaly is
correctly recorded in the ledger note.

## 5. Findings

1. **No discrepancy in the claimed physics.** Eqs. 2.112, 3.141 and 3.142 and
   the §3.2.6.1 prescriptions (keep the GOS PDFs, $\delta_F$ ratio, one common
   $\mathcal N$) are implemented as derived. The "less than $E_c$" reading of
   §3.2.6.1 is a misprint, contradicted by §2.6 and the §7.1 footnote and by
   the purpose of $E_c$.
2. **Interpretation note (no action).** "The ratio of the cross sections
   obtained from the GOS model with and without $\delta_F$" is read per shell,
   $\sigma^{(0)}_i(\delta_F)/\sigma^{(0)}_i(0)$. A material-total ratio would be
   the other reading. The per-shell reading is the natural one for a per-shell
   substitution. The two differ by $\le0.7\%$ at 100 keV, but by up to 4.5%
   (Si), 8.9% (SiO₂) and 6.1% (MoS₂) at 1 MeV. The material-total ratio is
   0.996–0.998 there, because the outer oscillators dominate $\sigma^{(0)}$.
   The choice therefore matters only in the MeV range.
3. **Scope (documented).** Electron only: the positron $F^{(+)}$ and
   $\sigma^{(+)}_{\rm si}$ path is not covered. The EABS restriction (§3.2.6) is
   not applied. Both are stated in the ledger assumptions.
4. **Minor, marker hygiene.** `shell_rates.py::is_inner_shell` is listed in
   the ledger `Code` field but has no source/limiting-case docstring or
   `Validation:` marker (the other five symbols have one). Severity: low.
   Suggested fix (for the author): add a one-line
   `Validation: penelope-shell-rate-closure` and the §2.6/§7.1 source to its
   docstring.
5. **Test independence (low).** The anchor tests' `_stp` and `_eedl` helpers
   use the same catalog column and EEDL loader as the code. That is fine for a
   closure identity. The independent mass-column and molar-mass cross-check
   of §4.1 is not pinned. Optional: pin
   `stp_cs ≈ stopping_MeV_cm2_per_g·1e6·M/N_A` to $10^{-5}$.
6. **Physics caveat (not a code defect).** With $E_c=50$ eV, the shallow Si
   and S L shells and Mo N1 are "inner", and inner shells carry 18–62% of the
   adopted stopping. $\mathcal N$ therefore absorbs sizeable EEDL-versus-GOS
   differences (for example, Mo N1 EEDL/GOS ≈ 0.3), and the closure changes
   the IMFP by up to ±23%. This is what Eq. 3.142 prescribes, and the ledger
   notes already state that no IMFP or straggling measurement has validated it.

## 6. Verdict

- **Claim**: `penelope-shell-rate-closure` —
  `src/pyrite/montecarlo/transport/shell_rates.py::{close_shell_rates,inner_shell_cutoff_eV,is_inner_shell,eedl_inner_cross_sections,adopted_stopping_cs,catalog_shell_rate_closure}` —
  PENELOPE-2024 Eqs. 2.112, 3.141–3.142 (§3.2.6.1)
- **Filters**: units pass; limits pass; signs/conventions pass (misprint reading confirmed)
- **Re-derivation**: `matches`. No divergent term. $\mathcal N$ agrees to
  $10^{-4}$, limited only by the reference molar masses, and $s_i$, $\rho_i$ to
  $10^{-10}$.
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/beam-transport/penelope-shell-rate-closure-verification.md`
- **Suggested ledger change**: status `unverified` → `rederived`. Add this
  verification link to the Anchor. Optionally add the `is_inner_shell` marker
  (finding 4). Only a human may mark it `signed-off`.

## Commands

Run in `/tmp/pyrite-issue-93` with
`PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache`:

- `pdftotext -layout` of the PENELOPE-2024 PDF (§2.6 Eq. 2.112, §3.2.1–3.2.3,
  §3.2.6–3.2.6.1, §7.1 footnote);
- `uv run python` running the independent quadrature/closure script (session
  scratchpad, not committed);
- `uv run pyrite-dev test tests/montecarlo/test_shell_rates.py tests/montecarlo/test_shell_gos.py -q`
  (all pass; 3 strict xfails as designed).

The concurrent uncommitted `shell_gos.py` edit (windowed moments) was present
during the numeric runs. It routes `shell_gos_moments` through
`_windowed_moments(…, 0, ∞)`, which is algebraically identical to `HEAD`. The
independent quadrature reproduces the pinned `HEAD` ratios (see the
`penelope-shell-gos-moments` verification, Eq. 3.96 re-check).
