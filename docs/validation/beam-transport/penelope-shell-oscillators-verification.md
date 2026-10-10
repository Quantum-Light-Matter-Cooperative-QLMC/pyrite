# PENELOPE shell oscillators: independent verification

Validation: `penelope-shell-oscillators`. This is the fresh-context verifier
record for branch `issue-93-soft-hard-inelastic-transport` at `10dc6aea`. It
does not change the ledger status. Only a human may mark the claim
`signed-off`.

## Summary

- **Claim**: `penelope-shell-oscillators`, covering
  `src/pyrite/montecarlo/transport/shell_oscillators.py::{load_conduction_bands,build_shell_oscillators}`
  and `src/pyrite/data/conduction_band.toml`. Source: PENELOPE-2024 §3.2.1,
  Eqs. 3.51 and 3.60–3.65.
- **Filters**: units pass, limits pass, signs and conventions pass.
- **Re-derivation**: `matches`. Every oscillator label, strength, $U_k$ and
  $W_k$ agrees to $\le 2.4\times10^{-15}$ relative for all three materials in
  both the measured and default modes. $a$ agrees to 10 digits.
- **Verdict**: `rederived`.
- **Suggested ledger change**: set the status to `rederived` and add this
  record to **Anchor**. Reconcile the "within 10%" check in the ledger with
  the "within 7%" statement in the derivation (see item 4). A human applies
  the change.

| item | subject | verdict |
| --- | --- | --- |
| 1 | Eqs. 3.62–3.64, $\Omega_p$, formula-unit $Z$, measured $f_{cb}$/$W_{cb}$ | verified |
| 2 | measured conduction-band data | verified with caveats |
| 3 | shell-consumption rule and assignments | verified |
| 4 | numbers | verified; one wording discrepancy (SiO₂ is 7.002%, not "within 7%") |
| 5 | implementation against derivation, fail-closed paths | verified with caveats (hardening only) |

## Sources consulted

- PENELOPE-2024, NEA/MBDAV/R(2024)1 (local Zotero PDF, `pdftotext -layout`).
  I read §3.2 pp. 114–119, which covers Eqs. 3.50–3.65, and the Fig. 3.11
  discussion on pp. 128–129.
- The pinned SBETHE `sdbase/pdatconf.p14`. I recomputed its SHA-256 as
  `cd239554…62d2` and parsed it with my own whitespace parser. The occupations
  sum to $Z$ for O, Si, S and Mo.
- The Ding-lab ELF database pages
  `https://micro.ustc.edu.cn/database/ELF/Si.html` (Yang et al., PRB 100,
  245209, 2019) and `.../SiO2.html` (Da et al., JAP 113, 214303, 2013). I
  fetched them with `curl` and parsed them with my own `html.parser`
  subclass. The Si page has 7 tables and the SiO₂ page has 2. On each page
  the first table holds oscillator parameters $(a_i,\omega_{pi},\gamma_i)$,
  and the REELS-derived ELF is the 6-column table (Si: 2000 rows at 0.1 eV
  up to 200 eV; SiO₂: 800 rows up to 80 eV). The Si page also carries
  Palik crystal and doped ELF tables on a coarser grid.
- Saito et al., Microscopy 74, 117 (2025), PMC11957257 (HTML).
- Moynihan et al., arXiv:2012.09924 (fetched via `export.arxiv.org`,
  `pdftotext`).

## Filters

- **Units.** $\Omega_p$, $U_k$, $W_k$, $W_{cb}$ and $I$ are all in eV. $f$
  and $Z$ are electron counts per formula unit. $a$ is dimensionless. Inside
  the square root of Eq. 3.63 both terms are in eV², and Eq. 3.64 compares
  $\ln$(eV) on both sides. Pass.
- **Limits.** For one shell with $f=Z$ and no conduction band, Eq. 3.64
  gives $Z\ln I=Z\ln W$, so $W=I$ for any $a$. I checked this with my own
  code: H with a 0 eV threshold and $I=19.2$ gives $W=19.2$. As
  $a\to\infty$ the right-hand side of Eq. 3.64 grows without bound. As
  $a\to0$ it tends to the zero-binding floor
  $f_{cb}\ln W_{cb}+\sum f_k\ln\sqrt{2f_k\Omega_p^2/3Z}$, so a root exists
  exactly when $Z\ln I$ lies above that floor. Pass.
- **Signs and conventions.** The right-hand side of Eq. 3.64 is strictly
  increasing in $a$ because $U_k>0$, so the root is unique. $U_{cb}=0$ and
  $Q_{cb}=W_{cb}$. Pass.

## Item 1: source equations

From the manual text (pp. 117–118):

$$
W_{cb}=\sqrt{4\pi N f_{cb}\hbar^2e^2/m_e}=\sqrt{\frac{f_{cb}}{Z}}\,\Omega_p\quad(3.62),\qquad
W_k=\sqrt{(aU_k)^2+\frac{2}{3}\frac{f_k}{Z}\Omega_p^2}\quad(3.63),
$$

$$
Z\ln I=f_{cb}\ln W_{cb}+\sum_k f_k\ln\sqrt{(aU_k)^2+\tfrac{2}{3}\tfrac{f_k}{Z}\Omega_p^2}\quad(3.64),
\qquad \Omega_p^2=4\pi NZ\hbar^2e^2/m_e\quad(3.51).
$$

- **$\Omega_p$ is all-electron.** The manual describes it as "the plasma
  energy corresponding to the total electron density in the material,
  Eq. (3.51)". Verified.
- **Formula-unit $Z$.** The manual says oscillators "may pertain either to
  atoms or molecules", with $Z_M=xZ_X+yZ_Y$ (Eq. 3.65). The formula-unit
  choice cancels:
  $f_k\Omega_p^2/Z=4\pi N f_k\hbar^2e^2/m_e$ depends only on the electron
  density of shell $k$. Using the formula-unit $Z$ together with the
  all-electron $\Omega_p$ is therefore correct, and it is the only
  consistent choice. Verified.
- **Measured $f_{cb}$ and $W_{cb}$.** The manual says they "should be
  identified with the effective number of electrons (per atom or molecule)
  that participate in plasmon excitations and the plasmon energy … estimated,
  e.g., from electron energy-loss spectra or from measured optical data. When
  this information is not available, we will simply fix … [Eq. 3.62]". So
  the measured route is the preferred one, and a measured $W_{cb}$ replaces
  Eq. 3.62. The manual's aluminium example on p. 129 does exactly this: it
  sets $W_{cb}=15$ eV (measured) instead of Eq. 3.62's 15.8 eV and keeps
  $f_{cb}=3$. Verified.
- **Meaning of $f_{cb}$.** The manual does not require $f_{cb}$ to equal
  the chemical valence. The default is "electrons with $U<$ say 15 eV", and
  gold gets $f_{cb}=11$. The manual does cite Sternheimer's use of the
  lowest chemical valence as a similar approach. A valence count is
  therefore a defensible estimate of the "effective number" but not a
  quantity the manual prescribes.

## Item 2: measured conduction-band data

I read the tables independently.

| material | quantity | value found |
| --- | --- | --- |
| Si (Yang 2019 REELS ELF) | maximum | 3.75854 at 16.7 eV; within 1% of the maximum over 16.5–16.9 eV; FWHM 14.6–18.6 eV; $\varepsilon_1$ zero crossing 16.2 eV |
| Si (Palik crystal, same page) | maximum | 3.908 at 17.0 eV (coarse grid) |
| SiO₂ (Da 2013 REELS ELF) | maximum | 1.00648 at 23.6 eV; within 1% of the maximum over 23.3–24.4 eV; FWHM 16.5–30.4 eV; no $\varepsilon_1$ zero crossing |
| SiO₂ (Saito 2025) | q = 0 peak | "broad peak at 22 eV … could correspond to a volume plasmon"; about 24 eV for $q>0.4$ Å⁻¹; the authors note that surface loss lowers the $q=0$ peak. Their free-electron count is 4 + 2×6 = 16 valence electrons. |
| MoS₂ (Moynihan 2020) | plasmon | "π + σ bulk plasmon mode is seen at around 23 eV"; π bulk plasmon at 8.6 eV (thick) and about 8.3 eV |

- **Valence counts.** Si $3s^23p^2$ gives 4. SiO₂ gives
  $4+2\times6=16$ from O $2s^22p^4$. MoS₂ gives Mo 6 (4d+5s) plus
  $2\times6$ from S $3s^23p^4$, so 18. All correct.
- **ELF maximum as the plasmon energy.** Defensible. It is the conventional
  EELS "plasmon energy". For Si it agrees with Palik (17.0 eV) and lies
  0.5 eV above the $\varepsilon_1=0$ crossing. The δ-oscillator carries no
  width, however, and the log-mean energy of the valence ELF (weight
  $W\,{\rm ELF}$) is higher: 18.6 eV for Si up to 99 eV and 34 eV for SiO₂
  up to 80 eV. Because $a$ is re-solved, $I$ and stopping are preserved,
  but the choice moves the IMFP, as the manual notes on p. 129. This is a
  caveat, not an error.
- **Caveat: sample density versus catalog density for SiO₂.** The Da et al.
  page states a density of 2.65 g/cm³. The catalog `sio2` density is
  2.1999 g/cm³. Computed independently, $\Omega_p$ is 33.13 eV at 2.65
  g/cm³ (Eq. 3.62 with $f=16$ gives 24.20 eV) and 30.20 eV at 2.20 g/cm³
  (22.06 eV). The measured 23.6 eV is therefore within 2.5% of the
  free-electron value at the sample density but 7.0% above it at the
  catalog density. Saito's amorphous-SiO₂ 22 eV value is closer to the
  catalog density. I recommend either documenting this mismatch or
  choosing a value for a density-matched sample.
- **Caveat: partial f-sum.** A partial f-sum on the Da ELF table gives
  $n_{\rm eff}\approx11$ (ELF) or $\approx8$ ($\varepsilon_2$) at 80 eV,
  where the table ends. Si saturates near 3.5 below the $L_{2,3}$ edge. So
  $f_{cb}=16$ for SiO₂ is an upper-bound valence count, not a measured
  effective number. It is acceptable under the manual's wording, but the
  O 2s electrons contribute only partly to the 23.6 eV feature.
- **Caveat: MoS₂.** The MoS₂ value is a stated approximation ("around
  23 eV") read from spectra, not from a tabulated ELF. The 8.6 eV π plasmon
  strength is folded into the single oscillator, which the derivation
  already says.
- **Observation: Mo configuration in `pdatconf.p14`.** The file lists Mo
  as N4 ($4d_{3/2}$, occupation 4) plus O1 ($5s$, occupation 2), both at
  8.317 eV. This is not the free-atom $4d^55s^1$ configuration. Because
  both shells have equal $U$ and are consumed together, the valence total
  of 6 is unaffected. The labels only look odd.

Verdict: verified with caveats.

## Item 3: shell consumption

My own parser sorted shells by $U$ across the formula unit, with strengths
multiplied by atom counts.

| material | consumed shells (cumulative $f$) | next bound $U$ |
| --- | --- | --- |
| Si | M2 8.151 (2), M1 13.46 (4) | L2/L3 104 eV |
| SiO₂ | Si M2 (2), Si M1 (4), O L2+L3 13.62 (12), O L1 28.48 (16) | Si L2/L3 104 eV |
| MoS₂ | Mo N4+O1 8.317 (6), S M2+M3 10.36 (14), S M1 20.2 (18) | Mo N3 42 eV |

The assignments and gaps match the claim. No boundary splits a shell or an
equal-$U$ group. For comparison, the 15 eV default gives $f_{cb}=4$, 12 and
14. Verified.

## Item 4: numbers

My construction takes catalog $I$ and `plasma_energy_eV` as inputs. The
oscillator build and the bisection root (300 iterations) are my own code.

| material | $I$ (eV) | $\Omega_p$ (eV) | Eq. 3.62 (eV) | measured/Eq. 3.62 | $a$ (measured) | $a$ (default) | default $f_{cb}$, $W_{cb}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| silicon | 173.000 | 31.0498 | 16.597 | 1.0062 | 2.202355 | 2.207832 | 4, 16.597 |
| sio2 | 125.663 | 30.2011 | 22.056 | 1.0700 | 3.202500 | 2.726436 | 12, 19.101 |
| mos2 | 292.726 | 43.8116 | 21.608 | 1.0644 | 1.779698 | 1.801969 | 14, 19.056 |

- **Independent $\Omega_p$.** I computed
  $\Omega_p=\hbar\sqrt{n_ee^2/\varepsilon_0m_e}$ from CODATA constants and
  standard atomic weights. Si at 2.329 g/cm³ gives 31.048 eV, consistent
  with the helper's 31.050 eV at 2.3292 g/cm³. SiO₂ at 2.20 g/cm³ gives
  30.201 eV. MoS₂ gives 44.07 eV at 5.06 g/cm³, which scales to 43.81 eV at
  the catalog's 5.0002 g/cm³.
- **Closure.** The dipole sum closes exactly. Eq. 3.64 closes to $2\times10^{-16}$.
- **Discrepancy: wording.** SiO₂ is $23.6/22.0557=1.070016$, which is
  7.002% and not "within 7%". The statement is in
  `docs/validation/beam-transport/penelope-shell-oscillators.md:53`. The
  ledger check (`docs/validation/ledger-transport-background.md:116`) and
  `tests/montecarlo/test_shell_oscillators.py:109` use 10%. This is
  cosmetic; it does not affect the physics.
- **Caveat: Bragg-additive $I$ for SiO₂.** SiO₂'s $I=125.66$ eV is
  Bragg-additive with O at 95 eV (the gas-phase value). ICRU-37's tabulated
  SiO₂ value is 139.2 eV. The manual permits Bragg additivity only "when the
  value of the mean excitation energy of the compound is not known". This
  belongs to the SBETHE material-input claim, not to this one, but it
  shifts $a$.

Verdict: verified (with the wording discrepancy above).

## Item 5: implementation against derivation

I read the implementation only after completing items 1–4.

- **Symbolic comparison.** `shell_oscillators.py:151` implements Eq. 3.62
  with `total` as the formula-unit $Z$. Lines 156 and 170 implement
  Eq. 3.63. The `closure` function at lines 157–160 implements Eq. 3.64.
  Threshold strictness (`<` at line 127) matches the manual's "less than".
  The sort key $(U,Z,\text{designator})$ at line 123 is deterministic. The
  equal-$U$ check at line 140 is sufficient because the list is sorted, so
  any equal-$U$ remainder sits at `rest[0]`.
- **Numeric comparison.** With the SBETHE `load_atomic_shells` shells, all
  6 cases (3 materials, measured and default) give identical oscillator
  counts (5/5, 6/7, 17/18), labels, strengths and $U$. $W$ agrees to
  $\le2.4\times10^{-15}$ relative and $a$ to 10 digits.
- **Fail-closed paths, all confirmed to raise ValueError:**
  - split shell (Mo $f=2$) and split equal-$U$ group (Mo $f=4$);
  - $f_{cb}>Z$;
  - $f_{cb}=Z$, which leaves no bound shell;
  - default mode with every shell below threshold (H);
  - formula mismatch;
  - $I$ below the zero-binding floor, including $W_{cb}=\infty$;
  - non-positive or NaN $I$ and $\Omega_p$;
  - negative composition;
  - missing formula, doi, unknown element or NaN in the TOML.
- **Default $f_{cb}=0$.** Ne yields no conduction band, as intended.
- **Other probes that pass:**
  - integer and float formula counts compare equal;
  - `resonance_eV=None` with a measured $f$ correctly falls back to Eq. 3.62
    with that $f$ (SiO₂: 22.056 eV).
- **Hardening findings.** None of these affects the three packaged
  materials.
  1. `shell_oscillators.py:77` accepts `resonance_eV = inf`. The failure
     then surfaces later as a misleading "zero-binding" error. Use a
     finiteness check.
  2. Lines 130–142: a directly constructed `ConductionBand(0.0, …)` silently
     yields no conduction band while `conduction_source` still names the
     measured source. The loader blocks this case, but
     `build_shell_oscillators` does not.
  3. Line 169: nothing bounds the result. An unphysical measured $W_{cb}$
     (200 or 1000 eV for Si) gives $a<1$ and $W_k<U_k$ (for example, Si L2
     at $W=44.5$ eV with $U=104$ eV). A guard on $W_{cb}<I$ or $W_k\ge U_k$
     would fail closed.
  4. Line 121: a composition element missing from `shells` raises
     `KeyError` rather than `ValueError`.
- **Tests.** The measured values of $a$ are only range-checked
  (`test_shell_oscillators.py:107`). Pinning 2.2024, 3.2025 and 1.7797
  would anchor them.

Verdict: verified with caveats.

## Commands run

```text
pdftotext -layout "<PENELOPE-2024 PDF>" pen.txt; sed -n 6735,7000p pen.txt; sed -n 7585,7615p pen.txt
sha256sum ~/.local/share/pyrite/xsgen/reference-data/sbethe/sdbase/pdatconf.p14
curl -sSL https://micro.ustc.edu.cn/database/ELF/{Si,SiO2}.html; python3 elf.py (own HTMLParser)
curl -sSL -A Mozilla/5.0 https://pmc.ncbi.nlm.nih.gov/articles/PMC11957257/
curl -sSL https://export.arxiv.org/pdf/2012.09924v1; pdftotext moy.pdf
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python ver.py   # own build + bisection
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python <compare/edge-case probes>
PYRITE_MC_BACKEND=cpu pyrite-dev test tests/montecarlo/test_shell_oscillators.py   # 10 passed
```

## Recommended fixes

I have not applied any of these.

1. Change "within 7%" to "within 7.1%" or "within 10%" in
   `penelope-shell-oscillators.md:53`, or state 7.0% explicitly.
2. For the SiO₂ sample density (2.65 g/cm³ on the database page) versus the
   catalog density (2.20 g/cm³), document the mismatch in the TOML `basis`
   field and in the derivation, or choose a $W_{cb}$ for a density-matched
   sample.
3. Harden `load_conduction_bands` against non-finite values, make
   `build_shell_oscillators` reject a measured $f_{cb}\le0$, and add a
   sanity guard on $a$, $W_k\ge U_k$, or $W_{cb}<I$.
4. Pin the measured values of $a$ in the tests.

## Author resolution

After this record, the author applied fixes 1, 3 and 4. The derivation now
states "within 7.1%" with per-material values; the ledger and test use 7.1%.
`build_shell_oscillators` rejects non-finite or non-positive measured
parameters, a missing element, $W_{cb}\ge I$ and any bound $W_k\le U_k$.
The tests pin $a$ to $10^{-4}$. For fix 2, the SiO₂ density mismatch, the
partial f-sum, the peak-versus-log-mean choice and the Mo configuration are
documented as caveats. The tabulated 23.6 eV is kept, pending an owner
decision on a density-matched value. The ledger status is `rederived`
according to the methodology; sign-off remains human-only.

### Owner input change after verification

The owner then replaced the SiO₂ $W_{cb}$ with Saito et al.'s amorphous 22 eV
q=0 peak. Item 2 of this record reads that value independently. The
equations and code are unchanged. The SiO₂ factor becomes $a=3.4701$, and
22 eV lies 0.25% below Eq. 3.62 at catalog density. The status stays
`rederived` because only a verified input value changed; a fresh check can
confirm the new $a$.

## Addendum: WSe2 (issue #218)

Scope: only the new `wse2` entry in `src/pyrite/data/conduction_band.toml`,
with its rows in `penelope-shell-oscillators.md`, the test pin and the
ledger row (commit `be71f4ac`). The equations and code are unchanged and
were verified above.

- **Claim**: `penelope-shell-oscillators`, WSe₂ input. Code:
  `data/conduction_band.toml::wse2` used by
  `montecarlo/transport/shell_oscillators.py::build_shell_oscillators`.
  Source: Ahmad et al., J. Phys.: Condens. Matter 29, 165502 (2017),
  with PENELOPE-2024 Eqs. 3.62 and 3.64.
- **Filters**: units pass; limits pass ($f_{cb}+\sum_k f_k=142=Z$, and
  every bound $W_k>U_k$); signs and conventions pass.
- **Re-derivation**: `matches` ($a=2.03826$ against the pinned 2.0383).
- **Verdict**: `rederived`, with three wording fixes (see below). None of
  them changes a number.
- **Write-up**: `docs/validation/beam-transport/penelope-shell-oscillators-verification.md`
  (this addendum).

### Source

I fetched and read the arXiv preprint 1701.06293 (pdftotext, plus a render
of Fig. 1). I did not read the published JPCM version, which is paywalled.
`doi:10.1088/1361-648X/aa63a7` resolves to the IOP article page.

- Sample: "high quality thin films of a thickness of about 100 nm were
  prepared via mechanical exfoliation from single crystals". The authors
  "verified the pure hexagonal phase of WSe2 by measuring the electron
  diffraction profile". The material is 2H-WSe₂. They measured at 20 K
  with a 172 keV beam, $\Delta E=82$ meV and $\Delta q=0.04$ Å⁻¹.
- Geometry: the spectra run "along the ΓM direction of the Brillouin zone.
  The momentum transfer is 0.1 Å⁻¹, which corresponds to the optical
  limit". ΓM lies in the basal plane, and the exfoliated films are
  c-normal, so $q$ is in-plane.
- Value: "dominated by a strong feature at about 22 eV which can be
  associated with the volume plasmon, a collective excitation of all
  valence electrons." In Fig. 1, the undoped peak sits just above the
  20 eV tick, consistent with 22 eV. Fig. 1 plots the measured loss
  spectrum, not a tabulated KK loss function. That supports the doc's
  "read from text" and its ±0.5 eV reading-uncertainty caveat. The ±0.5 eV
  bound is the author's judgement; I did not derive it.
- Other features: a broad feature "around 44 eV" from "multiple scattering
  and shallow core level excitations from the W 5p levels", and excitons at
  1.8 and about 2.3 eV.

The TOML `basis`, `source` and `doi` fields, the table row and the WSe₂
caveat bullet all match the source.

### $f_{cb}=18$ and shell consumption

The chemical valence is W $5d^46s^2$ (6) plus $2\times$ Se $4s^24p^4$ (12),
giving 18. I recomputed the SHA-256 of the pinned `pdatconf.p14`
(`cd239554…62d2`), and it matches the record. My own parse gives
$\sum f=74$ for W and 34 for Se.

I sorted every shell in the formula unit by $U$ and weighted each shell's
occupation by stoichiometry:

| shells | $U$ (eV) | electrons | running total |
| --- | ---: | ---: | ---: |
| W O4 ($5d_{3/2}$, 4) + W P1 ($6s$, 2) | 8.667 | 6 | 6 |
| Se N2 + N3 ($4p$), $\times2$ | 9.75 | 8 | 14 |
| Se N1 ($4s$), $\times2$ | 20.15 | 4 | 18 |
| next: W N7 ($4f_{7/2}$) | 36.0 | — | — |

The count reaches 18 exactly at a shell boundary. No shell is split, and
each equal-$U$ group (W O4/P1, Se N2/N3) is consumed whole. The pdatconf W
configuration is O4 = 4 plus P1 = 2, which matches $5d^46s^2$. W N7, N6, O3
and O2 sit at 36, 38, 41 and 51 eV, which confirms "W 4f and 5p at
36–51 eV".

### Numbers

I computed these with my own script; no repository helper went into them.

- Density from the packaged CIF: $a=3.282$ Å, $c=12.96$ Å, $\gamma=120^\circ$,
  $V=120.896$ Å³, two formula units per cell, $M=341.78$ g/mol. That gives
  $\rho=9.3889$ g/cm³, the same as the catalog's 9.388917.
- All-electron plasma energy with $Z=142$:
  $\Omega_p=\hbar\sqrt{n_e e^2/(\varepsilon_0 m_e)}=56.9128$ eV. The
  repository `plasma_energy_eV("wse2")` gives 56.91282.
- Eq. 3.62: $\sqrt{18/142}\,\Omega_p=20.263$ eV. So 22.0 eV is $+8.57\%$
  above it (doc: 20.26 eV, 8.6%). It is under the 9% test tolerance.
- $I$: the Bragg-additive value from ICRU 37 elemental values (W 727 eV,
  Se 348 eV) is $\exp[(74\ln727+68\ln348)/142]=510.877$ eV. This equals the
  catalog `mean_excitation_eV`.
- Eq. 3.64, solved for $a$ by Brent root-finding over the 21 remaining
  bound shells, with $f_k$ = stoichiometry × occupation:
  $a=2.038262$, which rounds to 2.0383. The other convention, one
  oscillator per atom, gives 2.03917, so the 1e-4 pin tells the two apart.
  The pin matches the per-formula grouping that the earlier MoS₂ entry
  already uses. The smallest bound $W_k$ is 74.2 eV, above its $U_k$.
- Cutoff: $W_{cb}=22<50$ eV. `validate_shell_cutoff` requires
  cutoff $>W_{cb}$, so it passes.
- `pyrite-dev test tests/montecarlo/test_shell_oscillators.py`
  gave 20 passed, 0 skipped. The pinned data were present.

### Doc and ledger findings

These are wording issues, not discrepancies. I have not applied them.

1. `penelope-shell-oscillators.md`, h-BN caveat: "Its 6.9% excess over
   Eq. 3.62 is the largest in the table" is now stale. WSe₂ (8.6%) is
   larger, and the WSe₂ bullet says so. Drop the h-BN sentence or reword it
   to "second largest".
2. Same file, consumption paragraph: "Unlike the 15 eV default, this
   includes O 2s (28.5 eV) and S 3s (20.2 eV)" should also list Se 4s
   (20.15 eV). That shell lies above the 15 eV default and is consumed by
   the WSe₂ band.
3. Ledger `penelope-shell-oscillators` Notes: replace "its fresh-context
   verification is pending" with a pointer to this addendum. The other
   ledger text (Source, Checks with "within 9%", pinned 2.0383, and the
   WSe₂ shell assignment) is accurate.

### Suggested ledger change

Keep the status at `rederived`, since the WSe₂ input is rederived. In
Notes, replace "its fresh-context verification is pending" with "verified
in the WSe₂ addendum of the verification record". Never mark the row
`signed-off`; only a human may do that (#277).

## Addendum: Bell & Liang dichalcogenides (issue #218)

Scope: only the eight new `src/pyrite/data/conduction_band.toml` entries
(`mose2`, `ws2`, `mote2`, `nbs2`, `nbse2`, `2h_tas2`, `2h_tase2`,
`zrse2`), the claim that `hfs2` and `hfse2` cannot be represented, and the
matching rows, caveats, test pins and ledger text (commit `0457e5e9`). The
equations and code are unchanged and were verified above.

- **Claim**: `penelope-shell-oscillators`, Bell & Liang MX₂ inputs. Code:
  `data/conduction_band.toml::{mose2,ws2,mote2,nbs2,nbse2,2h_tas2,2h_tase2,zrse2}`
  used by `montecarlo/transport/shell_oscillators.py::build_shell_oscillators`.
  Source: M. G. Bell and W. Y. Liang, Adv. Phys. 25, 53 (1976), Table 4,
  with PENELOPE-2024 Eqs. 3.62 and 3.64.
- **Filters**: units pass; limits pass (for each material
  $f_{cb}+\sum_k f_k=Z$, every bound $W_k>U_k$ with a margin of at least
  25 eV, and $W_{cb}<I$); signs and conventions pass.
- **Re-derivation**: `matches`. All eight $W_{cb}$, $f_{cb}$, shell sets,
  deviations and $a$ values agree with the pins; the HfS₂/HfSe₂ failure
  holds.
- **Verdict**: `rederived`, with one provenance correction (the NbS₂ and
  NbSe₂ polytype labels are swapped relative to the source) and two
  wording fixes. None of them changes a number.
- **Write-up**: `docs/validation/beam-transport/penelope-shell-oscillators-verification.md`
  (this addendum).

### Source

I read the local PDF of the published article. The `pdftotext` layer
garbles chemical formulas (NbSe₂ comes out as "NbS%"), so I rendered and
read journal pp. 60–61, 66 (Table 1) and 82 (Table 4) as images.

- Geometry (abstract and §3.2): "a beam of 50 keV electrons is incident
  along the c-axis of the crystals and electrons inelastically scattered
  through an angle of 1 m radian are selected". Specimens were mounted
  "with the basal plane normal to the incident electron beam", and the
  aperture accepted $1.0\pm0.1$ mrad. For losses below 25 eV the momentum
  transfer makes $75^\circ<\theta<90^\circ$ with $c$, so $q$ is almost in
  the basal plane. This matches the c-normal catalog crystals.
- Specimens (§3.1): 50–100 nm thick, peeled or tape-cleaved from bulk
  crystals "grown from the synthesized compound by the vapour transport
  technique, except in the case of MoS₂ where natural molybdenite was
  used". The caveat bullet states this correctly.
- Table 4, main (largest-$n$) plasmon rows:

| key | Table 4 row | $\omega_L$ (eV) | $\omega_p$ (eV) | $n$ | TOML `resonance_eV` / `electrons_per_formula` |
| --- | --- | ---: | ---: | ---: | --- |
| `mose2` | MoSe₂ | 22.1 | 20.3 | 18 | 22.1 / 18 |
| `ws2` | WS₂ | 23.3 | 21.6 | 18 | 23.3 / 18 |
| `mote2` | α-MoTe₂ | 19.4 | 18.2 | 18 | 19.4 / 18 |
| `nbs2` | NbS₂ | 22.5 | 20.2 | 17 | 22.5 / 17 |
| `nbse2` | NbSe₂ | 21.0 | 19.0 | 17 | 21.0 / 17 |
| `2h_tas2` | 2H-TaS₂ | 22.0 | 20.2 | 17 | 22.0 / 17 |
| `2h_tase2` | 2H-TaSe₂ | 21.0 | 19.0 | 17 | 21.0 / 17 |
| `zrse2` | ZrSe₂ | 19.1 | 17.1 | 16 | 19.1 / 16 |

The key mapping is right: the S row goes to `2h_tas2` and the Se row to
`2h_tase2`. Each value is the largest-$n$ row. The lower rows match the
caveat and `basis` strings: partial plasmons at 8.1 (MoSe₂),
7.1 (MoTe₂), 8.5 (NbS₂) and 7.5 eV (NbSe₂), and carrier plasmons at 1.0
(NbS₂), 0.95 (NbSe₂), 1.2 (2H-TaS₂) and 1.0 eV (2H-TaSe₂). Table 4 also
gives MoS₂ 23.1 eV and WSe₂ 22.2 eV, which supports the corroboration
sentence. HfS₂ is 20.6 eV and HfSe₂ 19.5 eV, both with $n=16$.

- Eq. 19 of the paper is $\omega_L^2=\omega_p^2/\epsilon_c+\omega_T^2$.
  The tabulated $\epsilon_c(\omega_L)$ is 1.0–1.1 for every main plasmon,
  so the caveat's $\omega_L^2\approx\omega_p^2+\omega_T^2$ is a fair
  reduction. The text gives "a probable error of 10%" on $\omega_T$ and
  allows "an error of about 10%" on $\epsilon_c$. I found no stated error
  on the main-plasmon $\omega_L$. The caveat is accurate.
- Polytypes. Table 1 gives MoSe₂ 2H, WS₂ 3R and ZrSe₂ 1T. Table 4 names
  α-MoTe₂ (the 2H form), 2H-TaS₂ and 2H-TaSe₂. The Table 1 caption says
  "For the two metallic compounds investigated in detail, 2H-NbSe₂ and
  2H-TaS₂". The paper therefore states **2H for NbSe₂**. I found no
  polytype for **NbS₂** anywhere. Its only other label is "NbS₂ (hyp.)",
  the hypothetical octahedral form in the band-scheme figure, which is not
  the measured sample. The catalog phases are `mose2` 2H, `ws2` 2H,
  `mote2` 2H, `nbs2` 2H, `nbse2` 2H, `2h_tas2` 2H, `2h_tase2` 2H and
  `zrse2` 1T. So WS₂ (3R measured, 2H catalog) is the only stated
  mismatch, and the doc records it.

### Shell consumption

I parsed `pdatconf.p14` myself (SHA-256 `cd239554…62d2`, which matches the
record) and sorted the shells of each formula unit by $U$, with $f$ =
stoichiometry × occupation:

| key | shells consumed (ascending $U$) | $U$ range (eV) | total | next shell |
| --- | --- | --- | ---: | --- |
| `mose2` | Mo N4+O1 (6), Se N2+N3 (8), Se N1 (4) | 8.32–20.15 | 18 | Mo N3, 42 |
| `ws2` | W O4+P1 (6), S M2+M3 (8), S M1 (4) | 8.67–20.2 | 18 | W N7, 36 |
| `mote2` | Mo N4+O1 (6), Te O2+O3 (8), Te O1 (4) | 8.32–17.84 | 18 | Mo N3, 42 |
| `nbs2` | Nb O1 (2), Nb N4 (3), S M2+M3 (8), S M1 (4) | 7.07–20.2 | 17 | Nb N3, 38 |
| `nbse2` | Nb O1 (2), Nb N4 (3), Se N2+N3 (8), Se N1 (4) | 7.07–20.15 | 17 | Nb N3, 38 |
| `2h_tas2` | Ta O4+P1 (5), S M2+M3 (8), S M1 (4) | 8.14–20.2 | 17 | Ta N7, 28 |
| `2h_tase2` | Ta O4+P1 (5), Se N2+N3 (8), Se N1 (4) | 8.14–20.15 | 17 | Ta N7, 28 |
| `zrse2` | Zr N4+O1 (4), Se N2+N3 (8), Se N1 (4) | 7.73–20.15 | 16 | Zr N3, 33 |

Every count lands on a whole-shell boundary and consumes each equal-$U$
group whole. The sets equal the test parametrization in
`tests/montecarlo/test_shell_oscillators.py`. The metal valences in
pdatconf are Mo 6, W 6, Nb 5, Ta 5 and Zr 4; with $2\times6$ chalcogen
electrons they give Bell & Liang's 18, 17 and 16. The consumption
paragraph's list (metal outer $d$+$s$, chalcogen outer $s$+$p$, Te 5s at
17.8 eV) is correct.

HfS₂ and HfSe₂: Hf O4+P1 (4 electrons, 7.25 eV) plus the chalcogen $p$
(8, 9.75 or 10.36 eV) gives 12. The next shell is Hf N7 ($4f_{7/2}$,
8 electrons, 20.0 eV), below S M1 (20.2 eV) and Se N1 (20.15 eV). The
count therefore jumps from 12 to 20, and $n=16$, which needs the chalcogen
$s$ without the 4f, cannot end on a boundary. Hf N6 (21.0 eV) lies above
the chalcogen $s$, so only N7 sits strictly "between" the $p$ and $s$
shells. The conclusion holds either way. Had a band been buildable, both
would also have exceeded the 12% check: Eq. 3.62 gives 18.31 and
17.16 eV, which puts the measured values 12.5% and 13.6% above it.

### Numbers

I used my own script: densities from the packaged CIF cells and site
counts with IUPAC atomic weights, $\Omega_p=\hbar c\sqrt{4\pi r_e n_e}$
with all $Z$ electrons, $I$ from ICRU 37 elemental values, and Eq. 3.64
solved by Brent's method. No repository helper went into these numbers.
I compared the results with the catalog density, `mean_excitation_eV` and
`plasma_energy_eV` afterwards; all agree to the printed digits.

| key | $Z$ | $\rho$ (g/cm³) | $\Omega_p$ (eV) | $I$ (eV) | Eq. 3.62 (eV) | B&L $\omega_p$ | excess | $a$ | pin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mose2` | 110 | 6.9626 | 50.048 | 375.26 | 20.246 | 20.3 | +9.16% | 1.900831 | 1.9008 |
| `ws2` | 106 | 7.7619 | 52.490 | 476.99 | 21.630 | 21.6 | +7.72% | 2.003276 | 2.0033 |
| `mote2` | 146 | 7.7985 | 51.888 | 466.60 | 18.219 | 18.2 | +6.48% | 1.592179 | 1.5922 |
| `nbs2` | 73 | 4.5641 | 41.974 | 288.53 | 20.256 | 20.2 | +11.08% | 1.779153 | 1.7792 |
| `nbse2` | 109 | 6.4497 | 48.240 | 372.50 | 19.051 | 19.0 | +10.23% | 1.910086 | 1.9101 |
| `2h_tas2` | 105 | 7.0738 | 50.166 | 470.99 | 20.186 | 20.2 | +8.99% | 2.066772 | 2.0668 |
| `2h_tase2` | 141 | 8.6911 | 54.796 | 506.32 | 19.027 | 19.0 | +10.37% | 2.080046 | 2.0800 |
| `zrse2` | 108 | 5.4773 | 44.400 | 364.03 | 17.090 | 17.1 | +11.76% | 1.905555 | 1.9056 |

- The doc's deviations (9.2, 7.7, 6.5, 11.1, 10.2, 9.0, 10.4, 11.8%) and
  its Eq. 3.62 column are reproduced.
- Bell & Liang's own $\omega_p$ agrees with Eq. 3.62 at catalog density
  within 0.06 eV for all eight, which confirms "to 0.1 eV". Their
  $\omega_p$ is the free-electron value for $n$ electrons per formula unit,
  the same quantity as $\sqrt{f_{cb}/Z}\,\Omega_p$.
- All eight $a$ round to the pins at $10^{-4}$.
- $W_{cb}\le23.3$ eV $<50$ eV for all eight, so `validate_shell_cutoff`
  passes. $W_{cb}<I$ and every bound $W_k-U_k\ge25$ eV.
- The test tolerance went from 9% to 12%; ZrSe₂ (11.76%) leaves 0.24%
  margin. This is acceptable for literature inputs, but any re-read of a
  catalog density for `zrse2` could trip it.
- `env -u PYRITE_ONLINE_TESTS pyrite-dev test tests/montecarlo/test_shell_oscillators.py`
  gave 30 passed, 0 skipped. The pinned data were present, so the
  hafnium test ran.

### Doc and ledger findings

These are not numerical discrepancies, and I have not applied them.

1. **Polytype labels swapped (provenance).** The source states 2H for
   NbSe₂ (Table 1 caption) and gives no polytype for NbS₂.
   - `conduction_band.toml::nbs2` `basis` says "2H-NbS2". Change it to
     "NbS2 (trigonal-prismatic metal; polytype not stated)".
   - `conduction_band.toml::nbse2` `basis` says "NbSe2". Change it to
     "2H-NbSe2".
   - In the `penelope-shell-oscillators.md` table, change the nbs2 row
     from "2H, metal" to "metal" and the nbse2 row from "metal" to
     "2H, metal".
   - In the Bell & Liang caveat, replace "the polytype of their NbSe₂ was
     not checked" with "they name 2H-NbSe₂ but give no polytype for NbS₂
     (vapour-grown NbS₂ is often 3R; like WS₂, the layers are the same)".
   - The catalog `nbs2` 2H phase is unaffected; this is the same kind of
     polytype caveat already recorded for WS₂.
2. **Stale superlative.** `penelope-shell-oscillators.md`, WSe₂ caveat:
   "Its 8.6% excess over Eq. 3.62 is the largest in the table" is now
   false, because ZrSe₂ (11.8%), NbS₂ (11.1%), TaSe₂ (10.4%), NbSe₂
   (10.2%) and MoSe₂ (9.2%) exceed it. Drop the sentence. The h-BN
   caveat's 6.9% sentence carries no superlative and is fine.
3. **Hf 4f wording (minor).** "places Hf N6/N7 (4f) at 20–21 eV, between
   the chalcogen $p$ … and $s$ (20.2 eV) shells" is true for N7 (20.0 eV)
   only; N6 (21.0 eV) lies above the chalcogen $s$. Suggested wording:
   "places Hf N7 ($4f_{7/2}$, 20.0 eV) below the chalcogen $s$ shell
   (20.15–20.2 eV), so the count jumps from 12 to 20 and 16 electrons do
   not end on a whole-shell boundary".
4. **Ledger Notes**: replace "fresh-context verification pending" with
   "verified in the Bell & Liang addendum of the verification record".
   The rest of the row (Source, the Checks wording "within 12%", "pinned
   $a$ for all 14 bands", "HfS₂/HfSe₂ whole-shell failure pinned", and
   the Notes text "6.5–11.8%", "3R measured, 2H catalog", "Hf 4f splits
   the $n=16$ valence") is accurate.

### Suggested ledger change

Keep the status at `rederived`, since the eight inputs are rederived. In
Notes, replace "fresh-context verification pending" with "verified in the
Bell & Liang addendum of the verification record". Apply finding 1 to
the TOML `basis` strings and the doc before relying on the polytype
provenance. Never mark the row `signed-off`; only a human may do that
(#277).
