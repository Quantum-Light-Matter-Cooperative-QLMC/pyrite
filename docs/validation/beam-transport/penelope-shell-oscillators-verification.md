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
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_shell_oscillators.py   # 10 passed
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
