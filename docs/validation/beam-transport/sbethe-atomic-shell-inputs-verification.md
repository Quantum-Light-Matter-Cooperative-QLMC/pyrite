# SBETHE atomic shell inputs: independent verification

Validation: `sbethe-atomic-shell-inputs`. Fresh-context verifier record for
branch `issue-93-soft-hard-inelastic-transport` at `10653d7a`. This record does
not change the ledger status; only a human may mark the claim `signed-off`.

## Sources consulted

- CSEWG, *ENDF-6 Formats Manual*, ENDF-102, BNL-224854-2023-INRE (ed. D. A.
  Brown, 28 September 2023), fetched from
  `https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf`. Appendix B.1
  (reaction type numbers MT 534–572) and §28 (MF=28 "Table of Subshell
  Designators", designator 1 = K = MT 534).
- Pinned SBETHE `sdbase/pdatconf.p14`, SHA-256
  `cd239554bb6e823692ea4611d443df8684b4cace06006fc271a4168cb78c62d2`
  (recomputed; matches `PDATCONF_SHA256`). Header and all 1562 data rows.
- Packaged `EEDL.endf` (2025 Cullen EEDL), SHA-256
  `f3ef54f6…1769c` (matches `EEDL_SHA256`), read with a verifier-written raw
  ENDF text parser (columns 67–75 for MAT/MF/MT; the TAB1 control record's
  first field for EPE), not with `endf_parserpy` or repository helpers. The
  file carries MF 1, 23, 26 only; no MF=28 relaxation data.
- Reference ionization potentials: Cs 3.894 eV (6s), Tl 6.108 eV (6p),
  Fr 4.07 eV (7s).

## Filters

- **Units.** Both files tabulate binding/ionization energies in eV; the
  diagnostic difference $\Delta E = E^{\rm EEDL}_{\rm b} - E^{\rm SBETHE}_{\rm b}$
  is in eV. Pass.
- **Limits.** Missing default source raises (`read_bytes`); checksum mismatch
  raises; an EEDL shell with no SBETHE $n,l$ raises; an SBETHE shell without an
  EEDL channel is omitted from the match tuple. Pass.
- **Conventions.** The central convention is designator numbering (below).
  Pass.

## Independent derivation

### ENDF-6 subshell designators

ENDF-6 Appendix B.1 assigns one MT per subshell in x-ray order with every
$j$ slot through $\ell=5$ for $n=6$:

$$
d = \mathrm{MT} - 533,\qquad
d:\ \mathrm{K}\,(1),\ \mathrm{L}_{1\text{–}3}\,(2\text{–}4),\ \mathrm{M}_{1\text{–}5}\,(5\text{–}9),\
\mathrm{N}_{1\text{–}7}\,(10\text{–}16),\ \mathrm{O}_{1\text{–}9}\,(17\text{–}25),\
\mathrm{P}_{1\text{–}11}\,(26\text{–}36),\ \mathrm{Q}_{1\text{–}3}\,(37\text{–}39).
$$

The manual lists MT 557 = O8 ($5g_{7/2}$), 558 = O9 ($5g_{9/2}$),
559 = P1 ($6s_{1/2}$), 569 = P11 ($6h_{11/2}$), 570 = Q1 ($7s_{1/2}$),
572 = Q3 ($7p_{3/2}$). Hence ENDF P1 is $d=26$ and Q1 is $d=37$. The MF=28
designator table states the same offset ($d=1\leftrightarrow$ MT 534).

Internal EEDL consistency with this numbering (raw parse):

| element | EEDL $d$ | EPE (eV) | ENDF label | known level |
| --- | --- | --- | --- | --- |
| Cs | 26 | 3.89 | P1 | $6s$ IP 3.894 eV |
| Tl | 27, 28 | 6.11, 6.11 | P2, P3 | $6p$ IP 6.108 eV |
| Fr | 37 | 4.0 | Q1 | $7s$ IP 4.07 eV |
| U | 22, 23, 29, 30, 37 | 6.03, 7.95, 6.0, 4.13, 6.0 | O6, O7, P4, P5, Q1 | $5f^3 6d^1 7s^2$ |

Under PENELOPE codes, Cs $d=26$ would read as $6p_{3/2}$ (unoccupied in
neutral Cs) and Fr $d=37$ would be undefined; the ENDF reading is the only
consistent one.

### `pdatconf.p14` code semantics

The header documents column 2 as "Code number of the shell (IS)" and column 3
as x-ray notation. Over all rows the code↔(label, $nlj$) map is a bijection:
codes 1–23 = K … O7 (identical to ENDF $d$), 24–27 = P1–P4, 29 = Q1. Code 28
(PENELOPE P5) never occurs. Every row's x-ray label is consistent with its
spectroscopic $nlj$ column. So the integer codes agree with ENDF through O7
and diverge from P1 on (PENELOPE 24 vs ENDF 26).

### Label to $n,\ell$

For subshell index $i$ within a shell letter ($i=1$ for K),

$$
\ell = \lfloor i/2 \rfloor,\qquad
j = \begin{cases} \ell - \tfrac12 & i \text{ even},\\ \ell + \tfrac12 & i \text{ odd}, i>1.\end{cases}
$$

This reproduces every Appendix B.1 entry checked, including O8 $5g_{7/2}$,
O9 $5g_{9/2}$, P8–P9 $6g$, P10 $6h_{9/2}$, P11 $6h_{11/2}$, Q3 $7p_{3/2}$.

### Occupations

Independent parse: 99 elements ($Z=1\ldots99$), $\sum_s f_{Z,s}=Z$ for every
$Z$, minimum ionization energy 3.89 eV, no negative profile or width, codes
unique per element, energies non-increasing within each element.

### Join rule and recomputed numbers

Rule derived from the claim: exact x-ray label match; else the unique SBETHE
shell with equal $(n,\ell)$; else raise. For the merged case the per-$n\ell$
rate is $\sigma_{n\ell}=\sum_j\sigma_{n\ell j}$. Verifier script
(raw-parsed EEDL, own label table, own $n,\ell$ rule) results:

- all 99 elements join, no raise;
- 41 merged EEDL channels; maximum $\lvert\Delta E\rvert$ = 3.33 eV (Ga N3 onto
  N2); next B L3 3.09, Al M3 3.08, Np O7 3.03 eV;
- maximum exact-label $\lvert\Delta E\rvert$ = 17.92 eV (Mn L1); heavy P shells
  above 5 eV: Pb P1 5.05, Bi P1 5.69, Po P1 8.61, Am P2 −7.59, Am P3 −11.08 eV;
- SBETHE shells with no EEDL channel: Pd O1, Ce–Eu and Tb–Tm O4, Ir P1,
  Th O6, Pu/Am/Cf/Es P4.

## Comparison with the implementation

Read after the derivation above.

- `EEDL_SUBSHELL_LABELS` (`src/pyrite/montecarlo/eedl_ionization.py:22`) equals
  the Appendix B.1 order for all 39 entries; the parser sets
  `shell_designator=mt - 533` (`eedl_ionization.py:161`). Match.
- `_nl` (`src/pyrite/montecarlo/shell_configuration.py:49`) gives
  `'spdfgh'[index // 2]`; tabulated for K … Q3 it returns the same $n,\ell$ as
  the rule above, including 5g, 6g, 6h. Match.
- `match_eedl_shells` (`shell_configuration.py:111`) implements exact label,
  then unique `orbital[:2] == _nl(label)`, else raise. Run on the pinned data,
  it yields 41 merges, max merged 3.333 eV (Ga N3), max exact 17.92 eV
  (Mn L1), identical merge list to the verifier's. Match.
- `spectrum/characteristic.py:52` aliases `_SHELL_LABELS = EEDL_SUBSHELL_LABELS`;
  the removed map on `main` is dict-equal to the new one, so characteristic
  relaxation lookup behaviour is unchanged and was already ENDF-ordered. No
  silent mislabel found. `_SHELL_ORDER` (`characteristic.py:53`) has no reader
  (pre-existing).

## Physical adequacy of the partner merge

All 41 merged EEDL channels have EEDL binding energies $\le 11.3$ eV (largest
C L3 11.26 eV) and are open valence shells (B/C 2p, Al/Si 3p, Sc–V 3d, Ga/Ge
4p, Y–Mo 4d, In/Sn 5p, La/Gd/Lu–W 5d, Ce–Eu 4f, Tl/Pb 6p, Ac–Np, Cm, Bk 6d, Pa–Am
5f). All lie far below the 50 eV relaxation cutoff used by `characteristic.py`,
and valence holes do not select a radiative or Coster–Kronig path, so the $j$
label is not load-bearing for relaxation. The merge is defensible as stated.
Caveat: the summed rate carries EEDL's $n\ell$ occupancy, not SBETHE's; the
two agree only where the configurations agree.

## Findings

1. **Label uniqueness is not enforced.** `load_atomic_shells` checks unique
   designators (`shell_configuration.py:104`) but not unique labels, while
   `match_eedl_shells` keys `by_label` on labels (`shell_configuration.py:136`).
   An alternate-path file with two codes carrying one label would collapse
   silently. The pinned file is unaffected (bijection verified). The
   derivation's "duplicate labels" fixture wording overstates the test, which
   uses a duplicated designator row.
2. **Label/orbital consistency is not checked** (the join mixes label-derived
   and `orbital[:2]`-derived $n,\ell$). Consistent in the pinned file.
3. **Element completeness** (all of 1–99) is asserted only in the optional
   fetched-source test, not in the loader.
4. The derivation's phrase "attach an unoccupied spin-orbit partner"
   (ledger claim) means unoccupied in SBETHE; EEDL does populate it.

None changes a number or join in the pinned data.

## Commands

```text
sha256sum ~/.local/share/pyrite/xsgen/reference-data/sbethe/sdbase/pdatconf.p14
curl -sSL -o endf6.pdf https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf; pdftotext -layout
python3 pd.py     # verifier parse of pdatconf.p14
python3 eedl.py   # verifier raw ENDF parse of EEDL.endf MF=23
python3 join.py   # verifier join and item-4 numbers
PYRITE_MC_BACKEND=cpu uv run pyrite-dev test tests/montecarlo/test_shell_configuration.py tests/montecarlo/test_characteristic.py
  -> 33 passed, 1 warning
PYRITE_MC_BACKEND=cpu uv run python -c ...  # implementation join over 99 elements
```

## Verdict

- Item 1 (loader, occupations, fail-closed): verified with caveats (findings
  1–3).
- Item 2 (ENDF designators, PENELOPE divergence after O7): verified against
  ENDF-102 (2023) Appendix B.1.
- Item 3 (partner merge, raise, no-rate): verified.
- Item 4 (99 join, 41 merges, $\le 3.4$ eV Ga N3, 17.9 eV Mn L1, 5–11 eV heavy
  P): verified.

Suggested status: `rederived` (human applies).

## Author resolution

Following this record, `load_atomic_shells` rejects duplicate labels and any
label whose $n,\ell$ disagrees with its orbital, with fixture tests (findings
1–2). The ledger claim now says "unoccupied in SBETHE", and the derivation
notes that merged rates carry EEDL occupancy (finding 4). Finding 3 remains:
only the optional fetched-source test asserts element completeness. The
ledger status is `rederived` according to the methodology's definition; sign-off remains
human-only.
