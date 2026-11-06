# `elsepa-vendor-reference`

## Scope and independence

Validation: `elsepa-vendor-reference`. Fresh-context validation on 2026-11-06 of the "State between energies (#656)" note only. The rest of the row (deck writer, parser, CDF and the vendor reference comparison) was not reviewed, so the row stays `filtered`.

Independence is imperfect. The verifier read the ledger note and the guard body before deriving the scaling, so the methodology's derive-first order was not kept for the guard. Treat that part as a source-to-code audit that was also checked numerically.

The merged PR removed the verifier's worktree during the run. The verifier reran the scan and the guard from `git archive d4a84cde` (the merged head). The W numbers and the corrected-table comparison ran against the earlier build and were not all repeated; the vendored source, `src/pyrite/xsgen`, its data pins and `tests/xsgen` are identical between the two revisions. The scripts and outputs were not committed.

## Mechanism

`DPWA` adds `CGL(3)` to the spin-flip series without checking `NPH` (`vendor/xsgen/elsepa/elsepa2020.f:3808`). `DPWA0` sets coefficients only up to `NPH` (`:3425-3432`, `:3438-3443`). The `ISCH=2` path starts at $L=1$, so $N_{\rm PH}\ge2$ (`:3305`), and `CFL(2)` and `CGL(2)` are always current. The `ISCH=1` path needs `L.GT.10`, so $N_{\rm PH}\ge12$. A panel with $N_{\rm PH}=2$ therefore reads `CGL(3)` from the last earlier panel with $N_{\rm PH}\ge3$ in the same process.

A build that zeroes `CGL(3)` when $N_{\rm PH}<3$ made shared and per-energy runs byte-identical on 13 decks: Fermi Z=2, 6 (positron), 14, 79; point Z=6; uniform and Helm Z=1, 2; free-atom `MABS 2` Z=1 and Z=14 (positron); muffin-tin Si and W (positron). Z=1 on the production grid, both species, was checked separately. Review found no other carried state that changes results:

- `ISUMP` is reset at `:3116` and `:4386`.
- `CFM`, `CGM` and `NPHM` are read only when $N_{\rm PH}\ge250$, and are rebuilt in that case.
- `DPWAC0` refills its block for each panel with `IZINF=1`.
- `EFIELD` and the `.dfs` read run on every `ELSEPA` call.

This is evidence from review and these decks, not a proof for all decks. That a fresh process reads zero relies on zero-initialized common blocks under gfortran on Linux. It is inferred from the equality above and was not printed directly.

**Unreached path.** At label 3, `IF(IND.LT.5.OR.IADD.EQ.1) GO TO 6` (`:3330`, and `:4673` in `DPWAI0`) exits without setting `NPH`. That happens if the phase shifts first fall below `EPSCUT` at exactly $L=100$. Reading the code, such a panel would use the previous panel's `NPH`, or 0 in a fresh process, and per-energy generation would not repair it. It did not occur in the scan below. It was read, not triggered.

## Screened-Rutherford scaling

With $\mu=\sin^2(\theta/2)$ and $d\Omega=4\pi\,d\mu$, the screened Mott cross section is

$$
\frac{d\sigma}{d\Omega}=\frac{(Z\alpha\hbar c)^2}{4(pc)^2\beta^2}\,\frac{1-\beta^2\mu}{(\mu+A)^2}.
$$

The vendor's `TCS1=2*ECS1` is $\sigma_1=\int(1-\cos\theta)\,d\sigma$ with $1-\cos\theta=2\mu$, so

$$
\sigma_1=\frac{2\pi(Z\alpha\hbar c)^2}{(pc)^2\beta^2}\,L,\qquad
L=\ln\frac{1+A}{A}-\frac{1}{1+A}-\beta^2\,O(1).
$$

The prefactor is an area, because $\hbar c/pc$ is a length. With $A\approx(\hbar/2pa)^2$, $L\approx\ln(4p^2a^2/\hbar^2)-1$. The slope on a clean table is then

$$
\frac{d\ln(\sigma_1p^2\beta^2)}{d\ln E}=\frac{2}{L}\,\frac{d\ln p}{d\ln E},
$$

where $d\ln p/d\ln E$ runs from $1/2$ (non-relativistic) to $1$ (ultra-relativistic). For H at 25 keV, $L\approx8$ gives about 0.13, against a measured 0.134.

A contaminated table holds $\sigma_1$ nearly constant. With $p^2\beta^2\propto(\gamma^2-1)^2/\gamma^2$, the limit is

$$
\text{slope}=\frac{4\gamma}{\gamma+1}-\frac{2(\gamma-1)}{\gamma}.
$$

This tends to 2 in both limits, has a minimum of 1.657 near $\gamma\approx2.4$, and equals 1.67 at 1 MeV and 1.99 at 100 MeV. The bound of 1 in `check_transport_scaling` separates the two regimes.

## Comparison with the implementation

| Check | Result |
|---|---|
| W, `MNUCL 1`, deck [1, 100] MeV, $\sigma_1$ shared/fresh at 100 MeV | 92.834 (1.22096e-22 / 1.31521e-24 cm²); zero build gives 1 and identical bytes |
| Same, from the last non-factorized production panel (1.25 MeV) on the production grid above it | 72.4 |
| Fermi Z=1-103, both species, $E>1$ keV, fresh processes | 10300 panels: $N_{\rm PH}=2$ only at Z=1, 25 and 30 keV; none below 2; 530 panels at 3 |
| Muffin-tin (`ISCH=1`, `:909`, `:920`) | minimum $N_{\rm PH}$ 15 (Si), 32 (W positron); released Si, W, HOPG electron and Si, W positron panels match fresh runs in DCS and $\sigma_1$ |
| Guard code | `gamma=1+E/mc^2`, `gamma**2-1` is $(p/mc)^2$, `1-gamma**-2` is $\beta^2$; domain matches `:226` and `:161`; neither condition depends on species |
| Largest slope, released free-atom tables (`-4` and `-5`) | electron 0.1565 (Z=31), positron 0.2016 (Z=83) |
| Largest slope, contaminated point nucleus, Z=6, 13, 14, 26, 29, 47, 74, 79 | electron 1.929-1.977, positron 1.902-1.975 |
| Largest slope, clean point nucleus, sampled Z up to 103 | at most 0.211 |
| Guard on contaminated data | raises on W (1.95); silent on contaminated H |
| `-5` Z=1 against per-energy runs, all 61 panels, both species | exactly equal in DCS, a0 DCS, Sherman, error and the three scalars; absorption not compared |
| Shared Z=1 run | differs from fresh only at 25 and 30 keV; reproduces the `-4` DCS exactly |
| Archives | SHA-256 match the pins; 108 of 108 shared members byte-identical in each archive |

Two guard mismatches affect non-production decks only and are harmless: free-atom `MABS` panels up to 1 MeV are not factorized but are still checked, and `IHEF=1` factorized panels above 100 MeV are skipped.

Other nuclear models also reach $N_{\rm PH}=2$. Shared-run $\sigma_1$ errors are up to $+6.05\times10^{-4}$ for uniform Z=1 (25-60 keV) and $+1.24\times10^{-4}$, $+1.73\times10^{-4}$ for Helm Z=1 (25, 30 keV). `model_record` covers these through `z == 1`. Uniform and Helm Z=2 showed no effect on this grid.

The change from `-4` to `-5` in Z=1 is:

| Species | $\sigma_1$, 25 keV | $\sigma_1$, 30 keV | DCS, 30 keV |
|---|---|---|---|
| electron | $-1.215\times10^{-4}$ | $-1.729\times10^{-4}$ | $-0.188\,\%$ to $+0.078\,\%$ |
| positron | $-1.168\times10^{-4}$ | $-1.664\times10^{-4}$ | $-0.184\,\%$ to $+0.070\,\%$ |

## Adjudication

The note's mechanism, numbers, coverage, guard and table claims match, with no discrepancy. Verdict for this note: `rederived`. The row stays `filtered`, and `anchored` is not supported because the verifier did not read or run `tests/xsgen/`.

Not checked:

- scan panels at or below 1 keV, excluded only by the `ISCH=1` argument above
- off-grid energies, and non-Fermi models beyond Z=1-2, point-nucleus Z=6 and W
- other compilers
- 10 of 13 muffin-tin crystals, and most free-atom panels against fresh runs
- `process_per_energy` key behaviour beyond reading `deck.py`

The 54 unchanged tables in each archive keep their shared-process provenance. Fresh runs matched them only on the sampled panels (Z=3, 6, 14, 79 electron; Z=3, 14, 79 positron; the muffin-tin tables above; 7-15 panels per table).

Only a human may mark this signed off.
