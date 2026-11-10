# `bremslib-angular-schiff`

## Scope and question

Validation: `bremslib-angular-schiff`. This is an implementation-context comparison for #86, not fresh-context validation. It asks whether the photon angular shape stored in the BremsLib double-differential tables, which PyRITE uses for `S(θ)` in `mc_brem_spectrum(cross_section_model="bremslib")`, agrees with an independent analytic reference. EEDL has no angular data, so this is BremsLib-only evidence. Magnitudes are covered by `brem-source-comparison`; only the shape is compared here.

## Reference

Koch and Motz, *Rev. Mod. Phys.* **31**, 920 (1959), Table I, Formula 2BS (Schiff's formula, screened Born approximation, small angle, $E_0\gg mc^2$). With total energies in units of $mc^2$, $E_0=1+T/mc^2$, $E=E_0-k/mc^2$, $y=E_0\theta$:

$$
\frac{d\sigma}{dk\,d\Omega}\propto
\frac{16y^2E}{(y^2+1)^4E_0}-\frac{(E_0+E)^2}{(y^2+1)^2E_0^2}
+\left[\frac{E_0^2+E^2}{(y^2+1)^2E_0^2}-\frac{4y^2E}{(y^2+1)^4E_0}\right]\ln M(y),
$$

$$
\frac1{M(y)}=\left(\frac{k}{2E_0E}\right)^2+\left(\frac{Z^{1/3}}{111\,(y^2+1)}\right)^2 .
$$

The published prefactor $(4Z^2r_0^2/137)(dk/k)\,y\,dy$ carries $y\,dy\propto d\Omega$, so the bracket is the density per steradian. The expression was transcribed from the rendered page, not from OCR text, and the bracket structure matches the Geant4 physics-reference Tsai form. Setting $Z=0$ in $M$ gives Formula 2BN(a).

## Method

`validation/brem_angular.py` reads each table node $(T,\kappa)$ directly from the staged `scaled_ddcs_mb_sr` (no interpolation, so parser or interpolation defects cannot enter) and evaluates the Schiff density on the same $\theta$ grid. Both are cut at $\theta\le\min(1\,\mathrm{rad},40/E_0)$ and normalized over that window with $d\Omega=2\pi\sin\theta\,d\theta$. The statistic is the enclosed-flux angle $\theta_p$, for $p=0.5$ and $0.9$, as the ratio BremsLib/Schiff. `checks/brem_angular_comparison.py` sweeps every catalogue element (`release.catalogue_elements()`), every table incident energy from 5 to 30 MeV, and $\kappa=0.1$–$0.8$. `gate_tolerance` gates $Z\ge3$ (`SCHIFF_MIN_Z`); lighter elements are compared, printed and recorded under `ungated_elements` in the JSON report.

## Results (2026-11-10)

42 catalogue elements, 248 nodes each (#1064).

| Z | $\theta_{50}$, $\theta_{90}$ ratio | |
|---|---|---|
| 1 (H) | 0.926–1.015 | reported |
| 3–20 | 0.967–1.024 | gated |
| 22–42 | 0.984–1.049 | gated |
| 46–83 | 0.990–1.137 | gated |

Over the 10168 gated nodes ($Z\ge3$, 41 elements), $\theta_{50}$ spans 0.967–1.097 and $\theta_{90}$ 0.968–1.137. The tip $\kappa>0.8$ spans 0.99–1.15 and is reported, not gated. The 2026-09-25 measurement covered 24 elements ($Z=5$–83, 6144 nodes) with the same envelope.

- **Gate:** $|\text{ratio}-1|\le0.06$ for $3\le Z<46$, $0.15$ for $Z\ge46$. The tolerances are set from the measured envelope, not derived.
- **High-Z excess.** The BremsLib distribution is up to 14 % broader than Schiff at $Z\ge46$. Schiff is a Born approximation; the partial-wave library includes Coulomb distortion, which grows with $\alpha Z$. This is a reference limitation, not a demonstrated BremsLib defect.
- **Low energy.** Below 5 MeV, $E_0$ is too small for the small-angle formula. W at 1 MeV differs from Schiff by 15–25 % in peak angle. The check is silent there, and the shape is unvalidated by this reference.

## Hydrogen and the $Z$ floor

Hydrogen is up to 7.4 % narrower than Schiff at $\kappa=0.1$–$0.2$ and $T\approx8$–30 MeV. The deficit grows as $\kappa$ falls and $T$ rises, i.e. as $q_{\min}=k/(2E_0E)$ falls and the screening term of $M$ dominates, and fades to 0.97–0.99 by $\kappa=0.5$.

- **Method.** BremsLib uses the screened Born approximation with a Kohn–Sham atom, not partial waves, for $Z\le2$ at $T_1-k\ge0.5$ MeV (BremsLib v2.0.8 manual, p. 1). The library's H DDCS equals its `DDCS_Born` file at the 12 sampled nodes (5–30 MeV, $\kappa=0.1,0.2,0.5$), and at the same nodes Born and partial-wave ratios to Schiff differ by at most 1.5 % for Li, C and Al, so the method switch is not the cause.
- **Screening.** In the Born approximation the shape depends on $Z$ only through the atomic form factor, and in Formula 2BS only through $Z^{1/3}/111$ in $M$. Schiff's $M$ comes from the Thomas–Fermi-scaled Yukawa potential $(Ze/r)e^{-r/a}$, $a=111Z^{-1/3}$ (Koch and Motz, Sec. II-E(3), p. 940). Thomas–Fermi and Hartree–Fock screening functions differ by 1–2 % for $Z\ge3$ and by 4–5 % for $Z=1,2$ (Seltzer and Berger, *Nucl. Instrum. Methods B* **12**, 95 (1985), Sec. 2.2.2, p. 98, Fig. 1 and Table 1). The gate therefore starts at $Z=3$.
- **Not a radius.** Matching hydrogen's complete-screening logarithm gives a Yukawa radius of 122.8 instead of 111 (in $\hbar/mc$), which moves H only from 0.926 to 0.933. The residual is the shape of the hydrogen form factor, not a fitted constant.
- **Not electron–electron.** Staging multiplies each BremsLib node by the scalar $1+\eta/Z$, which cancels in the normalized shape. Neither side carries an electron–electron angular shape, so the e–e share, about half of hydrogen's cross section, does not enter this comparison. The `brem-source-comparison` floor (#1068) bounds magnitudes and the SB85 $\eta$ range; it differs from this one deliberately.

## Verdict (implementation context)

- **Claim:** `bremslib-angular-schiff`. BremsLib angular shape agrees with the screened-Born small-angle reference to within ±6 % ($3\le Z<46$) and ±15 % ($Z\ge46$) in enclosed-flux angle for 5–30 MeV.
- **Not covered:** hydrogen (reported, not gated); shape below 5 MeV; magnitude; electron–electron angular contribution; polarization. Schiff is itself an approximation, so an independent tabulated or measured DDCS (PENELOPE, Kissel–Quarles–Pratt) is still open.
- **Status:** `filtered`. Fresh-context verification is pending; only a human may mark it signed off.
