# `bremslib-electron-electron`

## Scope and source

Validation: `bremslib-electron-electron`. This fresh-context verification began with the ledger row, the module docstring of `pyrite._bremslib_electron`, the header of `data/bremsstrahlung_electron_electron.toml`, and the primary source: S. M. Seltzer and M. J. Berger, *Nucl. Instrum. Methods B* **12**, 95 (1985) (SB85), Eqs. 16–20, 27–29 and Table 4. It read pages 104, 114 and 115–120 of the PDF at 110–200 dpi. The implementation bodies were inspected only after the derivation and the numerical checks below.

The claim is that each staged BremsLib electron–nucleus table is multiplied, node by node, by $1+\eta(Z,T,k/T)/Z$, with $\eta$ from SB85 Table 4. The claim also covers a declared interpolation scheme and a fitted small-$k/T$ shape. BremsLib itself contains no electron–electron term.

## Independent derivation

### Scaling factor

SB85 Eq. 18 writes the electron–atom cross section as the sum of a nuclear term, proportional to $Z^2$, and $Z$ unit-charge electron targets:

$$
\left(4\alpha r_e^2\right)^{-1}k\frac{d\sigma}{dk}
=Z^2\chi_n+Z\chi_e
=Z\left(Z+\eta\right)\chi_n,
\qquad
\eta(Z,T_1,k)\equiv\frac{\chi_e}{\chi_n}
=\frac{d\sigma_e/dk}{(1/Z^2)\,d\sigma_n/dk}
\quad\text{(Eq. 19)}.
$$

BremsLib tabulates the nuclear part only, so its SDCS is proportional to $Z^2\chi_n$. Dividing the electron–atom value by it gives

$$
\boxed{\frac{d\sigma_{\rm atom}/dk}{d\sigma_n/dk}=1+\frac{\eta}{Z}}.
$$

The factor is dimensionless and depends on $(Z,T,k)$ only. At fixed $(T,k)$ it multiplies the photon-energy distribution as a scalar. If the electron–electron photon is assigned the nuclear angular distribution at the same $(T,k/T)$, the DDCS takes the same factor and $\int D\,d\Omega=\chi$ survives. SB85 gives no angular data, so this assignment is a modelling assumption and cannot be derived from the source.

### Integrated contribution

The SB85 Eq. 16 integral at fixed $T_1$ is $\Phi^{(n)}_{\rm rad}\propto\int_0^{T_1}\chi_n\,dk$, up to $T$-only prefactors. $\Phi^{(e)}_{\rm rad}$ is the same integral with $\chi_e=\eta\chi_n$. Eqs. 27 and 29 then give

$$
\bar\eta(Z,T_1)=\frac{\Phi^{(e)}_{\rm rad}}{\Phi^{(n)}_{\rm rad}}
=\frac{\int_0^1\eta\,\chi_n\,dx}{\int_0^1\chi_n\,dx},
\qquad
\Phi^{\rm tot}_{\rm rad}=\Phi^{(n)}_{\rm rad}\left(1+\frac{\bar\eta}{Z}\right),
$$

with $x=k/T_1$. The weight is $\chi_n\,dx$, because the $\beta^2$ and $1/(T_1+mc^2)$ prefactors are constant at fixed $T_1$. So the ledger's $\chi_n$-weighted mean is the correct definition of $\bar\eta$.

### Tip limit

From SB85 Eq. 20, the highest photon energy for an electron target is

$$
k_{\max}=\frac{mc^2T_1}{T_1+2mc^2-\beta_1\left(T_1+mc^2\right)}.
$$

At every finite $T_1$, $k_{\max}<T_1$. The bound tends to $T_1/2$ as $\beta_1\to0$ and to $T_1$ as $\beta_1\to1$. Therefore $\eta(x=1)=0$ exactly, which is Table 4's $k/T_1=1.00$ column. Evaluating Eq. 20 gives $x_{\max}=0.9917$ at 30 MeV and $0.9938$ at 40 MeV. These values agree with Table 4: for $Z=6$ the $0.99$ column is zero at 30 MeV and $0.057$ at 40 MeV.

### Small-$k/T$ shape

Table 4 has no node between $x=0$ and $0.1$, but at MeV energies $\eta$ falls steeply there ($Z=6$, 10 MeV: $2.553\to1.090$). The declared shape is

$$
\eta(x)=\eta(0.1)+\left[\eta(0)-\eta(0.1)\right]g(x),\qquad
g(x)=\frac{(1+x/\kappa_s)^{-1}-(1+0.1/\kappa_s)^{-1}}{1-(1+0.1/\kappa_s)^{-1}}.
$$

Directly, $g(0)=1$ and $g(0.1)=0$, so $\eta$ is continuous with the Table 4 values at both ends. The function $g$ is monotone and convex for $\kappa_s>0$. As $1/\kappa_s\to0$, expanding to first order in $u=1/\kappa_s$ gives $g\to1-x/0.1$, the linear interpolant. So "$1/\kappa_s=0$ means linear" is the continuous limit of the same family and introduces no discontinuity in $T$. Since $\eta(0)$ and $\eta(0.1)$ come from Table 4, $1/\kappa_s(T)$ is the only free parameter. It is fixed by one condition, the Eq. 27 value of $\bar\eta$. The shape is a PyRITE fit; it is not stated in SB85.

## Cheap filters

| Check | Result |
|---|---|
| Units | $\eta$, $\bar\eta$, $g$ and $1+\eta/Z$ are dimensionless; SDCS (mb) and DDCS (mb/sr) keep their units |
| Algebraic form | $(Z^2\chi_n+Z\chi_e)/(Z^2\chi_n)=1+\eta/Z$, not $1+\eta$ or $1+\eta/Z^2$ |
| Sign | $\eta\ge0$ in every tabulated entry, so the factor is $\ge1$ |
| Tip | $\eta(1)=0$ follows from Eq. 20, $k_{\max}<T_1$ |
| Soft limit | $g(0)=1$ returns Table 4's $\eta(0)$ |
| Linear limit | $1/\kappa_s\to0$ gives linear interpolation |
| DDCS normalization | a per-node scalar factor preserves $\int D\,d\Omega=\chi$ |

All filters pass.

## Numerical checks (independent of the implementation)

Scratch scripts parsed the pinned NIST `BREMX.DAT` (`~/.local/share/pyrite/validation/seltzer-berger/nist_brems.data`; layout 57 $T$ by 30 $x$ for each of $Z=1$–100, with $T$ varying fastest) and the packaged TOML. They used no PyRITE code. They set $\chi_n=\chi_{\rm tot}/(1+\eta/Z)$, interpolated $\chi_{\rm tot}$ linearly in $x$ on the SB nodes $0,0.05,0.1,\dots$, and integrated on a $10^{-8}$–1 grid of 6000 points.

**Continuous model.** The $\chi_n$-weighted mean of the modelled $\eta$, with the stored $1/\kappa_s$, against Table 4's $\bar\eta$:

| $Z$ | 1 MeV | 3 MeV | 10 MeV | 30 MeV | 100 MeV |
|---|---|---|---|---|---|
| 6 | $+0.0005$ | $+0.0002$ | $+0.0005$ | $+0.0008$ | $+0.0009$ |
| 29 | $-0.0006$ | $-0.0003$ | $0.0000$ | $+0.0002$ | $-0.0004$ |
| 92 | $+0.0006$ | $+0.0007$ | $-0.0004$ | $-0.0008$ | $-0.0001$ |

The largest $\lvert\Delta\bar\eta\rvert$ over all six $Z$ and all tabulated $T$ from 10 keV to 100 MeV is $0.0043$. That is within the ledger's 0.005 and close to Table 4's three-decimal rounding. For comparison, linear interpolation in $(0,0.1)$ overestimates $\bar\eta$ by 4–8 % at 1–100 MeV, and holding $\eta(0.1)$ flat underestimates it by 4–10 %. Both bracket the fit, confirming the row's Notes.

**Staged 18-node grid.** The staged table is evaluated at the nodes $x\in\{0,0.004,0.01,0.02,0.035,0.06,0.1,\dots,0.9,0.95,0.975,x_{\rm top}\}$, and runtime interpolation is linear between nodes. Integrated that way, with the top value held to $x=1$, the effective $\bar\eta$ is biased high by $+0.001$ to $+0.006$. The worst case is $+0.0062$ at $Z=6$, 3 MeV. About half the bias comes from chords across the convex $g$ between sub-nodes, and half from the coarser $x\ge0.1$ BremsLib grid, which loses Table 4's 0.97 and 0.99 nodes. The effect on radiative stopping is $\Delta\bar\eta/Z\le0.10\,\%$ for carbon and smaller for heavier $Z$. This is not a factor error. However, the sentence in [Bremsstrahlung](../../physics/radiation-physics/bremsstrahlung.md) that "the radiative stopping integral reproduces $1+\bar\eta/Z$ ... within 0.005 in $\bar\eta$" holds for the continuous model, not for the 18-node staged table. A caveat: here $\chi_n$ is SB's, used as a proxy for BremsLib's.

**Transcription.** These rows were compared entry by entry, including $\bar\eta$, against the page images:

- $Z=6$: 0.5, 10, 100 and 10000 MeV;
- $Z=47$: 0.5, 5, 10, 60, 100 and 10000 MeV, plus the $\bar\eta$ column;
- $Z=92$: 0.1, 3, 30 and 200 MeV.

All match. Two printed values for $Z=47$ are misprints. At 5 MeV, $x=0.10$ is printed as $0.012$, between the neighbours $0.989$ and $1.032$. At 60 MeV, $x=0.70$ is printed as $1.102$, between $0.995$ and $1.037$, which breaks monotonicity in $T$. Reading both as $1.012$ is consistent with the neighbouring values.

## Comparison with the implementation

`electron_electron_eta`:

- Interpolates the six $\eta$ tables linearly in $\ln Z$, clamped to $[6,92]$.
- Interpolates linearly in $\ln T$, clamped to $[10^{-3},10^{4}]$ MeV, and interpolates `inverse_small_kappa_scale` with the same $\ln T$ weights.
- Uses `np.interp` in $x$ on the Table 4 nodes.
- For $x<0.1$ with $1/\kappa_s>0$, replaces the value with $\eta(0.1)+[\eta(0)-\eta(0.1)]g(x)$. Here $\eta(0)$ and $\eta(0.1)$ are the $Z$- and $T$-interpolated table columns 0 and 1, and $g$ is the expression derived above, with `edge` $=(1+0.1/\kappa_s)^{-1}$.

`electron_electron_factor` returns `1.0 + eta / float(atomic_number)`, which is the $1+\eta/Z$ derived above.

`_refine_reduced_energy` inserts $x=0.004,0.01,0.02,0.035,0.06$ with weight $w=x/0.1$ on both SDCS and DDCS. `_along_reduced_energy` in `brem_bremslib.py` evaluates the runtime value piecewise-linearly in $x$, so the sub-node values are the electron–nucleus interpolant the runtime already served, and that interpolant is unchanged. The DDCS is linear in the same weights, so every sub-node DDCS still integrates to its SDCS.

`prepare_bremslib_table` calls the factor after both refinements, evaluated at the staged nodes with the last node at `top_reduced_energy`. It multiplies the SDCS by `factor` and the DDCS by `factor[..., None]`. A per-node scalar preserves the angular normalization. With `electron_electron=False` the product is skipped.

Symbolic comparison: every term matches. The factor is $1+\eta/Z$ with $Z$ to the first power, $\eta$ is used rather than $\bar\eta$ at each node, and the $g$ shape is identical. The interpolation axes ($\ln Z$, $\ln T$, linear $x$) match the ledger. The anchor tests `tests/montecarlo/brem/test_bremsstrahlung_electron_electron.py` and the electron/sub-node nodes in `test_bremsstrahlung_bremslib.py` pass locally (8 passed).

## Findings (wording only; no divergent factor)

1. **Staged-grid $\bar\eta$ tolerance.** The ledger limiting case ("the $\chi_n$-weighted mean of $\eta$ reproduces $\bar\eta$ within 0.005") is true for the continuous $\eta$ model (max $0.0043$). For the 18-node staged table that consumers integrate, the bias is $+0.001$ to $+0.006$ (max $0.0062$, C at 3 MeV). The physics page's "radiative stopping integral ... within 0.005" should say "continuous model", or "within 0.007 on the staged grid".
2. **"Linear in $k/T$ below 0.5 MeV."** $1/\kappa_s=0$ only at tabulated $T\le0.2$ MeV. Between 0.2 and 0.5 MeV the scale ramps linearly in $\ln T$ from 0 to 13.6, so $\eta$ is mildly curved there. This is continuous and harmless, but the ledger's Assumptions statement is loose.
3. **Data-file comment.** The comment "Below 0.5 MeV eta(0) <= eta(0.1)" is false at 0.1–0.2 MeV: $Z=6$ has $0.237>0.219$ at 0.1 MeV and $0.461>0.378$ at 0.2 MeV, and $Z=92$ has $0.351>0.324$ at 0.2 MeV. The linear rule still reproduces $\bar\eta$ there, within the 0.0043 bound above, so only the comment is wrong.
4. **Above $x_{\max}$.** At high $T$, interpolating linearly between Table 4's 0.97/0.99 nodes and the staged 0.975/top nodes spreads a small $\eta$ into $x_{\max}<x<1$, where the exact $\chi_e$ is zero. This is part of the staged bias in finding 1 and is not separately significant.
5. **Angular assumption.** Giving the electron–electron photon the electron–nucleus DDCS shape is declared and is not testable against SB85.

## Verdict

`rederived`. Independent derivation from SB85 Eqs. 18–20, 27–29 reproduces several things:

- the factor $1+\eta/Z$;
- the $\chi_n$-weighted definition of $\bar\eta$;
- the tip limit;
- continuity of $g$ at both ends;
- the linear limit of $g$;
- the $\bar\eta$ reproduction of the continuous model within 0.005 for all six $Z$ from 10 keV to 100 MeV.

The implementation matches symbolically. The findings concern documentation precision only. The small-$k/T$ shape and the angular assignment are PyRITE modelling choices that SB85 does not supply. Human sign-off remains pending.

## Sub-node scheme (#1349, fresh-context verification 2026-11-20)

Scope: the electron–nucleus SDCS at the five sub-nodes $x=k/T_1\in\{0.004,0.01,0.02,0.035,0.06\}$ inserted at each library $T_1$ (`_bremslib_reduced_energy.py::refine_reduced_energy`), the Born table it reads (`xsgen/bremslib/read.py::parse_born_table`, `xsgen/bremslib/convert.py::build_table`), and the call order in `_bremslib_table.py::prepare_bremslib_table`. The $1+\eta/Z$ factor above is unchanged and was not re-derived.

### Source scheme

The scheme was reconstructed from the Interpolate_DCS v1.0.7 manual (Sec. 2 pp. 2–3; Sec. 4 parameters 30–33, 38–58), `Default_parameters.inp`, and the knot files `k_T1_knots1.txt` and `k_T1_knots_ref_ratio.txt`. The Fortran sources (`Interpolate_DCS.f90`, `Spline_fit/Spline_fit.f90`) were read for the exact fitting rules; nothing was copied. Let $x_j$ be the 13 library nodes, with the top node $x_{\rm top}=0.99$ for $T_1\le5$ keV, $1-50\,{\rm eV}/T_1$ up to 0.5 MeV and $0.9999$ above. Let $y_j$ be the scaled SDCS.

- $T_1<T_{\rm thr}=30$ keV (strict; `knots0` is empty): $S(x)$ is linear in $x$ between $x_0$ and $x_1$.
- $30\ {\rm keV}\le T_1\le0.999$ MeV: $S$ is the cubic spline $s$ that is $C^2$ on the knots $\{0,0.1,0.3,0.5,0.6,0.7,0.9,0.975,x_{\rm top}\}$. Its parameters are the knot values and both end slopes (`SplineFit0`: $m+2$ unknowns). It minimises

$$
\sum_{j=0}^{12} w_j\bigl(s(x_j)-y_j\bigr)^2,\qquad
w_j=\frac{1}{y_j^2}\times\min\!\Bigl(1,\bigl(0.01/\lvert\delta_j\rvert\bigr)^2\Bigr),
$$

where $\delta_j$ is the relative uncertainty.

- $T_1>0.999$ MeV: let $B_i$ be the 20-node Born SDCS at $x\in\{0,10^{-4},\dots,0.054,0.1,\dots,0.9\}$. Then $S(x)=r(x)\,b(x)$. Here $r$ is the same weighted fit of $y_j/B(x_j)$ over $x=0,\dots,0.9$ (10 points) with knots $\{0,0.2,0.4,0.5,0.7,0.8,0.9\}$. $b$ is the program's Born "fit" with $n_{\rm knots}=20-2$: knot indices $1,3,\dots,18,20$, with free end slopes. That is 20 parameters for 20 points, so it interpolates exactly with no knot at the 2nd and 19th points, which is the not-a-knot interpolating spline.
- The program also smooths $\ln y$ against $\ln T_1$ (`T1_spline=2`: a least-squares spline with knots at every other grid $T_1$, 0.01–30 MeV, or to 100 MeV at $x=0$). It replaces $y_j$ by the smoothed value and adds the smoothing residual in quadrature to $\delta_j$. A non-positive Born value is excluded from that column's $\ln B$–$\ln T_1$ not-a-knot spline (`exclude_zero`), so it is interpolated from neighbouring $T_1$.

The PyRITE intent, as declared, omits the $T_1$ smoothing and the $\delta_j$ weighting ($w_j=1/y_j^2$). It adds the anchor

$$
S_{\rm sub}(x)=S(x)+\Bigl(1-\frac{x}{0.1}\Bigr)\bigl(y_0-S(0)\bigr)+\frac{x}{0.1}\bigl(y_1-S(0.1)\bigr),
$$

so the library nodes are served unchanged. A sub-node shape is $(1-x/0.1)\,f_0+(x/0.1)\,f_1$ of the unit-integral neighbour shapes, so $\int_{4\pi}{\rm DDCS}=S_{\rm sub}$.

### Cheap filters

| Filter | Result |
|---|---|
| Units | pass: scaled SDCS in mb throughout; the Born ratio is dimensionless; the shape mix has unit integral |
| Linear data | pass: the plain fit contains straight lines and the anchor adds zero, so a linear $y$ is exact. The Born scheme is exact only at the anchored nodes, as declared |
| $T_1<30$ keV | pass: linear (`t1 < 0.03`, same strict test as the program). $T_1=30$ keV is a spline row in both |
| Nodes unchanged | pass: $y_0,y_1,\dots,y_{12}$ are bit-identical at 66 $(Z,T_1)$ rows |
| DDCS integral | pass: staged tables built end to end from the library for Z = 3 and 5 (`build_table` → `prepare_bremslib_table`, e–e off) have $\max\lvert\int{\rm DDCS}/{\rm SDCS}-1\rvert\le7\times10^{-16}$ |
| Call order | pass: sub-nodes at library $T_1$ rows, then the $T_1$ refinement, then the $1+\eta/Z$ factor |

### Numerical comparison

The independent implementation is a scratch script outside the repository that does not import `pyrite._bremslib_reduced_energy`. It builds its own B-spline design matrix on the clamped knot vector, uses SciPy not-a-knot `CubicSpline`, and parses the library files itself. It was compared with the Linux executable (defaults, run in a scratch directory) and with the code.

1. **Full program model vs executable.** Adding the $T_1$ smoothing, the $\delta_j$ weights and the per-column Born exclusion reproduces `Interpolate_DCS.exe` to $\le1.5\times10^{-14}$ at all 66 rows: Z = 1, 3, 5, 6, 29, 79 and $T_1$ = 0.01, 0.02, 0.03, 0.04, 0.1, 0.5, 0.8, 1, 3, 10, 30 MeV. This includes Z = 5, 3 MeV. The scheme above is therefore the program's complete $k/T_1$ algorithm for the SDCS.
2. **Declared scheme vs code.** At the same 66 rows, the code matches the independent anchored scheme to $\le2\times10^{-15}$. At Z = 5, 3 MeV the comparison is against the plain-spline fallback, and the `PhysicsFallbackWarning` fires there and nowhere else. Staged tables built end to end for Z = 3 and 5 agree to $1.2\times10^{-15}$.
3. **Regression pins.** In `test_bremslib_reduced_energy.py`, `INTERPOLATE_DCS` matches the executable to $3\times10^{-8}$ (its 8-digit rounding). `LI_SDCS` and `LI_BORN_1MEV` match `SDCS_3.txt` and `SDCS_Born_3.txt` to $1.3\times10^{-10}$. The code's sub-nodes are within $8.5\times10^{-5}$ (0.5 MeV) and $1.2\times10^{-4}$ (1 MeV) of the pins, inside `rtol=3e-4`. The tests pass (11 selected).

Served values are the runtime linear interpolation on $\{0,\text{sub-nodes},0.1\}$. They were compared with the executable at $x$ = 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03, 0.035, 0.045, 0.06, 0.07, 0.08 and 0.09, on every library $T_1$ row from 10 keV to 30 MeV:

| $Z$ | served vs exe | sub-nodes vs exe | chord of own curve | linear (before) |
|---|---|---|---|---|
| 3 | 0.651 % (10 MeV) | 0.651 % (10 MeV) | 0.558 % (0.6 MeV, $x\approx0.079$) | 5.77 % (0.8 MeV) |
| 5 | 0.763 % (10 MeV) | 0.763 % (10 MeV) | 0.476 % (0.6 MeV) | 4.73 % |
| 79 | 0.144 % (1 MeV) | 0.064 % | 0.133 % | 0.62 % |
| 1 | 1.240 % (0.5 MeV) | 0.173 % | 1.182 % (0.6 MeV) | 13.15 % |
| all 1–100 | 1.240 % (H, 0.5 MeV) | 1.215 % (Z = 24, 30 MeV) | — | 13.15 % |

### Findings

The code has no divergent term. Four documentation figures are wrong or too broad:

1. **Uncertainty down-weighting "< 1e-4".** This holds for the sampled Z = 1, 3, 5, 6, 29, 79, where no library $\lvert\delta_j\rvert$ exceeds 1 %. Across all Z, the library `RelErr` reaches 0.166 at 0.03–30 MeV. With the program's smoothing on, switching the weighting off moves the sub-nodes by up to 1.03 % (Z = 39, 30 MeV), and by more than $10^{-4}$ in 321 of 3100 rows. With raw values and raw $\delta_j$ the effect is up to 0.52 %. The largest effects are at Z = 34–46, 25–30 MeV. The omission itself is justified, because released tables drop `sdcs_rel_err`. The stated magnitude is not. This affects the `_bremslib_reduced_energy` module docstring, the ledger Assumptions and step 2 of the physics page.
2. **Born vs plain spline "≤ 1.5 % for Z = 1–79".** The bound is 1.498 % for the six sampled Z. Over all Born rows (1–30 MeV) the maximum is 2.99 % (Z = 43, 30 MeV), and 29 of 1599 rows exceed 1.5 %, all at 25–30 MeV. The statement should name the sampled Z or say ≤ 3.0 %. This affects the ledger Notes and the module docstring.
3. **Chord "up to 0.53 %".** On the sampled grid the chord of the code's own curve reaches 0.558 % (Li, 0.6 MeV, $x\approx0.079$). Say "≤ 0.56 %". Also, "the rest (≤ 0.8 %, Li 10 MeV)" misattributes the maximum: Li's sub-node gap peaks at 0.65 % (10 MeV), and the 0.76 % maximum is B at 10 MeV. That gap is the $T_1$ smoothing together with, for some Z, the $\delta_j$ weighting.
4. **The 0.76 % / 0.8 % agreement** is correct as scoped (Li, B, Au; 0.763 %). Over all Z the served values are within 1.24 % (H, chord-dominated) and the sub-nodes within 1.22 % (Z = 24, 30 MeV). A reader may otherwise generalise the scoped figure.

Note on the Born-zero fallback. The program does not fall back at Z = 5, 3 MeV: it drops the zero from that column's $\ln T_1$ spline and keeps the Born-ratio scheme. The code's plain-spline fallback is a declared deviation. It sits 0.09–0.24 % below the executable at the five sub-nodes. Interpolating the Born value in $\ln T_1$ instead would reproduce the program, which is optional.

### Verdict

`rederived` for the sub-node part. The independent reconstruction of the program's $k/T_1$ scheme reproduces the executable to round-off. With the declared omissions it matches the code to $2\times10^{-15}$ at every tested row and end to end. Findings 1–3 are documentation corrections, not code discrepancies. Human sign-off remains pending.
