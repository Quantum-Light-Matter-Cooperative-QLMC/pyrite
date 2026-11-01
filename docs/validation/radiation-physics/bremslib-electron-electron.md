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
