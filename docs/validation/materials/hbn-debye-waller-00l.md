# `hbn-debye-waller-00l`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 3.45` is the room-temperature h-BN basal Debye–Waller coefficient $B_{33}$ (with $U_{33}=0.0437$ Å$^2$). It applies only to the pinned $(002)/(004)$ reflections.
- Code: `data/catalog/crystals/hbn.toml::B_ang2`. The cell is in `data/cifs/hbn.cif`.
- Source: R. S. Pease, *Acta Cryst.* **5**, 356–361 (1952), doi:10.1107/S0365110X52001064. Table 1, Fig. 3 and the text on p. 358. Local PDF: Zotero `RV4Z2ZWJ`.
- Ledgered check: a weighted fit of $\ln[(F_o/F_c)^2]$ against $l^2$ over $002$–$008$ gives slope $-0.03888(834)$. That yields $B_{33}=-2c^2\cdot\text{slope}=3.45(74)$ Å$^2$ and $U_{33}=0.0437(94)$ Å$^2$.
- Intended quantity and units: an amplitude Debye–Waller coefficient for the $c^*$ direction, in Å$^2$. It enters as $\exp(-B_{33}s^2)$ with $s=\sin\theta/\lambda$.
- Scope (#1480): $B_{33}$ is valid **only for g parallel to c**. Non-basal h-BN reflections need $W=\tfrac12(U_\perp g_\perp^2+U_{33}g_z^2)$ with a sourced in-plane $U_\perp$, which PyRITE does not package; graphite values must not be reused.

This pass is a fresh-context verification. The derivation and fit below come from the source PDF and the ledger row only. They were written before reading the TOML, the CIF, `debye_waller`, or any other Debye–Waller write-up.

## Source extraction

Each item was checked against the page image of p. 358, not just the text layer.

- **Tabulated quantity.** Pease's text says: "The observed values of $F^2_{hkl}$, together with those calculated for structures A and B, are given in Table 1." The table title is "Observed and calculated $F^2_{hkl}$", and the column headers are $F_c^2(A)$, $F_c^2(B)$ and $F_o^2$. **Every Table 1 entry is already a squared structure factor.**
- **Model behind $F_c$.** "The calculated values are based on atomic scattering powers given in the International Tables, and contain no allowance for temperature factor." So $F_c^2$ is the static ($B=0$) value, and $\ln(F_o^2/F_c^2)$ measures the full $B$, not a residual.
- **Scale.** The $110$ reflection is the standard: $F_c^2=F_o^2=100$. The $00l$ values are relative to it, and structures A and B give the same $F_c^2$ for $00l$.
- **Temperature.** Pease calls these "the room-temperature results" (p. 358). Lattice spacings were measured at $35\pm0.5$ °C, where $c=6.6612\pm0.0005$ Å. His expansion law $c=6.6516+2.74\times10^{-4}\,t$ gives $c=6.657$ Å at 20 °C.
- **Fig. 3.** The plot is "$\log_e(F_o/F_c)^2$ versus $l^2$ for the $00l$ reflexions". It shows a straight line from about $+0.39$ at $l^2=0$ to about $-1.05$ at $l^2\approx71$, so its slope is about $-0.020$. Pease: "The slope of the line corresponds to a Debye temperature of 410° K." He does not state a numerical $B$ or slope, his weights, or the formula he used for $\Theta$.

Transcribed $00l$ rows of Table 1 (all values are $F^2$):

| $hkl$ | $F_c^2$ | $F_o^2$ | $l^2$ | $y=\ln(F_o^2/F_c^2)$ | $\sigma_y=\sigma(F_o^2)/F_o^2$ |
|---|---|---|---|---|---|
| 002 | 390 | $560\pm100$ | 4 | $+0.3618$ | 0.1786 |
| 004 | 150 | $158\pm10$ | 16 | $+0.0520$ | 0.0633 |
| 006 | 86 | $60\pm10$ | 36 | $-0.3600$ | 0.1667 |
| 008 | 59 | $26\pm6$ | 64 | $-0.8194$ | 0.2308 |

The $y$ values agree with the plotted Fig. 3 points: about $+0.36$, $+0.05$, $-0.36$ and $-0.82$. This confirms that Fig. 3 plots the logarithm of the Table 1 ratio as printed, with no second squaring. Fig. 3's error bars for 002 and 004 match $\sigma_y$. The bars for 006 and 008 look narrower than $10/60$ and $6/26$ (roughly $\pm0.06$ and $\pm0.17$), so their weights are tested below.

## Independent derivation

An atom with mean-square displacement $\langle u_z^2\rangle$ along $c$ multiplies its scattering amplitude by $\exp(-2\pi^2\langle u_z^2\rangle g_z^2)$. In the anisotropic convention $T=\exp(-2\pi^2\sum U_{ij}h_ih_ja^*_ia^*_j)$, the reflection $\mathbf h=(0,0,l)$ keeps only the $U_{33}$ term. In the hexagonal cell $c^*=1/c$, so

$$
T(00l)=\exp\!\left(-\frac{2\pi^2U_{33}l^2}{c^2}\right)
=\exp\!\left(-B_{33}s^2\right),\qquad
B_{33}=8\pi^2U_{33},\quad s=\frac{\sin\theta}{\lambda}=\frac{1}{2d_{00l}}=\frac{l}{2c}.
$$

Pease's $F_c$ is static. If B and N share one $U_{33}$, which is the only assumption under which a single basal scalar is exact, then with an unknown overall scale $K$:

$$
F_o^2=K\,F_c^2\,e^{-2B_{33}s^2}
\quad\Longrightarrow\quad
\ln\frac{F_o^2}{F_c^2}=\ln K-\frac{B_{33}}{2c^2}\,l^2 .
$$

So for a fit $y=\alpha+\beta l^2$ with $y=\ln(F_o^2/F_c^2)$,

$$
\boxed{B_{33}=-2c^2\beta},\qquad U_{33}=\frac{B_{33}}{8\pi^2}.
$$

The intercept has to be a free parameter. The data are scaled to $110$, which carries its own in-plane factor $e^{-2B_{11}s_{110}^2}$ and the errors in the scattering factors that Pease discusses. The fitted $\alpha\approx+0.37$ absorbs those effects. The slope does not depend on them.

The key point is that $y$ is the logarithm of the tabulated ratio itself. Writing $\ln[(F_o/F_c)^2]$ with Table 1 values put in as if they were amplitudes gives $2y$, which doubles both $\beta$ and $B_{33}$.

**Weights.** Only $F_o^2$ carries an uncertainty, and the theoretical $F_c^2$ is treated as exact. Propagation gives $\sigma_y=\sigma(F_o^2)/F_o^2$, and the fit is weighted least squares with $w=1/\sigma_y^2$. The parameter errors use the absolute covariance. The fit has $\chi^2=0.25$ for 2 degrees of freedom, so Birge scaling would not enlarge them.

## Fit results

The fit was run as a throwaway scratch script, not committed. It uses numpy only, with no PyRITE helpers. All fits use $c=6.6612$ Å.

| fit | slope $\beta$ | $B_{33}$ (Å$^2$) | $U_{33}$ (Å$^2$) |
|---|---|---|---|
| **Table 1 as $F^2$, weights $1/\sigma_y^2$ (correct)** | $-0.01944(417)$ | **$1.73(37)$** | **$0.0218(47)$** |
| unweighted | $-0.01945$ | 1.73 | 0.0219 |
| 006/008 weights read from Fig. 3 bars | $-0.0196$ to $-0.0202$ | 1.74–1.79 | 0.022–0.023 |
| omit 002 | — | $1.66(40)$ | 0.021 |
| $c=6.657$ Å (20 °C) or $c=6.661$ Å (CIF) | — | 1.723 / 1.725 | — |
| **Table 1 values squared again (ledger's procedure)** | $-0.03888(834)$ | $3.450(740)$ | $0.0437(94)$ |

The ledger's slope, $B_{33}$, $U_{33}$ and their uncertainties are reproduced **exactly** only by the last row, which squares the Table 1 values a second time. That is exactly a factor of 2.

## Cross-check against Pease's Debye temperature

Debye–Waller theory with zero-point motion, for a monatomic Debye solid of mass $m$, gives

$$
B=\frac{6h^2}{mk_B\Theta}\left[\frac{\phi(x)}{x}+\frac14\right],\qquad
\phi(x)=\frac1x\int_0^x\frac{t\,dt}{e^t-1},\quad x=\frac{\Theta}{T}.
$$

With $\Theta=410$ K, the mean BN atomic mass $\bar m=12.409$ u and $T=293$ K, this gives $B=1.70$ Å$^2$. The range is 1.67–1.78 Å$^2$ for $T=288$–308 K. Using the boron or nitrogen mass instead gives 1.95 or 1.51 Å$^2$. Leaving out the zero-point term gives 1.14 Å$^2$.

Inverting the fit gives $\Theta=407^{+55}_{-39}$ K for $B=1.73(37)$ Å$^2$, which matches Pease's 410 K. The catalog value of 3.45 Å$^2$ would imply $\Theta\approx284$ K, which contradicts the source. The visually read slope of the Fig. 3 line ($\approx-0.020$) gives the same answer: $B_{33}\approx1.8$ Å$^2$, not 3.45.

## Cheap filters

- **Units:** pass. $l^2$ is dimensionless, $\beta$ is dimensionless, $c^2$ is in Å$^2$, so $B_{33}$ is in Å$^2$. $8\pi^2$ is dimensionless.
- **Limits:** pass. $B_{33}\to0$ gives $F_o^2=KF_c^2$, a flat line. Larger $l$ or $B$ suppresses $F_o^2$. Only $U_{33}$ enters $00l$.
- **Signs/conventions:** the negative exponent and $B=8\pi^2U$ both pass. The conversion $B=-2c^2\beta$ is correct for a fit to $\ln F^2$. **Fails on one point:** the quantity tabulated in the source is $F^2$, not $F$, so $y$ must not be squared again.
- **Reference data:** $c$ and the $F_c^2$ and $F_o^2$ values (with their errors) match Table 1 and p. 357. The "room temperature" label and the 410 K value match p. 358. Fig. 3 is consistent with the transcription.

## Implementation comparison

Inspected after the derivation above was written.

- `crystals/hbn.toml` stores `B_ang2 = 3.45` and pins `hkl_families = [[0,0,2],[0,0,4]]`. That is the ledger value, which is twice the source-derived $1.73(37)$ Å$^2$. Its uncertainty interval does not contain the source value: the difference is $1.72$ Å$^2$, about $2.1\sigma$ of the ledger's own $\pm0.74$ and $4.7\sigma$ of the correct $\pm0.37$. Pease's 410 K gives 1.70 Å$^2$ independently.
- `data/cifs/hbn.cif` has $a=2.504$ Å and $c=6.661$ Å, which matches Pease's 2.50399 and 6.6612. The cell is not the source of the error: changing $c$ moves $B$ by less than 0.003 Å$^2$.
- `materials/crystal.py::debye_waller` returns $\exp[-B(g/4\pi)^2]=\exp(-Bs^2)$ on the amplitude, with intensity $\exp(-2Bs^2)$. That is exactly the convention derived above, so the code applies `B_ang2` correctly. The error is only in the stored number.
- Numeric impact on the pinned reflections ($c=6.661$ Å). The intensity ratio for 3.45 against 1.73 Å$^2$ is $\exp[-2(1.72)s^2]$. That is 0.925 for $(002)$ ($s^2=0.0225$ Å$^{-2}$, a $-7.5\%$ change in $\lvert F\rvert^2$) and 0.733 for $(004)$ ($s^2=0.0902$ Å$^{-2}$, a $-27\%$ change).
- The TOML's `validation_id` is `hbn-structure`. Nothing in the TOML names `hbn-debye-waller-00l`. This is a provenance-marker finding only.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions **fail** because the tabulated $F^2$ is squared twice.
- **Re-derivation:** **differs**. The first divergent factor is a **factor of 2 in the slope**. The ledger fits $\ln[(F_o/F_c)^2]$ with Table 1's $F_o^2$ and $F_c^2$ entered as amplitudes, which computes $2\ln(F_o^2/F_c^2)$. Pease tabulates $F^2$ (Table 1 title, column headers and text), and his $F_c^2$ has no temperature factor. The correct values are slope $-0.01944(417)$, $B_{33}=1.73(37)$ Å$^2$ and $U_{33}=0.0218(47)$ Å$^2$. Pease's 410 K Debye temperature gives 1.70 Å$^2$ independently. The Debye–Waller relation, the sign, $B=8\pi^2U$ and the code's application of `B_ang2` all agree.
- **Verdict:** `discrepancy`.
- **Suggested ledger change:** set Status to `discrepancy`. Change Checks to: "Table 1 tabulates $F^2$, and $F_c^2$ has no temperature factor. A weighted ($1/\sigma_y^2$, $\sigma_y=\sigma(F_o^2)/F_o^2$) fit of $\ln(F_o^2/F_c^2)$ against $l^2$ over 002–008 gives slope $-0.01944(417)$, $B_{33}=1.73(37)$ Å$^2$ and $U_{33}=0.0218(47)$ Å$^2$. This is consistent with $\Theta=410$ K, which gives 1.70 Å$^2$. The recorded $-0.03888(834)$/3.45(74) squared Table 1 a second time." Record in Notes that the catalog value of 3.45 is 2× too large: correcting it raises pinned $\lvert F_{002}\rvert^2$ by about 8% and $\lvert F_{004}\rvert^2$ by about 36%. Changing the catalog needs a separate authorized change, with a golden regeneration and a model/tables identity bump. A human applies the edit. Never mark this claim `signed-off`.

## Resolution (#1235)

The catalog now stores `B_ang2 = 1.73`, the fitted $B_{33}$ above, and the ledger row is `rederived`. The regenerated golden `(002)` structure-factor amplitude at 10 keV moves from 16.0921 to 16.7282. That ratio, 1.03953, equals $\exp(1.72\,s^2)$ with $s=1/c$, so only the Debye–Waller factor changed. The value stays a direction-specific basal $B_{33}$, not $B_{\rm iso}$.

The identity generation is `hbn-basal-b33-v2` (`campaign/profiles.py::_HBN_DEBYE_WALLER_MODEL`). Resolved case keys already hash `B_ang2`, but dataset crystallography omits it. The conditional marker invalidates both caches for h-BN films and radiating layers, preserving other materials’ identities.

## Current texture scope (#1301)

The catalog default represents **pyrolytic/fiber-textured BN**, by user decision,
with random in-plane grain azimuths and reciprocal `surface_hkl = [0, 0, 1]`.
`texture = "fiber"` replaces the fixed `(002)/(004)` pair: automatic selection
can retain higher allowed basal orders. Every candidate still has $g\parallel c$,
so the current `B_ang2 = 1.73` remains B33 and is never applied to an in-plane
vector by this selector. No mosaic broadening parameter changes.

An ordered single-crystal h-BN flake is a different orientation model. Its
in-plane reflections require independently sourced B11 data, which this catalog
entry does not supply. See
[the candidate-domain model](../../physics/materials/reflection-selection.md#fiber-texture).
