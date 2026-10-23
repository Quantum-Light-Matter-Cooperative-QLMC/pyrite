# `bremslib-high-energy-extension`

## Scope and question

Validation: `bremslib-high-energy-extension`. This is a fresh-context verification (2026-10-23) for #439. It was written from the ledger row, the derivation docstrings and Koch and Motz Formula 2BS before the implementation bodies were read. The question has two parts. Does the staged BremsLib table fill the holes above the last complete row ($T_c = 30$ MeV) by the stated SDCS tip rule and the stated angular-shape rule? Do the stated limiting cases hold? Whether the model is physically right above 30 MeV is a separate question, evidenced only by the back-test and ESTAR numbers in the ledger. This document does not re-measure them.

Code under review: `xsgen/bremslib/convert.py::build_table`, `_bremslib_extension.py::extend_above_complete`, `_bremslib_extension.py::schiff_density`, `_bremslib_table.py::prepare_bremslib_table`.

## Independent derivation

### Notation

$T$ is the incident kinetic energy, $x = k/T$ the reduced photon energy, and $\chi(T,x)$ the scaled SDCS. The library's unit-normalized angular shape is $S_{\rm BL}(T,x,\theta)$, with

$$
\int S_{\rm BL}\,d\Omega = 2\pi\int_0^\pi S_{\rm BL}(\theta)\sin\theta\,d\theta = 1 .
$$

The library $x$ grid is $\{0, 0.1, \dots, 0.9, 0.95, 0.975, x_{\rm top}\}$, 13 columns. Above 500 keV, $x_{\rm top} = 0.9999$. An extension row $T > T_c$ carries $\chi(T,x)$ for $x \le 0.95$ (11 columns) and the DDCS at $x = 0$ only.

### (a) SDCS tip

Assume that the ratio of each tip column $x_m \in \{0.975, x_{\rm top}\}$ to the last library column does not depend on $T$ for $T \ge T_c$:

$$
\frac{\chi(T,x_m)}{\chi(T,0.95)} = \frac{\chi(T_c,x_m)}{\chi(T_c,0.95)}
\quad\Longrightarrow\quad
\chi(T,x_m) = \chi(T,0.95)\,\frac{\chi(T_c,x_m)}{\chi(T_c,0.95)} .
$$

The ratio is dimensionless, so $\chi(T,x_m)$ keeps the units of $\chi$ (mb). At $T = T_c$ it is the identity. $x_{\rm top} = 0.9999$ on both rows, so the ratio is taken at the same physical node. Every library value $\chi(T,x\le0.95)$ is untouched.

### (b) Koch and Motz Formula 2BS

Koch and Motz, *Rev. Mod. Phys.* **31**, 920 (1959), Table I, Formula 2BS (Schiff 1951, Born approximation, Thomas–Fermi screening, small angle, $E_0 \gg 1$). Energies are total energies in units of $mc^2$: $E_0 = 1 + T/mc^2$, $E = E_0 - k$ with $k$ in $mc^2$, and $y = E_0\theta$.

$$
d\sigma_{k,\theta}=\frac{4Z^2r_0^2}{137}\frac{dk}{k}\,y\,dy
\left\{\frac{16y^2E}{(y^2+1)^4E_0}-\frac{(E_0+E)^2}{(y^2+1)^2E_0^2}
+\left[\frac{E_0^2+E^2}{(y^2+1)^2E_0^2}-\frac{4y^2E}{(y^2+1)^4E_0}\right]\ln M(y)\right\},
$$

$$
\frac{1}{M(y)}=\left(\frac{k}{2E_0E}\right)^2+\left(\frac{Z^{1/3}}{111\,(y^2+1)}\right)^2 .
$$

Call the brace $B(Z,T,x,\theta)$. The measure is $y\,dy = E_0^2\,\theta\,d\theta$. For small angles $\theta\,d\theta \approx \sin\theta\,d\theta = d\Omega/2\pi$, so $B$ is a density per steradian up to a factor that depends only on $T$. At large $\theta$ the Jacobian between "per $y\,dy$" and "per sr" is $\theta/\sin\theta$. The next subsection shows that this ambiguity has no effect on the extension.

### (c) Angular shape for $x > 0$

Define the Schiff ratio at fixed $(T,\theta)$:

$$
R(T,x,\theta) = \frac{B(T,x,\theta)}{B(T,0,\theta)} ,
$$

the residual on the last complete row:

$$
C(x,\theta) = \frac{S_{\rm BL}(T_c,x,\theta)}{S_{\rm BL}(T_c,0,\theta)\,R(T_c,x,\theta)} ,
$$

and the extension shape:

$$
\tilde S(T,x,\theta) = S_{\rm BL}(T,0,\theta)\,R(T,x,\theta)\,C(x,\theta),
\qquad
S(T,x,\theta) = \frac{\tilde S}{2\pi\int_0^\pi \tilde S\sin\theta\,d\theta} .
$$

**Invariances (measure check).** Replace $B$ by $a(T,x)\,J(T,\theta)\,B$, where $a$ is any per-node constant and $J$ is any $\theta$ Jacobian at fixed $T$, such as $\theta/\sin\theta$ or the $E_0^2$ in $y\,dy$. Then $R(T)$ picks up $a(T,x)/a(T,0)$ and $J$ cancels. $C$ picks up the inverse factor at $T_c$, so $\tilde S$ changes by a constant in $\theta$ that the final normalization removes. Likewise, the normalization constants of $S_{\rm BL}$ cancel. The only measure that affects the result is therefore the one used in the final normalization. It must be $d\Omega = 2\pi\sin\theta\,d\theta$, the same measure as the library's $S_{\rm BL}$. One consequence: the prefactor, the $1/k$ and the per-$dy$ versus per-sr question in Formula 2BS cannot bias the extension. The cost is that $R$ must be formed at one $T$ and one $\theta$ grid.

**Limits.**

- $x = 0$: $R = 1$ and $C = 1$ identically, so $S(T,0) = S_{\rm BL}(T,0)$, the library node.
- $T \to T_c$: $\tilde S = S_{\rm BL}(T_c,0)\,R(T_c)\,C = S_{\rm BL}(T_c,x)$. That is already unit-normalized, so the complete row is recovered exactly wherever $S_{\rm BL}(T_c,0)R(T_c) > 0$.
- $x \to 0^+$: $B$ is continuous in $k$ at every fixed $\theta$, so $R \to 1$ pointwise. The limit is not uniform in $\theta$. At large $y$ the screening term falls off as $y^{-4}$, and the $(k/2E_0E)^2$ term dominates $1/M$ even for small $k$. This is a property of 2BS, not a defect. It is harmless here, because the extension evaluates $R$ only at library columns $x \ge 0.1$. Staging then interpolates linearly in $x$ between the $x = 0$ node and the 0.1 node, so keV photons get the library $k = 0$ shape to $O(x/0.1)$.
- Positivity: $C$ and the result need $B > 0$. At large $y$, $\ln M \to 2\ln(2E_0E/k)$, and positivity needs $\ln M > (E_0+E)^2/(E_0^2+E^2) \le 2$. At the tip $x = 0.9999$, $k/(2E_0E) \approx 0.49$, so $\ln M \approx 1.43$, against a right-hand side of about 1.01 at 100 MeV. It is positive. This was confirmed numerically below.

**Modelling choice, not a convention error.** $C$ is held fixed at the same absolute $\theta$, not at the reduced angle $E_0\theta$. The ledger and the docstring both state $C(x,\theta)$, and the $T$-dependent narrowing of the forward peak enters through $S_{\rm BL}(T,0)$ and $R(T)$. A reduced-angle residual would be an equally defensible alternative. The ledger's 15 to 30 MeV back-test (1–3 % in $\theta_{50}$, $\theta_{90}$) supports the fixed-$\theta$ choice in the peak. It does not support it in the tail beyond 1 rad (10–55 %).

### (d) Normalization ordering

To reproduce the definition above, staging must:

1. normalize the complete-row shapes and the extension $x = 0$ shapes over $d\Omega$;
2. form $\tilde S$;
3. normalize $\tilde S$ over $d\Omega$;
4. multiply by $\chi$, so that every DDCS node integrates to its SDCS.

The same quadrature rule must be used in steps 1–3 and in the runtime angular integral. Refinement in $T$ and $x$ must come after the fill, so that sub-nodes interpolate filled nodes.

## Comparison with the implementation

| Item | Derivation | Implementation | Result |
|---|---|---|---|
| Tip | $\chi(T,0.95)\,\chi(T_c,x_m)/\chi(T_c,0.95)$ | `sdcs[c+1:, 11:] = sdcs[c+1:, 10, None] * sdcs[c, 11:] / sdcs[c, 10]` with `_LIBRARY_COLUMNS = 11` | matches |
| Tip $x$ | same node on both rows | uses column index; `top_reduced_energy` is 0.9999 for both 30 and 50–100 MeV | matches |
| 2BS bracket | Formula 2BS | `schiff_density`: same three terms, $\ln M$ as `-np.log(inv_M)`, $E_0 = 1 + T/mc^2$, $k = xT/mc^2$, $E = E_0 - k$, $y^2 = (E_0\theta)^2$ | matches |
| $R$ | $B(T,x)/B(T,0)$ at fixed $T$ and $\theta$ | `schiff_ratios(row)` on `theta_common`, with $x$ from `reduced[row, 1:]` (actual top node) | matches |
| $C$ | $S_{\rm BL}(T_c,x)/[S_{\rm BL}(T_c,0)R(T_c)]$ | `residual = shape[c,1:] / (shape[c,0]*schiff_ratios(c))`, set to 0 where the denominator is $\le 0$ | matches where the denominator is $> 0$ (see note) |
| $\theta$ grid | one grid for $T$ and $T_c$ | every node resampled linearly onto the union grid `theta_common` | matches; exact for each node's own linear interpolant |
| Normalization | $d\Omega$, before and after the fill | `solid_angle_integral` (exact $2\pi\int f\sin\theta$ of the linear interpolant) on complete rows and extension $x = 0$, then on all nodes after `extend_above_complete` | matches |
| Ordering | fill, then refine | `extend_above_complete`, then `_refine_incident_grid`, then `_refine_reduced_energy`, then the $1+\eta/Z$ factor | matches |
| Stored data | library kept, tip missing | `build_table` keeps rows with all 11 leading columns $> 0$ and a $k = 0$ node, stores the tip as NaN, and counts `complete_rows` | matches |
| Refusal | no fill without data | `build_table` drops rows lacking columns or the $k = 0$ node. `_check_complete` requires 13 nodes on complete rows and 1 on extension rows. `extend_above_complete` raises on no complete row, on non-finite or non-positive library SDCS, and on non-positive $B$. `prepare_bremslib_table` raises on a missing extension $k = 0$ node or a zero integral | matches |

Note on the `where=anchor > 0` guard: if $S_{\rm BL}(T_c,0,\theta) = 0$ at some $\theta$ while $S_{\rm BL}(T_c,x,\theta) > 0$, the residual is silently zeroed there, and the $T \to T_c$ limit fails at that point. The released library has no such point (see below), so the guard is inert today. It is not a refusal, though, and a future library with a zero in the $k = 0$ node would be truncated rather than rejected. This is a robustness finding, not a discrepancy.

## Numeric checks (2026-10-23)

The scratch scripts were run against the installed release tables. They used an independent 2BS transcription with the CODATA 2022 $mc^2$ from SciPy, and an independently written exact $\int f\sin\theta$ for the linear interpolant. The comparison does not reuse `extend_above_complete`, `schiff_density` or `solid_angle_integral`.

- **2BS transcription.** For $Z \in \{1,6,14,29,74,92,100\}$, $T \in \{5,30,50,100\}$ MeV, $x \in \{0,0.05,0.5,0.95,0.975,0.9999\}$ and 2001 angles on $[0,\pi]$, the maximum relative difference is $5.3\times10^{-9}$. The residual comes from the $mc^2$ constant: the code's 0.51099895 MeV against CODATA 2022's 0.51099895069. $B > 0$ everywhere in this sweep.
- **Full reconstruction, C, Si and W.** The library rows are 30 MeV ($T_c$, `complete_rows = 66`) and 50, 60, 80, 100 MeV. Against `prepare_bremslib_table(..., electron_electron=False)` at the original nodes:
  - extension SDCS: $\le 2.2\times10^{-16}$ relative;
  - DDCS: $\le 6.6\times10^{-10}$ of the node maximum, the same $mc^2$ residual;
  - 30 MeV row: $4\times10^{-16}$;
  - every node's angular integral equals its SDCS to $4.4\times10^{-16}$.

  The held tip ratios $\chi(x_m)/\chi(0.95)$ are $(0.6986, 0.0831)$ for C, $(0.7059, 0.1482)$ for Si and $(0.7772, 0.3505)$ for W.
- **$T \to T_c$.** Applying the extension formula to the 30 MeV row reproduces its library shapes to $4\times10^{-16}$ for all three elements.
- **Guard is inert.** No $\theta$ point with $S_{\rm BL}\le0$ occurs on the 30 MeV row (all $x$) or on any extension $x = 0$ node for C, Si or W.
- **Measure invariance.** On a synthetic four-row table, replacing `schiff_density` by $(\sin\theta + 10^{-3})(1 + 7x + T)\,B$ changes the raw filled shapes by up to 12.8 %. After the $d\Omega$ renormalization the change is $1.1\times10^{-15}$. This confirms that the result depends only on the final normalization measure.
- **Synthetic limits and refusal.** The $x = 0$ column and the complete rows are returned bit-identical. A NaN or a zero in an extension row's $x \le 0.95$ SDCS, or `complete_rows = 0`, each raises `ValueError`.
- **$x \to 0^+$ of 2BS** (W, 100 MeV, $\theta \in [0,\pi]$): $\max\lvert R-1\rvert$ is 0.37, 0.067 and $2.3\times10^{-5}$ at $x = 10^{-2}$, $10^{-4}$ and $10^{-6}$. The limit is pointwise and non-uniform at large $\theta$, as derived. Staging never evaluates $R$ below $x = 0.1$.
- **Anchors.** `tests/montecarlo/test_bremsstrahlung_extension.py` and the `build_table` extension test in `tests/xsgen/test_bremslib.py`: 15 passed.

## Verdict

- **Claim:** `bremslib-high-energy-extension`. The implementation computes exactly the stated tip rule and the stated angular-shape rule, with the residual $C$ on the common $\theta$ grid, normalized over $d\Omega$ before and after the fill. The library values are kept, and rows lacking data are refused.
- **Re-derivation:** matches. No divergent factor. The 2BS angular measure ($y\,dy$ versus $d\Omega$) and all per-node prefactors cancel in the ratio-of-ratios, so they cannot bias the result.
- **Findings, not discrepancies:**
  1. the `anchor > 0` guard zeros the residual instead of refusing; it is inert on the released library;
  2. $C$ held at fixed absolute $\theta$ rather than reduced angle is an unvalidated modelling assumption beyond the 15 to 30 MeV back-test, weakest in the tail beyond 1 rad;
  3. physical accuracy above 30 MeV rests on the ledger's ESTAR and back-test evidence, not on this verification.
- **Status:** `rederived` suggested. Only a human may mark it signed off.
