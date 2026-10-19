# epdl-photon-differential-data

Independent fresh-context verification of the per-atom EPDL2025 differential photon data: subshell photoionization cross sections, the coherent form factor and the incoherent scattering function. It was written from the ledger row in `docs/validation/ledger-crystallography-atomic-data.md`, the public docstrings and signatures, the pinned upstream tape, and the ENDF-6 conventions. The implementation bodies (`src/pyrite/materials/photon_differential.py`, `scripts/release_epdl_table.py::build_differential_table`) were read only after the reference expressions and the independent reader below were fixed.

Signatures under review:

- `photoionization_subshells(element) -> tuple[PhotoionizationSubshell, ...]`, where each subshell has `designator`, `binding_energy_eV`, `energy_eV` and `cross_section_ang2`
- `subshell_photoionization_cross_sections_ang2(element, E_eV) -> (designators, array)`, per atom, in $\mathring{\mathrm A}^2$, with shape $(n_{\rm sub},\ \ldots)$
- `coherent_form_factor(element, x_inv_ang)`, which returns $F(x,Z)$ (dimensionless)
- `incoherent_scattering_function(element, x_inv_ang)`, which returns $S(x,Z)$ (dimensionless)

On the reference side, every number comes from `EPDL2025.ALL` (SHA-256 `59bbd8c5…c43fd`, verified before parsing) and `EADL2025.ALL`. Both were read by a new fixed-width 80-column ENDF-6 reader that shares no code with PyRITE or `endf_parserpy`. It parses the HEAD record, the TAB1 records (NR/NP, NBT/INT, $(x,y)$ pairs), the MF=23 `EPE`/`EFL` taken from C1/C2 of the TAB1 control record, and the MF=28 MT=533 `SUBI`/`EBI` lists. The scripts live outside the repository (`/tmp/claude-0/verify/endf.py`, `check.py`, `compare.py`, `sum522.py`, `edge.py`). The packaged table was read only through the public loaders.

## Independent derivation

### Momentum-transfer variable and its unit

For elastic scattering of a photon of wavelength $\lambda$ through angle $\theta$, the momentum transfer is

$$
q=\lvert\mathbf k'-\mathbf k\rvert=2k\sin\frac{\theta}{2}=\frac{4\pi\sin(\theta/2)}{\lambda}.
$$

The ENDF-6 File 27 abscissa is $x=\sin(\theta/2)/\lambda$, so $q=4\pi x$. With $\lambda$ in $\mathring{\mathrm A}$, $x$ is in $\mathring{\mathrm A}^{-1}$. The descriptive text on the tape writes this as `M = Momentum transfer = SIN(THETA/2)/LAMBDA/4*PI`. That string is ambiguous, so the unit and the $4\pi$ are fixed numerically by the hydrogen anchor below, not by the comment.

The kinematic range is $0\le x\le 1/\lambda$. For the 100 GeV top of EPDL, $\lambda=hc/E=1.24\times10^{-7}\,\mathring{\mathrm A}$, so $x_{\max}\approx 8\times10^{6}\,\mathring{\mathrm A}^{-1}$. A grid that ends at $10^{9}$ must therefore be in $\mathring{\mathrm A}^{-1}$. In $\mathrm{cm}^{-1}$, $10^9\,\mathrm{cm}^{-1}=10\,\mathring{\mathrm A}^{-1}$ would not cover Compton saturation at high energy.

### Hydrogen 1s closed forms

The ground-state density is $\rho(r)=e^{-2r/a_0}/(\pi a_0^3)$. Its Fourier transform is

$$
F_{\rm H}(q)=\int\rho(r)\,e^{i\mathbf q\cdot\mathbf r}\,{\rm d}^3r
=\frac{16}{(4+q^2a_0^2)^2}
=\Bigl[1+\Bigl(\frac{a_0q}{2}\Bigr)^2\Bigr]^{-2},\qquad q=4\pi x .
$$

For one electron, the incoherent scattering function in the Waller–Hartree form is $S=1-\lvert F\rvert^2$. The limits are $F(0)=1$, $S(0)=0$, $F\to0$ and $S\to1$ as $x\to\infty$.

### General limits

$F(0,Z)=\int\rho=Z$. Since $S(x,Z)=\sum_{n\ne0}\lvert\langle n\vert\sum_j e^{i\mathbf q\cdot\mathbf r_j}\vert0\rangle\rvert^2$, $S(0,Z)=0$. As $q\to\infty$ the cross terms dephase, $S\to Z$, and $F\to0$.

### Subshell cross sections, designators, binding energies

ENDF-6 MF=23 assigns MT 534, 535, … to K, L1, L2, L3, M1, … (the tape's MF=1 directory lists `534 K (1S1/2)`, `535 L1`, `541 M4 (3D3/2)`, `542 M5 (3D5/2)`). EADL MF=28 uses the same contiguous `SUBI` numbering, so the designator is

$$
i=\mathrm{MT}-533\quad(1={\rm K},\ 2={\rm L1},\ 3={\rm L2},\ 4={\rm L3},\ 5={\rm M1},\ \ldots).
$$

`EPE` (C1 of the TAB1 control record) is the subshell edge. With the 2023 "threshold start with XC=0" convention it equals the first energy knot. Units are barn per atom, and $1\,{\rm b}=10^{-24}\,{\rm cm}^2=10^{-8}\,\mathring{\mathrm A}^2$. The upstream sum rule is $\sum_i\sigma_i=\sigma_{\rm MT522}$.

### Evaluation law

Every TAB1 on the tape declares a single range with `INT = 2`, which means lin-lin. My reader confirmed this for all 1912 MF=23 MT 522/534–599 and MF=27 MT 502/504 sections. At a query $u$, take the last knot with $x_k\le u$ (search to the right) and interpolate linearly to $x_{k+1}$. At a repeated knot $x_k=x_{k+1}$ this selects the upper value, which is right-continuity. A subshell is zero below its first knot. Outside the tabulated domain ($1\,{\rm eV}\le E\le 100\,{\rm GeV}$, $0\le x\le10^9\,\mathring{\mathrm A}^{-1}$) the value is undefined, and NaN is the honest return.

A thinned lin-lin table that reproduces every skipped upstream node to relative tolerance $\epsilon$ also bounds the error between nodes. On each upstream interval the absolute difference is linear and the upstream value is linear, so their ratio is monotone and its extremes lie at the nodes. Upstream nodes plus interval midpoints therefore form a sufficient test set.

## Cheap filters

| Filter | Result |
|---|---|
| Units of $x$ | H 1s with $q=4\pi x$, $x$ in $\mathring{\mathrm A}^{-1}$: $\max\lvert F-F_{\rm H}\rvert=3.8\times10^{-4}$ and $\max\lvert S-(1-F_{\rm H}^2)\rvert=5.9\times10^{-4}$ over $x=0.02$–$1\,\mathring{\mathrm A}^{-1}$. Alternatives fail by $O(1)$: $q=x$ gives $0.91$, $q=2\pi x$ gives $0.39$. Spot check: $x=0.1$ gives analytic $0.8109$ against tape $0.81082$. |
| Cross-section units | barn $\to\mathring{\mathrm A}^2$ factor $10^{-8}$ |
| $F(0)=Z$, $S(0)=0$ | exact on the tape for all 100 elements. The loaders return $F(0)=Z$ and $S(0)=0$. |
| Large-$x$ limits | tape $S(10^9)/Z=1$ exactly and $F(10^9)/Z\le1.9\times10^{-20}$. Loaders: $\max_Z F(10^3)/Z=3.1\times10^{-6}$, $\min_Z S(10^3)/Z=0.9995$. |
| Designator convention | the tape's MF=1 directory confirms MT 534 = K, MT 541/542 = M4/M5. |

All filters pass.

## Comparison with the implementation

### Code reading

```text
designator=int(mt[k]) - _SUBSHELL_MT_OFFSET          # 533
cross_section_ang2=_readonly(y_k * _BARN_TO_ANG2)    # 1.0e-8
X_MAX_INV_ANG = 1.0e9
inside = (flat >= lo) & (flat <= hi); out = np.full(flat.shape, np.nan)
i = np.clip(np.searchsorted(knots_e, x, side="right") - 1, 0, knots_e.size - 2)
out[inside] = s0 + (x - e0) / (e1 - e0) * (s1 - s0)   # zero below knots_e[0]
```

These lines match the derivation term by term: $\mathrm{MT}-533$, $10^{-8}$, lin-lin, `side="right"` (right-continuous), zero below the first knot, and NaN outside the closed band. The clip to `size - 2` reaches no division by zero. No packaged section ends in a repeated knot, and none has a triple knot. 1609 sections begin with the $(\mathrm{EPE},0),(\mathrm{EPE},\sigma)$ onset pair. The generator refuses any `INT` other than 2, refuses any `EPE` that is not the first knot, and requires every MF=27 grid to start at $x=0$. It keeps zero values and repeated knots verbatim and thins only strictly increasing positive runs.

### Numerical comparison (independent reader against public loaders)

| Check | Result |
|---|---|
| Elements and rows | Z = 1–100. 1812 rows ordered by $(Z,\mathrm{MF},\mathrm{MT})$. |
| Designators | the loader's set equals $\{\mathrm{MT}-533\}$ from the tape for every Z, ascending from K |
| `EPE` = first knot | 0 exceptions among all MF=23 subshell sections. `EFL` = 0 everywhere. |
| EPDL against EADL MF=28 | subshell sets identical and `EPE` = `EBI` (to $10^{-6}$ relative) for all 100 elements, with 0 mismatches |
| Subshells, all knots and midpoints, all Z | max relative error $4.99941\times10^{-4}$ (La, designator 13, 616.6 eV) |
| $F$, all knots and midpoints, all Z | max relative error $4.99951\times10^{-4}$ (Na, $x=3.0625$) |
| $S$, all knots and midpoints, all Z | max relative error $5.000014\times10^{-4}$ (F, $x=0.0253$) |
| Right-continuity | the value at every repeated upstream knot equals the upper upstream value to $\le5\times10^{-4}$ |
| Zero below threshold | every subshell is exactly 0 at all test energies below its first knot |
| Out of domain | NaN for $E=0.5$, $1-10^{-9}$, $10^{11}(1+10^{-9})$ eV and NaN input; NaN for $x=-10^{-12}$, $10^9(1+10^{-9})$ and NaN input; finite at the closed endpoints |
| Knot count | $1\,229\,249\to1\,208\,833$ (1.66% removed, under 2% as stated) |
| Regeneration | `build_differential_table` on the pinned tape gives SHA `d36a50d1…bb166`, byte-identical to the packaged table and to `EPDL_DIFFERENTIAL_SHA256` |
| Anchors | `tests/materials/test_photon_differential.py`: 17 passed |

The $S$ maximum exceeds $5\times10^{-4}$ by $1.4\times10^{-9}$ in relative-error units. That excess is float32 rounding of the stored values ($\epsilon_{32}\approx6\times10^{-8}$ relative), applied after the float64 thinning test. It is not a thinning or interpolation error. Read the stored tolerance as $5\times10^{-4}$ plus float32 rounding.

### Upstream quirk

On the tape, Ni designators 8 and 9 (M4, M5; MT 541, 542) and Ag designator 14 (N5; MT 547) start at 10 eV with nonzero values (1.33, 2.67 and 5.58 Mb) and no zero onset knot. MT 522 instead ramps from below into 10 eV. Using only the upstream tape, $\sum_i\sigma_i$ against MT 522 differs by up to 92.3% (Ag, 9.94 eV) and 78.6% (Ni, 9.96 eV) inside the interval just below 10 eV. Everywhere else, for every element, the difference is at most $4.8\times10^{-5}$ (upstream rounding). The loaders reproduce the tape: 0 at 9.99 eV and the upstream value at 10 eV. The ledger's statement is correct.

## Findings

- No divergence in any factor, unit, sign, convention or policy.
- Docstring wording (not physics): the module docstring says the binding energies "can differ from EADL's by a few eV". On EPDL2025 against EADL2025, the subshell sets and `EPE` = `EBI` are identical for all 100 elements, which is what the ledger Checks state. The sentence is stale or overcautious. The task owner may reword it.
- Packaging: this page is not yet listed in a `docs/validation/evidence-index.md` toctree. The task owner should add `radiation-physics/epdl-photon-differential-data`.

## Verdict

`rederived`: the independent reader and the derivation match the implementation. Human sign-off is pending.
