# Validation: `elsepa-muffin-tin-inputs`

## Scope and independence

Validation: `elsepa-muffin-tin-inputs`. Fresh-context validation on 2026-09-25. I derived the expressions below from the ledger row, the docstrings of `nearest_neighbour_distance_ang`, `elemental_solid`, `elemental_solid_for_composition`, `joined_arrays`, `resolve_layer_tables`, `ElsepaDeck.muffin_tin` and `ElsepaDeck.render`, and the vendored ELSEPA 2020 sources before I read any implementation body. Those sources are the `elscata.f` input reader and its `DNNEL`/`VMOLE`/`EEX1Z` tables, the muffin-tin block of `elsepa2020.f`, the upstream `readme.txt` and `Al.in`, and the upstream note *Atoms-in-solids.pdf* (Salvat, Jablonski & Powell, ELSEPA 2020, after Bote *et al.*, J. Electron Spectrosc. Relat. Phenom. **175** (2009) 41). I computed the nearest-neighbour distances and densities with a separate script that parses the catalog CIFs directly, without PyRITE crystal helpers. This validates the inputs PyRITE hands to ELSEPA and the way it joins the resulting tables. It does not validate ELSEPA's muffin-tin physics itself, and it does not authorize a human `signed-off` status.

## Source model

The ELSEPA 2020 note (Eqs. 5–7) models an atom in an elementary solid with the electron density

$$
\rho_e(r)=\rho_{\rm at}(r)+\rho_{\rm at}(2R_{\rm mt}-r)+\rho_0,\qquad
\int_0^{R_{\rm mt}}\rho_e(r)\,4\pi r^2\,dr=Z,
$$

and the constant potential $V(R_{\rm mt})=-\Delta E-i\Gamma$ for $r\ge R_{\rm mt}$. The source defines the radius as user input "or ... internally as half the inter-atomic distance". The code fills that default from `DNNEL(IZ)` (Kittel), stored in cm. The code imposes these constraints:

- `RMUF` is in **cm** (`elscata.f` header; `elsepa2020.f` converts it with `RMT=RMUF/A0B`). A value below $10^{-9}$ cm falls back to $\tfrac12$`DNNEL`. `DNNEL(15)` is $-1$, so phosphorus has no fallback and needs an explicit radius.
- If `NELEC` differs from `IZ`, the code resets `MUFFIN` to 0, so the muffin-tin model applies only to neutral atoms of one element.
- In muffin-tin mode the code sets `IHEF=0` whatever the deck says.
- `MABS 2` selects LDA-II. If `VABSA` is not positive, it takes the default $A_{\rm abs}=0.75$. If `VABSD` is omitted, it becomes the tabulated free-atom first-excitation energy `EEX1Z(IZ)` (C 1.26 eV, Si 0.78 eV, P 1.41 eV). The absorption potential is switched off only for `EV.GT.1.0D6`, so it is still active at exactly 1 MeV.
- `elscata` reads each line as `(A6,1X,A12)`, with the keyword in columns 1–6 and the value in columns 8–19. The input format has **no density keyword**. In muffin-tin mode the atomic density is always `VMOL=VMOLE(IZ)`, the PENELOPE natural-state table. It is replaced by $(2R_{\rm mt})^{-3}$ only if the sphere exceeds the Wigner–Seitz radius. In `elsepa2020.f` the density `VMOL1` appears only in the background absorption term $\sigma_{\rm abs,bk}=2\Gamma/(\hbar\mathcal N v)$ (note Eq. 14) and in that Wigner–Seitz guard. It does not enter the DCS, $\sigma_{\rm el}$, or $\sigma_{\rm tr,1}$.

## Independent derivation

**Radius.** Let a crystal have lattice rows $\mathbf a_k$ and basis fractional coordinates $\mathbf f_i$. The nearest-neighbour distance is

$$
d_{\rm nn}=\min_{i,j,\mathbf n\in\mathbb Z^3,\,(i,\mathbf n)\ne(j,\mathbf 0)}
\bigl\lvert(\mathbf f_j-\mathbf f_i+\mathbf n)\cdot A\bigr\rvert,
\qquad
R_{\rm mt}=\frac{d_{\rm nn}}{2}\ [\text{Å}]
=\frac{d_{\rm nn}}{2}\times10^{-8}\ [\text{cm}].
$$

**Truncation bound.** Write a candidate separation as $\mathbf v=\sum_k c_k\mathbf a_k$ with $c_k=n_k+\delta_k$ and $\lvert\delta_k\rvert<1$. Let $\mathbf b_k$ be the dual vectors, $\mathbf a_i\cdot\mathbf b_k=\delta_{ik}$, and $d_k=1/\lvert\mathbf b_k\rvert$ the spacing of the lattice planes. Then $\lvert c_k\rvert=\lvert\mathbf v\cdot\mathbf b_k\rvert\le\lvert\mathbf v\rvert/d_k$. Any candidate outside the block $n_k\in\{-2,\dots,2\}$ has some $\lvert n_k\rvert\ge3$, so $\lvert c_k\rvert>2$ and $\lvert\mathbf v\rvert>2d_k$. The $5\times5\times5$ search is therefore exact whenever

$$
d_{\rm nn}^{(\pm2)}\le 2\min_k d_k .
$$

**Density.** $n$ is in atoms/Å$^3$, $1\ \text{Å}^{-3}=10^{24}\ \text{cm}^{-3}$, and $A$ is in g/mol:

$$
\rho=\frac{10^{24}\,n\,A}{N_A}\quad[\mathrm{g\,cm^{-3}}].
$$

**Join.** Let $E_\star=\max\{E_i^{\rm mt}\le 1\ \text{MeV}\}$. The joined table holds every muffin-tin row with $E\le E_\star$ and every free-atom row with $E>E_\star$, copied verbatim on one shared $\mu$ grid. The node $E_\star$ belongs to the muffin-tin side. The joined energy axis is strictly increasing, and no row mixes the two models.

**Matching.** A layer with composition $\{(X,n)\}$ uses crystal $c$ if $X_c=X$ and $\lvert n_c-n\rvert\le10^{-6}\,n$. Compounds never match.

## Numeric evaluation (independent script)

| crystal | $n$ [Å$^{-3}$] | $d_{\rm nn}$ [Å] ($\pm2$ = $\pm6$ block) | $2\min_k d_k$ [Å] | $\rho$ [g cm$^{-3}$] | ELSEPA `DNNEL` [Å] |
| --- | --- | --- | --- | --- | --- |
| `silicon` | 0.049943 | 2.35165 | 10.86 | 2.3292 | 2.350 |
| `diamond` | 0.176301 | 1.54447 | 7.13 | 3.5163 | 1.540 |
| `hopg` | 0.113637 | 1.42086 | 4.26 | 2.2665 | — (C entry is diamond) |
| `black_phosphorus` | 0.052651 | 2.22360 | 6.63 | 2.7080 | $-1$ (none) |

These four are the only one-element catalog crystals; the only catalog medium, `sio2`, is a compound. For Si and C, $d_{\rm nn}=a\sqrt3/4$ reproduces `DNNEL` to $+0.07\,\%$ and $+0.29\,\%$, i.e. within Kittel's rounding. The truncation bound holds with a margin of at least $3\times$ for every crystal.

## Source-to-code comparison

- `nearest_neighbour_distance_ang` searches `product(range(-2, 3), repeat=3)` and masks index `shifts.shape[0] // 2`, which is the zero shift $(0,0,0)$, only for the self-pair. Its output agrees with my $\pm6$ search to all printed digits for all four crystals.
- `elemental_solid` sets `radius_cm = 0.5 * d_nn * 1.0e-8` and `density = n * A / (Avogadro * 1e-24)`. Both are identical to the expressions above. The values match the table to $\le 10^{-5}$ relative; the only differences come from the tabulated $A$.
- `ElsepaDeck.render` writes `IZ, MNUCL 3, NELEC=IZ, MELEC 4, MUFFIN 1, RMUF, IELEC -1, MEXCH 1, MCPOL 0, MABS 2, VABSA 7.50000E-01, IHEF 0, EV …`. The keywords are left-justified to 6 characters followed by one blank, and the longest rendered Si line is 18 characters (under the 19-column limit). `RMUF 1.17582E-08` is in cm. Omitting `VABSD` gives `EEX1Z(IZ)`, as the ledger notes state. The `IHEF` keyword matches the source's 4-character literal under Fortran blank padding. It is forced to 0 anyway, which is what `__post_init__` requires. `NELEC = IZ` satisfies the muffin-tin precondition. These fields match `Al.in` except that `Al.in` uses the Al-specific `VABSA 0.835` and `VABSD 0`.
- `joined_arrays` uses `low = E_mt <= 1e6`, `top = max(E_mt[low])`, and `high = E_free > top`, then concatenates. This is identical to the join rule above. It refuses unequal `mu` grids.
- `elemental_solid_for_composition` uses `np.isclose(n_c, n, rtol=1e-6, atol=0)`. The two carbon crystals differ in $n$ by 55 %, so no two catalog crystals can collide. The Si substrate of `mos2-on-sio2-si` takes its composition from `CATALOG.crystal("silicon").composition` and matches `silicon` exactly. A one-element layer whose density differs by more than $10^{-6}$ (for example a user-defined Si at 2.33 g cm$^{-3}$, which is $4\times10^{-4}$ off) silently falls back to free atoms. This is the documented behaviour.

**Stored production tables** (`resolve_layer_tables` on the installed tables, 41 muffin-tin nodes from 100 eV to 1 MeV, 606 $\mu$ nodes):

| crystal | $\sigma_{\rm el}^{\rm mt}/\sigma_{\rm el}^{\rm free}$ at 1 MeV | $\sigma_{\rm tr,1}^{\rm mt}/\sigma_{\rm tr,1}^{\rm free}$ at 1 MeV | $\sigma_{\rm el}$ jump from 1 MeV to 1.25 MeV | max $\lvert 4\pi\int{\rm DCS}\,d\mu/\sigma_{\rm el}-1\rvert$ (mt rows) |
| --- | --- | --- | --- | --- |
| `silicon` | 0.685 | 0.991 | $\times1.41$ | $3.9\times10^{-3}$ (1 MeV) |
| `diamond` | 0.522 | 0.978 | $\times1.85$ | $3.0\times10^{-3}$ |
| `hopg` | 0.463 | 0.972 | $\times2.09$ | $3.0\times10^{-3}$ |
| `black_phosphorus` | 0.709 | 0.991 | $\times1.36$ | $3.8\times10^{-3}$ |

Across that interval the free-atom table alone changes by $\times0.966$. The rows are verbatim copies: `np.array_equal` holds on both sides of the join. The joined energy axis is strictly increasing and every DCS is positive. The step is therefore a positive, monotone ramp under log-log interpolation over $[1, 1.25]$ MeV. The sampler interpolates quantiles in $\ln E$ between two valid normalized CDFs, so it needs no special handling. $\sigma_{\rm tr,1}$ is continuous to 1–3 %, which means the physical cost is extra near-forward events above 1 MeV, not a change in angular diffusion. The mixed-model ramp is a documented modelling choice, not an implementation error.

## Divergences (documentation and scope; no formula divergence)

1. **Density does not reach ELSEPA.** The ledger claim reads "the deck … takes … its density as $\rho=10^{24}nA/N_A$". `ElsepaDeck.render` writes no density, and `elscata` has no keyword that could carry one. `density_g_cm3` enters only `material_identity` in `muffin_tin_request`, i.e. the table key. ELSEPA instead uses `VMOLE(IZ)`. For carbon that is $1.0028\times10^{23}$ cm$^{-3}$, about 2.0 g cm$^{-3}$, where diamond has $1.763\times10^{23}$ and hopg $1.136\times10^{23}$. That density feeds only the stored `absorption_cm2` background term and the Wigner–Seitz guard. The guard does not trigger for any of the four crystals: $R_{\rm mt}<R_{\rm WS}(\texttt{VMOLE})$ in every case. None of the sampler fields are affected. The formula PyRITE computes is correct. The claim should say that the density is a table-identity input, not a deck input, and that the stored absorption cross section uses ELSEPA's tabulated density.
2. **The truncation argument in the `nearest_neighbour_distance_ang` docstring is false as stated.** The docstring claims the $5\times5\times5$ block suffices when "shortest lattice vector is not shorter than half its longest — true of every catalog crystal". That condition fails for `hopg` ($2.461<6.711/2$) and `black_phosphorus` ($3.314<10.478/2$). The result is still exact, because the plane-spacing bound $d_{\rm nn}\le2\min_k d_k$ holds for every crystal with at least $3\times$ margin.
3. **The step size is understated for carbon.** The ledger note ("about 30 % for Si") and the module docstring ("about 30 %"; "$\sigma_{\rm tr,1}$ agrees to 1–2 %") are accurate for Si. For carbon, however, the 1 MeV $\sigma_{\rm el}$ ratio is 0.52 for diamond and 0.46 for hopg, and $\sigma_{\rm tr,1}$ differs by 2.2 % and 2.8 %.
4. **The closure bound is scoped to two energies.** The ledger check "closes … within `2e-3`" is anchored at 1 keV and 100 keV only. On the production grid, trapezoidal closure on the native $\mu$ grid reaches $3.9\times10^{-3}$ for Si at 1 MeV, and exceeds $2\times10^{-3}$ at 8–10 of the 41 energies for each crystal. This is quadrature error on a forward-peaked DCS, not a physics error.
5. **The low-energy floor rationale uses the wrong density model name.** It cites a "DHFS/muffin-tin treatment", but the deck uses `MELEC 4`, which is Dirac–Fock, not DHFS.
6. **Model caveat (not a code error).** The touching-sphere packing fraction $\tfrac43\pi R_{\rm mt}^3 n$ is 0.34 for Si and diamond, 0.30 for black P, and only 0.17 for hopg. For layered graphite, most of each atom's Wigner–Seitz volume therefore sits in the constant interstitial potential, so the ELSEPA muffin-tin construction is cruder there than for the diamond-cubic solids it was tabulated for. No test anchors `black_phosphorus` (radius or matching).

## Evidence and verdict

`tests/xsgen/test_elsepa_catalog.py` passes (19 tests). Independent CIF parsing, a $\pm6$-cell brute-force search, the plane-spacing bound, the ELSEPA `DNNEL` table, and the installed production tables all support the ledgered equations.

- **Filters:** units pass (`RMUF` in cm; $\rho$ in g cm$^{-3}$); limits pass (diamond-cubic $a\sqrt3/4$ reproduces `DNNEL` for Si and C); conventions pass (fixed-column keywords, crossover node on the muffin-tin side, no blending).
- **Re-derivation:** matches for the radius, the density formula, the deck fields, the join, and the matching. The divergences are wording and scope only; see items 1–5 above.
- **Verdict:** `rederived`.
- **Human sign-off:** pending.

## Adjudication

Accepted 2026-09-25. The ledger claim now says the crystal density enters only the table identity; the assumptions name the Dirac–Fock (`MELEC 4`) model; the closure check is scoped to its anchor energies; the notes give the per-crystal 1 MeV steps and the HOPG packing caveat; and the `nearest_neighbour_distance_ang` and module docstrings carry the plane-spacing criterion and the corrected wording. The status moved to `rederived`.
