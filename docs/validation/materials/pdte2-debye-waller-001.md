# `pdte2-debye-waller-001`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 0.61` is a common basal projection for 1T-PdTe2 `(001)`, not an isotropic displacement parameter.
- Code: `data/catalog/crystals/pdte2.toml::B_ang2`.
- Source: Pell, Mironov & Ibers, *Acta Cryst. C* **52**, 1331--1332 (1996), doi:10.1107/S0108270195016246; COD 2004955.
- Intended quantity and units: one amplitude Debye--Waller coefficient in square angstroms for the reciprocal `c*` direction.

## Independent derivation

The source CIF gives unit-occupancy sites with

$$
U_{33,\mathrm{Pd}}=0.0074(4)\ {\rm \AA^2},\qquad
U_{33,\mathrm{Te}}=0.0079(3)\ {\rm \AA^2}.
$$

For one Pd and two Te atoms per formula unit, the declared common-site approximation is the multiplicity-weighted projection

$$
\bar U_{33}=\frac{U_{33,\mathrm{Pd}}+2U_{33,\mathrm{Te}}}{3}
           =0.0077333\ {\rm \AA^2}.
$$

Using the crystallographic convention `B = 8 pi^2 U`,

$$
B_{33}=8\pi^2\bar U_{33}=0.61062\ {\rm \AA^2},
$$

which rounds to `0.61 Ang^2`. Independent propagation of the quoted uncertainties gives `sigma(U33) = 0.000240 Ang^2` and `sigma(B33) = 0.0190 Ang^2`; the stored precision is appropriate. The site values differ by `0.0005(5) Ang^2`, so a common basal projection is consistent within one combined standard uncertainty.

For a basal `(00l)` reflection with `g = 2 pi l / c`, the amplitude factor is

$$
\exp\left(-\frac{B_{33}g^2}{16\pi^2}\right)
=\exp\left(-\frac{B_{33}l^2}{4c^2}\right)
=\exp\left(-\frac{2\pi^2U_{33}l^2}{c^2}\right).
$$

This equivalence fixes the direction and rules out interpreting `0.61` as `U33` or applying a second Debye--Waller factor.

## Cheap filters

- Units: `U33` and `B33` are both in square angstroms; `8 pi^2` is dimensionless and the exponent is dimensionless.
- Limits: `U33 -> 0` gives unit amplitude; increasing `U33` or `l` suppresses the reflection; `(001)` probes only the `c*` projection.
- Signs/conventions: passive thermal motion requires a negative exponent. `B = 8 pi^2 U`, not its inverse. This scalar is direction-specific and must not be reused as `Biso` for non-basal reflections.

## Implementation comparison

Inspection after fixing the derivation found:

- the source CIF independently downloaded from COD 2004955 contains `Pd U33=0.0074(4)` and `Te U33=0.0079(3) Ang^2`;
- `crystals/pdte2.toml` stores `B_ang2 = 0.61` and pins only basal `(001)`;
- `debye_waller` returns `exp[-B (g / 4 pi)^2]`, exactly the independently derived amplitude factor;
- `structure_factor` applies that factor once to each atom.

A separate numeric calculation used only the source cell/basis and `xraydb.f0` (no PyRITE crystallography helper). At 10 keV, changing the common scalar from `0.60` to `0.61 Ang^2` changes `|F001|^2` by `-0.0191239%`. Replacing the site-specific source tensors by the common `0.61 Ang^2` projection changes it by `-0.0714434%`. These reproduce the ledger values (`-0.019%` and `-0.071%`).

Non-physics documentation finding: `data/cifs/pdte2.cif` still says the catalog uses the old `0.6 Ang^2` placeholder. Production TOML and its golden snapshot use `0.61 Ang^2`.

Focused verification passed `test_packaged_catalog_matches_independent_serialized_golden`, but the broader Pd/Pt surface-contract parameterization still hard-codes `0.6 Ang^2` and fails for PdTe2 (`0.61` obtained). This stale assertion blocks advancing the claim to `anchored`; it does not change the re-derivation.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, sign, unit, or convention.
- **Verdict:** `rederived`.
- **Suggested ledger change:** `unverified -> rederived`; human applies it. Do not mark `anchored` until stale test assertion is reconciled, and never mark `signed-off` without human certification.

## Independent fresh-context re-derivation (2026-11-15)

Fresh-context verifier pass. The derivation and numbers below were produced from the COD 2004955 CIF (downloaded independently) and the ledger row only, before reading the TOML, the repository CIF, or the sections above.

### Source extraction

COD 2004955 (Pell, Mironov & Ibers 1996): $P\bar 3m1$, $Z=1$, $T=113(2)$ K, $a=4.024(3)$ Å, $c=5.113(4)$ Å, $\gamma=120^\circ$. Pd1 on 1a $(0,0,0)$; Te1 on 2d $(\tfrac13,\tfrac23,z)$ with $z=0.26628(8)$; both occupancies 1. Anisotropic parameters (Å$^2$):

| site | $U_{11}=U_{22}$ | $U_{33}$ | $U_{12}$ | $U_{13}=U_{23}$ | $U_{\rm eq}$ (CIF) |
|---|---|---|---|---|---|
| Pd1 | 0.0057(3) | 0.0074(4) | 0.00284(15) | 0 | 0.0063(3) |
| Te1 | 0.0062(3) | 0.0079(3) | 0.00309(14) | 0 | 0.0068(3) |

Transcription filters: $U_{12}=U_{11}/2$ and $U_{13}=U_{23}=0$, as the $\bar 3m$ and $3m$ site symmetries require. The hexagonal equivalent $U_{\rm eq}=\tfrac13\left[\tfrac43(U_{11}+U_{22}-U_{12})+U_{33}\right]$ gives 0.00627 (Pd) and 0.00677 Å$^2$ (Te), which reproduce the CIF's 0.0063 and 0.0068.

### Debye–Waller factor for $(00l)$

The CIF (Trueblood et al. 1996) convention is

$$
T_j(\mathbf h)=\exp\!\left(-2\pi^2\sum_{ij}U^{ij}_j\,h_ih_j\,a^*_ia^*_j\right).
$$

For $\mathbf h=(0,0,l)$ only the $i=j=3$ term survives because $h_1=h_2=0$. This holds whatever $U_{13}$ and $U_{23}$ are, and here they are zero anyway. In the hexagonal cell $\mathbf c\perp\mathbf a,\mathbf b$, so $c^*=\lvert\mathbf a\times\mathbf b\rvert/V=1/c$ and

$$
T_j(00l)=\exp\!\left(-\frac{2\pi^2U_{33,j}\,l^2}{c^2}\right).
$$

With $B=8\pi^2U$ and $\sin\theta/\lambda=\lvert\mathbf g\rvert/2=l/(2c)$, this is $\exp[-B_{33}(\sin\theta/\lambda)^2]$ exactly. The numerical check agrees: for $(001)$, $s^2=(\sin\theta/\lambda)^2=0.0095629$ Å$^{-2}$, and $2\pi^2U_{33}g^2=B_{33}s^2=0.0055874$ (Pd) and $0.0059649$ (Te).

### What a single common scalar must equal

Take $f_j=f_j^0(s)+f_j'+if_j''$. Then

$$
F_{00l}=f_{\rm Pd}T_{\rm Pd}+2f_{\rm Te}\cos(2\pi lz)\,T_{\rm Te}.
$$

A common $U$ reproduces $F$ exactly only if the site factors are combined with **amplitude weights** $w_{\rm Pd}=f_{\rm Pd}$ and $w_{\rm Te}=2f_{\rm Te}\cos(2\pi lz)$. To first order in $U$,

$$
U_{\rm eff}\simeq\frac{w_{\rm Pd}U_{33,\rm Pd}+w_{\rm Te}U_{33,\rm Te}}{w_{\rm Pd}+w_{\rm Te}}.
$$

For $(001)$, $\cos(2\pi z)=-0.10211$, so the Te pair enters with a *negative* weight ($w_{\rm Te}\approx-9.87$ against $w_{\rm Pd}\approx43.2$ electrons). Then $U_{\rm eff}$ falls *below* both site values. Neither multiplicity weighting, $\bar U=(U_{\rm Pd}+2U_{\rm Te})/3$, nor scattering-power weighting is the exact $(001)$ scalar. Both are documented approximations. If $U_{33,\rm Pd}=U_{33,\rm Te}$, every weighting coincides, which gives the limiting case.

Arithmetic of the ledgered choice: $\bar U_{33}=(0.0074+2\times0.0079)/3=0.0077333$ Å$^2$, so $B_{33}=8\pi^2\bar U_{33}=0.61060$ Å$^2$. Propagating the uncorrelated uncertainties gives $\sigma(B_{33})=0.019$ Å$^2$. These agree with the ledger.

### Numeric check at 10 keV (independent form factors)

Form factors come from xraylib 4.3.0 (`FF_Rayl`, `Fi`, `Fii`; oracle group), not from PyRITE helpers. At $s=0.097790$ Å$^{-1}$: Pd $f^0=43.296$, $f'=-0.075$, $\lvert f''\rvert=2.742$; Te $f^0=48.237$, $f'=+0.076$, $\lvert f''\rvert=4.468$. The same $f''$ sign convention is used for both sites, so $\lvert F\rvert^2$ does not depend on it. The basis is Pd $(0,0,0)$ and Te at $\pm z$.

| displacement model | $B$ (Å$^2$) | $\lvert F_{001}\rvert^2$ | vs site-specific | vs $B=0.60$ |
|---|---|---|---|---|
| site-specific $U_{33}$ | — | 1103.765 | 0 | +0.0525% |
| common, placeholder | 0.60 | 1103.186 | −0.0524% | 0 |
| common, catalog | 0.61 | 1102.975 | **−0.0715%** | **−0.0191%** |
| common, unrounded multiplicity | 0.61060 | 1102.963 | −0.0727% | −0.0203% |
| common, scattering-power weighted | 0.61156 | — | −0.0745% | — |
| common, exact for $(001)$ | **0.5726** | 1103.765 | 0 | — |

The ledger's −0.019% and −0.071% are reproduced for the rounded catalog value $B=0.61$. The scalar that reproduces the site-specific $\lvert F_{001}\rvert^2$ exactly is $B_{\rm eff}=8\pi^2(0.0072518)=0.5726$ Å$^2$; the linearised amplitude-weight estimate gives $U_{\rm eff}=0.0072521$. That differs from 0.61 Å$^2$ by 0.038 Å$^2$, about $2\sigma(B_{33})$, but moves $\lvert F_{001}\rvert^2$ by only 0.07%.

### Comparison with the implementation

- `crystals/pdte2.toml::B_ang2 = 0.61` equals the multiplicity-weighted $B_{33}$ rounded to two decimals. Its comment cites Pd $U_{33}=0.0074$ and Te $U_{33}=0.0079$, matching the source.
- The repository `data/cifs/pdte2.cif` cell ($a=4.024$, $c=5.113$, $\gamma=120^\circ$), unit occupancies, and sites Pd $(0,0,0)$, Te $(\tfrac13,\tfrac23,0.26628)$, and Te $(\tfrac23,\tfrac13,0.73372)\equiv-z$ match the source. It carries no $U_{ij}$, so the TOML scalar is the only displacement input.
- Stale documentation (non-physics): the repository CIF header still says the catalog uses `B_ang2=0.6` as "a shared modeling placeholder, not the displacement parameter refined by the source". That contradicts the TOML. The stale test assertion noted above has since been reconciled: `tests/materials/test_pd_pt_crystals.py` now expects 0.61 for `pdte2`, and `-k pdte2` passes (2 tests).
- Marker gap (finding only): the TOML's `validation_id` is `pdte2-p3m1-structure`, and nothing in the TOML or CIF names `pdte2-debye-waller-001`. The provenance comment is the only link.

### Verdict of this pass

- Filters: units pass; limits pass ($U\to0\Rightarrow T=1$; equal site $U_{33}$ makes every weighting exact); signs/conventions pass (negative exponent, $B=8\pi^2U$, only $U_{33}c^{*2}$ enters $(00l)$).
- Re-derivation: **matches** the ledgered arithmetic and both sensitivity numbers. Modelling caveat: multiplicity weighting is not the amplitude-exact $(001)$ scalar, which is $0.573$ Å$^2$ because the Te term interferes destructively. The resulting −0.07% $\lvert F_{001}\rvert^2$ bias is the ledger's own disclosed number and is far below any plausible PXR/CBS tolerance.
- Verdict: `rederived`. Human sign-off remains pending.
