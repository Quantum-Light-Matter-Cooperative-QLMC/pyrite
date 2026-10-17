# `cr-uniaxial-thermal-projection` — uniaxial displacement covariance projected on a planar harmonic

Verifier: fresh context (2026-10-16). Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/materials/channeling/thermal.py::{UniaxialThermalSite.b_factor_along,UniaxialThermalSite.covariance}`, `src/pyrite/materials/channeling/potential.py::_normal_b_factors`, `src/pyrite/data/channeling_thermal.toml::mos2`. Also covers the hexagonal, multi-element case of `cr-planar-continuum-potential`.

Independence note: the formulas in §1–§3 were derived from the ledger claim and the source description before the thermal/potential code bodies were read. The verifier had, however, already read the earlier anisotropic-TDS write-up for the transverse-covariance claim. The Schönfeld, Huang & Moss (1983) table could not be opened by the verifier, so §5 only checks the arithmetic of the stored values; a later source re-read (see the ledger notes) confirmed the transcription.

## 1. Debye–Waller factor of a planar harmonic

Take Gaussian site displacements $\mathbf u$ with zero mean and Cartesian covariance $U=\langle\mathbf u\mathbf u^{\mathsf T}\rangle$ (Å$^2$). For a Fourier component $\mathbf g$, $\langle e^{-i\mathbf g\cdot\mathbf u}\rangle=e^{-\mathbf g\cdot U\mathbf g/2}$. With $s=g/4\pi$ and $\hat{\mathbf n}=\mathbf g/g$,

$$
\tfrac12\mathbf g\cdot U\mathbf g=8\pi^2 s^2\,\hat{\mathbf n}\cdot U\hat{\mathbf n}\equiv B_ns^2,\qquad B_n=8\pi^2\,\hat{\mathbf n}\cdot U\hat{\mathbf n}.
$$

A planar harmonic $n$ has $\mathbf g_n=n\mathbf g_1$, so the same $\hat{\mathbf n}$ applies to all harmonics and the factor is $e^{-B_ns_n^2}$ with one $B_n$ per element.

## 2. Uniaxial covariance

$U=u_bI+(u_c-u_b)\hat{\mathbf c}\hat{\mathbf c}^{\mathsf T}$ equals $\mathrm{diag}(u_b,u_b,u_c)$ with $\hat{\mathbf c}=\hat{\mathbf e}_3$. Then $\hat{\mathbf n}\cdot U\hat{\mathbf n}=u_b+(u_c-u_b)\cos^2\theta=u_b\sin^2\theta+u_c\cos^2\theta$, with $\cos\theta=\hat{\mathbf n}\cdot\hat{\mathbf c}$:

$$
B_n=8\pi^2\bigl[u_b\sin^2\theta+u_c\cos^2\theta\bigr].
$$

Limits. $u_b=u_c=u$ gives $8\pi^2u$ for every plane. A basal plane has $\theta=0$ and $B_n=8\pi^2u_c$. A prismatic plane has $\theta=\pi/2$ and $B_n=8\pi^2u_b$. Units: Å$^2$, and the exponent $B_ns^2$ is dimensionless. $0\le\cos^2\theta\le1$, so $B_n$ lies between the two principal values.

Site-symmetry applicability. In $P6_3/mmc$, Mo $2c$ ($\bar6m2$) and S $4f$ ($3m$) both lie on a threefold axis along $c$. A threefold axis forces $U_{11}=U_{22}=2U_{12}$ in crystal axes and $U_{13}=U_{23}=0$, which is the Cartesian $\mathrm{diag}(u_b,u_b,u_c)$. The $6_3$ screw is a rotation about $c$, so it leaves this tensor unchanged. All sites of an element therefore share one Cartesian $U$, and the per-element sum in the planar structure factor is exact.

## 3. Numerical conversion of the stored rows

The stored rows are $\langle u^2\rangle$ in Å$^2$ and $B=8\pi^2u$:

| site | $u_b$ | $u_c$ | $8\pi^2u_b$ | $8\pi^2u_c$ |
| --- | --- | --- | --- | --- |
| Mo | 0.0048 | 0.0077 | 0.379 | 0.608 |
| S | 0.0038 | 0.0049 | 0.300 | 0.387 |

These reproduce the ledgered source column (Mo 0.38/0.61, S 0.30/0.39 Å$^2$). The Schönfeld table itself was not read by the verifier. Basal and prismatic limits and the interpolation $u_b\sin^2\theta+u_c\cos^2\theta$ hold at $\cos^2\theta=0.0472$ for $(101)$.

## 4. Comparison with the implementation and independent numerics

`b_factor_along`: $8\pi^2[u_b(1-c^2)+u_cc^2]$ with $c^2=\cos^2\theta$. This matches §2. `covariance`: $u_bI+(u_c-u_b)\hat{\mathbf c}\hat{\mathbf c}^{\mathsf T}$ with $\hat{\mathbf c}$ normalized. This matches. `_normal_b_factors`: $\cos^2\theta=(\mathbf g\cdot\mathbf c)^2/(g^2c^2)$ with $\mathbf c$ the third direct-lattice vector. This matches, provided that vector lies on the Cartesian $z$-axis (true for the hexagonal catalog cell). Temperature window enforced at 295 ± 10 K.

Independent rebuild of the MoS$_2$ planar potential. Own hexagonal lattice ($a=3.16$, $c=12.294$ Å), own reciprocal basis, own $P6_3/mmc$ basis ($\mathrm{Mo}\ (\tfrac13,\tfrac23,\tfrac14),(\tfrac23,\tfrac13,\tfrac34)$; $\mathrm S\ (\tfrac13,\tfrac23,z),(\tfrac23,\tfrac13,z-\tfrac12),(\tfrac23,\tfrac13,1-z),(\tfrac13,\tfrac23,\tfrac12-z)$ with $z=0.621$), `xraydb.f0` for the form factors, and

$$
U_n=-\frac{4\pi e^2}{Vg_n^2}\sum_j\bigl(Z_j-f_{0j}(g_n)\bigr)e^{i2\pi n\,\mathbf h\cdot\mathbf R_j}\,e^{-g_n^2\,\hat{\mathbf n}\cdot U_j\hat{\mathbf n}/2}.
$$

| plane | true period | harmonics compared | max $\lvert\Delta U_n\rvert$ | max $\lvert\Delta U(x)\rvert$ |
| --- | --- | --- | --- | --- |
| $(001)$ | $c/2$ (odd $00l$ extinct, $\lvert U_1\rvert\sim10^{-15}$ eV) | 11 | 2.4e-7 eV | 1.7e-6 eV |
| $(100)$ | $a\sqrt3/2$ | 11 | 1.8e-7 eV | 5.6e-7 eV |
| $(110)$ | $a/2$ | 11 | 2.0e-7 eV | 6.6e-7 eV |
| $(101)$ | $d_{101}$ | 11 | 3.8e-8 eV | 1.7e-7 eV |
| $(102)$ | $d_{102}$ | 11 | 7.4e-8 eV | 2.1e-7 eV |
| $(111)$ | $d_{111}/2$ (odd harmonics extinct) | 9 | 3.4e-8 eV | 7.3e-8 eV |

The code's plane-normal $B_n$ equals my $8\pi^2\hat{\mathbf n}\cdot U\hat{\mathbf n}$ for Mo on all six planes to full precision. The residual $\sim10^{-7}$ eV is the constant-level difference between the two form-factor evaluations. The extinctions follow the $P6_3/mmc$ rule: $h-k=3m$ with $l$ odd is forbidden. $(001)$ has minima at $z/c=0.121,\,0.25,\,0.379$ (S, Mo, S) and the barrier in the van der Waals gap. The well depth is 66.4 eV for $(001)$ and 28.8 eV for $(100)$. These are large because Mo ($Z=42$) and S dominate the sum, unlike Si.

## 5. Findings

- No sign, factor or convention divergence in the projection, the covariance or the planar sum.
- Data gap: the Schönfeld, Huang & Moss (1983) Table 3 values were not independently read, so the transcription and the $B$-versus-$U$ reading of the table are not independently confirmed. The stored values do equal the ledgered B column through $8\pi^2$.
- Catalog `mos2.toml` `B_ang2 = 0.47` uses the 3R deposit's $U_{33}$ (Mo 0.0075, S 0.0051), which gives $B_{33}=0.466$. The 2H rows give $(2\cdot0.0077+4\cdot0.0049)/6\cdot8\pi^2=0.458$. The 2 % spread is within the cross-polytype transfer already recorded in the materials audit.
- The ledger notes the specimen had $\sim5\%$ 3R and that the sulfur $z=0.621$ comes from the structure record, not the Schönfeld $0.1274$. With $z=0.621$ the S–Mo offset in the cell is $0.129\,c$. This difference moves the S wells by $\sim0.02$ Å and is not a projection issue.
- The period reduction keys on $\lvert U_n\rvert$ above a threshold (existing caveat in `cr-planar-continuum-potential`). For MoS$_2$ the skipped harmonics are $10^{-15}$ eV against $O(1)$ eV, so it is safe here.
