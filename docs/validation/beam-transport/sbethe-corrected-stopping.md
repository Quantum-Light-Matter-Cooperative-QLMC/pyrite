# SBETHE corrected collision stopping in transport

Validation: `sbethe-corrected-stopping`.

## Independent derivation from Salvat's source

Salvat's `sbethe.f` header (lines 5–37) defines the electronic stopping quantity and limits the corrected Bethe formula to energies above `ECUT`. For an electron projectile, `BETHE` (lines 2319–2397) sets the heavy-projectile Lindhard–Sørensen and Barkas terms to zero. Its cross section is

$$
\sigma_{\mathrm{stop}}(E)=2\,\mathrm{CONS}(E)\,Z_{\mathrm{mol}}
\left[L_0(E)+\frac{f(\gamma)}{2}
-C_{\mathrm{shell}}(E)-\frac{\delta_F(E)}{2}\right].
$$

Here `CONS` is the kinematic cross-section prefactor, `ZT` is the source's electron count $Z_{\mathrm{mol}}$ per molecule, and `CSHC` is the interpolated DHFS shell correction. `DENSIT` (lines 2135–2210) constructs the Fano density correction from the material oscillator-strength distribution; `BETHE` interpolates it as `CDEN` $=\delta_F/2$. Each bracketed term is dimensionless; the cross section has units eV cm$^2$ per molecule.

The shell-corrected `stp.dat` column (lines 440–478) is

$$
S_{\mathrm{eV/\mathring A}}(E)
=\sigma_{\mathrm{stop}}(E)n_{\mathrm{mol}}
\left(10^{-8}\,\frac{\mathrm{cm}}{\mathring A}\right),
\qquad n_{\mathrm{mol}}=\mathrm{VMOL}
\quad[\mathrm{cm}^{-3}].
$$

Thus $[\sigma n_{\mathrm{mol}}]=\mathrm{eV/cm}$ and the factor $10^{-8}$ converts distance from cm to Å. The separate no-shell column adds back $2\,\mathrm{CONS}\,Z_{\mathrm{mol}}\,n_{\mathrm{mol}}C_{\mathrm{shell}}$ before that distance conversion. For the transport convention,

$$
S_{\mathrm{keV/\mathring A}}=10^{-3}S_{\mathrm{eV/\mathring A}},
\qquad \frac{dE}{ds}=-S_{\mathrm{keV/\mathring A}}.
$$

Given strictly positive, increasing energies $E_i$ and stopping magnitudes $S_i$, the specified interpolation for $E_i\leq E\leq E_{i+1}$ is

$$
f=\frac{\ln E-\ln E_i}{\ln E_{i+1}-\ln E_i},\qquad
S(E)=\exp\!\left[(1-f)\ln S_i+f\ln S_{i+1}\right]
=S_i^{1-f}S_{i+1}^{f}.
$$

At $E_i$ and $E_{i+1}$ this returns the native magnitudes. At the geometric energy midpoint it returns $\sqrt{S_iS_{i+1}}$; if the two magnitudes agree it is constant across the interval. It stays positive and continuous but need not have a continuous derivative at a node. No extrapolation follows from this definition. Transport energies and cutoffs must remain in the table domain; an endpoint clamp is justified only for roundoff after that domain check. The signed rate is non-positive.

Below the material-dependent `ECUT` (about 1 keV for electrons), Salvat uses the empirical extrapolation in lines 525–584. Therefore a 1 keV table node need not be an evaluation of the corrected Bethe equation. The validation target is the material-level output table over its declared domain.

## Implementation comparison

The parser's `parse_stopping` takes column two of `stp.dat` as the shell-corrected eV/Å magnitude and retains the no-shell column separately. `prepare_sbethe_stopping_table` converts both energy and stopping magnitude from eV to keV with $10^{-3}$, checks positive finite ordered nodes, and stores their logarithms. Both `_dEds_sbethe_scalar` and CUDA `_dEds_sbethe` evaluate the derived linear interpolation in log space, then negate the exponential. Their different binary-search implementations select the same adjacent nodes. The host-facing `sbethe_stopping_keV_per_ang` rejects energies outside the table, allowing a one-ULP outward tolerance at the endpoints.

`simulate_trajectories` checks every layer's minimum cutoff and maximum initial electron energy against its table before launching CPU or CUDA work. The CPU lockstep/per-electron core, CPU LUT, and exact CUDA core all select the SBETHE rate when a table is supplied. Midpoint and cutoff evaluations also select it. The LUT samples the same table but its interpolation introduces a separate bounded numerical approximation. Urban sampling scales its elemental reference mean by the ratio of the SBETHE and reference rates.

Production `_case_stopping_table_records` resolves each transport layer's table from that layer's composition. `STOPPING_MODEL` is `sbethe-corrected-v2` (v2: compound shell corrections weighted by electron fraction, a PyRITE modification of `sbethe.f`); the table manifest digest is included in dataset and public result identity, so a changed table produces a changed key. Tests pin the catalog tables' 1 keV and 1 GeV endpoints, positivity, shell-column difference, node and geometric-midpoint interpolation, domain rejection, cutoff distance, LUT rate, and CPU core paths. The CUDA parity test is gated on hardware; inspecting the source proves formula and route selection, not execution on a GPU in this verification.

As an independent numerical check, let $E_0=1\,\mathrm{keV}$, $S_0=8\,\mathrm{eV/\mathring A}$, $E_1=4\,\mathrm{keV}$, and $S_1=4\,\mathrm{eV/\mathring A}$. At $E=2\,\mathrm{keV}$ the prediction is

$$
\frac{dE}{ds}=-10^{-3}\sqrt{8\cdot4}
=-0.00565685424949\,\mathrm{keV/\mathring A}.
$$

The scalar interpolation anchor uses that same midpoint. No divergent factor, sign, exponent, or column was found. The independent derivation and implementation comparison support `rederived`. On an NVIDIA GeForce RTX 5080 with driver 610.47, the SBETHE/Urban first-row CPU comparison and the exact and LUT CUDA cutoff anchors passed. The latter use a constant $0.1\,\mathrm{keV/\mathring A}$ stopping table and independently predict a $50\,\mathring A$ range for a $5\,\mathrm{keV}$ loss. Human sign-off remains separate.

## Compound shell correction (`sbethe-corrected-v2`)

This section re-derives the PyRITE modification of `sbethe.f` that weights a compound's shell correction by electron fraction. The verifier was a fresh context. The task handed over the patched source line before this derivation, so independence here comes from the additivity derivation and the numerical reconstruction below, not from blind code reading.

### Derivation from Bragg additivity

For an element $j$ with atomic number $Z_j$ and atom density $N_j$, the shell-corrected Bethe stopping is

$$
S_j = K(E)\,N_jZ_j\left[L_{0,j}(E)+\frac{f(\gamma)}{2}-\frac{C_j(E)}{Z_j}-\frac{\delta_j}{2}\right],
$$

with $C_j$ the total atomic shell correction and $C_j/Z_j$ its per-electron value. The kinematic factor $K=2\,\mathrm{CONS}$ is the same for every element. A molecule holds $n_j$ atoms of element $j$, so $N_j=n_jN_{\rm mol}$ and $Z_T=\sum_j n_jZ_j$. Bragg additivity of the shell term gives

$$
\sum_j K N_{\rm mol}n_jZ_j\,\frac{C_j}{Z_j}
=K N_{\rm mol}Z_T\left[\sum_j \frac{n_jZ_j}{Z_T}\,\frac{C_j}{Z_j}\right].
$$

SBETHE factors $K N_{\rm mol} Z_T$ out of the whole bracket: `SFACT=2*CONS*ZT` in `BETHE`, then multiplication by `VMOL` $=N_{\rm mol}$. The quantity subtracted inside the bracket must therefore be the compound's per-electron shell correction:

$$
\boxed{\;\frac{C}{Z}\bigg|_{\rm compound}=\sum_j w_j\,\frac{C_j}{Z_j},\qquad w_j=\frac{n_jZ_j}{Z_T}\;}
$$

The weights obey $\sum_j w_j = 1$. Substituting back, the shell term is $K\sum_j N_jC_j$, which is the additive sum of atomic corrections.

### Normalization of the SBETHE quantities

- **Table quantity.** Each `sdbase/eshcor-ZZ.tab` table has the column header `C'(gamma)/Z`, so `SHELL` as read is per electron. SBETHE adds the relativistic term $\frac{Z_j-S_0}{2Z_j}R_c(\gamma)$, which also carries a factor $1/Z_j$, so it stays per electron. `stplogb.dat` labels the `CSHC` column `(C/Z)`. The heavy-projectile terms `CLS` and `CBAR` are divided by `SFACT`, which is the same per-electron convention.
- **Indices.** `ZT` is always $\sum_j$ `STF(j)*IZ(j)`. This holds for the element prompt (`STF(1)=1`, `ZT=IZZ`), for the stoichiometric and weight-fraction prompts (weight fractions are normalized to the largest index), for the `pdcompos.pen` path, and for a reread material file, which stores the `ZT` that `MATER` computed. `ODTRAN` only copies `ZT`.
- **Scale invariance.** Under $n_j\to\lambda n_j$ the code scales $Z_T\to\lambda Z_T$ and $N_{\rm mol}\to N_{\rm mol}/\lambda$. Then $w_j$, $Z_TN_{\rm mol}$, $L_0$ (through $I$) and the density correction are all unchanged.

The patched lines are

```text
SHC(NSHC)=STF(JZ)*DBLE(IZE)*SHELL/ZT
X2(N1)=STF(JZ)*DBLE(IZE)*SHELL/ZT
```

Here `MERGE2` sums the per-element tables on the union grid, and `FINTRP` interpolates the sum as `CSHC`. The result is exactly $\sum_j w_j\,(C_j/Z_j)$. Upstream used `STF(JZ)*SHELL`, giving $\sum_j n_j\,(C_j/Z_j)$. That expression is not normalized. For an element it equals the derived value only when $n=1$, and for a compound only by accident. It scales linearly with $\lambda$.

### Limits

- **Single element.** $w=n Z/(nZ)=1$ for any index $n$, so `CSHC` is the element's own `SHELL`, with at most one rounding in $(nZ)\,\mathrm{SHELL}/(nZ)$. Upstream agrees only when $n=1$, which is what the element prompt sets.
- **Index rescaling.** The derived expression is invariant under rescaling. Upstream's `CSHC` grows by $\lambda$.

### Numerical evidence

As an independent reconstruction, a scratch script read the raw `eshcor` tables and added the relativistic term using each table's own $S_0$. It then formed $\sum_j w_j\,C_j/Z_j$ by log-linear interpolation, which is `MERGE2`/`FINTRP` mode 2. This was compared with the `CSHC` column that the compiled vendored program writes to `stplogb.dat`:

| material (indices) | $E$ (eV) | `CSHC` patched | independent $\sum_j w_jC_j/Z_j$ | `CSHC` upstream |
| --- | --- | --- | --- | --- |
| water (2:1) | $10^4$ | 3.06054e-2 | 3.06054e-2 | 5.10720e-2 |
| water (20:10) | $10^4$ | 3.06054e-2 | 3.06054e-2 | 5.10720e-1 |
| water (2:1) | $10^6$ | 3.35822e-3 | 3.35821e-3 | 1.91035e-2 |
| ZrTe$_5$ (1:5) | $10^4$ | 6.82907e-2 | 6.82880e-2 | 4.09895e-1 |
| ZrTe$_5$ (1/6:5/6) | $10^4$ | 6.82907e-2 | 6.82880e-2 | 6.83159e-2 |
| Te (1) | $10^4$ | 6.81899e-2 | 6.81868e-2 | 6.81899e-2 |

Water reconstructs to 6 figures. The ZrTe$_5$ and Te residuals of about $5\times10^{-5}$ relative come from tabulation-grid interpolation that `MERGE2` resamples. Patched water and ZrTe$_5$ are identical under index rescaling, and pure Te is bit-identical between upstream and patched builds. Upstream ZrTe$_5$ at 1:5 gave $9.332$ against the invariant $10.468\,\mathrm{MeV\,cm^2/g}$ at 10 keV.

The test `test_the_real_collision_stopping_power_of_water_reproduces_estar` uses ESTAR water values that were independently recomputed. The Berger–Seltzer/ICRU 37 electron formula with no shell correction, $I=75$ eV, $Z/A=10/18.01528$ and $\delta\approx0$ at keV energies, is

$$
\frac{S}{\rho}=0.153537\,\frac{Z}{A}\,\frac{1}{\beta^2}
\left[\ln\frac{\tau^2(\tau+2)}{2(I/mc^2)^2}+1-\beta^2
+\frac{\tau^2/8-(2\tau+1)\ln2}{(\tau+1)^2}\right].
$$

It gives 119.78, 75.214, 38.410 and 22.561 $\mathrm{MeV\,cm^2/g}$ at 1, 2, 5 and 10 keV. These match the test's 119.8, 75.21, 38.41 and 22.56 and SBETHE's no-shell column (22.561 at 10 keV).

With `PYRITE_EXTERN_CODES_TESTS=1`, `tests/xsgen/test_extern_codes.py` passed all SBETHE tests, including the normalization-invariance and ESTAR anchors (23 passed). The one failure, `test_direction_resolved_ddcs_reproduces_the_vendor_interpolation` for the Au 2.5 MeV, 10° case, is a bremsstrahlung DDCS check; commit `d66c326d` touches no bremsstrahlung code.

No divergent factor, weight or normalization was found. The modified `CSHC` supports `rederived`.
