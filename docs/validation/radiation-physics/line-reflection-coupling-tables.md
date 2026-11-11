# Reflection coupling tables

Claim: `line-reflection-coupling-tables`. Independent verification, 2026-11-11.
This is an evaluation-order and storage claim; it introduces no new material
model. The reader need is an explanation of the equation and its verification.

## Independent derivation (recorded before reading implementation bodies)

Inputs examined first: the selected ledger record, its `pxr-amplitude` and
`cbs-amplitude` source records, and AST-extracted signatures/docstrings of
`reflection_coupling_tables`, `chi_g` and `U_g`. No existing derivation prose
or implementation body was read before writing this section.

Source: {cite:t}`feranchuk2000`, Eqs. (3) and (4), p. 4227. The original
paper's text was read through the [author-uploaded full text](https://www.researchgate.net/publication/12238791_Parametric_x-ray_radiation_and_coherent_bremsstrahlung_from_nonrelativistic_electrons_in_crystals)
(the publisher PDF returned HTTP 401). Eq. (3) has a negative susceptibility
prefactor and positive basis phase; Eq. (4) has the positive electrostatic
potential branch, nuclear charge minus electron form factor, and one thermal
factor. The paper assumes frequencies far from atomic resonances; the
energy-dependent anomalous extension below is the stated model contract,
not a claim that Eq. (3) itself supplies anomalous reference data.

Let $V$ be the cell volume in $\mathrm{\AA}^3$, $\mathbf R_i$ fractional
basis positions, and $\mathbf g=2\pi A^{-T}\mathbf h$ in
$\mathrm{\AA}^{-1}$. For a common isotropic $B$ in $\mathrm{\AA}^2$,

$$
p_i=\exp(2\pi i\mathbf h\cdot\mathbf R_i),\qquad
D_g=\exp[-B g^2/(16\pi^2)],\qquad
q= g/(4\pi).
$$

The amplitude thermal factor is applied once, after the sum; applying it to
intensity instead would double its exponent. Independent-atom X-ray scattering
uses the dimensionless electron-count amplitude
$f_{0,e}(q)+f'_e(E)+i f''_e(E)$. The static charge distribution uses
$Z_e-f_{0,e}(q)$ only: frequency-dependent absorption/dispersion is not part of
Poisson's equation for the static screened potential. Set the anomalous
correction to zero where the selected material-model policy omits it.

Partition the finite basis into sets of identical elements. Distributivity gives

$$
P_{g,e}=\sum_{i\in e}p_i,\qquad
c_g=\sum_e P_{g,e}f_{0,e}(q),\qquad
a_g=\sum_e P_{g,e}[Z_e-f_{0,e}(q)],
$$

$$
S_g(E)=D_g\left[c_g+\sum_eP_{g,e}(f'_e(E)+i f''_e(E))\right].
$$

Restoring units in the source susceptibility with
$\lambda=2\pi\hbar c/E$ and $r_e=e^2/(m_ec^2)$ gives

$$
\boxed{\chi_g(E)=-\frac{r_e\lambda^2}{\pi V}S_g(E)},\qquad
\boxed{\frac{U_g}{m_ec^2}
=\frac{4\pi e^2}{Vm_ec^2g^2}D_g a_g}.
$$

Here $e^2$ is in $\mathrm{eV\,\AA}$ and $m_ec^2$ in eV.
The function's literal `U_g/m_e` notation denotes division by the rest energy,
not by a mass in kilograms. Both outputs are dimensionless:
$r_e\lambda^2/V$ and $e^2/(Vg^2m_ec^2)$ each have zero length/energy degree.
The source's $U_g$ excludes the inverse volume and an electron charge;
the code's eV-valued `U_g` includes both, as its docstring states.

Limits and conventions: an element-wise extinct reflection has every
$P_{g,e}=0$, hence both couplings vanish (up to finite-sum roundoff). A compound
cancellation of $c_g$ alone need not survive anomalous corrections. At fixed
reflection the static coupling is independent of photon energy; the
non-resonant susceptibility scales as $E^{-2}$. For $B=0$, $D_g=1$;
positive $B$ suppresses amplitudes. Reversing $\mathbf h$ conjugates the
static coupling and the non-anomalous structure factor. With complex anomalous
factors, conjugating only the basis phase does not conjugate $i f''$;
Friedel conjugacy must not be imposed on the full anomalous susceptibility.
$g=0$ is outside the nonzero-reflection CBS derivation.

A mosaic rotation changes a reciprocal vector's laboratory direction, but not
its length, basis Miller phases, cell volume or common isotropic thermal
factor. Therefore all orientation rows for reflection $j$ can read table $j$.
The row map must duplicate the reflection index in the same order as geometry
rows, without duplicating orientation weights or changing interpolation indices.

## Implementation comparison and exact ledger discrepancy

The bodies were inspected only after the independent expressions above were
written. Reviewed revision: `032ec80ccf762443e4fdc814085204ea817a5e41`;
working-tree sources were also read directly. `reflection_coupling_tables`
implements the boxed expressions term for term: positive basis phase,
$g/(4\pi)$ elastic-factor argument, one $D_g$, negative susceptibility
prefactor, positive static-potential prefactor, and no anomalous correction in
$U_g$. Dispersion is evaluated once per element. Real/imaginary energy arrays
are float64 on the selected backend before `_reflection_tables` casts them to
`REAL`. The historical `henke_dispersion` name loads Chantler/FFAST data;
`cromer_mann_f0` loads Waasmaier–Kirfel data. Neither historical name describes
the present underlying data source.

**The ledger claim as written differs from the listed spectrum consumers.**
The first exact divergent convention is the susceptibility passed to
`_reflection_tables`: it calls `emission_coupling_tables`, which evaluates
negative Miller indices and conjugates the result. The spectrum table is
therefore

$$
\chi^{\rm line}_g=\chi_{-g}^{*}
=-\frac{r_e\lambda^2D_g}{\pi V}
\sum_eP_{g,e}[f_{0,e}+f'_e-i f''_e],
$$

whereas the ledger's unqualified $\chi_g$ expression contains $+i f''_e$.
The static row remains $U_g/(m_ec^2)$ because its form factors are real.
This is an exact sign difference in the anomalous imaginary atomic factor,
not a summation-order error. In a noncentrosymmetric crystal it can change
modulus as well as phase. The direct materials helper matches its own claim;
the full ledger record's statement about the line paths needs correction.
The physical choice of emission Fourier convention belongs to the separate
`line-energy-dispersion` claim; this verification does not certify it merely
because the wrapper documents it.

The storage/indexing part matches. `_batched_reflection_tables` appends
`i_hkl` in the same inner orientation loop as `G`, `ES`, `EP` and `WM`.
`ROW_TABLE` therefore repeats each reflection index once per orientation.
The same map reaches both `_interp_gather_cxr_tables` and
`run_coherent_prologue_kernel`. CPU gather uses the mapped row and the
original energy bracket. The CUDA elementwise gather computes
`table_row[g] * n_tab`; the coherent prologue uses
`table_row[reflection]` with `row * n_samples` in `_interp_row`.
Endpoint branches retain that mapped row. Geometry and orientation weights
are not multiplied into the stored couplings. Without mosaic the map is the
identity. These are source checks; device execution was not tested.

## Numerical evidence and tolerance

The requested focused command passed: `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv
run pyrite-dev test tests/materials/test_crystallography.py::test_reflection_coupling_tables_match_per_atom_couplings`
— **4 passed in 0.97 s**. That anchor checks production helper parity,
including NaN masks; by itself it is not independent reference evidence.

An additional scratch oracle reconstructed the direct lattice matrix from
catalog lattice lengths/angles, obtained the reciprocal vector by solving
$A^T\mathbf g=2\pi\mathbf h$, recomputed the cell volume as
$\lvert\det A\rvert$, and summed atoms directly. It called xraydb's raw
Waasmaier–Kirfel/Chantler factors and atomic numbers, without calling
`chi_g`, `U_g`, `structure_factor`, `henke_dispersion`,
`cromer_mann_f0`, `debye_waller` or `reciprocal_g_vector`.
The constants were explicitly transcribed at their declared precision.
This independently checks algebra, metric, argument conversion and
normalization; it shares the production atomic database and does not
independently validate that database's experimental accuracy.

Four reflections $(102),(103),(\overline{1}0\overline{2}),(110)$ were
checked at 100, 1838, 1840, 2000, 8000, 12657, 12659, 30000 and 900000 eV,
including both sides of Si/Se edge neighborhoods. Largest complex absolute
error divided by the largest reference modulus for that reflection:

| Material and model | Susceptibility | Static coupling |
| --- | ---: | ---: |
| HOPG, anomalous, $B=0$ | $8.96\times10^{-16}$ | $8.08\times10^{-16}$ |
| h-BN, anomalous, $B=0.4\,\mathrm{\AA}^2$ | $2.73\times10^{-16}$ | $6.25\times10^{-16}$ |
| h-BN, no anomalous terms, same $B$ | $1.86\times10^{-16}$ | $6.25\times10^{-16}$ |
| MoSe2, automatic edge-prone terms, $B=0.6\,\mathrm{\AA}^2$ | $4.15\times10^{-16}$ | $3.79\times10^{-16}$ |
| 4H-SiC, anomalous, $B=0.5\,\mathrm{\AA}^2$ | $5.17\times10^{-16}$ | $1.08\times10^{-15}$ |

For example, independent 4H-SiC (102) at 8000 eV gives
$\chi_g=2.8121062901210447\times10^{-6}
-5.062328688317329\times10^{-6}i$ and
$U_g/(m_ec^2)=-1.0753862067864172\times10^{-6}
+5.673967431566599\times10^{-6}i$.

An initially included extinct 4H-SiC (002) point illustrates why a relative
error against the residual of a zero structure factor is unsuitable:
different summation orders produce different rounding residuals. It was
excluded from the relative-error table, not interpreted as a finite physical
amplitude. The existing anchor's $10^{-12}$ absolute floor times the
reflection's peak coupling handles energy-dependent zero crossings and
centric imaginary components. For an everywhere extinct reflection, a
scale based on the sum of absolute atomic amplitudes would be more robust
than its own rounding residual. The table above uses a preregistered
$2\times10^{-13}$ peak-relative arithmetic threshold; observed errors are
near float64 rounding, so the anchor's looser $10^{-12}$ threshold is an
arithmetic-equivalence budget, not a claim about atomic-data uncertainty.

An independent mask check used the raw Chantler energy grids. At 0.5 eV,
exact lower endpoint 1.01 eV, 5 eV, 100 eV, exact upper endpoint
966266.74 eV and 2 MeV, anomalous HOPG/h-BN and edge-prone MoSe2 had
susceptibility NaNs exactly outside the strict open interval; static rows
stayed finite. h-BN without anomalous terms stayed finite throughout.
The mask is the union of unsupported elemental domains, even when an
extinct phase weight multiplies a NaN: zero times NaN is deliberately not
replaced by zero. Both real and imaginary masks were checked.

A CPU gather probe independently evaluated four coupling components with
nonidentity map `[1, 0, 1, 0]`, two segment rows, different interpolation
fractions and both endpoint branches. All outputs equalled the explicit
nested-loop reference exactly. The coherent prologue and fused CUDA gather
were inspected but not executed. No GPU runtime parity, performance result,
or old remote artifact is claimed as evidence for this revision.

## Filter results and adjudication

| Filter | Result |
| --- | --- |
| Units, per-cell normalization, constants | Pass; both table outputs dimensionless |
| Limits and thermal placement | Pass for nonzero reflections and declared material policy |
| Basis phase and static-potential sign | Pass for the direct materials helper |
| Anomalous convention across ledger and consumers | **Failure:** consumers use $-i f''$ after conjugate Friedel pairing; ledger says $+i f''$ |
| Reference values and provenance | No experimental values transcribed; atomic data shared with production, explicitly limited evidence |
| Tolerance | Float64 arithmetic budget; zero-amplitude caveat recorded above |
| CPU indexing evidence | Pass; independent gather probe and source flow |
| CUDA parity evidence | Untested for this revision |

Verdict: **discrepancy for the ledger record as written**. The direct
`reflection_coupling_tables` equation is independently rederived; storage
sharing and row mapping agree with the inspected consumers. Suggested owner
edit: distinguish the direct materials result $\chi_g$ from the line result
$\chi_{-g}^{*}$; state that the line wrapper reverses Miller indices and
conjugates before `REAL` conversion, with the physical emission convention
owned by `line-energy-dispersion`. Preserve the static $U_g$ expression,
NaN contract and pending hardware evidence. After that explicit correction,
the corrected evaluation/storage claim can be recorded as `rederived`;
the current wording must not be promoted unchanged. Human sign-off remains
separate.

Only this validation document was changed by the verifier. Scratch scripts
and outputs were kept under `/tmp`. Docs tests, Sphinx build and positive
render inspection are delegated to the integrating owner by explicit task
instruction; they were not run concurrently here.
