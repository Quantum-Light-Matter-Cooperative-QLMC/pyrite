# Diffracted bremsstrahlung (DBR)

Validation: `diffracted-bremsstrahlung`. Status: **rederived**. Issue #1166, item 2.

## Need

A detector at a Bragg angle of reflection $g$ sees, besides PXR/CBS, forward radiation that the crystal Bragg-diffracts into it. For channeled electrons this is diffracted channeling radiation (DCR; [`cr-planar-zone-pxr-modulation`](../channeling/cr-planar-zone-pxr-modulation.md)). For every electron it includes incoherent bremsstrahlung. The photon travelling along $Q=k\hat n+g$ with $|Q|=k$ leaves along $\hat n$ at the Bragg energy

$$
E_B=-\frac{\hbar c\,g^2}{2\,\hat n\cdot g},\qquad \hat n\cdot g<0 .
$$

When $\hat Q$ lies inside the forward brem cone (the mirror geometry, $\hat Q\approx\hat v$), this line competes with PXR. Native scoring omitted it, with channeling on or off.

## Derivation

**Reciprocity.** The intensity an incoherent source radiates into vacuum direction $\hat n$ equals its emission into the crystal eigenmode that a plane wave from $\hat n$ excites, time reversed. Inside a two-beam Laue crystal that mode has a transmitted component ($\approx\hat n$) and a diffracted component ($Q$). For amplitudes $P=(T,D)$ along the inward depth $u$ from the exit face, the slowly varying two-beam equations give

$$
\frac{dP}{du}=iNP,\qquad N=\begin{pmatrix}k\chi_0/2n_s & a\\ b & q+k^2\chi_0/2Q_s\end{pmatrix},\qquad P(0)=(1,0),
$$

with $q=-\det/2Q_s$, $\det=g^2+2k\,\hat n\cdot g$, $a=k^2\chi_{-g}C_s/2kn_s$ (the code takes $ab$ with $|\chi_g|^2$, i.e. no anomalous Borrmann absorption), $b=k^2\chi_gC_s/2Q_s$, and $C_s$ the polarization factor ($1$ for σ, $|\hat k\cdot\hat Q|$ for π). Here $\operatorname{Im}\chi_0>0$ absorbs. With $\lambda_{1,2}$ the eigenvalues of $N$,

$$
D_s(u)=b\,\frac{e^{i\lambda_1u}-e^{i\lambda_2u}}{\lambda_1-\lambda_2}.
$$

A source radiating the unattenuated vacuum density $j(\hat Q,E)$ at depth $z$ couples to the mode's diffracted plane wave with amplitude $D_s(L-z)$, so

$$
\frac{d^2N}{dE\,d\Omega_n}=\sum_{\rm rows}j_{\rm row}(\hat Q,E)\;\tfrac12\sum_s|D_s(L-z_{\rm row},E)|^2 ,
$$

for unpolarized brem, to first order in $\chi$ ($|Q|=k$). No $\gamma_h/\gamma_0$ flux factor appears: the result is per steradian of $\hat n$, and reciprocity works with field amplitudes. $\operatorname{Im}\chi_0$ attenuates both legs of the path, $\operatorname{Re}\chi_0$ shifts the line by $\operatorname{Re}(N_{11}-N_{22}+q)$ in $q$, and $ab$ is the Pendellösung depletion.

**Kinematic limit.** For $ab,\chi_0\to0$, $|D|^2=|b|^2\,4\sin^2(qu/2)/q^2$. Since $dq/dE=-\hat n\cdot g/(\hbar cQ_s)$,

$$
\int dE\,|D_s|^2=|b|^2\,2\pi u\,\frac{\hbar c\,Q_s}{|\hat n\cdot g|},
$$

which is linear in the remaining depth. Averaged over a uniform emission depth this gives the "mean depth ½" of the #1148 DCR estimate. The tests check it against the textbook kinematic Laue integrated reflectivity $R_\theta=\pi k|\chi_g|^2C^2t/(2\gamma_0\sin2\theta_B)$, with $R_E=E\,R_\theta/\tan\theta_B$.

## Implementation

`spectrum/diffracted_brem.py` evaluates, per reflection:

1. $j$ for every brem row toward $\hat Q(E_B)$ at $E_B$, using native's continuum model (BremsLib direction-resolved, else isotropic EEDL, with the same fallbacks), cutoff, row clipping and channeling flux weight, but without the escape factor.
2. The rows' weights spread uniformly over each row's depth range onto 64 depth bins.
3. $S(E)=\sum_k w_k\,\tfrac12\sum_s|D_s(L-z_k,E)|^2$, on a grid resolving $2\pi/L$ in $q$ over $\pm200$ kinematic widths plus 20 Pendellösung widths, capped at ±1.5 keV.
4. Bin-mean averaging (band conserving) or node sampling onto the line grid, following `line_quadrature`.

`runner/diffracted_brem.py` adds the result to `spec` and `spec_coherent` for perfect single-slab crystal cases. Layer stacks, mosaic crystals and grooved faces add nothing.

## Checks

| check | result |
|---|---|
| $|D|^2$ vs `expm` of $N$ (strong coupling, absorbing) | 1e-8 |
| kinematic $\int dE|D|^2$ vs $|b|^2 2\pi u\hbar cQ_s/|n\cdot g|$ | 2e-3 (window tail) |
| bin-mean deposit conserves the band | 1e-9 |
| kinematic DBR vs native brem toward $\hat Q$ × $R_E$ × ½ (17 MeV Si, 2 µm, (1 1 −3), (−1 3 1)) | within 3% |
| full / estimate: (0 2 −2) 7 keV, (1 1 −3) 10 keV, (−1 3 1) 8.6 keV | 0.94, 0.99, 0.98 |
| DBR / native PXR, same reflection, mirror rays, 6–10 keV | 0.36–24 |

**Omitted direct-beam depletion (assumption, #1176).** The same mode's transmitted amplitude weights the native direct emission toward $\hat n$. Without absorption, flux conservation $\gamma_n|T|^2+\gamma_h|D|^2=\gamma_n$ means that emission loses $(\gamma_h/\gamma_n)|D|^2$ near $E_B$. That term is not scored. Relative to the scored gain it is of order $j(\hat n,E)/j(\hat Q,E)$. A one-ray estimate at the mirror rays integrates the depth-weighted $|T|^2-|T_0|^2$ against the gain:

| reflection | $j(\hat n)/j(\hat Q)$ | depletion / gain |
|---|---|---|
| (0 2 −2) | 3.2e-6 | −5.8e-6 |
| (1 1 −3) | 6.9e-6 | −1.0e-5 |
| (−1 3 1) | 3.9e-6 | −6.1e-6 |

The depletion is negligible, so DBR keeps no depletion term. For an isotropic source such as a characteristic line it is the same size as the gain. Kossel lines score both terms: see [`diffracted-characteristic-radiation`](diffracted-characteristic-radiation.md).

At (0 2 −2), $\sqrt{ab}\,t\approx0.6$: the Pendellösung depletion is −5% and two-beam absorption −4%. The independent estimate has neither.

## Fresh-context verification (2026-11-13)

An independent verifier (`physics-validator`) re-derived the claim.

| item | result |
|---|---|
| reciprocity: $j(\hat Q,E)\tfrac12\sum_s|D_s|^2$ per sr of $\hat n$, no $\gamma_h/\gamma_0$, $T$–$D$ cross terms absent for incoherent brem | pass |
| two-beam equations, $b$, $C_s$, closed form, small-split series | pass. **Fail**: the principal root let $\operatorname{Im}(\lambda_1-\lambda_2)<0$ and overflowed to NaN for $\tfrac{\mu}{2}|1/\gamma_h-1/\gamma_n|u>709$ (thick, absorbing, asymmetric Laue). Fixed with the $\operatorname{Im}\ge0$ branch and a finite-output test |
| $ab$ notation | concern: the code uses $|\chi_g|^2$, not $\chi_g\chi_{-g}$; stated as an assumption (no Borrmann absorption) |
| kinematic integral and $R_\theta$, with $\gamma_0=\hat Q\cdot\hat s$ | pass. (1 1 −3), 10 keV: kinematic / analytic 0.9990; full / kinematic 0.980 |
| `_segment_density` vs `mc_brem_spectrum` with escape off | 1.000000 (BremsLib, EEDL; channeling off and on) |
| refraction shift | pass |
| depth histogram, window, deposit | pass. The quarter-width step under-resolves only for $\sqrt{ab}L\gtrsim785$ (mm crystals) |
| scope guards; DBR in both `spec` and `spec_coherent` | pass; no overlap with the zone DCR source |

Consumer gap (documented): the adaptive `"cxr"` stopping measure, the energy-grid bandwidth and convergence probes and line roulette score lines without DBR. Verdict after the fixes: `rederived`. Probes: `scratch/verify_dbr_probe.py`, `scratch/test_verify_dbr_probe.py` in the task worktree (not committed).

