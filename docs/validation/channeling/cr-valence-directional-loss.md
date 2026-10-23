# `cr-valence-directional-loss` — direction-resolved valence oscillators of a uniaxial crystal

Validation: `cr-valence-directional-loss`.

## Claim

From the ledger row and the docstrings of the anchored symbols (read before
any implementation body):

- A uniaxial crystal answers a longitudinal transfer $\mathbf q$ at angle
  $\alpha$ to its optic axis $\hat{\mathbf c}$ through
  $\epsilon(\hat{\mathbf q},\omega)=\epsilon_\perp(\omega)\sin^2\alpha+\epsilon_\parallel(\omega)\cos^2\alpha$
  in the optical limit. The two loss functions are not mixed.
- The valence band becomes delta oscillators $(W_k,f_k(\cos^2\alpha))$ on one
  shared energy grid, with
  $f_k=N\,\frac{2}{\pi\Omega^2}\int_{\rm bin\,k}W\,\mathrm{ELF}(\alpha,W)\,dW$,
  $\mathrm{ELF}=\operatorname{Im}[-1/\epsilon]$ and
  $\Omega^2=4\pi n_{\rm val}r_e(\hbar c)^2$ [eV²]. Each direction is rescaled
  to exactly $N$ electrons after its raw closure is reported. A closure
  outside $1\pm$ `tolerance` raises.
- Per-direction diagnostics are the static sum
  $\frac{2}{\pi}\int\mathrm{ELF}/W\,dW$ against $1-1/\epsilon_1(0)$ and the
  logarithmic sum $\sum_k f_k\ln W_k$.
- Above the last tabulated energy $W_{\max}$ there is a flagged tail
  $\mathrm{ELF}(W)=\operatorname{Im}\epsilon(W_{\max})\,(W_{\max}/W)^3$.
- In the kernel, the distant strength of oscillator $k$ is $f_k(\cos^2\alpha_k)$
  with $\cos^2\alpha_k=(\hat{\mathbf c}\cdot\mathbf q)^2/q^2$ and
  $\mathbf q=(q_x,q_y,W_k/\hbar v)$. Close collisions use the sphere-averaged
  $\bar f_k$, taken uniform in $\cos\alpha$ (`DirectionalValence` docstring).
  Bound shells stay isotropic.
- The catalog conduction-band resonance becomes
  $W_{\rm cb}=\exp(\sum_k\bar f_k\ln W_k/N)$, so the catalog mean excitation
  energy $I$ is unchanged.
- `load_directional_valence` takes the plasma energy of the $N_{\rm cb}$
  conduction electrons as $\Omega_p\sqrt{N_{\rm cb}/N_{\rm total}}$.

Units: energies in eV, $n$ in Å⁻³, $r_e$ in Å, $\hbar c$ in eV Å, and $f_k$ in
electrons per formula unit.

## Independent derivation (before reading the bodies)

### (a) Plasma energy

For $n$ electrons per unit volume, $\omega_p^2=4\pi ne^2/m$ (Gaussian units).
Multiply by $\hbar^2$ and write $e^2=r_emc^2$:

$$
\Omega^2\equiv(\hbar\omega_p)^2=4\pi n\frac{e^2}{mc^2}(\hbar c)^2
=4\pi n\,r_e(\hbar c)^2 .
$$

Units: $[\text{Å}^{-3}][\text{Å}][\text{eV}^2\text{Å}^2]=\text{eV}^2$. If
$n_{\rm tot}=Z\,n_{\rm f}$ for $n_{\rm f}$ formula units per volume, then the
valence-only value is $n_{\rm val}=N\,n_{\rm f}$, so
$\Omega_{\rm val}=\Omega_p\sqrt{N/Z}$. This is the `load_directional_valence`
rule.

### (b) Uniaxial mixing

In the optical limit $q\to0$, the longitudinal dielectric function of an
anisotropic medium is the projection of the tensor on $\hat{\mathbf q}$:
$\epsilon_L(\hat{\mathbf q},\omega)=\hat{\mathbf q}\cdot\boldsymbol\epsilon(\omega)\cdot\hat{\mathbf q}$.
Fast-electron energy loss is longitudinal (the scalar potential obeys
$\nabla\cdot\boldsymbol\epsilon\nabla\phi=4\pi\rho_{\rm ext}$). In a uniaxial
crystal, $\boldsymbol\epsilon=\operatorname{diag}(\epsilon_\perp,\epsilon_\perp,\epsilon_\parallel)$
in the crystal frame. That gives

$$
\epsilon(\hat{\mathbf q},\omega)=\epsilon_\perp\sin^2\alpha+\epsilon_\parallel\cos^2\alpha,
\qquad
\frac{d^2\sigma}{dW\,d^3q}\propto\frac{1}{q^2}\operatorname{Im}\!\left[\frac{-1}{\epsilon(\hat{\mathbf q},W)}\right],
$$

which is Bassani–Tosatti / Tarrio–Schnatterly Sec. II. The tensor enters
linearly. The loss function is the nonlinear $-1/\epsilon$ of the mixture, not
$\sin^2\alpha\,\mathrm{ELF}_\perp+\cos^2\alpha\,\mathrm{ELF}_\parallel$. The
two agree only at $\alpha=0$ and $\alpha=\pi/2$, or when
$\epsilon_\perp=\epsilon_\parallel$.

Positivity: $\operatorname{Im}\epsilon(\alpha)=\sin^2\alpha\,\epsilon_{2\perp}+\cos^2\alpha\,\epsilon_{2\parallel}\ge0$
whenever both components are non-negative. Then
$\mathrm{ELF}=\operatorname{Im}\epsilon/\lvert\epsilon\rvert^2\ge0$ for every
$\alpha$.

### (c) f-sum per direction

Every component $\epsilon_j$ is causal, with
$\epsilon_j\to1-\omega_p^2/\omega^2$ as $\omega\to\infty$, and the mixture
$\epsilon(\alpha)$ inherits both properties because it is a convex combination
with weights $\sin^2\alpha+\cos^2\alpha=1$. The function $1/\epsilon(\alpha)$
is also causal (no zeros in the upper half plane for a passive medium), with
$1/\epsilon\to1+\omega_p^2/\omega^2$. The Kramers–Kronig asymptote of
$\operatorname{Re}(1/\epsilon)-1$ then gives the Nozières–Pines sum

$$
\int_0^\infty\omega\,\operatorname{Im}\!\left[\frac{-1}{\epsilon(\alpha,\omega)}\right]d\omega=\frac{\pi}{2}\omega_p^2
\;\Longrightarrow\;
\frac{2}{\pi\Omega^2}\int_0^\infty W\,\mathrm{ELF}(\alpha,W)\,dW=1 .
$$

The same $\omega_p$ holds in every direction. Hence

$$
\frac{df}{dW}(\alpha)=N\frac{2}{\pi\Omega^2}W\,\mathrm{ELF}(\alpha,W),\qquad
f_k(\alpha)=\int_{\rm bin\,k}\frac{df}{dW}dW,\qquad
\sum_kf_k(\alpha)=N\cdot\mathcal C(\alpha),
$$

with raw closure
$\mathcal C(\alpha)=\frac{2}{\pi\Omega^2}\int W\,\mathrm{ELF}\,dW=\sum_kf_k/N$.
Here $\Omega$ already belongs to the $N$ electrons, so no further $1/N$
appears. The ledger's
$\frac{2}{\pi\Omega^2N}\int\omega\,\mathrm{ELF}\,d\omega$ has one $1/N$ too
many if read literally. It is only correct if "$\int$" means $\sum f_k$.
This is checked against the code in the comparison below. The prefactor
matches the PENELOPE optical oscillator-strength density built from the loss
function, $df/dW=(2Z/\pi\Omega_p^2)\,W\,\operatorname{Im}(-1/\epsilon)$, with
$Z\to N$ and $\Omega_p\to\Omega$.

Rescaling each direction by $1/\mathcal C(\alpha)$ makes $\sum_kf_k=N$
exactly. The rescaling is an admitted normalization error that the raw
closure records.

### (d) Static sum

At $\omega=0$, the Kramers–Kronig relation for $1/\epsilon$ of an insulator
($\epsilon_2(0)=0$) gives

$$
1-\frac{1}{\epsilon_1(\alpha,0)}=\frac{2}{\pi}\int_0^\infty\frac{\mathrm{ELF}(\alpha,\omega)}{\omega}d\omega,
\qquad
\epsilon_1(\alpha,0)=\epsilon_{1\perp}(0)\sin^2\alpha+\epsilon_{1\parallel}(0)\cos^2\alpha .
$$

The integrand is dimensionless and the same in $\omega$ or $W$. The target
mixes $\epsilon_1(0)$ linearly, not $1/\epsilon_1(0)$.

### (e) High-energy tail

As $\omega\to\infty$, $\epsilon\to1$, so
$\mathrm{ELF}=\epsilon_2/(\epsilon_1^2+\epsilon_2^2)\to\epsilon_2$. A Drude
or Lorentz term $\epsilon_2=\omega_p^2\gamma\omega/[(\omega_0^2-\omega^2)^2+\gamma^2\omega^2]$
falls as $\omega_p^2\gamma/\omega^3$ for $\omega\gg\omega_0,\gamma$. The
free-electron tail is therefore $\omega^{-3}$, with amplitude fixed by
$\epsilon_2$ (not the screened ELF) at a point where $\lvert\epsilon\rvert$
is near 1:

$$
\mathrm{ELF}_{\rm tail}(\alpha,W)=\operatorname{Im}\epsilon(\alpha,W_{\max})\left(\frac{W_{\max}}{W}\right)^3,
\qquad
\operatorname{Im}\epsilon(\alpha)=\sin^2\alpha\,\epsilon_{2\perp}+\cos^2\alpha\,\epsilon_{2\parallel}.
$$

Just above a volume plasmon, $\epsilon_1<1$ and $\lvert\epsilon\rvert<1$.
There $\mathrm{ELF}>\epsilon_2$, so a tail matched to $\mathrm{ELF}(W_{\max})$
overestimates every higher-energy point. That is consistent with the ledger's
rejected 1.19/1.34 closure. The tail moments are analytic:

$$
\frac{2}{\pi\Omega^2}\int_{W_{\max}}^\infty W\,\mathrm{ELF}_{\rm tail}dW
=\frac{2}{\pi\Omega^2}\,\operatorname{Im}\epsilon(W_{\max})\,W_{\max}^2,
\qquad
\frac{2}{\pi}\int_{W_{\max}}^\infty\frac{\mathrm{ELF}_{\rm tail}}{W}dW
=\frac{2}{3\pi}\operatorname{Im}\epsilon(W_{\max}).
$$

The f-sum tail moment diverges logarithmically for $\omega^{-2}$ and is
finite for any exponent $>2$. The exponent 3 is the Drude choice that
Bell–Liang also fit. A real core-free valence band falls faster once its
oscillators are exhausted, so the tail is an upper-side assumption. The code
flags it.

If the tail strength is put into the binned oscillators, it needs an energy.
The f-sum-weighted log mean of the tail is
$\int_{W_m}^\infty(W_m^2/W^2)\ln W\,dW/W_m=\ln W_m+1$, that is
$W_{\rm tail}=eW_{\max}$. This is a reference for the comparison only. The
ledger does not fix it.

### (f) Direction of the transfer in the kernel

Frame: $x$ is the plane normal, $y$ is in-plane transverse, and $z$ is the
beam. For a delta oscillator of energy $W_k$, the minimum longitudinal
transfer for $W_k\ll E$ is $q_z=W_k/\hbar v$ (to leading order also for
relativistic $v$). The transverse kernel integrates over
$\mathbf q_\perp=(q_x,q_y)$, so

$$
\cos^2\alpha_k=\frac{(\hat{\mathbf c}\cdot\mathbf q)^2}{q_x^2+q_y^2+q_z^2},\qquad
\hat{\mathbf c}=\hat{\mathbf x}:\ \frac{q_x^2}{q^2},\quad
\hat{\mathbf c}=\hat{\mathbf y}:\ \frac{q_y^2}{q^2},\quad
\hat{\mathbf c}=\hat{\mathbf z}:\ \frac{q_z^2}{q^2}.
$$

In the radial δ-oscillator kernel, the distant-longitudinal weight of
oscillator $k$ is $f_k$ times a function of $q$ and $W_k$ only, namely the
$1/q^2$ optical-limit factor below the Bethe ridge. By (b) the optical-limit
direction dependence enters only through
$\operatorname{Im}[-1/\epsilon(\hat{\mathbf q})]$. Replacing
$f_k\to f_k(\cos^2\alpha_k)$ inside the distant weight is therefore exact
within the optical-limit assumption.

Isotropic limit: if $\epsilon_\perp=\epsilon_\parallel$, then $f_k$ does not
depend on $\alpha$ and the kernel reduces to the radial one for every
$\hat{\mathbf c}$.

The kernel is $\int d^2q\,w(\mathbf q)\,s_{\mathbf q}s_{\mathbf q}^\dagger$
with $w\ge0$, because $f_k(\alpha)\ge0$ by (b). It is a positive sum of
rank-one Hermitian terms, so it is Hermitian positive semidefinite for any
direction dependence of $w$.

### (g) Sphere average

Close collisions do not see the crystal axis, so one isotropic strength is
needed. The polycrystalline (orientation) average of the loss is the solid
angle mean of the ELF-derived strength. With $u=\cos\alpha$ and $t=u^2$:

$$
\bar f_k=\frac{1}{4\pi}\int f_k\,d\Omega=\frac12\int_{-1}^1f_k(u^2)\,du
=\int_0^1f_k(u^2)\,du=\int_0^1f_k(t)\frac{dt}{2\sqrt t}.
$$

On nodes uniform in $t=\cos^2\alpha$, a plain trapezoid in $t$ is wrong: it
gives $\int f\,dt$, which is uniform in $\cos^2\alpha$ and underweights
$\alpha\to\pi/2$. The correct weight is $1/(2\sqrt t)$, integrable at $t=0$,
or equivalently integration in $u$. Because each node row sums to $N$ after
rescaling, $\sum_k\bar f_k=N$ for any normalized weight. Since $f_k(t)$ is
nonlinear in $t$, the weight matters for the individual $\bar f_k$.

### (h) Mean excitation energy

The PENELOPE oscillator model gives
$Z\ln I=\sum_if_i\ln W_i$. The catalog uses one conduction-band oscillator
$(N,W_{\rm cb})$, and the bound-shell Sternheimer factor is solved against
that $W_{\rm cb}$. Replacing it by $\{(\bar f_k,W_k)\}$ leaves $\sum f\ln W$
unchanged if and only if

$$
N\ln W_{\rm cb}=\sum_k\bar f_k\ln W_k
\;\Longleftrightarrow\;
W_{\rm cb}=\exp\!\left(\frac{1}{N}\sum_k\bar f_k\ln W_k\right).
$$

The resonance must be fed to the Sternheimer solve before the bound shells
are fixed, so that the bound shells close $I$ with it. The individual
directions then carry $\sum_kf_k(\alpha)\ln W_k\ne N\ln W_{\rm cb}$, an
anisotropic Bethe logarithm, and only the orientation average reproduces $I$.

### Units and limits summary

| Quantity | Units | Limit or identity |
|---|---|---|
| $\Omega^2=4\pi nr_e(\hbar c)^2$ | eV² | $\Omega_{\rm val}=\Omega_p\sqrt{N/Z}$ |
| $f_k$ | electrons/formula | $\sum_kf_k=N\,\mathcal C$, so $N$ after rescaling |
| $\mathcal C=\frac{2}{\pi\Omega^2}\int W\,\mathrm{ELF}\,dW$ | 1 | $1$ for consistent data |
| $\frac{2}{\pi}\int\mathrm{ELF}/W\,dW$ | 1 | $1-1/\epsilon_1(\alpha,0)$ |
| tail | 1 | $\propto W^{-3}$, amplitude $\operatorname{Im}\epsilon(\alpha,W_{\max})$ |
| $\cos^2\alpha_k$ | 1 | $(\hat{\mathbf c}\cdot\mathbf q)^2/q^2$, $q_z=W_k/\hbar v$ |
| $\bar f_k$ | electrons/formula | uniform in $\cos\alpha$, weight $dt/2\sqrt t$ in $t=\cos^2\alpha$ |
| $W_{\rm cb}$ | eV | $\exp(\sum\bar f\ln W/N)$; equals $W_k$ for a single oscillator |

## Diff against the implementation

The bodies were read after the section above was written. Symbol by symbol:

| Item | Independent | Code | Verdict |
|---|---|---|---|
| Mixing | $\epsilon_\perp(1-t)+\epsilon_\parallel t$, $t=\cos^2\alpha$, then $-\operatorname{Im}(1/\epsilon)$ | `_mix`, `UniaxialDielectric.elf` | match; mixes $\epsilon$, not ELF |
| $\Omega^2$ | $4\pi nr_e(\hbar c)^2$ | `_PLASMA_COEFFICIENT` $=4\pi\cdot2.8179403262\times10^{-5}\,\text{Å}\cdot(\hbar c)^2$ | match |
| $\Omega_{\rm val}$ | $\Omega_p\sqrt{N/Z}$ | `load_directional_valence` | match |
| $f_k$ | $N\frac{2}{\pi\Omega^2}\int_{\rm bin}W\,\mathrm{ELF}\,dW$ | `norm * weight`, with `weight` the cumulative trapezoid of $W^2\mathrm{ELF}\,d\ln W$ interpolated to the bin edges | match ($W\,dW=W^2\,d\ln W$) |
| closure | $\sum_kf_k/N=\frac{2}{\pi\Omega^2}\int W\,\mathrm{ELF}\,dW$ | `raw.sum(axis=1) / electrons` | match; see finding 1 on the ledger notation |
| rescaling | after closure, raise outside tolerance | `strengths = raw * electrons / raw.sum` after the tolerance check | match |
| static sum | $\frac{2}{\pi}\int\mathrm{ELF}/W\,dW$, target $1-1/(\epsilon_{1\perp}(1-t)+\epsilon_{1\parallel}t)$ | `trapezoid(elf, log(fine))`, `target` | match (lower limit is the first tabulated energy, finding 3) |
| tail | $\operatorname{Im}\epsilon(\alpha,W_{\max})(W_{\max}/W)^3$ | `eps2_end * (e1/fine)**3`, `eps2_end` from the mixed $\epsilon$ at the last point | match; exponent 3 and the $\operatorname{Im}\epsilon$ amplitude agree. Tail binned into 12 geometric bins up to $10^{2.5}W_{\max}$, which drops a fraction $10^{-2.5}$ of the tail moment |
| $\bar f_k$ | $\int_0^1f_k(u^2)\,du$ | `_sphere_mean`: trapezoid on 2049 points uniform in $c=\cos\alpha$, node interpolant at $c^2$ | match; uniform in $\cos\alpha$, not in $\cos^2\alpha$ |
| $\cos^2\alpha_k$ | $(\hat{\mathbf c}\cdot\mathbf q)^2/q^2$, $q_z=W_k/\hbar v$ | `DirectionalChannels._cos2`: `p**2`, `q_y**2` or `distant_qmin_inv_ang**2` over `p**2+q_y**2+qmin**2` | match; `qmin` is the PENELOPE $q(Q_-)$, equal to $W_k/\hbar v$ to $2\times10^{-7}$ at 16.92 MeV |
| distant weight | radial weight with $f_k\to f_k(t_k)$ | channels built with $\bar f_k$, then `distant_ang2 * ratio`, `ratio` $=f_k(t)/\bar f_k$ by linear interpolation in $t$ | match |
| close | isotropic, $\bar f_k$ | `close(hypot(p, q_y))` from channels built on $\bar f_k$ | match |
| diagonal | azimuthal mean of $w$ at fixed $\lvert q_\perp\rvert$ (since $\rho_q$ depends on $\lvert q_\perp\rvert$ only) | `distant`: 16-point Gauss–Legendre mean over $\phi\in[0,\pi/2]$ | match (quarter-plane symmetry of $\cos^2\phi$, $\sin^2\phi$) |
| PSD | $w\ge0$, rank-one sum | strengths $\ge0$ from ELF $\ge0$ (linear interpolation of non-negative $\epsilon_2$); kernel symmetrized | match |
| $W_{\rm cb}$ | $\exp(\sum\bar f\ln W/N)$, before the Sternheimer solve | `mean_log_energy_eV`, passed as `conduction_resonance_eV` into `build_shell_oscillators` | match |

## Independent numerical checks

Scripts:
`/tmp/claude-1000/-home-alex-dev-pyrite/a6d7b22a-bb10-4e37-9fdd-dd3bf02926b0/scratchpad/check.py`
and `check2.py`. The references use `scipy.integrate.quad` on the closed-form
Lorentz tensor. No implementation helper serves as a reference.

Synthetic tensor ($\Omega=20$ eV, $N=18$): $\perp$ poles at 6 and 22 eV with
shares 0.3 and 0.7, $\parallel$ poles at 14 and 25 eV with shares 0.45 and
0.55. On $[0,\infty)$ the reference integrals reproduce f-sum $=1$ and the
static sum $=1-1/\epsilon_1(\alpha,0)$ to $10^{-6}$ for
$t\in\{0,\tfrac12,1\}$, which confirms (c)–(d).

| Check | $t$ | Code | Independent |
|---|---|---|---|
| (a) table 0.05–2000 eV, no tail: closure | 0, 0.5, 1 | 0.998250, 0.998210, 0.998170 | 0.998249, 0.998209, 0.998169 |
| (a) static sum | 0, 0.5, 1 | 0.79616, 0.72127, 0.55943 | exact 0.79641, 0.72153, 0.55954 |
| (b) table to 40 eV + tail: closure | 0, 0.5, 1 | 1.01899, 1.02811, 1.03705 | 1.01879, 1.02788, 1.03677 |
| (b) tail fraction | 0, 0.5, 1 | 0.1575, 0.1742, 0.1906 | 0.1573, 0.1739, 0.1903 |
| (b) static sum incl. tail $\frac{2}{3\pi}\operatorname{Im}\epsilon(W_{\max})$ | 0, 0.5, 1 | 0.79328, 0.71844, 0.55660 | 0.79323, 0.71838, 0.55653 |
| (b) closure with an ELF-matched tail | 0, 0.5, 1 | — | 1.196, 1.235, 1.276 (the rejected alternative overshoots, as the ledger reports for MoS₂) |
| (c) $\bar f_k$, $W_k=9.60$ eV | — | 0.13383 | uniform in $\cos\alpha$: 0.13402; uniform in $\cos^2\alpha$: 0.08594 |
| (c) $\bar f_k$, $W_k=27.35$ eV | — | 2.18894 | uniform in $\cos\alpha$: 2.18946; uniform in $\cos^2\alpha$: 2.01528 |
| (c) $f_k(t=0)$, $W_k=27.35$ eV | 0 | 2.57021 | 2.57100 (exact-angle quad, independently normalized) |

The $\sim2\times10^{-4}$ closure offset in (b) is the trapezoid across the
data-to-tail step at $W_{\max}$. The $\bar f_k$ differences ($\le0.15\%$) are
the error of linear interpolation between the 33 $\cos^2\alpha$ nodes. The
$\cos^2\alpha$-uniform alternative differs by up to 36%, so the check does
discriminate the convention.

Registered MoS₂ (`check2.py`): raw closure 0.886/0.929, static sum
0.866/0.816 against 0.933/0.889, and tail share 0.160/0.211 ($\perp$/$\parallel$).
These reproduce the ledger's Checks field.
$W_{\rm cb}=28.4145$ eV equals $\exp(\sum\bar f\ln W/N)$. $\sum_if_i\ln W_i/Z$
over the bound shells plus $\{\bar f_k,W_k\}$ equals $\ln I_{\rm catalog}=5.679235$
to all printed digits. For $c\in\{x,y,z\}$, the code's distant weight of the
dominant oscillator at $(q_x,q_y)=(0.03,0.02)$ Å⁻¹ equals
$\text{radial}_k\cdot f_k(t)/\bar f_k$, with $t$ computed independently from
$(\hat{\mathbf c}\cdot\mathbf q)^2/q^2$, to machine precision.

`pyrite-dev test tests/montecarlo/test_channeling_directional.py`: 14 passed.

## Findings

None changes a factor, sign or exponent in the implementation.

1. **Ledger notation (closure).** The ledger Claim writes the raw closure as
   $\frac{2}{\pi\Omega^2N}\int\omega\,\mathrm{ELF}\,d\omega$, and the
   `DirectionalValence.closure` docstring as
   "`(2 / pi Omega^2) int w ELF dw / N`". Since $\Omega$ is the plasma energy
   of the $N$ electrons, the dimensionless closure is
   $\frac{2}{\pi\Omega^2}\int W\,\mathrm{ELF}\,dW=\sum_kf_k/N$. The code
   computes this correctly (`raw.sum / electrons`, with `raw` already carrying
   $N$). Read literally, the written formula has one extra factor $1/N$.
   Suggested wording: "$\sum_kf_k/N=\frac{2}{\pi\Omega^2}\int W\,\mathrm{ELF}\,dW$".
2. **Shared $W_k$ convention.** Each $W_k$ is the log-centroid of the raw
   bin weight averaged uniformly over the $\cos^2\alpha$ nodes, not
   sphere-weighted. It is only a representative energy inside a narrow
   geometric bin, so the f-sum is unaffected. The per-direction
   $\sum f\ln W$ diagnostic and $W_{\rm cb}$ use these shared energies, so the
   $I$ closure is exact by construction for the chosen $W_k$. Not a
   discrepancy.
3. **Static-sum lower limit.** The static sum is integrated from the first
   tabulated energy, not from 0. For MoS₂ any sub-table ELF is missing from
   the diagnostic. This is diagnostic only and does not change the strengths.
4. **Tail.** The $W^{-3}$ exponent and the $\operatorname{Im}\epsilon$
   amplitude are the correct $\epsilon\to1$ Drude asymptote. A finite
   $W_{\max}$ still overshoots: for the synthetic tensor cut at 40 eV,
   closure is 1.02–1.04 against the exact 1. The ELF-matched alternative
   gives 1.20–1.28. For MoS₂ the tail carries 16–21% of $N$, so its strength
   distribution above 31.5 eV is an assumption rather than data, as the code
   labels it. The tail is truncated at $10^{2.5}W_{\max}$, which loses
   $0.3\%$ of the tail moment before rescaling.
5. **Provisional data (not adjudicated).** The MoS₂ static-sum deficit
   (0.866 vs 0.933 for $\perp$, 0.816 vs 0.889 for $\parallel$, about 7–8%) and
   the closure deficit (0.886/0.929) are data or provenance properties under
   #541, consistent with the assumed $\epsilon_{1\parallel}(0)=9$ and the
   missing data above 30 eV. They do not count against the algebra.

## Adjudication

Re-derivation **matches** the implementation for every ledgered element:
uniaxial $\epsilon$ mixing, the plasma energy, the f-sum strengths and closure,
per-direction rescaling, the static-sum target, the $W^{-3}$ tail matched to
$\operatorname{Im}\epsilon$, $\cos^2\alpha_k$ with $q_z=q(Q_-)\simeq W_k/\hbar v$,
the sphere average uniform in $\cos\alpha$, isotropic close collisions, the
PSD kernel, and the orientation-averaged $W_{\rm cb}$ that preserves $I$.
Verdict: `rederived`. Human sign-off remains in #277.
