# `tr-crossing-superposition`

## Claim and source

- Claim: the planar-stack TR vacuum amplitude of a straight charge is the sum over its interface crossings of the stack response to the own-field jump at the crossing velocity, phased $e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})}$. On transported tracks, the per-history coherent sum of those crossing amplitudes (entry, composition-changing layer boundaries, top/bottom exits) gives the TR yield per incident electron; histories add incoherently.
- Code: `montecarlo/spectrum/transition_radiation.py::interface_transition_radiation_amplitude` (helpers `_stack_media`, `_free_system`, `_particular_field`); `montecarlo/spectrum/transition_radiation_crossings.py::{crossing_amplitudes, crossing_transition_radiation, boundary_crossings}`.
- Sources: linearity of the Ginzburg–Frank boundary-value solve, already re-derived under [`tr-detected-omission-bound`](tr-detected-omission-bound.md); Ginzburg & Tsytovich 1990, Ch. 2. Regular-radiator reference: Garibyan, Sov. Phys. JETP 33, 23 (1971); Grichine, Phys. Lett. B 525, 225 (2002), as in the Geant4 XTR Physics Reference Manual.
- Units: Gaussian, $\hbar=c=1$, energies in eV, lengths and times in Å, $e^2=\alpha$. Output in photons eV$^{-1}$ sr$^{-1}$ per incident electron.
- Frozen revision: every source file was read at commit `b93eec8`.

The derivation below was written from the ledger row, the docstrings and the physics page only. The implementation bodies were read afterwards (see "Comparison with the implementation").

## Independent derivation

### Linearity and the crossing decomposition

Use the single-stack solve of the parent claim. The media are indexed $j=0,\dots,L+1$: $j=0$ is the semi-infinite substrate, $j=1,\dots,L$ the layers, and $j=L+1$ the vacuum above, the emission side. Interface $i\in\{0,\dots,L\}$ sits at $z_i$ (increasing) and separates media $i$ and $i+1$. In medium $j$ the total field is the free-wave field plus the particular field $\mathbf P_j(\mathbf v)\,e^{iqz}$, where $\mathbf P_j=(\mathbf E_{p,\perp},\mathbf H_{p,\perp})$ is the tangential own field of an unbounded straight charge in $\varepsilon_j$ and $q=(\omega-\mathbf k_\perp\cdot\mathbf v_\perp)/v_z$. Tangential continuity at $z_i$ then reads

$$
\mathbf F_{i+1}(z_i)-\mathbf F_i(z_i)=-\bigl[\mathbf P_{i+1}(\mathbf v)-\mathbf P_i(\mathbf v)\bigr]\,e^{iqz_i},
$$

with $\mathbf F_j$ the tangential free-wave field. The system $\mathsf M\mathbf x=\mathbf b$ is square and its matrix $\mathsf M$ depends only on $\omega$, $\mathbf k_\perp$, the $\varepsilon_j$ and the $z_i$, not on the source. The right-hand side is a sum of blocks $\mathbf b=\sum_i\mathbf b_i$, each supported on the four rows of one interface. The top-vacuum amplitude $\mathbf a$ is a fixed linear readout of $\mathbf x$, so

$$
\mathbf a=\sum_{i=0}^{L}\mathsf G_i\,\bigl[-\Delta\mathbf P_i(\mathbf v)\bigr]\,e^{iqz_i},
\qquad \Delta\mathbf P_i=\mathbf P_{i+1}-\mathbf P_i ,
$$

where $\mathsf G_i$ ($3\times4$) collects the top-wave columns of $\mathsf M^{-1}$ restricted to the rows of interface $i$, followed by the transversality map $(E_x,E_y)\mapsto(E_x,E_y,E_z)$. Equal media give $\Delta\mathbf P_i=0$ identically: an equal-composition boundary contributes nothing, and dropping it is exact.

### Crossing phase

Repeat the parent's source transform for a charge through an arbitrary point, $\rho=e\,\delta(\mathbf r-\mathbf r_c-\mathbf v(t-t_c))$. With $\rho(\mathbf k,\omega)=\int d^3r\,dt\,e^{-i(\mathbf k\cdot\mathbf r-\omega t)}\rho$,

$$
\rho(\mathbf k,\omega)=2\pi e\,\delta(\omega-\mathbf k\cdot\mathbf v)\,e^{i(\omega t_c-\mathbf k\cdot\mathbf r_c)} .
$$

After the $k_z$ integral the mixed-representation source carries $e^{iqz}\,e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c}-qz_c)}$. Evaluated at the crossing plane $z=z_c=z_i$, the phase factor that multiplies $-\Delta\mathbf P_i$ is

$$
\boxed{e^{iqz_i}\;\longrightarrow\;e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})}}
$$

for any reference point on the line. For the line through the origin at $t=0$, $t_c=z_i/v_z$ and $\mathbf r_{\perp,c}=\mathbf v_\perp z_i/v_z$, so $\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c}=(\omega-\mathbf k_\perp\cdot\mathbf v_\perp)z_i/v_z=qz_i$ exactly. The crossing form does not depend on the absolute $z$ origin. The only common reference left is the far-field origin $(0,0,z_{\rm top})$ at $t=0$ that the top wave is referenced to. The sign conventions are the parent's $e^{i(\mathbf k\cdot\mathbf r-\omega t)}$, so $+\omega t_c$ and $-\mathbf k_\perp\cdot\mathbf r_\perp$ are the correct signs. A charge that reaches the crossing later picks up $e^{+i\omega\Delta t}$, which under $e^{-i\omega t}$ is a delayed wave.

Per crossing $c$ at interface $i_c$ with velocity $\mathbf v_c$ and charge $s_ce$ ($s_c=\pm1$), the amplitude is therefore

$$
\mathbf a_c=s_c\,\mathsf G_{i_c}\bigl[-\Delta\mathbf P_{i_c}(\mathbf v_c)\bigr]\,e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})},
\qquad \mathbf k_\perp=\omega\hat{\mathbf n}_\perp ,
$$

and $\Delta\mathbf P$ is odd in the charge, so the charge sign is an overall factor per crossing. It matters only between crossings of opposite charge in one history (pair products). The speed is $\beta_c=\sqrt{1-\gamma_c^{-2}}$ with $\gamma_c=1+T_c/(m_ec^2)$, $m_ec^2=510.99895$ keV, so that $\mathbf v_c=\beta_c\hat{\mathbf u}_c$.

### Exit face and entrance face

Slab frame: the stack fills $0\le z\le Z$ with planes $z_0=0<z_1<\dots<z_N=Z$, $N$ = number of transport layers, $+\hat{\mathbf z}$ from the entrance face into the stack, and vacuum on both sides.

- **Exit face**, $n_z>0$: the solver frame is the slab frame. The substrate is the entrance-side vacuum, the layers run in slab order, and solver interface $i$ is slab plane $i$. The jump at plane $i$ is $\mathbf P_{\rm after}-\mathbf P_{\rm before}$ for a charge moving with $v_z>0$. For $v_z<0$ (a backscattered charge crossing upwards) the field jump is still $\mathbf P_{i+1}-\mathbf P_i$ in the solver's fixed media indexing, not "after minus before". The particular field, with its $1/\lvert v_z\rvert$, is valid for either sign of $v_z$.
- **Entrance face**, $n_z<0$: rotate by $\pi$ about $\hat{\mathbf x}$, $\mathsf R=\operatorname{diag}(1,-1,-1)$. Then $\hat{\mathbf n}'=\mathsf R\hat{\mathbf n}$ has $n'_z>0$, the entrance vacuum becomes the emission side, the exit vacuum becomes the substrate and the layers appear in reverse order. Slab medium $m$ (with $m=0$ the entrance vacuum, $m=1..N$ the layers and $m=N+1$ the exit vacuum) becomes solver medium $N+1-m$, so solver interface $i$ (between solver media $i$, $i+1$, that is slab media $N+1-i$, $N-i$) is slab plane $N-i$. $\mathsf R$ is proper, so $\mathbf E$ and the axial $\mathbf H$ both transform as $\mathsf R$. The inputs are $\mathbf v'_c=\mathsf R\mathbf v_c$, $\mathbf r'_c=\mathsf R\mathbf r_c$ and $t'_c=t_c$, and the output vector returns as $\mathbf a=\mathsf R^{-1}\mathbf a'=\mathsf R\mathbf a'$. The phase is invariant, $\mathbf k'_\perp\cdot\mathbf r'_\perp=\omega(n_xx+n_yy)=\mathbf k_\perp\cdot\mathbf r_\perp$. The layer thicknesses are unchanged, and $\lvert\mathbf a\rvert^2$ and $n_z^2$ are rotation invariant. The far-field origin is the transverse origin of the entrance plane.

Both faces use the same reference within a history, and every crossing amplitude is evaluated for one emission direction, so relative phases inside a history are frame independent.

### History sum and normalization

The parent's far-field factor gives, per history $h$ and per incident electron,

$$
\frac{d^2N}{dE\,d\Omega}=\frac{E\,n_z^2}{16\pi^4N_e}\sum_{h}\Bigl\lvert\sum_{c\in h}\mathbf a_c\Bigr\rvert^2
\qquad(\texttt{coherence="history"}),
$$

with $N_e$ the incident count, including histories with no crossing (in practice every primary enters). The sum runs over all tracks of the primary, secondaries included, with their charge signs. The dephased limit replaces $\lvert\sum_c\mathbf a_c\rvert^2$ by $\sum_c\lvert\mathbf a_c\rvert^2$. Any chunking of the crossing list must not split a history across two chunks unless the partial field sums are carried. Equivalently, if per-crossing cumulative sums are differenced at history ends, the result is exact for any chunk boundary only when the running sum continues across chunks or the chunks are history aligned.

### Limits

1. **Straight line, both senses**: with all $\mathbf v_c=\mathbf v$ and the crossings on the line, the sum equals the full-solve vector $\mathbf a$ term by term (above).
2. **Equal media**: $\Delta\mathbf P\equiv0$.
3. **Thickness average**: for a transparent film the cross term between the entry and exit amplitudes carries $e^{i(q-\lambda_f)d}$ times a Fabry–Perot series in $e^{2i\lambda_fd}$. Its average over $d$ vanishes unless $q$ is an odd multiple of $\lambda_f$, so $\langle\lvert\mathbf a_0+\mathbf a_1\rvert^2\rangle_d=\langle\lvert\mathbf a_0\rvert^2\rangle_d+\langle\lvert\mathbf a_1\rvert^2\rangle_d$.
4. **Substep invariance**: substeps do not cross a plane and add no crossings. The only substep dependence is the transported crossing time and point, which is quadrature-level.

### Regular-radiator reference

Take $N$ foils ($\varepsilon_1=1-\xi_1^2$, thickness $t_1$) separated by gaps ($\varepsilon_2=1-\xi_2^2$, $t_2$) at normal incidence, with $\gamma\gg1$, $\theta\ll1$ and $\xi_j\ll1$. Fresnel reflection is $O(\xi^4)$ and is neglected, and each interface radiates the Jackson 13.91 amplitude. Entry into a foil has amplitude $\propto-(1/A_2-1/A_1)$ and exit $\propto+(1/A_2-1/A_1)$, with $A_j=\gamma^{-2}+\theta^2+\xi_j^2$. Relative to its entry, the exit picks up the formation phase $\phi_1=(q-\lambda_1)t_1\simeq\omega t_1A_1/2$, and one period adds $\phi=\phi_1+\phi_2$. Summing the geometric series,

$$
\Bigl\lvert\sum_{k=0}^{N-1}(1-e^{i\phi_1})e^{ik\phi}\Bigr\rvert^2=4\sin^2\frac{\phi_1}{2}\,\frac{\sin^2(N\phi/2)}{\sin^2(\phi/2)} ,
$$

so with the single-interface density $\frac{\alpha}{\pi^2E}\theta^2(1/A_2-1/A_1)^2$ (parent, Jackson limit),

$$
\frac{d^2N}{dE\,d\Omega}=\frac{\alpha}{\pi^2E}\,\theta^2\Bigl(\frac1{A_2}-\frac1{A_1}\Bigr)^2\,4\sin^2\frac{\phi_1}2\,\frac{\sin^2(N\phi/2)}{\sin^2(\phi/2)} .
$$

Grichine's $\langle R^{(N)}\rangle=N\frac{(1-F_1)(1-F_2)}{1-F}+\frac{(1-F_1)^2F_2(1-F^N)}{(1-F)^2}$ with $F_j=e^{-i\phi_j}$ satisfies $2\operatorname{Re}\langle R^{(N)}\rangle=$ the same stack factor for real $\phi_j$ (checked numerically below). The ledger form $\frac{2\alpha}{\pi^2E}\theta^2(\dots)^2\operatorname{Re}\langle R^{(N)}\rangle$ therefore agrees with this derivation. It is also the Geant4 per-$d\theta^2$ form divided by $\pi$ ($d\Omega\simeq\pi\,d\theta^2$ after the azimuth). $N=1$ gives $2(1-\cos\phi_1)$ times one interface, the "twice on average, four times at peak" film limit.

## Comparison with the implementation

| Item | Independent | Code | Result |
| --- | --- | --- | --- |
| media indexing | substrate $j=0$, layers, vacuum last; interface $i$ between $i$, $i+1$ | `_stack_media`: substrate, layers, then `ones`; `_free_system` rows `4*i` couple media `i+1` (+) and `i` (−) | matches |
| jump | $-(\mathbf P_{i+1}-\mathbf P_i)$, fixed indexing for either sign of $v_z$ | `jump` = $-\mathbf P_{\rm face+1}+\mathbf P_{\rm face}$, the same sign pattern as the full-solve `rhs` | matches |
| $\mathsf G_i$ | top-wave rows of $\mathsf M^{-1}$ at the four rows of interface $i$ | `solve(M^T, select)`, reshaped `(B, size/4, 4, 2)`, indexed by `face` | matches |
| phase | $e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})}$, no $z$ | `exp(1j*(omega*t - k_perp·r_perp)/HBARC_EV_ANG)` | matches |
| $E_z$ | $-\mathbf k_\perp\cdot\mathbf E_\perp/\lambda_{\rm vac}$ | `-sum(k_perp*e_t)/lam[-1]` | matches |
| exit face | identity frame, interface $i$ | `flip = ones`, `face = interface` | matches |
| entrance face | $\mathsf R=\operatorname{diag}(1,-1,-1)$ on $\hat{\mathbf n}$, $\mathbf v$, $\mathbf r$; stack reversed; interface $N-i$; $\mathbf a=\mathsf R\mathbf a'$ | `_FLIP`, `layers[::-1]`, `thickness.size - interface`, `amplitude * (charge * flip)` | matches |
| charge | overall $s_c$ per crossing | `charge[:, None] * flip`; positron for `launch_kind == 3`, else $-1$ | matches |
| $\beta$ | $\sqrt{1-\gamma^{-2}}$, $\gamma=1+T/m_e$ | `gamma = 1 + T*1e3/M_E_EV`, `M_E_EV = 510998.95` | matches |
| crossing state | entry: row start, $T_{\rm start}$, $t_0+t_{\rm start}$; faces: row end, $T_{\rm end}$, $t_0+t_{\rm end}$ | identical; $z$ snapped to the plane | matches |
| secondary clock | continues the primary's | launch `t_ang` = parent row `t_end_ang`, `t0` = parent's (`secondaries.py`) | matches |
| history sum | $\sum_h\lvert\sum_{c\in h}\mathbf a_c\rvert^2$ | stable sort on `electron_id`; history-aligned chunks; `cumsum` differenced at history ends | matches |
| normalization | $E\,n_z^2/(16\pi^4N_e)$ | `omega_nz2 * total / (16 pi**4 * n_electrons)` | matches |
| chunk size | $\le$ `max_elements` per docstring | can exceed it by up to one history (see findings) | differs (memory bound only) |

On chunking: `hi = searchsorted(starts, first + per_chunk, side="right")` includes every history that *starts* at or before `first + per_chunk`, so chunks always end at a history boundary. The `cumsum` therefore never spans a chunk boundary and needs no carry. `ends` holds the chunk-local last index of each history, and `summed[..., 1:, :] -= cumulative[..., ends[:-1], :]` is the exact per-history difference.

## Numeric checks (own code, no module helpers in the reference)

I wrote an independent slab-frame vector solver. It keeps vacuum on both sides of the stack and reads the entrance-face ($-z$) and exit-face ($+z$) vacuum waves directly. It therefore uses no rotation, so it tests the module's $\pi$-rotation logic instead of repeating it. It has its own unknown layout, the particular field derived in the parent write-up, $\alpha=1/137.035999084$ and $\hbar c=1973.269804$ eV Å. The module's `ALPHA_FS = 1/137.035999` accounts for the constant $6.1\times10^{-10}$ yield offset ($3.1\times10^{-10}$ in amplitude) below.

- **Own-solver sanity (entrance face, no rotation)**: a charge entering a thick conductor ($\varepsilon=-10^9+10^9i$) at four random oblique velocities and backward directions matches the image-charge formula to $\le5.7\times10^{-5}$ ($O(\lvert\varepsilon\rvert^{-1/2})$).
- **Straight line, crossing phase**: twelve random absorbing stacks (1–4 layers, $\beta=0.2$–$0.97$, both senses of $v_z$, both charges, oblique), with the line through a random point $(\mathbf r_0,t_0)$. The sum of my per-crossing solves, each with phase $e^{i(\omega t_c-\mathbf k_\perp\cdot\mathbf r_{\perp,c})}$, equals my full solve translated by $e^{-i(\mathbf k_\perp\cdot\mathbf r_{\perp,s}+qz_s)}$ to $6.1\times10^{-12}$ of the peak on both faces. Here $\mathbf r_s=\mathbf r_0-\mathbf vt_0$.
- **Module against own solver, straight line**: `crossing_amplitudes(...).sum(-2)` agrees with my sum to $3.07\times10^{-10}$ in amplitude, and `crossing_transition_radiation` agrees to $6.13\times10^{-10}$ in yield, on both faces. This covers the entrance-face rotation, the reversed stack, the $N-i$ interface map and the back-rotated vector, all against my unrotated solve.
- **Transported tracks**: 40 primaries at 20 keV through Si/W/W/Si (800/800/800/600 Å). I extracted crossings from the raw rows independently (entry at the first-row start of each primary, ends of `LAYER_BOUNDARY`/`EXIT_TOP`/`EXIT_BOTTOM` rows, W/W boundary dropped). The extraction reproduces all 142 `boundary_crossings` records exactly (interface, history, time, kinetic energy, transverse point), with 52 matched boundaries dropped. The rows were relabelled into 20 two-track histories, with the second track a "secondary" and a quarter of the tracks positrons. My history-coherent yield, using my solver, $\beta$ from the recorded kinetic energy and my charge signs, then matches `crossing_transition_radiation` to $6.13\times10^{-10}$. That holds at 150 and 600 eV, $\theta=0.3$ and $1.0$, on both faces. The dephased mode matches $\sum_c\lvert\mathbf a_c\rvert^2$ to the same digit. The coherent/dephased ratios are 0.63–1.13, so the comparison is sensitive to the relative phases. (A genuine shell-secondary cascade could not run: the SBETHE stopping tables are not installed.)
- **Chunking and ordering**: a random permutation of the records, with `max_elements` = 1, 6, 13 and $2^{22}$ and a $3\times2$ shape $s$, reproduces the reference yield to $3.3\times10^{-16}$. With `max_elements` = 72 (`per_chunk` = 12), the chunks hold 14, 16, 14, 14, 19, … crossings.
- **Thickness average** (own solver, 1500 thicknesses from 0.2 to 40 µm, oblique track): $\langle\text{coherent}\rangle/\langle\text{dephased}\rangle$ = 0.9999 (transparent, exit face), 1.0071 ($\varepsilon=0.85$, entrance face, where Fresnel reflection is not small) and 0.9991 (weakly absorbing). The coherent maximum reaches $2.0$–$2.6\times$ the mean.
- **Equal media**: the jump amplitude is exactly 0 in my solver. The module test `test_charge_sign_and_matched_interfaces` asserts the same.
- **Regular radiator**: $2\operatorname{Re}\langle R^{(N)}\rangle$ equals the geometric-series factor to $5\times10^{-11}$ for $N=1,3,10$ at 1000 random real phase pairs, so the test's Grichine expression is algebraically my stack factor. The anchor geometry is 10 foils of 15 µm, $\hbar\omega_p=20.9$ eV, 200 µm vacuum gaps, $\gamma=2000$, 2–40 keV and $\gamma\theta\le3$. On it the module agrees with my small-angle reference to $1.07\times10^{-4}$ of the peak (gate $2\times10^{-4}$; worst point 3.3 keV, $\gamma\theta=1.49$). It agrees with the exact-phase stack factor to $3.0\times10^{-6}$ relative above $10^{-3}$ of the peak (gate $10^{-5}$), and with my full 19-layer vector solve (with Fresnel reflection) to $1.6\times10^{-8}$ at five points. That last figure is rounding in the near-cancelling own-field jump at $\gamma=2000$.
- **Repository tests**: `tests/montecarlo/test_transition_radiation_crossings.py` and `tests/montecarlo/test_transition_radiation.py`: 64 passed. The tracked `src/` and `tests/` were byte-identical to `b93eec8` during the runs.

### Anchor assessment

The anchor's reference is sound. Its normalization follows independently from the parent's Jackson-limit single-interface density times the coherent foil-sum factor ($N=1$ gives the film's $2(1-\cos\phi_1)$). Its stack factor is algebraically the transparent geometric series. The $2\times10^{-4}$-of-peak gate covers the small-angle-phase error with a $1.9\times$ margin. The $10^{-5}$ exact-phase gate is the sharper check, with a $3.4\times$ margin. Its remaining error comes from the small-angle interface amplitude and the neglected Fresnel reflection, both $O(\xi^2,\theta^2)$ suppressed at these energies. The citation text (Garibyan's JETP paper, Grichine's PLB paper, the Geant4 manual) was not available to this verifier, so the published normalization was confirmed by derivation rather than by transcription. The anchor covers only the exit face of a straight, normally incident, transparent stack. It is insensitive to the entrance-face rotation, the charge sign and history grouping. Those are covered by the consistency tests in the same file, which compare against the parent's `rederived` planar solve, and by the independent solver checks above.

### Physics assumption

- **Locally straight passage.** The crossing term is exact when the trajectory is straight, at constant speed, over the formation length $Z_j\sim1/\lvert q-\lambda_j\rvert$ on both sides of the crossing. A kink inside $Z_j$ couples the interface and bremsstrahlung terms (multiple-scattering suppression of TR in the Pafomov/Garibyan sense), and the per-crossing jump with $\mathbf P(\mathbf v_c)$ is then not separately defined. For tens-of-keV electrons at keV photons, $Z\simeq\hbar c/[E(1/\beta-\cos\theta)]$ is 0.1–1 Å, below the elastic mean free path. At EUV energies it approaches the mean free path, and for $\gamma\gg1$ it requires the multiple-scattering angle accumulated over $Z_j$ to stay well below $\max(\gamma^{-1},\theta,\xi)$. The code does not check this, and the ledger's Notes state the approximation.
- **Histories incoherent.** This is correct for the per-electron yield of uncorrelated arrivals. Bunch coherence (CTR) is explicitly outside the claim.
- **Secondaries coherent with their primary.** One history is one realization of the classical current, and its crossing phases are deterministic given the transported clock. The coherent sum is therefore the consistent classical model, and the secondary clocks do continue the primary's. Birth radiation of secondaries, the stopping of absorbed tracks and positron annihilation are not crossing terms. They belong to bremsstrahlung or other channels and are correctly not included.

## Findings

1. **Low, documentation/resource contract.** `crossing_transition_radiation` docstring: "`max_elements` bounds one chunk's `len(s) * crossings`". Chunks include every history that starts at or before `first + per_chunk`, so a chunk can exceed the bound by up to one history (19 crossings against `per_chunk` = 12 above), and a single history longer than `per_chunk` always does. Results are exact for every chunking. Suggested fix (owner): describe it as a target that never splits a history, or use the largest start not exceeding the limit.
2. **Low, traceability.** `boundary_crossings` defines the crossing set the claim names: entry, composition-changing boundaries, top/bottom exits, the equal-media drop, the interface index, and the kinetic-energy and time choices. It has no `Validation: tr-crossing-superposition` marker and is not in the row's `Code` field. I verified it independently (exact record agreement above). Suggested fix (owner): add it to `Code` and give it the marker.
3. **Info, scope.** The locally-straight-passage condition is not checked at runtime. Its validity window is stated above and belongs in the physics page's assumptions alongside the existing note. A real shell-secondary cascade was not run here (no SBETHE tables), so multi-track grouping was tested on relabelled transport rows. The CuPy path was not exercised.
4. **Info, anchor coverage.** The published-formula anchor exercises only the exit face of a straight, transparent, normally incident stack. Backward emission, oblique incidence, absorption and transported tracks rest on consistency with the `rederived` planar solve. That is adequate for `anchored` on this row, but the row's Anchor text should keep "other tests in the file check consistency with the exact solve".
5. **Low, docs.** This write-up is not yet in the `docs/validation/evidence-index.md` toctree (task owner).

No factor, sign, exponent, unit, index or convention diverges.

## Verdict

The crossing decomposition is reproduced term by term from the linearity of the parent's solve. That covers the per-crossing phase (identical to $e^{iqz_i}$ on a straight line, and correct for any line origin), the jump sign and media indexing, the entrance-face rotation with its reversed stack, $N-i$ map and back-rotation, the charge sign, $\beta$ from the crossing kinetic energy, history grouping with history-aligned chunks and `cumsum` differencing, and the normalization $E\,n_z^2\lvert\sum\mathbf a\rvert^2/(16\pi^4N_e)$. An independent unrotated solver agrees with the module to the $\alpha$-literal difference on both faces, for straight lines and for transported multi-track mixed-charge histories. The regular-radiator anchor's reference and tolerances are sound.

Verdict: `rederived`. Findings 1–2 are owner fixes that do not change any computed number. Only a human may mark the row `signed-off`.
