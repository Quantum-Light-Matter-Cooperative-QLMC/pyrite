# CUDA axial MCWF backend verification

Fresh verifier, no implementation authorship. Sources read first: ledger rows cr-axial-realspace-propagation/dissipator; their derivation documents; parent cr-lindblad-jump-operators. Original Strang/Kirkland/DCM papers were not independently retrieved. Scope is CUDA port of the recorded model, not renewed validation of material/GOS input tables.

## Independent derivation recorded before implementation inspection

With $k_0=p/\hbar$, $E=\gamma mc^2$, and $\hbar_v=\hbar c\beta$, expanding longitudinal dispersion gives

$$
i\hbar_v\,\partial_z\psi
=\left[-\frac{\hbar^2}{2\gamma m}\nabla_\perp^2+U-iW\right]\psi.
$$

Factoring $\psi=\exp(i\mathbf{k}_t\cdot\mathbf{r})\phi$, with $\phi$ periodic, replaces Fourier $\mathbf{k}$ by $\mathbf{k}+\mathbf{k}_t$. For interval $\Delta z$ the kinetic half-step is

$$
K_{1/2}=\exp\left[-\frac{i\Delta z\,\hbar^2\lvert\mathbf{k}+\mathbf{k}_t\rvert^2}{4\gamma m\hbar_v}\right],
$$

and the real-space factor is

$$
V=\exp\left[-\frac{i\Delta z\,U}{\hbar_v}-\frac{\Delta z\,C_{\mathrm{tot}}}{2}\right],
\qquad W=\frac{\hbar_v C_{\mathrm{tot}}}{2}.
$$

Physical discrete norm is $\Delta A\sum_j\lvert\phi_j\rvert^2$. Thus no loss conserves norm; uniform loss gives $\exp(-C_{\mathrm{tot}}z)$, exact regardless of kinetic phase. Dimensions: $\mathbf{k}$ in inverse angstrom, $U$ in eV, $\Delta z$ in angstrom; exponents dimensionless. Positive $C$ must decay norm. Electronic carrier kick must use plus $\mathbf{q}$, from $\exp(i\mathbf{q}\cdot\mathbf{r})\psi$.

For independent displaced atoms the second cumulant of $\exp(i\varphi)$ is covariance. Therefore TDS has $L_{\mathbf{u}}=\varphi_{\mathbf{u}}-\mathbb{E}\varphi$ with measure $\lambda p(\mathbf{u})\,d^2u$, Gaussian $\mathbf{u}$ with variance $a^2I$. For FFT synthesis $+i\mathbf{k}\cdot\mathbf{r}$,

$$
\varphi_{\mathbf{u}}(\mathbf{r})
=\sum_{\mathbf{k}}S_{\mathbf{k}}
\exp[i\mathbf{k}\cdot(\mathbf{r}-\mathbf{R})]
\exp(-i\mathbf{k}\cdot\mathbf{u}),
$$

and $\mathbb{E}\varphi$ uses $\exp(-a^2k^2/2)$. The mean subtraction is a spatial function, required coherently within EACH sampled jump. Random Fourier-coefficient resampling would change the kernel. The sum $\sum\lambda\mathbb{E}L^2=C_{\mathrm{grid}}$, and the numerical sink $\max(C_{\mathrm{full}}-C_{\mathrm{grid}},0)$ makes total TDS loss $\max(C_{\mathrm{full}},C_{\mathrm{grid}})$. A sink kills the trajectory, no physical kick or survivor renormalization contribution.

Electronic radial quadrature $c_q$ gives loss $\sum_qc_q\rho_q$ and selection mass $c_q\int dA\,\lvert\phi\rvert^2\rho_q$; sampled azimuth is uniform on $[0,2\pi)$. The jump is $\phi\mapsto\sqrt{\rho_q}\,\phi$ and $\mathbf{k}_t\mapsto\mathbf{k}_t+q(\cos\alpha,\sin\alpha)$, then normalize. For homogeneous density this is a compound Poisson process:

$$
\mathbb{E}\exp(ibk_x)
=\exp\left[z\rho\sum_qc_q\{J_0(bq)-1\}\right].
$$

For no-jump unnormalized evolution, $S(z)=\|\phi\|^2$ and

$$
\frac{dS}{dz}=-\int dA\,\lvert\phi\rvert^2C_{\mathrm{tot}}.
$$

Sample independent $\xi$ uniformly on $(0,1)$, trigger first $S\le\xi$, select mechanism proportional to its rate expectation at normalized $\phi$, apply normalized jump, reset $\xi$. Endpoint detection resolves jump depth to $\Delta z$ and permits at most one per step; this is a shared time discretization, not continuous-time exactness. Per-electron RNG identity must depend on seed and full uint64 electron key, and per-electron draw counters only; batching/removal/reordering must not perturb others. CPU historical NumPy Philox streams may remain distinct from a CUDA Philox4x32-10 scheme. Counter-based Philox round derives two $32\times32$ products and crosswise XOR/high-word mixing, with Weyl key advance each round; low and high halves of electron identity must be preserved.

Filters: units pass, zero-rate/unitarity and uniform-loss limits pass at expression level, covariance positivity and positive carrier-kick sign pass. Independent derivation complete before source bodies.

## Source comparison and numerical checks

Reviewed CUDA owners `_axial_cuda.py`, `_axial_cuda_jumps.py`, `_axial_cuda_dissipator.py`, `_axial_cuda_rng.py` and CPU `axial_mcwf.py` against the preceding expressions. Current worktree includes post-e0fcbd56 optimizations, checked as present 2026-10-21.

`_Batch` stores the following lagged spectrum:

```text
lagged=FFT(phi)/half
```

One step changes it to:

```text
FFT(V IFFT(half² lagged))
```

Reconstruction uses:

```text
IFFT(half lagged)
```

This is therefore exactly $K_{1/2}VK_{1/2}$. Norm $\Delta A\sum_j\lvert\mathrm{lagged}_j\rvert^2/N$ follows Parseval and the half-step is unit modulus. This validates phase order and FFT normalization without treating CPU parity as a derivation.

- `DeviceDissipator.tds_phase` uses shifted spectrum times $\exp(-i\mathbf{k}\cdot\mathbf{u})-\exp(-a^2k^2/2)$ before inverse FFT and $N$ multiplication. This is the coherent random-displacement operator, not independently sampled modes. The twice-grid convolution, density-overlap extension, mean cross term and positive annular slack match the finite trigonometric expansion. Per-source rates and per-mechanism rates are uploaded unchanged; omitted $\Delta A$ in source selection is common to all sources and cancels.
- Cached immutable spectra/envelope geometry/electronic nonzero indices leave the equations unchanged. Electronic expectations include the density spectrum's compensating $N$; smoothed density uses the inverse FFT of `electron_density*damp` multiplied by $N^2$, consistent with stored electron harmonics divided by $N$. Electronic kick is plus $\mathbf{q}$ and rebuilds the kinetic phase. Sink returns None, removes the incident contribution, and has its own event depth/count.
- Fused acceptance scans proposals in order and rejects a violated envelope loudly. A newly identified stride defect was reported and fixed by CUDA author before this review completed: `.real` gives a strided view, so the raw kernel must receive `cp.ascontiguousarray(weights)`; that call is present in the inspected `_accepted`. Latest hardware confirmation still pending at time of this paragraph.
- Single-entry fast path advances each electron draw counter to 1, preserving the address consumed by its degenerate categorical selection. Per-electron keys use all 64 mixed bits, draws use successive per-electron counters; changing batch membership or deaths does not reassign addresses. This is deliberately Philox4x32-10 rather than NumPy's Philox4x64 CPU stream.

Independent RNG oracle `/tmp/axial_philox_oracle.py` uses Python big-integer arithmetic and three published [Random123 known-answer vectors](https://raw.githubusercontent.com/DEShawResearch/random123/main/tests/kat_vectors). All three vectors agree. It extracts the actual raw CUDA kernel body, compiles its scalar operations for host C++ execution, and compares 80 scalar doubles bitwise to the independent oracle over keys 0, 1, $2^{32}+1$, $2^{64}-1$ and first draw offsets 0, 1, 2, $2^{33}$, $2^{64}-4$. All agree. This checks algorithm/arithmetic and full key/counter use; host compilation is explicitly not CUDA execution.

Independent historical CPU comparison `/tmp/axial_cpu_historical_check.py` loads main's historical axial class separately and compares with current default CPU at 12², Si $\langle001\rangle$, 1 MeV, 24 electrons, seed 17, $\Delta z=10$ Å, depths 0/1000/2000 Å. Depths, density observables, survival and sink depths agree bitwise; jump counts exactly tds=9, sink=6, electronic=29. This is a bounded regression sample, not a proof for every seed/input. Inspection finds CPU RNG creation/draw order unchanged.

Local targeted check: `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_channeling_axial_cuda.py -k 'backend_and_callback_refusals or cpu_jump_history or admission' --durations=0`: 3 passed, 5 deselected, 2.74 s. No local GPU or heavy jobs run.

Read actual worker-supplied `/tmp/issue620-cuda-1623.log`: 5 GPU tests pass in 4.74 s on NVIDIA GeForce RTX 5080, CuPy 14.2.0, CUDA runtime 13020. Tests cover device jump formulas, deterministic propagation, per-electron RNG vector/full-key/partition (verify exact staged test version before attributing individual names), and batch/recording stochastic parity. Log benchmark reports 48²/128 electrons/100 steps seed17 warm speedup 1.43822566349, below the predeclared 2x target. This log predates the latest raw-kernel acceptance optimization; it is evidence for that earlier source state only. Its test collection path is `checks/axial_cuda_tests.py`, so final source provenance must be supplied by the owner. No claim of personally running those GPU tests.

## Verifier output

- **Claim**: `cr-axial-realspace-propagation` — `_axial_cuda_jumps.py::DeviceJumps._half_kinetic`, `_axial_cuda.py::_Batch` — Strang 1968/Kirkland section 6, paraxial $K_{1/2}VK_{1/2}$ equation.
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`.
- **Re-derivation**: `matches` — kinetic plus-carrier symbol, half-step coefficient, potential decay sign and physical FFT norm all match.
- **Verdict**: `rederived` for CUDA equation port; retain existing base `anchored` scope.
- **Write-up**: `/tmp/cr-axial-cuda-independent-review.md`; owner integrates relevant text into `docs/validation/channeling/cr-axial-realspace-propagation.md`.
- **Suggested ledger change**: add CUDA owners and exact real-hardware anchor provenance only after final corrected source receives passing device tests; retain existing Airy/Gouy anchors and all paraxial/step-size limitations. Never signed-off.

- **Claim**: `cr-axial-realspace-dissipator` — `_axial_cuda_jumps.py::DeviceJumps`, `_axial_cuda_dissipator.py::DeviceDissipator`, `_axial_cuda.py::_Batch`, `_axial_cuda_rng.py::ElectronRNG` — DCM 1992 MCWF; parent Lindblad phase covariance and electronic LDA kernel.
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`.
- **Re-derivation**: `matches` — coherent mean-subtracted displacement, node-weighted localization/carrier kick, numerical sink, no-jump total rate, ordered rejection and full-key per-electron Philox agree. Latest acceptance-stride correction is symbolically correct, pending final GPU run.
- **Verdict**: `rederived`; no remaining equation discrepancy identified in inspected source.
- **Write-up**: `/tmp/cr-axial-cuda-independent-review.md`; owner integrates relevant text into `docs/validation/channeling/cr-axial-realspace-dissipator.md`.
- **Suggested ledger change**: retain `rederived` until final corrected device anchors pass and are named; then owner may promote covered CUDA unravelling scope to `anchored`. Preserve inhomogeneous-electronic per-jump coverage limitation and finite-step/8-sigma displacement cutoff. No original-paper re-retrieval or material/GOS revalidation in this review. Never signed-off.

No canonical docs were edited by this verifier, so docs build/render checks belong to the integrating owner's final artifact. This formatted scratch report supplies MyST math for canonical integration.

## Final candidate re-review: c6fb07ebf942030d6dab435ce50ee6d8edf09cf2

This section supersedes the earlier pending-final-hardware language. Inspected exact candidate via `git show c6fb07eb`, including new fused `_PROPOSE`, final `_accepted`, live-count control, cached phase/electronic data and stacked mechanism-rate reduction. No new derivation was taken from implementation: compare to the pre-body expressions recorded above.

The radial Gaussian probability in dimensionless annulus $[a,b]$ is $P=e^{-a^2/2}-e^{-b^2/2}$. Proposal annulus $i$ must have mass proportional to $P_i M_i$. Conditional sampling may use $Y=e^{-b^2/2}+U(e^{-a^2/2}-e^{-b^2/2})$, hence $r=\sigma\sqrt{-2\log Y}$. In `_PROPOSE`, the following literal calculation is exactly this inverse CDF:

```text
lo=decay[shell]
hi=decay[shell+1]
radius=sigma*sqrt(-2*log(hi+(lo-hi)*draws[count+i]))
```
 The first draw row selects cumulative annulus mass with strict `cumulative>target` (right-search convention); the third row gives $2\pi U$ uniform azimuth; the fourth row remains reserved for acceptance. `offsets[2*i], offsets[2*i+1]` are contiguous Cartesian samples. The wrapper receives a contiguous `(4,count)` buffer directly from per-electron RNG; no hidden draw, key truncation, independence change or per-mode TDS randomization is introduced. Fusing `cos/sin/log` can change only floating rounding, which the declared NumPy oracle tolerance covers. All cached probability/phase factors depend on immutable source geometry, not trajectory density.

Final `_accepted` passes contiguous float64 weights to the scalar raw kernel, so the earlier stride defect is resolved. It preserves first accepted proposal order, checks the envelope, and consumes the same proposal draw block. Host `live_count` is initialized from the device live mask and decremented exactly when `kept is None`; it therefore preserves the while-live termination and event order. Full bool-mask transfer plus host `flatnonzero` enumerates the same ascending electron indices as device `flatnonzero`. Stacked rate reduction preserves all three mechanism expectations and categorical ordering. Host electronic-q lookup uses the same immutable node value, so the plus-carrier kick is unchanged.

Read final `/tmp/issue620-cuda-1629.log` directly: 10 tests passed, 0 failed, on RTX 5080/CuPy14.2/CUDA13020. The staged collection includes the new direct NumPy inverse-CDF oracle, 8-seed replicated survival/population/jump-count comparison, stride regression, phase/weight/diagonal device parity, deterministic propagation and published-vector/full-key/counter-partition test. Read the maintained inverse-CDF test: 64 independent draw columns with explicit all-zero and nextafter-one endpoint columns; shell choices exact, displacement tolerance `rtol=1e-13`/`atol=1e-14`. This is an external array-computation oracle for the fused raw kernel, rather than calling the same kernel as its reference.

Read canonical raw evidence `docs/validation/check-records/axial-cuda-performance.json`. Independently checked all seven final-source SHA256 entries against `git show c6fb07eb:<path>` and all six overlapping remote-owner hash entries against those final entries. Also checked SHA256 of the final local log against its recorded final-stage hash. All agree. Remote test collection is a staged copy under `checks/axial_cuda_tests.py`; source manifest supplies the owning-file provenance. The log was produced by the CUDA owner, not personally executed by this verifier.

Unchanged declared 48²/128-electron/100-step seed17 representative warm benchmark now has CPU median 0.409848956158 s, CUDA median 0.195855707163 s, ratio 2.09260665464. Thus the predeclared $\ge2\times$ target is met on this device/workload. The 12²/8-electron workload remains slower on CUDA; the result does not establish universal acceleration or whole-device peak memory.

### Final verifier output and ledger recommendation

- **Claim**: `cr-axial-realspace-propagation` — `DeviceJumps._half_kinetic`, CUDA `_Batch` — paraxial split-step equation, Strang 1968/Kirkland section 6 as ledgered.
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`.
- **Re-derivation**: `matches` — no divergent factor/sign/FFT normalization; final optimizations preserve $K_{1/2}VK_{1/2}$.
- **Verdict**: `rederived`.
- **Write-up**: `/tmp/cr-axial-cuda-independent-review.md`, for owner integration into the ledgered propagation document.
- **Suggested ledger change**: maintain `anchored` and extend its expressly scoped CUDA port coverage with `test_cuda_deterministic_propagation_and_device_callbacks` and final hardware record c6fb07eb/job1629. Existing CPU Airy/Gouy/Strang anchors remain the physical references. Preserve finite-step, high-tilt and paraxial restrictions. No signed-off.

- **Claim**: `cr-axial-realspace-dissipator` — CUDA `DeviceDissipator`, `DeviceJumps`, `_Batch`, `ElectronRNG` — coherent phase-covariance unravelling and DCM waiting-time MCWF, electronic LDA kicks and separate numerical sink.
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`.
- **Re-derivation**: `matches` — final fused inverse CDF, weighted shell selection, contiguous ordered acceptance, unchanged full-key RNG, localization/kick and sink accounting match the pre-body independent expressions. No unresolved implementation discrepancy found in the reviewed scope.
- **Verdict**: `rederived`.
- **Write-up**: `/tmp/cr-axial-cuda-independent-review.md`, for owner integration into the ledgered dissipator document.
- **Suggested ledger change**: promote covered claim to `anchored`, linking existing direct TDS master-equation and homogeneous-electronic Lindblad regression anchors plus exact-candidate device operator/inverse-CDF/stride/RNG/replicated-statistics anchors from job1629. The lead reports the integrated CPU covariance/diagonal/master-equation and homogeneous-electronic tests passed; this verifier did not personally rerun that integrated suite. Owner must retain the concrete test evidence when applying the row. Coverage remains restricted: inhomogeneous electronic evolution has per-jump/device-parity checks and CPU/CUDA ensemble consistency, not a new independent full inhomogeneous-electronic master-equation oracle; numerical sink/clamp, correlated supercell-image, finite-step and 8-sigma cutoff limitations persist. Never signed-off.
