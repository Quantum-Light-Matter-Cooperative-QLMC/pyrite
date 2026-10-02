# Transport JIT-kernel walkthrough

This reference follows the arithmetic and control flow of the two CUDA transport kernels in [`transport/_jit_kernel.py`](../../../src/pyrite/montecarlo/transport/_jit_kernel.py) and [`transport/_jit_lut_kernel.py`](../../../src/pyrite/montecarlo/transport/_jit_lut_kernel.py), whose device-side constants and helpers live in [`transport/_jit_device.py`](../../../src/pyrite/montecarlo/transport/_jit_device.py) and whose host launchers live in [`transport/_jit_launch.py`](../../../src/pyrite/montecarlo/transport/_jit_launch.py):

- `_transport_kernel` evaluates elastic rates, stopping power, and optional Urban energy-loss straggling directly, and carries the opt-in shell soft/hard inelastic mode (helpers in `_jit_shell_device.py`; see [shell soft/hard transport](../../physics/beam-transport/shell-soft-hard-transport.md)) and the opt-in coupled BremsLib radiative mode (helpers in `_jit_radiative_device.py`; see [bremsstrahlung](../../physics/radiation-physics/hard-bremsstrahlung-events.md));
- `_transport_lut_kernel` linearly interpolates precomputed energy tables, samples optional Urban straggling through the same device sampler from the exact element tables, and does not support the shell mode.

Both are `cupyx.jit.rawkernel` implementations of the same ungrooved, per-electron transport algorithm. The examples below use artificial inputs so that every operation can be followed by hand. They explain implementation of existing claims; they do not independently revalidate the physical models. Validation status remains owned by the [`gpu-transport-core`](../../validation/ledger-transport-background.md#gpu-transport-core), [`electron-transport`](../../validation/ledger-transport-background.md#electron-transport), [`relativistic-bethe-stopping`](../../validation/ledger-transport-background.md#relativistic-bethe-stopping), [`transport-midpoint-stopping`](../../validation/ledger-transport-background.md#transport-midpoint-stopping), [`energy-controlled-propagation`](../../validation/ledger-transport-background.md#energy-controlled-propagation), and [`energy-loss-straggling`](../../validation/ledger-transport-background.md#energy-loss-straggling) records.

## Launch and ownership

The default `TransportKernelConfig` uses 128 threads per block. The wrapper launches

$$
N_{\rm blocks}=\left\lceil\frac{N_e}{128}\right\rceil
$$

blocks and computes the local electron index

```text
i = blockIdx.x * blockDim.x + threadIdx.x
```

inside the kernel. Threads with `i >= e_count` return immediately. Every other thread owns exactly one electron and runs it from its entering state until an exit, the energy cutoff, or `max_steps`. There is no shared memory, block synchronization, atomic work queue, or handoff between threads.

Two indices remain distinct:

- `i` is local to the current launch batch;
- `e = e_start + i` addresses the global electron state and RNG key.

If `e_start = 10`, `i = 2`, and `cap = 8`, the thread transports global electron 12. Its first segment goes to `slot = i * cap + 0 = 16`, not slot 96 and not a globally incremented slot. Segment $s$ always goes to

$$
\operatorname{slot}(i,s)=i\,\texttt{cap}+s.
$$

This electron-local addressing makes launch order irrelevant. `seg_count[i]` continues increasing after $s=\texttt{cap}$ even though the overflowing rows are not written. The driver can therefore allocate the measured capacity and replay the batch without changing its random streams.

## Counter-addressed random draws

The host supplies one 64-bit `stream_key[e]` per electron. Draw `c` from that key is

$$
z_c=\operatorname{SplitMix64}\!\left(
K+\Phi(c+1)\pmod{2^{64}}\right),
\qquad
u_c=(z_c\mathbin{\texttt{>>}}11)2^{-53},
$$

where $\Phi=\texttt{0x9E3779B97F4A7C15}$. The SplitMix64 finalizer is the literal three-step integer sequence

```text
x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
x = (x ^ (x >> 27)) * 0x94D049BB133111EB
x =  x ^ (x >> 31)
```

with unsigned 64-bit wraparound. Shifting by 11 leaves at most 53 bits, so its conversion to `float64` is exact.

For the artificial key $K=0$, the first three kernel calls produce:

| counter $c$ | $z_c$ (hexadecimal) | $z_c\mathbin{\texttt{>>}}11$ | $u_c$ |
|---:|---:|---:|---:|
| 0 | `e220a8397b1dcdaf` | 7956156453446585 | 0.8833108082136426 |
| 1 | `6e789e6aa1b965f4` | 3886858653415212 | 0.4315279970485100 |
| 2 | `06c45d188009454f` | 238094247788840 | 0.0264337715925977 |

Those values depend only on $(K,c)$, not the CUDA block size or scheduling. The ordinary transport counter is consumed in this order for one-element material:

1. free-path optical depth;
2. polar scattering draw;
3. azimuth draw.

A multi-element collision inserts an element-selection draw between steps 1 and 2. A layer boundary or numerical energy substep performs no scattering and therefore consumes none of those later draws.

## Direct kernel event loop

One iteration of `_transport_kernel` has five stages.

### 1. Resolve material and sample a collision

The current $z$ position selects layer $L$. For every live element in that layer, the kernel evaluates either the Browning/Mott rate

$$
R_i(E)=\frac{A_i}{E+B_i\sqrt E+C_i/\sqrt E}
$$

or the analytic screened-Rutherford rate. It sums the element rates, whose unit is cm$^{-1}$, and converts the mean free path to ångström:

$$
R(E)=\sum_iR_i(E),
\qquad
\lambda(E)=\frac{10^8}{R(E)}\ \text{Å}.
$$

A new physical flight draws one exponential optical-depth budget

$$
\tau=-\ln u,
\qquad
s_{\rm collision}=\tau\lambda(E).
$$

`tau_left = -1` is the sentinel for “no flight open.” A numerical energy substep keeps the remaining nonnegative `tau_left`, so it does not draw another collision.

### 2. Find the first geometry event

For an infinite transverse slab, only the current layer's top or bottom $z$ plane can truncate the proposed step. For example, with direction $d_z>0$, current position $z$, and lower layer boundary $z_{\rm bot}$,

$$
s_{\rm boundary}=\frac{z_{\rm bot}-z}{d_z}.
$$

The finite-footprint branch instead tests all six planes of the rectangular prism and retains the nearest strictly positive intersection. A geometry event wins only when its distance is strictly smaller than the sampled collision step. A top/bottom crossing is an outer exit only when it is also the stack's outer face; otherwise the next iteration begins in the adjacent layer after a small signed $z$ nudge.

The full equality ordering is load-bearing:

1. collision beats geometry when their distances are equal because geometry truncation uses strict `step_j > s_boundary`;
2. cutoff beats an equal collision, but an equal geometry event beats cutoff;
3. an energy-control cap uses strict `step_energy < step_j`, so it loses every equality;
4. sampled straggling stops at $\Delta E\ge E-E_{\rm cut}$, except an exact tie at a geometry event, where geometry wins.

### 3. Propagate energy and record a segment

The direct helper `_dEds_packed` sums the element stopping laws and returns $C(E)=dE/ds<0$ in keV/Å. Frozen propagation uses

$$
E_{\rm end}=E_{\rm start}+C(E_{\rm start})s,
\qquad
\beta=\sqrt{1-\left(1+E_{\rm start}/510.99895\right)^{-2}}.
$$

Midpoint propagation first predicts

$$
E_{\rm pred}=E_{\rm start}+C(E_{\rm start})s
$$

and then applies the explicit midpoint rule

$$
E_{\rm end}=E_{\rm start}
+sC\!\left(\frac{E_{\rm start}+E_{\rm pred}}{2}\right),
\qquad
\beta=\beta\!\left(\frac{E_{\rm start}+E_{\rm end}}{2}\right).
$$

The recorded row contains the incoming direction, segment midpoint, length, starting energy, starting clock, global electron id, and layer. Midpoint mode also records end energy/time plus `flight_id` and `substep_id`. The transport clock advances in the repository's $c=1$ length convention:

$$
t_{\rm end}=t_{\rm start}+s/\beta.
$$

### 4. Advance persistent state

Position, energy, and clock move to the end of the row. The flight's optical depth is consumed using the same mean free path that proposed the row:

$$
\tau_{\rm left}\mathrel{-}=s/\lambda(E_{\rm start}).
$$

Small negative roundoff is clamped to zero.

### 5. Dispatch the event

The row ends in exactly one of these control paths:

- **energy-limited substep:** retain direction, `flight_id`, and `tau_left`; increment only `substep_id`;
- **outer-face or side exit:** set the corresponding exit code and stop;
- **energy cutoff:** set `CUTOFF_STOPPED` and stop;
- **internal layer boundary:** close the flight, reset `tau_left = -1`, nudge into the adjacent layer, and draw from that layer on the next iteration;
- **elastic collision:** close the flight by incrementing `flight_id`, resetting `substep_id`, and setting `tau_left = -1`; then select an element and rotate the direction before the next iteration.

## Worked direct-kernel collision

Consider local thread `i = 2`, global electron `e = 12`, with this artificial entering state:

| quantity | value |
|---|---:|
| position | $(0,0,2)$ Å |
| direction | $(0,0,1)$ |
| layer bounds | $z\in[0,10]$ Å |
| energy | 30 keV |
| cutoff | 5 keV |
| clock | 0 Å |
| total elastic rate | $2.0\times10^7$ cm$^{-1}$ |
| stopping rate | $-0.1$ keV/Å |

To keep the arithmetic exact, prescribe the next uniform as $u=e^{-1}$. The kernel obtains

$$
\tau=-\ln(e^{-1})=1,
\qquad
\lambda=\frac{10^8}{2.0\times10^7}=5\ \text{Å},
\qquad
s_{\rm collision}=5\ \text{Å}.
$$

The bottom boundary is $(10-2)/1=8$ Å away, so the collision at 5 Å wins. The frozen cutoff distance is

$$
s_{\rm cutoff}=\frac{5-30}{-0.1}=250\ \text{Å},
$$

so no cutoff occurs. The energy and speed are

$$
E_{\rm end}=30-0.1(5)=29.5\ \text{keV},
$$

$$
\beta(30\ \text{keV})
=\sqrt{1-\left(1+30/510.99895\right)^{-2}}
=0.3283761764.
$$

The thread writes its first row to slot $2\times8=16$:

| output | value |
|---|---:|
| `seg_dir` | $(0,0,1)$ |
| `seg_mid` | $(0,0,4.5)$ Å |
| `seg_len` | 5 Å |
| `seg_E` | 30 keV |
| `seg_t0` | 0 Å |
| `seg_id` | 12 |
| `seg_lay` | 0 |

Persistent state becomes $\mathbf r=(0,0,7)$ Å, $E=29.5$ keV, and $t=5/\beta=15.22643955$ Å. The budget update gives $\tau_{\rm left}=1-5/5=0$.

There is one element, so selection consumes no random draw. Let its screened parameter be $\alpha=1/3$, prescribe the polar draw $R=1/4$, and prescribe the azimuth draw $u_\phi=1/4$. The kernel evaluates

$$
\cos\theta
=1-\frac{2\alpha R}{1+\alpha-R}
=1-\frac{1/6}{13/12}
=\frac{11}{13},
\qquad
\phi=2\pi u_\phi=\frac{\pi}{2}.
$$

For the entering $+z$ direction, the kernel's generated orthonormal frame is $\mathbf u=(0,1,0)$ and $\mathbf w=(-1,0,0)$. Therefore

$$
\mathbf d_{\rm new}
=\cos\theta\,\mathbf d
+\sin\theta\cos\phi\,\mathbf u
+\sin\theta\sin\phi\,\mathbf w
=\left(-\frac{4\sqrt3}{13},0,\frac{11}{13}\right),
$$

or approximately $(-0.53293871,0,0.84615385)$. The kernel normalizes this vector once more against floating-point drift. Immediately before selecting the element and performing this rotation, it has already incremented `flight_id`, reset `substep_id`, and marked `tau_left = -1`, so the next iteration draws a new collision.

## Numerical substeps reuse the collision draw

Now switch the same artificial flight to midpoint propagation and set `max_dE_frac = 0.01`; this is the supported combination, and midpoint mode records `flight_id` and `substep_id`. Because the toy stopping rate is constant, the midpoint energy arithmetic remains identical. Its deterministic cap is

$$
s_{\rm energy}=\frac{0.01(30)}{0.1}=3\ \text{Å},
$$

which truncates the proposed 5 Å collision flight. The first row is `(flight_id, substep_id) = (0, 0)`, does not scatter, and leaves

$$
\tau_{\rm left}=1-3/5=0.4.
$$

On the next iteration, if the mean free path remains 5 Å, the same budget proposes $0.4(5)=2$ Å. No new uniform is drawn. That second row has id `(0, 1)` and reaches the originally sampled collision after a total of $3+2=5$ Å. Only then does the kernel rotate the direction and advance to `flight_id = 1`. This is the concrete reason numerical refinement does not create extra elastic collisions.

## Energy-LUT kernel

`_transport_lut_kernel` retains the same thread ownership, RNG, geometry, segment writes, optical-depth budget, midpoint rule, and direction rotation. It replaces repeated material formulas with linear interpolation on a grid uniform in $\ln E$, so resolution is constant in *relative* energy and a slowing electron keeps it near the cutoff.

For grid minimum $E_{\min}$ and inverse log spacing $h_{\ln}^{-1}$, it computes

$$
x=\bigl(\ln E-\ln E_{\min}\bigr)h_{\ln}^{-1},
\qquad
i=\lfloor x\rfloor,
\qquad
f=x-i,
$$

clamped to the first or last interval -- one logarithm, one multiply, one integer conversion, no search. The clamp tests the *upper* bound first, so a non-finite coordinate ($E=0$ gives $-\infty$, $E<0$ gives NaN) fails both comparisons and lands on the lower clamp rather than reaching $\lfloor x\rfloor$. `_lut_index_frac_scalar`, `_jit_device.py::_lut_lerp_at`, and the two inline forms in `_jit_lut_kernel.py` all spell out this same branch order; they must stay identical or the CPU and CUDA cores would clamp differently.

Any table $T$ is then read as

$$
T(E)=T_i+f(T_{i+1}-T_i).
$$

Values are interpolated directly -- only the coordinate is logarithmic. Interpolating $\ln T$ instead was measured and rejected: it costs an exponential per table read in the innermost loop, and the residual error at the shipped resolution is set by model joins, where the transform buys nothing.

### Worked LUT interpolation

Take an energy grid $[10,20,40]$ keV -- uniform in $\ln E$ with node ratio 2 -- and $E=20\sqrt2\approx28.284$ keV, so $\ln E_{\min}=\ln 10$, $h_{\ln}^{-1}=1/\ln 2$, $x=1.5$, $i=1$, and $f=0.5$. If the two bracketing total rates are $2.0\times10^7$ and $4.0\times10^7$ cm$^{-1}$, the kernel obtains

$$
R(28.284)=2.0\times10^7+0.5(4.0-2.0)\times10^7
=3.0\times10^7\ \text{cm}^{-1},
$$

$$
\lambda(28.284)=\frac{10^8}{3.0\times10^7}=\frac{10}{3}\ \text{Å}.
$$

Likewise, bracketing stopping values $-0.08$ and $-0.12$ keV/Å interpolate to $-0.10$ keV/Å, while inverse-speed values 3.2 and 3.0 interpolate to 3.1.

For a two-element layer, suppose element 0's cumulative probabilities at the same nodes are 0.25 and 0.35. Its interpolated threshold is 0.30; element 1's threshold remains 1. A selection draw $u=0.28$ chooses element 0. A draw $u=0.30$ chooses element 1 because the source comparison is strict `cumulative > u`, not `>=`.

After the energy update, the kernel recomputes the interpolation coordinate at the new energy before reading the selected element's `lut_alpha`. Scattering therefore uses the post-flight energy, matching the direct kernel.

With straggling on, the LUT kernel follows the CPU LUT core's straggled branch: the deterministic `max_dE_frac` cap uses the interpolated `lut_dEds`, then the Urban loss is sampled per element from the exact padded `(L_Js, L_Zs, L_ks, L_coeffs, L_E_cross)` tables the shared driver uploads, and the cutoff is the sampled-loss crossing test. Under SBETHE stopping the elemental `C_i` are rescaled by `lut_dEds / _dEds_packed`, as on the host. The clock still reads `lut_inv_beta`. Both kernels call the same `_urban_sample_compound` device function, so they sample one law in one draw order.

## Straggling stream separation

When enabled in either kernel, Urban loss does not consume from the base free-path/scattering counter. It hashes the electron key into a separate `urban_key`, then derives a `flight_key` from `(urban_key, flight_id, substep_id)`. A local straggling counter starts at zero for that key.

Consequently, enabling straggling cannot shift which base uniforms select the elastic collision distance or angles. Within the straggling stream, the kernel draws exact Poisson channel counts by inverse CDF, splitting large means into bounded chunks whose independent Poisson counts add. Continuum ionisation then uses one additional uniform per sampled quantum. The sampled row loss is accumulated in `stragg_dE`; if it crosses the cutoff, the row length is fluidly interpolated to the crossing and the applied end energy is exactly the cutoff.

CPU/CUDA straggling is not promised bit-for-bit after transcendental functions: a last-bit host/device `exp` difference can move a draw across a Poisson CDF boundary, after which the stochastic trajectories legitimately diverge. The separate-stream and fixed-seed contracts remain deterministic on each backend.

## Direct and LUT kernel comparison

| Concern | Direct `_transport_kernel` | `_transport_lut_kernel` |
|---|---|---|
| Thread ownership | one electron to completion | same |
| Default block size | 128 | 128 |
| Elastic rate | per-element formula | interpolated total rate |
| Element selection | recomputed rates | interpolated cumulative table |
| Stopping power | `_dEds_packed` | interpolated `lut_dEds` |
| Speed | `_beta_from_keV` | interpolated `lut_inv_beta` |
| Scatter parameter | formula or Mott table | interpolated `lut_alpha` |
| Midpoint propagation | direct stopping evaluations | repeated table interpolation |
| Energy substeps | supported | supported |
| Urban straggling | supported | supported (exact element tables, LUT cap and clock) |
| Geometry and output slots | identical structure | identical structure |

## Reading and changing these kernels safely

- Arithmetic is `float64`; changing precision changes chaotic trajectories, not merely their final formatting.
- Launch geometry must not affect results. Output slots and random draws are functions of electron identity, never thread scheduling.
- A capacity overflow must keep transporting and report the true segment count. Stopping at `cap` would make replay capacity change the trajectory.
- An energy-limited substep must preserve `tau_left`, direction, and `flight_id`, consume no base RNG draw, and skip collision/exit dispatch.
- Internal layer crossings intentionally close the current physical flight; the fresh layer draw follows from the exponential law's memorylessness.
- The direct and CPU per-electron cores keep arithmetic and draw order aligned, but host/device transcendental functions permit few-ulp first-step differences and eventual trajectory divergence.
- Focused anchors are in [`test_transport_per_electron.py`](../../../tests/montecarlo/test_transport_per_electron.py), [`test_transport_energy_model.py`](../../../tests/montecarlo/test_transport_energy_model.py), and [`test_straggling_cuda.py`](../../../tests/montecarlo/test_straggling_cuda.py). Performance history and GPU validation evidence remain in [GPU electron transport RawKernel](gpu-transport-rawkernel.md).
