# GPT GDF electron injection

Validation: `gpt-gdf-injection`. Independent derivation recorded on
2026-09-23 before inspecting `src/pyrite/montecarlo/gdf.py`.

## Source and scope

The sources are the supplied GPT time-output contract, special relativity
$E=\gamma mc^2$, and elementary ray-plane intersection. A record contains
lab-frame position in metres, velocity components $\boldsymbol\beta=\mathbf v/c$,
electron mass $m$ in kg, negative single-electron charge $q$ in C, and positive
multiplicity $n$. All records belong to one common snapshot time. The contract
is an electron ensemble, with field-free straight extrapolation to a flat
entrance and incoherent emission; it does not define coherent bunch radiation.

## Independent derivation

For $0<\beta=\|\boldsymbol\beta\|<1$,

$$
\gamma=(1-\beta^2)^{-1/2},\qquad
K_{\rm keV}=\frac{(\gamma-1)mc^2}{10^3 e},\qquad
\hat{\mathbf d}_{\rm lab}=\frac{\boldsymbol\beta}{\beta},
$$

where $e$ is the positive elementary charge in C. The denominator converts
joules to keV. Rationalizing avoids cancellation at small speed:

$$
\gamma-1=\frac{\gamma^2\beta^2}{\gamma+1}.
$$

Thus $K\to mv^2/2$ as speed tends to zero; zero speed itself has no injection
direction. Electron multiplicities give signed charge and positive average
current magnitude at repetition frequency $f$:

$$
N=\sum_i n_i,\qquad Q=\sum_i q_i n_i<0,\qquad I=|Q|f.
$$

For identical electron charge $q_i=-e$, $|Q|=eN$. Current normalization by a
configured $I$ implies electron arrival rate $I/e$, independently of arbitrary
common scaling of the imported multiplicities. Bunch-charge normalization
instead uses $fN$ electrons per second, equal to $|Q|f/e$ for electrons.

Draw complete records with replacement using

$$
p_i=\frac{n_i}{N},\qquad
\widehat A=\frac1M\sum_{k=1}^M A(X_{J_k}),\qquad
\mathbb E[\widehat A]=\sum_i\frac{n_i}{N}A(X_i).
$$

For stochastic transport, $A(X_i)$ here means its conditional expected score.
The current-normalized rate is $(I/e)\widehat A$. Multiplicity is already
accounted for in $p_i$ and must not multiply individual scores again. Selecting
one row index for position, direction, energy, and time preserves all their
joint correlations; separate marginal draws do not. Equal multiplicities
recover uniform sampling. Misses contribute zero but remain in $M$.

Let $R$ be the orthogonal `sample_to_lab_R` rotation acting on column vectors,
and let the explicitly specified lab origin be
$\mathbf o=(0,0,z_{\rm origin})$. With $a=10^{10}$ angstroms per metre,

$$
\mathbf r_i=aR^T(\mathbf x_i-\mathbf o),\qquad
\hat{\mathbf d}_i=R^T\hat{\mathbf d}_{{\rm lab},i}.
$$

For row arrays these are right multiplications by $R$. Rotation preserves
length and speed. Solve the ray equation at the sample entrance:

$$
\mathbf r_{i,\rm ent}=\mathbf r_i+s_i\hat{\mathbf d}_i,\qquad
s_i=-\frac{r_{iz}}{d_{iz}},\qquad r_{i,\rm ent,z}=0.
$$

The signed path $s_i$ may be negative: a snapshot downstream of the entrance
is back-extrapolated. Taking its absolute value would destroy the time and
transverse-position relation. Forward entrance requires $d_{iz}>0$;
parallel rays have no unique intersection. Under the $c=1$ convention, time
is stored as a length, so relative to the common snapshot:

$$
\Delta t_{i,\rm ang}=\frac{s_i}{\beta_i}.
$$

An absolute clock would additionally contain $act_{\rm snapshot}$; the common
term is unnecessary for these incoherent relative offsets. At zero tilt with
records already on the entrance, transverse positions and directions are
unchanged and offsets vanish. At $\beta=0.6$, $\gamma=1.25$ and
$K=0.25mc^2$; an untilted position one metre upstream with forward axial
velocity has $s=10^{10}$ angstroms and
$\Delta t=10^{10}/0.6$ angstroms. A position one metre downstream reverses
both signs.

## Filters

Units pass: energy is joules divided by joules per keV, charge is coulombs,
current is coulombs per second, positions and clock offsets are angstroms.
Limits pass: nonrelativistic energy, equal-weight sampling, zero tilt and
on-plane identity. Signs and conventions pass: electron charge remains
negative, average current magnitude remains positive, orthogonal coordinate
change uses the inverse rotation, and drift and clock use the same signed
path.

## Implementation comparison

Inspection after recording the derivation finds term-for-term agreement:

- `gamma` uses the velocity norm; `energy` uses the rationalized expression
  and the supplied mass, with the correct joule-to-keV conversion.
- Electron mass and charge are both checked against the physical constants
  with relative tolerance $10^{-6}$ and zero absolute tolerance. The charge
  check is essential: accepting arbitrary negative charges would invalidate
  the electron-rate normalization. Retaining the accepted input charges in
  the charge sum permits a relative normalization difference up to that
  input tolerance, rather than asserting mathematically exact equal charges.
- `signed` is the multiplicity-weighted charge sum and `absolute` is its
  negative, equal to its magnitude for the accepted all-electron ensemble.
- Rescaling multiplicities by their maximum before normalizing preserves
  $p_i$ and avoids overflow in the probability sum. One sampled index array
  selects every phase-space component. No second multiplicity is returned.
- `sample_to_lab_R` explicitly maps sample column vectors to lab columns;
  the adapter correctly right-multiplies lab row vectors by this rotation.
  Subtracting the specified lab-z origin before rotation agrees exactly
  with the derived affine transform.
- `distance` is the signed intersection distance in metres, `position_ang`
  converts the resulting entrance point by $10^{10}$, and `time_ang`
  converts the signed distance divided by speed by the same factor.
  This is a snapshot-relative clock, not an absolute GPT time.
- The code restricts all selected-source directions to $d_z>10^{-6}$,
  a stricter numerical domain than the geometric $d_z>0$ condition.
  Rejection occurs before resampling, so invalid rows cannot be silently
  discarded to bias the ensemble.

When `nmacro` is absent, `pyrite_current` uses uniform row probabilities and
leaves bunch charge unknown. This extends the weighted contract only under
an explicit equal-weight interpretation; `gdf_charge` correctly refuses
missing multiplicities. Arbitrary unknown unequal weights cannot be
reconstructed from such a file.

## Verdict and verification boundary

**Verdict: rederived.** Units, limits, and signs/conventions pass; no divergent
term remains in the inspected adapter under the electron charge tolerance,
forward-ray restriction, and equal-weight fallback stated above. This review
validates the adapter equations and sampling estimator, not coherent emission,
GPU behavior, raw-file format provenance, or downstream rate wiring.

The implementation's $\beta=0.6$ expression reduces directly to
$\gamma^2\beta^2/(\gamma+1)=0.25$, and its axial drift reproduces both
signed one-metre examples above. Automated numeric, transport, documentation,
and rendered-math checks are owned by the integrating agent; this verifier
has not independently run them. Suggested ledger change: `filtered` to
`rederived`, with this write-up and the accepted charge tolerance noted.
Human sign-off remains pending.


## Independent screen-clock extension

A fresh verifier derived this extension on 2026-09-23 from the supplied GPT
screen contract before reading the implementation body. Unlike a common-time
snapshot, each screen row supplies its own lab crossing time $t_i$ in seconds.
Positions, velocities, rotation, and the explicit lab origin retain the
conventions above. Assume field-free constant velocity between each crossing
and the entrance, including signed backward extrapolation.

Choose one reference from the complete selected input block, before any
resampling:

$$
t_{\rm ref}=\min_j t_j.
$$

If $s_i$ denotes the signed ray distance in angstroms defined above, the
physical entrance event occurs at $t_i+s_i/(ac\beta_i)$. Subtracting the common
reference and expressing time as an angstrom length gives

$$
\boxed{T_{i,\rm ent}=ac(t_i-t_{\rm ref})+\frac{s_i}{\beta_i}},
\qquad a=10^{10}\ {\rm angstrom/m}.
$$

Thus the time term must be added to the signed flight term, with no absolute
value and no division of the crossing-time term by $\beta_i$. If the code
measures signed flight distance in metres, multiply the whole sum
$c(t_i-t_{\rm ref})+s_{i,\rm m}/\beta_i$ by $a$.

Units pass: both terms are lengths in angstroms. Limits pass: equal crossing
times recover the snapshot-relative clock; rows on the entrance retain their
relative crossing times; zero tilt retains the ordinary axial drift. A common
shift of every input time cancels exactly in the mathematical expression.
Signs pass: downstream records have negative flight contributions. An
entrance time may therefore be negative relative to the earliest screen
crossing; clamping it would be incorrect. The reference is a clock convention,
not a claim that the earliest crossing is the earliest entrance event.

For a numeric check, a row on the entrance crossing $10^{-9}$ s after the
reference has $T=2.99792458\times10^9$ angstroms. With $\beta=0.6$ and a signed
one-metre upstream path, its entrance clock is additionally
$10^{10}/0.6$ angstroms; a one-metre downstream path subtracts that quantity.
The selected-source reference must be shared across all sampled particles,
including repeated or omitted input rows; taking a sampled minimum changes
clock provenance with the random draw.


### Screen implementation comparison and verdict

After recording the extension, inspection of `GDFBeam.sample` finds exact
agreement: `distance / speed * 1e10` implements the signed flight term and
`(self.arrival_time_s[indices] - self.arrival_time_s.min()) * c * 1e10`
implements the crossing offset with the same complete-record sample indices.
The minimum uses the complete selected source, so the reference remains fixed
across sample counts and seeds. `load_gdf_beam` requires screen `t` arrays to
be finite, numeric, one dimensional, and the same length as the positions.
Time-output blocks leave `arrival_time_s` absent, preserving the snapshot
limit. No restriction to nonnegative entrance clocks is imposed.

**Screen-extension verdict: rederived.** Units, limits, and signs/conventions
pass; no divergent factor or clock convention was found. This validates the
supplied per-row crossing-time contract and field-free projection, not the
external GPT writer's format provenance, acceleration between planes, or
coherent radiation. Suggested ledger edit: retain `rederived`, extend the
source and scope to GPT screen crossing times, and include the signed flight
plus source-referenced crossing-time equation. Human sign-off remains pending.

## Independent shape-only energy extension

A fresh verifier derived this extension on 2026-09-23 before reading the
implementation of the optional `energy_keV` argument to `GDFBeam.sample`.
The supplied shape-only contract retains each complete record's position,
normalized velocity direction, and sampling multiplicity, but replaces its
kinetic energy by a common positive finite sweep energy $K$ in keV. Screen
crossing times are discarded: the new ensemble uses a common reference clock
at its imported positions. This is a prescribed ensemble transformation,
not acceleration through an unspecified field or a reconstruction of the
original GPT time history.

From the special-relativistic energy relation $E=\gamma m_e c^2$, define the
electron rest energy $M=m_e c^2/(10^3e)$ in keV and $x=K/M$. Then

$$
\gamma=1+x,\qquad
\beta_K^2=1-\frac{1}{(1+x)^2}
=\frac{x(x+2)}{(1+x)^2}.
$$

The subtraction form loses precision when $x$ is small. A positive-factor
form that avoids this cancellation is

$$
\boxed{\beta_K=\sqrt{\frac{x}{1+x}}
                    \sqrt{\frac{x+2}{1+x}}}.
$$

It also avoids explicitly squaring $x$. The nonrelativistic limit is
$\beta_K\sim\sqrt{2K/M}$; the ultrarelativistic limit is $\beta_K\to1$.
At $K=M/4$, $\gamma=5/4$ and $\beta_K=3/5$ exactly. Zero energy is excluded
because a stationary electron has no finite drift clock for a nonzero path.

The spatial ray and sampling law remain those derived above. In particular,
$s_i=-r_{iz}/d_{iz}$ in angstroms depends on direction and position, not speed.
The transformed injection is therefore

$$
K_i=K,\qquad
\mathbf r_{i,\rm ent}=\mathbf r_i+s_i\hat{\mathbf d}_i,\qquad
\boxed{T_{i,\rm ent}=\frac{s_i}{\beta_K}},\qquad
p_i=\frac{n_i}{\sum_j n_j}.
$$

No term involving the imported crossing time remains. At the entrance every
clock is zero, including screen rows with distinct original crossing times.
For a one-metre upstream axial path and $K=M/4$, the clock is
$10^{10}/0.6$ angstroms; the same downstream path gives its negative.
Increasing $K$ reduces the magnitude of either signed drift clock. A common
rescaling of multiplicities leaves the sampled ensemble unchanged.

Units pass: $x$ and speed are dimensionless, while signed path divided by
speed is an angstrom clock. Limits pass: on-plane zero, nonrelativistic speed,
and ultrarelativistic flight time. Signs and conventions pass: the direction
and signed projection are preserved, and the clock uses the new speed without
retaining a crossing-time contribution from the original beam.

### Shape-only implementation comparison and verdict

Inspection after the independent derivation finds algebraic agreement.
`GDFBeam.sample` computes the new dimensionless energy using keV-to-joule
conversion and the physical electron mass, then evaluates

```text
sqrt((k / (1 + k)) * (1 + 1 / (1 + k)))
```

The second factor equals $(x+2)/(x+1)$, so this expression is exactly the
derived speed. It avoids small-energy subtraction and large-energy squaring.
At $x=1/4$ its factors are $1/5$ and $9/5$, giving $\sqrt{9/25}=3/5$.
Finite positive energy validation excludes the singular zero-speed limit;
extremely small subnormal energies remain subject to ordinary floating-point
underflow and the adapter's finite-projection guard.

The sampled indices and probabilities are unchanged. Positions and normalized
directions are calculated before replacing speed, so the entrance rays retain
their original geometry. `energies.fill(energy_keV)` assigns every returned
history the scalar sweep energy. `distance / speed * 1e10` uses the new speed
and signed distance in metres. The screen-time addition is guarded by
`energy_keV is None`, correctly omitting all imported crossing-time offsets
for shape-only injection while preserving the original full-record behavior.

**Shape-only verdict: rederived.** Units, limits, and signs/conventions pass;
no divergent term or clock convention was found. The claim concerns the
prescribed monoenergetic ensemble, not preservation of momentum, emittance,
original time-energy correlations, or physical acceleration between planes.
The integrating agent owns automated numeric tests, documentation tests, and
rendered-math checks. Suggested ledger edit: retain `rederived`, extend the
scope to shape-only monoenergetic injection, and record the stable relativistic
speed and signed drift-only clock. Human sign-off remains pending.
