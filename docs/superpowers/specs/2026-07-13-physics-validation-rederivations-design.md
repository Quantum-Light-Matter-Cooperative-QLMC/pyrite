# Physics Validation Rederivations Design

## Goal

Advance three independent physics claims from the validation ledger through
fresh-context rederivation and root-context adjudication:

- `line-energy-dispersion`
- `finite-time-lineshape`
- `absorption-length`

This work may advance a claim only as far as `rederived` or `anchored`. Only a
human may mark a claim `signed-off`.

## Claim boundaries

Each claim is an independent work unit with its own derivation write-up at
`docs/validation/<id>.md`.

### Line-energy dispersion

Derive the emitted-photon dispersion relation
`omega = (v dot g) / (1 - v dot n_hat)` from phase matching. Check dimensions,
the reciprocal-vector sign convention, the observation-direction convention,
and behavior under the repository's positive-tilt convention. Compare the
result with both production geometry and the validation-only line-energy
helper.

### Finite-time lineshape

Derive the squared modulus of a constant-amplitude finite-time Fourier
integral. State the repository's `sinc` convention explicitly, recover the
zero-detuning value, and demonstrate the distributional infinite-duration
limit including its normalization. Compare the result with the finite-segment
factor in the coherent spectrum.

### Absorption length

Starting from the complex refractive index or atomic scattering factor, derive
the linear attenuation coefficient and absorption length expressed through
`f2`. Check number-density and wavelength units, positivity, the
Beer--Lambert convention for intensity rather than field amplitude, and the
transparent-medium limit. Compare the result with the crystal-material helper.

## Independence protocol

One fresh-context agent is assigned to each claim. An agent receives only the
cited physics source or starting law, the intended quantity, the public
signature or input/output description, and the validation checklist. It must
not inspect the owning implementation, tests that encode the implementation,
or another agent's derivation before recording its independent expression.

Each agent writes a standalone document containing:

1. source and starting assumptions;
2. step-by-step derivation;
3. units, normalization, sign, and coordinate checks as applicable;
4. at least one limiting case;
5. a final implementation-neutral expression; and
6. any ambiguity or missing source information that prevents a clean result.

## Adjudication and changes

The root context reviews each derivation against the owning implementation
only after the independent expression is durable. It records a symbolic,
dimensional, and, where useful, numerical diff in the same validation document.

- A match advances the ledger status to `rederived`.
- A mismatch advances it to `discrepancy` and records the exact difference;
  production physics is not silently changed.
- If a focused regression anchor is missing and an unambiguous analytic value
  exists, add the smallest CPU test or check that pins the claim. A green anchor
  may advance the claim to `anchored`.
- Existing docstrings must retain a derivation source, assumptions, a limiting
  case, and the matching `Validation: <id>` marker. Surgical docstring repairs
  are in scope; unrelated refactoring is not.

## Verification

For every claim, verify units, conventions, normalization, and relevant limits.
Run the smallest tests covering any edited implementation or anchor, then run
the ledger/document consistency checks available in the repository. Physics
review must use a context that did not author the production change. The
existing unrelated worktree modifications are out of scope and must remain
untouched.
