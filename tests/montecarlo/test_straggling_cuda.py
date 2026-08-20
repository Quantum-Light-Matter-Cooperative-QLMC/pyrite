"""Straggling on the CUDA transport kernels.

Slice F of ``feature/energy-loss-straggling``. Two disjoint halves:

* **The fail-closed raise**, which needs no device and runs everywhere. The
  CUDA *LUT* kernel (``_transport_lut_kernel``) is deliberately not wired for
  straggling -- it carries no per-element split to sample from, no duplicated
  Urban sampler, and ``run_transport_lut_kernel`` does not accept the
  straggling parameters at all. ``simulate_trajectories`` must therefore raise
  for that combination rather than silently return an unstraggled result or an
  opaque ``TypeError``, matching the ``energy_model`` fail-closed precedent the
  task doc names. Asserted here rather than merely documented.

* **Hardware parity**, gated on an actual CUDA device. Slices D, E and F all
  ran on a machine with no ``cupy`` and no CUDA device, so the straggling code
  in ``transport_jit_kernel.py`` has never been compiled or executed. These
  tests are the check that transcription is correct; they are the reason the
  slice F report can name what is unverified rather than hand-waving it. Run
  them on a GPU box (``pyrite remote``).

The parity claim is deliberately *not* bit-for-bit and is not stated per row.
``_urban_poisson_scalar`` branches on a ``log``/``exp`` comparison, so a
last-bit libm difference between the host and the device can move a Poisson
count by one at a CDF boundary (slice C's own recommendation says few-ulp, not
bit-for-bit). Before slice F that only perturbed a diagnostic; now it changes
the electron's energy, so the trajectory diverges from that row onward. The
comparable quantity is therefore the *first* row of each electron -- taken at
the unperturbed start energy, before any straggling draw has been applied --
which is exactly the scope
``test_transport_per_electron.py::test_cuda_first_step_agrees_with_the_cpu_reference``
already uses for the deterministic path.
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories

NO_LUT = TransportLUTConfig(enabled=False)

BASE_KWARGS = dict(
    E0_keV=25.0,
    Ne=64,
    thickness_ang=4000.0,
    element="C",
    n_atoms_per_ang3=0.1136,
    seed=1234,
    max_steps=4000,
)


def _run(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


# --- fail-closed: the one path slice F deliberately left unwired -------------


def test_cuda_lut_core_raises_rather_than_running_unstraggled():
    """``straggling=True`` with ``transport_core='cuda'`` and the LUT enabled
    (the default LUT configuration) must raise ``NotImplementedError``.

    This runs without a CUDA device because the guard sits in
    ``simulate_trajectories`` ahead of the ``transport_jit_kernel`` import, by
    construction: the point of a fail-closed guard is that it fires before
    anything device-specific is reached. The message must name both escapes so
    a caller who hits it can act on it.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        _run(straggling=True, transport_core="cuda")
    message = str(excinfo.value)
    assert "CUDA LUT core" in message
    assert "TransportLUTConfig(enabled=False)" in message
    assert "per-electron" in message


def test_cuda_lut_core_does_not_raise_with_straggling_off():
    """The guard is scoped to ``straggling=True`` and must not have made the
    ordinary CUDA LUT path unreachable. Without a device this resolves away
    from CUDA before the guard, so the assertion is that *no*
    ``NotImplementedError`` escapes, on either kind of machine."""
    try:
        _run(straggling=False, transport_core="cuda")
    except NotImplementedError:  # pragma: no cover - would be the bug
        pytest.fail("the straggling guard fired with straggling off")
    except Exception:
        # No CUDA device on this machine: `resolve_transport_core` refuses the
        # request long before the straggling guard. That is not what this test
        # is about.
        pass


# --- hardware parity ---------------------------------------------------------

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")

CUDA = dict(transport_core="cuda", transport_lut_config=NO_LUT)
HOST = dict(transport_core="per-electron", transport_lut_config=NO_LUT)


@pytest.mark.hardware
@requires_cuda
def test_cuda_straggling_off_is_bit_for_bit():
    """Property 1 on the device: the straggling-off path is untouched by slice
    F's restructuring of the kernel's energy close into an
    ``if straggle_on: ... else: ...`` pair."""
    default = _run(**CUDA)
    explicit_off = _run(**CUDA, straggling=False)
    for key, left in default.items():
        right = explicit_off[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=key)
        elif isinstance(left, (int, float, np.number)):
            assert left == right, key


@pytest.mark.hardware
@requires_cuda
def test_cuda_straggling_is_deterministic():
    """The straggling draw is counter-addressed, so it must not depend on
    launch order or on anything else the device varies between runs."""
    a = _run(**CUDA, straggling=True)
    b = _run(**CUDA, straggling=True)
    for key in ("L_ang", "E_keV", "E_start_keV", "straggle_dE_keV", "elec_id"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.hardware
@requires_cuda
def test_cuda_straggling_conserves_energy_sensibly():
    """Property 2 on the device: the loss is applied and lands somewhere
    physical. Same invariants as the host cores' own test."""
    Ne = BASE_KWARGS["Ne"]
    on = _run(**CUDA, straggling=True, thickness_ang=60000.0, max_steps=20000)
    assert np.all(np.isfinite(on["E_start_keV"]))
    assert np.all(on["E_start_keV"] >= 5.0)
    assert np.all(on["L_ang"] >= 0.0)
    assert np.all(np.isfinite(on["straggle_dE_keV"]))
    assert np.all(on["straggle_dE_keV"] >= 0.0)
    channels = (
        on["n_backscattered"] + on["n_transmitted"] + on["n_side_exited"] + on["n_cutoff_stopped"]
    )
    assert channels == Ne
    # The `stragg_loss > 0` guard: a NaN from a 0/0 crossing would show up in
    # `pos` (hence `r_mid`) rather than raising, so it is checked directly.
    assert np.all(np.isfinite(on["r_mid"]))


@pytest.mark.hardware
@requires_cuda
def test_cuda_first_row_agrees_with_the_cpu_reference_under_straggling():
    """The parity claim, at the scope where it holds.

    Both cores draw the same numbers from the same counter-addressed streams
    and take the same branches, so each electron's first row -- taken at the
    unperturbed start energy -- differs only by libm rounding. Later rows are
    not compared: a last-bit difference in a Poisson lambda can flip a count at
    a CDF boundary, and once the loss is applied that changes the electron's
    energy and the trajectory diverges from there. The tolerance is the same
    ``rtol=1e-12`` the deterministic first-step parity test uses.
    """
    cpu = _run(**HOST, straggling=True)
    gpu = _run(**CUDA, straggling=True)

    cpu_first = np.flatnonzero(np.diff(cpu["elec_id"], prepend=-1))
    gpu_first = np.flatnonzero(np.diff(gpu["elec_id"], prepend=-1))
    assert np.array_equal(cpu["elec_id"][cpu_first], gpu["elec_id"][gpu_first])
    np.testing.assert_allclose(cpu["L_ang"][cpu_first], gpu["L_ang"][gpu_first], rtol=1e-12)
    # The applied loss itself: the first row's end energy is the start energy
    # minus that row's own draw, so this is the direct check that the device
    # applied the same loss the host did.
    np.testing.assert_allclose(cpu["E_keV"][cpu_first], gpu["E_keV"][gpu_first], rtol=1e-12)
    np.testing.assert_allclose(cpu["r_mid"][cpu_first], gpu["r_mid"][gpu_first], rtol=1e-12)


@pytest.mark.hardware
@requires_cuda
def test_cuda_straggling_matches_the_host_in_distribution():
    """The ensemble statement, since the pathwise one only survives one row.

    A flipped Poisson count is a rare, unbiased perturbation, so the summed
    sampled loss over a whole run must agree between host and device to
    sampling error rather than to ulp. A transcription error in a channel
    weight, a level, or the draw order would move this well outside the
    tolerance while leaving the first-row test intact only by luck."""
    cpu = _run(**HOST, straggling=True, Ne=512)
    gpu = _run(**CUDA, straggling=True, Ne=512)
    assert gpu["straggle_dE_keV"].sum() == pytest.approx(cpu["straggle_dE_keV"].sum(), rel=0.02)
    assert gpu["n_cutoff_stopped"] == pytest.approx(cpu["n_cutoff_stopped"], rel=0.05, abs=2)
