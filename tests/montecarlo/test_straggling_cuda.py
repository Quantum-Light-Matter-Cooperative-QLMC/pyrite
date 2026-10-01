"""Straggling on the CUDA transport kernels.

Slice F of ``feature/energy-loss-straggling`` wired the exact CUDA kernel;
issue #280 wired the CUDA *LUT* kernel through the same shared
``_urban_sample_compound`` device sampler. Two halves:

* **Core selection**, which needs no device: a straggled production case keeps
  the CUDA LUT core instead of being forced onto the exact one.

* **Hardware parity**, gated on an actual CUDA device and run through
  ``pyrite remote``. Slices D, E and F were authored on a machine without CUDA,
  so these tests close the transcription gap.

The parity claim is deliberately *not* bit-for-bit and is not stated per row.
``_urban_poisson_scalar`` inverts a floating-point CDF, so a last-bit libm
difference between the host and the device can move a Poisson count by one at a
CDF boundary (slice C's own recommendation says few-ulp, not bit-for-bit).
Before slice F that only perturbed a diagnostic; now it changes
the electron's energy, so the trajectory diverges from that row onward. The
comparable quantity is therefore the *first* row of each electron -- taken at
the unperturbed start energy, before any straggling draw has been applied --
which is exactly the scope
``test_transport_per_electron.py::test_cuda_first_step_agrees_with_the_cpu_reference``
already uses for the deterministic path. LUT-versus-exact agreement is a
distribution claim on top of that, with the LUT table error bounded separately
by ``test_transport_lut.py``.

Validation: energy-loss-straggling
"""

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
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
    elastic_model="sr",
)


def _run(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


# --- core selection ----------------------------------------------------------


def test_straggled_production_case_keeps_the_cuda_lut(monkeypatch):
    """Straggling no longer forces the exact CUDA kernel (#280)."""
    from pyrite.montecarlo.transport import batching

    monkeypatch.setattr(batching, "_cuda_transport_available", lambda: True)
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE", raising=False)
    case = dict(
        build_cases(material_sweep("silicon"), 4, 4, energy_model="midpoint", straggling=True)[0]
    )
    case["Ne"] = 10**5
    assert runner._case_transport_core(case) == "cuda"

    class Launched(Exception):
        pass

    seen = {}

    def launch(*args, **kwargs):
        seen.update(kwargs)
        raise Launched

    monkeypatch.setattr(runner, "simulate_trajectories", launch)
    with pytest.raises(Launched):
        runner._transport_case(case)
    assert seen["transport_core"] == "cuda"
    assert seen["straggling"] is True
    assert "transport_lut_config" not in seen


# --- hardware parity ---------------------------------------------------------

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")

CUDA = dict(transport_core="cuda", transport_lut_config=NO_LUT)
HOST = dict(transport_core="per-electron", transport_lut_config=NO_LUT)
CUDA_LUT = dict(transport_core="cuda")
HOST_LUT = dict(transport_core="per-electron")


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
def test_cuda_sbethe_straggling_first_row_matches_host():
    """Exact CUDA uses the same SBETHE mean and Urban rescaling as the host."""
    table = {
        "stopping_energy_eV": np.array([5.0e3, 10.0e3, 25.0e3]),
        "stopping_eV_per_angstrom": np.array([1.0, 1.0, 1.0]),
    }
    common = dict(
        straggling=True,
        energy_model="midpoint",
        stopping_tables=[table],
    )
    cpu = _run(**HOST, **common)
    gpu = _run(**CUDA, **common)

    cpu_first = np.flatnonzero(np.diff(cpu["elec_id"], prepend=-1))
    gpu_first = np.flatnonzero(np.diff(gpu["elec_id"], prepend=-1))
    assert np.array_equal(cpu["elec_id"][cpu_first], gpu["elec_id"][gpu_first])
    np.testing.assert_allclose(cpu["L_ang"][cpu_first], gpu["L_ang"][gpu_first], rtol=1e-12)
    np.testing.assert_allclose(
        cpu["E_end_keV"][cpu_first],
        gpu["E_end_keV"][gpu_first],
        rtol=1e-12,
    )


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


# --- CUDA LUT kernel (#280) --------------------------------------------------


@pytest.mark.hardware
@requires_cuda
def test_cuda_lut_straggling_off_is_bit_for_bit():
    """The LUT kernel's straggled branch leaves the off path untouched."""
    default = _run(**CUDA_LUT)
    explicit_off = _run(**CUDA_LUT, straggling=False)
    assert "straggle_dE_keV" not in default
    for key, left in default.items():
        right = explicit_off[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=key)
        elif isinstance(left, (int, float, np.number)):
            assert left == right, key


@pytest.mark.hardware
@requires_cuda
def test_cuda_lut_straggling_is_deterministic():
    a = _run(**CUDA_LUT, straggling=True, energy_model="midpoint", max_dE_frac=0.02)
    b = _run(**CUDA_LUT, straggling=True, energy_model="midpoint", max_dE_frac=0.02)
    for key in ("L_ang", "E_keV", "E_end_keV", "straggle_dE_keV", "elec_id", "substep_id"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize(
    "numerics",
    [
        dict(energy_model="frozen"),
        dict(energy_model="midpoint", max_dE_frac=0.02),
        dict(
            energy_model="midpoint",
            stopping_tables=[
                {
                    "stopping_energy_eV": np.array([5.0e3, 10.0e3, 25.0e3]),
                    "stopping_eV_per_angstrom": np.array([1.0, 1.0, 1.0]),
                }
            ],
        ),
    ],
    ids=["frozen", "midpoint-cap", "midpoint-sbethe"],
)
def test_cuda_lut_first_row_matches_the_cpu_lut_core_under_straggling(numerics):
    """Transcription check of the LUT kernel against its CPU twin.

    Same scope as the exact-kernel first-row test: identical LUT inputs,
    streams and branches, so only libm rounding separates the two on the row
    taken at the unperturbed start energy. The SBETHE case exercises the
    ``stopping_scale`` rescaling of ``C_i`` to the LUT stopping.
    """
    cpu = _run(**HOST_LUT, straggling=True, **numerics)
    gpu = _run(**CUDA_LUT, straggling=True, **numerics)

    cpu_first = np.flatnonzero(np.diff(cpu["elec_id"], prepend=-1))
    gpu_first = np.flatnonzero(np.diff(gpu["elec_id"], prepend=-1))
    assert np.array_equal(cpu["elec_id"][cpu_first], gpu["elec_id"][gpu_first])
    np.testing.assert_allclose(cpu["L_ang"][cpu_first], gpu["L_ang"][gpu_first], rtol=1e-12)
    end = "E_end_keV" if "E_end_keV" in cpu else "E_keV"
    np.testing.assert_allclose(cpu[end][cpu_first], gpu[end][gpu_first], rtol=1e-12)
    np.testing.assert_allclose(cpu["r_mid"][cpu_first], gpu["r_mid"][gpu_first], rtol=1e-12)


def _pooled(core, numerics, seeds, Ne):
    """Per-electron sampled loss and cutoff count pooled over ``seeds``."""
    losses, n_cutoff = [], 0
    for seed in seeds:
        out = _run(
            **core,
            straggling=True,
            seed=seed,
            Ne=Ne,
            thickness_ang=60000.0,
            max_steps=20000,
            **numerics,
        )
        losses.append(out["straggle_dE_keV"])
        n_cutoff += int(out["n_cutoff_stopped"])
    return np.concatenate(losses), n_cutoff


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize(
    "numerics",
    [dict(energy_model="frozen"), dict(energy_model="midpoint", max_dE_frac=0.02)],
    ids=["no-cap", "binding-cap"],
)
def test_cuda_lut_matches_exact_cuda_in_distribution(numerics):
    """LUT and exact CUDA cores agree in distribution under straggling.

    Mean, variance and upper tail of the per-electron sampled loss, and the
    cutoff fraction, pooled over seeds. Tolerances are four standard errors of
    *independent* samples; the two cores share stream keys, so their runs are
    positively correlated and the test is conservative. The LUT table error
    (``test_transport_lut.py``) is far below these Monte Carlo errors.
    """
    seeds, Ne = (11, 22, 33, 44), 1024
    exact, exact_cut = _pooled(CUDA, numerics, seeds, Ne)
    lut, lut_cut = _pooled(CUDA_LUT, numerics, seeds, Ne)
    n = exact.size
    assert lut.size == n

    se_mean = np.sqrt((exact.var(ddof=1) + lut.var(ddof=1)) / n)
    assert abs(lut.mean() - exact.mean()) < 4.0 * se_mean

    # Var(s^2) ~ (mu4 - sigma^4)/n per sample.
    def var_se(x):
        c = x - x.mean()
        return np.sqrt(max((np.mean(c**4) - np.mean(c**2) ** 2) / n, 0.0))

    se_var = np.hypot(var_se(exact), var_se(lut))
    assert abs(lut.var(ddof=1) - exact.var(ddof=1)) < 4.0 * se_var

    # Upper tail: fraction of LUT electrons above the exact 95th percentile.
    q95 = np.quantile(exact, 0.95)
    tail = np.mean(lut > q95)
    assert abs(tail - 0.05) < 4.0 * np.sqrt(2.0 * 0.05 * 0.95 / n)

    p_exact, p_lut = exact_cut / n, lut_cut / n
    p_pool = 0.5 * (p_exact + p_lut)
    assert abs(p_lut - p_exact) < 4.0 * np.sqrt(2.0 * p_pool * (1.0 - p_pool) / n) + 1.0 / n
