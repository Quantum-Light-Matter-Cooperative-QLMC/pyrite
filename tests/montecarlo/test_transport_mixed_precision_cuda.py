"""Mixed-precision build of the exact CUDA transport kernel (#838).

The transpile tests need CuPy but no device: they check that the default build
has no narrowing cast and that every FP32 helper twin is free of ``double``.
The comparisons run the mixed kernel on a device through ``pyrite remote``.
They pin its contracts: it runs, its continuation schedule is still exact, and
it tracks the FP64 kernel at rounding scale. They are not the #510 refinement
gates.

Validation: gpu-mixed-precision-tables
Validation: gpu-mixed-precision-math
"""

import importlib.util
import re

import numpy as np
import pytest

from pyrite.montecarlo.transport import simulate_trajectories

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")
# Transpiling needs CuPy installed, not a device.
requires_cupy = pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="no CuPy")

# Kernel parameter types as `run_transport_kernel` passes them; every array
# not listed is float64. Only the transpiler sees these, so no device is used.
_SCALARS = {
    np.int32: (
        "e_start e_count cap max_steps n_layers elastic_model_code_arg energy_model_code_arg "
        "finite_footprint_arg sbethe_on_arg sbethe_width max_el el_n_mu el_guide_cells "
        "straggle_on_arg inelastic_on_arg il_width il_max_ch sec_on_arg rad_on_arg rad_max_el "
        "rad_max_t rad_n_red cont_on_arg"
    ).split(),
    np.float64: "max_dE_frac z_total width_ang height_ang il_cutoff_eV rad_cutoff_eV".split(),
}
_ARRAYS = {
    np.uint64: "stream_key il_keys rad_keys cont_u64".split(),
    np.uint8: "alive mott_has_table".split(),
    np.int32: (
        "L_sbethe_n L_nel mott_start mott_len el_start el_len el_guide seg_id seg_lay seg_flight "
        "seg_substep seg_count il_n il_nch il_ch_branch rad_nel rad_nt rad_z cont_slots cont_i32"
    ).split(),
    np.int16: "il_ch_code seg_hard_ch seg_rad_Z".split(),
    np.int8: "seg_event exit_code".split(),
}
# ELSEPA, midpoint, SBETHE, straggling, shell, coupled radiative, continuation;
# and Mott, midpoint, straggling.
_SWITCHES = [(2, 1, 0, 1, 1, 1, 1, 1), (1, 1, 0, 0, 1, 0, 0, 0)]


def _transpile(switches, precision):
    from cupyx.jit import _compile as jit_compile
    from cupyx.jit import _cuda_types

    from pyrite.montecarlo.transport import _jit_kernel_variants as variants
    from pyrite.montecarlo.transport._jit_precision import MIXED_TABLES

    kernel = variants.transport_kernel_variant(
        variants.TransportKernelSwitches(*switches), precision
    )
    code = kernel._func.__code__
    in_types = []
    for name in code.co_varnames[: code.co_argcount]:
        scalar = next((t for t, names in _SCALARS.items() if name in names), None)
        if scalar is not None:
            in_types.append(_cuda_types.Scalar(scalar))
            continue
        dtype = next((t for t, names in _ARRAYS.items() if name in names), np.float64)
        if precision == "mixed" and name in MIXED_TABLES:
            dtype = np.float32
        in_types.append(_cuda_types.CArray(np.dtype(dtype), 1, True, True))
    return jit_compile.transpile(
        kernel._func, ['extern "C"', "__global__"], kernel._mode, tuple(in_types), _cuda_types.void
    ).code


def _device_functions(code):
    """``{base name: (signature, body)}`` of every generated device function."""
    out = {}
    for match in re.finditer(r"^__device__ [^(]*? (\w+)\((.*?)\) \{", code, re.M):
        depth, end = 1, match.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(code[end], 0)
            end += 1
        out[re.sub(r"_\d+$", "", match.group(1))] = (match.group(0), code[match.end() : end])
    return out


@requires_cupy
@pytest.mark.parametrize("switches", _SWITCHES)
def test_mixed_helpers_compute_in_fp32_and_fp64_build_never_narrows(switches):
    from pyrite.montecarlo.transport._jit_precision import MIXED_HELPERS

    fp64 = _transpile(switches, "fp64")
    mixed = _transpile(switches, "mixed")
    # `table_real`/`math_real` are identity casts in the default build.
    assert "(float)" not in fp64 and "<float" not in fp64
    helpers = {name for names in MIXED_HELPERS.values() for name in names}
    functions = _device_functions(mixed)
    called = helpers & functions.keys()
    assert called
    for name in called:
        signature, body = functions[name]
        assert "double" not in signature + body, name
    # Helpers outside the set keep FP64: no draw or stopping power is narrowed.
    assert functions["_stream_uniform"][0].startswith("__device__ double")
    body = mixed.split("__global__")[1]
    assert "log((float)" in body
    assert "log(_stream_uniform" in body  # free-path draws stay FP64


@requires_cupy
def test_fp64_precision_adds_no_globals():
    from pyrite.montecarlo.transport import _jit_kernel
    from pyrite.montecarlo.transport._jit_precision import fp32_twins, precision_globals

    assert precision_globals("fp64") == {}
    assert _jit_kernel.table_real is np.float64 and _jit_kernel.math_real is np.float64
    twins = fp32_twins()
    for twin in twins.values():
        namespace = twin._func.__globals__
        assert not any(isinstance(value, np.float64) for value in namespace.values())
        for name, other in twins.items():
            if name in namespace:
                assert namespace[name] is other
    with pytest.raises(ValueError, match="transport precision"):
        precision_globals("fp32")


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"transport_core": "per-electron"}, "requires transport_core='cuda'"),
        ({"transport_precision": "fp32"}, "transport_precision must be one of"),
    ],
)
def test_mixed_precision_is_refused_off_the_cuda_core(kw, message):
    with pytest.raises(ValueError, match=message):
        simulate_trajectories(
            E0_keV=30.0,
            Ne=4,
            thickness_ang=1.0e4,
            element="C",
            n_atoms_per_ang3=0.176,
            elastic_model="sr",
            radiative_model="uncoupled",
            **({"transport_precision": "mixed", "transport_core": "cuda"} | kw),
        )


def _cases():
    from tests.montecarlo.test_transport_continuation_cuda import (
        NO_LUT,
        _elsepa,
        _radiative,
        _shell,
    )
    from tests.montecarlo.test_transport_kernel_variants_cuda import _sbethe_elsepa

    return {
        "elsepa-midpoint": (_elsepa, {"Ne": 400}),
        "mott": (_elsepa, {"elastic_model": "mott", "transport_lut_config": NO_LUT, "Ne": 200}),
        "sr-radiative": (_radiative, {}),
        "shell-all": (_shell, {"secondaries": True, "straggling": True, "radiative": True}),
        "elsepa-sbethe-radiative": (_sbethe_elsepa, {"Ne": 300}),
    }


CASE_IDS = ["elsepa-midpoint", "mott", "sr-radiative", "shell-all", "elsepa-sbethe-radiative"]


def _host(value):
    return cupy.asnumpy(value) if isinstance(value, cupy.ndarray) else value


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("case", CASE_IDS)
def test_mixed_tracks_fp64_at_rounding_scale(case):
    from tests.montecarlo.test_transport_continuation_cuda import REFERENCE

    run, kw = _cases()[case]
    fp64 = run(REFERENCE, **kw)
    mixed = run(REFERENCE, transport_precision="mixed", **kw)
    for name in ("L_ang", "E_end_keV"):
        values = _host(mixed[name])
        assert np.isfinite(values).all(), name
    # The mixed kernel ran: trajectories diverge at rounding scale.
    assert not np.array_equal(_host(fp64["L_ang"])[:64], _host(mixed["L_ang"])[:64])
    ne = kw.get("Ne", 200)
    for name in ("n_transmitted", "n_backscattered"):
        # A few histories may change outcome; a gross bias would move many.
        assert abs(int(mixed[name]) - int(fp64[name])) <= max(3, ne // 20), name
    path = float(np.sum(_host(fp64["L_ang"])))
    assert abs(float(np.sum(_host(mixed["L_ang"]))) - path) <= 0.05 * path


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("case", ["elsepa-sbethe-radiative", "shell-all"])
def test_mixed_continuation_matches_full_history(case):
    from tests.montecarlo.test_transport_continuation_cuda import (
        REFERENCE,
        _assert_identical,
        _continued,
    )

    run, kw = _cases()[case]
    kw = kw | {"transport_precision": "mixed"}
    _assert_identical(run(REFERENCE, **kw), run(_continued(7, 37), **kw))


@pytest.mark.hardware
@requires_cuda
def test_mixed_precision_needs_the_specialized_kernel(monkeypatch):
    from pyrite.montecarlo.transport import _jit_launch
    from tests.montecarlo.test_transport_continuation_cuda import REFERENCE, _elsepa

    monkeypatch.setattr(_jit_launch, "_SPECIALIZE_TRANSPORT_KERNEL", False)
    with pytest.raises(ValueError, match="specialized kernel"):
        _elsepa(REFERENCE, Ne=8, transport_precision="mixed")
