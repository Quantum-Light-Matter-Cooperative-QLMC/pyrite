"""CUDA-gated CPU-vs-device parity for the coherent line spectrum (#298).

The backend is fixed at import, so each backend runs this file as a child
script and writes its spectrum; the test compares them. The input puts segment
midpoints across a finite footprint, so pieces reach and pass its side faces --
where the compiled CPU prism exit once disagreed with the device array path.
Coherent spectra amplify that escape-path gap through the propagation phase.

Validation: finite-transverse-crystal
"""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = pytest.mark.skipif(_NDEV < 1, reason="CUDA device required")

# fp64 device and host evaluate the same float64 formulas; only reduction
# order differs.
FP64_PEAK_TOL = 1e-9
# float32 device: each segment phase ~ |g.r| + omega |d| <= ~5e4 rad here
# rounds by eps32 ~ 6e-8 relative, ~3e-3 rad, and the coherent sum carries an
# error of that order relative to peak. The bound scales with the largest
# phase: midpoints at z ~ 1e6 Ang (g.r ~ 2e6 rad) reach ~5e-2.
FP32_PEAK_TOL = 1e-2


def _segments():
    from pyrite.montecarlo.transport import beta_from_keV

    rng = np.random.default_rng(0)
    n, ne, length, energy = 400, 10, 500.0, 30.0
    tilt = 0.05 * rng.normal(size=(n, 2))
    v_hat = np.column_stack([tilt, np.ones(n)])
    v_hat /= np.linalg.norm(v_hat, axis=1)[:, None]
    start = np.tile(np.arange(n // ne), ne) * length
    r_mid = np.column_stack(
        [rng.uniform(-5e3, 5e3, n), rng.uniform(-5e3, 5e3, n), rng.uniform(0.0, 2e4, n)]
    )
    return {
        "r_mid": r_mid,
        "v_hat": v_hat,
        "L_ang": np.full(n, length),
        "E_keV": np.full(n, energy),
        "t_ang": start / beta_from_keV(energy),
        "elec_id": np.repeat(np.arange(ne), n // ne),
        "layer": np.zeros(n, dtype=int),
        "Ne": ne,
        "thickness_ang": 1e6,
        "crystal_width_ang": 1e4,
        "crystal_height_ang": 1e4,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _spectrum():
    from pyrite.montecarlo.spectrum.lines import mc_spectrum

    n_hat = np.array([1.0, 0.0, 0.01]) / np.linalg.norm([1.0, 0.0, 0.01])
    return mc_spectrum(
        _segments(),
        np.arange(700.0, 1500.0, 0.5),
        "hopg",
        [(0, 0, 2)],
        B_ang2=0.8,
        n_hat=n_hat,
        coherent=True,
    )


def _run(tmp_path, name, **env_overrides):
    env = dict(os.environ)
    env.pop("PYRITE_TEST_BACKEND", None)
    env.pop("PYRITE_FP64", None)
    env.update(env_overrides)
    out = tmp_path / f"{name}.npy"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, __file__, str(out)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(pathlib.Path(__file__).resolve().parents[2]),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return np.load(out)


@pytest.mark.slow
def test_coherent_spectrum_matches_across_backends(tmp_path):
    host = _run(tmp_path, "cpu", PYRITE_MC_BACKEND="cpu")
    peak = np.max(np.abs(host))
    assert peak > 0.0
    fp64 = _run(tmp_path, "cuda64", PYRITE_MC_BACKEND="cuda", PYRITE_FP64="1")
    assert np.max(np.abs(fp64 - host)) / peak <= FP64_PEAK_TOL
    fp32 = _run(tmp_path, "cuda32", PYRITE_MC_BACKEND="cuda")
    assert np.max(np.abs(fp32 - host)) / peak <= FP32_PEAK_TOL


if __name__ == "__main__":
    np.save(sys.argv[1], np.asarray(_spectrum(), dtype=np.float64))
