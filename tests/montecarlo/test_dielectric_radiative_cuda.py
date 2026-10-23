"""Device/Numba parity for the opt-in 30--100 MeV dielectric BremsLib mode.

The synthetic SDCS avoids remote tables. These tests require a CUDA GPU: a
CPU-only CI runner skips them rather than claiming device validation.

Issue #439; numerical/implementation validation, not physical validation.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
try:
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required", allow_module_level=True)
except cupy.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

from cupyx import jit  # noqa: E402

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table  # noqa: E402
from pyrite.montecarlo.transport._jit_radiative import (  # noqa: E402
    radiative_layer_moments_scalar,
    radiative_moments_scalar,
    sample_hard_photon_energy_scalar,
)
from pyrite.montecarlo.transport._jit_radiative_device import (  # noqa: E402
    RAD_HARD_RATE,
    RAD_SOFT,
    _rad_layer_moment,
    _rad_moment,
    _rad_sample_photon_eV,
)
from pyrite.montecarlo.transport.hard_radiative import pack_radiative_layer_tables  # noqa: E402
from tests.helpers.bremslib import synthetic_bremslib_arrays  # noqa: E402


@jit.rawkernel()
def _probe_atom(incident, nominal, top, chi, energies, cutoffs, plasma, uniforms, n_cases, out):
    i = jit.grid(1)
    if i >= n_cases:
        return
    energy = energies[i]
    cutoff = cutoffs[i]
    ep = plasma[i]
    out[3 * i] = _rad_moment(
        RAD_SOFT,
        incident,
        nominal,
        top,
        chi,
        np.int32(0),
        np.int32(incident.size),
        np.int32(nominal.size),
        np.int32(incident.size),
        np.float64(14.0),
        energy,
        cutoff,
        ep,
    )
    out[3 * i + 1] = _rad_moment(
        RAD_HARD_RATE,
        incident,
        nominal,
        top,
        chi,
        np.int32(0),
        np.int32(incident.size),
        np.int32(nominal.size),
        np.int32(incident.size),
        np.float64(14.0),
        energy,
        cutoff,
        ep,
    )
    out[3 * i + 2] = _rad_sample_photon_eV(
        incident,
        nominal,
        top,
        chi,
        np.int32(0),
        np.int32(incident.size),
        np.int32(nominal.size),
        np.int32(incident.size),
        np.float64(14.0),
        energy,
        cutoff,
        uniforms[i],
        ep,
    )


@jit.rawkernel()
def _probe_layers(
    nel,
    nt,
    z,
    ncm3,
    incident,
    nominal,
    top,
    chi,
    plasma,
    max_el,
    max_t,
    n_red,
    energy,
    cutoff,
    n_layers,
    out,
):
    # The device helpers take an int32 layer; jit.grid is uint32.
    if jit.grid(1) != 0:
        return
    for layer in range(n_layers):
        out[2 * layer] = _rad_layer_moment(
            RAD_SOFT,
            layer,
            nel,
            nt,
            z,
            ncm3,
            incident,
            nominal,
            top,
            chi,
            max_el,
            max_t,
            n_red,
            energy,
            cutoff,
            plasma,
        )
        out[2 * layer + 1] = _rad_layer_moment(
            RAD_HARD_RATE,
            layer,
            nel,
            nt,
            z,
            ncm3,
            incident,
            nominal,
            top,
            chi,
            max_el,
            max_t,
            n_red,
            energy,
            cutoff,
            plasma,
        )


@pytest.fixture(scope="module")
def tables():
    grid = np.array([10.0, 30.0, 60.0, 100.0])
    return {
        symbol: prepare_bremslib_table(
            synthetic_bremslib_arrays(t1_MeV=grid),
            atomic_number=z,
            key=f"synthetic/{symbol}",
        )
        for symbol, z in (("Si", 14), ("O", 8))
    }


def test_cuda_dielectric_atom_moments_and_quantiles(tables):
    table = tables["Si"]
    cases = [
        (T, c, p, u)
        for T in (30e6, 100e6)
        for c in (100.0, 1000.0, 10000.0)
        for p in (0.0, 31.0, 75.0)
        for u in (0.0, 0.02, 0.5, 0.98)
    ]
    incident = table.incident_energy_keV
    nominal = table.nominal_reduced_energy
    top = table.top_reduced_energy
    chi = table.scaled_sdcs_mb
    inputs = np.asarray(cases, dtype=np.float64)
    out = cupy.empty(3 * len(cases), dtype=cupy.float64)
    _probe_atom(
        (1,),
        (128,),
        (
            cupy.asarray(incident),
            cupy.asarray(nominal),
            cupy.asarray(top),
            cupy.asarray(chi.reshape(-1)),
            *(cupy.asarray(inputs[:, i]) for i in range(4)),
            np.int32(len(cases)),
            out,
        ),
    )
    actual = cupy.asnumpy(out).reshape(-1, 3)
    for i, (energy, cutoff, plasma, uniform) in enumerate(cases):
        expected = radiative_moments_scalar(incident, nominal, top, chi, 14, energy, cutoff, plasma)
        sampled = sample_hard_photon_energy_scalar(
            incident, nominal, top, chi, 14, energy, cutoff, uniform, plasma
        )
        np.testing.assert_allclose(actual[i, :2], expected[:2], rtol=3e-7, atol=1e-30)
        np.testing.assert_allclose(actual[i, 2], sampled, rtol=3e-7, atol=5e-5)
        assert cutoff <= actual[i, 2] <= energy


def test_cuda_layer_plasma_is_selected_per_material(tables):
    compositions = [[("Si", 0.05)], [("Si", 0.025), ("O", 0.050)]]
    packed = pack_radiative_layer_tables(compositions, tables, 10e6, 100e6, dielectric=True)
    nel, nt, z, ncm3, incident, nominal, top, chi, plasma = packed
    assert plasma.shape == (2,) and not np.isclose(plasma[0], plasma[1])
    for energy in (30e6, 100e6):
        out = cupy.empty(2 * len(compositions), dtype=cupy.float64)
        _probe_layers(
            (1,),
            (32,),
            (
                cupy.asarray(nel),
                cupy.asarray(nt.reshape(-1)),
                cupy.asarray(z.reshape(-1)),
                cupy.asarray(ncm3.reshape(-1)),
                cupy.asarray(incident.reshape(-1)),
                cupy.asarray(nominal.reshape(-1)),
                cupy.asarray(top.reshape(-1)),
                cupy.asarray(chi.reshape(-1)),
                cupy.asarray(plasma),
                np.int32(z.shape[1]),
                np.int32(incident.shape[2]),
                np.int32(nominal.shape[2]),
                np.float64(energy),
                np.float64(1000.0),
                np.int32(len(compositions)),
                out,
            ),
        )
        device = cupy.asnumpy(out).reshape(-1, 2)
        for layer in range(2):
            rates = np.zeros(z.shape[1])
            expected = radiative_layer_moments_scalar(
                nel,
                nt,
                z,
                ncm3,
                incident,
                nominal,
                top,
                chi,
                layer,
                energy,
                1000.0,
                rates,
                plasma,
            )
            np.testing.assert_allclose(device[layer], expected, rtol=3e-7, atol=1e-30)
