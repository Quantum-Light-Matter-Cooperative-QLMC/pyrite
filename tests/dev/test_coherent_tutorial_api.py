"""Issue #777: public coherent tutorial compatibility contracts (no tables)."""

from dataclasses import replace

import numpy as np
import pytest


def test_temporal_intensity_has_common_time_axis_and_allows_signed_bins():
    from pyrite.results import TemporalIntensity

    p = TemporalIntensity(
        time_fs=np.array([0.0, 0.2, 0.4]),
        incoherent=np.array([1.0, 2.0, 3.0]),
        coherent=np.array([2.0, -0.1, 4.0]),
    )
    assert p.coherent is not None and p.coherent[1] < 0
    assert p.time_fs.shape == p.incoherent.shape == p.coherent.shape
    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalIntensity([0, 0, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="match time_fs"):
        TemporalIntensity([0, 1, 2], [1, 2])


def test_microtrain_is_public_and_accepted_by_beam():
    from pyrite.beams import Beam, LongitudinalDistribution, PointSpot
    from pyrite.campaign.longitudinal import LongitudinalDistribution as StoredDistribution
    from pyrite.units import ureg

    assert LongitudinalDistribution is StoredDistribution
    policy = LongitudinalDistribution(kind="microtrain", envelope_rms_fs=1.0)
    beam = Beam(kinetic_energy=30 * ureg.keV, transverse=PointSpot(), longitudinal=policy)
    assert beam.to_spec().longitudinal is policy
    assert replace(beam, longitudinal=policy).longitudinal is policy


def test_temporal_opt_in_validates_bool_and_keeps_default_off():
    from pyrite.config import Numerics

    assert Numerics().temporal_profile is False
    assert Numerics(temporal_profile=True).temporal_profile is True
    with pytest.raises(TypeError, match="temporal_profile"):
        Numerics(temporal_profile=1)


def test_result_accepts_opt_in_profile_without_changing_spectral_contract():
    from pyrite.results import CxrSpectra, Result, SpectralDensity, TemporalIntensity

    spectrum = SpectralDensity(np.array([100., 200.]), np.array([1., 2.]))
    profile = TemporalIntensity(np.array([0., 1.]), np.array([1., 2.]))
    kwargs = dict(cxr=CxrSpectra(spectrum), brem=spectrum, case={}, provenance={})
    assert Result(**kwargs).temporal is None
    assert Result(**kwargs, temporal=profile).temporal is profile
    with pytest.raises(TypeError, match="TemporalIntensity"):
        Result(**kwargs, temporal=object())
