"""Pin line_index / line_quality / line_metrics outputs on synthetic spectra.

Guards the peak-finding fast path (issue #232): optimizations must leave every
value byte-identical. Expected values were recorded from the pre-optimization
implementation; they are exact (==) on purpose."""

import numpy as np
import pytest

from pyrite.results import Settings, line_index, line_metrics, line_quality

_E = np.linspace(1000.0, 5000.0, 400)


def _gauss(c, w, a=1.0):
    return a * np.exp(-(((_E - c) / w) ** 2))


def _spectra():
    rng = np.random.default_rng(232)
    out = {}
    out["noisy_many_maxima"] = np.abs(rng.normal(0.0, 1.0, _E.size)) + _gauss(2500.0, 30.0, 6.0)
    out["noisy_only"] = rng.random(_E.size)
    out["zero"] = np.zeros_like(_E)
    out["flat"] = np.full_like(_E, 2.0)
    out["pedestal_small_bump"] = _gauss(3000.0, 900.0, 10.0) + _gauss(2200.0, 15.0, 0.6)
    out["comparable_peaks"] = (
        _gauss(1800.0, 20.0, 1.0)
        + _gauss(2600.0, 20.0, 0.97)
        + _gauss(3400.0, 45.0, 0.95)
        + _gauss(4200.0, 12.0, 0.9)
    )
    out["sharp_beats_tall"] = _gauss(2000.0, 8.0, 0.7) + _gauss(3500.0, 120.0, 1.0)
    out["with_negative"] = _gauss(2500.0, 25.0, 1.0) - 0.2 + 0.01 * rng.random(_E.size)
    return out


def _record(spec, seed):
    brem = 0.05 + 0.02 * np.random.default_rng(seed).random(_E.size)
    return {"E_grid": _E, "spec": spec, "brem": brem, "scale": 2.5}


nan = float("nan")
_METRICS = ("sharpness", "prominence")


def _observed():
    got = {}
    for i, (name, spec) in enumerate(_spectra().items()):
        for metric in _METRICS:
            m = line_metrics(_record(spec, i), Settings(), metric=metric)
            got[(name, metric)] = (
                line_index(spec, metric=metric),
                line_quality(spec),
                {k: m[k] for k in sorted(m) if k != "hit_frac"},
            )
    return got


EXPECTED = {
    ("noisy_many_maxima", "sharpness"): (
        15,
        0.052093292487182,
        {
            "coherent_brem_ratio": 14.819178785265905,
            "coherent_flux": 44912.807636349244,
            "coherent_flux_per_na": 8982.561527269849,
            "fwhm_eV": 10.795662936350936,
            "line_brem_ratio": 18.870798491960542,
            "line_eV": 1150.375939849624,
            "line_flux": 465.1952615325665,
            "line_flux_per_na": 93.0390523065133,
            "line_frac": 0.009702983225732851,
            "line_quality": 0.052093292487182,
            "peak_flux": 100.05867085058642,
            "peak_flux_per_na": 20.011734170117286,
            "peak_sample_spacing_eV": 10.025062656641694,
            "peak_spectral_flux_density": 100.05867085058642,
            "peak_spectral_flux_density_per_na": 20.011734170117286,
            "total_flux": 47943.52939813838,
            "total_flux_per_na": 9588.705879627676,
        },
    ),
    ("noisy_many_maxima", "prominence"): (
        150,
        0.052093292487182,
        {
            "coherent_brem_ratio": 14.819178785265905,
            "coherent_flux": 44912.807636349244,
            "coherent_flux_per_na": 8982.561527269849,
            "fwhm_eV": 41.594099914374056,
            "line_brem_ratio": 55.60910060330514,
            "line_eV": 2503.7593984962405,
            "line_flux": 5180.607575112358,
            "line_flux_per_na": 1036.1215150224716,
            "line_frac": 0.10805644974717941,
            "line_quality": 0.052093292487182,
            "peak_flux": 100.05867085058642,
            "peak_flux_per_na": 20.011734170117286,
            "peak_sample_spacing_eV": 10.025062656641467,
            "peak_spectral_flux_density": 100.05867085058642,
            "peak_spectral_flux_density_per_na": 20.011734170117286,
            "total_flux": 47943.52939813838,
            "total_flux_per_na": 9588.705879627676,
        },
    ),
    ("noisy_only", "sharpness"): (
        166,
        0.013688165560833269,
        {
            "coherent_brem_ratio": 8.52291933638893,
            "coherent_flux": 25456.38983630064,
            "coherent_flux_per_na": 5091.277967260128,
            "fwhm_eV": 10.398101873464384,
            "line_brem_ratio": 6.1615011263193304,
            "line_eV": 2664.160401002506,
            "line_flux": 145.35696136660812,
            "line_flux_per_na": 29.07139227332162,
            "line_frac": 0.005110428247961974,
            "line_quality": 0.013688165560833269,
            "peak_flux": 12.464388732406047,
            "peak_flux_per_na": 2.4928777464812093,
            "peak_sample_spacing_eV": 10.025062656641694,
            "peak_spectral_flux_density": 12.464388732406047,
            "peak_spectral_flux_density_per_na": 2.4928777464812093,
            "total_flux": 28443.20560113061,
            "total_flux_per_na": 5688.6411202261215,
        },
    ),
    ("noisy_only", "prominence"): (
        293,
        0.013688165560833269,
        {
            "coherent_brem_ratio": 8.52291933638893,
            "coherent_flux": 25456.38983630064,
            "coherent_flux_per_na": 5091.277967260128,
            "fwhm_eV": 30.14439621005795,
            "line_brem_ratio": 10.31250961190769,
            "line_eV": 3937.34335839599,
            "line_flux": 699.9518714611886,
            "line_flux_per_na": 139.99037429223773,
            "line_frac": 0.02460875476825178,
            "line_quality": 0.013688165560833269,
            "peak_flux": 12.464388732406047,
            "peak_flux_per_na": 2.4928777464812093,
            "peak_sample_spacing_eV": 10.025062656641467,
            "peak_spectral_flux_density": 12.464388732406047,
            "peak_spectral_flux_density_per_na": 2.4928777464812093,
            "total_flux": 28443.20560113061,
            "total_flux_per_na": 5688.6411202261215,
        },
    ),
    ("zero", "sharpness"): (
        0,
        0.0,
        {
            "coherent_brem_ratio": 0.0,
            "coherent_flux": 0.0,
            "coherent_flux_per_na": 0.0,
            "fwhm_eV": 0.0,
            "line_brem_ratio": nan,
            "line_eV": 1000.0,
            "line_flux": 0.0,
            "line_flux_per_na": 0.0,
            "line_frac": 0.0,
            "line_quality": 0.0,
            "peak_flux": 0.0,
            "peak_flux_per_na": 0.0,
            "peak_sample_spacing_eV": 10.02506265664158,
            "peak_spectral_flux_density": 0.0,
            "peak_spectral_flux_density_per_na": 0.0,
            "total_flux": 2996.190537685393,
            "total_flux_per_na": 599.2381075370786,
        },
    ),
    ("zero", "prominence"): (
        0,
        0.0,
        {
            "coherent_brem_ratio": 0.0,
            "coherent_flux": 0.0,
            "coherent_flux_per_na": 0.0,
            "fwhm_eV": 0.0,
            "line_brem_ratio": nan,
            "line_eV": 1000.0,
            "line_flux": 0.0,
            "line_flux_per_na": 0.0,
            "line_frac": 0.0,
            "line_quality": 0.0,
            "peak_flux": 0.0,
            "peak_flux_per_na": 0.0,
            "peak_sample_spacing_eV": 10.02506265664158,
            "peak_spectral_flux_density": 0.0,
            "peak_spectral_flux_density_per_na": 0.0,
            "total_flux": 2996.190537685393,
            "total_flux_per_na": 599.2381075370786,
        },
    ),
    ("flat", "sharpness"): (
        0,
        0.0,
        {
            "coherent_brem_ratio": 33.25011404989377,
            "coherent_flux": 100000.0,
            "coherent_flux_per_na": 20000.0,
            "fwhm_eV": 0.0,
            "line_brem_ratio": nan,
            "line_eV": 1000.0,
            "line_flux": 0.0,
            "line_flux_per_na": 0.0,
            "line_frac": 0.0,
            "line_quality": 0.0,
            "peak_flux": 25.0,
            "peak_flux_per_na": 5.0,
            "peak_sample_spacing_eV": 10.02506265664158,
            "peak_spectral_flux_density": 25.0,
            "peak_spectral_flux_density_per_na": 5.0,
            "total_flux": 103007.50848102187,
            "total_flux_per_na": 20601.501696204374,
        },
    ),
    ("flat", "prominence"): (
        0,
        0.0,
        {
            "coherent_brem_ratio": 33.25011404989377,
            "coherent_flux": 100000.0,
            "coherent_flux_per_na": 20000.0,
            "fwhm_eV": 0.0,
            "line_brem_ratio": nan,
            "line_eV": 1000.0,
            "line_flux": 0.0,
            "line_flux_per_na": 0.0,
            "line_frac": 0.0,
            "line_quality": 0.0,
            "peak_flux": 25.0,
            "peak_flux_per_na": 5.0,
            "peak_sample_spacing_eV": 10.02506265664158,
            "peak_spectral_flux_density": 25.0,
            "peak_spectral_flux_density_per_na": 5.0,
            "total_flux": 103007.50848102187,
            "total_flux_per_na": 20601.501696204374,
        },
    ),
    ("pedestal_small_bump", "sharpness"): (
        120,
        0.0,
        {
            "coherent_brem_ratio": 65.57302593634556,
            "coherent_flux": 199266.58953243753,
            "coherent_flux_per_na": 39853.31790648751,
            "fwhm_eV": 17.540110826055297,
            "line_brem_ratio": 78.21138992229882,
            "line_eV": 2203.0075187969924,
            "line_flux": 3197.542007123579,
            "line_flux_per_na": 639.5084014247158,
            "line_frac": 0.015805516720628996,
            "line_quality": 0.0,
            "peak_flux": 124.99612267274125,
            "peak_flux_per_na": 24.99922453454825,
            "peak_sample_spacing_eV": 10.025062656641467,
            "peak_spectral_flux_density": 124.99612267274125,
            "peak_spectral_flux_density_per_na": 24.99922453454825,
            "total_flux": 202305.43952734052,
            "total_flux_per_na": 40461.087905468106,
        },
    ),
    ("pedestal_small_bump", "prominence"): (
        199,
        0.0,
        {
            "coherent_brem_ratio": 65.57302593634556,
            "coherent_flux": 199266.58953243753,
            "coherent_flux_per_na": 39853.31790648751,
            "fwhm_eV": 1490.898008882748,
            "line_brem_ratio": 65.57302593634556,
            "line_eV": 2994.9874686716794,
            "line_flux": 199266.58953243753,
            "line_flux_per_na": 39853.31790648751,
            "line_frac": 0.9849789012000723,
            "line_quality": 0.0,
            "peak_flux": 124.99612267274125,
            "peak_flux_per_na": 24.99922453454825,
            "peak_sample_spacing_eV": 10.025062656641694,
            "peak_spectral_flux_density": 124.99612267274125,
            "peak_spectral_flux_density_per_na": 24.99922453454825,
            "total_flux": 202305.43952734052,
            "total_flux_per_na": 40461.087905468106,
        },
    ),
    ("comparable_peaks", "sharpness"): (
        319,
        0.2423959029274388,
        {
            "coherent_brem_ratio": 0.6891861743859061,
            "coherent_flux": 2059.3699247912164,
            "coherent_flux_per_na": 411.8739849582432,
            "fwhm_eV": 20.93691906305139,
            "line_brem_ratio": 5.0917947530827,
            "line_eV": 4197.994987468672,
            "line_flux": 239.06444623835537,
            "line_flux_per_na": 47.81288924767107,
            "line_frac": 0.047363050367927435,
            "line_quality": 0.2423959029274388,
            "peak_flux": 12.375001828116108,
            "peak_flux_per_na": 2.4750003656232216,
            "peak_sample_spacing_eV": 10.02506265664124,
            "peak_spectral_flux_density": 12.375001828116108,
            "peak_spectral_flux_density_per_na": 2.4750003656232216,
            "total_flux": 5047.488377147289,
            "total_flux_per_na": 1009.4976754294578,
        },
    ),
    ("comparable_peaks", "prominence"): (
        80,
        0.2423959029274388,
        {
            "coherent_brem_ratio": 0.6891861743859061,
            "coherent_flux": 2059.3699247912164,
            "coherent_flux_per_na": 411.8739849582432,
            "fwhm_eV": 33.84778715623247,
            "line_brem_ratio": 5.510716685058445,
            "line_eV": 1802.0050125313282,
            "line_flux": 442.84801887694977,
            "line_flux_per_na": 88.56960377538995,
            "line_frac": 0.08773631275348,
            "line_quality": 0.2423959029274388,
            "peak_flux": 12.375001828116108,
            "peak_flux_per_na": 2.4750003656232216,
            "peak_sample_spacing_eV": 10.025062656641694,
            "peak_spectral_flux_density": 12.375001828116108,
            "peak_spectral_flux_density_per_na": 2.4750003656232216,
            "total_flux": 5047.488377147289,
            "total_flux_per_na": 1009.4976754294578,
        },
    ),
    ("sharp_beats_tall", "sharpness"): (
        100,
        0.30646947626526044,
        {
            "coherent_brem_ratio": 0.9246951400879193,
            "coherent_flux": 2782.7525459186645,
            "coherent_flux_per_na": 556.5505091837329,
            "fwhm_eV": 14.75267827229095,
            "line_brem_ratio": 3.715185955037069,
            "line_eV": 2002.5062656641603,
            "line_flux": 123.8367443424413,
            "line_flux_per_na": 24.767348868488263,
            "line_frac": 0.021380190109713093,
            "line_quality": 0.30646947626526044,
            "peak_flux": 12.48773772242097,
            "peak_flux_per_na": 2.497547544484194,
            "peak_sample_spacing_eV": 10.025062656641694,
            "peak_spectral_flux_density": 12.48773772242097,
            "peak_spectral_flux_density_per_na": 2.497547544484194,
            "total_flux": 5792.1255006138545,
            "total_flux_per_na": 1158.4251001227708,
        },
    ),
    ("sharp_beats_tall", "prominence"): (
        249,
        0.30646947626526044,
        {
            "coherent_brem_ratio": 0.9246951400879193,
            "coherent_flux": 2782.7525459186645,
            "coherent_flux_per_na": 556.5505091837329,
            "fwhm_eV": 200.04646728356147,
            "line_brem_ratio": 5.902548380694481,
            "line_eV": 3496.240601503759,
            "line_flux": 2657.571410997003,
            "line_flux_per_na": 531.5142821994007,
            "line_frac": 0.4588249012759396,
            "line_quality": 0.30646947626526044,
            "peak_flux": 12.48773772242097,
            "peak_flux_per_na": 2.497547544484194,
            "peak_sample_spacing_eV": 10.025062656641467,
            "peak_spectral_flux_density": 12.48773772242097,
            "peak_spectral_flux_density_per_na": 2.497547544484194,
            "total_flux": 5792.1255006138545,
            "total_flux_per_na": 1158.4251001227708,
        },
    ),
    ("with_negative", "sharpness"): (
        150,
        1.1204592134521372,
        {
            "coherent_brem_ratio": -3.060747453367758,
            "coherent_flux": -9194.209725969806,
            "coherent_flux_per_na": -1838.8419451939612,
            "fwhm_eV": 42.687380621085595,
            "line_brem_ratio": 2.4407840667571974,
            "line_eV": 2503.7593984962405,
            "line_flux": 242.11697989830589,
            "line_flux_per_na": 48.42339597966118,
            "line_frac": nan,
            "line_quality": 1.1204592134521372,
            "peak_flux": 9.844109491440198,
            "peak_flux_per_na": 1.9688218982880397,
            "peak_sample_spacing_eV": 10.025062656641467,
            "peak_spectral_flux_density": 9.844109491440198,
            "peak_spectral_flux_density_per_na": 1.9688218982880397,
            "total_flux": -6190.299777158654,
            "total_flux_per_na": -1238.059955431731,
        },
    ),
    ("with_negative", "prominence"): (
        150,
        1.1204592134521372,
        {
            "coherent_brem_ratio": -3.060747453367758,
            "coherent_flux": -9194.209725969806,
            "coherent_flux_per_na": -1838.8419451939612,
            "fwhm_eV": 42.687380621085595,
            "line_brem_ratio": 2.4407840667571974,
            "line_eV": 2503.7593984962405,
            "line_flux": 242.11697989830589,
            "line_flux_per_na": 48.42339597966118,
            "line_frac": nan,
            "line_quality": 1.1204592134521372,
            "peak_flux": 9.844109491440198,
            "peak_flux_per_na": 1.9688218982880397,
            "peak_sample_spacing_eV": 10.025062656641467,
            "peak_spectral_flux_density": 9.844109491440198,
            "peak_spectral_flux_density_per_na": 1.9688218982880397,
            "total_flux": -6190.299777158654,
            "total_flux_per_na": -1238.059955431731,
        },
    ),
}


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("name", list(_spectra()))
def test_pinned_line_outputs(name, metric):
    idx, quality, metrics = _observed()[(name, metric)]
    e_idx, e_quality, e_metrics = EXPECTED[(name, metric)]
    assert idx == e_idx
    assert quality == e_quality
    # repr comparison: exact floats, and NaN == NaN
    assert repr(metrics) == repr(e_metrics)
    assert metrics["line_quality"] == quality


def test_top_geometries_shares_line_metrics_cache(monkeypatch):
    from pyrite.results import metrics as _metrics
    from pyrite.results import top_geometries

    store = {}
    for i, amp in enumerate((1.0, 2.0, 3.0)):
        rec = _record(_gauss(2500.0 + 100.0 * i, 25.0, amp), i)
        rec["case"] = {
            "name": f"m{i}",
            "E0_keV": 30.0,
            "tilt_deg": 10.0 * i,
            "tilt_azim_deg": 0.0,
            "thickness_ang": 1.0e4,
        }
        store.setdefault(f"m{i}", {})[30.0] = rec
    _metrics._LINE_METRICS_CACHE.clear()
    real = _metrics.line_metrics
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_metrics, "line_metrics", counting)
    first = top_geometries(store, Settings())
    second = top_geometries(store, Settings())
    assert calls["n"] == 3
    assert first.equals(second)
    _metrics._LINE_METRICS_CACHE.clear()
