"""A bounded CPU counting example; simulated data, not a detector calibration."""

# tutorial-start
import matplotlib.pyplot as plt
import numpy as np

from pyrite import Beam, Numerics, Scene, simulate
from pyrite.beams import PointBunch, PointSpot
from pyrite.config import AngularSampling, CxrReflections, FixedSampling
from pyrite.detectors import IdealPhotonCounter, PixelGrid, PlanarDetector, PlanarPose
from pyrite.geometry import Footprint, Slab
from pyrite.observations import Acquisition, Observation


def calculate():
    beam = Beam(
        energy_keV=30.0,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
        rep_rate_hz=1_000_000.0,
        bunch_charge_pc=0.01,
    )
    panel = PlanarDetector(
        PlanarPose.from_observation(distance_mm=100.0, polar_deg=135.0),
        pixels=PixelGrid(shape=(16, 16), pitch_mm=(1.0, 1.0)),
    )
    scene = Scene(
        beam,
        Slab("hopg", 1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        panel,
    )
    observation = Observation(
        detector="detector",
        response=IdealPhotonCounter(),
        sampling=AngularSampling((4, 4)),
        acquisition=Acquisition(
            exposure_s=1.0,
            measured_edges_eV=(500.0, 4_000.0),
            mode="poisson",
            seed=23,
        ),
    )
    result = simulate(
        scene,
        observation=observation,
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=64, brem_electrons=64),
            cxr_reflections=CxrReflections(n_families=2),
        ),
    )
    batch = result.acquire(
        region=(slice(None), slice(None)),
        components=("cxr", "brem", "characteristic"),
    )
    return result, batch


# tutorial-end


def plot(data):
    expected, realized = (np.asarray(data[k]) for k in ("expected", "realized"))
    fig, axes = plt.subplots(2, 1, figsize=(6, 7.5), layout="constrained")
    for ax, image, title in zip(
        axes,
        (expected, realized),
        ("Expected registered counts", "Simulated Poisson realization"),
        strict=True,
    ):
        artist = ax.imshow(image, origin="lower", vmin=0, vmax=max(realized.max(), expected.max()))
        ax.set(xlabel="Pixel column", ylabel="Pixel row", title=title)
        ax.set_xticks((0, 5, 10, 15))
        ax.set_yticks((0, 5, 10, 15))
        fig.colorbar(artist, ax=ax, label="Counts / pixel in 1 s (0.5–4 keV)")
    return fig


if __name__ == "__main__":
    result, batch = calculate()
    plot(
        {
            "expected": batch.expected.sum(axis=-1).reshape(16, 16),
            "realized": batch.total_counts.reshape(16, 16),
        }
    )
    plt.show()


def compute():
    """Cache actual counts plus resolved source and observation identities."""
    import warnings

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result, batch = calculate()
    return {
        "expected": batch.expected.sum(axis=-1).reshape(16, 16).tolist(),
        "realized": batch.total_counts.reshape(16, 16).tolist(),
        "measured_edges_eV": batch.measured_edges_eV.tolist(),
        "components": batch.components,
        "outside_window_expected": {
            "underflow": float(batch.underflow_expected.sum()),
            "overflow": float(batch.overflow_expected.sum()),
            "below_cut": float(batch.below_cut_expected.sum()),
        },
        "seed": result.case["seed"],
        "case": dict(result.case),
        "provenance": dict(result.provenance),
        "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in recorded}),
    }
