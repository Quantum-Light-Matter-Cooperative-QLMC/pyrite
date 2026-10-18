"""Deterministic alternating-stack reference for the experimental Born DTR field.

Run with ``python -m pyrite.validation.dtr_reference --output-dir DIR``.
No Monte Carlo transport, CBS, mosaic average or production case is scored.
This is implementation evidence, not the published-result anchor for #581.

Validation: dtr-kinematic-boundary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from ..materials import CATALOG
from ..materials.crystal import CRYSTALS, HBARC_EV_ANG, M_E_EV, chi_g, reciprocal_g_vector
from ..montecarlo.spectrum.dtr import dtr_crossing_amplitudes, pxr_straight_amplitude
from ..montecarlo.spectrum.transition_radiation import (
    beta_from_kinetic_eV,
    interface_transition_radiation_amplitude,
    medium_dielectric,
)

REFERENCE_PROFILE = Path(__file__).with_name("reference_data") / "dtr_stack.toml"
DIAGNOSTIC_MODEL = "dtr-first-born-diagnostic-v1"


def reference_spectrum(
    profile: Path = REFERENCE_PROFILE,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return density columns [photons/eV/sr/electron] and resolved provenance.

    The TOML profile selects a symmetric, normally traversed A/B/A stack, one
    basal reflection in the central crystal layer, and a narrow observation
    direction offset from exact backscatter. The Maxwell frequency convention
    requires the conjugate susceptibility of the positive-hkl Friedel mate;
    g points along -z. See the DTR derivation for that explicit transfer sign.

    Limits: all diffracted terms vanish with chi_transfer; the on/off columns
    use the identical deterministic straight passage. No sampling seed exists.

    Validation: dtr-kinematic-boundary
    """
    raw = Path(profile).read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    layer = config["crystal_layer"]
    crystal = config["layers"][layer]["material"]
    hkl = tuple(config["reflection"])
    if hkl[:2] != (0, 0) or hkl[2] <= 0:
        raise ValueError("reference profile requires a positive basal (00l) reflection")
    _, gmag = reciprocal_g_vector(hkl, CRYSTALS[crystal]["lattice"])
    beta = beta_from_kinetic_eV(config["kinetic_energy_keV"] * 1000)
    angle = float(config["observation_offset_rad"])
    if not 0 < angle < np.pi / 2:
        raise ValueError("observation_offset_rad must be between 0 and pi/2")
    n = np.array([np.sin(angle), 0.0, np.cos(angle)])
    center = HBARC_EV_ANG * gmag / (1 / beta + n[2])
    half_width, nodes = config["energy_half_width_eV"], config["energy_nodes"]
    if not 0 < half_width < center or not isinstance(nodes, int) or nodes < 3:
        raise ValueError(
            "reference requires 0 < energy_half_width_eV < line energy and energy_nodes >= 3"
        )
    energy = np.linspace(center - half_width, center + half_width, nodes)
    eps = [medium_dielectric(row["material"], energy) for row in config["layers"]]
    layers = [(e, row["thickness_ang"]) for e, row in zip(eps, config["layers"], strict=True)]
    z = np.r_[0.0, np.cumsum([d for _, d in layers])]
    velocity = np.array([0.0, 0.0, -beta])
    crossings: dict[str, Any] = dict(
        velocity=np.tile(velocity, (len(z), 1)),
        time_ang=(z[-1] - z) / beta,
        position_ang=np.column_stack([np.zeros(len(z)), np.zeros(len(z)), z]),
        interface=np.arange(len(z)),
    )
    chi = np.conj(
        chi_g(crystal, hkl, energy, B_ang2=CATALOG.crystal(crystal).B_ang2, use_henke=True)
    )
    shared: dict[str, Any] = dict(
        photon_energy_eV=energy, direction=n, layers=layers, substrate_eps=1.0
    )
    diffraction: dict[str, Any] = dict(
        crystal_layer=layer,
        reciprocal_vector_ang=[0.0, 0.0, -gmag],
        chi_transfer=chi,
        lattice_origin_ang=config["lattice_origin_ang"],
    )
    # All sources are -e. This common sign cancels in every table entry, but
    # retaining it makes the vectors consistent with transported electrons.
    dtr = -dtr_crossing_amplitudes(**shared, **diffraction, **crossings).sum(axis=-2)
    pxr = -pxr_straight_amplitude(
        **shared, **diffraction, velocity=velocity, time_ang=0.0, position_ang=[0.0, 0.0, z[-1]]
    )
    tr = -interface_transition_radiation_amplitude(**shared, **crossings).sum(axis=-2)
    scale = energy * n[2] ** 2 / (16 * np.pi**4)

    def density(field):
        return scale * np.sum(np.abs(field) ** 2, axis=-1)

    line = density(pxr + dtr)
    columns = dict(
        energy_eV=energy,
        pxr=density(pxr),
        dtr=density(dtr),
        pxr_dtr_cross=2 * scale * np.sum((pxr * dtr.conj()).real, axis=-1),
        line=line,
        tr=density(tr),
        total=density(tr + pxr + dtr),
    )
    columns["tr_line_cross"] = 2 * scale * np.sum((tr * (pxr + dtr).conj()).real, axis=-1)
    if not all(np.all(np.isfinite(value)) for value in columns.values()):
        raise ValueError(
            "reference spectrum contains non-finite values; check dielectric data range"
        )
    lam = energy * np.sqrt(eps[layer] - np.sum(n[:2] ** 2))
    d = layers[layer][1]
    extinction = 2 * HBARC_EV_ANG * np.abs(lam) / (energy**2 * np.abs(chi))
    formation = HBARC_EV_ANG / np.abs(-energy / beta + lam)
    absorption = HBARC_EV_ANG / (2 * lam.imag)
    digest = hashlib.sha256(raw + DIAGNOSTIC_MODEL.encode())
    for value in [np.asarray([gmag]), *eps, chi]:
        digest.update(np.asarray(value, dtype="<c16").tobytes())
    i = nodes // 2
    metadata = dict(
        name=config["name"],
        model=DIAGNOSTIC_MODEL,
        profile_sha256=hashlib.sha256(raw).hexdigest(),
        resolved_sha256=digest.hexdigest(),
        config=config,
        center_eV=float(center),
        direction=n.tolist(),
        gamma_inverse_squared=float((1 + config["kinetic_energy_keV"] * 1000 / M_E_EV) ** -2),
        abs_chi0=float(abs(eps[layer][i] - 1)),
        extinction_scale_ang=float(extinction[i]),
        formation_scale_ang=float(formation[i]),
        absorption_depth_ang=float(absorption[i]),
        max_born_coupling=float(np.max(d / extinction)),
        dtr_on_off_peak_fraction=float(np.max(abs(line - columns["pxr"])) / np.max(columns["pxr"])),
        units="photons/eV/sr per incident electron",
        validation_status="unverified",
        limitations=[
            "first Born only",
            "straight passage, no transport",
            "no CBS or mosaic averaging",
            "not a production campaign profile",
            "no independent or published-result validation",
        ],
    )
    return columns, metadata


def main(argv: list[str] | None = None) -> None:
    """Write a reference CSV, provenance JSON and comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=REFERENCE_PROFILE, help="diagnostic TOML profile"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="directory for CSV, JSON and PNG outputs"
    )
    args = parser.parse_args(argv)
    start = time.perf_counter()
    columns, metadata = reference_spectrum(args.profile)
    metadata["calculation_seconds"] = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "spectrum.csv",
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    (args.output_dir / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for name, label in (
        ("pxr", "PXR (DTR off)"),
        ("line", "PXR + DTR (amplitudes summed)"),
        ("dtr", "DTR alone"),
    ):
        ax.plot(columns["energy_eV"], columns[name], label=label)
    ax.set(
        xlabel="Photon energy [eV]",
        ylabel="photons / eV / sr / electron",
        title="SiO2 / MoS2 / SiO2 — experimental Born diagnostic",
    )
    ax.legend()
    fig.savefig(args.output_dir / "spectrum.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
