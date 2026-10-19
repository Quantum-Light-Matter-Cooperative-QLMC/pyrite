"""Deterministic alternating-stack reference for experimental DTR fields.

Run with ``python -m pyrite.validation.dtr_reference --output-dir DIR``.
No Monte Carlo transport, CBS, mosaic average or production case is scored.
This is implementation evidence, not the published-result anchor for #581.

Validation: dtr-kinematic-boundary
Validation: dtr-two-beam-boundary
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
from ..montecarlo.spectrum.dtr_two_beam import two_beam_straight_field
from ..montecarlo.spectrum.transition_radiation import (
    beta_from_kinetic_eV,
    interface_transition_radiation_amplitude,
    medium_dielectric,
)

REFERENCE_PROFILE = Path(__file__).with_name("reference_data") / "dtr_stack.toml"
DIAGNOSTIC_MODEL = "dtr-first-born-diagnostic-v1"
TWO_BEAM_DIAGNOSTIC_MODEL = "dtr-born-two-beam-comparison-v1"


def reference_spectrum(
    profile: Path = REFERENCE_PROFILE,
    *,
    two_beam: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return density columns [photons/eV/sr/electron] and resolved provenance.

    The TOML profile selects a symmetric, normally traversed A/B/A stack, one
    basal reflection in the central crystal layer, and a narrow observation
    direction offset from exact backscatter. The Maxwell frequency convention
    requires the conjugate susceptibility of the positive-hkl Friedel mate;
    g points along -z. See the DTR derivation for that explicit transfer sign.

    Limits: all diffracted terms vanish with chi_transfer; the on/off columns
    use the identical deterministic straight passage. No sampling seed exists.
    ``two_beam=True`` appends the repeated-diffraction source-row densities
    and their signed cross term. Existing columns remain the Born diagnostic;
    the dynamical line row contains PXR and DTR together, without a separate
    DTR-off interpretation. Both solvers remain unverified.

    Validation: dtr-kinematic-boundary
    Validation: dtr-two-beam-boundary
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
    reverse = None
    if two_beam:
        # Reverse the Miller indices before conjugating the frequency
        # convention. In an absorbing crystal this is not conj(chi_transfer).
        reverse = np.conj(
            chi_g(
                crystal,
                tuple(-h for h in hkl),
                energy,
                B_ang2=CATALOG.crystal(crystal).B_ang2,
                use_henke=True,
            )
        )
        field = two_beam_straight_field(
            **shared,
            **diffraction,
            velocity=velocity,
            time_ang=0.0,
            position_ang=[0.0, 0.0, z[-1]],
            chi_reverse=reverse,
        )
        columns.update(
            two_beam_transition=density(field.transition),
            two_beam_line=density(field.line),
            two_beam_cross=2 * scale * np.sum((field.transition * field.line.conj()).real, axis=-1),
            two_beam_total=density(field.total),
        )
        columns["two_beam_minus_born_total"] = columns["two_beam_total"] - columns["total"]
    if not all(np.all(np.isfinite(value)) for value in columns.values()):
        raise ValueError(
            "reference spectrum contains non-finite values; check dielectric data range"
        )
    lam = energy * np.sqrt(eps[layer] - np.sum(n[:2] ** 2))
    d = layers[layer][1]
    extinction = 2 * HBARC_EV_ANG * np.abs(lam) / (energy**2 * np.abs(chi))
    formation = HBARC_EV_ANG / np.abs(-energy / beta + lam)
    absorption = HBARC_EV_ANG / (2 * lam.imag)
    model = TWO_BEAM_DIAGNOSTIC_MODEL if two_beam else DIAGNOSTIC_MODEL
    digest = hashlib.sha256(raw + model.encode())
    for value in [np.asarray([gmag]), *eps, chi]:
        digest.update(np.asarray(value, dtype="<c16").tobytes())
    if reverse is not None:
        digest.update(np.asarray(reverse, dtype="<c16").tobytes())
    i = nodes // 2
    metadata = dict(
        name=config["name"],
        model=model,
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
            "Born and two-beam diagnostics" if two_beam else "first Born only",
            "straight passage, no transport",
            "no CBS or mosaic averaging",
            "not a production campaign profile",
            "no independent or published-result validation",
        ],
    )
    if two_beam:
        metadata["two_beam_born_peak_fraction"] = float(
            np.max(np.abs(columns["two_beam_minus_born_total"])) / np.max(columns["total"])
        )
        metadata["two_beam_accounting"] = (
            "transition: observed source row including diffraction extinction; "
            "line: second source row including PXR and DTR; "
            "total = transition + line + signed cross term (densities)"
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
    parser.add_argument(
        "--two-beam", action="store_true", help="append experimental two-beam/Born comparison"
    )
    args = parser.parse_args(argv)
    start = time.perf_counter()
    columns, metadata = reference_spectrum(args.profile, two_beam=args.two_beam)
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

    if args.two_beam:
        fig, (ax, difference) = plt.subplots(
            2,
            1,
            figsize=(7.5, 6),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        curves = (
            ("total", "Born total (TR + PXR + DTR)"),
            ("two_beam_total", "Two-beam total (source rows summed)"),
        )
        difference.plot(columns["energy_eV"], columns["two_beam_minus_born_total"])
        difference.axhline(0, color="0.5", linewidth=0.8)
        difference.set(xlabel="Photon energy [eV]", ylabel="Two-beam − Born\n[same units]")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        curves = (
            ("pxr", "PXR (DTR off)"),
            ("line", "PXR + DTR (amplitudes summed)"),
            ("dtr", "DTR alone"),
        )
        ax.set_xlabel("Photon energy [eV]")
    for i, (name, label) in enumerate(curves):
        ax.plot(
            columns["energy_eV"],
            columns[name],
            label=label,
            linestyle="--" if args.two_beam and i == 1 else "-",
        )
    ax.set(
        ylabel="photons / eV / sr / electron",
        title=f"{metadata['name']}\nExperimental {'two-beam / Born' if args.two_beam else 'Born'} diagnostic",
    )
    ax.legend()
    fig.savefig(args.output_dir / "spectrum.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
