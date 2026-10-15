"""Energy and trajectory bookkeeping for precision diagnostics, independent of CUDA."""

import numpy as np


def trajectory_checks(result, n_primaries, energy_keV, cutoff_eV):
    from pyrite.perf.precision import grouped_scores

    ids = np.asarray(result["electron_id"])
    events = np.asarray(result["event_kind"])
    start = np.asarray(result["E_start_keV"])
    end = np.asarray(result["E_end_keV"])
    photons = np.asarray(result["hard_radiative_k_eV"]) * 1e-3
    direction = np.asarray(result["v_hat"])
    for name, value in result.items():
        if (
            isinstance(value, np.ndarray)
            and value.dtype.kind == "f"
            and not np.isfinite(value).all()
        ):
            raise ValueError(f"nonfinite trajectory field: {name}")
    if np.any(end < 0) or np.any(start < end - 1e-10) or np.any(np.asarray(result["L_ang"]) < 0):
        raise ValueError("invalid trajectory energy or length")
    if (
        np.any(photons < 0)
        or np.any((photons > 0) & (photons < cutoff_eV * 1e-3 - 1e-8))
        or np.any(photons > end + 1e-8)
    ):
        raise ValueError("invalid photon energy")
    norm_error = float(np.max(np.abs(np.linalg.norm(direction, axis=1) - 1)))
    if norm_error > 1e-10:
        raise ValueError("direction normalization failed")
    terminal = np.isin(events, [4, 5, 6, 7])
    if not np.array_equal(np.sort(ids[terminal]), np.arange(n_primaries)):
        raise ValueError("incomplete primary terminal records")
    hard = np.flatnonzero(events == 9)
    if np.any(hard + 1 >= ids.size) or np.any(ids[hard] != ids[hard + 1]):
        raise ValueError("missing post-photon continuation")
    debit_error = float(np.max(np.abs(end[hard] - start[hard + 1] - photons[hard]), initial=0))
    continuous = grouped_scores(ids, start - end, n_primaries)
    emitted = grouped_scores(ids, photons, n_primaries)
    residual = grouped_scores(ids[terminal], end[terminal] - photons[terminal], n_primaries)
    ledger_error = float(np.max(np.abs(continuous + emitted + residual - energy_keV)))
    if debit_error > 1e-9 or ledger_error > 1e-7:
        raise ValueError(
            f"primary energy conservation failed: debit={debit_error:.17g} keV, ledger={ledger_error:.17g} keV"
        )
    scores = {
        "photon_energy_keV_per_primary": emitted,
        "continuous_loss_keV_per_primary": continuous,
        "terminal_energy_keV_per_primary": residual,
    }
    transmitted = events == 5
    mu = direction[:, 2]
    for index, (low, high) in enumerate(
        zip([0, 0.5, 0.9, 0.99], [0.5, 0.9, 0.99, 1.0000000001], strict=True)
    ):
        selected = transmitted & (mu >= low) & (mu < high)
        scores[f"transmitted_mu_bin_{index}"] = grouped_scores(
            ids[selected], np.ones(np.count_nonzero(selected)), n_primaries
        )
    return scores, {
        "energy_ledger_max_error_keV": ledger_error,
        "photon_debit_max_error_keV": debit_error,
        "direction_norm_max_error": norm_error,
    }


def eligible_variants(root, protocol, scenes):
    """Admit timing arms only after complete trajectories and 1-keV component checks."""
    import json

    eligible, reasons = [], {}
    for variant in protocol["variants"]:
        try:
            probe = json.loads((root / f"probe-{variant}.json").read_text(encoding="utf-8"))
            if variant in protocol.get("full_screen_variants", []) and not all(
                r["screening_passed"] for r in probe["reports"]
            ):
                raise ValueError("full endpoint component screen failed")
            for scene in scenes:
                sample = (
                    root
                    / f"{scene['id']}-{protocol['preflight']['histories']}-{protocol['preflight']['seed']}-{variant}.json"
                )
                if sample.with_suffix(".failed.json").exists():
                    raise ValueError(f"failed trajectory preflight: {sample.name}")
                record = json.loads(sample.read_text(encoding="utf-8"))
                if (
                    record.get("schema") != "pyrite.equal-precision.precision-sample.v1"
                    or record["precision_source"]["variant_sha256"]
                    != probe["source"]["variant_sha256"]
                ):
                    raise ValueError("preflight source/schema mismatch")
            for element in sorted({s["element"] for s in scenes}):
                report = next(r for r in probe["reports"] if r["element"] == element)
                if not (report["finite"] and report["nonnegative"] and report["photon_support"]):
                    raise ValueError("invalid component probe")
                with np.load(root / f"probe-{variant}-{element}.npz") as raw:
                    selected = raw["cutoff_eV"] == 1000
                    if not selected.any():
                        raise ValueError("missing operating-cutoff probe")
                    expected, actual = raw["expected"][selected], raw["actual"][selected]
                    scale = np.maximum(
                        np.abs(expected[:, :2]), np.max(np.abs(expected[:, :2]), axis=0) * 1e-6
                    )
                    error = np.max(np.abs(actual[:, :2] - expected[:, :2]) / scale)
                    inverse = np.max(
                        np.abs(actual[:, 2] - expected[:, 2]) / raw["energy_eV"][selected]
                    )
                    if (
                        not np.isfinite(actual).all()
                        or error > protocol["preflight"]["moment_scaled_error_limit"]
                        or inverse
                        > protocol["preflight"]["inverse_error_over_incident_energy_limit"]
                    ):
                        raise ValueError("operating-cutoff component screen failed")
            eligible.append(variant)
        except (OSError, ValueError, KeyError, StopIteration) as error:
            reasons[variant] = str(error)
    return eligible, reasons
