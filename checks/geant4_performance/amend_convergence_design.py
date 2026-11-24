"""Owner amendments of the frozen #1041 convergence design, made before confirmation.

The v1 freeze (``convergence_manifest.py``) hashes its own generator, so
amendments live here and derive from the committed v1 file by hash rather than
editing it. ``--check`` re-derives a committed amendment and refuses drift.
"""

import argparse
import copy
import json
import math
import runpy
from pathlib import Path

from pyrite._util.hashing import file_sha256

HERE = Path(__file__).resolve().parent
V1 = HERE / "convergence-design-v1.json"

V1_1_PAIRING = (
    "crn-paired only for (config, setting, observable) cells verified by "
    "audit_primary_pairing.py, with identical seeds and global primary offsets in both "
    "arms; otherwise independent"
)
ELSEPA_REASON = (
    "Owner 2026-11-24: ELSEPA energy-grid convergence is a physics-data question split to "
    "#1426; not qualified in #1041"
)


def amend_v1_1(frozen, frozen_sha256, gpu_hours):
    """Owner amendment of 2026-11-24, made before any confirmation run.

    Relaxes the per-observable target RSE from 2 % to 5 % (allowances are
    recomputed as S*eps/3/4/k, never rescaled), moves the ELSEPA
    energy-density ladder out of scope (``unsupported``, owned by #1426; it
    keeps its rows, multiplicity and family share like any unsupported
    control), admits audited common-random-number pairing and
    raises the confirmation GPU-hour ceiling. Rows, multiplicity and alpha
    allocation are unchanged.
    """
    if frozen["schema"] != "pyrite.geant4-performance.convergence-design.v1":
        raise ValueError("unsupported convergence design")
    if "amendment" in frozen or frozen["target_rse"] != 0.02:
        raise ValueError("v1.1 amends only the original v1 freeze")
    old_ceiling = frozen["resources"]["confirmation_gpu_hours"]
    if isinstance(gpu_hours, bool) or not isinstance(gpu_hours, (int, float)):
        raise ValueError("invalid GPU-hour ceiling")
    if not math.isfinite(gpu_hours) or gpu_hours < old_ceiling:
        raise ValueError("v1.1 may only extend the GPU-hour ceiling")
    out = copy.deepcopy(frozen)
    out["target_rse"] = 0.05
    scale = {(s["scene"], s["workload"], s["observable"]): s["scale"] for s in out["scales"]}
    scenes = {c["id"]: c["scene"] for c in out["configurations"]}
    for row in out["expected_rows"]:
        share = round(row["total_allowance"] / row["allowance"])
        total = scale[scenes[row["config"]], row["workload"], row["observable"]] * 0.05 / 3
        row["total_allowance"] = total
        row["allowance"] = total / share
        if row["control"] == "elsepa_energy_density":
            row["support"], row["reason"] = "unsupported", ELSEPA_REASON
    for config in out["configurations"]:
        for control in config["controls"]:
            if control["id"] == "elsepa_energy_density":
                control["support"], control["reason"] = "unsupported", ELSEPA_REASON
    out["seed_policy"]["pairing"] = V1_1_PAIRING
    out["resources"]["confirmation_gpu_hours"] = gpu_hours
    out["amendment"] = dict(
        version="1.1",
        amends="convergence-design-v1.json",
        amends_sha256=frozen_sha256,
        decided="2026-11-24",
        decided_by="owner (issue #1041)",
        before_confirmation=True,
        changes=[
            "target_rse 0.02 -> 0.05; allowances recomputed as S*0.05/3/4/k",
            "elsepa_energy_density unresolved -> unsupported (split to #1426)",
            "pairing: audited common-random-number pairs admitted",
            f"confirmation_gpu_hours {old_ceiling} -> {gpu_hours}",
        ],
    )
    return out


def derive(gpu_hours):
    """Check v1 still reproduces, then apply the v1.1 amendment."""
    manifest = runpy.run_path(str(HERE / "convergence_manifest.py"))
    frozen = json.loads(V1.read_text())
    if frozen != manifest["make_manifest"](frozen["pilot_files_sha256"]):
        raise ValueError("frozen v1 design no longer reproduces")
    out = amend_v1_1(frozen, file_sha256(V1), gpu_hours)
    out["amendment"]["amender_sha256"] = file_sha256(Path(__file__))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "convergence-design-v1.1.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu-hours", type=float, help="confirmation GPU-hour ceiling to freeze")
    group.add_argument("--check", action="store_true", help="re-derive the committed amendment")
    args = parser.parse_args()
    if args.check:
        frozen = json.loads(args.output.read_text())
        if derive(frozen["resources"]["confirmation_gpu_hours"]) != frozen:
            raise ValueError("amended design differs from its recorded derivation")
        result = frozen
    else:
        if args.output.exists():
            parser.error("refusing to overwrite a frozen amendment; use a new version/path")
        result = derive(args.gpu_hours)
        manifest = runpy.run_path(str(HERE / "convergence_manifest.py"))
        manifest["write_manifest"](args.output, result)
    print(
        json.dumps(
            dict(
                version=result["amendment"]["version"],
                target_rse=result["target_rse"],
                gpu_hours=result["resources"]["confirmation_gpu_hours"],
            )
        )
    )


if __name__ == "__main__":
    main()
