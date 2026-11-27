"""Read-only Pixi workspace diagnostics, including a pre-activation lock probe."""

import argparse
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

_LOCK_CONFLICT = re.compile(
    r"environment '([^']+)' for platform (\S+) are out of date because "
    r"'([^']+)' is locked as a conda package but only requested by pypi dependencies"
)


def manifest_errors(manifest: dict) -> list[str]:
    """Enforce explicit ownership for known cross-feature Conda/PyPI overlaps."""
    pixi = manifest["tool"]["pixi"]
    features = pixi.get("feature", {})
    errors = []
    if "packaging" not in pixi.get("dependencies", {}):
        errors.append("Declare packaging in [tool.pixi.dependencies] (shared runtime dependency).")
    for name in ("oracle", "docs", "notebooks"):
        if "matplotlib-base" not in features.get(name, {}).get("dependencies", {}):
            errors.append(f"Declare matplotlib-base in [tool.pixi.feature.{name}.dependencies].")
    for name, env in pixi["environments"].items():
        if name.startswith("runtime-"):
            extras = set(env["features"]) & (set(manifest["dependency-groups"]) | {"apps", "plots"})
            if extras:
                errors.append(f"{name}: developer-only features {', '.join(sorted(extras))}")
    return errors


def lock_conflicts(output: str) -> list[str]:
    """Extract actionable solver diagnostics without dumping the environment."""
    return [
        f"{env}/{platform}: {package} is Conda-locked but PyPI-only requested; "
        "declare it in the narrow owning feature, not automatically in the base feature."
        for env, platform, package in dict.fromkeys(_LOCK_CONFLICT.findall(output))
    ]


def lock_errors(root: Path) -> list[str]:
    pixi = shutil.which("pixi")
    if not pixi:
        return ["pixi is not on PATH; install Pixi before checking the lock."]
    # --locked refuses stale locks; --no-install prevents environment changes.
    # Pixi checks all environment/platform rows even when selecting default.
    result = subprocess.run(
        [pixi, "-v", "run", "--locked", "--no-install", "-e", "default", "echo", "lock-ok"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    conflicts = lock_conflicts(result.stderr + "\n" + result.stdout)
    return conflicts or [
        "Pixi lock probe failed; inspect `pixi -vvv run --locked --no-install -e default echo lock-ok`."
    ]


def run(root: Path, *, check_lock: bool = False) -> int:
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    errors = manifest_errors(manifest)
    if check_lock:
        errors.extend(lock_errors(root))
    for error in errors:
        print(f"doctor: {error}")
    if not errors:
        scope = "manifest and lock" if check_lock else "manifest policy"
        print(f"doctor: {scope} OK")
    return int(bool(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="store_true", help="probe Pixi lock satisfiability")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    raise SystemExit(run(root, check_lock=args.lock))


if __name__ == "__main__":
    main()
