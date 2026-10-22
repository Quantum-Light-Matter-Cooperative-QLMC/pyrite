"""Install the #510 per-history tally into an isolated copy of pinned PenRed.

Input is a clean checkout of PenRed at the pinned commit. Rejects any other
source: the two registry files the patch edits must match their pinned hashes.
The tally is copied next to the generic tallies and registered in both
(DICOM and non-DICOM) type lists; no physics source is touched. Build
afterwards with the commands in README.md.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PIN = "145a8a357f08affa551f592300fa6dfcab98a762"
REGISTRY = {
    "src/tallies/generic/includes/genericTallies.hh": (
        "0e0fd340586c9ec19acccd4a4d37eb8e34e7fb3a5ba53572cf2eb562867e373c"
    ),
    "src/tallies/generic/source/genericTallies.cpp": (
        "d89b9ead55f116aafdde385a123ea30157029525ee264f2e4331cc463146c433"
    ),
}
TALLY = Path(__file__).with_name("penred")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"expected {count} occurrence(s) of {old!r}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="PenRed checkout at the pinned commit")
    args = parser.parse_args()
    root = args.source
    before = {name: sha256(root / name) for name in REGISTRY}
    if before != REGISTRY:
        raise ValueError(f"source does not match pinned PenRed {PIN}")
    header, source = (root / name for name in REGISTRY)
    header.write_text(
        replace_once(
            replace_once(
                header.read_text(),
                '#include "tallyDetectionEDep.hh"\n',
                '#include "tallyDetectionEDep.hh"\n#include "tallyPyritePrimary.hh"\n',
            ),
            "pen_tallyTracking,\n",
            "pen_tallyTracking,\n\t\t\t\t\t   pen_tallyPyritePrimary,\n",
            count=2,
        )
    )
    source.write_text(
        replace_once(
            source.read_text(),
            '#include "tallyDetectionEDep.cpp"\n',
            '#include "tallyDetectionEDep.cpp"\n#include "tallyPyritePrimary.cpp"\n',
        )
    )
    added = {}
    for suffix, folder in [(".hh", "includes"), (".cpp", "source")]:
        target = root / "src/tallies/generic" / folder / f"tallyPyritePrimary{suffix}"
        shutil.copyfile(TALLY / target.name, target)
        added[str(target.relative_to(root))] = sha256(target)
    record = {
        "penred_commit": PIN,
        "before": before,
        "after": {name: sha256(root / name) for name in REGISTRY},
        "added": added,
        "physics_sources_modified": False,
    }
    (root / "benchmark-patch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
