"""Compare request-owned BremsLib staging with a pinned pre-change module."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["baseline", "candidate"])
    parser.add_argument("audit", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    args, adapter_args = parser.parse_known_args()
    from pyrite.montecarlo.transport import hard_radiative

    source = Path(hard_radiative.__file__)
    if args.variant == "baseline":
        if args.baseline_source is None:
            parser.error("baseline requires --baseline-source")
        package = importlib.import_module("pyrite.montecarlo.transport")
        name = package.__name__ + "._completion_baseline"
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        # The result assembler resolves this routine from its owning module at
        # call time. Keep the rest of the current transport preparation intact.
        hard_radiative.complete_hard_radiative_events = module.complete_hard_radiative_events
        source = args.baseline_source
    sys.argv = [sys.argv[0], "learned32", str(args.audit), *adapter_args]
    runpy.run_path(str(Path(__file__).with_name("diagnose_kernel.py")))["main"]()
    audit = json.loads(args.audit.read_text())
    audit.update(
        variant=args.variant,
        completion_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    sample_path = Path(adapter_args[adapter_args.index("--output") + 1])
    sample = json.loads(sample_path.read_text())
    sample.update(
        diagnostic_variant=args.variant, completion_source_sha256=audit["completion_source_sha256"]
    )
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")


if __name__ == "__main__":
    main()
