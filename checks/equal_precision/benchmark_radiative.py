"""Compare the production radiative kernel with a retained pre-change source.

Use separate fresh processes for baseline/candidate. The diagnostic wrapper
keeps full matching warmups, synchronized boundaries and trajectory hashes.
"""

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
    if args.variant == "baseline" and args.baseline_source is None:
        parser.error("baseline requires --baseline-source")
    if args.variant == "baseline":
        package = importlib.import_module("pyrite.montecarlo.transport")
        name = package.__name__ + "._jit_radiative_device"
        if name in sys.modules:
            raise RuntimeError("baseline must be loaded before CUDA kernel imports")
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        package._jit_radiative_device = module
    from pyrite.montecarlo.transport import _jit_radiative_device

    source = Path(_jit_radiative_device.__file__)
    sys.argv = [sys.argv[0], "baseline", str(args.audit), *adapter_args]
    namespace = runpy.run_path(str(Path(__file__).with_name("diagnose_kernel.py")))
    namespace["main"]()
    audit = json.loads(args.audit.read_text())
    audit.update(
        variant=args.variant, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    sample_path = Path(adapter_args[adapter_args.index("--output") + 1])
    sample = json.loads(sample_path.read_text())
    sample.update(diagnostic_variant=args.variant, radiative_source_sha256=audit["source_sha256"])
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")


if __name__ == "__main__":
    main()
