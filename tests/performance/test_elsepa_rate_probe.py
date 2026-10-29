"""The #792 rate-cost probe must keep a single, compilable insertion site."""

import ast
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PROBE = ROOT / "checks/equal_precision/probe_elsepa_rate_cost.py"
TRANSPORT = ROOT / "src/pyrite/montecarlo/transport"


@pytest.mark.parametrize(
    "module,anchor,probe",
    [("cores", "CPU_ANCHOR", "CPU_PROBE"), ("_jit_kernel", "CUDA_ANCHOR", "CUDA_PROBE")],
)
def test_probe_anchor_is_unique_and_transformed_source_parses(module, anchor, probe):
    names = runpy.run_path(str(PROBE))
    source = (TRANSPORT / f"{module}.py").read_text()
    assert source.count(names[anchor]) == 1
    transformed = source.replace(names[anchor], names[probe].format(k=2) + names[anchor])
    ast.parse(transformed)
