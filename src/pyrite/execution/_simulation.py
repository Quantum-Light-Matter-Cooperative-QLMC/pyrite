"""Adapt one resolved simulation to the existing batch executor contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .._executor_contract import execution_provenance as execution_provenance
from .._executor_contract import validate_executor as validate_executor
from .artifacts import (
    CaseArtifactRef,
    CaseArtifactStore,
    DirectionalArtifactRef,
    load_directional_result,
)
from .protocol import Executor

if TYPE_CHECKING:
    from ..montecarlo.case import Case


def execute_simulation(
    executor: Executor,
    case: Case,
    table_markers: Mapping[str, str],
    *,
    directions: Any = None,
) -> dict[str, Any]:
    """Dispatch once, validate worker artifacts, and return ordinary runner arrays.

    The caller owns observation/Result assembly. Directions travel beside the
    case and scheduler metadata never enters its serialized scientific inputs.
    """
    options: dict[str, Any] = {} if directions is None else {"observation_directions": [directions]}
    outputs = executor.execute([case], progress=False, keep_results=True, **options)
    if len(outputs) != 1 or outputs[0] is None:
        raise RuntimeError("executor must return exactly one completed simulation result")
    output = outputs[0]
    if isinstance(output, (CaseArtifactRef, DirectionalArtifactRef)):
        from ..campaign.profiles import case_content_key

        store = getattr(executor, "store", None)
        if not isinstance(store, CaseArtifactStore):
            raise ValueError("executor returned an artifact without its CaseArtifactStore")
        key = case_content_key(case, xsgen_tables=table_markers)
        if isinstance(output, DirectionalArtifactRef):
            if directions is None:
                raise ValueError("executor returned unexpected directional artifacts")
            output = load_directional_result(
                store, output, material=case["crystal"], case_key=key, directions=directions
            )
        else:
            output = store.load(output, expected_material=case["crystal"], expected_content_key=key)
    if not isinstance(output, dict):
        raise TypeError("executor must return a runner output dictionary or checked artifact")
    if directions is None:
        return output
    if not isinstance(output.get("directional"), dict):
        raise ValueError("executor omitted requested directional spectra")
    return output["directional"]
