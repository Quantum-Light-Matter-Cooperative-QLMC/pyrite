"""The execution contract is cheap to test: no Monte Carlo or GPU required."""

from dataclasses import FrozenInstanceError

import pytest

from pyrite.execution import (
    ExecutionState,
    LocalExecutor,
    ResourceRequest,
    RetryPolicy,
    WorkUnit,
    iter_work_units,
)


def test_work_planner_yields_contiguous_bounded_units():
    cases = [{"id": i} for i in range(5)]
    units = list(
        iter_work_units(
            cases,
            cases_per_unit=2,
            case_key=lambda case: f"content-key-{case['id']}",
            estimate_cost=lambda case: case["id"] + 1.0,
        )
    )
    assert [(unit.start, unit.stop, unit.size) for unit in units] == [
        (0, 2, 2),
        (2, 4, 2),
        (4, 5, 1),
    ]
    assert [unit.unit_id for unit in units] == [
        "work-00000000-00000002",
        "work-00000002-00000004",
        "work-00000004-00000005",
    ]
    assert units[1].case_keys == ("content-key-2", "content-key-3")
    assert [unit.estimated_cost for unit in units] == [3.0, 7.0, 5.0]
    assert units == list(
        iter_work_units(
            cases,
            cases_per_unit=2,
            case_key=lambda case: f"content-key-{case['id']}",
            estimate_cost=lambda case: case["id"] + 1.0,
        )
    )
    with pytest.raises(FrozenInstanceError):
        units[0].start = 99


def test_work_planner_empty_and_lazy():
    assert list(iter_work_units([], cases_per_unit=4)) == []
    calls = []
    cases = [0, 1, 2, 3]
    units = iter_work_units(
        cases, cases_per_unit=2, case_key=lambda c: (calls.append(c), str(c))[1]
    )
    assert calls == []
    assert next(units).case_keys == ("0", "1")
    assert calls == [0, 1]
    assert next(units).case_keys == ("2", "3")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_invalid_chunk_size(bad):
    with pytest.raises(ValueError, match="cases_per_unit"):
        list(iter_work_units([{}], cases_per_unit=bad))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"cpu_cores": 0}, "cpu_cores must be a positive integer"),
        ({"cpu_cores": "3"}, "cpu_cores must be a positive integer"),
        ({"memory_bytes": -2}, "memory_bytes must be a nonnegative integer"),
        ({"gpu_count": -1}, "gpu_count must be a nonnegative integer"),
        ({"accelerator": "cuda2"}, "unsupported accelerator requirement"),
        ({"accelerator": "cpu", "gpu_count": 1}, "cpu-only work cannot request GPUs"),
        ({"gpu_memory_bytes": 100}, "gpu_memory_bytes requires gpu_count > 0"),
        ({"accelerator": "cuda"}, "device-specific work must request at least one GPU"),
    ],
)
def test_invalid_resources(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ResourceRequest(**kwargs)


def test_work_and_retry_validation():
    assert ExecutionState.SUCCEEDED.value == "succeeded"
    assert RetryPolicy().max_attempts == 1
    with pytest.raises(ValueError, match=r"max_attempts must be a positive integer"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match=r"unit_id must be nonempty"):
        WorkUnit("", 0, 1)
    with pytest.raises(ValueError, match=r"work unit must cover at least one nonnegative index"):
        WorkUnit("unit", 2, 2)
    with pytest.raises(ValueError, match=r"case_keys must align with the work interval"):
        WorkUnit("unit", 0, 2, case_keys=("only-one",))
    with pytest.raises(ValueError, match=r"estimated_cost must be finite and nonnegative"):
        WorkUnit("unit", 0, 1, estimated_cost=float("nan"))


def test_local_executor_delegates_options_and_does_not_add_parallelism():
    observed = {}
    received = []
    cases = [{"id": 0}, {"id": 1}]

    def synthetic_runner(payload, *, callback, should_stop, **options):
        observed.update(options)
        assert payload is cases
        assert should_stop() is False
        for i, case in enumerate(payload):
            callback(i, case, {"value": i})
        return [None] * len(payload)

    executor = LocalExecutor(synthetic_runner)
    result = executor.execute(
        cases,
        callback=lambda i, case, output: received.append((i, case, output)),
        should_stop=lambda: False,
        keep_results=False,
        max_workers=3,
        progress=False,
        transport_only=False,
    )
    assert result == [None, None]
    assert received == [(i, case, {"value": i}) for i, case in enumerate(cases)]
    assert observed == {
        "keep_results": False,
        "max_workers": 3,
        "progress": False,
        "transport_only": False,
    }
