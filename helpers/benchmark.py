"""Run, list and profile FlagCV benchmarks."""

import math
from contextlib import nullcontext
from functools import partial
from statistics import median

import pytest

from conftest import emit
from helpers.baseline import comparison, external_result, read_json, summarize
from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES, STATEFUL_PROTOCOL
from helpers.launch_boundary import CURRENT_STREAM_OUT
from workloads.core import python_cases


def _measure_ms(candidate, args, *, warmup_ms=PROFILES["default"].warmup_ms,
                measurement_ms=MEASUREMENT_MS):
    import torch

    device = args[0].device
    if device.type == "cpu":
        pytest.fail("Device timing requires an accelerator")

    from triton.testing import do_bench

    with torch.get_device_module(device).device(device):
        return do_bench(lambda: candidate(*args), warmup=warmup_ms,
                        rep=measurement_ms, return_mode="median")


def _profile(config, case_id, candidate, build):
    import torch

    warmup_inputs = build()
    capture_inputs = build()
    device = capture_inputs[0].device
    device_module = torch.get_device_module(device) if device.type != "cpu" else None
    device_scope = device_module.device(device) if device_module else nullcontext()
    with device_scope:
        warmup_candidate = candidate.prepare(*warmup_inputs) if hasattr(candidate, "prepare") else candidate
        capture_candidate = candidate.prepare(*capture_inputs) if hasattr(candidate, "prepare") else candidate
        for _ in range(config.getoption("profile_warmup")):
            warmup_candidate(*warmup_inputs)
        if device_module:
            device_module.synchronize()
        scope = config.hook.pytest_flaggems_profile_scope(
            backend=config._flagcv_runtime.vendor,
            case_id=case_id,
        )
        before = config._evaluation.calls
        with scope if scope is not None else nullcontext():
            for _ in range(config.getoption("profile_iterations")):
                capture_candidate(*capture_inputs)
            if device_module:
                device_module.synchronize()
        assert (
            config._evaluation.calls > before
        ), "Profile did not execute its candidate"


def _prepare_baseline(op_name, args, vendor):
    import torch

    if vendor != "nvidia":
        raise ValueError("OpenCV CUDA baseline requires NVIDIA")
    from adapters.opencv_cuda import prepare

    dst_index = 1 if op_name in ("bitwise_not", "flip") else 2
    baseline_args = list(args)
    baseline_args[dst_index] = torch.empty_like(args[dst_index])
    baseline_args = tuple(baseline_args)
    return prepare(op_name, baseline_args), baseline_args


def _validate_outputs(op_name, candidate, args, baseline, baseline_args):
    """Validate both implementations before timing on identical inputs."""
    import torch

    from helpers.correctness import poison_output
    from helpers.tolerance import (
        assert_bitwise_equal,
        assert_exact,
        assert_relative_close,
    )
    from reference import get_reference

    dst_index = 1 if op_name in ("bitwise_not", "flip") else 2
    sources, dst, extra = args[:dst_index], args[dst_index], args[dst_index + 1 :]
    snapshots = tuple(source.clone() for source in sources)
    expected = torch.empty_like(dst)
    reference_args = tuple(source.clone() for source in snapshots)
    get_reference(op_name, dst.device)(*reference_args, expected, *extra)
    if op_name == "flip":
        compare = assert_bitwise_equal
    elif dst.dtype == torch.uint8:
        compare = assert_exact
    else:
        compare = assert_relative_close

    scope = (
        torch.get_device_module(dst.device).device(dst.device)
        if dst.device.type != "cpu"
        else nullcontext()
    )
    with scope:
        for function, operands in ((baseline, baseline_args), (candidate, args)):
            if function is None:
                continue
            target = operands[dst_index]
            if all(source.data_ptr() != target.data_ptr() for source in operands[:dst_index]):
                poison_output(target, expected)
            assert (
                function(*operands) is operands[dst_index]
            ), "Operator must return preallocated dst"
            compare(operands[dst_index], expected)
            for source, snapshot in zip(operands[:dst_index], snapshots):
                if source.data_ptr() == target.data_ptr():
                    source.copy_(snapshot)
                else:
                    assert_bitwise_equal(source, snapshot)
    if baseline is not None:
        compare(dst, baseline_args[dst_index])


def _mutated_inputs(op, args):
    if not args or not hasattr(args[0], "data_ptr"):
        return []
    dst_index = 1 if op in ("bitwise_not", "flip") else 2
    dst = args[dst_index]
    return [source for source in args[:dst_index] if source.data_ptr() == dst.data_ptr()][:1]


def _measure_stateful(run, args, mutated):
    import torch

    device = args[0].device
    device_module = torch.get_device_module(device)
    with device_module.device(device):
        state = [(tensor, tensor.clone()) for tensor in mutated]

        def restore():
            for tensor, saved in state:
                tensor.copy_(saved)

        try:
            for _ in range(5):
                restore()
                run(*args)
            samples = []
            start, stop = (device_module.Event(enable_timing=True) for _ in range(2))
            for _ in range(25):
                restore()
                device_module.synchronize()
                start.record()
                run(*args)
                stop.record()
                stop.synchronize()
                samples.append(start.elapsed_time(stop))
            return median(samples), samples
        finally:
            restore()
            device_module.synchronize()


def _target(args, vendor):
    if not args or getattr(getattr(args[0], "device", None), "type", None) != "cuda":
        return None
    import torch
    from runtime.identity import observe

    return observe(vendor, torch, args[0].device)


def _aten_candidate_boundary(candidate, args):
    if getattr(candidate, "execution_boundary", None) is not CURRENT_STREAM_OUT:
        return False
    return _check_stream_allocation(candidate, args)


def _check_stream_allocation(function, args):
    import torch

    device = args[0].device
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = function(*args)
        return (result is args[-1] and torch.cuda.current_stream(device) == stream
                and torch.cuda.max_memory_allocated(device) == before)


def _time_case(config, op_name, candidate, args, row):
    profile = PROFILES[config.getoption("benchmark_profile")]
    warmup = config.getoption("warmup")
    warmup = profile.warmup_ms if warmup is None else warmup
    measurement = config.getoption("iter") or MEASUREMENT_MS
    mutated = _mutated_inputs(op_name, args)
    protocol = profile.name
    if mutated:
        protocol = STATEFUL_PROTOCOL
    elif warmup != profile.warmup_ms or measurement != MEASUREMENT_MS:
        protocol = f"triton-do-bench-no-graph-{warmup}ms-warmup-{measurement}ms-median"
    native_mode = config.getoption("baseline")
    vendor = getattr(getattr(config, "_flagcv_runtime", None), "vendor", "cpu")
    baseline, baseline_args = None, None
    native_status, native_reason = "unavailable", "no equivalent native comparator on this device"
    if native_mode == "none":
        native_status, native_reason = "disabled", "native timing disabled"
    candidate_samples, baseline_samples, stateful_samples = [], [], []
    if hasattr(candidate, "prepare"):
        candidate = candidate.prepare(*args)
    _validate_outputs(op_name, candidate, args, None, None)
    baseline_version = None
    native_comparator = None
    prepare_native = None
    if native_mode == "opencv-cuda" or (native_mode == "auto" and vendor == "nvidia"):
        if not mutated:
            native_comparator = "opencv_cuda_live"
            prepare_native = partial(_prepare_baseline, op_name, args, vendor)
    elif native_mode == "auto" and vendor in ("amd", "hygon"):
        from adapters import aten_hip

        native_status, native_reason = aten_hip.qualification(op_name, args, row["dtype"], vendor)
        if native_status == "eligible" and not _aten_candidate_boundary(candidate, args):
            native_status, native_reason = "pending", "candidate current-stream/preallocated launch boundary not verified"
        if native_status == "eligible":
            native_comparator = "aten_hip_out"
            prepare_native = partial(aten_hip.prepare, op_name, args)

    def time(function, operands, inputs_to_restore):
        calls = 0

        def observed(*values):
            nonlocal calls
            calls += 1
            return function(*values)

        if inputs_to_restore:
            latency, samples = _measure_stateful(observed, operands, inputs_to_restore)
        else:
            latency = _measure_ms(observed, operands, warmup_ms=warmup, measurement_ms=measurement)
            samples = []
        assert calls > 0, "Candidate timer did not execute the candidate"
        assert math.isfinite(latency) and latency > 0
        return latency, samples

    def time_native():
        nonlocal native_status, native_reason
        if native_status == "passed":
            try:
                latency, _ = time(baseline, baseline_args, [])
                baseline_samples.append(latency)
            except Exception as error:
                baseline_samples.clear()
                native_status, native_reason = "failed", str(error)

    try:
        if prepare_native is not None:
            try:
                baseline, baseline_args = prepare_native()
                baseline_version = baseline.version
                _validate_outputs(op_name, None, args, baseline, baseline_args)
                if native_comparator == "aten_hip_out" and not _check_stream_allocation(baseline, baseline_args):
                    raise RuntimeError("ATen changed the output, current stream or allocated temporary device memory")
                native_status, native_reason = "passed", None
            except Exception as error:
                native_status, native_reason = "failed", str(error)

        for sample in range(config.getoption("benchmark_samples")):
            if sample % 2:
                time_native()
            latency, samples = time(candidate, args, mutated)
            candidate_samples.append(latency)
            if mutated:
                stateful_samples.append(samples)
            if sample % 2 == 0:
                time_native()
    finally:
        if baseline is not None:
            baseline.close()

    config._evaluation.timings += 1
    latency = median(candidate_samples)
    native_latency = median(baseline_samples) if baseline_samples else None
    native_speedup = native_latency / latency if native_latency else None
    base = config.getoption("base_data")
    target_profile = config.getoption("target_profile")
    observations = read_json(target_profile).get("cases", {}) if target_profile else {}
    target = _target(args, vendor)
    external = external_result(base, row, latency, protocol, target,
                               observations.get(row["case_id"]))
    return {
        "latency": latency, "latency_base": native_latency, "speedup": native_speedup,
        "latency_samples_ms": candidate_samples, "native_latency_samples_ms": baseline_samples,
        "stateful_samples_ms": stateful_samples,
        "timing_protocol": protocol, "warmup_ms": None if mutated else warmup,
        "measurement_ms": None if mutated else measurement,
        "native_status": native_status, "native_reason": native_reason,
        "baseline_source": native_comparator if native_status == "passed" else None,
        "native_comparator": native_comparator,
        "baseline_version": baseline_version, "accuracy_checked": True,
        "device_identity": target,
        "kernel_dispatch": getattr(candidate, "dispatch", None),
        **external, **comparison(native_status, native_reason, native_speedup, external),
    }


def run(request, config, *, op_name, make_inputs):
    """Run the operator's Core cases from the shared inventory."""
    inventory = tuple(python_cases(op_name))
    rows = [row for _, row in inventory]
    state = config._evaluation
    if config.getoption("list_cases"):
        emit(
            config,
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "benchmarks": [
                    {
                        "op_name": op_name,
                        "level": "core",
                        "phase": "timing",
                        "cases": rows,
                    }
                ],
            },
        )
        return

    selected = getattr(config, "_flagcv_selected_cases", set())
    candidate = request.getfixturevalue("candidate")
    device = config.getoption("device")
    workloads = [(case, row) for case, row in inventory
                 if not selected or row["case_id"] in selected]
    probe_start = len(state.records)
    by_dtype = {}
    for case, row in workloads:
        case_id = row["case_id"]
        build = partial(make_inputs, case, device)
        if config.getoption("profile_only"):
            _profile(config, case_id, candidate, build)
            return
        if candidate.source == "dummy":
            state.run_probe(candidate, build, request.node.nodeid, case_id)
            continue

        args = build()
        timing = _time_case(config, op_name, candidate, args, row)
        result = {
            "case_id": case_id,
            "legacy_case_id": row["legacy_case_id"],
            "config": row["config"],
            "shape_detail": row["shape"],
            "params": row["params"],
            "candidate_source": candidate.source,
            "measurement_scope": "delivery" if candidate.source == "builtin" else "experiment",
            **timing,
        }
        by_dtype.setdefault(row["dtype"], []).append(result)

    if candidate.source == "dummy":
        expected = [row["case_id"] for _, row in workloads]
        observed = [record["case_id"] for record in state.records[probe_start:]]
        assert observed == expected, "Dummy records differ from the selected workloads"
        return

    details = [
        {"op_name": op_name, "dtype": dtype, "level": "core", "result": results,
         "summary": summarize(results)}
        for dtype, results in by_dtype.items()
    ]
    emit(
        config,
        {
            op_name: {
                "details": details,
                "result": "passed",
                "test_case": request.node.nodeid,
            }
        },
    )
