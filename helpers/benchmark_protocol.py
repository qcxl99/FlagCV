"""Shared definitions for Python kernel timing."""

from typing import NamedTuple


class BenchmarkProtocol(NamedTuple):
    name: str
    warmup_ms: int
    samples: int


MEASUREMENT_MS = 100
STATEFUL_PROTOCOL = "no-graph-device-event-reset-outside-5warmup-25sample-median"
PROFILES = {
    "default": BenchmarkProtocol("triton-do-bench-no-graph-25ms-warmup-100ms-median", 25, 3),
    "long": BenchmarkProtocol("flagcv-python-triton-event-median/2", 1000, 3),
}
