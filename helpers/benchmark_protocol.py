"""Shared definitions for the Python kernel timing profiles."""
from typing import NamedTuple


class BenchmarkProtocol(NamedTuple):
    name: str
    warmup_ms: int
    samples: int


MEASUREMENT_MS = 100
PROFILES = {
    "default": BenchmarkProtocol("flagcv-python-triton-event-median/3", 100, 2),
    "long": BenchmarkProtocol("flagcv-python-triton-event-median/2", 1000, 3),
}
