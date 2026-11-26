"""List, probe and time the flip workloads."""

import pytest

from helpers.benchmark import run
from workloads.flip import OP_NAME, make_inputs


@pytest.mark.flip
def test_flip_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        make_inputs=make_inputs,
    )
