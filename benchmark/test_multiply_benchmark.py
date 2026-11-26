"""List, probe and time the multiply workloads."""

import pytest

from helpers.benchmark import run
from workloads.multiply import OP_NAME, make_inputs


@pytest.mark.multiply
def test_multiply_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        make_inputs=make_inputs,
    )
