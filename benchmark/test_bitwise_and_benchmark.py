"""List, probe and time the bitwise_and workloads."""

import pytest

from helpers.benchmark import run
from workloads.bitwise_and import OP_NAME, make_inputs


@pytest.mark.bitwise_and
def test_bitwise_and_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        make_inputs=make_inputs,
    )
