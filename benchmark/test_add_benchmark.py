"""List, probe and time the add workloads."""

import pytest

from helpers.benchmark import run
from workloads.add import OP_NAME, make_inputs


@pytest.mark.add
def test_add_benchmark(request, pytestconfig):
    run(
        request,
        pytestconfig,
        op_name=OP_NAME,
        make_inputs=make_inputs,
    )
