"""Verify the single-channel OpenCV-compatible subtract profile."""

from functools import partial

import pytest

from helpers.correctness import check
from helpers.tolerance import assert_exact, assert_relative_close
from workloads.subtract import CORRECTNESS_CASES, OP_NAME, case_id, make_inputs

@pytest.mark.subtract
@pytest.mark.parametrize(
    ("shape", "dtype", "fixed_inputs"),
    [pytest.param(*case, id=case_id(case)) for case in CORRECTNESS_CASES],
)
def test_subtract(
    shape, dtype, fixed_inputs, candidate, native_reference, pytestconfig
):
    case = (shape, dtype, fixed_inputs)
    assert_output = assert_exact if dtype == "uint8" else assert_relative_close
    check(
        OP_NAME,
        partial(make_inputs, case),
        candidate,
        native_reference,
        pytestconfig,
        source_count=2,
        assert_output=assert_output,
    )
