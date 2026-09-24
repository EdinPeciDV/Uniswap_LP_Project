"""Regression tests on real mainnet swaps captured by scripts/verify_*_swaps.py.

These run offline: the fixtures hold the pre-trade pool state and the amounts
the contracts actually produced, and our math must reproduce them to the wei.
"""
import json
from pathlib import Path

import pytest

from lp import v2
from lp.v3_math import swap_within_tick

FIX = Path(__file__).parent / "fixtures"
V2 = json.loads((FIX / "v2_swaps.json").read_text())
V3 = [f for f in json.loads((FIX / "v3_swaps.json").read_text()) if f["kind"] != "price_limit"]


@pytest.mark.parametrize("f", V2, ids=[f["tx"][:10] for f in V2])
def test_v2_mainnet_swap(f):
    if f["a0_in"]:
        a_in, a_out, r_in, r_out = f["a0_in"], f["a1_out"], f["r0"], f["r1"]
    else:
        a_in, a_out, r_in, r_out = f["a1_in"], f["a0_out"], f["r1"], f["r0"]
    if f["kind"] == "exact_in_match":
        assert v2.get_amount_out(a_in, r_in, r_out) == a_out
    else:
        assert v2.get_amount_in(a_out, r_in, r_out) == a_in


@pytest.mark.parametrize("f", V3, ids=[f["tx"][:10] for f in V3])
def test_v3_mainnet_swap(f):
    zero_for_one = f["amount0"] > 0
    if f["kind"] == "exact_in":
        specified = f["amount0"] if zero_for_one else f["amount1"]
    else:
        specified = f["amount1"] if zero_for_one else f["amount0"]
    got = swap_within_tick(f["sqrt_price_before"], f["liquidity"], specified, zero_for_one, 500)
    assert got == (f["amount0"], f["amount1"], f["sqrt_price_after"])
