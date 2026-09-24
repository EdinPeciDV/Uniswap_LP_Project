import math

import pytest

from lp.v3_math import Q96, amount0_delta, amount1_delta, compute_swap_step, swap_within_tick

SQRT_P = int(math.sqrt(1 / 2500 * 1e12) * Q96)   # ~2500 USDC/ETH in USDC(6)/WETH(18) pool units
L = 2 * 10**18


def test_amount_deltas_match_float_formulas():
    sb = SQRT_P * 101 // 100
    d0 = amount0_delta(SQRT_P, sb, L, False)
    d1 = amount1_delta(SQRT_P, sb, L, False)
    assert d0 == pytest.approx(L * (Q96 / SQRT_P - Q96 / sb), rel=1e-12)
    assert d1 == pytest.approx(L * (sb - SQRT_P) / Q96, rel=1e-12)
    # rounding up is never smaller and at most 1 unit larger
    assert 0 <= amount0_delta(SQRT_P, sb, L, True) - d0 <= 1
    assert 0 <= amount1_delta(SQRT_P, sb, L, True) - d1 <= 1


@pytest.mark.parametrize("zero_for_one", [True, False])
def test_exact_in_then_exact_out_roundtrip(zero_for_one):
    amt_in = 10**9 if zero_for_one else 10**17
    a0, a1, sp = swap_within_tick(SQRT_P, L, amt_in, zero_for_one, 500)
    out = -(a1 if zero_for_one else a0)
    b0, b1, sp2 = swap_within_tick(SQRT_P, L, -out, zero_for_one, 500)
    paid = b0 if zero_for_one else b1
    assert paid <= amt_in           # asking for the same output never costs more
    # exact-output charges (almost) the minimal input: paying it as exact-input yields the output
    # (at most one unit short from the two roundings), and paying 0.1% less clearly does not
    def out_for(x):
        c0, c1, _ = swap_within_tick(SQRT_P, L, x, zero_for_one, 500)
        return -(c1 if zero_for_one else c0)
    assert out - out_for(paid) <= 1
    assert out_for(paid * 999 // 1000) < out


def test_fee_is_charged_on_input():
    step = compute_swap_step(SQRT_P, 4295128740, L, 10**10, 500)
    assert step.fee_amount == pytest.approx(10**10 * 0.0005, rel=1e-3)
    assert step.amount_in + step.fee_amount == 10**10


def test_price_moves_the_right_way():
    _, _, down = swap_within_tick(SQRT_P, L, 10**9, True, 500)    # sell USDC for WETH -> P(WETH/USDC) falls
    _, _, up = swap_within_tick(SQRT_P, L, 10**17, False, 500)
    assert down < SQRT_P < up
