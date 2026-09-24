"""Uniswap v3 swap math in exact integer arithmetic.

Ported from v3-core: FullMath, UnsafeMath, SqrtPriceMath and SwapMath.
Python ints are arbitrary precision, so we emulate the uint256/uint160
overflow checks explicitly where the Solidity code depends on them.
"""
from __future__ import annotations

from dataclasses import dataclass

Q96 = 1 << 96
Q128 = 1 << 128
MAX_UINT256 = (1 << 256) - 1
MAX_UINT160 = (1 << 160) - 1
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342


# ---------------------------------------------------------------- FullMath
def mul_div(a: int, b: int, d: int) -> int:
    if d == 0:
        raise ZeroDivisionError
    r = a * b // d
    if r > MAX_UINT256:
        raise OverflowError("mulDiv overflow")
    return r


def mul_div_rounding_up(a: int, b: int, d: int) -> int:
    r = mul_div(a, b, d)
    if (a * b) % d:
        r += 1
    return r


def div_rounding_up(a: int, b: int) -> int:
    return a // b + (1 if a % b else 0)


# ----------------------------------------------------------- SqrtPriceMath
def next_sqrt_price_from_amount0_rounding_up(sqrt_p: int, liquidity: int, amount: int, add: bool) -> int:
    if amount == 0:
        return sqrt_p
    numerator1 = liquidity << 96
    product = amount * sqrt_p
    if add:
        if product <= MAX_UINT256:
            denominator = numerator1 + product
            if denominator <= MAX_UINT256:  # the Solidity `denominator >= numerator1` no-overflow check
                return mul_div_rounding_up(numerator1, sqrt_p, denominator)
        return div_rounding_up(numerator1, numerator1 // sqrt_p + amount)
    if product > MAX_UINT256 or numerator1 <= product:
        raise ValueError("price would go to or below zero")
    return mul_div_rounding_up(numerator1, sqrt_p, numerator1 - product)


def next_sqrt_price_from_amount1_rounding_down(sqrt_p: int, liquidity: int, amount: int, add: bool) -> int:
    if add:
        quotient = (amount << 96) // liquidity if amount <= MAX_UINT160 else mul_div(amount, Q96, liquidity)
        res = sqrt_p + quotient
        if res > MAX_UINT160:
            raise OverflowError
        return res
    quotient = div_rounding_up(amount << 96, liquidity) if amount <= MAX_UINT160 else mul_div_rounding_up(amount, Q96, liquidity)
    if sqrt_p <= quotient:
        raise ValueError("price would go to or below zero")
    return sqrt_p - quotient


def next_sqrt_price_from_input(sqrt_p: int, liquidity: int, amount_in: int, zero_for_one: bool) -> int:
    if zero_for_one:
        return next_sqrt_price_from_amount0_rounding_up(sqrt_p, liquidity, amount_in, True)
    return next_sqrt_price_from_amount1_rounding_down(sqrt_p, liquidity, amount_in, True)


def next_sqrt_price_from_output(sqrt_p: int, liquidity: int, amount_out: int, zero_for_one: bool) -> int:
    if zero_for_one:
        return next_sqrt_price_from_amount1_rounding_down(sqrt_p, liquidity, amount_out, False)
    return next_sqrt_price_from_amount0_rounding_up(sqrt_p, liquidity, amount_out, False)


def amount0_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    numerator1 = liquidity << 96
    numerator2 = sqrt_b - sqrt_a
    if round_up:
        return div_rounding_up(mul_div_rounding_up(numerator1, numerator2, sqrt_b), sqrt_a)
    return mul_div(numerator1, numerator2, sqrt_b) // sqrt_a


def amount1_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if round_up:
        return mul_div_rounding_up(liquidity, sqrt_b - sqrt_a, Q96)
    return mul_div(liquidity, sqrt_b - sqrt_a, Q96)


# ---------------------------------------------------------------- SwapMath
@dataclass
class SwapStep:
    sqrt_price_next: int
    amount_in: int
    amount_out: int
    fee_amount: int


def compute_swap_step(sqrt_current: int, sqrt_target: int, liquidity: int,
                      amount_remaining: int, fee_pips: int) -> SwapStep:
    """SwapMath.computeSwapStep. amount_remaining > 0 = exact input, < 0 = exact output."""
    zero_for_one = sqrt_current >= sqrt_target
    exact_in = amount_remaining >= 0
    amount_in = amount_out = 0

    if exact_in:
        remaining_less_fee = mul_div(amount_remaining, 10**6 - fee_pips, 10**6)
        amount_in = (amount0_delta(sqrt_target, sqrt_current, liquidity, True) if zero_for_one
                     else amount1_delta(sqrt_current, sqrt_target, liquidity, True))
        if remaining_less_fee >= amount_in:
            sqrt_next = sqrt_target
        else:
            sqrt_next = next_sqrt_price_from_input(sqrt_current, liquidity, remaining_less_fee, zero_for_one)
    else:
        amount_out = (amount1_delta(sqrt_target, sqrt_current, liquidity, False) if zero_for_one
                      else amount0_delta(sqrt_current, sqrt_target, liquidity, False))
        if -amount_remaining >= amount_out:
            sqrt_next = sqrt_target
        else:
            sqrt_next = next_sqrt_price_from_output(sqrt_current, liquidity, -amount_remaining, zero_for_one)

    reached_max = sqrt_target == sqrt_next
    if zero_for_one:
        if not (reached_max and exact_in):
            amount_in = amount0_delta(sqrt_next, sqrt_current, liquidity, True)
        if not (reached_max and not exact_in):
            amount_out = amount1_delta(sqrt_next, sqrt_current, liquidity, False)
    else:
        if not (reached_max and exact_in):
            amount_in = amount1_delta(sqrt_current, sqrt_next, liquidity, True)
        if not (reached_max and not exact_in):
            amount_out = amount0_delta(sqrt_current, sqrt_next, liquidity, False)

    if not exact_in and amount_out > -amount_remaining:
        amount_out = -amount_remaining

    if exact_in and sqrt_next != sqrt_target:
        fee_amount = amount_remaining - amount_in
    else:
        fee_amount = mul_div_rounding_up(amount_in, fee_pips, 10**6 - fee_pips)
    return SwapStep(sqrt_next, amount_in, amount_out, fee_amount)


def swap_within_tick(sqrt_p: int, liquidity: int, amount_specified: int, zero_for_one: bool,
                     fee_pips: int) -> tuple[int, int, int]:
    """Simulate a swap that does not cross an initialized tick.

    Returns (amount0, amount1, sqrt_price_after) with the pool's sign convention
    (positive = paid into the pool, negative = paid out), exactly as the Swap event logs them.
    """
    target = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
    step = compute_swap_step(sqrt_p, target, liquidity, amount_specified, fee_pips)
    if amount_specified >= 0:
        specified_used = step.amount_in + step.fee_amount
        calculated = -step.amount_out
    else:
        specified_used = -step.amount_out
        calculated = step.amount_in + step.fee_amount
    # the "specified" side is token0 when (zero_for_one == exact_in)
    if zero_for_one == (amount_specified >= 0):
        amount0, amount1 = specified_used, calculated
    else:
        amount0, amount1 = calculated, specified_used
    return amount0, amount1, step.sqrt_price_next


# ----------------------------------------------------------- conversions
def sqrt_price_x96_to_price(sqrt_price_x96: int, dec0: int, dec1: int) -> float:
    """Human price of token0 in units of token1."""
    return (sqrt_price_x96 / Q96) ** 2 * 10 ** (dec0 - dec1)
