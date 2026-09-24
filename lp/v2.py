"""Uniswap v2 constant-product AMM, reproduced from the Solidity contracts.

Two layers:
  * integer functions that match UniswapV2Library / UniswapV2Pair to the wei
  * float helpers for LP economics (value, impermanent loss, delta)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FEE_NUM = 997   # 0.30% fee: amountIn * 997 / 1000
FEE_DEN = 1000


# --------------------------------------------------------------------------
# Exact integer math (UniswapV2Library.sol)
# --------------------------------------------------------------------------
def get_amount_out(amount_in: int, reserve_in: int, reserve_out: int) -> int:
    """Max output for an exact input. Identical to UniswapV2Library.getAmountOut."""
    if amount_in <= 0:
        raise ValueError("INSUFFICIENT_INPUT_AMOUNT")
    if reserve_in <= 0 or reserve_out <= 0:
        raise ValueError("INSUFFICIENT_LIQUIDITY")
    amount_in_with_fee = amount_in * FEE_NUM
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in * FEE_DEN + amount_in_with_fee
    return numerator // denominator


def get_amount_in(amount_out: int, reserve_in: int, reserve_out: int) -> int:
    """Min input for an exact output. Identical to UniswapV2Library.getAmountIn."""
    if amount_out <= 0:
        raise ValueError("INSUFFICIENT_OUTPUT_AMOUNT")
    if reserve_in <= 0 or reserve_out <= amount_out:
        raise ValueError("INSUFFICIENT_LIQUIDITY")
    numerator = reserve_in * amount_out * FEE_DEN
    denominator = (reserve_out - amount_out) * FEE_NUM
    return numerator // denominator + 1


def k_invariant_holds(r0: int, r1: int, a0_in: int, a1_in: int, a0_out: int, a1_out: int) -> bool:
    """The check UniswapV2Pair.swap() enforces: fee-adjusted balances must not shrink k."""
    b0 = r0 + a0_in - a0_out
    b1 = r1 + a1_in - a1_out
    b0_adj = b0 * 1000 - a0_in * 3
    b1_adj = b1 * 1000 - a1_in * 3
    return b0_adj * b1_adj >= r0 * r1 * 1000**2


# --------------------------------------------------------------------------
# A small stateful pool, useful for simulation and tests
# --------------------------------------------------------------------------
@dataclass
class Pair:
    reserve0: int
    reserve1: int

    def swap_exact_in(self, amount_in: int, zero_for_one: bool) -> int:
        r_in, r_out = (self.reserve0, self.reserve1) if zero_for_one else (self.reserve1, self.reserve0)
        out = get_amount_out(amount_in, r_in, r_out)
        if zero_for_one:
            self.reserve0 += amount_in
            self.reserve1 -= out
        else:
            self.reserve1 += amount_in
            self.reserve0 -= out
        return out

    @property
    def k(self) -> int:
        return self.reserve0 * self.reserve1


# --------------------------------------------------------------------------
# LP economics (floats). Price p = token1 per token0 (e.g. USDC per ETH).
# --------------------------------------------------------------------------
def impermanent_loss(price_ratio: float) -> float:
    """IL of a full-range LP vs holding, as a fraction (negative = loss).

    IL(r) = 2*sqrt(r)/(1+r) - 1, where r = P_now / P_entry.
    """
    r = price_ratio
    return 2.0 * math.sqrt(r) / (1.0 + r) - 1.0


def lp_value(L: float, p: float) -> float:
    """Value (in token1 units) of liquidity L at price p. x=L/sqrt(p), y=L*sqrt(p) -> V=2L*sqrt(p)."""
    return 2.0 * L * math.sqrt(p)


def lp_delta(L: float, p: float) -> float:
    """dV/dp = amount of token0 held = L/sqrt(p). This is the hedge ratio."""
    return L / math.sqrt(p)


def lp_gamma(L: float, p: float) -> float:
    """d2V/dp2 = -L / (2 p^1.5). Always negative: an LP is short gamma."""
    return -0.5 * L / p**1.5


def breakeven_vol(fee_apr: float) -> float:
    """Annualised volatility at which fees exactly pay for IL (continuous-time approx).

    For a full-range LP, expected IL drag per year ~ sigma^2 / 8, so fees cover
    it when fee_apr >= sigma^2 / 8  ->  sigma* = sqrt(8 * fee_apr).
    """
    return math.sqrt(8.0 * fee_apr)
