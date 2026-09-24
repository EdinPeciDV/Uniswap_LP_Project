"""Concentrated-liquidity LP positions (Uniswap v3) in float math.

Convention: x = risky asset (ETH), y = numeraire (USDC), p = y per x.
A position has liquidity L on the price range [pa, pb]. Full range is
pa -> 0, pb -> inf, which reduces exactly to the v2 formulas.

Liquidity is symmetric in the two tokens (L = sqrt(x*y) for full range),
so L in human units maps to the pool's raw L by a constant factor
sqrt(10**dec_x * 10**dec_y).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RangePosition:
    L: float
    pa: float
    pb: float

    # ------------------------------------------------------------- holdings
    def amounts(self, p: float) -> tuple[float, float]:
        """(x, y) held by the position at price p."""
        sa, sb = math.sqrt(self.pa), math.sqrt(self.pb) if math.isfinite(self.pb) else math.inf
        s = math.sqrt(p)
        if p <= self.pa:
            return self.L * (1 / sa - 1 / sb), 0.0
        if p >= self.pb:
            return 0.0, self.L * (sb - sa)
        return self.L * (1 / s - 1 / sb), self.L * (s - sa)

    def value(self, p: float) -> float:
        x, y = self.amounts(p)
        return x * p + y

    def delta(self, p: float) -> float:
        """dV/dp = x(p): ETH exposure to short if you want to be delta-neutral."""
        return self.amounts(p)[0]

    def in_range(self, p: float) -> bool:
        return self.pa < p < self.pb

    # --------------------------------------------------------- constructors
    @staticmethod
    def from_capital(capital: float, p: float, pa: float, pb: float) -> "RangePosition":
        """Position worth `capital` (in y units) at price p, with pa < p < pb."""
        if not pa < p < pb:
            raise ValueError("entry price must be inside the range")
        unit = RangePosition(1.0, pa, pb)
        return RangePosition(capital / unit.value(p), pa, pb)

    @staticmethod
    def symmetric(capital: float, p: float, width: float) -> "RangePosition":
        """Range [p/(1+w), p*(1+w)], geometric around p. width=inf -> full range."""
        if math.isinf(width):
            return RangePosition.from_capital(capital, p, 0.0, math.inf)
        return RangePosition.from_capital(capital, p, p / (1 + width), p * (1 + width))


def capital_efficiency(width: float) -> float:
    """Liquidity per $ of a symmetric range relative to full range, at the centre price.

    For [p/k, p*k] with k = 1+width this is 1 / (1 - 1/sqrt(k)).
    """
    k = 1 + width
    return 1.0 / (1.0 - 1.0 / math.sqrt(k))


def range_il(price_ratio: float, width: float) -> float:
    """IL vs HODL (fraction) for a symmetric range entered at the centre."""
    p0 = 1.0
    pos = RangePosition.symmetric(1.0, p0, width)
    x0, y0 = pos.amounts(p0)
    p1 = price_ratio
    hodl = x0 * p1 + y0
    return pos.value(p1) / hodl - 1.0
