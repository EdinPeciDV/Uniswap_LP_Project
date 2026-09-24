"""Hourly backtest of Uniswap v3 LP strategies on real on-chain data.

Fee income is taken from the pool itself, not estimated from volume:
feeGrowthGlobal{0,1}X128 is the fee paid per unit of in-range liquidity, net
of the protocol fee. A position with liquidity L that stays in range over an
interval therefore earns exactly L * delta(feeGrowthGlobal). We scale that by
L_pool / (L_pool + L_ours) so our own liquidity dilutes the fees we'd get.

Simplifications (see README): hourly granularity (an interval half in range
earns half), fees are swept to USDC as they accrue, the hedge is an ETH perp
short at the pool price with Hyperliquid funding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .position import RangePosition
from .v3_math import Q128

L_SCALE = math.sqrt(1e18 * 1e6)   # human L (ETH x USDC) -> raw pool L (wei x USDC units)
HOURS_PER_YEAR = 24 * 365
MOD256 = 1 << 256


@dataclass
class Strategy:
    name: str
    width: float = math.inf          # symmetric range [p/(1+w), p(1+w)]; inf = full range
    rebalance: bool = False          # re-centre when price leaves the range
    hedge: bool = False              # short delta ETH on a perp
    hedge_every: int = 24            # hours between hedge rebalances
    kind: str = "lp"                 # "lp" or "hodl"


@dataclass
class Costs:
    swap_fee: float = 0.0005         # pool fee paid on the swap needed to re-centre
    gas_usd: float = 5.0             # burn + collect + swap + mint on mainnet
    perp_fee: float = 0.00045        # taker fee on hedge adjustments


@dataclass
class Result:
    strategy: Strategy
    equity: np.ndarray
    fees: float
    hedge_pnl: float
    funding: float
    costs: float
    rebalances: int
    time_in_range: float
    extra: dict = field(default_factory=dict)


@dataclass
class Market:
    """Arrays aligned on the hourly samples."""
    t: np.ndarray            # block timestamps
    p: np.ndarray            # ETH price in USDC
    L_pool: np.ndarray       # active liquidity (raw)
    fee_usd_per_L: np.ndarray  # USD fees per unit of *human* L over interval i -> i+1 (if in range)
    fee_share_lp: np.ndarray   # LP share of swap fees in each interval (1 or 3/4 after the fee switch)
    funding: np.ndarray        # funding rate paid by longs over interval i -> i+1

    @property
    def years(self) -> float:
        return (self.t[-1] - self.t[0]) / (365 * 24 * 3600)


def build_market(pool: pd.DataFrame, funding: pd.DataFrame | None = None) -> Market:
    p = pool["price"].to_numpy(float)
    fg0 = pool["fee_growth0_x128"].tolist()
    fg1 = pool["fee_growth1_x128"].tolist()
    d0 = np.array([((b - a) % MOD256) / Q128 for a, b in zip(fg0[:-1], fg0[1:])])  # USDC raw per raw L
    d1 = np.array([((b - a) % MOD256) / Q128 for a, b in zip(fg1[:-1], fg1[1:])])           # WETH raw per raw L
    fee_usd = L_SCALE * (d0 / 1e6 + d1 / 1e18 * p[1:])
    fp = pool["fee_protocol"].to_numpy()
    # feeProtocol packs token0 in the low 4 bits, token1 in the high 4; protocol takes 1/n of the fee
    n0 = fp[1:] % 16
    share = np.where(n0 > 0, 1 - 1 / np.where(n0 > 0, n0, 1), 1.0)
    f = np.zeros(len(p) - 1)
    if funding is not None:
        fmap = dict(zip(funding["hour"], funding["funding"]))
        f = np.array([fmap.get(h, 0.0) for h in pool["hour"].to_numpy()[1:]])
    return Market(t=pool["timestamp"].to_numpy(float), p=p,
                  L_pool=np.array([float(x) for x in pool["liquidity"]]),
                  fee_usd_per_L=fee_usd, fee_share_lp=share, funding=f)


def run(m: Market, s: Strategy, capital: float = 100_000.0, costs: Costs = Costs(),
        fee_multiplier: np.ndarray | float = 1.0) -> Result:
    """Simulate one strategy. fee_multiplier lets you run counterfactuals (e.g. no protocol fee)."""
    p = m.p
    n = len(p)
    if s.kind == "hodl":
        x0, y0 = capital / 2 / p[0], capital / 2
        eq = x0 * p + y0
        return Result(s, eq, 0.0, 0.0, 0.0, 0.0, 0, 1.0)

    pos = RangePosition.symmetric(capital, p[0], s.width)
    fees = hedge_pnl = funding_pnl = 0.0
    rebal_cost = hedge_cost = 0.0     # rebalance costs come out of the position, hedge costs out of cash
    rebalances, in_range_hours = 0, 0.0
    short = 0.0
    if s.hedge:
        short = pos.delta(p[0])
        hedge_cost += costs.perp_fee * short * p[0]
    eq = np.empty(n)
    eq[0] = capital - hedge_cost
    fm = np.broadcast_to(np.asarray(fee_multiplier, float), (n - 1,))

    for i in range(n - 1):
        p0, p1 = p[i], p[i + 1]
        frac = (float(pos.in_range(p0)) + float(pos.in_range(p1))) / 2  # numpy bools would OR, not add
        in_range_hours += frac
        dilution = m.L_pool[i] / (m.L_pool[i] + pos.L * L_SCALE)
        fees += pos.L * m.fee_usd_per_L[i] * frac * dilution * fm[i]

        if s.hedge:
            hedge_pnl += -short * (p1 - p0)
            funding_pnl += short * p0 * m.funding[i]   # positive funding: longs pay shorts

        if s.rebalance and not pos.in_range(p1):
            value = pos.value(p1)
            x, y = pos.amounts(p1)
            target = RangePosition.symmetric(value, p1, s.width)
            tx, ty = target.amounts(p1)
            swapped = abs(x - tx) * p1                  # notional we must swap to re-centre
            c = costs.swap_fee * swapped + costs.gas_usd
            rebal_cost += c
            pos = RangePosition.symmetric(value - c, p1, s.width)
            rebalances += 1
            if s.hedge:
                short, c = _rehedge(pos, p1, short, costs)
                hedge_cost += c
        elif s.hedge and (i + 1) % s.hedge_every == 0:
            short, c = _rehedge(pos, p1, short, costs)
            hedge_cost += c

        eq[i + 1] = pos.value(p1) + fees + hedge_pnl + funding_pnl - hedge_cost

    return Result(s, eq, fees, hedge_pnl, funding_pnl, rebal_cost + hedge_cost, rebalances,
                  in_range_hours / (n - 1))


def _rehedge(pos: RangePosition, p: float, short: float, costs: Costs) -> tuple[float, float]:
    target = pos.delta(p)
    return target, costs.perp_fee * abs(target - short) * p


def metrics(r: Result, m: Market, capital: float = 100_000.0) -> dict:
    eq = r.equity
    rets = np.diff(np.log(eq))
    yrs = m.years
    total = eq[-1] / capital - 1
    vol = rets.std() * math.sqrt(HOURS_PER_YEAR)
    peak = np.maximum.accumulate(eq)
    return {
        "strategy": r.strategy.name,
        "final_value": eq[-1],
        "total_return": total,
        "cagr": (eq[-1] / capital) ** (1 / yrs) - 1,
        "ann_vol": vol,
        "sharpe": (rets.mean() * HOURS_PER_YEAR) / vol if vol > 0 else float("nan"),
        "max_drawdown": (eq / peak - 1).min(),
        "fees": r.fees,
        "fee_apr": r.fees / capital / yrs,
        "hedge_pnl": r.hedge_pnl,
        "funding": r.funding,
        "costs": r.costs,
        "rebalances": r.rebalances,
        "time_in_range": r.time_in_range,
    }
