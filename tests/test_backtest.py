import math

import numpy as np
import pytest

from lp.backtest import Costs, Market, Strategy, metrics, run
from lp.position import RangePosition

N = 49


def flat_market(price=2000.0, fee_per_L=0.01, funding=0.0, prices=None):
    p = np.full(N, price) if prices is None else np.asarray(prices, float)
    return Market(t=np.arange(len(p)) * 3600.0, p=p, L_pool=np.full(len(p), 1e30),
                  fee_usd_per_L=np.full(len(p) - 1, fee_per_L), fee_share_lp=np.ones(len(p) - 1),
                  funding=np.full(len(p) - 1, funding))


def test_hodl_tracks_half_eth():
    m = flat_market(prices=np.linspace(2000, 3000, N))
    r = run(m, Strategy("h", kind="hodl"), 10_000)
    assert r.equity[-1] == pytest.approx(5_000 * 3000 / 2000 + 5_000)


def test_flat_price_earns_exactly_L_times_fee_growth():
    m = flat_market()
    r = run(m, Strategy("full"), 10_000)
    L = 10_000 / (2 * math.sqrt(2000))
    assert r.fees == pytest.approx(L * 0.01 * (N - 1))
    assert r.equity[-1] == pytest.approx(10_000 + r.fees)


def test_out_of_range_position_earns_nothing():
    prices = [2000] + [3000] * (N - 1)            # jumps above a ±10% range and stays there
    r = run(flat_market(prices=prices), Strategy("narrow", 0.10), 10_000)
    # only the first interval, in range at its start but not its end, earns (half credit)
    L = RangePosition.symmetric(10_000, 2000, 0.10).L
    assert r.fees == pytest.approx(0.5 * L * 0.01)
    assert r.time_in_range < 0.05


def test_rebalance_recenters_and_pays_costs():
    prices = [2000] * 10 + [2500] * (N - 10)
    r = run(flat_market(prices=prices), Strategy("rb", 0.10, rebalance=True), 10_000,
            costs=Costs(swap_fee=0.001, gas_usd=10))
    assert r.rebalances == 1
    assert r.time_in_range > 0.95
    assert r.costs > 10


def test_hedge_removes_first_order_price_risk():
    prices = 2000 * np.exp(np.linspace(0, 0.02, N))     # small steady move
    unhedged = run(flat_market(fee_per_L=0, prices=prices), Strategy("u"), 10_000, costs=Costs(perp_fee=0))
    hedged = run(flat_market(fee_per_L=0, prices=prices), Strategy("h", hedge=True, hedge_every=1), 10_000,
                 costs=Costs(perp_fee=0))
    assert abs(hedged.equity[-1] - 10_000) < 0.01 * abs(unhedged.equity[-1] - 10_000)


def test_positive_funding_pays_the_short():
    r = run(flat_market(fee_per_L=0, funding=1e-4), Strategy("h", hedge=True), 10_000, costs=Costs(perp_fee=0))
    assert r.funding > 0 and r.equity[-1] > 10_000


def test_metrics_keys():
    m = flat_market(prices=np.linspace(2000, 2100, N))
    d = metrics(run(m, Strategy("full"), 10_000), m, 10_000)
    assert {"total_return", "sharpe", "max_drawdown", "fee_apr"} <= d.keys()
