import math
import pytest
from lp import v2
from lp.position import RangePosition, capital_efficiency, range_il


def test_full_range_reduces_to_v2():
    pos = RangePosition.symmetric(10_000, 2_000, math.inf)
    for p in (500, 1_000, 2_000, 4_000, 9_000):
        assert pos.value(p) == pytest.approx(v2.lp_value(pos.L, p))
        assert range_il(p / 2_000, math.inf) == pytest.approx(v2.impermanent_loss(p / 2_000))


def test_value_is_continuous_at_range_edges():
    pos = RangePosition.symmetric(10_000, 2_000, 0.10)
    for edge in (pos.pa, pos.pb):
        assert pos.value(edge * (1 - 1e-12)) == pytest.approx(pos.value(edge * (1 + 1e-12)), rel=1e-9)


def test_out_of_range_holds_single_asset():
    pos = RangePosition.symmetric(10_000, 2_000, 0.10)
    x, y = pos.amounts(pos.pa * 0.9)
    assert y == 0 and x > 0          # price fell below range -> all ETH
    x, y = pos.amounts(pos.pb * 1.1)
    assert x == 0 and y > 0          # price rose above range -> all USDC


def test_delta_is_derivative_of_value():
    pos = RangePosition.symmetric(10_000, 2_000, 0.2)
    p, h = 2_150.0, 1e-3
    fd = (pos.value(p + h) - pos.value(p - h)) / (2 * h)
    assert pos.delta(p) == pytest.approx(fd, rel=1e-6)


def test_capital_efficiency_matches_liquidity_ratio():
    for w in (0.01, 0.05, 0.2, 1.0):
        narrow = RangePosition.symmetric(1_000, 1_500, w)
        full = RangePosition.symmetric(1_000, 1_500, math.inf)
        assert narrow.L / full.L == pytest.approx(capital_efficiency(w))


def test_narrow_ranges_amplify_il():
    # same price move hurts a narrower range more (until the range is exited)
    il = [range_il(1.04, w) for w in (math.inf, 0.5, 0.1, 0.05)]
    assert il == sorted(il, reverse=True)
    assert all(v <= 0 for v in il)
