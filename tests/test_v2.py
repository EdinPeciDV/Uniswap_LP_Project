import math
import pytest
from lp import v2


def test_amount_out_matches_known_value():
    # 1 ETH into a 100 ETH / 200,000 USDC pool (raw units)
    out = v2.get_amount_out(10**18, 100 * 10**18, 200_000 * 10**6)
    # closed form: 997*200000e6 / (100*1000 + 997) with 1e18 scaling
    expected = (10**18 * 997 * 200_000 * 10**6) // (100 * 10**18 * 1000 + 10**18 * 997)
    assert out == expected


def test_amount_in_is_inverse_of_amount_out():
    r_in, r_out = 5_000 * 10**18, 12_000_000 * 10**6
    for want in (1, 10**6, 1_234_567_890, 10**12):
        a_in = v2.get_amount_in(want, r_in, r_out)
        assert v2.get_amount_out(a_in, r_in, r_out) >= want
        assert v2.get_amount_out(a_in - 1, r_in, r_out) < want  # minimal input


def test_k_never_decreases_and_invariant_check_passes():
    pair = v2.Pair(10**24, 3 * 10**15)
    k0 = pair.k
    for i in range(50):
        r0, r1 = pair.reserve0, pair.reserve1
        amt = 10**20 + i * 10**19
        out = pair.swap_exact_in(amt, zero_for_one=True)
        assert v2.k_invariant_holds(r0, r1, amt, 0, 0, out)
        assert not v2.k_invariant_holds(r0, r1, amt, 0, 0, out + 1)  # 1 wei more breaks k
        assert pair.k >= k0
        k0 = pair.k


@pytest.mark.parametrize("r,expected", [(1.0, 0.0), (4.0, -0.2), (0.25, -0.2)])
def test_impermanent_loss_closed_form(r, expected):
    assert v2.impermanent_loss(r) == pytest.approx(expected)


def test_il_matches_simulated_lp_vs_hodl():
    L, p0, p1 = 1_000.0, 2_000.0, 3_100.0
    hodl = (L / math.sqrt(p0)) * p1 + L * math.sqrt(p0)
    assert v2.lp_value(L, p1) / hodl - 1 == pytest.approx(v2.impermanent_loss(p1 / p0))


def test_delta_is_derivative_of_value():
    L, p, h = 500.0, 2500.0, 1e-3
    fd = (v2.lp_value(L, p + h) - v2.lp_value(L, p - h)) / (2 * h)
    assert v2.lp_delta(L, p) == pytest.approx(fd, rel=1e-8)
    fd2 = (v2.lp_value(L, p + 1) - 2 * v2.lp_value(L, p) + v2.lp_value(L, p - 1))
    assert v2.lp_gamma(L, p) == pytest.approx(fd2, rel=1e-5)
