"""Replay real Uniswap v3 USDC/WETH 0.05% swaps with lp.v3_math, bit for bit.

Each v3 Swap event logs the pool's post-trade sqrtPriceX96 and active
liquidity, so the previous Swap gives us the exact pre-trade state (as long
as no Mint/Burn happened in between). For every swap that stays inside one
tick (liquidity unchanged) we rerun SwapMath.computeSwapStep as both an
exact-input trade, an exact-output trade and a partial fill stopped by a
price limit, and require that amount0, amount1 AND
the resulting sqrtPriceX96 all match the event exactly.

Usage: python scripts/verify_v3_swaps.py [--blocks 9000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp.chain import (RPC, TOPIC_V3_BURN, TOPIC_V3_MINT, TOPIC_V3_SWAP,  # noqa: E402
                      V3_USDC_WETH_005, decode_v3_swap)
from lp.v3_math import compute_swap_step, swap_within_tick  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEE_PIPS = 500  # 0.05%


def replay(sqrt_p: int, liq: int, s) -> str | None:
    """Return 'exact_in' / 'exact_out' if the event is reproduced exactly, else None."""
    zero_for_one = s.amount0 > 0
    # exact input: the positive leg was specified
    specified_in = s.amount0 if zero_for_one else s.amount1
    specified_out = s.amount1 if zero_for_one else s.amount0  # negative
    for kind, amt in (("exact_in", specified_in), ("exact_out", specified_out)):
        try:
            a0, a1, sp = swap_within_tick(sqrt_p, liq, amt, zero_for_one, FEE_PIPS)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        if (a0, a1, sp) == (s.amount0, s.amount1, s.sqrt_price_x96):
            return kind
    # partial fill: the swap stopped at the trader's sqrtPriceLimitX96. Then the step
    # reaches its target exactly, so the amounts are fully determined by (pre, post) price.
    step = compute_swap_step(sqrt_p, s.sqrt_price_x96, liq, specified_in, FEE_PIPS)
    if step.sqrt_price_next == s.sqrt_price_x96:
        paid, received = step.amount_in + step.fee_amount, -step.amount_out
        if (paid, received) == ((s.amount0, s.amount1) if zero_for_one else (s.amount1, s.amount0)):
            return "price_limit"
    return None


def main(n_blocks: int) -> dict:
    rpc = RPC()
    head = rpc.block_number()
    start = head - n_blocks
    logs = rpc.get_logs(V3_USDC_WETH_005, [[TOPIC_V3_SWAP, TOPIC_V3_MINT, TOPIC_V3_BURN]], start, head, chunk=500)
    logs.sort(key=lambda l: (int(l["blockNumber"], 16), int(l["logIndex"], 16)))

    counts: Counter[str] = Counter()
    fixtures = []
    state = None  # (sqrtPriceX96, liquidity) after the last swap, None if unknown
    for log in logs:
        if log["topics"][0] != TOPIC_V3_SWAP:
            state = None  # a Mint/Burn may have changed active liquidity
            continue
        s = decode_v3_swap(log)
        if state is None:
            counts["skipped: unknown pre-state"] += 1
        elif s.liquidity != state[1]:
            counts["crossed initialized tick(s)"] += 1
        else:
            kind = replay(state[0], state[1], s)
            counts[f"{kind}_match" if kind else "no_match"] += 1
            if kind and len(fixtures) < 25:
                fixtures.append({"block": s.block, "tx": s.tx, "sqrt_price_before": state[0],
                                 "liquidity": state[1], "amount0": s.amount0, "amount1": s.amount1,
                                 "sqrt_price_after": s.sqrt_price_x96, "kind": kind})
        state = (s.sqrt_price_x96, s.liquidity)

    matched = counts["exact_in_match"] + counts["exact_out_match"] + counts["price_limit_match"]
    replayable = matched + counts["no_match"]
    result = {"pool": V3_USDC_WETH_005, "from_block": start, "to_block": head,
              "swaps": sum(counts.values()), "breakdown": dict(counts.most_common()),
              "single_tick_swaps_replayed": replayable,
              "bit_exact_match_rate": matched / replayable if replayable else None}
    print(json.dumps(result, indent=2))
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "v3_verification.json").write_text(json.dumps(result, indent=2))
    (ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "fixtures" / "v3_swaps.json").write_text(json.dumps(fixtures, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=9_000, help="look-back window (~12s per block; free RPCs keep ~10k blocks of logs)")
    main(ap.parse_args().blocks)
