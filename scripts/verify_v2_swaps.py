"""Replay real Uniswap v2 USDC/WETH swaps with lp.v2 and check them to the wei.

For each Swap event we take the reserves from the preceding Sync event (the
pool state right before the trade), then check:
  1. the pair's k-invariant holds with the 0.3% fee (it must - the contract enforces it)
  2. output == getAmountOut(input)      -> a standard exact-input router swap
  3. input  == getAmountIn(output)      -> a standard exact-output router swap
  4. anything else = trader took less than the max (aggregators, MEV bots, fee-on-transfer)

Usage: python scripts/verify_v2_swaps.py [--blocks 9000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp import v2  # noqa: E402
from lp.chain import RPC, TOPIC_V2_SWAP, TOPIC_V2_SYNC, V2_USDC_WETH, V2Swap, V2Sync, decode_v2_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def classify(pre: V2Sync, s: V2Swap) -> str:
    if not v2.k_invariant_holds(pre.r0, pre.r1, s.a0_in, s.a1_in, s.a0_out, s.a1_out):
        return "k_violated"
    one_sided = (s.a0_in > 0) != (s.a1_in > 0) and (s.a0_out > 0) != (s.a1_out > 0)
    if not one_sided:
        return "multi_sided (flash swap)"
    if s.a0_in:
        a_in, a_out, r_in, r_out = s.a0_in, s.a1_out, pre.r0, pre.r1
    else:
        a_in, a_out, r_in, r_out = s.a1_in, s.a0_out, pre.r1, pre.r0
    if a_out == v2.get_amount_out(a_in, r_in, r_out):
        return "exact_in_match"
    if a_out < r_out and a_in == v2.get_amount_in(a_out, r_in, r_out):
        return "exact_out_match"
    return "below_max_output"


def main(n_blocks: int) -> dict:
    rpc = RPC()
    head = rpc.block_number()
    start = head - n_blocks
    logs = rpc.get_logs(V2_USDC_WETH, [[TOPIC_V2_SWAP, TOPIC_V2_SYNC]], start, head)
    events = sorted((decode_v2_log(l) for l in logs), key=lambda e: (e.block, e.log_index))

    counts: Counter[str] = Counter()
    fixtures = []
    prev_sync = cur_sync = None
    for e in events:
        if isinstance(e, V2Sync):
            prev_sync, cur_sync = cur_sync, e
            continue
        if prev_sync is None:
            continue  # need the state before this trade
        # the Sync emitted by this swap sits right before it; its reserves must equal pre + in - out
        post_ok = (cur_sync.r0 == prev_sync.r0 + e.a0_in - e.a0_out and
                   cur_sync.r1 == prev_sync.r1 + e.a1_in - e.a1_out)
        if not post_ok:
            counts["state_mismatch (donation/skim)"] += 1
            continue
        c = classify(prev_sync, e)
        counts[c] += 1
        if c.endswith("match") and len(fixtures) < 25:
            fixtures.append({"block": e.block, "tx": e.tx, "r0": prev_sync.r0, "r1": prev_sync.r1,
                             "a0_in": e.a0_in, "a1_in": e.a1_in, "a0_out": e.a0_out, "a1_out": e.a1_out,
                             "kind": c})

    total = sum(counts.values())
    matched = counts["exact_in_match"] + counts["exact_out_match"]
    result = {"pool": V2_USDC_WETH, "from_block": start, "to_block": head, "swaps": total,
              "breakdown": dict(counts.most_common()),
              "router_formula_match_rate": matched / total if total else None,
              "k_invariant_violations": counts["k_violated"]}
    print(json.dumps(result, indent=2))
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "v2_verification.json").write_text(json.dumps(result, indent=2))
    (ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "fixtures" / "v2_swaps.json").write_text(json.dumps(fixtures, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=9_000, help="look-back window (~12s per block; free RPCs keep ~10k blocks of logs)")
    main(ap.parse_args().blocks)
