"""Download one year of hourly on-chain pool state + perp funding into data/.

  data/pool_hourly.csv     sqrtPriceX96, tick, active liquidity, feeGrowthGlobal0/1,
                           protocol-fee setting and ETH price, read at the block at
                           each hour boundary (archive eth_call via Multicall3)
  data/funding_hourly.csv  Hyperliquid ETH perp hourly funding rate

Usage: python scripts/fetch_data.py [--days 365]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp.data import DATA, HOUR, fetch_hyperliquid_funding, fetch_pool_hourly  # noqa: E402


def main(days: int) -> None:
    DATA.mkdir(exist_ok=True)
    t = time.time()
    pool = fetch_pool_hourly(days)
    pool.to_csv(DATA / "pool_hourly.csv", index=False)
    print(f"pool: {len(pool)} hourly rows, blocks {pool.block.iloc[0]}..{pool.block.iloc[-1]} "
          f"({time.time() - t:.0f}s)")
    lag = (pool["timestamp"] - pool["hour"]).abs()
    print(f"block timestamp vs hour boundary: median {lag.median():.0f}s, max {lag.max():.0f}s")

    start_ms, end_ms = int(pool.hour.iloc[0]) * 1000, (int(pool.hour.iloc[-1]) + HOUR) * 1000
    funding = fetch_hyperliquid_funding(start_ms, end_ms)
    funding.to_csv(DATA / "funding_hourly.csv", index=False)
    print(f"funding: {len(funding)} hourly rows, mean {funding.funding.mean() * 24 * 365:.1%} APR")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    main(ap.parse_args().days)
