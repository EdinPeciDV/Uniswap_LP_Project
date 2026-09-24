"""Historical data: hourly Uniswap v3 pool state from an archive node, and
hourly ETH perp funding from Hyperliquid (used to cost the delta hedge)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .chain import (MULTICALL3, RPC, SEL_FEE_GROWTH0, SEL_FEE_GROWTH1, SEL_LIQUIDITY, SEL_SLOT0,
                    V3_USDC_WETH_005, _signed, words)
from .v3_math import Q96

SEL_BLOCK_TS = "0x0f28c97d"  # Multicall3.getCurrentBlockTimestamp()
HOUR = 3600
DATA = Path(__file__).resolve().parents[1] / "data"


def hourly_blocks(rpc: RPC, start_ts: int, end_ts: int, anchor_every: int = 7 * 24 * HOUR) -> pd.DataFrame:
    """Block number at each hour boundary.

    Exact binary-search anchors every `anchor_every` seconds, linear
    interpolation in between (post-Merge blocks are ~12s, so the error is a
    few blocks); the true timestamp of every sampled block is recorded later.
    """
    anchor_ts = list(range(start_ts, end_ts, anchor_every)) + [end_ts]
    anchors = [rpc.block_at_timestamp(t) for t in anchor_ts]
    hours = np.arange(start_ts, end_ts + 1, HOUR)
    blocks = np.interp(hours, anchor_ts, anchors).round().astype(int)
    return pd.DataFrame({"hour": hours, "block": blocks})


def fetch_pool_hourly(days: int = 365, pool: str = V3_USDC_WETH_005, workers: int = 8,
                      end_ts: int | None = None) -> pd.DataFrame:
    rpc = RPC()
    end_ts = end_ts or (int(time.time()) // HOUR) * HOUR - HOUR
    start_ts = end_ts - days * 24 * HOUR
    grid = hourly_blocks(rpc, start_ts, end_ts)
    calls = [(pool, SEL_SLOT0), (pool, SEL_LIQUIDITY), (pool, SEL_FEE_GROWTH0),
             (pool, SEL_FEE_GROWTH1), (MULTICALL3, SEL_BLOCK_TS)]

    def read(block: int):
        s0, liq, fg0, fg1, ts = rpc.multicall(calls, int(block))
        w = words(s0)
        return {"block": int(block), "timestamp": words(ts)[0], "sqrt_price_x96": w[0], "tick": _signed(w[1]),
                "fee_protocol": w[5], "liquidity": words(liq)[0],
                "fee_growth0_x128": words(fg0)[0], "fee_growth1_x128": words(fg1)[0]}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(read, grid["block"]))
    df = pd.DataFrame(rows)
    df.insert(0, "hour", grid["hour"].values)
    # ETH price in USDC: pool price is WETH per USDC in raw units (token0=USDC 6dp, token1=WETH 18dp)
    df["price"] = [1.0 / ((int(s) / Q96) ** 2 * 1e-12) for s in df["sqrt_price_x96"]]
    for c in ("sqrt_price_x96", "liquidity", "fee_growth0_x128", "fee_growth1_x128"):
        df[c] = df[c].astype(str)  # uint256 doesn't fit in int64; keep exact
    return df


def fetch_hyperliquid_funding(start_ms: int, end_ms: int, coin: str = "ETH") -> pd.DataFrame:
    """Hourly funding rates for the Hyperliquid ETH perp (positive = longs pay shorts)."""
    out, cursor = [], start_ms
    while cursor < end_ms:
        r = requests.post("https://api.hyperliquid.xyz/info", timeout=30,
                          json={"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": end_ms})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out += batch
        cursor = batch[-1]["time"] + 1
        time.sleep(0.2)
    df = pd.DataFrame(out)
    df["hour"] = (df["time"] // 1000 // HOUR) * HOUR
    df["funding"] = df["fundingRate"].astype(float)
    return df.groupby("hour", as_index=False)["funding"].sum()


def load_pool_hourly(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "pool_hourly.csv")
    for c in ("sqrt_price_x96", "liquidity", "fee_growth0_x128", "fee_growth1_x128"):
        df[c] = df[c].astype(str).map(int)
    return df


def load_funding(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA / "funding_hourly.csv")
