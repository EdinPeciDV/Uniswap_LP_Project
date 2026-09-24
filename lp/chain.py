"""Minimal Ethereum JSON-RPC access: no web3 dependency, just requests.

Covers what this project needs: eth_call (incl. historical blocks via an
archive RPC), eth_getLogs with automatic range splitting, block lookups by
timestamp, and Multicall3 so one request reads several pool fields at once.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

# Public endpoints. drpc serves archive state (historical eth_call) on its free
# tier; publicnode is fast for recent blocks. Override with ETH_RPC_URL.
DEFAULT_RPCS = ["https://eth.drpc.org", "https://ethereum-rpc.publicnode.com"]

MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"

# ---- Uniswap mainnet addresses -------------------------------------------
V2_USDC_WETH = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"   # token0 USDC, token1 WETH, 0.30%
V3_USDC_WETH_005 = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # token0 USDC, token1 WETH, 0.05%

# ---- event topics (keccak256 of the signatures) ----------------------------
TOPIC_V2_SWAP = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
TOPIC_V2_SYNC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"
TOPIC_V3_SWAP = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
TOPIC_V3_MINT = "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
TOPIC_V3_BURN = "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"

# ---- function selectors ----------------------------------------------------
SEL_SLOT0 = "0x3850c7bd"
SEL_LIQUIDITY = "0x1a686502"
SEL_FEE_GROWTH0 = "0xf3058399"
SEL_FEE_GROWTH1 = "0x46141319"
SEL_GET_RESERVES = "0x0902f1ac"
SEL_AGGREGATE = "0x252dba42"  # Multicall3.aggregate((address,bytes)[])


class RPCError(RuntimeError):
    pass


class RPC:
    def __init__(self, urls: list[str] | None = None, timeout: float = 30, retries: int = 4):
        env = os.environ.get("ETH_RPC_URL")
        self.urls = [env] if env else (urls or DEFAULT_RPCS)
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self._id = 0

    def call(self, method: str, params: list, url: str | None = None):
        last = None
        urls = [url] if url else self.urls
        for attempt in range(self.retries):
            refusals = 0  # providers that answered but refused (range/archive limits): retrying won't help
            for u in urls:
                self._id += 1
                try:
                    r = self.session.post(u, json={"jsonrpc": "2.0", "id": self._id, "method": method,
                                                   "params": params}, timeout=self.timeout)
                    data = r.json()
                except (requests.RequestException, ValueError) as e:
                    last = e
                    continue
                if "result" in data:
                    return data["result"]
                last = RPCError(f"{u}: {data.get('error')}")
                msg = str(data.get("error", "")).lower()
                if any(s in msg for s in ("range", "too many", "limit", "exceed", "archive", "not supported")):
                    refusals += 1
            if refusals == len(urls):
                raise last  # every provider refused; for getLogs the caller splits the range
            time.sleep(0.5 * 2**attempt)
        raise RPCError(f"{method} failed after retries: {last}")

    # ------------------------------------------------------------ helpers
    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def block_timestamp(self, block: int) -> int:
        return int(self.call("eth_getBlockByNumber", [hex(block), False])["timestamp"], 16)

    def eth_call(self, to: str, data: str, block: int | str = "latest") -> str:
        tag = hex(block) if isinstance(block, int) else block
        return self.call("eth_call", [{"to": to, "data": data}, tag])

    def get_logs(self, address: str, topics: list, from_block: int, to_block: int, chunk: int = 2_000) -> list[dict]:
        """eth_getLogs over a block range, halving the chunk on provider range errors."""
        out, start = [], from_block
        while start <= to_block:
            end = min(start + chunk - 1, to_block)
            try:
                out += self.call("eth_getLogs", [{"address": address, "topics": topics,
                                                  "fromBlock": hex(start), "toBlock": hex(end)}])
                start = end + 1
            except RPCError:
                if chunk <= 10:
                    raise
                chunk //= 2
        return out

    def block_at_timestamp(self, ts: int, lo: int | None = None, hi: int | None = None) -> int:
        """Last block with timestamp <= ts (binary search; post-Merge 12s slots give a tight first guess)."""
        head = self.block_number()
        head_ts = self.block_timestamp(head)
        guess = head - (head_ts - ts) // 12
        lo = lo if lo is not None else max(guess - 3_000, 0)
        hi = hi if hi is not None else min(guess + 3_000, head)
        while self.block_timestamp(lo) > ts:
            lo -= 5_000
        while hi < head and self.block_timestamp(hi) <= ts:
            hi = min(hi + 5_000, head)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.block_timestamp(mid) <= ts:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def multicall(self, calls: list[tuple[str, str]], block: int | str = "latest") -> list[bytes]:
        """Multicall3.aggregate: many (target, calldata) reads in a single eth_call."""
        raw = self.eth_call(MULTICALL3, encode_aggregate(calls), block)
        return decode_aggregate(raw)


# ----------------------------------------------------------------- ABI bits
def _word(x: int) -> str:
    return f"{x:064x}"


def _signed(x: int, bits: int = 256) -> int:
    return x - (1 << bits) if x >> (bits - 1) else x


def encode_aggregate(calls: list[tuple[str, str]]) -> str:
    """ABI-encode aggregate((address target, bytes callData)[])."""
    n = len(calls)
    tuples = []
    for target, data in calls:
        payload = bytes.fromhex(data.removeprefix("0x"))
        padded = payload.hex() + "00" * ((32 - len(payload) % 32) % 32)
        tuples.append(_word(int(target, 16)) + _word(0x40) + _word(len(payload)) + padded)
    offsets, pos = [], 32 * n
    for t in tuples:
        offsets.append(_word(pos))
        pos += len(t) // 2
    body = _word(n) + "".join(offsets) + "".join(tuples)
    return SEL_AGGREGATE + _word(0x20) + body


def decode_aggregate(raw: str) -> list[bytes]:
    b = bytes.fromhex(raw.removeprefix("0x"))
    word = lambda i: int.from_bytes(b[i:i + 32], "big")
    arr = word(32)                   # offset of bytes[] (after blockNumber)
    n = word(arr)
    base = arr + 32
    out = []
    for i in range(n):
        off = base + word(base + 32 * i)
        ln = word(off)
        out.append(b[off + 32: off + 32 + ln])
    return out


def words(data: bytes | str) -> list[int]:
    if isinstance(data, str):
        data = bytes.fromhex(data.removeprefix("0x"))
    return [int.from_bytes(data[i:i + 32], "big") for i in range(0, len(data), 32)]


# ------------------------------------------------------------ pool readers
@dataclass
class V3State:
    block: int
    sqrt_price_x96: int
    tick: int
    fee_protocol: int
    liquidity: int
    fee_growth0_x128: int
    fee_growth1_x128: int


def read_v3_state(rpc: RPC, pool: str, block: int | str = "latest") -> V3State:
    s0, liq, fg0, fg1 = rpc.multicall([(pool, SEL_SLOT0), (pool, SEL_LIQUIDITY),
                                      (pool, SEL_FEE_GROWTH0), (pool, SEL_FEE_GROWTH1)], block)
    w = words(s0)
    return V3State(block=block if isinstance(block, int) else -1,
                   sqrt_price_x96=w[0], tick=_signed(w[1]), fee_protocol=w[5],
                   liquidity=words(liq)[0], fee_growth0_x128=words(fg0)[0], fee_growth1_x128=words(fg1)[0])


@dataclass
class V2Swap:
    block: int
    log_index: int
    tx: str
    a0_in: int
    a1_in: int
    a0_out: int
    a1_out: int


@dataclass
class V2Sync:
    block: int
    log_index: int
    r0: int
    r1: int


def decode_v2_log(log: dict):
    t0 = log["topics"][0]
    w = words(log["data"])
    blk, li = int(log["blockNumber"], 16), int(log["logIndex"], 16)
    if t0 == TOPIC_V2_SWAP:
        return V2Swap(blk, li, log["transactionHash"], *w[:4])
    if t0 == TOPIC_V2_SYNC:
        return V2Sync(blk, li, w[0], w[1])
    return None


@dataclass
class V3Swap:
    block: int
    log_index: int
    tx: str
    amount0: int
    amount1: int
    sqrt_price_x96: int
    liquidity: int
    tick: int


def decode_v3_swap(log: dict) -> V3Swap:
    w = words(log["data"])
    return V3Swap(int(log["blockNumber"], 16), int(log["logIndex"], 16), log["transactionHash"],
                  _signed(w[0]), _signed(w[1]), w[2], w[3], _signed(w[4]))
