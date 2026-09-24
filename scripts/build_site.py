"""Build the static results site: docs/index.html (single file, data inlined).

Reruns the backtest from data/, downsamples the series to daily points and
injects them into scripts/site_template.html. Deploy docs/ as-is: GitHub Pages (branch main,
folder /docs) or Vercel (Root Directory = docs).

Usage: python scripts/build_site.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lp.backtest import build_market, metrics, run  # noqa: E402
from lp.data import DATA, load_funding, load_pool_hourly  # noqa: E402
from run_backtest import CAPITAL, STRATEGIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"   # GitHub Pages serves /docs; Vercel: set Root Directory to docs


def main() -> None:
    m = build_market(load_pool_hourly(), load_funding())
    dates = pd.to_datetime(m.t, unit="s")
    step = 24
    idx = np.arange(0, len(m.t), step)
    if idx[-1] != len(m.t) - 1:
        idx = np.append(idx, len(m.t) - 1)

    results = {s.name: run(m, s, CAPITAL) for s in STRATEGIES}
    table = [metrics(r, m, CAPITAL) for r in results.values()]

    # rolling 30-day full-range fee APR vs sigma^2/8
    win = 24 * 30
    fee_yield = pd.Series(m.fee_usd_per_L / (2 * np.sqrt(m.p[1:])))
    fee_apr = fee_yield.rolling(win).sum() * (24 * 365 / win)
    rv = pd.Series(np.diff(np.log(m.p))).rolling(win).std() * math.sqrt(24 * 365)
    gidx = idx[idx >= win][:-1] if idx[-1] >= len(fee_apr) else idx[idx >= win]
    gidx = np.clip(gidx, 0, len(fee_apr) - 1)

    attribution = []
    for n in ["Full range, hedged", "±50% rebal., hedged", "±20% rebal., hedged", "±10% rebal., hedged"]:
        r = results[n]
        net = r.equity[-1] - CAPITAL
        attribution.append({"name": n, "fees": r.fees, "funding": r.funding,
                            "il": net - r.fees - r.funding + r.costs, "costs": -r.costs, "net": net})

    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in dates[idx]],
        "eth": [round(float(x), 2) for x in m.p[idx]],
        "equity": {k: [round(float(v), 0) for v in r.equity[idx]] for k, r in results.items()},
        "theta": {"dates": [d.strftime("%Y-%m-%d") for d in dates[1:][gidx]],
                  "fee_apr": [round(float(fee_apr.iloc[i]) * 100, 3) for i in gidx],
                  "il_drag": [round(float(rv.iloc[i] ** 2 / 8) * 100, 3) for i in gidx]},
        "attribution": attribution,
        "table": table,
        "summary": json.loads((DATA / "summary.json").read_text()),
        "v2": json.loads((DATA / "v2_verification.json").read_text()),
        "v3": json.loads((DATA / "v3_verification.json").read_text()),
    }
    html = (ROOT / "scripts" / "site_template.html").read_text()
    blob = json.dumps(payload, default=float, separators=(",", ":")).replace("</", "<\\/")
    (SITE / "index.html").write_text(html.replace("/*__DATA__*/null", blob))
    print(f"docs/index.html written ({len(blob) / 1024:.0f} KB of data)")


if __name__ == "__main__":
    main()
