"""Run every strategy on the saved data, write the results table and figures.

Outputs:
  data/results.csv, data/results.md   metrics for every strategy
  data/summary.json                   headline numbers quoted in the README
  figures/*.png

Usage: python scripts/run_backtest.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp.backtest import Strategy, build_market, metrics, run  # noqa: E402
from lp.data import DATA, load_funding, load_pool_hourly  # noqa: E402
from lp.position import range_il  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
CAPITAL = 100_000.0

# categorical slots in fixed order + a neutral for the benchmark
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8983"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

STRATEGIES = [
    Strategy("HODL 50/50", kind="hodl"),
    Strategy("Full range"),
    Strategy("±50% static", 0.50),
    Strategy("±20% static", 0.20),
    Strategy("±10% static", 0.10),
    Strategy("±50% rebalanced", 0.50, rebalance=True),
    Strategy("±20% rebalanced", 0.20, rebalance=True),
    Strategy("±10% rebalanced", 0.10, rebalance=True),
    Strategy("±5% rebalanced", 0.05, rebalance=True),
    Strategy("Full range, hedged", hedge=True),
    Strategy("±50% rebal., hedged", 0.50, rebalance=True, hedge=True),
    Strategy("±20% rebal., hedged", 0.20, rebalance=True, hedge=True),
    Strategy("±10% rebal., hedged", 0.10, rebalance=True, hedge=True),
]


def style(ax, title=None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    pool, funding = load_pool_hourly(), load_funding()
    m = build_market(pool, funding)
    dates = pd.to_datetime(m.t, unit="s")

    results = {s.name: run(m, s, CAPITAL) for s in STRATEGIES}
    table = pd.DataFrame([metrics(r, m, CAPITAL) for r in results.values()])
    table.to_csv(DATA / "results.csv", index=False)

    # ---------------- markdown table
    def pct(x):
        return f"{x:+.1%}"
    md = ["| Strategy | Return | Fee APR | Fees $ | Hedge + funding $ | Costs $ | Rebal. | Ann. vol | Max DD |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        hf = r.hedge_pnl + r.funding
        md.append(f"| {r.strategy} | {pct(r.total_return)} | {r.fee_apr:.1%} | {r.fees:,.0f} | "
                  f"{hf:,.0f} | {r.costs:,.0f} | {r.rebalances} | {r.ann_vol:.1%} | {r.max_drawdown:.1%} |")
    (DATA / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---------------- headline numbers
    eth_vol = np.diff(np.log(m.p)).std() * math.sqrt(24 * 365)
    switch_idx = int(np.argmax(m.fee_share_lp < 1))
    counterfactual = np.where(m.fee_share_lp < 1, 1 / m.fee_share_lp, 1.0)
    no_switch = {name: run(m, s, CAPITAL, fee_multiplier=counterfactual).fees
                 for name, s in [(s.name, s) for s in STRATEGIES if s.kind == "lp"]}
    summary = {
        "period_start": str(dates[0]), "period_end": str(dates[-1]),
        "eth_start": m.p[0], "eth_end": m.p[-1], "eth_return": m.p[-1] / m.p[0] - 1,
        "eth_realized_vol": eth_vol, "breakeven_fee_apr_full_range": eth_vol**2 / 8,
        "mean_funding_apr": float(m.funding.mean() * 24 * 365),
        # the switch flipped between these two hourly samples
        "fee_switch_between": [str(dates[switch_idx]), str(dates[switch_idx + 1])],
        "fee_switch_blocks": [int(pool.block.iloc[switch_idx]), int(pool.block.iloc[switch_idx + 1])],
        "fees_lost_to_protocol_fee": {k: no_switch[k] - results[k].fees for k in no_switch},
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))

    # ---------------- figure 1: equity curves, unhedged vs hedged
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panels = [("Unhedged LP positions vs holding", ["Full range", "±20% rebalanced", "±5% rebalanced"]),
              ("Delta-hedged LP positions", ["Full range, hedged", "±50% rebal., hedged", "±20% rebal., hedged",
                                             "±10% rebal., hedged"])]
    for ax, (title, names) in zip(axes, panels):
        style(ax, title)
        hodl = results["HODL 50/50"].equity / 1000
        if ax is axes[0]:
            ax.plot(dates, hodl, color=NEUTRAL, linewidth=2, linestyle=(0, (4, 2)),
                    label=f"HODL 50/50 ({hodl[-1] / 100 - 1:+.0%})")
        for c, n in zip(C, names):
            y = results[n].equity / 1000
            ax.plot(dates, y, color=c, linewidth=1.6, label=f"{n} ({y[-1] / 100 - 1:+.0%})")
            ax.plot(dates[-1], y[-1], "o", color=c, markersize=5, markeredgecolor="white", markeredgewidth=1.5)
        ax.axhline(100, color=INK2, linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    axes[0].set_ylabel("Portfolio value ($k, started at 100)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "equity_curves.png", dpi=160)
    plt.close(fig)

    # ---------------- figure 2: P&L attribution for hedged strategies
    names = ["Full range, hedged", "±50% rebal., hedged", "±20% rebal., hedged", "±10% rebal., hedged"]
    comp = []
    for n in names:
        r = results[n]
        net = r.equity[-1] - CAPITAL
        gamma = net - r.fees - r.funding + r.costs   # LP price P&L + hedge P&L = what IL cost after hedging
        comp.append({"Swap fees": r.fees, "Funding": r.funding, "IL after hedge": gamma, "Costs": -r.costs,
                     "net": net})
    comp = pd.DataFrame(comp, index=names) / 1000
    fig, ax = plt.subplots(figsize=(10, 4.4))
    style(ax, r"Where the hedged P&L came from (\$k, 1 year, \$100k start)")
    parts = ["Swap fees", "Funding", "IL after hedge", "Costs"]
    width = 0.19
    x = np.arange(len(names))
    for j, (part, c) in enumerate(zip(parts, C)):
        ax.bar(x + (j - 1.5) * width, comp[part], width * 0.9, color=c, label=part)
    ax.scatter(x, comp["net"], color=INK, zorder=3, s=40, label="Net result")
    for xi, v in zip(x, comp["net"]):
        ax.annotate(f"net {v:+.1f}k", (xi, v), xytext=(0, -14 if v < 0 else 8), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, names, fontsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "pnl_attribution.png", dpi=160)
    plt.close(fig)

    # ---------------- figure 3: theta vs gamma (rolling 30d)
    win = 24 * 30
    full_fees_per_L = pd.Series(m.fee_usd_per_L)
    # full-range fee yield: fees per L over value per L (value = 2 L sqrt(p))
    fee_yield = full_fees_per_L / (2 * np.sqrt(m.p[1:]))
    fee_apr = fee_yield.rolling(win).sum() * (24 * 365 / win)
    rv = pd.Series(np.diff(np.log(m.p))).rolling(win).std() * math.sqrt(24 * 365)
    il_drag = rv**2 / 8
    fig, ax = plt.subplots(figsize=(10, 4.2))
    style(ax, "Full-range LP: fee income vs volatility cost (rolling 30 days, annualised)")
    d = dates[1:]
    ax.plot(d, fee_apr * 100, color=C[0], linewidth=2, label="Fee APR (theta)")
    ax.plot(d, il_drag * 100, color=C[1], linewidth=2, label="σ²/8 IL drag (gamma)")
    ax.axvline(dates[switch_idx + 1], color=INK2, linewidth=1, linestyle=(0, (2, 2)))
    ax.annotate("fee switch on (27 Dec):\nLPs keep 3/4 of fees", (dates[switch_idx + 1], ax.get_ylim()[0]),
                xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK2, va="bottom")
    ax.set_ylabel("% per year", color=INK2, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "theta_vs_gamma.png", dpi=160)
    plt.close(fig)

    # ---------------- figure 4: IL curves by range width
    r = np.linspace(0.5, 1.5, 401)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax, "Impermanent loss vs holding, by range width")
    for c, (w, n) in zip(C, [(math.inf, "Full range"), (0.5, "±50%"), (0.2, "±20%"), (0.1, "±10%")]):
        il = np.array([range_il(x, w) for x in r]) * 100
        ax.plot(r, il, color=c, linewidth=2, label=n)
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.set_xlabel("Price / entry price", color=INK2, fontsize=9)
    ax.set_ylabel("Loss vs HODL (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.tight_layout()
    fig.savefig(FIG / "il_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
