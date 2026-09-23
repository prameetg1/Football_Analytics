#!/usr/bin/env python3
"""Print a provider-agnostic analytics report for one or more matches.

Demonstrates the new `src.analytics` layer: an adapter turns a provider's raw
data into normalized schema, and the model bundle (passing network, possession,
possession-share, funnel, xG) produces a per-match report. Provider-agnostic —
any adapter implementing `ProviderAdapter` works.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/match_analytics_report.py 3869685 [3788741 ...]
"""

import argparse
import sys

from src.analytics.adapters.statsbomb import StatsBombAdapter
from src.analytics.analyze import analyze_match


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("match_ids", nargs="+", type=int,
                    help="StatsBomb match ids to analyse")
    args = ap.parse_args(argv)

    adapter = StatsBombAdapter()
    for mid in args.match_ids:
        try:
            rep = analyze_match(adapter, mid)
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"match {mid}: {exc}", file=sys.stderr)
            continue
        print(f"\n=== {rep.home_team} vs {rep.away_team} (match {mid}) ===")
        for side, s in rep.summary.items():
            name = s["team"] or f"side {side}"
            print(f"  {name:<14} poss {s['possession_share']:.3f}  "
                  f"pass/min {s['passing_rate']:.2f}  shots {s['shots']:>2}  "
                  f"xG {s['xg']:.2f}  goals {s['goals']}")
        top = rep.passing_network.degree_top(3)
        if top:
            print("  top pass-vol players (id, volume):",
                  " ".join(f"{pid}:{v}" for pid, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
