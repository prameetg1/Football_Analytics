"""Bulk retrieval for all non-StatsBomb sources.

Runs each registered retriever's `fetch`/`fetch_all` for the 11-source
registry, skipping StatsBomb (already fully cached). Per-source failures are
reported and do not abort the run. Skipping is resumable (files already on disk
are reused) so re-running is cheap.

Usage
-----
    python scripts/retrieve_sources.py [--sources skillcorner,wyscout,idsse]
                                       [--limit N]

`--sources` restricts to a comma-separated subset. `--limit` caps the number of
matches fetched per dense/event source (SkillCorner/IDSSE/Wyscout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.retrieval import get  # noqa: E402

# SkillCorner match ids exposed by the retriever.
# Wyscout match ids on the koenvo mirror.
_WYSCOUT_SAMPLE = (
    "2499841", "1694390", "1694391", "2355071", "2505363",
    "1983144", "2576338", "1799912", "2224827", "2101644",
    "1898292", "2412131", "1725700", "2590795", "1960968",
)

# id -> (home, away) from the SkillCorner catalogue (representative names).
_SKILLCORNER = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]

_IDSSE = (
    "J03WMX", "J03WN1", "J03WOH", "J03WOY",
    "J03WPY", "J03WQQ", "J03WR9",
)


def log(msg: str) -> None:
    print(f"[retrieve] {msg}", flush=True)


def fetch_skillcorner(limit: int | None) -> None:
    sc = get("skillcorner")
    ids = sc.list_resources()
    if limit:
        ids = ids[:limit]
    for mid in ids:
        log(f"skillcorner {mid} ...")
        try:
            d = sc.fetch(mid)
            n = len(list(d.iterdir()))
            log(f"  ok -> {d} ({n} files)")
        except Exception as exc:  # noqa: BLE001
            log(f"  FAIL {mid}: {exc}")


def fetch_idsse(limit: int | None) -> None:
    idsse = get("idsse")
    ids = idsse.list_resources()
    if limit:
        ids = ids[:limit]
    for mid in ids:
        log(f"idsse {mid} ...")
        try:
            d = idsse.fetch(mid)
            n = len(list(d.iterdir()))
            log(f"  ok -> {d} ({n} files)")
        except Exception as exc:  # noqa: BLE001
            log(f"  FAIL {mid}: {exc}")


def fetch_wyscout(limit: int | None) -> None:
    w = get("wyscout")
    ids = list(_WYSCOUT_SAMPLE)
    done = set(w.list_resources())
    todo = [i for i in ids if i not in done]
    if limit:
        todo = todo[:limit]
    for mid in todo:
        log(f"wyscout {mid} ...")
        try:
            p = w.fetch(mid)
            log(f"  ok -> {p}")
        except Exception as exc:  # noqa: BLE001
            log(f"  FAIL {mid}: {exc}")


def fetch_soccerdata(source: str, season: str | None = None) -> None:
    log(f"soccerdata {source} (season={season or 'default'}) ...")
    try:
        if season is not None:
            from src.analytics.retrieval.soccerdata import (
                SoccerDataRetriever)
            r = SoccerDataRetriever(source, season=season)
        else:
            r = get(source)
    except Exception as exc:  # noqa: BLE001
        log(f"  construct FAIL: {exc}")
        return
    try:
        resources = r.available_resources()
    except Exception as exc:  # noqa: BLE001
        resources = ["schedule"]
        log(f"  available_resources FAIL ({exc}), defaulting to schedule")
    for resource in resources:
        log(f"  {source} {resource} ...")
        try:
            p = r.fetch(resource)
            log(f"    ok -> {p}")
        except Exception as exc:  # noqa: BLE001
            log(f"    FAIL {resource}: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources",
                    default="skillcorner,idsse,wyscout,fbref,understat,"
                            "whoscored,sofascore,espn,football-data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--season", type=str, default=None,
                    help="soccerdata season id (e.g. 2026 = 2025-26) for "
                         "fbref/understat/espn/sofascore/whoscored")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.sources.split(",")}
    if "skillcorner" in wanted:
        fetch_skillcorner(args.limit)
    if "idsse" in wanted:
        fetch_idsse(args.limit)
    if "wyscout" in wanted:
        fetch_wyscout(args.limit)
    for source in ("fbref", "understat", "whoscored", "sofascore", "espn"):
        if source in wanted:
            fetch_soccerdata(source, args.season)
    log("done.")


if __name__ == "__main__":
    main()