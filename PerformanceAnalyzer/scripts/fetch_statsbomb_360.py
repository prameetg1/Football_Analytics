"""Bulk-download StatsBomb open data (events, lineups, 360) for matches that
have 360 freeze-frame data available.

Enumerates the `match_status_360 == "available"` matches across the given
competition/season pairs from the open-data match index, then downloads
events + lineups + 360 for each into `data/statsbomb/` using the same
`match_{match_id}_{kind}.json` naming convention as the loader, so downloaded
files are reused offline.

Safe to re-run: already-cached matches are skipped, and failures on individual
matches are logged and continue rather than aborting the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from src.config import STATSBOMB_DATA_DIR, STATSBOMB_DATA_URL

# (competition_id, season_id, display label) — the 360-enabled sets.
COMPETITION_SEASONS: list[tuple[int, int, str]] = [
    (43, 106, "FIFA World Cup 2022"),
    (55, 282, "UEFA Euro 2024"),
    (72, 107, "Women's World Cup 2023"),
    (9, 281, "Bundesliga 2023/24"),
    (11, 90, "La Liga 2020/21"),
    (7, 108, "Ligue 1 2021/22"),
    (44, 107, "MLS 2023"),
    (53, 315, "Women's Euro 2025"),
]

KINDS = ("events", "lineups", "360")
# Overrides to force-fetch matches that already exist locally (e.g. knockout
# matches cached with events/360 but missing lineups). Keyed by match_id.
FORCE_REFRESH: set[int] = set()   # extended by --refetch-ids


def http_get(session: requests.Session, url: str, timeout: int = 60) -> dict | list:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def match_ids_with_360(session: requests.Session, comp_id: int,
                       season_id: int) -> list[dict]:
    url = f"{STATSBOMB_DATA_URL}/matches/{comp_id}/{season_id}.json"
    data = http_get(session, url)
    return [m for m in data if m.get("match_status_360") == "available"]


def download_match(session: requests.Session, data_dir: Path, match_id: int,
                   kinds: tuple[str, ...], refetch: set[int]) -> list[str]:
    """Download missing (or forced) kinds for one match. Returns saved kinds."""
    saved: list[str] = []
    for kind in kinds:
        path = data_dir / f"match_{match_id}_{kind}.json"
        if path.exists() and match_id not in refetch:
            continue
        # three-sixty, events, lineups
        url = f"{STATSBOMB_DATA_URL}/{'three-sixty' if kind == '360' else kind}/{match_id}.json"
        try:
            payload = http_get(session, url)
        except Exception as exc:  # noqa: BLE001
            print(f"    [skip] {kind} {match_id}: {exc}", file=sys.stderr)
            continue
        path.write_text(json.dumps(payload))
        saved.append(kind)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commas", help="'cid,sid' pairs (repeatable/; separated)")
    ap.add_argument("--refetch-ids", help="comma-separated match ids to force")
    ap.add_argument("--limit", type=int, default=0,
                    help="only fetch first N matches (test run)")
    args = ap.parse_args()

    data_dir = Path(STATSBOMB_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.commas:
        sets = []
        for part in args.commas.split(";"):
            cid, sid = (int(x) for x in part.strip().split(","))
            sets.append((cid, sid, f"{cid}/{sid}"))
    else:
        sets = COMPETITION_SEASONS

    refetch = FORCE_REFRESH | set(
        int(x) for x in (args.refetch_ids or "").split(",") if x)

    session = requests.Session()
    session.headers["User-Agent"] = "performance-analyzer/1.0"

    fetched = 0
    for comp_id, season_id, label in sets:
        try:
            matches = match_ids_with_360(session, comp_id, season_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {label}: {exc}", file=sys.stderr)
            continue
        print(f"[{label}] {len(matches)} matches with 360")
        if args.limit:
            matches = matches[: args.limit]
        for i, m in enumerate(matches, 1):
            mid = m["match_id"]
            home = m.get("home_team", {}).get("home_team_name", "?")
            away = m.get("away_team", {}).get("away_team_name", "?")
            saved = download_match(session, data_dir, mid, KINDS, refetch)
            fetched += 1
            tag = ",".join(saved) if saved else "-"
            print(f"  ({i:>2}/{len(matches)}) {mid} {home} v {away} [{tag}]")
            time.sleep(0.1)   # be gentle on the mirror
    print(f"\nDone. Processed {fetched} match attempts.")


if __name__ == "__main__":
    main()
