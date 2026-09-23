"""Stage 7/9: validate a CV tracking table against StatsBomb open-data GT.

Usage:
    python scripts/validate_against_statsbomb.py --match-id 3788741 \
        --cv output/tracking.csv

Fetches (and caches) the StatsBomb match payloads, builds a ground-truth
tracking frame from the 360 freeze-frame data, and compares the given CV
tracking export. Prints a ValidationReport: position error (m), pass
recall/precision, and per-kind event recall/precision (Stage 9: recoveries,
shots, clearances, ...). Also prints whether the metrics gate would pass.

Because our sample footage (0bfacc_0.mp4) is an unknown excerpt, the script
does not assume it maps to `--match-id`; it is a closest-footage protocol: you
point it at whatever match your footage came from, or run it on synthetic GT
for tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.adapters.video.metrics.gate import MetricsGate  # noqa: E402
from src.analytics.adapters.statsbomb.alignment import (  # noqa: E402
    build_gt_frame,
    extract_event_locations,
    extract_pass_events,
)
from src.analytics.adapters.statsbomb.loader import StatsBombLoader  # noqa: E402
from src.analytics.adapters.statsbomb.validate import (  # noqa: E402
    ValidationReport,
    pass_match,
    position_error,
    validate_cv_against_gt,
    validate_events_against_gt,
)

GATE_KINDS = ("Pass", "Ball Recovery", "Shot", "Clearance",
              "Throw-in", "Goal Kick", "Corner", "Carry", "Duel")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-id", type=int, required=True)
    ap.add_argument("--cv", type=str, required=True,
                    help="CV tracking export (csv/parquet), TrackingFrame schema")
    ap.add_argument("--no-360", action="store_true",
                    help="skip 360 fetch (position-error validation needs it)")
    args = ap.parse_args()

    loader = StatsBombLoader()
    gt_df = build_gt_frame(loader, args.match_id, with_360=not args.no_360)

    if args.cv.endswith(".parquet"):
        cv_df = pd.read_parquet(args.cv)
    else:
        cv_df = pd.read_csv(args.cv)

    report = validate_cv_against_gt(cv_df, gt_df)
    gate = MetricsGate(enabled=True)
    gate.set_report(report)
    print(report)
    print(f"gate: {'PASS' if report.passes else 'BLOCK'} "
          f"(pos_err<={gate.max_position_error_m:.1f}m, "
          f"recall>={gate.min_pass_recall:.1f}, "
          f"precision>={gate.min_pass_precision:.1f})")

    # Stage 9: per-kind event gate. GT per kind is extracted from the real
    # StatsBomb events; CV events come from our extractor on the tracking table.
    from src.analytics.adapters.video.events import extract_events

    cv_events = extract_events(cv_df)
    cv_kinds = {e.kind for e in cv_events}
    gt_events: dict[str, list[tuple[float, float, float]]] = {}
    for kind in GATE_KINDS:
        if kind not in cv_kinds:
            continue
        if kind == "Pass":
            g = extract_pass_events(loader, args.match_id)
            g = g.dropna(subset=["end_x", "end_y"])
            gt = list(zip(g["timestamp_sec"], g["end_x"], g["end_y"]))
        else:
            sb_kind = {"Ball Recovery": "Ball Recovery", "Shot": "Shot",
                       "Clearance": "Clearance", "Throw-in": "Throw In",
                       "Goal Kick": "Goal Kick", "Corner": "Corner",
                       "Carry": "Carry", "Duel": "Duel"}.get(kind, kind)
            g = extract_event_locations(loader, args.match_id, (sb_kind,))
            if g.empty:
                continue
            gt = list(zip(g["timestamp_sec"], g["pitch_x"], g["pitch_y"]))
        gt_events[kind] = gt

    ev_report = validate_events_against_gt(cv_events, gt_events)
    print("\nStage-9 per-kind event gate:")
    for kind in sorted(ev_report.kinds):
        p, r, n_gt = ev_report.kinds[kind]
        print(f"  {kind:14s} precision={p:.2f} recall={r:.2f} (gt={n_gt})")
    print(f"passes        precision={ev_report.pass_precision:.2f} "
          f"recall={ev_report.pass_recall:.2f} (gt={ev_report.n_gt_passes})")
    print(f"event gate: {'PASS' if ev_report.passes else 'BLOCK'}")
    return 0 if report.passes and ev_report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())

