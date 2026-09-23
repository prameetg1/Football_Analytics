#!/usr/bin/env python3
"""Fit and cache the C2 scoring model from real StatsBomb shot data.

Pools shots from the WC 2022 knockout matches (round of 16 -> final) plus any
extra `--match-id`s, fits a regularised logistic P(goal | x, y), and writes the
coefficients to `data/statsbomb/shot_model.json` (consumed by `epv_grid`).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/fit_shot_model.py [--match-id 3788741 ...]
"""

import argparse

from src.config import EPV_SHOT_MODEL_PATH, STATSBOMB_DATA_DIR
from src.analytics.adapters.statsbomb.loader import StatsBombLoader
from src.analytics.adapters.statsbomb.shot_model import ShotModel, extract_shots, save_shot_model

WC_2022_COMP = 43
WC_2022_SEASON = 106
KNOCKOUT_STAGES = ("Round of 16", "Quarter-finals", "Semi-finals",
                   "Final", "3rd Place Final")


def knockout_match_ids(loader: StatsBombLoader) -> list[int]:
    ms = loader.matches(WC_2022_COMP, WC_2022_SEASON)
    return [m["match_id"] for m in ms
            if m["competition_stage"]["name"] in KNOCKOUT_STAGES]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit StatsBomb shot model")
    ap.add_argument("--match-id", type=int, action="append", default=[],
                    help="extra match ids (repeatable)")
    ap.add_argument("--min-goals", type=int, default=25,
                    help="abort if the pooled shots contain fewer goals")
    args = ap.parse_args()

    loader = StatsBombLoader()
    ids = sorted(set(args.match_id) | set(knockout_match_ids(loader)))
    print(f"pooling shots from {len(ids)} matches ...")
    frames = []
    goals = 0
    for mid in ids:
        try:
            shots = extract_shots(loader.events(mid))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! match {mid}: {exc}")
            continue
        goals += int(shots["goal"].sum())
        frames.append(shots)
        print(f"  match {mid}: {len(shots)} shots")
    all_shots = __import__("pandas").concat(frames, ignore_index=True)
    print(f"total {len(all_shots)} shots, {goals} goals")
    if goals < args.min_goals:
        print(f"aborting: only {goals} goals (< {args.min_goals})")
        return 1
    model = ShotModel.fit(all_shots)
    save_shot_model(model, EPV_SHOT_MODEL_PATH)
    print(f"saved {EPV_SHOT_MODEL_PATH}")
    print(f"intercept={model.intercept:.4f} coef_={[round(w, 5) for w in model.coef_]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
