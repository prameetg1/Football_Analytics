"""Concrete orchestration demo: run the full agent pipeline on a real cached match.

Uses Metrica sample_game_1 (dense tracking + events) from the local cache —
fully offline. Orchestrates Tactician → StatScientist → Visualizer → DomainExpert,
writes each agent's HTML report into a shared timestamped match folder, and
prints a summary of the resulting `MatchInsightReport`.
"""

from __future__ import annotations

from pathlib import Path

from src.analytics.agents import orchestrate
from src.analytics.adapters.metrica.adapter import MetricaAdapter

MATCH_DIR = Path("data/retrieval/metrica/sample_game_1")
MATCH_ID = 1


def main() -> None:
    adapter = MetricaAdapter(match_dir=MATCH_DIR, sample_every=5)

    print("=" * 70)
    print("ORCHESTRATION DEMO — Metrica sample_game_1")
    print("=" * 70)

    # 0. What data does this adapter give us?
    from src.analytics.adapters.resolver import Resolver
    rp = Resolver([adapter]).resolve(MATCH_ID,
                                     {"events", "dense_tracking", "lineups"})
    print("\n[resolver] capabilities -> source adapters:")
    for cap, src in rp.sources.items():
        print(f"    {cap:16s} -> {src}")

    # 1. Run the full pipeline (figures + HTML into a timestamped match folder)
    report = orchestrate(adapter, MATCH_ID)

    print("\n[errors]", report.errors or "none")
    print("\n[output folder]")
    print(f"    {report.output_dir.folder}")

    print("\n[html files written]")
    for f in sorted(report.html_written):
        print(f"    {f}")

    # 2. Spot-check each agent's typed output
    print("\n" + "-" * 70)
    print("TACTICAL (highlights)")
    print("-" * 70)
    for label, ta in (("Home", report.tactical.home),
                      ("Away", report.tactical.away)):
        print(f"  {label:4s} decentral={ta.decentralization:.3f} "
              f"central={ta.most_central_player} "
              f"presses={ta.press_events_total} "
              f"width={ta.formation_width:.1f}m")
    print("  insights:", *[f"\n    - {i}" for i in report.tactical.insights])

    print("\n" + "-" * 70)
    print("STATISTICAL (highlights)")
    print("-" * 70)
    for label, sa in (("Home", report.statistical.home),
                      ("Away", report.statistical.away)):
        print(f"  {label:4s} xG={sa.xg:.2f} goals={sa.goals} "
              f"over={sa.xg_overperformance:+.2f} "
              f"shots={sa.shots} vaep={sa.vaep_value:+.2f} "
              f"possession={sa.possession_share:.1%}")
    print("  insights:", *[f"\n    - {i}" for i in report.statistical.insights])

    print("\n" + "-" * 70)
    print("VISUAL (figures saved into match folder)")
    print("-" * 70)
    for p in report.visual.saved_paths:
        print(f"    {Path(p).relative_to(report.output_dir.folder)}")

    print("\n" + "-" * 70)
    print("NARRATIVE")
    print("-" * 70)
    print(report.narrative.full_narrative)

    print("\nOpen in a browser:")
    print(f"    file://{report.output_dir.folder / 'index.html'}")


if __name__ == "__main__":
    main()