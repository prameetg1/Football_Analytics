"""Statistical model layer over normalized analytics data (pitch-level too).

Models either run over the normalized event schema (core.py, xg.py) or over
pitch-location surfaces (pitch_control.py: pitch control, EPV, threat,
territory). All consume `src.analytics.schema`-compatible inputs.
"""

from src.analytics.models.core import (
    ExpectedPoints,
    Funnel,
    GoalLineValuation,
    PassNetworkSummary,
    PassingNetwork,
    Possession,
    expected_points_from_goals,
    funnel,
    possession_analysis,
    summarize_network,
)
from src.analytics.models.xg import (
    ShotAggregates,
    aggregate_shots,
)
from src.analytics.models.pitch_control import (
    epv_grid,
    expected_threat_grid,
    field_tilt,
    pitch_control_surface,
    space_creation_runs,
)
from src.analytics.models.snapshot import (
    SnapshotSummary,
    snapshot_summary,
    territory_series,
)
from src.analytics.models.flow import (
    FlowField,
    TeamFormation,
    flow_field,
    team_formation,
)
from src.analytics.models.pressure import (
    PressMap,
    PressWindows,
    press_map,
    press_windows,
)
from src.analytics.models.attacking import (
    ProgressivePassSummary,
    ShotMap,
    progressive_summary,
    shot_map,
)
from src.analytics.models.winprob import (
    GoalValueByMinute,
    ScoreCurve,
    goal_value_by_minute,
    score_curve,
    win_probability,
)
from src.analytics.models.rating import (
    DixonColesFit,
    MatchOdds,
    MatchResult,
    fit_dixon_coles,
    predict_match,
    table_from_results,
)
from src.analytics.models.vaep import (
    VaepAction,
    VaepSummary,
    hazard_surface,
    vaep,
    value_action,
)

__all__ = [
    "PassingNetwork", "Possession", "ExpectedPoints", "GoalLineValuation",
    "Funnel", "ShotAggregates", "possession_analysis", "funnel",
    "expected_points_from_goals", "aggregate_shots",
    "PassNetworkSummary", "summarize_network",
    "pitch_control_surface", "epv_grid", "expected_threat_grid",
    "field_tilt", "space_creation_runs",
    "SnapshotSummary", "snapshot_summary", "territory_series",
    "FlowField", "TeamFormation", "flow_field", "team_formation",
    "PressMap", "PressWindows", "press_map", "press_windows",
    "ProgressivePassSummary", "ShotMap", "progressive_summary", "shot_map",
    "ScoreCurve", "GoalValueByMinute", "win_probability", "score_curve",
    "goal_value_by_minute",
    "DixonColesFit", "MatchOdds", "MatchResult", "fit_dixon_coles",
    "predict_match", "table_from_results",
    "VaepAction", "VaepSummary", "hazard_surface", "value_action", "vaep",
]
