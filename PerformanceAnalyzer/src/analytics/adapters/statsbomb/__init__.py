"""StatsBomb open-data source + provider adapter: loading, alignment, validation."""

from src.analytics.adapters.statsbomb.adapter import StatsBombAdapter
from src.analytics.adapters.statsbomb.loader import StatsBombLoader, StatsBombMatch
from src.analytics.adapters.statsbomb.validate import (
    ValidationReport,
    detect_passes,
    nearest_assignment_mean_error,
    pass_match,
    position_error,
    validate_cv_against_gt,
)

__all__ = [
    "StatsBombAdapter",
    "StatsBombLoader",
    "StatsBombMatch",
    "ValidationReport",
    "detect_passes",
    "nearest_assignment_mean_error",
    "pass_match",
    "position_error",
    "validate_cv_against_gt",
]
