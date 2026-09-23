"""Metrics gate: refuse to emit metrics unless CV-vs-GT validation passes.

Stage 7's deliverable — a metric table is only surfaced when the gate is
enabled AND the match's StatsBomb validation report is within bounds. When the
gate is disabled (default) the pipeline runs exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import (
    GATE_MAX_POSITION_ERROR_M,
    GATE_MIN_PASS_PRECISION,
    GATE_MIN_PASS_RECALL,
    METRICS_GATE_ENABLED,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoided at runtime
    from src.analytics.adapters.statsbomb.validate import ValidationReport


class MetricGateBlocked(Exception):
    """Raised when the gate is enabled and the validation report fails."""


@dataclass
class MetricsGate:
    """Gates metric emission on a `ValidationReport`."""

    enabled: bool = METRICS_GATE_ENABLED
    max_position_error_m: float = GATE_MAX_POSITION_ERROR_M
    min_pass_recall: float = GATE_MIN_PASS_RECALL
    min_pass_precision: float = GATE_MIN_PASS_PRECISION
    _report: ValidationReport | None = field(default=None, repr=False)

    def set_report(self, report: ValidationReport) -> None:
        self._report = report

    def __bool__(self) -> bool:
        """True == metrics may be emitted."""
        if not self.enabled:
            return True          # soft: gate not engaged
        if self._report is None:
            return False         # enabled but never validated -> block
        return self._report.passes

    def check(self) -> None:
        """Raise MetricGateBlocked if emission is not permitted."""
        if not bool(self):
            raise MetricGateBlocked(
                "metric emission gated: validation report missing or out of "
                f"bounds (pos_err<={self.max_position_error_m:.1f}m, "
                f"recall>={self.min_pass_recall:.1f}, "
                f"precision>={self.min_pass_precision:.1f})")
