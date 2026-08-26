"""RS-ViSemDS: visual-semantic demonstration selection for remote sensing."""

from .adaptive_weights import (
    AdaptiveWeightResult,
    class_visual_log_evidence,
    compute_adaptive_weights,
    evidence_concentration,
    softmax_distribution,
)
from .calibration import TemperatureCalibrationResult, calibrate_temperatures
from .selector import (
    ScoredCandidate,
    select_adaptive_demonstrations,
)

__all__ = [
    "AdaptiveWeightResult",
    "ScoredCandidate",
    "TemperatureCalibrationResult",
    "calibrate_temperatures",
    "class_visual_log_evidence",
    "compute_adaptive_weights",
    "evidence_concentration",
    "select_adaptive_demonstrations",
    "softmax_distribution",
]
