"""Purga y embargo, metricas y test economico."""

from miax.eval.beta_study import (
    FACTOR_MAKERS,
    FoldData,
    ResidualFit,
    StudyResult,
    beta_horizon_bias,
    compare_schemes,
    fold_symbols,
    fold_windows,
    load_fold,
    project_fixed,
    residual_diagnostics,
    rolling_residuals,
    write_tables,
)
from miax.eval.idiosyncratic_signal import (
    SignalResult,
    contemporaneous_structure,
    decision_table,
    lagged_cross_correlation,
    signal_statistics,
    study_signal,
    write_signal_tables,
)

__all__ = [
    "FACTOR_MAKERS",
    "FoldData",
    "ResidualFit",
    "SignalResult",
    "StudyResult",
    "beta_horizon_bias",
    "compare_schemes",
    "contemporaneous_structure",
    "decision_table",
    "fold_symbols",
    "fold_windows",
    "lagged_cross_correlation",
    "load_fold",
    "project_fixed",
    "residual_diagnostics",
    "rolling_residuals",
    "signal_statistics",
    "study_signal",
    "write_signal_tables",
    "write_tables",
]
