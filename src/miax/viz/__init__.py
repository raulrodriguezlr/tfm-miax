"""Figuras para la memoria."""

from miax.viz.factor_comparison import (
    FACTOR_MODELS,
    FactorComparisonPaths,
    build_factor_comparison_report,
    compare_factors,
    factor_comparison_table,
    split_period,
)
from miax.viz.panel_overview import (
    PanelOverview,
    bar_validity_pct,
    build_panel_overview,
    correlation_by_liquidity,
    feature_quantiles,
    period_gap_causes,
    symbol_quality,
)
from miax.viz.raw_correlation import (
    ReportPaths,
    build_raw_correlation_report,
    correlation_summary,
    explained_variance,
    order_by_category,
    plot_raw_correlation,
    raw_correlation,
    save_figure,
    summarize_correlation,
    symbols_without_data,
)

__all__ = [
    "FACTOR_MODELS",
    "FactorComparisonPaths",
    "PanelOverview",
    "ReportPaths",
    "bar_validity_pct",
    "build_factor_comparison_report",
    "build_panel_overview",
    "build_raw_correlation_report",
    "compare_factors",
    "correlation_by_liquidity",
    "correlation_summary",
    "explained_variance",
    "factor_comparison_table",
    "feature_quantiles",
    "order_by_category",
    "period_gap_causes",
    "plot_raw_correlation",
    "raw_correlation",
    "save_figure",
    "split_period",
    "summarize_correlation",
    "symbol_quality",
    "symbols_without_data",
]
