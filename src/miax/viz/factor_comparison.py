"""Comparativa de los factores BTC, media y PCA trimestre a trimestre (MIAX-055)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_metrics import correlation_drop
from miax.features.factor_pca import PcaFactor
from miax.features.panel_store import PANEL_DIR, cached_period_panels
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import load_universe
from miax.utils.io import resolve_repo_path
from miax.utils.logging import get_logger
from miax.viz.raw_correlation import _NUMBER_FORMAT, TABLES_DIR

logger = get_logger(__name__)

# Fabricas, no instancias: cada periodo usa un modelo nuevo, nunca uno ya ajustado con otro tramo.
FACTOR_MODELS: dict[str, Callable[[], FactorModel]] = {
    "btc": BtcFactor,
    "mean": MeanFactor,
    "pca": PcaFactor,
}

# 2026 se reserva entero para la evaluacion final (D-37): elegir factor no puede verlo.
RESERVED_FROM = pd.Timestamp("2026-01-01", tz="UTC")

_METRIC_COLUMNS: tuple[str, ...] = (
    "n_symbols",
    "n_symbols_without_data",
    "symbols_without_data",
    "n_pairs",
    "raw_pair_mean",
    "residual_pair_mean",
    "raw_pair_abs_mean",
    "residual_pair_abs_mean",
    "correlation_drop",
    "raw_first_eigenvalue_share",
    "residual_first_eigenvalue_share",
)
_ROW_COLUMNS: tuple[str, ...] = (
    *_METRIC_COLUMNS,
    "signed_floor",
    "train_rows",
    "test_rows",
    "error",
)


class FactorComparisonPaths(NamedTuple):
    """Rutas de las dos tablas del informe."""

    by_period_csv: Path
    summary_csv: Path


def split_period(
    returns: pd.DataFrame, *, train_months: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte un periodo en train (los primeros `train_months` meses) y test (el resto)."""
    if train_months < 1:
        raise ValueError(f"train_months debe ser >= 1: {train_months}")
    if returns.empty:
        raise ValueError("returns esta vacio")
    cut = returns.index[0] + pd.DateOffset(months=train_months)
    # El test empieza justo cuando acaba el train (`< cut` frente a `>= cut`): no se solapan.
    train = returns[returns.index < cut]
    test = returns[returns.index >= cut]
    if train.empty or test.empty:
        raise ValueError(f"train ({len(train)} filas) o test ({len(test)} filas) vacio")
    return train, test


def _empty_row(error: str, train_rows: int, test_rows: int) -> dict[str, object]:
    """Fila con todas las metricas en `NaN` y el motivo del fallo."""
    row: dict[str, object] = dict.fromkeys(_ROW_COLUMNS, np.nan)
    row["symbols_without_data"] = ""
    row["train_rows"] = train_rows
    row["test_rows"] = test_rows
    row["error"] = error
    return row


def compare_factors(
    returns: pd.DataFrame,
    *,
    train_months: int = 2,
    models: Mapping[str, Callable[[], FactorModel]] | None = None,
    min_periods: int = 1000,
) -> pd.DataFrame:
    """Ajusta cada factor en el train de un periodo y mide en el test cuanta correlacion quita."""
    factories = FACTOR_MODELS if models is None else models
    try:
        train, test = split_period(returns, train_months=train_months)
    except ValueError as exc:
        logger.warning("Periodo sin resultado: %s", exc)
        empty = {name: _empty_row(str(exc), 0, 0) for name in factories}
        table = pd.DataFrame.from_dict(empty, orient="index", columns=list(_ROW_COLUMNS))
        table.index.name = "factor"
        return table
    rows: dict[str, dict[str, object]] = {}
    for name, factory in factories.items():
        try:
            # Ajuste solo con el train; el test se transforma con betas ya fijadas: sin fuga.
            residuals = factory().fit(train).transform(test)
            metrics = correlation_drop(test, residuals, min_periods=min_periods)
        except ValueError as exc:
            logger.warning("Factor %s sin resultado: %s", name, exc)
            rows[name] = _empty_row(str(exc), len(train), len(test))
            continue
        usable = int(metrics["n_symbols"]) - int(metrics["n_symbols_without_data"])
        # Suelo de la correlacion media con signo tras quitar un factor comun: -1/(N-1) (D-35).
        floor = -1.0 / (usable - 1) if usable > 1 else np.nan
        rows[name] = {
            **metrics,
            "signed_floor": floor,
            "train_rows": len(train),
            "test_rows": len(test),
            "error": "",
        }
    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(_ROW_COLUMNS))
    table.index.name = "factor"
    return table


def usable_quarters(by_period: pd.DataFrame, *, min_symbols: int = 30) -> pd.DataFrame:
    """Filas de los trimestres con todos los factores validos y al menos `min_symbols` monedas."""
    errors = by_period["error"].fillna("").astype(str)
    usable = by_period["n_symbols"] - by_period["n_symbols_without_data"]
    # Con pocas monedas el suelo -1/(N-1) domina y la comparacion entre periodos no es justa.
    # El filtro es por trimestre y no por fila: asi las columnas usan los mismos trimestres.
    row_ok = (errors == "") & (usable >= min_symbols)
    quarter_ok = row_ok.groupby(by_period["period_start"]).transform("all")
    return by_period[quarter_ok]


def factor_comparison_table(by_period: pd.DataFrame, *, min_symbols: int = 30) -> pd.DataFrame:
    """Resume el formato largo en una tabla con una columna por factor, lista para la memoria.

    Un trimestre solo entra si todos sus factores son validos, para comparar sobre los mismos.
    """
    kept = usable_quarters(by_period, min_symbols=min_symbols)
    if kept.empty:
        raise ValueError(f"ningun periodo valido con al menos {min_symbols} monedas")
    order = [name for name in FACTOR_MODELS if name in set(kept["factor"])]
    order += [name for name in dict.fromkeys(kept["factor"]) if name not in order]
    summary: dict[str, dict[str, object]] = {}
    for name in order:
        part = kept[kept["factor"] == name]
        drop = part["correlation_drop"]
        starts = pd.to_datetime(part["period_start"])
        summary[name] = {
            "n_periods": int(part["period_start"].nunique()),
            "correlation_drop_median": drop.median(),
            "correlation_drop_min": drop.min(),
            "correlation_drop_max": drop.max(),
            "raw_pair_abs_mean_median": part["raw_pair_abs_mean"].median(),
            "residual_pair_abs_mean_median": part["residual_pair_abs_mean"].median(),
            "residual_pair_mean_median": part["residual_pair_mean"].median(),
            "signed_floor_median": part["signed_floor"].median(),
            "residual_first_eigenvalue_share_median": (
                part["residual_first_eigenvalue_share"].median()
            ),
            "first_period": f"{starts.min():%Y-%m-%d}",
            "last_period": f"{starts.max():%Y-%m-%d}",
        }
    return pd.DataFrame(summary)


def factor_comparison_by_period(
    universe: pd.DataFrame | None = None,
    *,
    interval: str = "1m",
    train_months: int = 2,
    reserved_from: pd.Timestamp = RESERVED_FROM,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
) -> pd.DataFrame:
    """Compara los tres factores en cada periodo previo a `reserved_from`; formato largo."""
    if reserved_from.tzinfo is None:
        raise ValueError(f"reserved_from debe estar en UTC: {reserved_from}")
    full = load_universe() if universe is None else universe
    # Solo periodos que acaban antes de `reserved_from`: 2026 no se mira para elegir factor (D-37).
    kept = full[pd.to_datetime(full["period_end"], utc=True) <= reserved_from]
    if kept.empty:
        raise ValueError(f"ningun periodo termina antes de {reserved_from:%Y-%m-%d}")
    frames: list[pd.DataFrame] = []
    for period_start, panel in cached_period_panels(
        kept, interval, base_dir=base_dir, cache_dir=cache_dir
    ):
        logger.info("Comparando factores en %s", f"{period_start:%Y-%m-%d}")
        part = compare_factors(panel.features["return"], train_months=train_months)
        part.insert(0, "period_start", f"{period_start:%Y-%m-%d}")
        frames.append(part.reset_index())
        del panel
    return pd.concat(frames, ignore_index=True)


def build_factor_comparison_report(
    *,
    universe: pd.DataFrame | None = None,
    interval: str = "1m",
    train_months: int = 2,
    min_symbols: int = 30,
    reserved_from: pd.Timestamp = RESERVED_FROM,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
    tables_dir: str | Path = TABLES_DIR,
) -> FactorComparisonPaths:
    """Compara los tres factores en cada periodo previo a 2026 y escribe las dos tablas en CSV."""
    by_period = factor_comparison_by_period(
        universe,
        interval=interval,
        train_months=train_months,
        reserved_from=reserved_from,
        base_dir=base_dir,
        cache_dir=cache_dir,
    )
    summary = factor_comparison_table(by_period, min_symbols=min_symbols)

    out = resolve_repo_path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_period_csv = out / "factor_comparison_by_period.csv"
    summary_csv = out / "factor_comparison.csv"
    by_period.to_csv(by_period_csv, index=False, float_format=_NUMBER_FORMAT, lineterminator="\n")
    # Las columnas del resumen son `object` (int, float y texto): `float_format` no las toca.
    printable = summary.astype(object).apply(
        lambda column: column.map(
            lambda v: _NUMBER_FORMAT % v if isinstance(v, float) and not np.isnan(v) else v
        )
    )
    printable.to_csv(summary_csv, index_label="metric", lineterminator="\n")
    medians = ", ".join(
        f"{name}={summary.loc['correlation_drop_median', name]:.4f}" for name in summary.columns
    )
    logger.info(
        "Comparativa de factores: %s y %s; caida mediana: %s", by_period_csv, summary_csv, medians
    )
    return FactorComparisonPaths(by_period_csv=by_period_csv, summary_csv=summary_csv)
