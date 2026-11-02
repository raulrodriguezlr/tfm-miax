"""Resumen descriptivo de los paneles de la cache, trimestre a trimestre (MIAX-214).

Todo es descriptivo del periodo entero de cada panel: no es una feature ni un filtro, y no debe
entrar en ningun modelo (usaria el periodo completo, es decir, el futuro). Las causas de hueco
son ex post (D-17) y la correlacion es la cruda de MIAX-046 (D-30).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from miax.features.alignment import grid_ms
from miax.features.contract import validate_panel
from miax.features.gaps import (
    GapCause,
    cause_counts,
    classify_klines,
    download_ends_from_universe,
    listings_from_universe,
)
from miax.features.panel import Panel
from miax.features.panel_store import PANEL_DIR, cached_period_panels
from miax.features.resample import bar_factor, resample_validity
from miax.features.tails import tail_report
from miax.ingest.klines import interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import to_ms
from miax.utils.logging import get_logger
from miax.viz.raw_correlation import raw_correlation, summarize_correlation

logger = get_logger(__name__)

DEFAULT_QUANTILES: tuple[float, ...] = (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999)
DEFAULT_HORIZONS: tuple[str, ...] = ("5m", "15m", "60m")


class PanelOverview(NamedTuple):
    """Tablas del resumen: una fila por trimestre y, segun la tabla, por simbolo o por feature."""

    periods: pd.DataFrame
    quality: pd.DataFrame
    feature_quantiles: pd.DataFrame
    tails: pd.DataFrame
    bar_validity: pd.DataFrame
    gap_causes: pd.DataFrame
    correlation: pd.DataFrame


def symbol_quality(panel: Panel) -> pd.DataFrame:
    """Por simbolo: `valid_pct` (% de minutos validos) y el % de `NaN` de cada feature."""
    table = pd.DataFrame(index=pd.Index(panel.symbols, name="symbol"))
    table["valid_pct"] = panel.valid.mean().to_numpy() * 100
    for name, frame in panel.features.items():
        table[f"{name}_nan_pct"] = frame.isna().mean().to_numpy() * 100
    return table


def feature_quantiles(
    panel: Panel, *, quantiles: Sequence[float] = DEFAULT_QUANTILES
) -> pd.DataFrame:
    """Cuantiles de cada feature (filas) sobre todos sus valores no `NaN`, de todos los simbolos."""
    if not quantiles or any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError(f"quantiles debe ser no vacio y estar en [0, 1]: {tuple(quantiles)}")
    rows: dict[str, np.ndarray] = {}
    for name, frame in panel.features.items():
        values = frame.to_numpy(dtype="float64").ravel()
        values = values[~np.isnan(values)]
        # Una feature sin ningun valor da NaN en todos los cuantiles, no un error ni un 0 (D-02).
        rows[name] = (
            np.quantile(values, list(quantiles)) if values.size else np.full(len(quantiles), np.nan)
        )
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"q{q:g}" for q in quantiles])


def _complete_bars(
    valid_open: pd.DataFrame, base_interval: str, target_interval: str
) -> pd.DataFrame:
    """Recorta a las barras completas alineadas al reloj: sin el principio ni el final partidos."""
    factor = bar_factor(base_interval, target_interval)
    base = interval_ms(base_interval) or 0  # `bar_factor` ya ha comprobado que tiene paso fijo
    slots = grid_ms(valid_open.index)
    aligned = np.flatnonzero(slots % (base * factor) == 0)
    first = int(aligned[0]) if aligned.size else len(slots)
    usable = (len(slots) - first) // factor * factor
    if usable == 0:
        raise ValueError(f"panel mas corto que una barra de {target_interval}")
    return valid_open.iloc[first : first + usable]


def bar_validity_pct(
    panel: Panel,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    *,
    min_valid_share: float = 0.8,
) -> pd.DataFrame:
    """Por simbolo, % de barras validas a cada horizonte (D-23); `1m` es el % de minutos validos."""
    base = interval_ms(panel.interval)
    if base is None:
        raise ValueError(f"El intervalo {panel.interval!r} no tiene paso fijo")
    # `panel.valid` va indexado al cierre (D-19): la fila `tau` es la vela abierta en `tau - paso`.
    # Volver a la apertura es solo cambiar la etiqueta de cada fila, sin mover ningun dato en el
    # tiempo, asi que no hay fuga; `resample_validity` agrupa despues cada barra con sus propios
    # minutos.
    opens = pd.DatetimeIndex((grid_ms(panel.valid.index) - base).astype("datetime64[ms]"), tz="UTC")
    valid_open = panel.valid.set_axis(opens, axis=0)
    result = pd.DataFrame(index=pd.Index(panel.symbols, name="symbol"))
    result[panel.interval] = panel.valid.mean().to_numpy() * 100
    for horizon in horizons:
        # Solo barras completas: la primera vela del panel (cierra en `start`) pertenece a la barra
        # anterior y la ultima barra no tiene su ultimo minuto dentro del panel.
        bars = _complete_bars(valid_open, panel.interval, horizon)
        result[horizon] = (
            resample_validity(bars, panel.interval, horizon, min_valid_share=min_valid_share)
            .mean()
            .to_numpy()
            * 100
        )
    return result


def period_gap_causes(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    listed_ms: dict[str, int] | None = None,
    data_end_ms: dict[str, int] | None = None,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> pd.DataFrame:
    """Slots de cada simbolo (filas) por causa de hueco (columnas) en `[start_ms, end_ms)`.

    La etiqueta es ex post (D-17): solo describe, no sirve de feature.
    """
    causes = classify_klines(
        symbols,
        interval,
        start_ms,
        end_ms,
        base_dir=base_dir,
        listed_ms=listed_ms,
        data_end_ms=data_end_ms,
    )
    return cause_counts(causes)


def correlation_by_liquidity(
    corr: pd.DataFrame, valid_pct: pd.Series, *, illiquid_below: float = 97.0
) -> dict[str, float | int]:
    """Correlacion media de los pares entre liquidas y de los pares con alguna iliquida.

    Es iliquida la moneda con datos y menos de `illiquid_below` % de minutos validos (umbral
    arbitrario); una media sin pares queda en `NaN`.
    """
    pct = valid_pct.reindex(corr.index)
    if pct.isna().any():
        raise ValueError("valid_pct no tiene todas las monedas de la matriz")
    matrix = corr.to_numpy(dtype="float64")
    # Una moneda sin datos (diagonal NaN) no tiene pares: no cuenta como iliquida.
    illiquid = (pct < illiquid_below).to_numpy() & ~np.isnan(np.diag(matrix))
    rows, cols = np.triu_indices(len(corr), k=1)
    values = matrix[rows, cols]
    has_value = ~np.isnan(values)
    both_liquid = has_value & ~illiquid[rows] & ~illiquid[cols]
    any_illiquid = has_value & (illiquid[rows] | illiquid[cols])
    return {
        "n_illiquid": int(illiquid.sum()),
        "pair_mean_liquid": float(values[both_liquid].mean()) if both_liquid.any() else np.nan,
        "pair_mean_with_illiquid": (
            float(values[any_illiquid].mean()) if any_illiquid.any() else np.nan
        ),
    }


def _correlation_row(
    returns: pd.DataFrame,
    valid_pct: pd.Series,
    *,
    factor: str,
    min_periods: int,
    high: float,
    max_nan_share: float,
    illiquid_below: float,
) -> dict[str, float | int | str]:
    """Resumen de correlacion cruda de un periodo; una fila vacia con `error` si no se puede."""
    label = f"{returns.index[0]:%Y-%m-%d}"
    corr = raw_correlation(returns, min_periods=min_periods)
    try:
        summary, _, without = summarize_correlation(
            corr, factor=factor, high=high, max_nan_share=max_nan_share, label=label
        )
    except ValueError as exc:
        logger.warning("Periodo %s sin resumen de correlacion: %s", label, exc)
        return {"n_symbols_in_universe": len(corr), "error": str(exc)}
    return {
        "n_symbols_in_universe": len(corr),
        "n_symbols_without_data": len(without),
        "symbols_without_data": ";".join(without),
        "error": "",
        **summary,
        **correlation_by_liquidity(corr, valid_pct, illiquid_below=illiquid_below),
    }


def build_panel_overview(
    universe: pd.DataFrame,
    interval: str = "1m",
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    cache_dir: str | Path = PANEL_DIR,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    min_valid_share: float = 0.8,
    factor: str = "BTCUSDT",
    min_periods: int = 1000,
    high: float = 0.5,
    max_nan_share: float = 0.05,
    illiquid_below: float = 97.0,
    include_gap_causes: bool = True,
) -> PanelOverview:
    """Recorre los paneles de la cache (cada uno se carga una vez) y arma las tablas del resumen.

    Construye los paneles que falten; `include_gap_causes=False` omite las causas de hueco, que
    releen las velas (unos segundos por trimestre), y deja `gap_causes` vacia.
    """
    listed = listings_from_universe(universe) if include_gap_causes else None
    data_end = download_ends_from_universe(universe) if include_gap_causes else None
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene paso fijo")
    periods: dict[pd.Timestamp, dict[str, object]] = {}
    quality: dict[pd.Timestamp, pd.DataFrame] = {}
    quants: dict[pd.Timestamp, pd.DataFrame] = {}
    tails: dict[pd.Timestamp, pd.DataFrame] = {}
    bars: dict[pd.Timestamp, pd.DataFrame] = {}
    causes: dict[pd.Timestamp, pd.DataFrame] = {}
    correlation: dict[pd.Timestamp, dict[str, float | int | str]] = {}
    for period_start, panel in cached_period_panels(
        universe, interval, base_dir=base_dir, cache_dir=cache_dir
    ):
        logger.info("Resumen del panel %s", f"{period_start:%Y-%m-%d}")
        # `cached_period_panels` ya valida al cargar; se repite para dejar el resultado anotado.
        try:
            validate_panel(panel)
            contract_ok, contract_error = True, ""
        except ValueError as exc:
            contract_ok, contract_error = False, str(exc)
        periods[period_start] = {
            "n_rows": panel.shape[0],
            "n_symbols": panel.shape[1],
            "n_features": panel.shape[2],
            "contract_ok": contract_ok,
            "contract_error": contract_error,
        }
        quality[period_start] = symbol_quality(panel)
        quants[period_start] = feature_quantiles(panel, quantiles=quantiles)
        tails[period_start] = tail_report(panel.features["return"])
        bars[period_start] = bar_validity_pct(panel, horizons, min_valid_share=min_valid_share)
        correlation[period_start] = _correlation_row(
            panel.features["return"],
            quality[period_start]["valid_pct"],
            factor=factor,
            min_periods=min_periods,
            high=high,
            max_nan_share=max_nan_share,
            illiquid_below=illiquid_below,
        )
        if include_gap_causes:
            # Las filas del panel van por el cierre: sus velas se abrieron un paso antes, asi que
            # las causas se piden en ese rango (mismos slots que `valid`) y no mira mas alla.
            start_ms = to_ms(period_start) - step
            causes[period_start] = period_gap_causes(
                panel.symbols,
                interval,
                start_ms,
                start_ms + panel.shape[0] * step,
                listed_ms=listed,
                data_end_ms=data_end,
                base_dir=base_dir,
            )
        del panel
    return PanelOverview(
        periods=pd.DataFrame.from_dict(periods, orient="index").rename_axis("period_start"),
        quality=_stack(quality, "symbol"),
        feature_quantiles=_stack(quants, "feature"),
        tails=_stack(tails, "symbol"),
        bar_validity=_stack(bars, "symbol"),
        gap_causes=_stack(causes, "symbol")
        if causes
        else pd.DataFrame(columns=[c.name for c in GapCause]),
        correlation=pd.DataFrame.from_dict(correlation, orient="index").rename_axis("period_start"),
    )


def _stack(frames: dict[pd.Timestamp, pd.DataFrame], level: str) -> pd.DataFrame:
    """Une tablas por trimestre en una con indice `(period_start, level)`."""
    table = pd.concat(frames, names=["period_start"])
    table.index = table.index.set_names(["period_start", level])
    return table
