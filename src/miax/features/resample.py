"""Remuestreo de las velas de 1m a barras de 5m, 15m o 1h, con validez recalculada por barra."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from miax.features.alignment import align_klines, build_time_grid, grid_ms
from miax.features.gaps import classify_klines
from miax.features.returns import log_returns
from miax.features.validity import mask_invalid, validity_mask
from miax.features.volatility import check_window_params
from miax.ingest.klines import candle_offset_ms, interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR

# Como se agrega cada campo de una vela: primero, maximo, minimo, ultimo o suma de lo observado.
FIELD_AGGREGATION: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_asset_volume": "sum",
    "num_trades": "sum",
    "taker_buy_base": "sum",
    "taker_buy_quote": "sum",
}

# Campos de precio cuyo primer valor y extremos solo salen de minutos validos (no de velas rancias).
PRICE_EXTREMES_FIELDS: tuple[str, ...] = ("open", "high", "low")


def _fixed_step(interval: str) -> int:
    """Devuelve el paso de `interval` en ms; `ValueError` si no tiene duracion fija."""
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    return step


def bar_factor(base_interval: str, target_interval: str) -> int:
    """Cuantas velas base caben en una barra objetivo; `ValueError` si no es un multiplo entero."""
    base = _fixed_step(base_interval)
    target = _fixed_step(target_interval)
    if candle_offset_ms(base_interval) or candle_offset_ms(target_interval):
        raise ValueError("las barras se alinean al reloj desde la epoca: no se admiten semanas")
    if target < base:
        raise ValueError(f"{target_interval!r} es mas corto que {base_interval!r}: no se agrega")
    if target % base:
        raise ValueError(f"{target_interval!r} no es multiplo entero de {base_interval!r}")
    return target // base


def _check_base_grid(index: pd.DatetimeIndex, base_interval: str, target_interval: str) -> int:
    """Comprueba que `index` es la rejilla contigua del origen en barras completas; da el factor."""
    factor = bar_factor(base_interval, target_interval)
    if not len(index):
        raise ValueError("indice vacio: no hay nada que remuestrear")
    base = _fixed_step(base_interval)
    target = _fixed_step(target_interval)
    slots = grid_ms(index)
    if len(slots) > 1 and not (np.diff(slots) == base).all():
        raise ValueError(f"el indice no es la rejilla contigua de {base_interval!r}")
    if slots[0] % target:
        raise ValueError(f"el indice no empieza en un multiplo de {target_interval!r}")
    if len(slots) % factor:
        raise ValueError(f"la longitud {len(slots)} no es multiplo de {factor} (barra incompleta)")
    return factor


def _bar_index(index: pd.DatetimeIndex, factor: int) -> pd.DatetimeIndex:
    """Aperturas de barra: la primera fila de cada grupo, sin `freq` como el resto del paquete."""
    return pd.DatetimeIndex(grid_ms(index)[::factor].astype("datetime64[ms]"), tz="UTC")


def resample_validity(
    valid: pd.DataFrame,
    base_interval: str,
    target_interval: str,
    *,
    min_valid_share: float = 0.8,
) -> pd.DataFrame:
    """Barra valida si su ultimo minuto es valido y la fraccion de minutos validos es suficiente.

    La fraccion se compara con `>= min_valid_share`; con 1.0 es la regla estricta.
    """
    check_window_params(1, min_valid_share)
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in valid.dtypes):
        raise ValueError("valid debe ser bool en todas sus columnas: un NaN no es False")
    factor = _check_base_grid(valid.index, base_interval, target_interval)
    # Cada barra agrupa solo las `factor` filas de su propio intervalo: no usa filas posteriores.
    groups = valid.to_numpy(dtype=bool).reshape(-1, factor, valid.shape[1])
    last_valid = groups[:, -1, :]
    share = groups.sum(axis=1) / factor
    result = last_valid & (share >= min_valid_share)
    return pd.DataFrame(result, index=_bar_index(valid.index, factor), columns=valid.columns)


def _aggregate(groups: np.ndarray, how: str) -> np.ndarray:
    """Agrega un array `(barras, minutos, simbolos)` sobre los minutos sin warnings con `NaN`."""
    observed = ~np.isnan(groups)
    any_observed = observed.any(axis=1)
    if how == "last":
        return groups[:, -1, :]
    if how == "first":
        first = np.argmax(observed, axis=1)
        return np.take_along_axis(groups, first[:, None, :], axis=1)[:, 0, :]
    if how == "max":
        top = np.where(observed, groups, -np.inf).max(axis=1)
        return np.where(any_observed, top, np.nan)
    if how == "min":
        bottom = np.where(observed, groups, np.inf).min(axis=1)
        return np.where(any_observed, bottom, np.nan)
    if how == "sum":
        total = np.where(observed, groups, 0.0).sum(axis=1)
        return np.where(any_observed, total, np.nan)
    raise ValueError(f"agregacion desconocida: {how!r}")


def resample_fields(
    fields: Mapping[str, pd.DataFrame],
    valid: pd.DataFrame,
    base_interval: str,
    target_interval: str,
    *,
    min_valid_share: float = 0.8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Agrega cada campo a barras y devuelve los campos enmascarados y la validez de barra.

    `open`, `high` y `low` ignoran los minutos no validos (repiten el cierre anterior); lo no
    valido es `NaN`, nunca 0.
    """
    unknown = [name for name in fields if name not in FIELD_AGGREGATION]
    if unknown:
        raise ValueError(f"Campos no remuestreables: {unknown}; validos: {list(FIELD_AGGREGATION)}")
    bar_valid = resample_validity(
        valid, base_interval, target_interval, min_valid_share=min_valid_share
    )
    factor = _check_base_grid(valid.index, base_interval, target_interval)
    result: dict[str, pd.DataFrame] = {}
    for name, frame in fields.items():
        if not frame.index.equals(valid.index) or not frame.columns.equals(valid.columns):
            raise ValueError(f"{name}: el indice y las columnas deben coincidir con los de valid")
        if not all(pd.api.types.is_float_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError(f"{name} debe ser float en todas las columnas")
        values = frame.to_numpy(dtype="float64")
        if name in PRICE_EXTREMES_FIELDS:
            # Un minuto no valido (p. ej. NO_TRADES) repite el cierre anterior: no es de la barra.
            values = np.where(valid.to_numpy(dtype=bool), values, np.nan)
        # Cada barra agrega solo las `factor` filas de su propio intervalo: ninguna es posterior.
        groups = values.reshape(-1, factor, frame.shape[1])
        bars = pd.DataFrame(
            _aggregate(groups, FIELD_AGGREGATION[name]),
            index=bar_valid.index,
            columns=frame.columns,
        )
        result[name] = mask_invalid(bars, bar_valid)
    return result, bar_valid


def _check_range(start_ms: int, end_ms: int, target_interval: str) -> None:
    """Exige un rango no vacio cuyos extremos son multiplos del intervalo objetivo."""
    target = _fixed_step(target_interval)
    if start_ms % target or end_ms % target:
        raise ValueError(
            f"start_ms y end_ms deben ser multiplos de {target_interval!r}: "
            "una barra parcial mezclaria horizontes"
        )
    if start_ms >= end_ms:
        raise ValueError(f"Rango vacio o invertido: [{start_ms}, {end_ms})")


def resample_from_cache(
    symbols: Sequence[str],
    base_interval: str,
    target_interval: str,
    start_ms: int,
    end_ms: int,
    *,
    fields: Sequence[str] = ("close",),
    min_valid_share: float = 0.8,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Lee la cache y da las barras de `[start_ms, end_ms)` por campo, mas la validez de barra.

    Los campos van indexados por la apertura de la barra; `start_ms` y `end_ms` deben ser
    multiplos de `target_interval`.
    """
    bar_factor(base_interval, target_interval)
    check_window_params(1, min_valid_share)
    _check_range(start_ms, end_ms, target_interval)
    # Solo se leen minutos de `[start_ms, end_ms)`: cada barra usa filas de su propio intervalo.
    aligned = align_klines(
        symbols, base_interval, start_ms, end_ms, fields=fields, base_dir=base_dir
    )
    causes = classify_klines(symbols, base_interval, start_ms, end_ms, base_dir=base_dir)
    return resample_fields(
        aligned,
        validity_mask(causes),
        base_interval,
        target_interval,
        min_valid_share=min_valid_share,
    )


def horizon_returns_from_cache(
    symbols: Sequence[str],
    target_interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_interval: str = "1m",
    min_valid_share: float = 0.8,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> pd.DataFrame:
    """Lee la cache y da los retornos log a horizonte `target_interval` conocidos en el rango.

    La fila `tau` es el retorno de la barra que cierra en `tau` (`open_time + H`); con
    `target_interval == base_interval` coincide con `returns_from_cache`.
    """
    bar_factor(base_interval, target_interval)
    check_window_params(1, min_valid_share)
    _check_range(start_ms, end_ms, target_interval)
    step = _fixed_step(target_interval)
    grid = build_time_grid(start_ms, end_ms, target_interval)
    # El retorno conocido en tau usa las barras de tau - H y tau - 2H, ambas ANTERIORES a tau:
    # el primer slot tiene retorno sin mirar al futuro.
    wide_start = start_ms - 2 * step
    wide_end = end_ms - step
    fields, bar_valid = resample_from_cache(
        symbols,
        base_interval,
        target_interval,
        wide_start,
        wide_end,
        fields=("close",),
        min_valid_share=min_valid_share,
        base_dir=base_dir,
    )
    returns = log_returns(fields["close"], bar_valid, target_interval)
    # La primera fila (conocida en start - H) cae fuera del rango pedido: se quita.
    returns = returns.iloc[1:]
    if not returns.index.equals(grid):
        raise RuntimeError("el indice de retornos no coincide con la rejilla del rango")
    return returns
