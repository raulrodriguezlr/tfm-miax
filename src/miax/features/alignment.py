"""Alineacion de las series de klines a una rejilla temporal comun en UTC; lo que falta es NaN."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from miax.ingest.klines import KLINE_DTYPES, interval_ms, slot_bounds
from miax.ingest.store import DEFAULT_BASE_DIR, load_klines
from miax.utils.logging import get_logger

logger = get_logger(__name__)

# Campos con sentido de serie: los numericos de las klines salvo los tiempos y `ignore`.
ALIGNABLE_FIELDS = tuple(c for c in KLINE_DTYPES if c not in ("open_time", "close_time", "ignore"))


def _check_fields(fields: Sequence[str]) -> list[str]:
    """Devuelve `fields` como lista si todos son alineables; si no, `ValueError`."""
    invalid = [f for f in fields if f not in ALIGNABLE_FIELDS]
    if invalid:
        raise ValueError(f"Campos no alineables: {invalid}; validos: {list(ALIGNABLE_FIELDS)}")
    return list(fields)


def _grid_ms(grid: pd.DatetimeIndex) -> np.ndarray:
    """Devuelve la rejilla como `int64` en ms desde la epoca UTC."""
    return grid.tz_convert("UTC").as_unit("ms").asi8


def _positions(open_time: np.ndarray, grid_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Da la mascara de velas dentro de la rejilla y su fila en ella; falla si hay duplicadas."""
    # Solo cuenta el rango de la rejilla [primer slot, ultimo slot]; el resto se ignora sin mirar.
    inside = (open_time >= grid_ms[0]) & (open_time <= grid_ms[-1])
    times = open_time[inside]
    if len(np.unique(times)) != len(times):
        raise ValueError("open_time duplicado")
    rows = np.searchsorted(grid_ms, times)
    # `searchsorted` cae al slot siguiente si la vela no esta en la rejilla: se detecta aqui.
    if not np.array_equal(grid_ms[rows], times):
        raise ValueError("open_time fuera de la rejilla (desalineado)")
    return inside, rows


def build_time_grid(start_ms: int, end_ms: int, interval: str = "1m") -> pd.DatetimeIndex:
    """Crea la rejilla de aperturas de vela UTC en `[start_ms, end_ms)`, sin `freq`."""
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    first_slot, _, expected = slot_bounds(start_ms, end_ms, interval)
    slots = first_slot + np.arange(expected, dtype="int64") * step
    # Con huecos no hay frecuencia regular: el indice nace con `freq = None` y no se infiere.
    return pd.DatetimeIndex(slots.astype("datetime64[ms]"), tz="UTC")


def align_frame(frame: pd.DataFrame, grid: pd.DatetimeIndex, fields: Sequence[str]) -> pd.DataFrame:
    """Reindexa las velas de un simbolo a `grid` en `float64`; las que faltan quedan `NaN`."""
    columns = _check_fields(fields)
    missing = [c for c in ["open_time", *columns] if c not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas para alinear: {missing}")
    result = np.full((len(grid), len(columns)), np.nan, dtype="float64")
    if len(grid) and len(frame):
        inside, rows = _positions(frame["open_time"].to_numpy(dtype="int64"), _grid_ms(grid))
        for j, field in enumerate(columns):
            result[rows, j] = frame[field].to_numpy(dtype="float64")[inside]
    return pd.DataFrame(result, index=grid, columns=columns)


def align_klines(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    fields: Sequence[str] = ("close",),
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> dict[str, pd.DataFrame]:
    """Alinea los simbolos a una rejilla comun en `[start_ms, end_ms)` y da un ancho por campo.

    Cada DataFrame es `tiempo x simbolo` en `float64` (`num_trades` tambien, porque un entero no
    admite `NaN`); lo que falta es `NaN`, nunca 0 ni relleno (D-02), y un simbolo sin cache es
    una columna entera de `NaN`.
    """
    if not symbols:
        raise ValueError("symbols vacio: no hay nada que alinear")
    if not fields:
        raise ValueError("fields vacio: hay que pedir al menos un campo")
    columns = _check_fields(fields)
    names = list(dict.fromkeys(symbols))
    grid = build_time_grid(start_ms, end_ms, interval)
    if not len(grid):
        raise ValueError("rejilla vacia: el rango no contiene ningun slot")
    grid_ms = _grid_ms(grid)
    # Un array por campo, rellenado simbolo a simbolo: no se acumulan frames intermedios.
    arrays = {f: np.full((len(grid), len(names)), np.nan, dtype="float64") for f in columns}
    for k, symbol in enumerate(names):
        # `load_klines` es inclusivo en `end_ms` y la rejilla es semiabierta: se pide `end_ms - 1`
        # para que la vela de `end_ms` (futuro del rango) no entre.
        candles = load_klines(
            symbol,
            interval,
            base_dir=base_dir,
            start_ms=start_ms,
            end_ms=end_ms - 1,
            columns=["open_time", *columns],
        )
        if candles.empty:
            logger.warning("Sin velas de %s %s en el rango: columna de NaN", symbol, interval)
            continue
        try:
            inside, rows = _positions(candles["open_time"].to_numpy(dtype="int64"), grid_ms)
        except ValueError as exc:
            raise ValueError(
                f"{symbol} {interval}: {exc}; revisa la cache con "
                f"`python -m miax.ingest validate --symbol {symbol} --interval {interval}`"
            ) from exc
        for field in columns:
            arrays[field][rows, k] = candles[field].to_numpy(dtype="float64")[inside]
    # `copy=False` evita duplicar cada array al crear su DataFrame (pico de 2x la memoria): los
    # arrays son locales y no se vuelven a tocar, asi que compartirlos es seguro.
    return {f: pd.DataFrame(arrays[f], index=grid, columns=names, copy=False) for f in columns}
