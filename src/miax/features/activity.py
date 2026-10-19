"""Actividad (volumen, operaciones) relativa al propio pasado, comparable entre activos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from miax.features.alignment import (
    align_klines,
    build_time_grid,
    shift_to_close,
)
from miax.features.gaps import classify_klines
from miax.features.validity import mask_invalid, validity_mask
from miax.features.volatility import check_window_params
from miax.ingest.klines import interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR


def log_relative_to_past(
    values: pd.DataFrame,
    valid: pd.DataFrame,
    interval: str,
    *,
    window: int = 10080,
    min_valid_share: float = 0.5,
) -> pd.DataFrame:
    """Logaritmo de `values` menos la media del log de las `window` filas anteriores validas.

    La base de la fila `tau` no incluye la propia vela; es `NaN` si menos de `min_valid_share` de la
    ventana es valida. La salida se indexa por el cierre de la vela (`open_time + intervalo`).
    """
    check_window_params(window, min_valid_share)
    if interval_ms(interval) is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    # `mask_invalid` ya exige mismo indice y columnas, `values` float y `valid` bool.
    masked = mask_invalid(values, valid)
    if not (values.index.is_monotonic_increasing and values.index.is_unique):
        raise ValueError("el indice de values debe ser estrictamente creciente")
    # Un NaN enmascarado es un hueco legitimo; en un slot valido, no finito o <= 0 es dato corrupto.
    raw = masked.to_numpy()
    corrupt = valid.to_numpy(dtype=bool) & ~(np.isfinite(raw) & (raw > 0))
    bad = masked.columns[corrupt.any(axis=0)].tolist()
    if bad:
        raise ValueError(f"valor no finito o <= 0 en slots validos de: {bad}")
    # Se liberan los intermedios en cuanto dejan de usarse: con muchos pares a 1m manda el pico.
    logs = np.log(masked)
    del masked, raw, corrupt
    # La ventana de tau son las filas tau-window..tau-1 de `logs`: se suma hasta tau-1 y se
    # desplaza 1, asi la base no incluye la propia vela ni el futuro.
    # NaN -> 0 solo para sumar; el recuento de validos (no el de filas) decide si la base vale.
    total = logs.fillna(0.0).rolling(window, min_periods=1).sum().shift(1)
    count = logs.notna().astype(float).rolling(window, min_periods=1).sum().shift(1)
    # El denominador del share es `window` fijo: una ventana parcial no tiene mejor cobertura.
    enough = (count / window) >= min_valid_share
    # Si count == 0 `enough` es False (min_valid_share > 0): no hay division por cero.
    base = total / count.where(enough)
    del total, count, enough
    relative = (logs - base).astype("float64")
    del logs, base
    return relative.set_axis(shift_to_close(relative.index, interval), axis=0)


def relative_volume_from_cache(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    window: int = 10080,
    min_valid_share: float = 0.5,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> pd.DataFrame:
    """Lee la cache y da el volumen relativo (`quote_asset_volume`) conocido en `[start, end)`.

    Pide `window + 1` velas anteriores al rango (pasado) para que la base del primer slot este
    completa; el indice es exactamente `build_time_grid(start_ms, end_ms, interval)`.
    """
    check_window_params(window, min_valid_share)
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    grid = build_time_grid(start_ms, end_ms, interval)
    if not len(grid):
        raise ValueError("rejilla vacia: el rango no contiene ningun slot")
    # El slot conocido en start es la vela de start - paso, y su base son las `window` velas
    # anteriores a esa: todo es ANTERIOR a start_ms (pasado), asi el primer slot no mira adelante.
    wide_start = start_ms - (window + 1) * step
    wide_end = end_ms - step
    volume = align_klines(
        symbols, interval, wide_start, wide_end, fields=("quote_asset_volume",), base_dir=base_dir
    )["quote_asset_volume"]
    causes = classify_klines(symbols, interval, wide_start, wide_end, base_dir=base_dir)
    relative = log_relative_to_past(
        volume, validity_mask(causes), interval, window=window, min_valid_share=min_valid_share
    )
    # La salida cubre [wide_start + paso, end_ms): las `window` primeras filas caen antes de
    # start_ms y solo servian para llenar la base del primer slot, asi que se quitan.
    relative = relative.iloc[window:]
    if not relative.index.equals(grid):
        raise RuntimeError("el indice de volumen relativo no coincide con la rejilla del rango")
    return relative
