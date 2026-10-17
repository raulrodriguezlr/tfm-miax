"""Retornos logaritmicos a 1 paso sobre precios alineados, validos segun la mascara."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from miax.features.alignment import align_klines, build_time_grid, grid_ms
from miax.features.gaps import classify_klines
from miax.features.validity import mask_invalid, return_validity, validity_mask
from miax.ingest.klines import interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR


def log_returns(close: pd.DataFrame, valid: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Retorno logaritmico de cierre a cierre a 1 paso, indexado por el instante en que se conoce.

    La fila `tau` es el retorno de la vela que cierra en `tau` (`open_time + intervalo`); `NaN`
    donde esa vela o la anterior no son validas, y la primera fila es `NaN`.
    """
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    # `mask_invalid` ya exige mismo indice y columnas, `close` float y `valid` bool.
    prices = mask_invalid(close, valid)
    # Un NaN enmascarado es un hueco legitimo; en un slot valido, no finito o <= 0 es dato corrupto.
    values = prices.to_numpy()
    corrupt = valid.to_numpy(dtype=bool) & ~(np.isfinite(values) & (values > 0))
    bad = prices.columns[corrupt.any(axis=0)].tolist()
    if bad:
        raise ValueError(f"precio no finito o <= 0 en slots validos de: {bad}")
    # diff() solo trae el cierre de t-1 a la fila t: no mira al futuro (la 1.a fila no tiene t-1).
    steps = mask_invalid(np.log(prices).diff(), return_validity(valid))
    # La fila pasa de la apertura al cierre de su vela: el instante en que el retorno ya se conoce.
    known_at = pd.DatetimeIndex((grid_ms(steps.index) + step).astype("datetime64[ms]"), tz="UTC")
    return steps.set_axis(known_at, axis=0)


def returns_from_cache(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> pd.DataFrame:
    """Lee la cache y da los retornos log conocidos en `[start_ms, end_ms)`, ya calculados.

    El primero de un periodo que empieza a las 00:00 es el de la vela 23:59, conocido a las 00:00.
    """
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    grid = build_time_grid(start_ms, end_ms, interval)
    if not len(grid):
        raise ValueError("rejilla vacia: el rango no contiene ningun slot")
    # El retorno conocido en tau es el de la vela de tau - paso, que necesita la de tau - 2*paso:
    # ambas son ANTERIORES a tau (pasado), asi el primer slot tiene retorno sin mirar al futuro.
    wide_start = start_ms - 2 * step
    wide_end = end_ms - step
    close = align_klines(
        symbols, interval, wide_start, wide_end, fields=("close",), base_dir=base_dir
    )["close"]
    causes = classify_klines(symbols, interval, wide_start, wide_end, base_dir=base_dir)
    returns = log_returns(close, validity_mask(causes), interval)
    # La primera fila (conocida en start - paso) cae fuera del rango pedido: se quita.
    returns = returns.iloc[1:]
    if not returns.index.equals(grid):
        raise RuntimeError("el indice de retornos no coincide con la rejilla del rango")
    return returns
