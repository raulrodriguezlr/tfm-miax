"""Volatilidad realizada por ventana sobre retornos log, reescalada por los que faltan."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from miax.features.alignment import build_time_grid
from miax.features.returns import returns_from_cache
from miax.ingest.klines import interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR


def check_window_params(window: int, min_valid_share: float) -> None:
    """Valida los parametros de la ventana; lanza `ValueError` si no valen."""
    if window < 1:
        raise ValueError(f"window debe ser >= 1, no {window}")
    if not 0.0 < min_valid_share <= 1.0:
        raise ValueError(f"min_valid_share debe estar en (0, 1], no {min_valid_share}")


def realized_volatility(
    returns: pd.DataFrame, *, window: int = 60, min_valid_share: float = 0.8
) -> pd.DataFrame:
    """Raiz de la suma de `r^2` de la ventana de `window` filas que acaba en `tau`, reescalada.

    Reescala por `window / n` (con `n` los retornos no `NaN` de la ventana) y da `NaN` si
    `n / window < min_valid_share`; las filas que faltan al principio cuentan como no validas.
    """
    check_window_params(window, min_valid_share)
    if not all(pd.api.types.is_float_dtype(dtype) for dtype in returns.dtypes):
        raise ValueError("returns debe ser float en todas las columnas")
    if not (returns.index.is_monotonic_increasing and returns.index.is_unique):
        raise ValueError("el indice de returns debe ser estrictamente creciente")
    # Ventana solo hacia atras e inclusiva de tau: r_s con s <= tau ya se conocen en tau, sin shift.
    # NaN -> 0 solo para sumar; el recuento de validos (no el de filas) decide si la fila vale.
    squares = returns.pow(2).fillna(0.0).rolling(window, min_periods=1).sum()
    valid = returns.notna().astype(float).rolling(window, min_periods=1).sum()
    # El denominador del share es `window` fijo: una ventana parcial no tiene mejor cobertura.
    enough = (valid / window) >= min_valid_share
    # Si n == 0 `enough` es False (min_valid_share > 0): n queda NaN y no hay division por cero.
    rescaled = squares * (window / valid.where(enough))
    # Salvaguarda: el redondeo de la suma deslizante podria dejar un negativo diminuto,
    # y `sqrt` daria NaN con aviso.
    return np.sqrt(rescaled.clip(lower=0.0)).where(enough).astype("float64")


def realized_volatility_from_cache(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    window: int = 60,
    min_valid_share: float = 0.8,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> pd.DataFrame:
    """Lee la cache y da la volatilidad realizada conocida en `[start_ms, end_ms)`.

    Pide `window - 1` retornos anteriores al rango (pasado) para que la ventana del primer slot
    este completa; el indice es exactamente `build_time_grid(start_ms, end_ms, interval)`.
    """
    check_window_params(window, min_valid_share)
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    grid = build_time_grid(start_ms, end_ms, interval)
    if not len(grid):
        raise ValueError("rejilla vacia: el rango no contiene ningun slot")
    # Los window - 1 retornos extra son ANTERIORES a start_ms (pasado): la ventana no mira adelante.
    lead = window - 1
    returns = returns_from_cache(
        symbols, interval, start_ms - lead * step, end_ms, base_dir=base_dir
    )
    volatility = realized_volatility(returns, window=window, min_valid_share=min_valid_share)
    # Se quitan las `lead` primeras filas, que solo servian para llenar la ventana del primer slot.
    volatility = volatility.iloc[lead:]
    if not volatility.index.equals(grid):
        raise RuntimeError("el indice de volatilidad no coincide con la rejilla del rango")
    return volatility
