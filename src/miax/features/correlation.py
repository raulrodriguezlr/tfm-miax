"""Correlacion por pares entre activos y sus autovalores, sin rellenar huecos con 0 (D-02)."""

from __future__ import annotations

import numpy as np
import pandas as pd

_NEGATIVE_EIGENVALUE_TOLERANCE = 1e-8


def raw_correlation(returns: pd.DataFrame, *, min_periods: int = 1000) -> pd.DataFrame:
    """Pearson por pares con observaciones comunes; `NaN` si un par tiene menos de `min_periods`.

    Una serie constante, toda `NaN` o con menos de `min_periods` datos queda con su fila, columna y
    diagonal en `NaN`: no se inventa una correlacion de 1 consigo misma.
    """
    if min_periods < 2:
        raise ValueError(f"min_periods debe ser al menos 2: {min_periods}")
    if not returns.columns.is_unique:
        raise ValueError("returns tiene columnas duplicadas")
    # Solo pares de observaciones del mismo instante: sin retardos, no se mira otra fila.
    corr = returns.astype("float64").corr(method="pearson", min_periods=min_periods)
    values = corr.to_numpy(copy=True)
    # Solo se fija a 1 la diagonal que pandas pudo calcular (sin varianza o sin datos es NaN).
    diagonal = np.diag(values).copy()
    np.fill_diagonal(values, np.where(np.isnan(diagonal), np.nan, 1.0))
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def symbols_without_data(corr: pd.DataFrame) -> list[str]:
    """Simbolos con la diagonal en `NaN` (sin datos, constantes o con menos de `min_periods`).

    Mantiene el orden de la matriz; `raw_correlation` deja asi a esas series.
    """
    diagonal = np.diag(corr.to_numpy(dtype="float64"))
    return [
        str(symbol) for symbol, value in zip(corr.index, diagonal, strict=True) if np.isnan(value)
    ]


def off_diagonal(corr: pd.DataFrame) -> np.ndarray:
    """Triangulo superior sin la diagonal, como vector (incluye los `NaN`)."""
    rows, cols = np.triu_indices(len(corr), k=1)
    return corr.to_numpy(dtype="float64")[rows, cols]


def count_missing_pairs(corr: pd.DataFrame) -> int:
    """Pares distintos (triangulo superior, sin diagonal) de la matriz sin valor."""
    return int(np.isnan(off_diagonal(corr)).sum())


def explained_variance(
    corr: pd.DataFrame, *, max_nan_share: float = 0.05, require_psd: bool = True
) -> np.ndarray:
    """Autovalores de la matriz de correlacion, de mayor a menor, como fraccion de N.

    Los pares sin valor se sustituyen por la media de los pares con valor; `ValueError` si faltan
    mas del `max_nan_share` de los pares o, con `require_psd`, si algun autovalor baja de `-1e-8`.
    """
    if not 0.0 <= max_nan_share < 1.0:
        raise ValueError(f"max_nan_share debe estar en [0, 1): {max_nan_share}")
    n = len(corr)
    if n < 2 or corr.shape[1] != n:
        raise ValueError("la matriz de correlacion debe ser cuadrada y de al menos 2x2")
    values = corr.to_numpy(dtype="float64", copy=True)
    pairs = off_diagonal(corr)
    missing = np.isnan(pairs)
    if missing.mean() > max_nan_share:
        raise ValueError(
            f"faltan {missing.mean():.1%} de los pares, mas que max_nan_share={max_nan_share:.1%}"
        )
    if missing.any():
        # Rellenar con 0 mentiria (D-02): se usa la media de los pares observados.
        fill = float(np.nanmean(pairs))
        values[np.isnan(values)] = fill
    np.fill_diagonal(values, 1.0)
    values = (values + values.T) / 2
    eigenvalues = np.linalg.eigvalsh(values)[::-1]
    if require_psd and eigenvalues[-1] < -_NEGATIVE_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"la matriz de correlacion no es semidefinida positiva: autovalor minimo "
            f"{eigenvalues[-1]:.3g} (el relleno de pares sin valor no lo garantiza)"
        )
    return eigenvalues / n
