"""Metricas descriptivas de cuanto factor comun quita la residualizacion (MIAX-054)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from miax.features.correlation import (
    explained_variance,
    off_diagonal,
    raw_correlation,
    symbols_without_data,
)
from miax.utils.logging import get_logger

logger = get_logger(__name__)


def correlation_drop(
    returns: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    min_periods: int = 1000,
    max_nan_share: float = 0.05,
) -> dict[str, float | int | str]:
    """Cuanto cae la correlacion entre activos al pasar de `returns` a sus `residuals`.

    Compara los mismos minutos y pares; `correlation_drop` es la clave principal y es descriptiva de
    la muestra completa: no es una feature ni un filtro.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(residuals, pd.DataFrame):
        raise ValueError("returns y residuals deben ser DataFrame")  # noqa: TRY004
    # Se valida aqui: mas abajo el ValueError de `explained_variance` se convierte en NaN.
    if not 0.0 <= max_nan_share < 1.0:
        raise ValueError(f"max_nan_share debe estar en [0, 1): {max_nan_share}")
    if not returns.index.equals(residuals.index):
        raise ValueError("returns y residuals deben tener el mismo indice")
    if not returns.columns.is_unique or not residuals.columns.is_unique:
        raise ValueError("returns o residuals tienen columnas duplicadas")
    if set(returns.columns) != set(residuals.columns):
        raise ValueError("returns y residuals deben tener las mismas columnas")
    non_str = [c for c in returns.columns if not isinstance(c, str)]
    if non_str:
        raise ValueError(f"los nombres de columna deben ser str: {non_str}")
    residuals = residuals[list(returns.columns)]

    # Misma muestra en los dos sentidos: donde uno es NaN el otro tampoco cuenta (minuto sin factor,
    # BTC con BtcFactor). Nada de rellenar con 0 (D-02).
    raw = returns.where(residuals.notna())
    residuals = residuals.where(returns.notna())
    # Solo pares del mismo instante: la metrica no mira otras filas, no hay fuga.
    corr_raw = raw_correlation(raw, min_periods=min_periods)
    corr_res = raw_correlation(residuals, min_periods=min_periods)

    apart = set(symbols_without_data(corr_raw)) | set(symbols_without_data(corr_res))
    without = [str(c) for c in returns.columns if c in apart]
    if len(returns.columns) - len(without) < 2:
        raise ValueError(f"menos de 2 simbolos con datos; sin datos: {without}")
    if without:
        logger.warning(
            "%d simbolos sin datos, fuera de la metrica: %s", len(without), ", ".join(without)
        )
        corr_raw = corr_raw.drop(index=without, columns=without)
        corr_res = corr_res.drop(index=without, columns=without)

    pairs_raw = off_diagonal(corr_raw)
    pairs_res = off_diagonal(corr_res)
    both = ~np.isnan(pairs_raw) & ~np.isnan(pairs_res)
    if not both.any():
        raise ValueError("no hay ningun par con valor en la correlacion cruda y la residual")
    pairs_raw = pairs_raw[both]
    pairs_res = pairs_res[both]

    # Los autovalores se calculan sobre los mismos pares que la media: fuera de `both` va NaN
    # (la diagonal se conserva) y asi ambos usan el mismo conjunto y el mismo tipo de relleno.
    rows, cols = np.triu_indices(len(corr_raw), k=1)
    keep = np.zeros((len(corr_raw), len(corr_raw)), dtype=bool)
    keep[rows[both], cols[both]] = True
    keep = keep | keep.T | np.eye(len(corr_raw), dtype=bool)
    corr_raw = corr_raw.where(keep)
    corr_res = corr_res.where(keep)

    raw_abs = float(np.abs(pairs_raw).mean())
    res_abs = float(np.abs(pairs_res).mean())
    if raw_abs == 0.0:
        raise ValueError("la correlacion cruda media es 0: no hay correlacion que quitar")
    # Valor absoluto: con factor media o PCA la media con signo residual cae a -1/(N-1) por
    # construccion (D-35, D-36) y daria una caida enganosa.
    drop = 1.0 - res_abs / raw_abs

    # Si faltan demasiados pares para el primer autovalor el campo queda en NaN: la metrica
    # principal no depende de el.
    shares: dict[str, float] = {}
    for name, matrix in (("raw", corr_raw), ("residual", corr_res)):
        try:
            # Solo se usa el primer autovalor, bien definido aunque el minimo salga negativo.
            shares[name] = float(
                explained_variance(matrix, max_nan_share=max_nan_share, require_psd=False)[0]
            )
        except ValueError as error:
            logger.warning("Sin primer autovalor %s: %s", name, error)
            shares[name] = float("nan")

    return {
        "n_symbols": int(returns.shape[1]),
        "n_symbols_without_data": len(without),
        "symbols_without_data": ";".join(without),
        "n_pairs": int(both.sum()),
        "raw_pair_mean": float(pairs_raw.mean()),
        "residual_pair_mean": float(pairs_res.mean()),
        "raw_pair_abs_mean": raw_abs,
        "residual_pair_abs_mean": res_abs,
        "correlation_drop": float(drop),
        "raw_first_eigenvalue_share": shares["raw"],
        "residual_first_eigenvalue_share": shares["residual"],
    }
