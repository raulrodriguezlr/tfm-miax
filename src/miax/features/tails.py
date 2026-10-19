"""Colas de los retornos: informe descriptivo y recorte opcional con limites ajustados en train."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def _check_quantile(q: float, name: str) -> None:
    """Lanza `ValueError` si el cuantil no esta en el intervalo abierto (0, 1)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"{name} debe estar en (0, 1): {q}")


def _check_unique_columns(frame: pd.DataFrame, name: str) -> None:
    """Lanza `ValueError` si `frame` tiene columnas duplicadas."""
    if not frame.columns.is_unique:
        dups = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ValueError(f"{name} tiene columnas duplicadas: {dups}")


def _check_finite(returns: pd.DataFrame, name: str) -> None:
    """Lanza `ValueError` nombrando las columnas con `inf` o `-inf` (datos corruptos, D-02)."""
    has_inf = np.isinf(returns.astype("float64")).any()
    if has_inf.any():
        bad = [str(c) for c in returns.columns[has_inf.to_numpy()]]
        raise ValueError(f"{name} contiene infinitos en las columnas: {bad}")


def tail_report(
    returns: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.001, 0.999),
    sigma_levels: Sequence[float] = (5.0, 10.0),
    top_share: float = 0.001,
) -> pd.DataFrame:
    """Describe las colas de cada simbolo (una fila por simbolo), ignorando los `NaN`.

    `max_abs_z`, `std` y los conteos por sigma son descriptivos de la muestra pasada, toda ella:
    NO deben usarse como filtro ni como umbral de recorte (miraria al futuro).
    """
    for q in quantiles:
        _check_quantile(q, "quantiles")
    if any(s <= 0 for s in sigma_levels):
        raise ValueError(f"sigma_levels debe ser positivo: {tuple(sigma_levels)}")
    if not 0.0 < top_share <= 1.0:
        raise ValueError(f"top_share debe estar en (0, 1]: {top_share}")
    _check_unique_columns(returns, "returns")
    _check_finite(returns, "returns")
    rows: dict[str, dict[str, float]] = {}
    for symbol in returns.columns:
        x = returns[symbol].dropna().astype("float64")
        n = len(x)
        std = float(x.std()) if n > 1 else float("nan")
        # Sin varianza (std = 0) la curtosis no esta definida: NaN, no 0.
        kurt = float(x.kurt()) if (n > 1 and std > 0) else float("nan")
        row: dict[str, float] = {"n": n, "std": std, "kurtosis": kurt}
        for q in quantiles:
            row[f"q{q:g}"] = float(x.quantile(q)) if n else float("nan")
        # Con std = 0 o NaN no hay z: se deja NaN en vez de dividir por cero.
        z = (x.abs() / std) if (n > 1 and std > 0) else pd.Series(np.nan, index=x.index)
        row["max_abs_z"] = float(z.max()) if n else float("nan")
        for s in sigma_levels:
            row[f"n_gt_{s:g}sd"] = int((z > s).sum())
        squares = x.to_numpy() ** 2
        total = squares.sum()
        if n and total > 0:
            # El epsilon evita el truncado por coma flotante (100 * 0.29 = 28.999...).
            k = max(1, math.floor(n * top_share + 1e-9))
            row["top_var_share"] = float(np.sort(squares)[-k:].sum() / total)
        else:
            row["top_var_share"] = float("nan")
        rows[str(symbol)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def fit_clip_bounds(
    train: pd.DataFrame, *, lower_q: float = 0.001, upper_q: float = 0.999
) -> pd.DataFrame:
    """Da los cuantiles por simbolo (filas `lower` y `upper`) de lo que recibe."""
    _check_quantile(lower_q, "lower_q")
    _check_quantile(upper_q, "upper_q")
    if lower_q >= upper_q:
        raise ValueError(f"lower_q debe ser menor que upper_q: {lower_q} >= {upper_q}")
    _check_unique_columns(train, "train")
    _check_finite(train, "train")
    # No hay fuga: solo se usa el frame recibido; quien llama pasa unicamente train, nunca test.
    lower = train.quantile(lower_q)
    upper = train.quantile(upper_q)
    return pd.DataFrame({"lower": lower, "upper": upper}).T


def clip_returns(returns: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Copia con cada columna recortada a sus limites; los `NaN` siguen siendo `NaN`.

    Falla si `lower > upper`; una columna con limites `NaN` queda sin recortar (D-02).
    """
    _check_unique_columns(returns, "returns")
    _check_unique_columns(bounds, "bounds")
    if set(returns.columns) != set(bounds.columns):
        raise ValueError("las columnas de `returns` y de `bounds` no coinciden")
    lower = bounds.loc["lower", returns.columns]
    upper = bounds.loc["upper", returns.columns]
    inverted = (lower > upper).to_numpy()  # comparar con NaN da False: no se valida esa columna
    if inverted.any():
        raise ValueError(
            f"`bounds` con lower > upper en: {[str(c) for c in returns.columns[inverted]]}"
        )
    # `clip` conserva los NaN (lo invalido sigue invalido, D-02/D-18): no se rellena nada con 0.
    return returns.clip(lower=lower, upper=upper, axis=1)
