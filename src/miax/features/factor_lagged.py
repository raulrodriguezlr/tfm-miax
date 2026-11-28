"""Factor con retardos: ademas del factor del mismo minuto, quita el de los L minutos anteriores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from miax.features.factor import FactorModel, NotFittedError
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BAR = pd.Timedelta(minutes=1)


def lag_matrix(
    factor: pd.Series, index: pd.DatetimeIndex, *, lags: int, bar: pd.Timedelta
) -> np.ndarray:
    """Matriz T x L con `f(t - l*bar)` para l=1..L, tomado por tiempo; NaN si falta ese instante."""
    # Solo se leen instantes estrictamente anteriores a t (l >= 1); si falta uno queda NaN y
    # nunca se usa otro minuto: sin fuga ni retardos desalineados.
    columns = [
        factor.reindex(index - step * bar).to_numpy(dtype="float64") for step in range(1, lags + 1)
    ]
    return np.column_stack(columns)


class LaggedFactor(FactorModel):
    """Envuelve un modelo de factor y quita tambien `sum_l gamma_l * f(t - l*bar)` del residuo."""

    def __init__(
        self,
        base: FactorModel,
        *,
        lags: int = 5,
        bar: pd.Timedelta = DEFAULT_BAR,
        min_periods: int = 1000,
    ) -> None:
        if lags < 1:
            raise ValueError(f"lags debe ser >= 1, es {lags}")
        if bar <= pd.Timedelta(0):
            raise ValueError(f"bar debe ser positivo, es {bar}")
        if min_periods < 2:
            raise ValueError(f"min_periods debe ser >= 2, es {min_periods}")
        if base.is_fitted:
            raise ValueError("base ya esta ajustado: se exige un modelo sin ajustar")
        self._base = base
        self.lags = lags
        self.bar = bar
        self.min_periods = min_periods
        self._gammas: pd.DataFrame | None = None

    @property
    def base(self) -> FactorModel:
        """Modelo de factor envuelto (se ajusta dentro de `fit`)."""
        return self._base

    @property
    def gammas(self) -> pd.DataFrame:
        """Coeficientes de retardo (moneda x lag_1..lag_L) ajustados en train y congelados."""
        self._require_fitted()
        if self._gammas is None:
            raise NotFittedError("LaggedFactor sin ajustar: falta llamar a fit")
        return self._gammas.copy()

    @property
    def betas(self) -> pd.Series:
        """Betas contemporaneas del modelo base (identicas a las suyas)."""
        betas = getattr(self._base, "betas", None)
        if betas is None:
            raise AttributeError("el modelo base no tiene betas")
        return betas

    @property
    def symbol(self) -> str:
        """Simbolo del factor del modelo base, si lo tiene."""
        symbol = getattr(self._base, "symbol", None)
        if symbol is None:
            raise AttributeError("el modelo base no tiene symbol")
        return symbol

    def get_state(self) -> dict[str, object]:
        """Retardos, barra, minimo de filas, gammas y estado del modelo base."""
        return {
            "lags": self.lags,
            "bar_ns": int(self.bar.value),
            "min_periods": self.min_periods,
            "gammas": self._gammas,
            "base": self._base.get_state() if self._base.is_fitted else None,
        }

    def _fit(self, returns: pd.DataFrame) -> None:
        self._base.fit(returns)
        residuals = self._base.transform(returns)
        factors = self._base.factors(returns)
        if factors.shape[1] < 1:
            raise ValueError("el modelo base no tiene factor: no hay nada que retardar")
        factor = factors.iloc[:, 0]
        # Los retardos de train son factor pasado (t-l); los gammas se congelan para test (D-03).
        lagged = lag_matrix(factor, returns.index, lags=self.lags, bar=self.bar)
        e = residuals.to_numpy(dtype="float64")
        gammas = np.full((returns.shape[1], self.lags), np.nan)
        lag_ok = ~np.isnan(lagged).any(axis=1)
        for i in range(returns.shape[1]):
            rows = lag_ok & ~np.isnan(e[:, i])
            if rows.sum() < self.min_periods:
                continue
            design = np.column_stack([np.ones(int(rows.sum())), lagged[rows]])
            coef, _, rank, _ = np.linalg.lstsq(design, e[rows, i], rcond=None)
            if rank < design.shape[1]:
                continue
            gammas[i] = coef[1:]  # la constante no se guarda ni se resta (como D-34)
        names = [f"lag_{k}" for k in range(1, self.lags + 1)]
        table = pd.DataFrame(gammas, index=pd.Index(returns.columns, dtype=object), columns=names)
        no_gamma = [str(s) for s in table.index[table.isna().any(axis=1)]]
        logger.info(
            "LaggedFactor: %d de %d monedas sin gamma; mediana por retardo: %s",
            len(no_gamma),
            len(table),
            ", ".join(f"{c}={float(table[c].median()):.4g}" for c in names),
        )
        self._gammas = table

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        return self._base.factors(returns)

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if self._gammas is None:
            raise NotFittedError("LaggedFactor sin ajustar: falta llamar a fit")
        base_residuals = self._base.transform(returns)
        # Retardos por tiempo (t - l*bar) con gammas de train: solo pasado, sin fuga. Los primeros
        # `lags` minutos de la muestra no tienen pasado en la entrada y su residuo queda NaN.
        lagged = lag_matrix(factors.iloc[:, 0], returns.index, lags=self.lags, bar=self.bar)
        gammas = self._gammas.reindex(returns.columns).to_numpy(dtype="float64")
        out = base_residuals.to_numpy(dtype="float64") - lagged @ gammas.T
        return pd.DataFrame(out, index=returns.index, columns=returns.columns)
