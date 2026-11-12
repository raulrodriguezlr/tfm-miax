"""Factor comun = media transversal de retornos: residuo = r - beta * media, beta OLS de train."""

from __future__ import annotations

import pandas as pd

from miax.features.factor import FactorModel, NotFittedError, ols_betas
from miax.utils.logging import get_logger

logger = get_logger(__name__)


class MeanFactor(FactorModel):
    """Factor = media simple de los retornos de cada minuto; residuo = r - beta*f sin alfa."""

    def __init__(self, *, min_valid_share: float = 0.5, min_periods: int = 1000) -> None:
        if not 0 < min_valid_share <= 1:
            raise ValueError(f"min_valid_share debe estar en (0, 1], es {min_valid_share}")
        if min_periods < 2:
            raise ValueError(f"min_periods debe ser >= 2, es {min_periods}")
        self.min_valid_share = min_valid_share
        self.min_periods = min_periods
        self._betas: pd.Series | None = None

    def get_state(self) -> dict[str, object]:
        """Fraccion minima de validos, minimo de filas y betas ajustadas en train."""
        return {
            "min_valid_share": self.min_valid_share,
            "min_periods": self.min_periods,
            "betas": self._betas,
        }

    def _mean_factor(self, returns: pd.DataFrame) -> pd.Series:
        """Media de la fila sin NaN; NaN si la fraccion de validos queda bajo el minimo."""
        # El factor de t solo usa retornos del mismo minuto con pesos fijos 1/N: sin fuga.
        mean = returns.mean(axis=1, skipna=True)
        share = returns.notna().sum(axis=1) / returns.shape[1]
        return mean.where(share >= self.min_valid_share)

    def _fit(self, returns: pd.DataFrame) -> None:
        factor = self._mean_factor(returns)
        betas = ols_betas(returns, factor, min_periods=self.min_periods)
        n_nan = int(betas.isna().sum())
        logger.info(
            "MeanFactor: %d de %d betas NaN; %.2f%% de filas de train con factor NaN",
            n_nan,
            len(betas),
            100.0 * float(factor.isna().mean()),
        )
        self._betas = betas

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        return self._mean_factor(returns).rename("mean").to_frame()

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if self._betas is None:
            raise NotFittedError("MeanFactor sin ajustar: falta llamar a fit")
        # Solo el factor del mismo minuto y betas congeladas de train: sin fuga.
        betas = self._betas.reindex(returns.columns).to_numpy()
        out = returns.to_numpy() - factors[["mean"]].to_numpy() * betas[None, :]
        return pd.DataFrame(out, index=returns.index, columns=returns.columns)
