"""Factor comun = retorno de BTC: residuo = r - beta * r_BTC, con beta OLS ajustada en train."""

from __future__ import annotations

import numpy as np
import pandas as pd

from miax.features.factor import FactorModel, NotFittedError, ols_betas
from miax.utils.logging import get_logger

logger = get_logger(__name__)


class BtcFactor(FactorModel):
    """Factor = retorno de `symbol`; residuo = r - beta*f sin alfa, y el de BTC es NaN."""

    def __init__(self, symbol: str = "BTCUSDT", *, min_periods: int = 1000) -> None:
        if min_periods < 2:
            raise ValueError(f"min_periods debe ser >= 2, es {min_periods}")
        self.symbol = symbol
        self.min_periods = min_periods
        self._betas: pd.Series | None = None

    def get_state(self) -> dict[str, object]:
        """Simbolo del factor, minimo de filas y betas ajustadas en train."""
        return {"symbol": self.symbol, "min_periods": self.min_periods, "betas": self._betas}

    def _fit(self, returns: pd.DataFrame) -> None:
        if self.symbol not in returns.columns:
            raise ValueError(f"falta el simbolo del factor {self.symbol!r} en returns")
        betas = ols_betas(returns, returns[self.symbol], min_periods=self.min_periods)
        n_nan = int(betas.drop(self.symbol).isna().sum())
        logger.info("BtcFactor: %d de %d betas NaN (sin BTC)", n_nan, len(betas) - 1)
        self._betas = betas

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns[[self.symbol]].rename(columns={self.symbol: "btc"})

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if self._betas is None:
            raise NotFittedError("BtcFactor sin ajustar: falta llamar a fit")
        # Solo el factor del mismo minuto y betas congeladas de train: sin fuga.
        betas = self._betas.reindex(returns.columns).to_numpy()
        out = returns.to_numpy() - factors[["btc"]].to_numpy() * betas[None, :]
        res = pd.DataFrame(out, index=returns.index, columns=returns.columns)
        res[self.symbol] = np.nan
        return res
