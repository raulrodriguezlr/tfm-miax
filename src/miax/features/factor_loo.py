"""Factores sin la propia moneda (leave-one-out): el factor de cada activo excluye a ese activo."""

from __future__ import annotations

from abc import abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from miax.features.factor import (
    FactorModel,
    NotFittedError,
    _validate_returns,
    ols_betas_paired,
)
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.utils.logging import get_logger

logger = get_logger(__name__)


class _LooFactor(FactorModel):
    """Base comun: `factors()` es el factor de mercado unico y `loo_factors()` el de cada moneda."""

    def __init__(self, *, min_valid_share: float = 0.5, min_periods: int = 1000) -> None:
        if not 0 < min_valid_share <= 1:
            raise ValueError(f"min_valid_share debe estar en (0, 1], es {min_valid_share}")
        if min_periods < 2:
            raise ValueError(f"min_periods debe ser >= 2, es {min_periods}")
        self.min_valid_share = min_valid_share
        self.min_periods = min_periods
        self._market: FactorModel = self._make_market()
        self._betas: pd.Series | None = None

    @abstractmethod
    def _make_market(self) -> FactorModel:
        """Modelo estandar (con todas las monedas) que da el nodo de mercado de `factors()`."""

    @abstractmethod
    def _fit_loo(self, returns: pd.DataFrame) -> None:
        """Ajusta lo propio de cada familia (cargas) y las betas contra el factor LOO."""

    @abstractmethod
    def _loo_array(self, returns: pd.DataFrame) -> np.ndarray:
        """Factor LOO (tiempo x moneda) con lo ajustado, solo con retornos del mismo minuto."""

    def _extra_state(self) -> dict[str, object]:
        return {}

    def _fit(self, returns: pd.DataFrame) -> None:
        self._fit_loo(returns)

    def get_state(self) -> dict[str, object]:
        """Parametros, factor de mercado estandar y betas LOO ajustados en train."""
        return {
            "min_valid_share": self.min_valid_share,
            "min_periods": self.min_periods,
            "market": self._market.get_state(),
            "betas": self._betas,
            **self._extra_state(),
        }

    @property
    def betas(self) -> pd.Series:
        """Betas de cada simbolo contra su factor LOO, ajustadas en train y congeladas."""
        self._require_fitted()
        if self._betas is None:
            raise NotFittedError(f"{type(self).__name__} sin ajustar: falta llamar a fit")
        return self._betas.copy()

    def fit(self, returns: pd.DataFrame) -> Self:
        """Ajusta el modelo estandar y el LOO con los mismos datos de train."""
        self._symbols = None
        self._betas = None
        _validate_returns(returns)
        if returns.shape[0] == 0 or returns.shape[1] == 0:
            raise ValueError("returns no puede tener 0 filas ni 0 columnas")
        if returns.shape[1] < 2:
            raise ValueError(f"{type(self).__name__} necesita al menos 2 columnas")
        self._market = self._make_market().fit(returns)
        return super().fit(returns)

    def loo_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Factor de cada moneda sin ella misma (tiempo x moneda), con las columnas de entrada."""
        aligned, _, _ = self._prepare(returns)
        out = pd.DataFrame(
            self._loo_array(aligned), index=aligned.index, columns=aligned.columns, dtype="float64"
        )
        return out[list(returns.columns)]

    def _fit_betas(self, returns: pd.DataFrame, loo: pd.DataFrame) -> None:
        # Betas solo con train y congeladas: `transform` no las recalcula.
        betas = ols_betas_paired(returns, loo, min_periods=self.min_periods)
        logger.info(
            "%s: %d de %d betas NaN; beta media %.3f",
            type(self).__name__,
            int(betas.isna().sum()),
            len(betas),
            float(betas.mean()),
        )
        self._betas = betas

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Nodo de mercado unico (D-34): el mismo factor estandar, con todas las monedas.
        return self._market.factors(returns)

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if self._betas is None:
            raise NotFittedError(f"{type(self).__name__} sin ajustar: falta llamar a fit")
        # Factor LOO del mismo minuto y betas congeladas de train: sin fuga.
        betas = self._betas.reindex(returns.columns).to_numpy()
        out = returns.to_numpy(dtype="float64") - self._loo_array(returns) * betas[None, :]
        return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def _others_matrix(n: int) -> np.ndarray:
    """Matriz N x N con 1 fuera de la diagonal y 0 en ella: suma las demas columnas."""
    return 1.0 - np.eye(n)


class LooMeanFactor(_LooFactor):
    """Factor de i = media de los retornos de las demas monedas; residuo = r_i - beta_i*f_(-i)."""

    def _make_market(self) -> FactorModel:
        return MeanFactor(min_valid_share=self.min_valid_share, min_periods=self.min_periods)

    def _fit_loo(self, returns: pd.DataFrame) -> None:
        loo = pd.DataFrame(
            self._loo_array(returns), index=returns.index, columns=returns.columns, dtype="float64"
        )
        self._fit_betas(returns, loo)

    def _loo_array(self, returns: pd.DataFrame) -> np.ndarray:
        # El factor de t usa solo retornos del mismo minuto de las demas monedas: sin fuga.
        r = returns.to_numpy(dtype="float64")
        n = r.shape[1]
        present = ~np.isnan(r)
        others = _others_matrix(n)
        # El producto por la matriz de "demas" pone un 0 exacto donde iria r_i: f_(-i) no depende
        # de r_i ni en el ultimo bit.
        total = np.where(present, r, 0.0) @ others
        count = present.astype("float64") @ others
        ok = (count / (n - 1) >= self.min_valid_share) & (count > 0)
        f = np.full(r.shape, np.nan)
        np.divide(total, count, out=f, where=ok)
        return f


class LooPcaFactor(_LooFactor):
    """Factor de i = primer componente de las demas monedas; residuo = r_i - beta_i*f_(-i)."""

    def __init__(self, *, min_valid_share: float = 0.5, min_periods: int = 1000) -> None:
        super().__init__(min_valid_share=min_valid_share, min_periods=min_periods)
        self._loadings: pd.DataFrame | None = None

    def _make_market(self) -> FactorModel:
        return PcaFactor(min_valid_share=self.min_valid_share, min_periods=self.min_periods)

    def _extra_state(self) -> dict[str, object]:
        return {"loadings_matrix": self._loadings}

    @property
    def loadings_matrix(self) -> pd.DataFrame:
        """Cargas N x N: la columna i son las cargas del factor de i (NaN en su diagonal)."""
        self._require_fitted()
        if self._loadings is None:
            raise NotFittedError("LooPcaFactor sin ajustar: falta llamar a fit")
        return self._loadings.copy()

    def fit(self, returns: pd.DataFrame) -> Self:
        """Ajusta el modelo estandar y el LOO con los mismos datos de train."""
        self._loadings = None
        return super().fit(returns)

    def _fit_loo(self, returns: pd.DataFrame) -> None:
        # Correlaciones y desviaciones salen solo de train y se congelan: sin fuga.
        counts = returns.notna().sum()
        stds = returns.std()
        # No constante por exactitud (como PcaFactor): std() de una constante != 0 da ~1e-19.
        usable_mask = (counts >= self.min_periods) & (returns.max() != returns.min())
        usable = [str(c) for c in returns.columns[usable_mask.to_numpy()]]
        if len(usable) < 3:
            raise ValueError(
                f"LooPcaFactor necesita al menos 3 simbolos utilizables (>= {self.min_periods} "
                f"retornos y no constante); hay {len(usable)} de {returns.shape[1]}"
            )
        corr = returns[usable].corr(min_periods=self.min_periods).to_numpy(dtype="float64")
        if np.isnan(corr).any():
            i, j = np.argwhere(np.isnan(corr))[0]
            raise ValueError(
                f"correlacion NaN entre {usable[i]} y {usable[j]}: menos de "
                f"{self.min_periods} filas comunes o varianza nula en ellas"
            )
        sigma = stds[usable].to_numpy(dtype="float64")
        loadings = pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
        for k, symbol in enumerate(usable):
            keep = np.arange(len(usable)) != k
            sub = corr[np.ix_(keep, keep)]
            _, eigvecs = np.linalg.eigh(sub)
            vec = eigvecs[:, -1]  # eigh ordena de menor a mayor: el ultimo es el mayor autovalor
            raw = sigma[keep] * vec
            mean = float(raw.mean())
            if not np.isfinite(mean) or mean == 0.0:
                raise ValueError(
                    f"cargas del primer componente sin {symbol} degeneradas: media 0 o no finita"
                )
            # Media 1 fija escala y signo: el factor sube cuando sube el mercado.
            others = [s for s, flag in zip(usable, keep, strict=True) if flag]
            loadings.loc[others, symbol] = raw / mean
        self._loadings = loadings
        loo = pd.DataFrame(
            self._loo_array(returns), index=returns.index, columns=returns.columns, dtype="float64"
        )
        self._fit_betas(returns, loo)

    def _loo_array(self, returns: pd.DataFrame) -> np.ndarray:
        if self._loadings is None:
            raise NotFittedError("LooPcaFactor sin ajustar: falta llamar a fit")
        # El factor de t usa solo retornos del mismo minuto y cargas congeladas de train: sin fuga.
        b = self._loadings.reindex(index=returns.columns, columns=returns.columns).to_numpy(
            dtype="float64"
        )
        r = returns.to_numpy(dtype="float64")
        has_load = ~np.isnan(b)
        b_full = np.where(has_load, b, 0.0)
        present = ~np.isnan(r)
        # La carga de i en su propio factor es 0 exacto: r_i no entra ni en el ultimo bit.
        num = np.where(present, r, 0.0) @ b_full
        den = present.astype("float64") @ (b_full * b_full)
        n_loaded = has_load.sum(axis=0)
        count = present.astype("float64") @ has_load.astype("float64")
        share = np.divide(count, n_loaded, out=np.zeros_like(count), where=n_loaded > 0)
        ok = (share >= self.min_valid_share) & (den > 0) & (n_loaded > 0)
        f = np.full(r.shape, np.nan)
        np.divide(num, den, out=f, where=ok)
        return f
