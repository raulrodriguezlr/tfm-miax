"""Factor comun = primer componente principal de la correlacion de train: residuo = r - b*f."""

from __future__ import annotations

import numpy as np
import pandas as pd

from miax.features.factor import FactorModel, NotFittedError
from miax.utils.logging import get_logger

logger = get_logger(__name__)


class PcaFactor(FactorModel):
    """Factor = primer componente principal de la correlacion de train; residuo = r - b*f."""

    def __init__(self, *, min_valid_share: float = 0.5, min_periods: int = 1000) -> None:
        if not 0 < min_valid_share <= 1:
            raise ValueError(f"min_valid_share debe estar en (0, 1], es {min_valid_share}")
        if min_periods < 2:
            raise ValueError(f"min_periods debe ser >= 2, es {min_periods}")
        self.min_valid_share = min_valid_share
        self.min_periods = min_periods
        self._loadings: pd.Series | None = None
        self._explained_share: float | None = None

    def get_state(self) -> dict[str, object]:
        """Fraccion minima de validos, minimo de filas, cargas y varianza explicada de train."""
        return {
            "min_valid_share": self.min_valid_share,
            "min_periods": self.min_periods,
            "loadings": self._loadings,
            "explained_share": self._explained_share,
        }

    @property
    def loadings(self) -> pd.Series:
        """Cargas del primer componente en unidades de retorno (media 1); hacen de betas en test."""
        self._require_fitted()
        if self._loadings is None:
            raise NotFittedError("PcaFactor sin ajustar: falta llamar a fit")
        return self._loadings.copy()

    @property
    def explained_share(self) -> float:
        """Fraccion de la correlacion explicada por el primer componente en train."""
        self._require_fitted()
        if self._explained_share is None:
            raise NotFittedError("PcaFactor sin ajustar: falta llamar a fit")
        return self._explained_share

    def _fit(self, returns: pd.DataFrame) -> None:
        # Correlaciones y desviaciones salen solo de train y se congelan: sin fuga.
        counts = returns.notna().sum()
        stds = returns.std()
        # No constante por exactitud (como ols_betas): std() de una constante != 0 da ~1e-19.
        usable_mask = (counts >= self.min_periods) & (returns.max() != returns.min())
        usable = [str(c) for c in returns.columns[usable_mask.to_numpy()]]
        if len(usable) < 2:
            raise ValueError(
                f"PcaFactor necesita al menos 2 simbolos utilizables (>= {self.min_periods} "
                f"retornos y no constante); hay {len(usable)} de {returns.shape[1]}"
            )
        # Correlacion y no covarianza: una moneda muy volatil (FTT en 2022T4) arrastraria el
        # componente. Por pares: en 2023T3 solo el 12,8 % de los minutos de train estan completos.
        corr = returns[usable].corr(min_periods=self.min_periods)
        corr_values = corr.to_numpy(dtype="float64")
        if np.isnan(corr_values).any():
            i, j = np.argwhere(np.isnan(corr_values))[0]
            raise ValueError(
                f"correlacion NaN entre {usable[i]} y {usable[j]}: menos de "
                f"{self.min_periods} filas comunes o varianza nula en ellas"
            )
        eigvals, eigvecs = np.linalg.eigh(corr_values)
        vec = eigvecs[:, -1]  # eigh ordena de menor a mayor: el ultimo es el mayor autovalor
        explained = float(eigvals[-1] / eigvals.sum())
        raw = stds[usable].to_numpy(dtype="float64") * vec
        mean = float(raw.mean())
        if not np.isfinite(mean) or mean == 0.0:
            raise ValueError("cargas del primer componente degeneradas: media 0 o no finita")
        # Media 1 fija escala y signo: el factor sube cuando sube el mercado.
        loadings = pd.Series(np.nan, index=returns.columns, dtype="float64")
        loadings[usable] = raw / mean
        self._loadings = loadings
        self._explained_share = explained
        valid = loadings.dropna()
        logger.info(
            "PcaFactor: %d de %d cargas NaN; primer componente explica %.2f%%; "
            "carga min %.3f (%s), max %.3f (%s)",
            int(loadings.isna().sum()),
            len(loadings),
            100.0 * explained,
            float(valid.min()),
            valid.idxmin(),
            float(valid.max()),
            valid.idxmax(),
        )

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        if self._loadings is None:
            raise NotFittedError("PcaFactor sin ajustar: falta llamar a fit")
        # El factor de t solo usa retornos del mismo minuto y cargas congeladas de train: sin fuga.
        b = self._loadings.reindex(returns.columns).to_numpy(dtype="float64")[None, :]
        r = returns.to_numpy(dtype="float64")
        has_load = ~np.isnan(b)
        present = ~np.isnan(r) & has_load
        b_full = np.where(present, b, 0.0)
        num = (b_full * np.where(present, r, 0.0)).sum(axis=1)
        den = (b_full * b_full).sum(axis=1)
        n_loaded = int(has_load.sum())
        share = present.sum(axis=1) / n_loaded if n_loaded else np.zeros(len(r))
        ok = (share >= self.min_valid_share) & (den > 0)
        f = np.full(len(r), np.nan)
        f[ok] = num[ok] / den[ok]
        return pd.DataFrame({"pca": f}, index=returns.index)

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if self._loadings is None:
            raise NotFittedError("PcaFactor sin ajustar: falta llamar a fit")
        # Solo el factor del mismo minuto y cargas congeladas de train: sin fuga.
        b = self._loadings.reindex(returns.columns).to_numpy(dtype="float64")
        out = returns.to_numpy() - factors[["pca"]].to_numpy() * b[None, :]
        return pd.DataFrame(out, index=returns.index, columns=returns.columns)
