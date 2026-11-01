"""Interfaz comun de los modelos de factor: ajustar en train, proyectar en test."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd


class NotFittedError(RuntimeError):
    """Se uso un modelo de factor antes de llamar a `fit`."""


def _same(a: object, b: object) -> bool:
    """Igualdad robusta para estados (NaN iguales); `TypeError` si el tipo no esta soportado."""
    if isinstance(a, pd.DataFrame | pd.Series):
        return type(a) is type(b) and a.equals(b)  # type: ignore[arg-type]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
        except TypeError:  # dtypes sin NaN (objeto, texto): igualdad normal
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(b)  # type: ignore[arg-type]
            and all(_same(x, y) for x, y in zip(a, b, strict=True))  # type: ignore[arg-type]
        )
    if a is None or isinstance(a, bool | int | float | complex | str | bytes | np.generic):
        both_nan = (
            isinstance(a, float | np.floating)
            and isinstance(b, float | np.floating)
            and bool(np.isnan(a))
            and bool(np.isnan(b))
        )
        return both_nan or bool(a == b)
    raise TypeError(f"tipo no soportado en el estado del modelo: {type(a).__name__}")


def _validate_returns(returns: pd.DataFrame) -> None:
    """Comprueba la entrada (tiempo x simbolo, indice UTC creciente, float64, sin inf)."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError(f"returns debe ser un DataFrame, no {type(returns).__name__}")  # noqa: TRY004
    problems: list[str] = []
    index = returns.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        problems.append(f"el indice debe ser DatetimeIndex en UTC, es {index.dtype}")
    elif not index.is_unique:
        problems.append("el indice tiene duplicados")
    elif not index.is_monotonic_increasing:
        problems.append("el indice no es creciente")
    if not returns.columns.is_unique:
        problems.append("hay simbolos (columnas) duplicados")
    non_str = [c for c in returns.columns if not isinstance(c, str)]
    if non_str:
        problems.append(f"los nombres de columna deben ser str: {non_str}")
    bad = [str(c) for c, dtype in returns.dtypes.items() if dtype != np.float64]
    if bad:
        problems.append(f"columnas que no son float64: {bad}")
    elif np.isinf(returns.to_numpy(dtype="float64")).any():
        problems.append("hay valores infinitos")
    if problems:
        raise ValueError(
            "returns no cumple la interfaz de FactorModel:\n- " + "\n- ".join(problems)
        )


class FactorModel(ABC):
    """Modelo de factor comun: `fit` solo con train, `transform` y `factors` sin reajustar nada."""

    _symbols: tuple[str, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        """Indica si ya se llamo a `fit`."""
        return self._symbols is not None

    @property
    def symbols(self) -> tuple[str, ...]:
        """Simbolos (columnas) vistos en `fit`, en su orden."""
        return self._require_fitted()

    @abstractmethod
    def get_state(self) -> dict[str, object]:
        """Parametros ajustados (betas, cargas...).

        Solo admite ndarray, DataFrame/Series, dict, list/tuple y escalares.
        """

    def fit(self, returns: pd.DataFrame) -> Self:
        """Ajusta el modelo; llamar solo con datos de train."""
        self._symbols = None  # un reajuste fallido no deja el modelo como ajustado
        _validate_returns(returns)
        if returns.shape[0] == 0 or returns.shape[1] == 0:
            raise ValueError("returns no puede tener 0 filas ni 0 columnas")
        self._fit(returns.copy())
        self._symbols = tuple(str(c) for c in returns.columns)
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Devuelve los residuos con lo ajustado en `fit`, sin reajustar nada."""
        aligned, factors, state = self._prepare(returns)
        residuals = self._residuals(aligned, factors)
        self._check_output(residuals, aligned, "_residuals")
        # El NaN del retorno y del factor de la fila se propaga: lo que falta nunca pasa a ser 0.
        mask = aligned.isna().to_numpy() | factors.isna().any(axis=1).to_numpy()[:, None]
        out = residuals.astype("float64").mask(mask)
        if np.isinf(out.to_numpy()).any():
            raise ValueError("_residuals devolvio valores infinitos")
        self._check_state(state)
        return out[list(returns.columns)]

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y devuelve los residuos de la misma muestra (solo train)."""
        return self.fit(returns).transform(returns)

    def factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Devuelve la(s) serie(s) de factor (tiempo x k) con lo ajustado en `fit`."""
        _, factors, _ = self._prepare(returns)
        return factors

    def _require_fitted(self) -> tuple[str, ...]:
        if self._symbols is None:
            raise NotFittedError(f"{type(self).__name__} no esta ajustado: llamar antes a fit")
        return self._symbols

    def _prepare(
        self, returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        """Valida, reordena a los simbolos de `fit` y calcula los factores comprobando el estado."""
        fitted = self._require_fitted()
        _validate_returns(returns)
        seen = tuple(str(c) for c in returns.columns)
        if set(seen) != set(fitted):
            missing = sorted(set(fitted) - set(seen))
            extra = sorted(set(seen) - set(fitted))
            raise ValueError(f"simbolos distintos a los de fit: faltan {missing}, sobran {extra}")
        aligned = returns[list(fitted)].copy()
        # Instantanea del estado ajustado: si transform lo cambia, hay fuga (D-03).
        state = copy.deepcopy(self.get_state())
        factors = self._factors(aligned)
        if not isinstance(factors, pd.DataFrame) or not factors.index.equals(aligned.index):
            raise ValueError("_factors debe devolver un DataFrame con el indice de la entrada")
        factors = factors.astype("float64")
        if np.isinf(factors.to_numpy()).any():
            raise ValueError("_factors devolvio valores infinitos")
        self._check_state(state)
        return aligned, factors, state

    def _check_output(self, out: pd.DataFrame, ref: pd.DataFrame, name: str) -> None:
        if (
            not isinstance(out, pd.DataFrame)
            or not out.index.equals(ref.index)
            or list(out.columns) != list(ref.columns)
        ):
            raise ValueError(
                f"{name} debe devolver un DataFrame con el indice y columnas de la entrada"
            )

    def _check_state(self, before: dict[str, object]) -> None:
        if not _same(before, self.get_state()):
            raise RuntimeError(
                f"{type(self).__name__} cambio su estado ajustado al proyectar: fuga de datos"
            )

    @abstractmethod
    def _fit(self, returns: pd.DataFrame) -> None:
        """Ajusta los parametros del factor con `returns` (train)."""

    @abstractmethod
    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Serie(s) de factor tiempo x k con el indice de `returns`, usando solo lo ajustado."""

    @abstractmethod
    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """Residuos con el indice y las columnas de `returns`, usando solo lo ajustado."""


class NoFactor(FactorModel):
    """Referencia sin residualizar: factor vacio y residuo igual al retorno."""

    def get_state(self) -> dict[str, object]:
        """Sin parametros ajustados."""
        return {}

    def _fit(self, returns: pd.DataFrame) -> None:
        return None

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=returns.index, columns=pd.Index([]), dtype="float64")

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return returns.copy()
