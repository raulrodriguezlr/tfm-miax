"""Tests de que la garantia 8 falla si un modelo recalcula algo con test (D-03, MIAX-051)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from miax.features.factor import FactorModel, NotFittedError, ols_betas
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from tests.unit.factor_contract import check_factor_model, make_returns

SYMS = ("AAA", "BBB", "CCC", "DDD")
BTC_SYMS = ("BTCUSDT", "AAA", "BBB", "CCC")
MIN_PERIODS = 10  # el trozo de la garantia 8 tiene 13 filas


def _with_common_factor(
    df: pd.DataFrame, seed: int, scale: np.ndarray, factor_col: str | None = None
) -> pd.DataFrame:
    """Suma a `df` un factor comun F*scale; si hay `factor_col`, esa columna es solo F."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, len(df))
    out = df + f[:, None] * scale
    if factor_col is not None:
        out[factor_col] = pd.Series(f, index=df.index).where(df[factor_col].notna())
    return out


def _data(
    symbols: tuple[str, ...], factor_col: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test sinteticos con un factor comun y NaN dispersos en ambos."""
    scale = np.array([1.0, 0.5, 1.5, 1.0])
    train = make_returns(80, symbols, seed=1, nan_frac=0.05)
    test = make_returns(40, symbols, seed=2, start="2024-01-04 08:00", nan_frac=0.05)
    return (
        _with_common_factor(train, 101, scale, factor_col),
        _with_common_factor(test, 102, scale, factor_col),
    )


def _btc_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _data(BTC_SYMS, factor_col="BTCUSDT")


def _mean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _data(SYMS)


# ---------- modelos tramposos ----------


class _BtcRefitBetasOnTest(BtcFactor):
    """Recalcula las betas con la muestra de test en cada `_residuals`."""

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        betas = ols_betas(returns, returns[self.symbol], min_periods=self.min_periods)
        betas = betas.reindex(returns.columns).to_numpy()
        out = returns.to_numpy() - factors[["btc"]].to_numpy() * betas[None, :]
        res = pd.DataFrame(out, index=returns.index, columns=returns.columns)
        res[self.symbol] = np.nan
        return res


class _MeanRefitBetasOnTest(MeanFactor):
    """Recalcula las betas con la muestra de test en cada `_residuals`."""

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        betas = ols_betas(returns, factors["mean"], min_periods=self.min_periods)
        betas = betas.reindex(returns.columns).to_numpy()
        out = returns.to_numpy() - factors[["mean"]].to_numpy() * betas[None, :]
        return pd.DataFrame(out, index=returns.index, columns=returns.columns)


class _MeanDivideByTestStd(MeanFactor):
    """Divide el residuo por la desviacion tipica de la muestra de test."""

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return super()._residuals(returns, factors) / returns.std()


class _MeanSubtractTestMean(MeanFactor):
    """Resta al residuo la media de la muestra de test."""

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return super()._residuals(returns, factors) - returns.mean()


class _PcaRefitOnTest(PcaFactor):
    """Recalcula correlacion, primer autovector y cargas con test y proyecta con ellas."""

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        corr = returns.corr(min_periods=self.min_periods).to_numpy(dtype="float64")
        _, eigvecs = np.linalg.eigh(corr)
        raw = returns.std().to_numpy(dtype="float64") * eigvecs[:, -1]
        b = (raw / raw.mean())[None, :]
        r = returns.to_numpy(dtype="float64")
        present = ~np.isnan(r)
        b_full = np.where(present, b, 0.0)
        num = (b_full * np.where(present, r, 0.0)).sum(axis=1)
        den = (b_full * b_full).sum(axis=1)
        f = np.full(len(r), np.nan)
        ok = (present.mean(axis=1) >= self.min_valid_share) & (den > 0)
        f[ok] = num[ok] / den[ok]
        return pd.DataFrame({"pca": f}, index=returns.index)


class _MeanLargeMinPeriodsRefit(MeanFactor):
    """Recalcula `ols_betas` con test pero con un `min_periods` mayor que sus filas: da NaN."""

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        betas = ols_betas(returns, factors["mean"], min_periods=1000)
        betas = betas.reindex(returns.columns).to_numpy()
        out = returns.to_numpy() - factors[["mean"]].to_numpy() * betas[None, :]
        return pd.DataFrame(out, index=returns.index, columns=returns.columns)


# ---------- la garantia 8 caza a los tramposos ----------


@pytest.mark.parametrize(
    ("make_model", "data"),
    [
        pytest.param(
            lambda: _BtcRefitBetasOnTest(min_periods=MIN_PERIODS), _btc_data, id="btc-refit-betas"
        ),
        pytest.param(
            lambda: _MeanRefitBetasOnTest(min_periods=MIN_PERIODS),
            _mean_data,
            id="mean-refit-betas",
        ),
        pytest.param(
            lambda: _MeanDivideByTestStd(min_periods=MIN_PERIODS), _mean_data, id="mean-test-std"
        ),
        pytest.param(
            lambda: _MeanSubtractTestMean(min_periods=MIN_PERIODS), _mean_data, id="mean-test-mean"
        ),
        pytest.param(
            lambda: _PcaRefitOnTest(min_periods=MIN_PERIODS), _mean_data, id="pca-refit-on-test"
        ),
    ],
)
def test_cheater_fails_contract(
    make_model: Callable[[], FactorModel],
    data: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Un modelo que recalcula algo con test incumple la garantia 8 y el mensaje cita D-03."""
    train, test = data()
    with pytest.raises(AssertionError, match="D-03"):
        check_factor_model(make_model, train, test)


@pytest.mark.parametrize(
    ("make_model", "data"),
    [
        pytest.param(lambda: BtcFactor(min_periods=MIN_PERIODS), _btc_data, id="btc"),
        pytest.param(lambda: MeanFactor(min_periods=MIN_PERIODS), _mean_data, id="mean"),
        pytest.param(lambda: PcaFactor(min_periods=MIN_PERIODS), _mean_data, id="pca"),
    ],
)
def test_honest_models_pass_contract(
    make_model: Callable[[], FactorModel],
    data: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Control: con los mismos datos, los modelos honestos pasan el contrato entero."""
    train, test = data()
    check_factor_model(make_model, train, test)


def test_known_limit_cheater_with_large_min_periods_passes() -> None:
    """Limite conocido, no garantia: con `min_periods` > filas de test las betas son NaN siempre.

    El tramposo devuelve NaN en el test entero y en el trozo, asi que la garantia 8 no lo ve;
    es un limite del metodo (la trampa no produce residuos validos), no una proteccion.
    """
    train, test = _mean_data()
    check_factor_model(lambda: _MeanLargeMinPeriodsRefit(min_periods=MIN_PERIODS), train, test)


# ---------- betas ----------

_MODELS = [
    pytest.param(lambda: BtcFactor(min_periods=MIN_PERIODS), _btc_data, id="btc"),
    pytest.param(lambda: MeanFactor(min_periods=MIN_PERIODS), _mean_data, id="mean"),
    pytest.param(lambda: PcaFactor(min_periods=MIN_PERIODS), _mean_data, id="pca"),
]


@pytest.mark.parametrize(("make_model", "data"), _MODELS)
def test_betas_not_fitted_raises(
    make_model: Callable[[], FactorModel],
    data: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """`betas` antes de `fit` lanza NotFittedError."""
    with pytest.raises(NotFittedError):
        _ = make_model().betas  # type: ignore[attr-defined]


@pytest.mark.parametrize(("make_model", "data"), _MODELS)
def test_betas_series_float64_indexed_by_symbols(
    make_model: Callable[[], FactorModel],
    data: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Tras `fit`, `betas` es una Series float64 indexada por los simbolos de fit."""
    train, _ = data()
    m = make_model().fit(train)
    betas = m.betas  # type: ignore[attr-defined]
    assert isinstance(betas, pd.Series)
    assert betas.dtype == np.float64
    assert list(betas.index) == list(train.columns)


@pytest.mark.parametrize(("make_model", "data"), _MODELS)
def test_betas_returns_a_copy(
    make_model: Callable[[], FactorModel],
    data: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Mutar la copia devuelta no cambia `model.betas` ni `get_state()`."""
    train, _ = data()
    m = make_model().fit(train)
    before = m.betas.copy()  # type: ignore[attr-defined]
    mutated = m.betas  # type: ignore[attr-defined]
    mutated.iloc[:] = 999.0
    pd.testing.assert_series_equal(m.betas, before)  # type: ignore[attr-defined]
    state = m.get_state()
    stored = state["betas"] if "betas" in state else state["loadings"]
    pd.testing.assert_series_equal(stored, before)  # type: ignore[arg-type]


def test_btc_and_mean_betas_equal_state() -> None:
    """En BtcFactor y MeanFactor, `betas` es igual a `get_state()["betas"]`."""
    for model, data in (
        (BtcFactor(min_periods=MIN_PERIODS), _btc_data()),
        (MeanFactor(min_periods=MIN_PERIODS), _mean_data()),
    ):
        model.fit(data[0])
        pd.testing.assert_series_equal(model.betas, model.get_state()["betas"])  # type: ignore[arg-type]


def test_pca_betas_equal_loadings() -> None:
    """En PcaFactor, `betas` es igual a `loadings`."""
    train, _ = _mean_data()
    m = PcaFactor(min_periods=MIN_PERIODS).fit(train)
    pd.testing.assert_series_equal(m.betas, m.loadings)


@pytest.mark.parametrize(("make_model", "data"), _MODELS)
def test_transform_test_uses_train_betas(
    make_model: Callable[[], FactorModel],
    data: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """`transform(test)` es test - betas * factors(test): los residuos usan las betas de train."""
    train, test = data()
    m = make_model().fit(train)
    betas = m.betas.reindex(test.columns)  # type: ignore[attr-defined]
    factor = m.factors(test).iloc[:, 0]
    expected = test - np.outer(factor.to_numpy(), betas.to_numpy())
    if isinstance(m, BtcFactor):
        expected[m.symbol] = np.nan
    # Los NaN de entrada y del factor de la fila se propagan a la fila entera
    mask = test.isna().to_numpy() | factor.isna().to_numpy()[:, None]
    expected = expected.mask(mask)
    pd.testing.assert_frame_equal(m.transform(test), expected)
