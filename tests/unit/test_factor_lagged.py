"""Tests del factor con retardos (MIAX-216): contrato, fuga, retardo por tiempo y gammas."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import miax.features.factor_lagged as fl
from miax.features import LaggedFactor as ExportedLaggedFactor
from miax.features.factor import FactorModel, NoFactor, NotFittedError, _same
from miax.features.factor_btc import BtcFactor
from miax.features.factor_lagged import DEFAULT_BAR, LaggedFactor, lag_matrix
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from tests.unit.factor_contract import check_factor_model, make_returns
from tests.unit.leakage import assert_no_lookahead

MIN_PERIODS = 10
LAGS = 3
HOUR = pd.Timedelta(hours=1)
BTC = "BTCUSDT"
COINS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
BETAS = np.array([0.5, 1.0, 1.5, 0.8])


def _index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="min", tz="UTC")


def _lag_panel(
    n: int,
    *,
    gamma: np.ndarray | None = None,
    seed: int = 0,
    start: str = "2024-01-01",
    drift: float = 0.0,
) -> pd.DataFrame:
    """Minutos con BTC de factor y `r_i = b_i f(t) + g_i f(t-1) + drift + ruido` por moneda."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, n)
    f_prev = np.concatenate([[0.0], f[:-1]])
    g = np.zeros(len(COINS)) if gamma is None else gamma
    data = {BTC: f}
    for k, symbol in enumerate(COINS):
        noise = rng.normal(0.0, 0.002, n)
        data[symbol] = BETAS[k] * f + g[k] * f_prev + drift + noise
    return pd.DataFrame(data, index=_index(n, start))


# ---------- contrato y fuga ----------


def _hourly_btc_base() -> FactorModel:
    return BtcFactor("AAA", min_periods=MIN_PERIODS)


_HOURLY_BASES: dict[str, Callable[[], FactorModel]] = {
    "btc": _hourly_btc_base,
    "mean": lambda: MeanFactor(min_periods=MIN_PERIODS),
    "pca": lambda: PcaFactor(min_periods=MIN_PERIODS),
}


def _hourly_lagged(kind: str) -> FactorModel:
    return LaggedFactor(_HOURLY_BASES[kind](), lags=2, bar=HOUR, min_periods=MIN_PERIODS)


@pytest.mark.parametrize("kind", list(_HOURLY_BASES))
def test_contract_lagged_factor(kind: str) -> None:
    """Cumple el contrato de `FactorModel` con `lookback` igual a los retardos."""
    train = make_returns(80, seed=1, nan_frac=0.05)
    test = make_returns(40, seed=2, start="2024-01-04 08:00", nan_frac=0.05)
    check_factor_model(lambda: _hourly_lagged(kind), train, test, lookback=2)


def test_contract_lookback_zero_still_rejects_lagged_factor() -> None:
    """Sin `lookback` el punto 8 del contrato sigue fallando con un modelo que mira atras."""
    train = make_returns(80, seed=1, nan_frac=0.05)
    test = make_returns(40, seed=2, start="2024-01-04 08:00", nan_frac=0.05)
    with pytest.raises(AssertionError):
        check_factor_model(lambda: _hourly_lagged("mean"), train, test)


def _factor_data(kind: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Panel horario de 120 filas con factor comun, NaN dispersos y el inicio de test (fila 80)."""
    n = 120
    base = make_returns(n, ("AAA", "BBB", "CCC", "DDD"), seed=7, nan_frac=0.02)
    factor = np.random.default_rng(1007).normal(0.0, 0.01, n)
    data = base + factor[:, None] * np.linspace(0.5, 1.5, 4)
    if kind == "btc":
        data["AAA"] = pd.Series(factor, index=base.index).where(base["AAA"].notna())
    return data, data.index[80]


@pytest.mark.parametrize("move_nans", [False, True])
@pytest.mark.parametrize("kind", list(_HOURLY_BASES))
@pytest.mark.parametrize("cut_pos", [3, 20, 39])  # con 2 retardos, 3 deja una fila no NaN
def test_lagged_transform_and_factors_ignore_the_future(
    kind: str, cut_pos: int, move_nans: bool
) -> None:
    """Con fit(train), residuos y factor de las filas < corte no cambian al perturbar >= corte."""
    data, test_start = _factor_data(kind)
    cut = data.index[80 + cut_pos]

    def residuals(d: pd.DataFrame) -> pd.DataFrame:
        model = _hourly_lagged(kind).fit(d[d.index < test_start])
        return model.transform(d[d.index >= test_start])

    def factors(d: pd.DataFrame) -> pd.DataFrame:
        model = _hourly_lagged(kind).fit(d[d.index < test_start])
        return model.factors(d[d.index >= test_start])

    assert_no_lookahead(residuals, data, cut, move_nans=move_nans, what=f"{kind}: residuos")
    assert_no_lookahead(factors, data, cut, move_nans=move_nans, what=f"{kind}: factors")


@pytest.mark.parametrize("kind", list(_HOURLY_BASES))
def test_lagged_fit_state_ignores_test(kind: str) -> None:
    """El estado ajustado con fit(train) (gammas y base) no cambia al perturbar todo el test."""
    data, test_start = _factor_data(kind)

    def state(d: pd.DataFrame) -> object:
        return _hourly_lagged(kind).fit(d[d.index < test_start]).get_state()

    for pos in (0, 20, 39):
        assert_no_lookahead(state, data, data.index[80 + pos], what=f"{kind}: estado")


def test_leak_control_future_factor_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: si los retardos miraran f(t+l), `assert_no_lookahead` salta."""
    data, test_start = _factor_data("mean")
    real = fl.lag_matrix

    def cheating(
        factor: pd.Series, index: pd.DatetimeIndex, *, lags: int, bar: pd.Timedelta
    ) -> np.ndarray:
        return real(factor, index, lags=lags, bar=-bar)

    def residuals(d: pd.DataFrame) -> pd.DataFrame:
        model = _hourly_lagged("mean").fit(d[d.index < test_start])
        return model.transform(d[d.index >= test_start])

    assert_no_lookahead(residuals, data, data.index[100], what="control honesto")
    monkeypatch.setattr(fl, "lag_matrix", cheating)
    with pytest.raises(AssertionError, match="fuga de futuro"):
        assert_no_lookahead(residuals, data, data.index[100], what="control tramposo")


# ---------- retardo por tiempo ----------


def test_lag_matrix_by_hand_with_a_gap() -> None:
    """Columna l = f(t - l*bar) por tiempo; donde falta ese instante hay NaN, no otro minuto."""
    index = _index(7)
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], index=index).drop(index[3])
    out = lag_matrix(factor, index, lags=2, bar=DEFAULT_BAR)
    nan = np.nan
    expected = np.array(
        [
            [nan, nan],  # t=0: sin pasado
            [1.0, nan],  # t=1
            [2.0, 1.0],  # t=2
            [3.0, 2.0],  # t=3 (su propio factor falta, pero el pasado existe)
            [nan, 3.0],  # t=4: falta f(3)
            [5.0, nan],  # t=5: f(4)=5, falta f(3)
            [6.0, 5.0],  # t=6
        ]
    )
    assert out.shape == (7, 2)
    np.testing.assert_array_equal(out, expected)


def test_lag_matrix_respects_bar_and_a_different_index() -> None:
    """Con barra de 5 minutos el retardo 1 es t-5min, y el indice de salida puede ser otro."""
    index = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    factor = pd.Series(np.arange(10.0), index=index)
    out = lag_matrix(factor, index[3:6], lags=1, bar=pd.Timedelta(minutes=5))
    np.testing.assert_array_equal(out[:, 0], [2.0, 3.0, 4.0])
    # con barra de 1 minuto esos instantes no existen en el factor
    assert np.isnan(lag_matrix(factor, index[3:6], lags=1, bar=DEFAULT_BAR)).all()


def test_missing_minute_makes_the_residual_nan_not_another_minute() -> None:
    """Si falta t-l, el residuo de t es NaN; lejos del hueco coincide con la version sin hueco."""
    train = _lag_panel(4000, gamma=np.array([0.5, 0.3, -0.4, 0.2]), seed=1)
    test = _lag_panel(300, gamma=np.array([0.5, 0.3, -0.4, 0.2]), seed=2, start="2024-02-01")
    model = LaggedFactor(BtcFactor(BTC, min_periods=100), lags=5, min_periods=100).fit(train)
    full = model.transform(test)
    j = 150
    gapped_input = test.drop(test.index[j])
    gapped = model.transform(gapped_input)
    # filas t = j+1 .. j+5 pierden un retardo
    for step in range(1, 6):
        assert gapped.loc[test.index[j + step], list(COINS)].isna().all()
        assert full.loc[test.index[j + step], list(COINS)].notna().all()
    after = test.index[j + 6 :]
    before = test.index[:j]
    # el residuo de BTC es NaN siempre; se compara solo con las monedas
    pd.testing.assert_frame_equal(
        gapped.loc[after, list(COINS)], full.loc[after, list(COINS)], check_freq=False
    )
    pd.testing.assert_frame_equal(
        gapped.loc[before, list(COINS)], full.loc[before, list(COINS)], check_freq=False
    )
    assert test.index[j] not in gapped.index


# ---------- retraso conocido ----------


def _corr_with_lag1(residuals: pd.DataFrame, factor: pd.Series) -> np.ndarray:
    lagged = lag_matrix(factor, residuals.index, lags=1, bar=DEFAULT_BAR)[:, 0]
    out = []
    for symbol in COINS:
        ok = residuals[symbol].notna().to_numpy() & ~np.isnan(lagged)
        out.append(np.corrcoef(residuals[symbol].to_numpy()[ok], lagged[ok])[0, 1])
    return np.array(out)


def test_known_lag_is_recovered_and_the_residual_loses_it() -> None:
    """Con `r = b f(t) + g f(t-1) + ruido` las gammas recuperan g y `corr(e, f(t-1))` cae a ~0."""
    gamma = np.array([0.5, 0.3, -0.4, 0.0])
    train = _lag_panel(20000, gamma=gamma, seed=3)
    test = _lag_panel(10000, gamma=gamma, seed=4, start="2024-03-01")
    lagged = LaggedFactor(BtcFactor(BTC, min_periods=1000), lags=3, min_periods=1000).fit(train)
    base = BtcFactor(BTC, min_periods=1000).fit(train)
    table = lagged.gammas.loc[list(COINS)]
    np.testing.assert_allclose(table["lag_1"].to_numpy(), gamma, atol=0.03)
    assert table[["lag_2", "lag_3"]].abs().to_numpy().max() < 0.03
    factor = test[BTC]
    corr_base = _corr_with_lag1(base.transform(test), factor)
    corr_lagged = _corr_with_lag1(lagged.transform(test), factor)
    assert np.abs(corr_base[:3]).min() > 0.5
    assert np.abs(corr_lagged).max() < 0.03


# ---------- igualdad con el base ----------


def _fit_pair(
    train: pd.DataFrame, *, lags: int = 3, min_periods: int = 1000
) -> tuple[LaggedFactor, BtcFactor]:
    lagged = LaggedFactor(BtcFactor(BTC, min_periods=100), lags=lags, min_periods=min_periods)
    return lagged.fit(train), BtcFactor(BTC, min_periods=100).fit(train)


def test_betas_factors_and_state_match_the_base() -> None:
    """Betas y `factors()` son los del base ajustado con el mismo train; el estado anida el base."""
    gamma = np.array([0.5, 0.3, -0.4, 0.1])
    train = _lag_panel(5000, gamma=gamma, seed=5)
    test = _lag_panel(500, gamma=gamma, seed=6, start="2024-03-01")
    lagged, base = _fit_pair(train)
    pd.testing.assert_series_equal(lagged.betas, base.betas)
    pd.testing.assert_frame_equal(lagged.factors(test), base.factors(test))
    assert lagged.symbol == BTC and lagged.symbols == base.symbols
    state = lagged.get_state()
    assert _same(state["base"], base.get_state())
    assert state["lags"] == 3 and state["bar_ns"] == DEFAULT_BAR.value
    assert state["min_periods"] == 1000
    assert list(lagged.gammas.columns) == ["lag_1", "lag_2", "lag_3"]


def test_residual_is_base_minus_lagged_gamma_by_hand() -> None:
    """e_lag = e_base - sum_l gamma_l f(t-l), calculado a mano con la matriz de retardos."""
    gamma = np.array([0.5, 0.3, -0.4, 0.1])
    train = _lag_panel(5000, gamma=gamma, seed=5)
    test = _lag_panel(500, gamma=gamma, seed=6, start="2024-03-01")
    lagged, base = _fit_pair(train)
    lagged_f = lag_matrix(test[BTC], test.index, lags=3, bar=DEFAULT_BAR)
    expected = (
        base.transform(test).to_numpy()
        - lagged_f @ lagged.gammas.reindex(test.columns).to_numpy().T
    )
    got = lagged.transform(test).to_numpy()
    np.testing.assert_allclose(got[3:, 1:], expected[3:, 1:], atol=1e-15)


def test_first_rows_are_nan_and_the_next_is_not() -> None:
    """Las primeras L filas de la muestra son NaN y la fila L+1 ya tiene residuo."""
    train = _lag_panel(5000, gamma=np.array([0.5, 0.3, -0.4, 0.1]), seed=5)
    test = _lag_panel(100, seed=6, start="2024-03-01")
    lagged, _ = _fit_pair(train, lags=4)
    out = lagged.transform(test)
    assert out.iloc[:4].isna().all().all()
    assert out.iloc[4][list(COINS)].notna().all()


def test_btc_residual_is_nan_with_btc_base() -> None:
    """El residuo del propio BTC es NaN (como en el base) y su gamma tambien."""
    train = _lag_panel(5000, seed=5)
    test = _lag_panel(100, seed=6, start="2024-03-01")
    lagged, _ = _fit_pair(train)
    assert lagged.transform(test)[BTC].isna().all()
    assert lagged.gammas.loc[BTC].isna().all()


def test_coin_with_fewer_than_min_periods_complete_rows_has_nan_gamma_and_residual() -> None:
    """Con menos de `min_periods` filas completas la gamma y el residuo de esa moneda son NaN."""
    train = _lag_panel(3000, gamma=np.array([0.5, 0.3, -0.4, 0.1]), seed=5)
    train.iloc[:2500, train.columns.get_loc("CCCUSDT")] = np.nan  # 500 filas < 1000
    test = _lag_panel(100, seed=6, start="2024-03-01")
    lagged, base = _fit_pair(train)
    assert not np.isnan(base.betas["CCCUSDT"])  # el base si la ajusta (min_periods=100)
    assert lagged.gammas.loc["CCCUSDT"].isna().all()
    out = lagged.transform(test)
    assert out["CCCUSDT"].isna().all()
    assert out[["AAAUSDT", "BBBUSDT", "DDDUSDT"]].iloc[3:].notna().all().all()


def test_ols_constant_is_not_subtracted() -> None:
    """Un nivel constante en los retornos no se resta: el residuo conserva la media."""
    drift = 0.001
    train = _lag_panel(6000, gamma=np.array([0.5, 0.3, -0.4, 0.1]), seed=5, drift=drift)
    test = _lag_panel(5000, gamma=np.array([0.5, 0.3, -0.4, 0.1]), seed=6, drift=drift)
    lagged, _ = _fit_pair(train)
    means = lagged.transform(test)[list(COINS)].mean()
    np.testing.assert_allclose(means.to_numpy(), drift, atol=2e-4)


def test_fit_is_deterministic() -> None:
    """Dos ajustes con el mismo train dan el mismo estado y los mismos residuos."""
    train = _lag_panel(4000, gamma=np.array([0.5, 0.3, -0.4, 0.1]), seed=5)
    test = _lag_panel(300, seed=6, start="2024-03-01")
    a, _ = _fit_pair(train)
    b, _ = _fit_pair(train)
    assert _same(a.get_state(), b.get_state())
    pd.testing.assert_frame_equal(a.transform(test), b.transform(test))


# ---------- validacion ----------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lags": 0}, "lags"),
        ({"lags": -2}, "lags"),
        ({"bar": pd.Timedelta(0)}, "bar"),
        ({"bar": pd.Timedelta(minutes=-1)}, "bar"),
        ({"min_periods": 1}, "min_periods"),
    ],
)
def test_constructor_rejects_bad_parameters(kwargs: dict[str, object], match: str) -> None:
    """Retardos < 1, barra no positiva y `min_periods` < 2 dan `ValueError`."""
    with pytest.raises(ValueError, match=match):
        LaggedFactor(MeanFactor(), **kwargs)  # type: ignore[arg-type]


def test_constructor_rejects_an_already_fitted_base() -> None:
    """Un base ya ajustado da `ValueError`."""
    base = MeanFactor(min_periods=10).fit(make_returns(50, seed=1))
    with pytest.raises(ValueError, match="ya esta ajustado"):
        LaggedFactor(base)


def test_not_fitted_errors() -> None:
    """Antes de `fit`, transform, factors, symbols y gammas dan `NotFittedError`."""
    model = LaggedFactor(MeanFactor(min_periods=10), lags=2, bar=HOUR, min_periods=10)
    data = make_returns(30, seed=1)
    for call in (model.transform, model.factors, lambda _d: model.symbols, lambda _d: model.gammas):
        with pytest.raises(NotFittedError):
            call(data)
    assert model.get_state()["base"] is None
    assert not model.is_fitted


def test_base_without_factor_columns_is_rejected() -> None:
    """Un base sin columnas de factor (`NoFactor`) da `ValueError` y deja el modelo sin ajustar."""
    model = LaggedFactor(NoFactor(), lags=2, bar=HOUR, min_periods=10)
    with pytest.raises(ValueError, match="no tiene factor"):
        model.fit(make_returns(50, seed=1))
    assert not model.is_fitted


def test_betas_and_symbol_raise_when_the_base_lacks_them() -> None:
    """Sobre una media no hay `symbol`; sobre un base sin betas, `betas` da `AttributeError`."""
    data = make_returns(60, seed=1)
    mean = LaggedFactor(MeanFactor(min_periods=10), lags=2, bar=HOUR, min_periods=10).fit(data)
    assert len(mean.betas) == 4
    with pytest.raises(AttributeError):
        mean.symbol  # noqa: B018
    nofactor = LaggedFactor(NoFactor(), lags=2, bar=HOUR, min_periods=10)
    with pytest.raises(AttributeError):
        nofactor.betas  # noqa: B018


def test_export_from_package() -> None:
    """`LaggedFactor` se exporta desde `miax.features`."""
    assert ExportedLaggedFactor is LaggedFactor
