"""Casos borde de la residualizacion con los factores BTC, media y PCA (MIAX-059)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from tests.unit.factor_contract import make_returns

MIN_PERIODS = 10
SYMS = ("BTCUSDT", "AAA", "BBB", "CCC")
SCALE = np.array([1.0, 0.8, 1.2, 1.0])

MODELS = [
    pytest.param(lambda: BtcFactor(min_periods=MIN_PERIODS), id="btc"),
    pytest.param(lambda: MeanFactor(min_periods=MIN_PERIODS), id="mean"),
    pytest.param(lambda: PcaFactor(min_periods=MIN_PERIODS), id="pca"),
]
OLS_MODELS = MODELS[:2]


def _data(n: int, seed: int, start: str) -> pd.DataFrame:
    """Retornos sinteticos con un factor comun, sin NaN."""
    base = make_returns(n, SYMS, seed=seed, start=start)
    f = np.random.default_rng(seed + 100).normal(0.0, 0.01, n)
    return base + f[:, None] * SCALE


def _train() -> pd.DataFrame:
    return _data(120, seed=1, start="2024-01-01")


def _test() -> pd.DataFrame:
    # El test empieza despues de las 120 horas de train: tramos contiguos sin solape.
    return _data(40, seed=2, start="2024-01-06")


# ---------- serie constante ----------


@pytest.mark.parametrize("make_model", OLS_MODELS)
@pytest.mark.parametrize("value", [0.0, 0.001])
def test_constant_coin_ols_models_beta_zero(
    make_model: Callable[[], FactorModel], value: float
) -> None:
    """Con BTC y media, una moneda constante en train tiene beta ~0 y su residuo es su retorno.

    Documenta el comportamiento actual: el PCA, en cambio, la deja fuera con carga NaN.
    """
    train, test = _train(), _test()
    train["AAA"] = value
    test["AAA"] = value
    m = make_model().fit(train)
    assert abs(m.betas["AAA"]) < 1e-12  # type: ignore[attr-defined]
    res = m.transform(test)
    np.testing.assert_allclose(res["AAA"].to_numpy(), value, atol=1e-12)
    assert np.isfinite(res[["BBB", "CCC"]].to_numpy()).all()


def test_constant_btc_gives_all_nan_residuals() -> None:
    """Con BTC constante en train el factor no varia: todas las betas y residuos son NaN."""
    train, test = _train(), _test()
    train["BTCUSDT"] = 0.001
    m = BtcFactor(min_periods=MIN_PERIODS).fit(train)
    assert m.betas.isna().all()
    assert m.transform(test).isna().all().all()


@pytest.mark.parametrize("make_model", OLS_MODELS)
def test_all_constant_ols_models_give_all_nan(make_model: Callable[[], FactorModel]) -> None:
    """Con todas las monedas constantes el factor no varia: betas y residuos NaN, sin excepcion."""
    train = _train()
    for value, col in zip((0.0, 0.001, -0.002, 0.003), SYMS, strict=True):
        train[col] = value
    m = make_model().fit(train)
    assert m.betas.isna().all()  # type: ignore[attr-defined]
    assert m.transform(_test()).isna().all().all()


def test_all_constant_pca_raises() -> None:
    """Con todas las monedas constantes el PCA no tiene 2 simbolos utilizables: ValueError."""
    train = _train()
    train[list(SYMS)] = 0.001
    with pytest.raises(ValueError, match="al menos 2"):
        PcaFactor(min_periods=MIN_PERIODS).fit(train)


# ---------- un solo activo ----------


def test_btc_only_symbol() -> None:
    """Con solo BTC el factor es su retorno y el residuo es todo NaN, sin excepcion."""
    train, test = _train()[["BTCUSDT"]], _test()[["BTCUSDT"]]
    m = BtcFactor(min_periods=MIN_PERIODS).fit(train)
    assert list(m.betas.index) == ["BTCUSDT"]
    np.testing.assert_array_equal(m.factors(test)["btc"].to_numpy(), test["BTCUSDT"].to_numpy())
    assert m.transform(test).isna().all().all()


def test_pca_single_column_raises() -> None:
    """El PCA con una sola columna lanza ValueError: necesita al menos 2 simbolos."""
    with pytest.raises(ValueError, match="al menos 2"):
        PcaFactor(min_periods=MIN_PERIODS).fit(_train()[["AAA"]])


# ---------- activos sin solape temporal ----------


def test_btc_coin_without_overlap_with_btc() -> None:
    """Una moneda con datos solo donde BTC falta: beta y residuo NaN; las demas, intactas."""
    train, test = _train(), _test()
    half = len(train) // 2
    train.iloc[:half, train.columns.get_loc("BTCUSDT")] = np.nan
    train.iloc[half:, train.columns.get_loc("AAA")] = np.nan
    m = BtcFactor(min_periods=MIN_PERIODS).fit(train)
    ref = BtcFactor(min_periods=MIN_PERIODS).fit(train.drop(columns="AAA"))
    assert np.isnan(m.betas["AAA"])
    pd.testing.assert_series_equal(m.betas[["BBB", "CCC"]], ref.betas[["BBB", "CCC"]])
    res = m.transform(test)
    ref_res = ref.transform(test.drop(columns="AAA"))
    assert res["AAA"].isna().all()
    pd.testing.assert_frame_equal(res[["BBB", "CCC"]], ref_res[["BBB", "CCC"]])


def test_mean_coin_alone_in_its_period() -> None:
    """Una moneda sola en su tramo: ahi el factor es NaN, y su beta y su residuo tambien."""
    train, test = _train(), _test()
    half = len(train) // 2
    train.iloc[half:, train.columns.get_loc("AAA")] = np.nan
    train.iloc[:half, [train.columns.get_loc(c) for c in ("BTCUSDT", "BBB", "CCC")]] = np.nan
    m = MeanFactor(min_periods=MIN_PERIODS).fit(train)
    assert np.isnan(m.betas["AAA"])
    assert np.isfinite(m.betas[["BTCUSDT", "BBB", "CCC"]]).all()
    res = m.transform(test)
    assert res["AAA"].isna().all()
    assert np.isfinite(res[["BTCUSDT", "BBB", "CCC"]].to_numpy()).all()


def test_mean_two_disjoint_blocks() -> None:
    """Dos bloques de monedas que nunca coinciden: el factor sigue definido y nada falla."""
    train, test = _train(), _test()
    half = len(train) // 2
    first = [train.columns.get_loc(c) for c in ("BTCUSDT", "AAA")]
    second = [train.columns.get_loc(c) for c in ("BBB", "CCC")]
    train.iloc[half:, first] = np.nan
    train.iloc[:half, second] = np.nan
    m = MeanFactor(min_periods=MIN_PERIODS).fit(train)
    assert np.isfinite(m.betas).all()
    assert np.isfinite(m.transform(test).to_numpy()).all()


# ---------- bordes del tramo de test ----------


@pytest.mark.parametrize("make_model", MODELS)
def test_coin_empty_in_test(make_model: Callable[[], FactorModel]) -> None:
    """Una moneda sin ningun dato en test (caso FTT 2022T4): su residuo es NaN y el resto sigue."""
    test = _test()
    test["BBB"] = np.nan
    m = make_model().fit(_train())
    res = m.transform(test)
    assert res["BBB"].isna().all()
    assert np.isfinite(res[["AAA", "CCC"]].to_numpy()).all()


def test_btc_coin_empty_in_test_leaves_others_unchanged() -> None:
    """Con BTC el residuo de cada moneda no depende de las demas: vaciar una no cambia el resto."""
    m = BtcFactor(min_periods=MIN_PERIODS).fit(_train())
    test = _test()
    full = m.transform(test)
    test["BBB"] = np.nan
    pd.testing.assert_frame_equal(m.transform(test).drop(columns="BBB"), full.drop(columns="BBB"))


@pytest.mark.parametrize("make_model", MODELS)
def test_single_row_test(make_model: Callable[[], FactorModel]) -> None:
    """Un test de una sola fila da lo mismo que esa fila dentro del test completo."""
    test = _test()
    m = make_model().fit(_train())
    full = m.transform(test)
    for i in (0, len(test) // 2, len(test) - 1):
        pd.testing.assert_frame_equal(m.transform(test.iloc[[i]]), full.iloc[[i]])


@pytest.mark.parametrize("make_model", MODELS)
def test_all_nan_test(make_model: Callable[[], FactorModel]) -> None:
    """Un test todo NaN da factor y residuos todo NaN, sin excepcion y nunca 0."""
    test = _test() * np.nan
    m = make_model().fit(_train())
    assert m.factors(test).isna().all().all()
    assert m.transform(test).isna().all().all()
