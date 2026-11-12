"""Tests de `MeanFactor`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from miax.features import MeanFactor as ExportedMeanFactor
from miax.features.factor import NotFittedError, ols_betas
from miax.features.factor_mean import MeanFactor
from tests.unit.factor_contract import check_factor_model, make_returns

SYMS = ("AAA", "BBB", "CCC", "DDD")


def _fit_clean(min_valid_share: float = 0.5) -> tuple[MeanFactor, pd.DataFrame]:
    """Modelo ajustado con 200 filas sin NaN y los retornos de train."""
    train = make_returns(200, SYMS, seed=11)
    return MeanFactor(min_valid_share=min_valid_share, min_periods=10).fit(train), train


# ---------- contrato ----------


def test_contract_mean_factor() -> None:
    """Cumple el contrato de FactorModel con NaN dispersos (incluye determinismo)."""
    train = make_returns(80, SYMS, seed=1, nan_frac=0.05)
    test = make_returns(40, SYMS, seed=2, start="2024-01-04 08:00", nan_frac=0.05)
    check_factor_model(lambda: MeanFactor(min_periods=10), train, test)


def test_export_from_package() -> None:
    """MeanFactor se exporta desde miax.features."""
    assert ExportedMeanFactor is MeanFactor


# ---------- factor = media de la fila ----------


def test_factor_is_row_mean_without_nan() -> None:
    """Sin NaN el factor es exactamente la media de la fila, en una sola columna 'mean'."""
    m, _ = _fit_clean()
    test = make_returns(30, SYMS, seed=12, start="2024-02-01")
    fac = m.factors(test)
    assert list(fac.columns) == ["mean"]
    np.testing.assert_array_equal(fac["mean"].to_numpy(), test.mean(axis=1).to_numpy())


def test_factor_is_mean_of_present_values() -> None:
    """Con NaN en algunas celdas el factor es la media de los retornos presentes."""
    m, _ = _fit_clean()
    test = make_returns(30, SYMS, seed=13, start="2024-02-01")
    test.iloc[2, 0] = np.nan
    test.iloc[5, [1, 2]] = np.nan
    fac = m.factors(test)["mean"]
    assert fac.iloc[2] == pytest.approx(test.iloc[2, 1:].mean())
    assert fac.iloc[5] == pytest.approx(test.iloc[5, [0, 3]].mean())
    assert fac.notna().all()


# ---------- umbral min_valid_share ----------


def _threshold_frame() -> pd.DataFrame:
    """Test de 3 filas con 4, 2 y 1 simbolos validos."""
    df = make_returns(3, SYMS, seed=14, start="2024-02-01")
    df.iloc[1, [2, 3]] = np.nan
    df.iloc[2, [1, 2, 3]] = np.nan
    return df


def test_threshold_inclusive_and_row_nan_below() -> None:
    """2 de 4 validos con umbral 0.5 define el factor; 1 de 4 da NaN en factor y todo el residuo."""
    m, _ = _fit_clean(min_valid_share=0.5)
    test = _threshold_frame()
    fac = m.factors(test)["mean"]
    assert np.isfinite(fac.iloc[0])
    assert fac.iloc[1] == pytest.approx(test.iloc[1, :2].mean())
    assert np.isnan(fac.iloc[2])
    res = m.transform(test)
    assert res.iloc[2].isna().all()  # incluida la moneda que si tenia retorno, nunca 0
    assert test.iloc[2, 0] == test.iloc[2, 0]  # AAA tenia retorno
    assert res.iloc[1, :2].notna().all()
    assert res.iloc[1, 2:].isna().all()


def test_threshold_one_requires_all_valid() -> None:
    """Con min_valid_share=1.0 basta un NaN en la fila para que el factor sea NaN."""
    m, _ = _fit_clean(min_valid_share=1.0)
    test = _threshold_frame()
    fac = m.factors(test)["mean"]
    assert np.isfinite(fac.iloc[0])
    assert np.isnan(fac.iloc[1])
    assert np.isnan(fac.iloc[2])
    assert m.transform(test).iloc[1].isna().all()


# ---------- fila toda NaN ----------


def test_all_nan_row_no_warning() -> None:
    """Una fila toda NaN da factor y residuo NaN sin RuntimeWarning."""
    m, _ = _fit_clean()
    test = make_returns(5, SYMS, seed=15, start="2024-02-01")
    test.iloc[2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fac = m.factors(test)
        res = m.transform(test)
    assert np.isnan(fac["mean"].iloc[2])
    assert res.iloc[2].isna().all()
    assert res.drop(res.index[2]).notna().all().all()


# ---------- betas y residuo ----------


def test_betas_are_ols_against_train_mean() -> None:
    """Las betas son exactamente ols_betas(train, media de train, min_periods)."""
    train = make_returns(200, SYMS, seed=16, nan_frac=0.05)
    m = MeanFactor(min_periods=20).fit(train)
    expected = ols_betas(
        train, train.mean(axis=1).where(train.notna().mean(axis=1) >= 0.5), min_periods=20
    )
    pd.testing.assert_series_equal(m.get_state()["betas"], expected)


def test_residual_is_r_minus_beta_times_factor_no_alpha() -> None:
    """El residuo es r - beta*factor celda a celda; el alfa NO se resta (queda en el residuo)."""
    rng = np.random.default_rng(17)
    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    f = rng.normal(0, 1, n)
    alphas = np.array([0.5, -0.4, 0.3, 0.0, 0.2, -0.1])
    betas_true = np.array([0.5, 1.5, 1.0, 0.8, 1.2, 1.0])
    data = alphas + np.outer(f, betas_true) + rng.normal(0, 0.01, (n, 6))
    cols = [f"S{i}" for i in range(6)]
    df = pd.DataFrame(data, index=idx, columns=cols)
    m = MeanFactor(min_periods=100).fit(df)
    betas = m.get_state()["betas"]
    fac = m.factors(df)["mean"]
    res = m.transform(df)
    np.testing.assert_allclose(res.to_numpy(), df.to_numpy() - np.outer(fac, betas), atol=1e-12)
    # sin restar alfa: la media del residuo de cada moneda es alfa_i - beta_i*media(factor), no 0
    expected_mean = df.mean() - betas * fac.mean()
    np.testing.assert_allclose(res.mean().to_numpy(), expected_mean.to_numpy(), atol=1e-12)
    assert (res.mean().abs() > 1e-3).any()


def test_ols_mean_property_betas_average_one_and_residuals_sum_zero() -> None:
    """Dependencia lineal: betas con media 1 y residuos de cada fila suman ~0 (train y test)."""
    train = make_returns(300, SYMS, seed=18)
    test = make_returns(50, SYMS, seed=19, start="2024-02-01")
    m = MeanFactor(min_periods=10).fit(train)
    betas = m.get_state()["betas"]
    assert np.isclose(betas.mean(), 1.0)
    for df in (train, test):
        sums = m.transform(df).sum(axis=1).to_numpy()
        assert np.allclose(sums, 0.0, atol=1e-12)


# ---------- sin fuga ----------


def test_betas_frozen_after_transform_of_scaled_test() -> None:
    """Proyectar un test escalado x10 no cambia las betas de train."""
    m, _ = _fit_clean()
    before = m.get_state()["betas"].copy()
    test = make_returns(40, SYMS, seed=20, start="2024-02-01")
    m.transform(test)
    m.transform(test * 10)
    m.factors(test * 10)
    pd.testing.assert_series_equal(m.get_state()["betas"], before)


def test_test_rows_are_independent() -> None:
    """Modificar la fila k de test no cambia factor ni residuo de las demas filas."""
    m, _ = _fit_clean()
    test = make_returns(30, SYMS, seed=21, start="2024-02-01")
    changed = test.copy()
    k = 10
    changed.iloc[k] = changed.iloc[k] * 100 + 0.5
    others = [i for i in range(len(test)) if i != k]
    pd.testing.assert_frame_equal(m.factors(test).iloc[others], m.factors(changed).iloc[others])
    pd.testing.assert_frame_equal(m.transform(test).iloc[others], m.transform(changed).iloc[others])
    assert not np.allclose(m.factors(test).iloc[k], m.factors(changed).iloc[k])


# ---------- casos borde ----------


def test_all_nan_column_in_train() -> None:
    """Una columna toda NaN en train tiene beta y residuo NaN y no afecta a las demas."""
    train = make_returns(200, SYMS, seed=22)
    train["DDD"] = np.nan
    m = MeanFactor(min_periods=10).fit(train)
    betas = m.get_state()["betas"]
    assert np.isnan(betas["DDD"])
    ref = MeanFactor(min_periods=10).fit(train[["AAA", "BBB", "CCC"]])
    pd.testing.assert_series_equal(betas[["AAA", "BBB", "CCC"]], ref.get_state()["betas"])
    test = make_returns(30, SYMS, seed=23, start="2024-02-01")
    test["DDD"] = np.nan
    res = m.transform(test)
    assert res["DDD"].isna().all()
    pd.testing.assert_frame_equal(
        res[["AAA", "BBB", "CCC"]], ref.transform(test[["AAA", "BBB", "CCC"]])
    )
    # con retornos en DDD en test: su residuo sigue NaN y las demas siguen definidas
    test2 = make_returns(30, SYMS, seed=23, start="2024-02-01")
    res2 = m.transform(test2)
    assert res2["DDD"].isna().all()
    assert res2[["AAA", "BBB", "CCC"]].notna().all().all()


def test_single_symbol() -> None:
    """Con N=1 el factor es el propio retorno, beta ~1 y residuo ~0."""
    train = make_returns(200, ("AAA",), seed=24)
    test = make_returns(30, ("AAA",), seed=25, start="2024-02-01")
    m = MeanFactor(min_periods=10).fit(train)
    assert m.get_state()["betas"]["AAA"] == pytest.approx(1.0)
    np.testing.assert_array_equal(m.factors(test)["mean"].to_numpy(), test["AAA"].to_numpy())
    assert np.allclose(m.transform(test)["AAA"].to_numpy(), 0.0, atol=1e-12)


# ---------- parametros ----------


@pytest.mark.parametrize("share", [0, 0.0, -0.1, 1.0000001, 2.0])
def test_invalid_min_valid_share_raises(share: float) -> None:
    """min_valid_share fuera de (0, 1] lanza ValueError."""
    with pytest.raises(ValueError):
        MeanFactor(min_valid_share=share)


def test_min_periods_lt_2_raises() -> None:
    """min_periods=1 lanza ValueError."""
    with pytest.raises(ValueError):
        MeanFactor(min_periods=1)


def test_min_valid_share_one_is_valid() -> None:
    """min_valid_share=1.0 es valido."""
    assert MeanFactor(min_valid_share=1.0).min_valid_share == 1.0


# ---------- sin ajustar ----------


def test_not_fitted_raises() -> None:
    """Llamar a transform y factors antes de fit lanza NotFittedError."""
    test = make_returns(10, SYMS, seed=26)
    m = MeanFactor(min_periods=10)
    with pytest.raises(NotFittedError):
        m.transform(test)
    with pytest.raises(NotFittedError):
        m.factors(test)
