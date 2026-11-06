"""Tests de `BtcFactor` y `ols_betas`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from miax.features.factor import ols_betas
from miax.features.factor_btc import BtcFactor
from tests.unit.factor_contract import check_factor_model, make_returns

SYMS = ("BTCUSDT", "AAA", "BBB", "CCC")


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")


def _synthetic(
    n: int = 500, beta: float = 1.5, alpha: float = 0.3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    r = alpha + beta * f + rng.normal(0, 0.01, n)
    idx = _idx(n)
    return pd.DataFrame({"X": r}, index=idx), pd.Series(f, index=idx)


# ---------- contrato ----------


def test_contract_btc_factor() -> None:
    """Contract btc factor."""
    train = make_returns(80, SYMS, seed=1, nan_frac=0.05)
    test = make_returns(40, SYMS, seed=2, start="2024-01-04 08:00", nan_frac=0.05)
    check_factor_model(lambda: BtcFactor(min_periods=10), train, test)


# ---------- ols_betas ----------


def test_ols_recovers_beta() -> None:
    """Ols recovers beta."""
    r, f = _synthetic()
    b = ols_betas(r, f, min_periods=100)
    assert b["X"] == pytest.approx(1.5, abs=0.01)


def test_ols_below_min_periods_nan() -> None:
    """Ols below min periods nan."""
    r, f = _synthetic(n=50)
    assert np.isnan(ols_betas(r, f, min_periods=51)["X"])
    assert np.isfinite(ols_betas(r, f, min_periods=50)["X"])


@pytest.mark.parametrize(("scale", "n"), [(1.0, 100), (1e-4, 100), (1e-4, 300000)])
def test_ols_constant_factor_nan(scale: float, n: int) -> None:
    """Ols constant factor nan."""
    idx = _idx(n)
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"X": rng.normal(0, 1, n)}, index=idx)
    f = pd.Series(0.1 * scale, index=idx)
    assert np.isnan(ols_betas(r, f, min_periods=10)["X"])


def test_ols_nearly_zero_factor_finite() -> None:
    """Ols nearly zero factor finite."""
    n = 1000
    idx = _idx(n)
    rng = np.random.default_rng(0)
    f = np.zeros(n)
    f[::100] = 1e-3
    r = pd.DataFrame({"X": 2.0 * f + rng.normal(0, 1e-6, n)}, index=idx)
    b = ols_betas(r, pd.Series(f, index=idx), min_periods=10)
    assert np.isfinite(b["X"])
    assert b["X"] == pytest.approx(2.0, abs=0.05)


def test_ols_gaps_use_common_rows_only() -> None:
    """Ols gaps use common rows only."""
    rng = np.random.default_rng(3)
    n = 200
    idx = _idx(n)
    f = rng.normal(0, 1, n)
    x = 0.5 * f + rng.normal(0, 0.1, n)
    y = -1.0 * f + rng.normal(0, 0.1, n)
    f[:10] = np.nan
    x[20:40] = np.nan
    y[50:55] = np.nan
    r = pd.DataFrame({"X": x, "Y": y}, index=idx)
    b = ols_betas(r, pd.Series(f, index=idx), min_periods=10)
    for col, arr in (("X", x), ("Y", y)):
        m = ~np.isnan(arr) & ~np.isnan(f)
        expected = np.polyfit(f[m], arr[m], 1)[0]
        assert b[col] == pytest.approx(expected, rel=1e-9)


def test_ols_min_periods_lt_2_raises() -> None:
    """Ols min periods lt 2 raises."""
    r, f = _synthetic(n=20)
    with pytest.raises(ValueError):
        ols_betas(r, f, min_periods=1)


def test_ols_index_mismatch_raises() -> None:
    """Ols index mismatch raises."""
    r, f = _synthetic(n=20)
    with pytest.raises(ValueError):
        ols_betas(r, f.iloc[1:], min_periods=2)
    with pytest.raises(ValueError):
        ols_betas(r, pd.Series(f.to_numpy(), index=f.index + pd.Timedelta("1min")), min_periods=2)


# ---------- BtcFactor ----------


def _btc_frame(n: int = 500, beta: float = 1.5, alpha: float = 0.3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    f = rng.normal(0, 1, n)
    return pd.DataFrame({"BTCUSDT": f, "AAA": alpha + beta * f + rng.normal(0, 0.01, n)}, index=idx)


def test_no_alpha_residual_keeps_shift() -> None:
    """No alpha residual keeps shift."""
    df = _btc_frame()
    m = BtcFactor(min_periods=100).fit(df)
    res = m.transform(df)
    assert res["AAA"].mean() == pytest.approx(0.3, abs=0.01)


def test_btc_column_all_nan() -> None:
    """Btc column all nan."""
    df = _btc_frame()
    res = BtcFactor(min_periods=100).fit(df).transform(df)
    assert res["BTCUSDT"].isna().all()
    assert res["AAA"].notna().all()


def test_factors_equal_btc_return() -> None:
    """Factors equal btc return."""
    df = _btc_frame()
    fac = BtcFactor(min_periods=100).fit(df).factors(df)
    assert list(fac.columns) == ["btc"]
    np.testing.assert_array_equal(fac["btc"].to_numpy(), df["BTCUSDT"].to_numpy())


def test_betas_frozen_from_train() -> None:
    """Betas frozen from train."""
    train = _btc_frame(beta=1.5, alpha=0.0)
    m = BtcFactor(min_periods=100).fit(train)
    idx = pd.date_range("2024-02-01", periods=300, freq="min", tz="UTC")
    rng = np.random.default_rng(9)
    f = rng.normal(0, 1, 300)
    test = pd.DataFrame({"BTCUSDT": f, "AAA": 0.5 * f}, index=idx)
    state = m.get_state()
    betas_before = state["betas"].copy()
    res = m.transform(test)
    expected = 0.5 * f - 1.5 * f
    np.testing.assert_allclose(res["AAA"].to_numpy(), expected, atol=0.02 * np.abs(f).max())
    pd.testing.assert_series_equal(m.get_state()["betas"], betas_before)


def test_missing_symbol_raises() -> None:
    """Missing symbol raises."""
    df = _btc_frame().rename(columns={"BTCUSDT": "ETHUSDT"})
    with pytest.raises(ValueError):
        BtcFactor(min_periods=10).fit(df)


def test_nan_beta_gives_nan_column() -> None:
    """Nan beta gives nan column."""
    df = _btc_frame()
    train = df.copy()
    train["BBB"] = np.nan
    m = BtcFactor(min_periods=100).fit(train)
    test = df.copy()
    test["BBB"] = 0.01
    res = m.transform(test)
    assert res["BBB"].isna().all()
    assert res["AAA"].notna().all()


def test_btc_too_few_rows_all_nan() -> None:
    """Btc too few rows all nan."""
    df = _btc_frame(n=50)
    m = BtcFactor(min_periods=100).fit(df)
    betas = m.get_state()["betas"]
    assert betas.isna().all()
    res = m.transform(df)
    assert res.isna().all().all()


def test_custom_symbol() -> None:
    """Custom symbol."""
    df = _btc_frame().rename(columns={"BTCUSDT": "ETHUSDT"})
    m = BtcFactor("ETHUSDT", min_periods=100).fit(df)
    res = m.transform(df)
    assert res["ETHUSDT"].isna().all()
    assert res["AAA"].mean() == pytest.approx(0.3, abs=0.01)
    np.testing.assert_array_equal(m.factors(df)["btc"].to_numpy(), df["ETHUSDT"].to_numpy())


def test_min_periods_lt_2_raises() -> None:
    """Min periods lt 2 raises."""
    with pytest.raises(ValueError):
        BtcFactor(min_periods=1)
