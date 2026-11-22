"""Tests de `correlation_drop` (MIAX-054): caida de correlacion al residualizar."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from miax.features import BtcFactor, MeanFactor, NoFactor, correlation_drop
from miax.features import factor_metrics as module
from miax.features.correlation import explained_variance, raw_correlation
from tests.unit.factor_contract import make_returns

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
MIN = 50
KEYS = [
    "n_symbols",
    "n_symbols_without_data",
    "symbols_without_data",
    "n_pairs",
    "raw_pair_mean",
    "residual_pair_mean",
    "raw_pair_abs_mean",
    "residual_pair_abs_mean",
    "correlation_drop",
    "raw_first_eigenvalue_share",
    "residual_first_eigenvalue_share",
]


@pytest.fixture(autouse=True)
def _propagate_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deja que caplog vea el logger del modulo (tiene propagate=False)."""
    monkeypatch.setattr(module.logger, "propagate", True)


def one_factor(
    n: int = 600, symbols: tuple[str, ...] = SYMBOLS, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Retornos r = beta*f + ruido; devuelve (retornos, factor, betas)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    f = pd.Series(rng.normal(0, 0.01, n), index=idx)
    betas = pd.Series(np.linspace(0.8, 1.2, len(symbols)), index=list(symbols))
    noise = rng.normal(0, 0.002, (n, len(symbols)))
    returns = pd.DataFrame(f.to_numpy()[:, None] * betas.to_numpy() + noise, index=idx)
    returns.columns = list(symbols)
    return returns, f, betas


def known_residuals(returns: pd.DataFrame, f: pd.Series, betas: pd.Series) -> pd.DataFrame:
    """Residuos = retorno - beta*f."""
    return returns - f.to_numpy()[:, None] * betas.to_numpy()[None, :]


def test_known_factor_gives_high_drop_and_small_residual_correlation() -> None:
    """Con un factor conocido quitado, la caida es alta y la correlacion residual pequena."""
    returns, f, betas = one_factor()
    out = correlation_drop(returns, known_residuals(returns, f, betas), min_periods=MIN)

    assert out["correlation_drop"] > 0.9
    assert out["residual_pair_abs_mean"] < 0.1
    assert out["raw_pair_abs_mean"] > 0.8
    assert out["n_pairs"] == 15


def test_no_factor_gives_zero_drop_and_equal_means() -> None:
    """`NoFactor` deja los residuos iguales a los retornos: caida 0 y medias iguales."""
    returns, _, _ = one_factor()
    residuals = NoFactor().fit(returns).transform(returns)

    out = correlation_drop(returns, residuals, min_periods=MIN)

    assert out["correlation_drop"] == 0.0
    assert out["raw_pair_mean"] == out["residual_pair_mean"]
    assert out["raw_pair_abs_mean"] == out["residual_pair_abs_mean"]
    assert out["raw_first_eigenvalue_share"] == out["residual_first_eigenvalue_share"]


def test_mean_factor_signed_mean_goes_negative_but_drop_uses_abs() -> None:
    """Con MeanFactor la media con signo residual baja a -1/(N-1) y la caida usa |rho|."""
    n_sym = 8
    symbols = tuple(f"S{i}" for i in range(n_sym))
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=4000, freq="min", tz="UTC")
    f = rng.normal(0, 0.01, 4000)
    returns = pd.DataFrame(
        f[:, None] + rng.normal(0, 0.01, (4000, n_sym)), index=idx, columns=list(symbols)
    )
    model = MeanFactor(min_periods=100).fit(returns)
    residuals = model.transform(returns)

    out = correlation_drop(returns, residuals, min_periods=100)

    assert out["residual_pair_mean"] == pytest.approx(-1 / (n_sym - 1), abs=0.05)
    assert out["residual_pair_mean"] < 0
    expected = 1.0 - out["residual_pair_abs_mean"] / out["raw_pair_abs_mean"]
    assert out["correlation_drop"] == pytest.approx(expected)
    # Con signo la caida seria mayor que 1 menos nada: la formula con signo daria otro valor.
    assert out["correlation_drop"] != pytest.approx(
        1.0 - out["residual_pair_mean"] / out["raw_pair_mean"]
    )


def test_nan_in_residual_masks_raw_return_of_same_minute() -> None:
    """Un NaN en el residuo de un minuto quita ese minuto del retorno crudo (calculo a mano)."""
    returns, f, betas = one_factor(n=300)
    residuals = known_residuals(returns, f, betas)
    residuals.iloc[10:60, 0] = np.nan
    residuals.iloc[100:120, 2] = np.nan

    out = correlation_drop(returns, residuals, min_periods=MIN)

    raw_masked = returns.where(residuals.notna())
    res_masked = residuals.where(returns.notna())
    pairs_raw = raw_correlation(raw_masked, min_periods=MIN).to_numpy()
    pairs_res = raw_correlation(res_masked, min_periods=MIN).to_numpy()
    upper = np.triu_indices(len(SYMBOLS), k=1)
    assert out["raw_pair_abs_mean"] == pytest.approx(np.abs(pairs_raw[upper]).mean())
    assert out["residual_pair_abs_mean"] == pytest.approx(np.abs(pairs_res[upper]).mean())
    # Y se nota: sin enmascarar, el cruda de esos pares seria otro.
    unmasked = raw_correlation(returns, min_periods=MIN).to_numpy()[upper]
    assert not np.isclose(np.abs(unmasked).mean(), out["raw_pair_abs_mean"], rtol=0, atol=1e-12)


def test_nan_in_returns_masks_residual_of_same_minute() -> None:
    """Un NaN en el retorno de un minuto quita ese minuto del residuo (calculo a mano)."""
    returns, f, betas = one_factor(n=300)
    residuals = known_residuals(returns, f, betas)
    returns.iloc[10:60, 1] = np.nan
    returns.iloc[200:230, 4] = np.nan

    out = correlation_drop(returns, residuals, min_periods=MIN)

    res_masked = residuals.where(returns.notna())
    pairs_res = raw_correlation(res_masked, min_periods=MIN).to_numpy()
    upper = np.triu_indices(len(SYMBOLS), k=1)
    assert out["residual_pair_abs_mean"] == pytest.approx(np.abs(pairs_res[upper]).mean())
    unmasked = raw_correlation(residuals, min_periods=MIN).to_numpy()[upper]
    assert not np.isclose(np.abs(unmasked).mean(), out["residual_pair_abs_mean"], atol=1e-12)


@pytest.mark.parametrize("mode", ["all_nan", "short"])
def test_symbol_without_data_is_set_aside_and_warns(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Una moneda toda NaN o con menos de `min_periods` se aparta y avisa."""
    returns, f, betas = one_factor(n=300)
    residuals = known_residuals(returns, f, betas)
    if mode == "all_nan":
        returns["CCC"] = np.nan
    else:
        returns.loc[returns.index[MIN - 10 :], "CCC"] = np.nan

    with caplog.at_level(logging.WARNING):
        out = correlation_drop(returns, residuals, min_periods=MIN)

    assert out["symbols_without_data"] == "CCC"
    assert out["n_symbols_without_data"] == 1
    assert out["n_symbols"] == 6
    assert out["n_pairs"] == 10
    assert any("CCC" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_fewer_than_two_symbols_with_data_raises() -> None:
    """Con menos de 2 monedas con datos la metrica no tiene sentido: ValueError."""
    returns, f, betas = one_factor(n=300)
    residuals = known_residuals(returns, f, betas)
    returns.loc[:, list(SYMBOLS[1:])] = np.nan

    with pytest.raises(ValueError, match="menos de 2"):
        correlation_drop(returns, residuals, min_periods=MIN)


def test_no_common_pair_raises() -> None:
    """Si ningun par tiene valor en las dos matrices, ValueError."""
    returns, f, betas = one_factor(n=300, symbols=("AAA", "BBB"))
    residuals = known_residuals(returns, f, betas)
    # Solapan solo 20 filas (< min_periods) pero cada una tiene datos de sobra por su cuenta.
    returns.iloc[:140, 0] = np.nan
    returns.iloc[160:, 1] = np.nan

    with pytest.raises(ValueError, match="ningun par"):
        correlation_drop(returns, residuals, min_periods=MIN)


def test_zero_raw_correlation_raises() -> None:
    """Con dos columnas ortogonales y de media cero la correlacion cruda es 0: ValueError."""
    idx = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    # Con 3 columnas pandas deja un -5.6e-17 en algun par (no es exactamente 0); con 2 si lo es.
    returns = pd.DataFrame({"A": [1.0, -1.0, 1.0, -1.0], "C": [1.0, -1.0, -1.0, 1.0]}, index=idx)
    residuals = pd.DataFrame(
        np.random.default_rng(0).normal(size=(4, 2)), index=idx, columns=list("AC")
    )

    with pytest.raises(ValueError, match="es 0"):
        correlation_drop(returns, residuals, min_periods=2)


def test_validation_errors() -> None:
    """Indices distintos, columnas distintas, duplicadas, no str o no DataFrame dan ValueError."""
    returns, f, betas = one_factor(n=100)
    residuals = known_residuals(returns, f, betas)

    with pytest.raises(ValueError, match="mismo indice"):
        correlation_drop(returns, residuals.iloc[1:], min_periods=MIN)
    with pytest.raises(ValueError, match="mismas columnas"):
        correlation_drop(returns, residuals.drop(columns="AAA"), min_periods=MIN)
    with pytest.raises(ValueError, match="mismas columnas"):
        correlation_drop(returns, residuals.rename(columns={"AAA": "ZZZ"}), min_periods=MIN)
    dup = returns.copy()
    dup.columns = ["AAA", "AAA", *SYMBOLS[2:]]
    with pytest.raises(ValueError, match="duplicadas"):
        correlation_drop(dup, residuals, min_periods=MIN)
    with pytest.raises(ValueError, match="duplicadas"):
        correlation_drop(returns, dup, min_periods=MIN)
    numeric = returns.copy()
    numeric.columns = range(len(SYMBOLS))
    with pytest.raises(ValueError, match="str"):
        correlation_drop(numeric, residuals.set_axis(range(len(SYMBOLS)), axis=1), min_periods=MIN)
    with pytest.raises(ValueError, match="DataFrame"):
        correlation_drop(returns.to_numpy(), residuals, min_periods=MIN)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="DataFrame"):
        correlation_drop(returns, residuals["AAA"], min_periods=MIN)  # type: ignore[arg-type]


@pytest.mark.parametrize("share", [-0.1, 1.0])
def test_invalid_max_nan_share_raises_instead_of_nan(share: float) -> None:
    """Un `max_nan_share` fuera de [0, 1) es ValueError, no un autovalor NaN silencioso."""
    returns, f, betas = one_factor(n=100)
    residuals = known_residuals(returns, f, betas)
    with pytest.raises(ValueError, match="max_nan_share"):
        correlation_drop(returns, residuals, min_periods=MIN, max_nan_share=share)


def test_residual_column_order_does_not_matter() -> None:
    """Las columnas de `residuals` en otro orden dan el mismo resultado."""
    returns, f, betas = one_factor()
    residuals = known_residuals(returns, f, betas)

    a = correlation_drop(returns, residuals, min_periods=MIN)
    b = correlation_drop(returns, residuals[list(SYMBOLS[::-1])], min_periods=MIN)

    assert a == b


def test_result_keys_order_and_python_types() -> None:
    """El dict trae las claves exactas y en orden, con tipos nativos de Python."""
    returns, f, betas = one_factor()
    out = correlation_drop(returns, known_residuals(returns, f, betas), min_periods=MIN)

    assert list(out) == KEYS
    for key in ("n_symbols", "n_symbols_without_data", "n_pairs"):
        assert type(out[key]) is int
    assert type(out["symbols_without_data"]) is str
    for key in KEYS[4:]:
        assert type(out[key]) is float


def _hole_case() -> tuple[pd.DataFrame, pd.DataFrame]:
    """8 monedas; el par (S0, S1) solo es NaN en los residuos (S0 residual constante)."""
    symbols = tuple(f"S{i}" for i in range(8))
    returns, f, betas = one_factor(n=400, symbols=symbols)
    residuals = known_residuals(returns, f, betas)
    residuals.iloc[:200, 0] = 0.0  # constante en todo el solape con S1
    residuals.iloc[200:, 1] = np.nan  # S1 solo existe en las primeras 200 filas
    return returns, residuals


def test_first_eigenvalue_uses_only_pairs_valid_in_both() -> None:
    """Un par que solo falta en una matriz se quita de las dos antes del primer autovalor."""
    returns, residuals = _hole_case()
    raw_m = raw_correlation(returns.where(residuals.notna()), min_periods=MIN)
    res_m = raw_correlation(residuals.where(returns.notna()), min_periods=MIN)
    assert np.isnan(res_m.loc["S0", "S1"])
    assert not np.isnan(raw_m.loc["S0", "S1"])

    out = correlation_drop(returns, residuals, min_periods=MIN)

    both_missing = raw_m.copy()
    both_missing.loc["S0", "S1"] = both_missing.loc["S1", "S0"] = np.nan
    res_missing = res_m
    expected_raw = explained_variance(both_missing, require_psd=False)[0]
    expected_res = explained_variance(res_missing, require_psd=False)[0]
    assert out["n_pairs"] == 27
    assert out["raw_first_eigenvalue_share"] == pytest.approx(expected_raw)
    assert out["residual_first_eigenvalue_share"] == pytest.approx(expected_res)
    # Si el par crudo siguiera dentro, el autovalor crudo seria otro.
    with_pair = explained_variance(raw_m, require_psd=False)[0]
    assert out["raw_first_eigenvalue_share"] != pytest.approx(with_pair, abs=1e-12)


def test_eigenvalue_is_nan_with_warning_when_too_many_pairs_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Si faltan mas pares que `max_nan_share`, el autovalor es NaN y avisa; el resto sigue."""
    returns, residuals = _hole_case()

    with caplog.at_level(logging.WARNING):
        out = correlation_drop(returns, residuals, min_periods=MIN, max_nan_share=0.0)

    assert np.isnan(out["raw_first_eigenvalue_share"])
    assert np.isnan(out["residual_first_eigenvalue_share"])
    assert not np.isnan(out["correlation_drop"])
    assert sum("primer autovalor" in r.getMessage() for r in caplog.records) == 2


def test_row_permutation_does_not_change_result() -> None:
    """Permutar las filas por igual en las dos entradas no cambia nada: es del mismo instante."""
    returns, f, betas = one_factor()
    residuals = known_residuals(returns, f, betas)
    returns.iloc[5:40, 1] = np.nan
    perm = np.random.default_rng(7).permutation(len(returns))

    a = correlation_drop(returns, residuals, min_periods=MIN)
    # El indice se permuta con las filas, asi sigue igual en ambas entradas.
    b = correlation_drop(returns.iloc[perm], residuals.iloc[perm], min_periods=MIN)

    assert a.keys() == b.keys()
    for key, value in a.items():
        if isinstance(value, float):
            assert b[key] == pytest.approx(value, abs=1e-12)
        else:
            assert b[key] == value


def test_btc_factor_fit_on_train_applied_to_test_sets_btc_aside() -> None:
    """BtcFactor ajustado en train y aplicado a un test posterior deja BTC apartado."""
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT")
    full, _, _ = one_factor(n=900, symbols=symbols, seed=5)
    train, test = full.iloc[:600], full.iloc[600:]
    assert train.index[-1] < test.index[0]
    # El factor de cada moneda es el propio BTC mas ruido: BtcFactor lo reconoce.
    rng = np.random.default_rng(11)
    for frame in (train, test):
        for col in symbols[1:]:
            frame.loc[:, col] = frame["BTCUSDT"] + rng.normal(0, 0.002, len(frame))
    model = BtcFactor(min_periods=MIN).fit(train)

    out = correlation_drop(test, model.transform(test), min_periods=MIN)

    assert out["symbols_without_data"] == "BTCUSDT"
    assert out["n_symbols_without_data"] == 1
    assert out["n_pairs"] == 6
    assert out["correlation_drop"] > 0.5


def test_make_returns_fixture_is_usable_with_no_factor() -> None:
    """`make_returns` del contrato sirve de entrada: caida 0 con `NoFactor`."""
    returns = make_returns(n=200, seed=1)

    out = correlation_drop(returns, returns.copy(), min_periods=MIN)

    assert out["correlation_drop"] == 0.0
