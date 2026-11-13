"""Tests de `PcaFactor`."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from miax.features import PcaFactor as ExportedPcaFactor
from miax.features.factor import NotFittedError
from miax.features.factor_pca import PcaFactor
from tests.unit.factor_contract import check_factor_model, make_returns

SYMS = ("AAA", "BBB", "CCC", "DDD")
LOADS4 = np.array([0.6, 1.0, 1.4, 1.0])


def _factor_frame(
    n: int,
    c: np.ndarray,
    seed: int,
    *,
    start: str = "2024-01-01",
    noise: float = 0.0005,
    symbols: tuple[str, ...] | None = None,
    nan_frac: float = 0.0,
) -> pd.DataFrame:
    """Retornos r_i = c_i * F + ruido pequeno, con un factor comun F y NaN dispersos opcionales."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, n)
    data = np.outer(f, c) + rng.normal(0.0, noise, (n, len(c)))
    if nan_frac > 0:
        data[rng.random(data.shape) < nan_frac] = np.nan
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    cols = list(symbols) if symbols is not None else [f"S{i}" for i in range(len(c))]
    return pd.DataFrame(data, index=idx, columns=cols, dtype="float64")


def _fit4(min_valid_share: float = 0.5) -> tuple[PcaFactor, pd.DataFrame]:
    """Modelo de 4 simbolos ajustado con 400 filas sin NaN con factor comun, y su train."""
    train = _factor_frame(400, LOADS4, seed=11, symbols=SYMS)
    return PcaFactor(min_valid_share=min_valid_share, min_periods=10).fit(train), train


# ---------- contrato ----------


def test_contract_pca_factor() -> None:
    """Cumple el contrato de FactorModel con NaN dispersos y un factor comun real en los datos."""
    train = _factor_frame(300, LOADS4, seed=1, symbols=SYMS, nan_frac=0.05)
    test = _factor_frame(60, LOADS4, seed=2, symbols=SYMS, start="2024-02-01", nan_frac=0.05)
    check_factor_model(lambda: PcaFactor(min_periods=10), train, test)


def test_contract_pca_factor_independent_noise() -> None:
    """Tambien cumple el contrato con ruido independiente (PC1 mal definido)."""
    train = make_returns(80, SYMS, seed=1, nan_frac=0.05)
    test = make_returns(40, SYMS, seed=2, start="2024-01-04 08:00", nan_frac=0.05)
    check_factor_model(lambda: PcaFactor(min_periods=10), train, test)


def test_export_from_package() -> None:
    """PcaFactor se exporta desde miax.features."""
    assert ExportedPcaFactor is PcaFactor


# ---------- recuperacion de un factor conocido ----------


def test_recovers_known_loadings() -> None:
    """Con r_i = c_i*F + ruido pequeno las cargas son proporcionales a c_i, con media 1."""
    c = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
    train = _factor_frame(2000, c, seed=3)
    m = PcaFactor(min_periods=100).fit(train)
    loadings = m.loadings
    assert loadings.mean() == pytest.approx(1.0, abs=1e-12)
    assert (loadings > 0).all()
    np.testing.assert_allclose(loadings.to_numpy(), c / c.mean(), rtol=0.05)
    assert 0.9 < m.explained_share <= 1.0


# ---------- signo ----------


def test_sign_flip_of_train_gives_same_loadings() -> None:
    """La correlacion y las std no cambian al invertir todos los retornos; la media 1 fija el signo.

    Por tanto las cargas de -train son las de train, y el factor de -test es el opuesto.
    """
    train = _factor_frame(500, LOADS4, seed=4, symbols=SYMS)
    test = _factor_frame(50, LOADS4, seed=5, symbols=SYMS, start="2024-03-01")
    a = PcaFactor(min_periods=10).fit(train)
    b = PcaFactor(min_periods=10).fit(-train)
    assert a.loadings.mean() == pytest.approx(1.0)
    assert b.loadings.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(a.loadings.to_numpy(), b.loadings.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(
        a.factors(-test).to_numpy(), -a.factors(test).to_numpy(), rtol=1e-12, atol=1e-15
    )


# ---------- correlacion, no covarianza (caso FTT) ----------


def test_correlation_not_covariance_volatile_coin_does_not_dominate() -> None:
    """Una moneda con volatilidad idiosincratica 10x y poca correlacion no domina las cargas."""
    rng = np.random.default_rng(6)
    n = 3000
    f = rng.normal(0.0, 0.01, n)
    others = f[:, None] + rng.normal(0.0, 0.005, (n, 5))
    ftt = 0.1 * f + rng.normal(0.0, 0.1, n)  # std ~10x la del resto, corr con f ~ 0.01
    data = np.column_stack([others, ftt])
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    cols = [f"S{i}" for i in range(5)] + ["FTT"]
    df = pd.DataFrame(data, index=idx, columns=cols)

    # PCA de covarianzas en el propio test: FTT acapara el primer autovector.
    _, vecs = np.linalg.eigh(np.cov(data, rowvar=False))
    cov_weight = vecs[-1, -1] ** 2
    assert cov_weight > 0.9

    loadings = PcaFactor(min_periods=100).fit(df).loadings
    b2 = loadings.to_numpy() ** 2
    pca_weight = b2[-1] / b2.sum()
    assert pca_weight < 0.3 < cov_weight
    # La carga de FTT es del orden de las demas, no varias veces mayor.
    assert abs(loadings["FTT"]) < 2.0 * loadings.drop("FTT").abs().max()


# ---------- ortogonalidad ----------


def test_residuals_orthogonal_to_loadings_each_row() -> None:
    """Dependencia lineal de los residuos (Epoca 4): sum_i b_i*res_i = 0 en cada fila."""
    train = _factor_frame(300, LOADS4, seed=7, symbols=SYMS)
    test = _factor_frame(60, LOADS4, seed=8, symbols=SYMS, start="2024-02-01")
    m = PcaFactor(min_periods=10).fit(train)
    for df in (train, test):
        res = m.transform(df)
        sums = (m.loadings * res).sum(axis=1).to_numpy()
        np.testing.assert_allclose(sums, 0.0, atol=1e-12)


def test_residuals_orthogonal_with_nan_cells() -> None:
    """Con NaN en celdas de test, la suma de b*res sobre los presentes tambien es ~0."""
    m, _ = _fit4(min_valid_share=0.5)
    test = _factor_frame(80, LOADS4, seed=9, symbols=SYMS, start="2024-02-01", nan_frac=0.2)
    res = m.transform(test)
    sums = (m.loadings * res).sum(axis=1, skipna=True).to_numpy()
    defined = m.factors(test)["pca"].notna().to_numpy()
    assert defined.any()
    np.testing.assert_allclose(sums[defined], 0.0, atol=1e-12)


# ---------- factor con huecos ----------


def test_factor_with_gaps_is_projection_over_present() -> None:
    """Con monedas ausentes el factor es exactamente sum(b*r)/sum(b^2) sobre las presentes."""
    m, _ = _fit4(min_valid_share=0.25)
    test = _factor_frame(10, LOADS4, seed=10, symbols=SYMS, start="2024-02-01")
    test.iloc[2, 0] = np.nan
    test.iloc[5, [1, 3]] = np.nan
    b = m.loadings
    fac = m.factors(test)["pca"]
    for k, cols in ((2, ["BBB", "CCC", "DDD"]), (5, ["AAA", "CCC"])):
        expected = (b[cols] * test.iloc[k][cols]).sum() / (b[cols] ** 2).sum()
        assert fac.iloc[k] == pytest.approx(expected, rel=1e-12)
    full = (b * test.iloc[0]).sum() / (b**2).sum()
    assert fac.iloc[0] == pytest.approx(full, rel=1e-12)


# ---------- umbral min_valid_share ----------


def _threshold_frame() -> pd.DataFrame:
    """Test de 3 filas con 4, 2 y 1 simbolos validos."""
    df = _factor_frame(3, LOADS4, seed=14, symbols=SYMS, start="2024-02-01")
    df.iloc[1, [2, 3]] = np.nan
    df.iloc[2, [1, 2, 3]] = np.nan
    return df


def test_threshold_inclusive_and_row_nan_below() -> None:
    """2 de 4 validos con umbral 0.5 define el factor; 1 de 4 da NaN en factor y todo el residuo."""
    m, _ = _fit4(min_valid_share=0.5)
    test = _threshold_frame()
    fac = m.factors(test)["pca"]
    assert np.isfinite(fac.iloc[0])
    assert np.isfinite(fac.iloc[1])
    assert np.isnan(fac.iloc[2])
    res = m.transform(test)
    assert res.iloc[2].isna().all()  # incluida AAA, que si tenia retorno: nunca 0
    assert not np.isnan(test.iloc[2, 0])
    assert res.iloc[1, :2].notna().all()
    assert res.iloc[1, 2:].isna().all()


def test_threshold_one_requires_all_valid() -> None:
    """Con min_valid_share=1.0 basta un NaN en la fila para que el factor sea NaN."""
    m, _ = _fit4(min_valid_share=1.0)
    test = _threshold_frame()
    fac = m.factors(test)["pca"]
    assert np.isfinite(fac.iloc[0])
    assert np.isnan(fac.iloc[1])
    assert np.isnan(fac.iloc[2])
    assert m.transform(test).iloc[1].isna().all()


# ---------- fila toda NaN ----------


def test_all_nan_row_no_warning() -> None:
    """Una fila toda NaN da factor y residuo NaN sin RuntimeWarning."""
    m, _ = _fit4()
    test = _factor_frame(5, LOADS4, seed=15, symbols=SYMS, start="2024-02-01")
    test.iloc[2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fac = m.factors(test)
        res = m.transform(test)
    assert np.isnan(fac["pca"].iloc[2])
    assert res.iloc[2].isna().all()
    assert res.drop(res.index[2]).notna().all().all()


# ---------- simbolos no utilizables ----------


def _unusable_train(kind: str) -> pd.DataFrame:
    """Train de 5 columnas donde la ultima (S4) es no utilizable segun `kind`."""
    train = _factor_frame(300, np.array([0.8, 1.0, 1.2, 1.0, 1.0]), seed=16)
    if kind == "all_nan":
        train["S4"] = np.nan
    elif kind == "constant":
        train["S4"] = 0.0  # retorno cero constante (precio parado): std exactamente 0
    elif kind == "constant_nonzero":
        train["S4"] = 0.001  # constante distinta de 0: std ~2e-19 por redondeo, no 0
    else:  # short: menos de min_periods (10) retornos
        train.loc[train.index[5:], "S4"] = np.nan
    return train


@pytest.mark.parametrize("kind", ["all_nan", "constant", "constant_nonzero", "short"])
def test_unusable_symbol_has_nan_loading_and_residual(kind: str) -> None:
    """Columna toda NaN, constante o corta: carga NaN y residuo NaN; el resto, cargas finitas."""
    train = _unusable_train(kind)
    m = PcaFactor(min_periods=10).fit(train)
    b = m.loadings
    assert np.isnan(b["S4"])
    good = b.drop("S4")
    assert np.isfinite(good).all()
    assert good.mean() == pytest.approx(1.0)
    test = _factor_frame(30, np.array([0.8, 1.0, 1.2, 1.0, 1.0]), seed=17, start="2024-02-01")
    res = m.transform(test)
    assert res["S4"].isna().all()
    assert res.drop(columns="S4").notna().all().all()


@pytest.mark.parametrize("kind", ["all_nan", "constant", "constant_nonzero", "short"])
def test_threshold_denominator_counts_only_loaded_symbols(kind: str) -> None:
    """Con 5 columnas y 1 sin carga, 2 de las 4 con carga presentes definen el factor (0.5)."""
    m = PcaFactor(min_valid_share=0.5, min_periods=10).fit(_unusable_train(kind))
    test = _factor_frame(3, np.array([0.8, 1.0, 1.2, 1.0, 1.0]), seed=18, start="2024-02-01")
    test.iloc[0, [2, 3]] = np.nan  # quedan S0, S1 (con carga) y S4 (sin carga): 2 de 4
    test.iloc[1, [1, 2, 3]] = np.nan  # solo S0 con carga: 1 de 4 aunque S4 este presente
    fac = m.factors(test)["pca"]
    assert np.isfinite(fac.iloc[0])
    assert np.isnan(fac.iloc[1])
    assert np.isfinite(fac.iloc[2])


# ---------- errores de fit ----------


def test_fit_fewer_than_two_usable_raises() -> None:
    """Con menos de 2 simbolos utilizables fit lanza ValueError."""
    train = _factor_frame(100, np.array([1.0, 1.0, 1.0]), seed=19)
    train[["S1", "S2"]] = np.nan
    with pytest.raises(ValueError, match="al menos 2"):
        PcaFactor(min_periods=10).fit(train)


def test_fit_pair_without_enough_common_rows_raises_naming_both() -> None:
    """Dos monedas sin solape (cada una con datos suficientes) hacen fallar fit nombrandolas."""
    train = _factor_frame(200, np.array([1.0, 1.0, 1.0]), seed=20, symbols=("AAA", "BBB", "CCC"))
    train.iloc[:100, 1] = np.nan  # BBB solo en la segunda mitad
    train.iloc[100:, 0] = np.nan  # AAA solo en la primera
    with pytest.raises(ValueError, match=r"AAA.*BBB|BBB.*AAA"):
        PcaFactor(min_periods=10).fit(train)
    assert not PcaFactor(min_periods=10).is_fitted


def test_fit_pair_with_zero_variance_overlap_raises() -> None:
    """Un par cuyo solapamiento tiene varianza nula en una moneda hace fallar fit."""
    train = _factor_frame(300, np.array([1.0, 1.0, 1.0]), seed=21, symbols=("AAA", "BBB", "CCC"))
    train.iloc[:100, 1] = np.nan  # BBB en filas 100..299
    train.iloc[200:, 0] = np.nan  # AAA en filas 0..199; solape 100..199
    train.iloc[100:200, 0] = 0.0  # AAA constante en el solape (pero varia en 0..99)
    m = PcaFactor(min_periods=10)
    with pytest.raises(ValueError, match="correlacion NaN"):
        m.fit(train)
    assert not m.is_fitted


# ---------- correlacion por pares no semidefinida positiva ----------


def test_pairwise_non_psd_correlation_gives_valid_explained_share() -> None:
    """Huecos que hacen la correlacion por pares no PSD: explained_share finito y en (0, 1]."""
    rng = np.random.default_rng(22)
    n = 300  # por bloque
    z = rng.normal(0.0, 1.0, (3 * n, 1))
    e = rng.normal(0.0, 0.2, (3 * n, 3))
    base = z + e
    df = np.full((3 * n, 3), np.nan)
    df[:n, [0, 1]] = base[:n, [0, 1]]  # A,B correlacionadas +
    df[n : 2 * n, [1, 2]] = base[n : 2 * n, [1, 2]]  # B,C correlacionadas +
    df[2 * n :, 0] = base[2 * n :, 0]  # A,C correlacionadas -
    df[2 * n :, 2] = -base[2 * n :, 2]
    idx = pd.date_range("2024-01-01", periods=3 * n, freq="h", tz="UTC")
    frame = pd.DataFrame(df, index=idx, columns=["AAA", "BBB", "CCC"])
    corr = frame.corr(min_periods=100).to_numpy()
    assert np.linalg.eigvalsh(corr).min() < 0  # el caso es realmente no PSD
    m = PcaFactor(min_periods=100).fit(frame)
    assert np.isfinite(m.explained_share)
    assert 0.0 < m.explained_share <= 1.0


# ---------- parametros ----------


@pytest.mark.parametrize("share", [0, 0.0, -0.1, 1.0000001, 2.0])
def test_invalid_min_valid_share_raises(share: float) -> None:
    """min_valid_share fuera de (0, 1] lanza ValueError."""
    with pytest.raises(ValueError):
        PcaFactor(min_valid_share=share)


def test_min_periods_lt_2_raises() -> None:
    """min_periods=1 lanza ValueError."""
    with pytest.raises(ValueError):
        PcaFactor(min_periods=1)


def test_min_valid_share_one_is_valid() -> None:
    """min_valid_share=1.0 es valido."""
    assert PcaFactor(min_valid_share=1.0).min_valid_share == 1.0


# ---------- sin ajustar ----------


def test_not_fitted_raises() -> None:
    """transform, factors, loadings y explained_share antes de fit lanzan NotFittedError."""
    test = make_returns(10, SYMS, seed=26)
    m = PcaFactor(min_periods=10)
    with pytest.raises(NotFittedError):
        m.transform(test)
    with pytest.raises(NotFittedError):
        m.factors(test)
    with pytest.raises(NotFittedError):
        _ = m.loadings
    with pytest.raises(NotFittedError):
        _ = m.explained_share


# ---------- sin fuga / estado congelado ----------


def test_loadings_returns_copy() -> None:
    """Mutar la Series devuelta por loadings no cambia el estado ni las cargas."""
    m, _ = _fit4()
    state_before = m.get_state()["loadings"].copy()
    got = m.loadings
    got[:] = 99.0
    pd.testing.assert_series_equal(m.loadings, state_before)
    pd.testing.assert_series_equal(m.get_state()["loadings"], state_before)


def test_state_frozen_after_transform_of_scaled_test() -> None:
    """Proyectar un test x10 no cambia cargas ni explained_share."""
    m, _ = _fit4()
    loadings = m.loadings
    share = m.explained_share
    test = _factor_frame(40, LOADS4, seed=27, symbols=SYMS, start="2024-02-01")
    m.transform(test)
    m.transform(test * 10)
    m.factors(test * 10)
    pd.testing.assert_series_equal(m.loadings, loadings)
    assert m.explained_share == share


def test_test_rows_are_independent() -> None:
    """Modificar la fila k de test no cambia factor ni residuo de las demas filas."""
    m, _ = _fit4()
    test = _factor_frame(30, LOADS4, seed=28, symbols=SYMS, start="2024-02-01")
    changed = test.copy()
    k = 10
    changed.iloc[k] = changed.iloc[k] * 100 + 0.5
    others = [i for i in range(len(test)) if i != k]
    pd.testing.assert_frame_equal(m.factors(test).iloc[others], m.factors(changed).iloc[others])
    pd.testing.assert_frame_equal(m.transform(test).iloc[others], m.transform(changed).iloc[others])
    assert not np.allclose(m.factors(test).iloc[k], m.factors(changed).iloc[k])


# ---------- rendimiento ----------


def test_transform_large_frame_is_fast() -> None:
    """Transform sobre ~130000 x 40 con NaN dispersos tarda pocos segundos."""
    rng = np.random.default_rng(29)
    n, k = 130_000, 40
    f = rng.normal(0.0, 0.001, n)
    data = f[:, None] * rng.uniform(0.5, 1.5, k) + rng.normal(0.0, 0.001, (n, k))
    data[rng.random(data.shape) < 0.05] = np.nan
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(k)])
    m = PcaFactor(min_periods=1000).fit(df.iloc[:5000])
    t0 = time.perf_counter()
    res = m.transform(df)
    elapsed = time.perf_counter() - t0
    assert res.shape == df.shape
    assert elapsed < 5.0
