"""Tests de los factores sin la propia moneda (MIAX-215): contrato, exclusion y fuga."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from miax.features.factor import FactorModel, NotFittedError, ols_betas, ols_betas_paired
from miax.features.factor_loo import LooMeanFactor, LooPcaFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from tests.unit.factor_contract import check_factor_model, make_returns, make_train_test
from tests.unit.leakage import assert_no_lookahead

MIN_PERIODS = 10
N_TRAIN, N_TEST = 80, 40

MAKERS: dict[str, Callable[[], FactorModel]] = {
    "loo_mean": lambda: LooMeanFactor(min_periods=MIN_PERIODS),
    "loo_pca": lambda: LooPcaFactor(min_periods=MIN_PERIODS),
}


def _panel(n: int, k: int, *, seed: int = 0, nan_frac: float = 0.0) -> pd.DataFrame:
    """Retornos con un factor comun, `k` monedas S0..S{k-1} y NaN dispersos."""
    symbols = tuple(f"S{i}" for i in range(k))
    base = make_returns(n, symbols, seed=seed, nan_frac=nan_frac)
    common = np.random.default_rng(seed + 500).normal(0.0, 0.01, n)
    return base + common[:, None] * np.linspace(0.5, 1.5, k)


# --- contrato ---


@pytest.mark.parametrize("kind", list(MAKERS))
def test_factor_model_contract(kind: str) -> None:
    """Los dos modelos cumplen el contrato comun de FactorModel."""
    train, test = make_train_test()
    check_factor_model(MAKERS[kind], train, test)


# --- fuga de datos ---


def _data() -> pd.DataFrame:
    return _panel(N_TRAIN + N_TEST, 4, seed=7, nan_frac=0.02)


@pytest.mark.parametrize("move_nans", [False, True])
@pytest.mark.parametrize("kind", list(MAKERS))
@pytest.mark.parametrize("cut_pos", [1, N_TEST // 2, N_TEST - 1])
def test_transform_and_loo_factors_ignore_the_future(
    kind: str, cut_pos: int, move_nans: bool
) -> None:
    """Con fit(train), residuos, factor y factor LOO de las filas < corte ignoran el futuro."""
    data = _data()
    start = data.index[N_TRAIN]
    cut = data.index[N_TRAIN + cut_pos]

    def fitted(d: pd.DataFrame) -> FactorModel:
        return MAKERS[kind]().fit(d[d.index < start])

    runs = {
        "transform": lambda d: fitted(d).transform(d[d.index >= start]),
        "factors": lambda d: fitted(d).factors(d[d.index >= start]),
        "loo_factors": lambda d: fitted(d).loo_factors(d[d.index >= start]),  # type: ignore[attr-defined]
    }
    for name, run in runs.items():
        assert_no_lookahead(run, data, cut, move_nans=move_nans, what=f"{kind}: {name}")


@pytest.mark.parametrize("kind", list(MAKERS))
def test_fitted_state_ignores_test(kind: str) -> None:
    """Betas y cargas ajustadas con fit(train) no cambian si se perturba todo el test."""
    data = _data()
    start = data.index[N_TRAIN]
    for pos in (0, N_TEST // 2, N_TEST - 1):
        assert_no_lookahead(
            lambda d: MAKERS[kind]().fit(d[d.index < start]).get_state(),
            data,
            data.index[N_TRAIN + pos],
            what=f"{kind}: estado",
        )


@pytest.mark.parametrize("kind", list(MAKERS))
def test_fit_with_test_is_detected_as_leakage(kind: str) -> None:
    """Ajustar con train+test (la trampa) hace fallar el ayudante antifuga."""
    data = _data()
    start = data.index[N_TRAIN]
    cut = data.index[N_TRAIN + N_TEST // 2]

    def cheat(d: pd.DataFrame) -> pd.DataFrame:
        return MAKERS[kind]().fit(d).transform(d[d.index >= start].iloc[: N_TEST // 2 - 1])

    with pytest.raises(AssertionError, match="fuga"):
        assert_no_lookahead(cheat, data, cut)


# --- la moneda i no entra en su factor ---


@pytest.mark.parametrize("kind", list(MAKERS))
def test_own_return_does_not_enter_own_factor(kind: str) -> None:
    """Cambiar r_i(t) no cambia f_(-i)(t) (exacto) y si el de las demas monedas."""
    train, test = _panel(300, 5, seed=1), _panel(60, 5, seed=2, nan_frac=0.05)
    model = MAKERS[kind]().fit(train)
    base = model.loo_factors(test)  # type: ignore[attr-defined]
    row, i = 17, 2
    changed = test.copy()
    changed.iloc[row, i] = 0.25
    after = model.loo_factors(changed)  # type: ignore[attr-defined]
    col = test.columns[i]
    pd.testing.assert_series_equal(base[col], after[col], check_exact=True)
    others = [c for c in test.columns if c != col]
    assert (base.loc[:, others].iloc[row] != after.loc[:, others].iloc[row]).all()
    # y el resto de filas no cambia en ninguna columna
    mask = np.arange(len(test)) != row
    pd.testing.assert_frame_equal(base[mask], after[mask], check_exact=True)


# --- LooMeanFactor a mano ---


def test_loo_mean_by_hand_with_gaps_and_threshold() -> None:
    """f_(-i) = media de las demas presentes; NaN bajo min_valid_share sobre N-1 (inclusivo)."""
    train = _panel(200, 5, seed=3)
    model = LooMeanFactor(min_valid_share=0.5, min_periods=MIN_PERIODS).fit(train)
    nan = np.nan
    values = np.array(
        [
            [0.01, 0.02, 0.03, 0.04, 0.05],  # todas
            [nan, 0.02, 0.03, 0.04, 0.05],  # falta la 0
            [0.01, nan, nan, 0.04, 0.05],  # faltan 1 y 2
            [nan, nan, nan, 0.04, 0.05],  # faltan 0, 1, 2
            [nan, nan, nan, nan, 0.05],  # solo la 4
            [nan, nan, nan, nan, nan],  # ninguna
        ]
    )
    test = pd.DataFrame(values, index=train.index[:6], columns=train.columns)
    got = model.loo_factors(test)  # type: ignore[attr-defined]
    expected = np.full(values.shape, nan)
    for t in range(values.shape[0]):
        for i in range(5):
            others = np.delete(values[t], i)
            present = others[~np.isnan(others)]
            # N-1 = 4: 2 presentes = 0,5 (inclusivo, vale); 1 presente = 0,25 (NaN)
            if len(present) / 4 >= 0.5:
                expected[t, i] = present.mean()
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-12, atol=0, equal_nan=True)
    assert np.isfinite(got.iloc[3, 0]) and np.isnan(got.iloc[3, 4])  # 2 de 4 vale; 0 de 4 no
    assert np.isnan(got.iloc[4, 0])  # 1 de 4


def test_loo_mean_betas_and_residuals_by_hand() -> None:
    """Betas = OLS contra su factor LOO; residuo = r - beta * f_(-i)."""
    train = _panel(400, 4, seed=4, nan_frac=0.05)
    model = LooMeanFactor(min_periods=MIN_PERIODS).fit(train)
    loo = model.loo_factors(train)  # type: ignore[attr-defined]
    expected = pd.Series(
        {c: ols_betas(train[[c]], loo[c], min_periods=MIN_PERIODS).iloc[0] for c in train.columns}
    )
    pd.testing.assert_series_equal(model.betas, expected, check_names=False, rtol=1e-12)
    test = _panel(50, 4, seed=5, nan_frac=0.05)
    got = model.transform(test)
    by_hand = test - model.loo_factors(test) * model.betas  # type: ignore[attr-defined]
    pd.testing.assert_frame_equal(got, by_hand)
    # LOO a mano de una moneda, desde los retornos
    c = "S1"
    others = test.drop(columns=c)
    manual = others.mean(axis=1, skipna=True).where(others.notna().mean(axis=1) >= 0.5)
    np.testing.assert_allclose(
        model.loo_factors(test)[c].to_numpy(),
        manual.to_numpy(),
        rtol=1e-12,
        equal_nan=True,  # type: ignore[attr-defined]
    )


# --- LooPcaFactor a mano ---


def test_loo_pca_loadings_by_hand() -> None:
    """Cargas de i = sigma * autovector mayor de la correlacion sin i, media 1, positivas."""
    train = _panel(500, 5, seed=6)
    model = LooPcaFactor(min_periods=MIN_PERIODS).fit(train)
    loadings = model.loadings_matrix
    corr = train.corr().to_numpy()
    sigma = train.std().to_numpy()
    for k, symbol in enumerate(train.columns):
        keep = np.arange(5) != k
        _, vecs = np.linalg.eigh(corr[np.ix_(keep, keep)])
        raw = sigma[keep] * vecs[:, -1]
        want = raw / raw.mean()
        np.testing.assert_allclose(loadings.loc[keep, symbol].to_numpy(), want, rtol=1e-9)
        assert np.isnan(loadings.loc[symbol, symbol])  # diagonal NaN
        assert loadings.loc[keep, symbol].mean() == pytest.approx(1.0)
        assert (loadings.loc[keep, symbol] > 0).all()


def test_loo_pca_factor_is_cross_sectional_regression_on_present_loadings() -> None:
    """f_(-i)(t) = sum b_j r_j / sum b_j^2 sobre las demas presentes con carga."""
    train = _panel(500, 5, seed=6)
    model = LooPcaFactor(min_periods=MIN_PERIODS).fit(train)
    b = model.loadings_matrix.to_numpy()
    test = _panel(30, 5, seed=8, nan_frac=0.15)
    r = test.to_numpy()
    got = model.loo_factors(test).to_numpy()  # type: ignore[attr-defined]
    assert np.isnan(got).any() and np.isfinite(got).any()
    for t in range(r.shape[0]):
        for i in range(5):
            use = ~np.isnan(r[t]) & ~np.isnan(b[:, i])
            if use.sum() / (~np.isnan(b[:, i])).sum() >= model.min_valid_share and use.any():
                want = (b[use, i] * r[t, use]).sum() / (b[use, i] ** 2).sum()
                assert got[t, i] == pytest.approx(want, rel=1e-10, abs=1e-15)
            else:
                assert np.isnan(got[t, i])


def test_loo_pca_unusable_coin_has_nan_beta_and_no_loadings() -> None:
    """Una moneda con pocos retornos: beta NaN, sin cargas y fuera de los factores ajenos."""
    train = _panel(300, 5, seed=9)
    train.iloc[: 300 - (MIN_PERIODS - 1), 4] = np.nan  # 9 retornos < 10
    model = LooPcaFactor(min_periods=MIN_PERIODS).fit(train)
    assert np.isnan(model.betas["S4"]) and model.betas.drop("S4").notna().all()
    loadings = model.loadings_matrix
    assert loadings["S4"].isna().all() and loadings.loc["S4"].isna().all()
    assert loadings.drop(index="S4", columns="S4").notna().sum().sum() == 4 * 3


def test_loo_pca_fewer_than_three_usable_raises() -> None:
    """Con menos de 3 monedas utilizables, fit lanza ValueError."""
    train = _panel(300, 4, seed=10)
    train.iloc[:, 2:] = np.nan
    with pytest.raises(ValueError):
        LooPcaFactor(min_periods=MIN_PERIODS).fit(train)


# --- factors() ---


@pytest.mark.parametrize(
    ("loo", "std"),
    [
        (
            lambda: LooMeanFactor(min_periods=MIN_PERIODS),
            lambda: MeanFactor(min_periods=MIN_PERIODS),
        ),
        (lambda: LooPcaFactor(min_periods=MIN_PERIODS), lambda: PcaFactor(min_periods=MIN_PERIODS)),
    ],
)
def test_factors_equal_standard_market_factor(
    loo: Callable[[], FactorModel], std: Callable[[], FactorModel]
) -> None:
    """factors() es el del modelo estandar ajustado con el mismo train (D-34)."""
    train, test = _panel(300, 5, seed=1, nan_frac=0.03), _panel(60, 5, seed=2, nan_frac=0.03)
    a, b = loo().fit(train).factors(test), std().fit(train).factors(test)
    assert a.shape[1] == 1 and a.equals(b)


@pytest.mark.parametrize("kind", list(MAKERS))
def test_loo_factors_keep_input_column_order(kind: str) -> None:
    """loo_factors respeta el orden de columnas de la entrada, con o sin permutarlas."""
    train, test = _panel(300, 4, seed=1), _panel(40, 4, seed=2)
    model = MAKERS[kind]().fit(train)
    base = model.loo_factors(test)  # type: ignore[attr-defined]
    perm = list(test.columns[::-1])
    out = model.loo_factors(test[perm])  # type: ignore[attr-defined]
    assert list(out.columns) == perm
    pd.testing.assert_frame_equal(out[list(test.columns)], base)
    assert list(base.columns) == list(test.columns)


@pytest.mark.parametrize("kind", list(MAKERS))
def test_not_fitted_errors(kind: str) -> None:
    """Antes de fit, betas, loo_factors y (PCA) cargas lanzan NotFittedError."""
    model = MAKERS[kind]()
    test = _panel(20, 4, seed=2)
    with pytest.raises(NotFittedError):
        _ = model.betas  # type: ignore[attr-defined]
    with pytest.raises(NotFittedError):
        model.loo_factors(test)  # type: ignore[attr-defined]
    if kind == "loo_pca":
        with pytest.raises(NotFittedError):
            _ = model.loadings_matrix  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", [LooMeanFactor, LooPcaFactor])
def test_parameter_validation(cls: type[FactorModel]) -> None:
    """min_valid_share fuera de (0, 1], min_periods < 2 o una sola columna dan ValueError."""
    for share in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="min_valid_share"):
            cls(min_valid_share=share)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="min_periods"):
        cls(min_periods=1)  # type: ignore[call-arg]
    cls(min_valid_share=1.0, min_periods=2)  # type: ignore[call-arg]  # bordes validos
    with pytest.raises(ValueError):
        cls(min_periods=MIN_PERIODS).fit(_panel(100, 1, seed=1))  # type: ignore[call-arg]


# --- resultado clave del ticket ---


def test_equal_betas_standard_residuals_sum_to_zero_but_loo_do_not() -> None:
    """Con betas iguales, el estandar da suma residual 0 y autovalor minimo 0; el LOO no."""
    from miax.eval.loo_factor import _model_metrics

    rng = np.random.default_rng(215)
    n, t, k = 20, 6000, 20
    cols = [f"S{i:02d}USDT" for i in range(k)]
    idx = pd.date_range("2024-01-01", periods=2 * t, freq="min", tz="UTC")
    data = pd.DataFrame(
        rng.normal(0, 1, (2 * t, 1)) + rng.normal(0, 1, (2 * t, k)), index=idx, columns=cols
    )
    train, test = data.iloc[:t], data.iloc[t:]
    got: dict[str, dict[str, float]] = {}
    makers = {
        "mean": lambda: MeanFactor(min_periods=100),
        "pca": lambda: PcaFactor(min_periods=100),
        "loo_mean": lambda: LooMeanFactor(min_periods=100),
        "loo_pca": lambda: LooPcaFactor(min_periods=100),
    }
    for name, make in makers.items():
        got[name], _ = _model_metrics(make().fit(train), test, min_obs=100)
    for name in ("mean",):  # el PCA solo lo cumple aproximadamente (cargas no exactamente iguales)
        assert abs(got[name]["min_eigenvalue"]) < 1e-8
        assert abs(got[name]["sum_ratio"]) < 1e-6
    for name in ("loo_mean", "loo_pca"):
        assert got[name]["min_eigenvalue"] > 0.05
        assert got[name]["sum_ratio"] > 0.1
    assert got["pca"]["sum_ratio"] < 0.5 * got["loo_pca"]["sum_ratio"]
    for name, m in got.items():
        assert m["pair_mean"] == pytest.approx(-1 / (n - 1), abs=0.01), name
        assert m["floor"] == pytest.approx(-1 / (n - 1))


# --- ols_betas_paired ---


def test_ols_betas_paired_matches_columnwise_ols_betas() -> None:
    """Un factor por columna da lo mismo que ols_betas columna a columna."""
    returns = _panel(300, 4, seed=1, nan_frac=0.05)
    factors = _panel(300, 4, seed=2, nan_frac=0.05)
    got = ols_betas_paired(returns, factors, min_periods=20)
    for c in returns.columns:
        want = ols_betas(returns[[c]], factors[c], min_periods=20).iloc[0]
        assert got[c] == pytest.approx(want, rel=1e-12)
    assert list(got.index) == list(returns.columns)


def test_ols_betas_paired_validation_and_nan_cases() -> None:
    """Indice o columnas distintas dan ValueError; min_periods y factor constante dan NaN."""
    returns = _panel(100, 3, seed=1)
    factors = _panel(100, 3, seed=2)
    with pytest.raises(ValueError):
        ols_betas_paired(returns, factors.iloc[1:])
    with pytest.raises(ValueError):
        ols_betas_paired(returns, factors[["S2", "S1", "S0"]])  # mismo conjunto, otro orden
    with pytest.raises(ValueError):
        ols_betas_paired(returns, factors.rename(columns={"S0": "X"}))
    with pytest.raises(ValueError, match="min_periods"):
        ols_betas_paired(returns, factors, min_periods=1)
    assert ols_betas_paired(returns, factors, min_periods=101).isna().all()
    flat = factors.copy()
    flat["S1"] = 0.003
    out = ols_betas_paired(returns, flat, min_periods=10)
    assert np.isnan(out["S1"]) and out[["S0", "S2"]].notna().all()
