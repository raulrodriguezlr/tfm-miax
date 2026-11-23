"""Tests del estudio de betas (MIAX-053): ventana movil, no-fuga, sesgo de Epps, metricas y CLI."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import miax.eval.__main__ as cli
from miax.eval import (
    ResidualFit,
    StudyResult,
    beta_horizon_bias,
    compare_schemes,
    fold_windows,
    project_fixed,
    residual_diagnostics,
    rolling_residuals,
    write_tables,
)
from miax.eval import beta_study as study_module
from miax.features.factor import FactorModel, ols_betas
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

DAY = 1440
MIN_MS = 60_000
SYMBOLS = ("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT")


def _returns(
    days: int, *, seed: int = 0, start: str = "2024-01-01", symbols: tuple[str, ...] = SYMBOLS
) -> pd.DataFrame:
    """Retornos de 1 minuto de un factor (la primera columna) mas ruido, con semilla fija."""
    rng = np.random.default_rng(seed)
    n = days * DAY
    index = pd.date_range(start, periods=n, freq="min", tz="UTC")
    factor = rng.normal(0.0, 0.001, n)
    data = {symbols[0]: factor}
    for k, symbol in enumerate(symbols[1:], start=1):
        data[symbol] = (0.5 + 0.3 * k) * factor + rng.normal(0.0, 0.001, n)
    return pd.DataFrame(data, index=index, dtype="float64")


def _btc() -> FactorModel:
    return BtcFactor(min_periods=100)


def _mean() -> FactorModel:
    return MeanFactor(min_periods=100)


def _pca() -> FactorModel:
    return PcaFactor(min_periods=100)


MAKERS: dict[str, Callable[[], FactorModel]] = {"btc": _btc, "mean": _mean, "pca": _pca}


# --- fold_windows ---


def test_fold_windows_train_ends_where_test_starts() -> None:
    """Train son 12 meses, acaba justo donde empieza test, y test es un trimestre."""
    start = pd.Timestamp("2024-04-01", tz="UTC")
    train_start, test_start, test_end = fold_windows(start)
    assert train_start == pd.Timestamp("2023-04-01", tz="UTC")
    assert test_start == start
    assert test_end == pd.Timestamp("2024-07-01", tz="UTC")


@pytest.mark.parametrize("bad", ["2024-02-01", "2024-04-02", "2024-04-01 01:00"])
def test_fold_windows_rejects_non_quarter_start(bad: str) -> None:
    """Un instante que no es el inicio de un trimestre da `ValueError`."""
    with pytest.raises(ValueError, match="inicio de un trimestre"):
        fold_windows(pd.Timestamp(bad, tz="UTC"))


def test_fold_windows_rejects_naive_timestamp() -> None:
    """Sin zona horaria UTC no hay fold."""
    with pytest.raises(ValueError, match="UTC"):
        fold_windows(pd.Timestamp("2024-04-01"))


# --- ventana movil exacta y no-fuga ---


class _Recorder(BtcFactor):
    """BtcFactor que apunta las filas con las que se ajusta y se proyecta."""

    log: list[tuple[pd.Timestamp, pd.Timestamp, int, pd.Timestamp]]

    def fit(self, returns: pd.DataFrame) -> _Recorder:
        """Anota el ultimo minuto y el numero de filas del ajuste."""
        self._fit_info = (returns.index.min(), returns.index.max(), len(returns))
        return super().fit(returns)  # type: ignore[return-value]

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Anota el primer minuto proyectado junto con lo ajustado."""
        _Recorder.log.append((*self._fit_info, returns.index.min()))  # type: ignore[arg-type]
        return super().transform(returns)


def test_rolling_matches_manual_fit_and_transform() -> None:
    """Con un bloque, coincide con `modelo.fit(ventana).transform(bloque)` hecho a mano."""
    returns = _returns(5)
    start, end = returns.index[3 * DAY], returns.index[4 * DAY]
    fit = rolling_residuals(_btc, returns, start, end, window_days=2, step_days=1)
    window = returns.loc[start - pd.Timedelta(days=2) : start - pd.Timedelta(minutes=1)]
    block = returns.loc[start : end - pd.Timedelta(minutes=1)]
    model = _btc().fit(window)
    pd.testing.assert_frame_equal(fit.residuals, model.transform(block))
    pd.testing.assert_series_equal(fit.factor, model.factors(block).iloc[:, 0], check_names=False)
    assert (fit.n_blocks, fit.n_failed) == (1, 0)


def test_rolling_window_ends_right_before_each_block() -> None:
    """La ventana tiene exactamente `window_days` y acaba un minuto antes del bloque."""
    returns = _returns(6)
    start, end = returns.index[3 * DAY], returns.index[6 * DAY - 1]
    _Recorder.log = []
    rolling_residuals(_Recorder, returns, start, end, window_days=2, step_days=1)
    assert len(_Recorder.log) == 3
    for fit_first, fit_last, n_rows, block_first in _Recorder.log:
        assert n_rows == 2 * DAY
        assert fit_last == block_first - pd.Timedelta(minutes=1)
        assert fit_first == block_first - pd.Timedelta(days=2)


def test_rolling_never_uses_current_or_future_returns() -> None:
    """Cambiar los retornos de un bloque y posteriores no altera los residuos de los anteriores."""
    returns = _returns(6)
    start, end = returns.index[3 * DAY], returns.index[6 * DAY - 1]
    base = rolling_residuals(_btc, returns, start, end, window_days=2, step_days=1)
    changed = returns.copy()
    second_block = returns.index[4 * DAY]
    rng = np.random.default_rng(99)
    tail = changed.index >= second_block
    changed.loc[tail] = rng.normal(0.0, 0.01, size=changed.loc[tail].shape)
    other = rolling_residuals(_btc, changed, start, end, window_days=2, step_days=1)
    before = base.residuals.index < second_block
    pd.testing.assert_frame_equal(base.residuals[before], other.residuals[before])
    pd.testing.assert_series_equal(base.factor[before], other.factor[before])
    assert not base.residuals[~before].equals(other.residuals[~before])


def test_rolling_block_depends_only_on_the_past_and_itself() -> None:
    """Un bloque da igual calculado en el test completo que empezando en su propio inicio."""
    returns = _returns(6)
    start, end = returns.index[3 * DAY], returns.index[6 * DAY - 1]
    full = rolling_residuals(_mean, returns, start, end, window_days=2, step_days=1)
    later = returns.index[4 * DAY]
    part = rolling_residuals(_mean, returns, later, end, window_days=2, step_days=1)
    pd.testing.assert_frame_equal(full.residuals.loc[later:], part.residuals)


def test_rolling_excludes_sparse_coins_in_window() -> None:
    """Una moneda con menos de `min_valid_share` en la ventana queda en `NaN`, no en 0."""
    returns = _returns(5)
    rng = np.random.default_rng(3)
    sparse = rng.random(len(returns)) < 0.8
    returns.loc[sparse, "CCCUSDT"] = np.nan
    start, end = returns.index[3 * DAY], returns.index[4 * DAY]
    fit = rolling_residuals(
        _btc, returns, start, end, window_days=2, step_days=1, min_valid_share=0.5
    )
    assert fit.residuals["CCCUSDT"].isna().all()
    assert fit.residuals["AAAUSDT"].notna().any()
    assert fit.n_failed == 0


def test_rolling_failed_block_is_nan_and_counted() -> None:
    """Un bloque sin ventana previa no ajusta: residuo `NaN` (nunca 0) y contado."""
    returns = _returns(3)
    start, end = returns.index[0], returns.index[2 * DAY]
    fit = rolling_residuals(_btc, returns, start, end, window_days=1, step_days=1)
    assert fit.n_blocks == 2
    assert fit.n_failed == 1  # el primer bloque no tiene datos antes
    first = fit.residuals.index < returns.index[DAY]
    assert fit.residuals[first].isna().all().all()
    assert fit.factor[first].isna().all()
    assert fit.residuals[~first].drop(columns="BTCUSDT").notna().all().all()
    assert not (fit.residuals.to_numpy() == 0.0).any()


def test_rolling_rejects_bad_parameters() -> None:
    """Parametros fuera de rango dan `ValueError`."""
    returns = _returns(3)
    start, end = returns.index[DAY], returns.index[2 * DAY]
    with pytest.raises(ValueError, match="window_days"):
        rolling_residuals(_btc, returns, start, end, window_days=0)
    with pytest.raises(ValueError, match="min_valid_share"):
        rolling_residuals(_btc, returns, start, end, window_days=1, min_valid_share=0.0)
    with pytest.raises(ValueError, match="vacio"):
        rolling_residuals(_btc, returns, end, start, window_days=1)


# --- horizonte ---


def _aggregate(returns: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Suma retornos log en barras no solapadas de `minutes` minutos."""
    values = returns.to_numpy().reshape(-1, minutes, returns.shape[1]).sum(axis=1)
    return pd.DataFrame(values, index=returns.index[::minutes], columns=returns.columns)


def test_horizon_fit_on_5m_projects_on_1m() -> None:
    """`fit(5m).transform(1m)` usa las betas de 5 min y el factor de 1 min."""
    returns = _returns(4)
    train, test = returns.iloc[: 3 * DAY], returns.iloc[3 * DAY :]
    train5 = _aggregate(train, 5)
    fit = project_fixed(_btc, train5, test)
    beta5 = ols_betas(train5, train5["BTCUSDT"], min_periods=100)
    expected = test.to_numpy() - test[["BTCUSDT"]].to_numpy() * beta5.to_numpy()[None, :]
    expected[:, 0] = np.nan  # el residuo del propio factor queda vacio
    np.testing.assert_allclose(fit.residuals.to_numpy(), expected, equal_nan=True)
    pd.testing.assert_series_equal(fit.factor, test["BTCUSDT"].rename("btc"), check_names=False)
    assert fit.betas is not None and "BTCUSDT" not in fit.betas.index
    pd.testing.assert_series_equal(fit.betas, beta5.drop("BTCUSDT"))
    # Con retornos de 1 min las betas serian otras: el horizonte importa.
    other = project_fixed(_btc, train, test)
    assert not np.allclose(other.residuals.to_numpy()[:, 1:], fit.residuals.to_numpy()[:, 1:])


def test_project_fixed_failure_is_nan_and_counted() -> None:
    """Un ajuste que falla (PCA sin pares con solape) deja el residuo en `NaN` y lo cuenta."""
    returns = _returns(2)
    train = returns.iloc[:DAY].copy()
    train.iloc[: DAY // 2, 1] = np.nan  # AAA y BBB solo comparten... nada en la 1.a mitad
    train.iloc[DAY // 2 :, 2] = np.nan
    train[["CCCUSDT", "BTCUSDT"]] = np.nan
    test = returns.iloc[DAY:]
    fit = project_fixed(lambda: PcaFactor(min_periods=100), train, test)
    assert (fit.n_blocks, fit.n_failed) == (1, 1)
    assert fit.residuals.isna().all().all() and fit.factor.isna().all()
    assert fit.betas is None


# --- sesgo de Epps ---


def _epps_returns(shares: list[float], seed: int = 7) -> pd.DataFrame:
    """Factor y monedas; `share` es la parte del retorno que llega un minuto tarde."""
    rng = np.random.default_rng(seed)
    n = 60_000
    index = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    factor = rng.normal(0.0, 0.001, n)
    late = np.concatenate([[0.0], factor[:-1]])
    data = {"BTCUSDT": factor}
    for k, share in enumerate(shares):
        data[f"C{k}USDT"] = (1 - share) * factor + share * late + rng.normal(0.0, 0.0005, n)
    return pd.DataFrame(data, index=index, dtype="float64")


def test_epps_beta_at_1m_is_biased_down_and_more_for_illiquid() -> None:
    """Con precios obsoletos la beta a 1 min sale menor que a 5 min, y mas cuanto mas iliquida."""
    shares = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    returns = _epps_returns(shares)
    betas = {
        "1m": _btc().fit(returns).betas.drop("BTCUSDT"),
        "5m": _btc().fit(_aggregate(returns, 5)).betas.drop("BTCUSDT"),
    }
    illiquidity = pd.Series(shares, index=betas["1m"].index)
    table = beta_horizon_bias(betas, illiquidity)
    row = table.iloc[0]
    assert row["horizon"] == "5m" and row["n_coins"] == 6
    ratios = betas["5m"] / betas["1m"]
    assert (ratios.iloc[1:] > 1.0).all()
    assert ratios.is_monotonic_increasing
    assert row["median_ratio_illiquid"] > row["median_ratio_mid"] > row["median_ratio_liquid"]
    assert row["median_ratio_liquid"] == pytest.approx(ratios.iloc[:2].median())
    assert row["spearman"] == pytest.approx(1.0)
    assert row["median_ratio"] == pytest.approx(ratios.median())


def test_bias_ignores_non_positive_base_beta_and_needs_base() -> None:
    """Una beta base <= 0 o NaN no da cociente; sin horizonte base es un error."""
    base = pd.Series({"A": 1.0, "B": -0.5, "C": np.nan, "D": 0.5, "E": 2.0})
    other = pd.Series({"A": 2.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 2.0})
    illiq = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5})
    row = beta_horizon_bias({"1m": base, "5m": other}, illiq).iloc[0]
    assert row["n_coins"] == 3  # A, D, E
    assert row["median_ratio"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="base"):
        beta_horizon_bias({"5m": other}, illiq)


def _tied_bias_inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Betas base, betas a 5m e iliquidez con 6 empates sobre 12 monedas."""
    rng = np.random.default_rng(3)
    symbols = [f"C{i:02d}" for i in range(12)]
    base = pd.Series(rng.uniform(0.5, 1.5, 12), index=symbols)
    other = base * pd.Series(rng.uniform(0.8, 1.6, 12), index=symbols)
    illiq = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], index=symbols)
    return base, other, illiq


def test_bias_with_tied_illiquidity_is_independent_of_column_order() -> None:
    """Con empates de iliquidez, Spearman y medianas por tercil no dependen del orden."""
    base, other, illiq = _tied_bias_inputs()
    expected = beta_horizon_bias({"1m": base, "5m": other}, illiq)
    rng = np.random.default_rng(11)
    for _ in range(8):
        order = list(rng.permutation(base.index))
        got = beta_horizon_bias({"1m": base.loc[order], "5m": other.loc[order]}, illiq.loc[order])
        pd.testing.assert_frame_equal(got, expected)


def test_bias_spearman_uses_average_ranks_manually() -> None:
    """El Spearman coincide con el Pearson de rangos promediados calculado con numpy."""
    base, other, illiq = _tied_bias_inputs()
    ratio = (other / base).to_numpy()
    x = illiq.to_numpy()

    def avg_rank(v: np.ndarray) -> np.ndarray:
        return np.array([(v < a).sum() + ((v == a).sum() + 1) / 2 for a in v])

    manual = np.corrcoef(avg_rank(x), avg_rank(ratio))[0, 1]
    row = beta_horizon_bias({"1m": base, "5m": other}, illiq).iloc[0]
    assert row["spearman"] == pytest.approx(manual)


def test_bias_terciles_break_ties_alphabetically() -> None:
    """A igualdad de iliquidez, el tercil se asigna por simbolo alfabetico."""
    symbols = ["D", "C", "B", "A", "F", "E"]
    base = pd.Series(1.0, index=symbols)
    other = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}).loc[symbols]
    illiq = pd.Series(0.5, index=symbols)  # todo empatado: orden A..F
    row = beta_horizon_bias({"1m": base, "5m": other}, illiq).iloc[0]
    assert row["median_ratio_liquid"] == pytest.approx(1.5)  # A, B
    assert row["median_ratio_mid"] == pytest.approx(3.5)  # C, D
    assert row["median_ratio_illiquid"] == pytest.approx(5.5)  # E, F


def test_bias_with_few_coins_has_no_spearman() -> None:
    """Con menos de 3 monedas no hay Spearman ni terciles, pero si la mediana."""
    base = pd.Series({"A": 1.0, "B": 1.0})
    row = beta_horizon_bias({"1m": base, "5m": base * 2}, pd.Series({"A": 0.1, "B": 0.2})).iloc[0]
    assert np.isnan(row["spearman"]) and np.isnan(row["median_ratio_liquid"])
    assert row["median_ratio"] == pytest.approx(2.0)


# --- metricas ---


def test_diagnostics_match_manual_numpy() -> None:
    """`var_removed`, `corr_lag0` y `corr_lag1` coinciden con un calculo manual con numpy."""
    rng = np.random.default_rng(5)
    n = 400
    index = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    f = rng.normal(size=n)
    r = 0.7 * f + rng.normal(size=n)
    e = r - 0.7 * np.concatenate([[0.0], f[:-1]]) * 0.5 + rng.normal(size=n) * 0.1
    r[[10, 50]] = np.nan
    e[[10, 77]] = np.nan
    returns = pd.DataFrame({"X": r}, index=index)
    residuals = pd.DataFrame({"X": e}, index=index)
    factor = pd.Series(f, index=index)
    table = residual_diagnostics(returns, residuals, factor, min_obs=10)
    ok = ~np.isnan(r) & ~np.isnan(e)
    assert table.loc["X", "var_removed"] == pytest.approx(
        1 - (e[ok] ** 2).sum() / (r[ok] ** 2).sum()
    )
    assert table.loc["X", "corr_lag0"] == pytest.approx(abs(np.corrcoef(e[ok], f[ok])[0, 1]))
    f_prev = np.concatenate([[np.nan], f[:-1]])
    ok1 = ~np.isnan(e) & ~np.isnan(f_prev)
    lag1 = np.corrcoef(e[ok1], f_prev[ok1])[0, 1]
    assert table.loc["X", "corr_lag1"] == pytest.approx(lag1)
    assert table.loc["X", "abs_corr_lag1"] == pytest.approx(abs(lag1))
    assert table.loc["X", "n_obs"] == ok.sum()


def test_diagnostics_pure_factor_and_lagged_residual() -> None:
    """Un residuo sin factor da `corr_lag0 ~ 0`; uno que conserva f(t-1) da `corr_lag1` alta."""
    rng = np.random.default_rng(6)
    n = 5000
    index = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    f = pd.Series(rng.normal(size=n), index=index)
    noise = rng.normal(size=n)
    returns = pd.DataFrame({"X": f + noise}, index=index)
    clean = pd.DataFrame({"X": noise}, index=index)
    lagged = pd.DataFrame({"X": f.shift(1).fillna(0.0).to_numpy() + 0.1 * noise}, index=index)
    t_clean = residual_diagnostics(returns, clean, f, min_obs=10)
    t_lagged = residual_diagnostics(returns, lagged, f, min_obs=10)
    assert t_clean.loc["X", "corr_lag0"] < 0.05
    assert t_clean.loc["X", "abs_corr_lag1"] < 0.05
    assert t_clean.loc["X", "var_removed"] == pytest.approx(0.5, abs=0.05)
    assert t_lagged.loc["X", "corr_lag1"] > 0.9
    assert t_lagged.loc["X", "corr_lag0"] < 0.05


def test_diagnostics_nan_residual_gives_nan_not_zero() -> None:
    """Una moneda sin residuo (el factor BTC) da `NaN`, y menos de `min_obs` tambien."""
    returns = _returns(1)
    fit = project_fixed(_btc, returns.iloc[: DAY // 2], returns.iloc[DAY // 2 :])
    table = residual_diagnostics(returns.iloc[DAY // 2 :], fit.residuals, fit.factor, min_obs=10)
    assert table.loc["BTCUSDT", ["var_removed", "corr_lag0", "corr_lag1"]].isna().all()
    assert table.loc["AAAUSDT", "var_removed"] > 0
    few = residual_diagnostics(returns.iloc[DAY // 2 :], fit.residuals, fit.factor, min_obs=10_000)
    assert few.drop(columns="n_obs").isna().all().all()


def test_diagnostics_require_same_index() -> None:
    """Entradas con indices distintos dan `ValueError`."""
    returns = _returns(1)
    with pytest.raises(ValueError, match="indice"):
        residual_diagnostics(returns, returns.iloc[1:], returns["BTCUSDT"])


# --- decision ---


def _diag(wins: int, folds: int = 12, question: str = "A") -> pd.DataFrame:
    """Diagnosticos de un solo factor: el esquema alternativo gana en `wins` folds."""
    rows = []
    for k in range(folds):
        win = k < wins
        rows.append((f"f{k:02d}", "fold", "btc", 0.5, 0.2, 0.1))
        better = (0.6, 0.1, 0.05) if win else (0.4, 0.3, 0.2)
        rows.append((f"f{k:02d}", "roll30" if question == "A" else "h5m", "btc", *better))
    frame = pd.DataFrame(
        rows, columns=["fold", "scheme", "factor", "var_removed", "corr_lag0", "abs_corr_lag1"]
    )
    return frame


@pytest.mark.parametrize(("wins", "passes"), [(9, True), (8, False), (12, True), (10, True)])
def test_decision_threshold_is_inclusive(wins: int, passes: bool) -> None:
    """9 de 12 folds ya cuenta (`>=`); 8 no."""
    table = study_module._decision_table(_diag(wins), pd.DataFrame(columns=["factor"]), min_wins=9)
    row = table[table["factor"] == "btc"].iloc[0]
    assert (row["n_wins"], row["n_folds"], bool(row["passes"])) == (wins, 12, passes)
    overall = table[table["factor"] == "all"].iloc[0]
    assert bool(overall["passes"]) is passes


def test_decision_a_needs_both_metrics_and_b_uses_lag1() -> None:
    """A exige mas varianza quitada Y menos corr_lag0; B solo mira `abs_corr_lag1`."""
    diag = _diag(12)
    diag.loc[diag["scheme"] == "roll30", "corr_lag0"] = 0.9  # peor en corr_lag0
    table = study_module._decision_table(diag, pd.DataFrame(columns=["factor"]), min_wins=9)
    assert table[table["factor"] == "btc"].iloc[0]["n_wins"] == 0
    diag_b = _diag(10, question="B")
    table_b = study_module._decision_table(diag_b, pd.DataFrame(columns=["factor"]), min_wins=9)
    assert table_b[table_b["factor"] == "btc"].iloc[0]["n_wins"] == 10


def test_decision_all_requires_every_factor() -> None:
    """El esquema pasa solo si pasa para los tres factores."""
    good, bad = _diag(12), _diag(5)
    bad["factor"] = "mean"
    diag = pd.concat([good, bad], ignore_index=True)
    table = study_module._decision_table(diag, pd.DataFrame(columns=["factor"]), min_wins=9)
    assert bool(table[table["factor"] == "btc"].iloc[0]["passes"])
    assert not bool(table[table["factor"] == "mean"].iloc[0]["passes"])
    assert not bool(table[table["factor"] == "all"].iloc[0]["passes"])


def test_decision_nan_is_not_a_win() -> None:
    """Un fold con metricas `NaN` no cuenta como victoria."""
    diag = _diag(12)
    diag.loc[(diag["scheme"] == "roll30") & (diag["fold"] == "f00"), "var_removed"] = np.nan
    table = study_module._decision_table(diag, pd.DataFrame(columns=["factor"]), min_wins=9)
    assert table[table["factor"] == "btc"].iloc[0]["n_wins"] == 11


# --- punta a punta con cache y universo sinteticos ---


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def _save(
    base_dir: Path,
    symbol: str,
    start: str,
    close: np.ndarray,
    step: int = 1,
    drop: np.ndarray | None = None,
) -> None:
    """Guarda velas de 1 minuto de `symbol` cada `step` minutos desde `start`, sin las `drop`."""
    rows = np.arange(0, len(close), step)
    if drop is not None:
        rows = rows[~drop[rows]]
    open_times = _ms(start) + rows * MIN_MS
    n = len(rows)
    frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": close[rows],
            "high": close[rows],
            "low": close[rows],
            "close": close[rows],
            "volume": [1.0] * n,
            "close_time": open_times + MIN_MS - 1,
            "quote_asset_volume": [100.0] * n,
            "num_trades": [5] * n,
            "taker_buy_base": [0.5] * n,
            "taker_buy_quote": [5.0] * n,
            "ignore": [0.0] * n,
        },
        columns=KLINE_COLUMNS,
    ).astype(KLINE_DTYPES)
    save_klines(frame, symbol, "1m", base_dir=base_dir)


def _world(base_dir: Path) -> pd.DataFrame:
    """Cache de 2023-12-01 a 2024-02-01 con 3 monedas buenas y una casi vacia, y su universo."""
    n = 62 * DAY
    rng = np.random.default_rng(11)
    factor = rng.normal(0.0, 0.001, n)
    symbols = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "SPARSEUSDT"]
    for k, symbol in enumerate(symbols):
        returns = (0.6 + 0.2 * k) * factor + rng.normal(0.0, 0.001, n)
        close = 100.0 * np.exp(np.cumsum(returns))
        _save(base_dir, symbol, "2023-12-01", close, step=5 if symbol == "SPARSEUSDT" else 1)
    rows = [("2024-01-01", "2024-02-01", s, "l1") for s in symbols]
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    return universe


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    """Cache de velas y universo sinteticos, compartidos por los tests punta a punta."""
    base = tmp_path_factory.mktemp("klines")
    return base, _world(base)


def _compare(world: tuple[Path, pd.DataFrame], cache_dir: Path, **kwargs: object) -> StudyResult:
    base, universe = world
    params: dict[str, object] = {
        "universe": universe,
        "factors": MAKERS,
        "windows": (7,),
        "horizons": ("1m", "5m", "15m"),
        "train_months": 1,
        "test_months": 1,
        "min_obs": 100,
        "min_wins": 1,
        "base_dir": base,
        "cache_dir": cache_dir,
    }
    params.update(kwargs)
    return compare_schemes([pd.Timestamp("2024-01-01", tz="UTC")], **params)  # type: ignore[arg-type]


def test_end_to_end_tables(world: tuple[Path, pd.DataFrame], tmp_path: Path) -> None:
    """Un fold, tres factores y cuatro esquemas; la moneda casi vacia queda fuera (D-37)."""
    result = _compare(world, tmp_path / "panels")
    diag = result.diagnostics
    assert list(diag["scheme"].unique()) == ["fold", "roll7", "h5m", "h15m"]
    assert list(diag["factor"].unique()) == ["btc", "mean", "pca"]
    assert len(diag) == 12 and (diag["fold"] == "2024-01-01").all()
    assert (diag["n_universe"] == 4).all()
    btc = diag[diag["factor"] == "btc"]
    # BTC no tiene residuo y SPARSE esta fuera: solo AAA y BBB cuentan con el factor BTC.
    assert (btc["n_coins"] == 2).all()
    assert (diag[diag["factor"] == "mean"]["n_coins"] == 3).all()
    assert (diag["n_failed"] == 0).all()
    assert diag[["var_removed", "corr_lag0", "abs_corr_lag1"]].notna().all().all()
    assert (diag[diag["scheme"] == "roll7"]["n_blocks"] == 5).all()
    bias = result.bias
    assert set(bias["factor"]) == {"btc", "mean"}  # el PCA no tiene beta OLS
    assert set(bias["horizon"]) == {"5m", "15m"}
    assert len(result.decision) == 3 * 4 and set(result.decision["factor"]) == {
        "btc",
        "mean",
        "pca",
        "all",
    }


def test_end_to_end_is_deterministic_byte_for_byte(
    world: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    """Mismas entradas, mismos CSV byte a byte, y el orden de filas y columnas es fijo."""
    first = write_tables(_compare(world, tmp_path / "p1"), out_dir=tmp_path / "a")
    second = write_tables(_compare(world, tmp_path / "p2"), out_dir=tmp_path / "b")
    assert [p.name for p in first] == [
        "beta_study_diagnostics.csv",
        "beta_study_bias.csv",
        "beta_study_decision.csv",
    ]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()
        assert b"\r" not in a.read_bytes()
    header = first[0].read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(study_module.DIAGNOSTIC_COLUMNS)


def test_end_to_end_unknown_quarter_raises(
    world: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    """Un trimestre que no esta en el universo da `ValueError`."""
    base, universe = world
    with pytest.raises(ValueError, match="no esta en el universo"):
        compare_schemes(
            [pd.Timestamp("2024-04-01", tz="UTC")],
            universe=universe,
            train_months=1,
            test_months=1,
            base_dir=base,
            cache_dir=tmp_path,
        )


def test_failed_fit_is_reported_in_the_table(
    world: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    """Un ajuste que falla (PCA) o betas `NaN` (BTC) dejan el residuo en `NaN`, contado."""
    makers: dict[str, Callable[[], FactorModel]] = {
        "btc": partial(BtcFactor, min_periods=10**7),
        "pca": partial(PcaFactor, min_periods=10**7),
    }
    result = _compare(world, tmp_path / "panels", factors=makers, horizons=("1m",))
    diag = result.diagnostics
    pca = diag[(diag["scheme"] == "fold") & (diag["factor"] == "pca")].iloc[0]
    assert pca["n_failed"] == 1 and pca["n_coins"] == 0 and np.isnan(pca["var_removed"])
    roll = diag[(diag["scheme"] == "roll7") & (diag["factor"] == "pca")].iloc[0]
    assert roll["n_failed"] == roll["n_blocks"] == 5
    btc = diag[(diag["scheme"] == "fold") & (diag["factor"] == "btc")].iloc[0]
    assert btc["n_failed"] == 0 and btc["n_coins"] == 0  # betas NaN: ninguna moneda con metrica
    assert np.isnan(btc["var_removed"])


def test_compare_rejects_bad_parameters(world: tuple[Path, pd.DataFrame], tmp_path: Path) -> None:
    """Ventanas o horizontes invalidos dan `ValueError` antes de leer nada."""
    with pytest.raises(ValueError, match="ventanas"):
        _compare(world, tmp_path, windows=(0,))
    with pytest.raises(ValueError):
        _compare(world, tmp_path, horizons=("7x",))


# --- cableado de datos de train y test (no-fuga) ---


@pytest.fixture
def log_records(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Hace visible en `caplog` el logger del estudio, que no propaga al raiz."""
    study_module.logger.addHandler(caplog.handler)
    caplog.set_level("DEBUG", logger=study_module.logger.name)
    yield caplog  # type: ignore[misc]
    study_module.logger.removeHandler(caplog.handler)


def _split_world(base_dir: Path, *, train_seed: int = 11, test_seed: int = 21) -> pd.DataFrame:
    """Cache 2023-12-01 a 2024-02-01 con train y test de cargas distintas, y una moneda fina.

    `THINUSDT` tiene el 60 % de minutos sin vela en train (40 % valido, queda fuera del fold)
    y todos en test; train y test salen de semillas independientes.
    """
    n = 31 * DAY
    train_rng = np.random.default_rng(train_seed)
    test_rng = np.random.default_rng(test_seed)
    f_train, f_test = train_rng.normal(0.0, 0.001, n), test_rng.normal(0.0, 0.001, n)
    thin_drop = np.concatenate([np.random.default_rng(13).random(n) < 0.6, np.zeros(n, dtype=bool)])
    symbols = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "THINUSDT"]
    for k, symbol in enumerate(symbols):
        train = (0.6 + 0.2 * k) * f_train + train_rng.normal(0.0, 0.001, n)
        test = (1.6 - 0.3 * k) * f_test + test_rng.normal(0.0, 0.001, n)
        close = 100.0 * np.exp(np.cumsum(np.concatenate([train, test])))
        _save(base_dir, symbol, "2023-12-01", close, drop=thin_drop if k == 3 else None)
    rows = [("2024-01-01", "2024-02-01", s, "l1") for s in symbols]
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    return universe


def _compare_split(tmp_path: Path, name: str, **world: int) -> StudyResult:
    """Corre el estudio sobre un mundo sintetico con train y test independientes."""
    base = tmp_path / f"klines_{name}"
    universe = _split_world(base, **world)
    return compare_schemes(
        [pd.Timestamp("2024-01-01", tz="UTC")],
        universe=universe,
        factors=MAKERS,
        windows=(7,),
        horizons=("1m", "5m", "15m"),
        train_months=1,
        test_months=1,
        min_obs=100,
        min_wins=1,
        base_dir=base,
        cache_dir=tmp_path / f"panels_{name}",
    )


def test_end_to_end_betas_and_bias_depend_only_on_train(tmp_path: Path) -> None:
    """Cambiar solo los datos de test no mueve la tabla de sesgo (betas de fold, 5m y 15m)."""
    base = _compare_split(tmp_path, "a")
    other_test = _compare_split(tmp_path, "b", test_seed=22)
    other_train = _compare_split(tmp_path, "c", train_seed=12)
    # Las betas (y con ellas el sesgo) salen de train: igual con otro test, distinto con otro train.
    pd.testing.assert_frame_equal(base.bias, other_test.bias)
    assert not base.bias.equals(other_train.bias)
    # Las metricas de test, en cambio, si reflejan el test (el contraste no es trivial).
    fold = base.diagnostics["scheme"] == "fold"
    assert not base.diagnostics[fold].equals(other_test.diagnostics[fold])
    # 5m y 15m no coinciden con 1m (si no, ajustarian con retornos de 1 min): cociente != 1.
    assert ((base.bias["median_ratio"] - 1.0).abs() > 1e-6).all()
    assert set(base.bias["horizon"]) == {"5m", "15m"}


def test_end_to_end_train_filter_ignores_test_coverage(tmp_path: Path) -> None:
    """La moneda con 40 % de train valido queda fuera aunque en test este completa."""
    diag = _compare_split(tmp_path, "thin").diagnostics
    # mean: BTC, AAA y BBB; btc: AAA y BBB (el factor no tiene residuo). THIN fuera en todo.
    assert (diag[diag["factor"] == "mean"]["n_coins"] == 3).all()
    assert (diag[diag["factor"] == "btc"]["n_coins"] == 2).all()
    assert (diag["n_universe"] == 4).all()


def _direct_inputs(
    train_seed: int = 1, test_seed: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Retornos de train y test (rangos contiguos) y los de train a 5 min."""
    train = _returns(3, seed=train_seed, start="2024-01-01")
    test = _returns(2, seed=test_seed, start="2024-01-04")
    return train, test, {"5m": _aggregate(train, 5)}


def _direct_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon_train: dict[str, pd.DataFrame],
    factors: dict[str, Callable[[], FactorModel]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Llama a `_fold_rows` con parametros fijos y devuelve (diagnostico, sesgo)."""
    bounds = (test.index[0], test.index[-1] + pd.Timedelta(minutes=1))
    valid = pd.Series([1.0, 0.9, 0.8, 0.7], index=list(SYMBOLS))
    diag, bias = study_module._fold_rows(
        "2024-01-04",
        list(SYMBOLS),
        valid,
        train,
        test,
        horizon_train,
        factors or {"btc": _btc, "mean": _mean},
        (1,),
        bounds,
        step_days=1,
        min_valid_share=0.5,
        min_obs=100,
    )
    return pd.DataFrame(diag), pd.DataFrame(bias)


def test_fold_rows_betas_come_from_train_only() -> None:
    """Las betas de `fold`, `h5m` y `h15m` no cambian al alterar el test y si al alterar train."""
    train, test, horizon = _direct_inputs()
    diag, bias = _direct_rows(train, test, horizon)
    train_b, test_b, horizon_b = _direct_inputs(train_seed=3, test_seed=4)
    other_test_diag, other_test_bias = _direct_rows(train, test_b, horizon)
    other_train_diag, other_train_bias = _direct_rows(train_b, test, horizon_b)
    pd.testing.assert_frame_equal(bias, other_test_bias)
    assert not bias.equals(other_train_bias)
    fold = diag["scheme"] == "fold"
    assert not diag[fold].equals(other_test_diag[fold])
    assert not diag[fold].equals(other_train_diag[fold])
    assert set(bias["horizon"]) == {"5m"}


def test_fold_rows_horizon_beta_is_fitted_on_horizon_returns() -> None:
    """El cociente del sesgo es `beta(5m) / beta(1m)` con retornos de train, no 1."""
    train, test, horizon = _direct_inputs()
    _, bias = _direct_rows(train, test, horizon)
    beta_1m = ols_betas(train, train["BTCUSDT"], min_periods=100)
    beta_5m = ols_betas(horizon["5m"], horizon["5m"]["BTCUSDT"], min_periods=100)
    ratio = (beta_5m / beta_1m).drop("BTCUSDT")
    row = bias[bias["factor"] == "btc"].iloc[0]
    assert row["median_ratio"] == pytest.approx(float(ratio.median()))
    assert abs(row["median_ratio"] - 1.0) > 1e-6


def test_fold_rows_skips_bias_and_warns_when_a_horizon_fit_fails(
    log_records: pytest.LogCaptureFixture,
) -> None:
    """Si falla el ajuste de un horizonte se omite el sesgo del fold y factor, con aviso."""

    class _NeedsManyRows(BtcFactor):
        def fit(self, returns: pd.DataFrame) -> _NeedsManyRows:
            """Falla con las pocas filas de los retornos de 5 min."""
            if len(returns) < 1000:
                raise ValueError("pocas filas")
            return super().fit(returns)  # type: ignore[return-value]

    train, test, horizon = _direct_inputs()
    diag, bias = _direct_rows(train, test, horizon, {"btc": _NeedsManyRows, "mean": _mean})
    assert set(bias["factor"]) == {"mean"}  # btc sin tabla de sesgo
    failed = diag[(diag["factor"] == "btc") & (diag["scheme"] == "h5m")].iloc[0]
    assert failed["n_failed"] == 1 and failed["n_coins"] == 0
    warnings = [r.getMessage() for r in log_records.records if r.levelname == "WARNING"]
    skipped = [m for m in warnings if "sesgo" in m]
    assert len(skipped) == 1
    assert "2024-01-04" in skipped[0] and "btc" in skipped[0] and "5m" in skipped[0]
    assert "un fold menos" in skipped[0]


def test_rolling_logs_error_with_first_reason_when_every_block_fails(
    log_records: pytest.LogCaptureFixture,
) -> None:
    """Si falla todo el esquema, se loguea `error` con el primer motivo; el residuo sigue `NaN`."""
    returns = _returns(3)
    start, end = returns.index[0], returns.index[DAY]
    fit = rolling_residuals(_btc, returns, start, end, window_days=1, step_days=1)
    assert fit.n_blocks == fit.n_failed == 1
    assert fit.residuals.isna().all().all() and not (fit.residuals.to_numpy() == 0.0).any()
    errors = [r.getMessage() for r in log_records.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "primer fallo: " in errors[0]
    reason = errors[0].split("primer fallo: ")[1]
    assert reason and reason != "None"
    block_warnings = [r.getMessage() for r in log_records.records if r.levelname == "WARNING"]
    assert any(reason in m for m in block_warnings)


# --- CLI ---


def test_cli_default_tests_are_the_twelve_quarters_of_d37() -> None:
    """Por defecto, 2023T1 a 2025T4, todos inicios de trimestre."""
    assert len(cli.DEFAULT_TESTS) == 12
    for text in cli.DEFAULT_TESTS:
        fold_windows(pd.Timestamp(text, tz="UTC"))
    assert cli.DEFAULT_TESTS[0] == "2023-01-01" and cli.DEFAULT_TESTS[-1] == "2025-10-01"


@pytest.mark.parametrize(
    "argv",
    [
        ["beta-study", "--tests", "2023-02-01"],
        ["beta-study", "--tests", "no-es-fecha"],
        ["beta-study", "--windows", "0"],
        ["beta-study", "--horizons", "7x"],
        ["beta-study", "--horizons", "30s"],
        ["otro-comando"],
        ["beta-study", "--windows", "abc"],
    ],
)
def test_cli_invalid_arguments_exit_2(argv: list[str]) -> None:
    """Argumentos invalidos salen con 2, sin traceback."""
    assert cli.main(argv) == 2


def test_cli_quarter_outside_universe_exits_1() -> None:
    """Un trimestre valido pero fuera del universo sale con 1."""
    assert cli.main(["beta-study", "--tests", "2000-01-01"]) == 1


def test_cli_is_a_thin_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """La CLI solo parsea y llama a `compare_schemes` y `write_tables`; devuelve 0."""
    calls: dict[str, object] = {}
    empty = pd.DataFrame()

    def fake_compare(starts: list[pd.Timestamp], **kwargs: object) -> StudyResult:
        calls["starts"], calls["kwargs"] = starts, kwargs
        return StudyResult(empty, empty, empty)

    def fake_write(result: StudyResult) -> list[Path]:
        calls["written"] = result
        return []

    monkeypatch.setattr(cli, "compare_schemes", fake_compare)
    monkeypatch.setattr(cli, "write_tables", fake_write)
    code = cli.main(
        [
            "beta-study",
            "--tests",
            "2024-01-01",
            "--windows",
            "10",
            "--horizons",
            "1m",
            "5m",
            "--rebuild",
        ]
    )
    assert code == 0
    assert calls["starts"] == [pd.Timestamp("2024-01-01", tz="UTC")]
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["windows"] == [10] and kwargs["horizons"] == ["1m", "5m"] and kwargs["rebuild"]
    assert "written" in calls


def test_cli_job_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un fallo esperado del trabajo (ValueError) sale con 1."""

    def boom(starts: list[pd.Timestamp], **kwargs: object) -> StudyResult:
        raise ValueError("sin cache")

    monkeypatch.setattr(cli, "compare_schemes", boom)
    assert cli.main(["beta-study", "--tests", "2024-01-01"]) == 1


def test_residual_fit_is_a_named_tuple() -> None:
    """`ResidualFit` desempaqueta en cinco campos con nombre."""
    assert ResidualFit._fields == ("residuals", "factor", "n_blocks", "n_failed", "betas")


def test_end_to_end_train_and_test_are_disjoint_and_contiguous(
    world: tuple[Path, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train acaba antes del primer minuto de test (tambien los de 5 y 15 min): sin solape."""
    seen: dict[str, object] = {}
    real = study_module._fold_rows

    def spy(*args: object, **kwargs: object) -> object:
        seen["train"], seen["test"], seen["horizon"] = args[3], args[4], args[5]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(study_module, "_fold_rows", spy)
    _compare(world, tmp_path / "panels")
    start = pd.Timestamp("2024-01-01", tz="UTC")
    train, test = seen["train"], seen["test"]
    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    assert train.index.max() < start == test.index.min()
    assert train.index.intersection(test.index).empty
    horizon = seen["horizon"]
    assert isinstance(horizon, dict) and set(horizon) == {"5m", "15m"}
    for frame in horizon.values():
        assert frame.index.max() < start
