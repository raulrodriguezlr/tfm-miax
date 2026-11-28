"""Tests del estudio del retraso respecto al mercado (MIAX-216): diagnostico, decision y CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import miax.eval.__main__ as cli
import miax.eval.market_lag as ml
from miax.eval import residual_diagnostics
from miax.eval.beta_study import FoldData
from miax.eval.idiosyncratic_signal import SignalResult, decision_table
from miax.eval.market_lag import (
    LAG_DECIDERS,
    LAG_MAKERS,
    SIGNAL_FACTORS,
    MarketLagResult,
    build_makers,
    compare_market_lag,
    diagnostic_columns,
    lag_diagnostics,
    write_market_lag_tables,
)
from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_lagged import LaggedFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.utils.config import ConfigError

BTC = "BTCUSDT"
COINS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
LAGS = 2
MIN_PERIODS = 100
START_A = pd.Timestamp("2024-01-01", tz="UTC")
START_B = pd.Timestamp("2024-04-01", tz="UTC")


def _index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="min", tz="UTC")


# ---------- constantes y fabricas ----------


def test_constants_and_signal_order_are_fixed() -> None:
    """El orden de los factores del contraste fija las semillas de 057 y no puede cambiar."""
    assert SIGNAL_FACTORS == ("btc", "mean", "pca", "mean_lag", "pca_lag")
    assert LAG_DECIDERS == ("mean_lag", "pca_lag")
    assert ml.DEFAULT_LAGS == 5
    assert set(ml._SIGNAL_RESERVED) >= {"factors", "horizons", "universe", "min_obs", "rebuild"}


def test_build_makers_wraps_each_base_with_the_requested_lags() -> None:
    """Seis fabricas: tres base y sus versiones con `lags` retardos, modelo nuevo cada vez."""
    makers = build_makers(3)
    assert list(makers) == ["btc", "mean", "pca", "btc_lag", "mean_lag", "pca_lag"]
    for name, base_type in (("btc", BtcFactor), ("mean", MeanFactor), ("pca", PcaFactor)):
        base = makers[name]()
        assert type(base) is base_type
        wrapped = makers[f"{name}_lag"]()
        assert isinstance(wrapped, LaggedFactor) and wrapped.lags == 3
        assert type(wrapped.base) is base_type
        assert makers[f"{name}_lag"]() is not wrapped
    assert LAG_MAKERS["mean_lag"]().lags == ml.DEFAULT_LAGS  # type: ignore[attr-defined]


def test_diagnostic_columns_layout() -> None:
    """Columnas fijas: identificadores, varianza, corr, abs_corr, gamma por retardo y error."""
    assert diagnostic_columns(2) == [
        "fold",
        "model",
        "n_coins",
        "var_removed",
        "var_removed_extra",
        "corr_lag1",
        "corr_lag2",
        "abs_corr_lag1",
        "abs_corr_lag2",
        "gamma_lag1",
        "gamma_lag2",
        "error",
    ]
    assert len(diagnostic_columns(5)) == 5 + 3 * 5 + 1


# ---------- lag_diagnostics ----------


def _diag_inputs(n: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """e1 = 0.5 f(t-1) + ruido, e2 = -0.5 f(t-2) + ruido; r = e + 0.3 f."""
    rng = np.random.default_rng(11)
    f = rng.normal(size=n)
    f1 = np.concatenate([[np.nan], f[:-1]])
    f2 = np.concatenate([[np.nan, np.nan], f[:-2]])
    e = pd.DataFrame(
        {
            "UP": 0.5 * np.nan_to_num(f1) + rng.normal(size=n),
            "DOWN": -0.5 * np.nan_to_num(f2) + rng.normal(size=n),
        },
        index=_index(n),
    )
    r = e + 0.3 * f[:, None]
    return r, e, pd.Series(f, index=e.index, name="factor")


def test_lag_diagnostics_correlations_by_hand_with_sign() -> None:
    """`corr_lag{l}` es el Pearson de e(t) con f(t-l) calculado a mano, con signo."""
    r, e, f = _diag_inputs()
    table = lag_diagnostics(r, e, f, lags=3, min_obs=100)
    assert list(table.columns) == ["var_removed", "corr_lag1", "corr_lag2", "corr_lag3"]
    for symbol in ("UP", "DOWN"):
        for step in (1, 2, 3):
            a, b = e[symbol].to_numpy()[step:], f.to_numpy()[:-step]
            assert table.loc[symbol, f"corr_lag{step}"] == pytest.approx(np.corrcoef(a, b)[0, 1])
    assert table.loc["UP", "corr_lag1"] > 0.3
    assert table.loc["DOWN", "corr_lag2"] < -0.3
    assert abs(table.loc["UP", "corr_lag2"]) < 0.1


def test_lag_diagnostics_var_removed_matches_residual_diagnostics() -> None:
    """`var_removed` y `corr_lag1` coinciden con los de `residual_diagnostics` sin huecos."""
    r, e, f = _diag_inputs()
    ours = lag_diagnostics(r, e, f, lags=2, min_obs=100)
    theirs = residual_diagnostics(r, e, f, min_obs=100)
    np.testing.assert_allclose(ours["var_removed"], theirs["var_removed"])
    np.testing.assert_allclose(ours["corr_lag1"], theirs["corr_lag1"])
    by_hand = 1.0 - (e["UP"] ** 2).sum() / (r["UP"] ** 2).sum()
    assert ours.loc["UP", "var_removed"] == pytest.approx(by_hand)


def test_lag_diagnostics_nan_below_min_obs_and_with_a_gap_by_time() -> None:
    """Con menos de `min_obs` datos todo es NaN; un hueco anula el retardo por tiempo, no otro."""
    r, e, f = _diag_inputs(600)
    r.iloc[:560, 0] = np.nan  # UP: 40 datos
    e = e.mask(r.isna())
    table = lag_diagnostics(r, e, f, lags=2, min_obs=100)
    assert table.loc["UP"].isna().all()
    assert table.loc["DOWN"].notna().all()
    # un NaN en el factor solo quita las filas t = j+1 y t = j+2
    f_gap = f.copy()
    f_gap.iloc[300] = np.nan
    gapped = lag_diagnostics(r, e, f_gap, lags=2, min_obs=100)
    keep = np.ones(600, dtype=bool)
    keep[301] = False
    a, b = e["DOWN"].to_numpy()[1:][keep[1:]], f_gap.to_numpy()[:-1][keep[1:]]
    ok = ~np.isnan(b)
    assert gapped.loc["DOWN", "corr_lag1"] == pytest.approx(np.corrcoef(a[ok], b[ok])[0, 1])


def test_lag_diagnostics_rejects_bad_input() -> None:
    """Indices distintos, `lags` < 1 o `min_obs` < 2 dan `ValueError`."""
    r, e, f = _diag_inputs(300)
    with pytest.raises(ValueError, match="indice"):
        lag_diagnostics(r, e.iloc[1:], f)
    with pytest.raises(ValueError, match="indice"):
        lag_diagnostics(r, e, f.iloc[1:])
    with pytest.raises(ValueError, match="lags"):
        lag_diagnostics(r, e, f, lags=0)
    with pytest.raises(ValueError, match="min_obs"):
        lag_diagnostics(r, e, f, min_obs=1)


# ---------- estudio con folds sinteticos ----------


def _fold_returns(n: int, seed: int, start: pd.Timestamp, gamma: float = 0.5) -> pd.DataFrame:
    """BTC como factor y cuatro monedas con retraso conocido respecto a el."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, n)
    f_prev = np.concatenate([[0.0], f[:-1]])
    data = {BTC: f}
    for k, symbol in enumerate(COINS):
        data[symbol] = (0.6 + 0.3 * k) * f + gamma * f_prev + rng.normal(0.0, 0.002, n)
    return pd.DataFrame(data, index=pd.date_range(start, periods=n, freq="min"))


def _fake_load(test_start: pd.Timestamp, **kwargs: object) -> FoldData:  # sustituye a `load_fold`
    train = _fold_returns(3000, seed=test_start.month, start=test_start - pd.Timedelta(days=30))
    test = _fold_returns(1500, seed=100 + test_start.month, start=test_start)
    symbols = [BTC, *COINS]
    return FoldData(
        f"{test_start:%Y-%m-%d}",
        symbols,
        symbols,
        pd.Series(1.0, index=symbols),
        train,
        test,
        test_start - pd.Timedelta(days=30),
        test_start,
        test_start + pd.Timedelta(days=90),
    )


def _makers() -> dict[str, object]:
    def lagged(base: type[FactorModel], **kw: object):  # noqa: ANN202
        return lambda: LaggedFactor(
            base(min_periods=MIN_PERIODS, **kw), lags=LAGS, min_periods=MIN_PERIODS
        )

    return {
        "btc": lambda: BtcFactor(BTC, min_periods=MIN_PERIODS),
        "mean": lambda: MeanFactor(min_periods=MIN_PERIODS),
        "pca": lambda: PcaFactor(min_periods=MIN_PERIODS),
        "btc_lag": lambda: LaggedFactor(
            BtcFactor(BTC, min_periods=MIN_PERIODS), lags=LAGS, min_periods=MIN_PERIODS
        ),
        "mean_lag": lagged(MeanFactor),
        "pca_lag": lagged(PcaFactor),
    }


class _SignalSpy:
    """Sustituye a `study_signal`: registra argumentos y devuelve un `by_fold` fijo."""

    def __init__(self, by_fold: pd.DataFrame | None = None) -> None:
        self.calls: list[tuple[list[pd.Timestamp], dict[str, object]]] = []
        self.by_fold = _by_fold({}) if by_fold is None else by_fold

    def __call__(self, starts: list[pd.Timestamp], **kwargs: object) -> SignalResult:
        self.calls.append((list(starts), kwargs))
        return SignalResult(self.by_fold, decision_table(self.by_fold))


def _by_fold(wins: dict[str, int], n: int = 12) -> pd.DataFrame:
    """`by_fold` de 057 reducido: `wins[f]` folds con p=0.01 y el resto con p=0.5, a 1 minuto."""
    rows = [
        {
            "fold": f"f{i:02d}",
            "factor": factor,
            "horizon": "1m",
            "p_cross": 0.01 if i < wins.get(factor, 0) else 0.5,
        }
        for factor in SIGNAL_FACTORS
        for i in range(n)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> _SignalSpy:
    """Folds sinteticos en lugar de `load_fold` y un espia en lugar de `study_signal`."""
    spy = _SignalSpy()
    monkeypatch.setattr(ml, "load_fold", _fake_load)
    monkeypatch.setattr(ml, "study_signal", spy)
    return spy


def _study(tests: list[str] | None = None, **kwargs: object) -> MarketLagResult:
    params: dict[str, object] = {
        "universe": pd.DataFrame(),
        "makers": _makers(),
        "lags": LAGS,
        "min_obs": 200,
        "run_signal": False,
    }
    params.update(kwargs)
    return compare_market_lag(tests or ["2024-01-01", "2024-04-01"], **params)  # type: ignore[arg-type]


def test_study_columns_order_and_extra_variance(world: _SignalSpy) -> None:
    """Columnas fijas, una fila por fold y modelo en orden, y `var_removed_extra` frente al base."""
    result = _study()
    diag = result.diagnostics
    assert list(diag.columns) == diagnostic_columns(LAGS)
    assert list(diag["fold"]) == ["2024-01-01"] * 6 + ["2024-04-01"] * 6
    assert list(diag["model"][:6]) == ["btc", "mean", "pca", "btc_lag", "mean_lag", "pca_lag"]
    assert (diag["error"] == "").all()
    for fold, group in diag.groupby("fold"):
        by_model = group.set_index("model")
        for name in ("btc", "mean", "pca"):
            assert np.isnan(by_model.loc[name, "var_removed_extra"])
            assert np.isnan(by_model.loc[name, "gamma_lag1"])
            expected = (
                by_model.loc[f"{name}_lag", "var_removed"] - by_model.loc[name, "var_removed"]
            )
            assert by_model.loc[f"{name}_lag", "var_removed_extra"] == pytest.approx(expected), fold
            assert by_model.loc[f"{name}_lag", "var_removed_extra"] > 0
        # con gamma conocida de 0.5, el retraso es lo que queda en el base y no en la version lag
        assert by_model.loc["btc", "abs_corr_lag1"] > 0.5
        assert by_model.loc["btc_lag", "abs_corr_lag1"] < 0.1
        assert by_model.loc["btc_lag", "gamma_lag1"] == pytest.approx(0.5, abs=0.1)
    assert world.calls == []


def test_study_is_deterministic(world: _SignalSpy) -> None:
    """Dos ejecuciones dan tablas identicas."""
    pd.testing.assert_frame_equal(_study().diagnostics, _study().diagnostics)


def test_value_error_in_a_model_gives_an_error_row_and_the_study_continues(
    world: _SignalSpy,
) -> None:
    """Un `ValueError` del modelo deja su fila con `error` y NaN; los demas siguen."""

    def broken() -> FactorModel:
        raise ValueError("ajuste imposible")

    makers = _makers() | {"mean": broken}
    diag = _study(makers=makers).diagnostics
    bad = diag[diag["model"] == "mean"]
    assert (bad["error"] == "ajuste imposible").all()
    assert bad[["var_removed", "corr_lag1", "n_coins"]].isna().all().all()
    other = diag[diag["model"] != "mean"]
    assert (other["error"] == "").all()
    assert other["var_removed"].notna().all()
    # la version lag cuyo base falla no tiene diferencia que reportar
    assert diag[diag["model"] == "mean_lag"]["var_removed_extra"].isna().all()
    assert diag[diag["model"] == "pca_lag"]["var_removed_extra"].notna().all()


def test_runtime_error_in_a_model_is_not_captured(world: _SignalSpy) -> None:
    """Un `RuntimeError` (p. ej. fuga detectada) se propaga y tumba el estudio."""

    def leaky() -> FactorModel:
        raise RuntimeError("cambio su estado ajustado")

    with pytest.raises(RuntimeError, match="estado"):
        _study(makers=_makers() | {"pca": leaky})


def test_study_rejects_bad_parameters(world: _SignalSpy) -> None:
    """`lags` < 1 y sin trimestres dan `ValueError`."""
    with pytest.raises(ValueError, match="lags"):
        _study(lags=0)
    with pytest.raises(ValueError, match="trimestre"):
        compare_market_lag([], universe=pd.DataFrame(), run_signal=False)


def test_skip_signal_never_calls_study_signal(world: _SignalSpy) -> None:
    """Con `run_signal=False` no se llama a `study_signal` y no hay tablas de senal."""
    result = _study(run_signal=False)
    assert world.calls == []
    assert result.signal_by_fold is None and result.signal_decision is None


def test_signal_gets_the_factors_in_fixed_order_at_one_minute(world: _SignalSpy) -> None:
    """Pasa btc, mean, pca, mean_lag y pca_lag en ese orden y a 1 min, vengan como vengan."""
    shuffled = dict(reversed(list(_makers().items())))
    result = _study(makers=shuffled, run_signal=True, signal_kwargs={"n_shifts": 7, "seed": 3})
    assert len(world.calls) == 1
    starts, kwargs = world.calls[0]
    assert starts == [START_A, START_B]
    factors = kwargs["factors"]
    assert isinstance(factors, dict)
    assert list(factors) == ["btc", "mean", "pca", "mean_lag", "pca_lag"]
    assert factors["mean_lag"] is shuffled["mean_lag"]
    assert tuple(kwargs["horizons"]) == ("1m",)  # type: ignore[arg-type]
    assert kwargs["min_obs"] == 200 and kwargs["n_folds_required"] == 12
    assert kwargs["n_shifts"] == 7 and kwargs["seed"] == 3
    assert kwargs["rebuild"] is False
    assert result.signal_by_fold is not None and result.signal_decision is not None


@pytest.mark.parametrize("key", sorted(ml._SIGNAL_RESERVED))
def test_reserved_signal_kwargs_raise(world: _SignalSpy, key: str) -> None:
    """Una clave que el estudio ya fija en `signal_kwargs` da `ValueError`."""
    with pytest.raises(ValueError, match="ya fijadas"):
        _study(run_signal=True, signal_kwargs={key: 1})
    assert world.calls == []


def test_signal_without_any_signal_model_raises(world: _SignalSpy) -> None:
    """Si ningun maker esta en `SIGNAL_FACTORS`, hay `ValueError`."""
    with pytest.raises(ValueError, match="ningun modelo"):
        _study(makers={"otro": _makers()["mean"]}, run_signal=True)  # type: ignore[dict-item]


def _verdicts(world: _SignalSpy, wins: dict[str, int], n: int = 12) -> dict[str, str]:
    world.by_fold = _by_fold(wins, n)
    decision = _study(run_signal=True).signal_decision
    assert decision is not None
    tail = decision.tail(2)
    assert list(tail["factor"]) == ["verdict_base", "verdict_lag"]
    return dict(zip(tail["factor"], tail["verdict"], strict=True))


def test_verdict_lag_and_base_apply_the_d44_rule(world: _SignalSpy) -> None:
    """12/12 con las versiones lag da senal, <=3 en una da pivote y el base se juzga aparte."""
    full = {"mean": 12, "pca": 12, "mean_lag": 12, "pca_lag": 12}
    assert _verdicts(world, full) == {"verdict_base": "signal", "verdict_lag": "signal"}
    survives_only_base = full | {"mean_lag": 12, "pca_lag": 3}
    assert _verdicts(world, survives_only_base) == {
        "verdict_base": "signal",
        "verdict_lag": "pivot",
    }
    vanished = full | {"mean_lag": 0, "pca_lag": 0}
    assert _verdicts(world, vanished) == {"verdict_base": "signal", "verdict_lag": "pivot"}
    base_pivot = full | {"mean": 2}
    assert _verdicts(world, base_pivot) == {"verdict_base": "pivot", "verdict_lag": "signal"}
    middle = full | {"pca_lag": 6}
    assert _verdicts(world, middle) == {"verdict_base": "signal", "verdict_lag": "inconclusive"}
    nine = full | {"mean_lag": 9, "pca_lag": 9}
    assert _verdicts(world, nine) == {"verdict_base": "signal", "verdict_lag": "signal"}
    eight = full | {"mean_lag": 8}
    assert _verdicts(world, eight) == {"verdict_base": "signal", "verdict_lag": "inconclusive"}


def test_verdict_with_fewer_than_twelve_folds_is_inconclusive(world: _SignalSpy) -> None:
    """Con 11 folds validos ninguno de los dos veredictos es senal ni pivote."""
    full = {"mean": 11, "pca": 11, "mean_lag": 11, "pca_lag": 11}
    assert _verdicts(world, full, n=11) == {
        "verdict_base": "inconclusive",
        "verdict_lag": "inconclusive",
    }
    zero = {"mean": 0, "pca": 0, "mean_lag": 0, "pca_lag": 0}
    assert _verdicts(world, zero, n=11) == {
        "verdict_base": "inconclusive",
        "verdict_lag": "inconclusive",
    }


def test_decision_table_keeps_per_factor_rows_before_the_verdicts(world: _SignalSpy) -> None:
    """Las filas por factor van antes de `verdict_base` y `verdict_lag`, sin `verdict` suelto."""
    world.by_fold = _by_fold({"mean": 12, "pca": 12, "mean_lag": 12, "pca_lag": 12})
    decision = _study(run_signal=True).signal_decision
    assert decision is not None
    assert list(decision.columns) == ["factor", "horizon", "n_folds", "n_significant", "verdict"]
    assert "verdict" not in set(decision["factor"])
    assert set(decision["factor"][:-2]) == set(SIGNAL_FACTORS)
    lag_rows = decision[decision["factor"].isin(LAG_DECIDERS)]
    assert set(lag_rows["verdict"]) == {"signal"}
    btc = decision[decision["factor"] == "btc"].iloc[0]
    assert btc["verdict"] == "report"


# ---------- escritura de tablas ----------


def _result(with_signal: bool) -> MarketLagResult:
    diag = pd.DataFrame({c: [np.nan] for c in diagnostic_columns(2)})
    diag["fold"], diag["model"], diag["error"] = "2024-01-01", "mean", ""
    diag["var_removed"] = 1 / 3
    if not with_signal:
        return MarketLagResult(diag, None, None)
    by_fold = _by_fold({"mean": 12, "pca": 12, "mean_lag": 12, "pca_lag": 12})
    by_fold["p_cross"] = by_fold["p_cross"] / 3
    return MarketLagResult(by_fold, by_fold, decision_table(by_fold))  # type: ignore[arg-type]


def test_write_tables_is_deterministic_and_uses_fixed_format(tmp_path: Path) -> None:
    """Mismos bytes en dos ejecuciones, `%.10g`, sin retorno de carro y columnas fijas."""
    first = write_market_lag_tables(_result(False), out_dir=tmp_path / "a")
    second = write_market_lag_tables(_result(False), out_dir=tmp_path / "b")
    assert [p.name for p in first] == ["market_lag_diagnostics.csv"]
    assert first[0].read_bytes() == second[0].read_bytes()
    assert b"\r" not in first[0].read_bytes()
    text = first[0].read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(diagnostic_columns(2))
    assert "0.3333333333" in text and "0.33333333333" not in text


def test_write_tables_without_signal_does_not_write_signal_tables(tmp_path: Path) -> None:
    """Sin senal solo se escribe el diagnostico; con senal, las tres tablas, en orden."""
    paths = write_market_lag_tables(_result(False), out_dir=tmp_path / "x")
    assert sorted(p.name for p in (tmp_path / "x").iterdir()) == ["market_lag_diagnostics.csv"]
    assert len(paths) == 1
    full = write_market_lag_tables(_result(True), out_dir=tmp_path / "y")
    assert [p.name for p in full] == [
        "market_lag_diagnostics.csv",
        "market_lag_signal_by_fold.csv",
        "market_lag_signal_decision.csv",
    ]
    assert all(p.parent == tmp_path / "y" for p in full)
    again = write_market_lag_tables(_result(True), out_dir=tmp_path / "z")
    for a, b in zip(full, again, strict=True):
        assert a.read_bytes() == b.read_bytes()
    assert "0.003333333333" in full[1].read_text(encoding="utf-8")


# ---------- CLI ----------


def _cli_result(with_signal: bool = True) -> MarketLagResult:
    diag = pd.DataFrame({"model": ["btc", "btc_lag"], "abs_corr_lag1": [0.04, 0.001]})
    decision = pd.DataFrame(
        {"factor": ["mean", "verdict_base", "verdict_lag"], "verdict": ["x"] * 3}
    )
    return MarketLagResult(diag, None, decision if with_signal else None)


@pytest.fixture
def cli_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Sustituye el estudio y la escritura de la CLI y apunta lo recibido."""
    calls: dict[str, object] = {}

    def fake_compare(tests: list[str], **kwargs: object) -> MarketLagResult:
        calls["tests"], calls["kwargs"] = tests, kwargs
        return _cli_result(with_signal=bool(kwargs["run_signal"]))

    monkeypatch.setattr(cli, "compare_market_lag", fake_compare)
    monkeypatch.setattr(
        cli, "write_market_lag_tables", lambda result: calls.setdefault("w", result)
    )
    return calls


def test_registry_has_market_lag_subcommand() -> None:
    """El registro de subcomandos incluye `market-lag`."""
    assert "market-lag" in cli._COMMANDS


def test_cli_market_lag_is_a_thin_layer(cli_calls: dict[str, object]) -> None:
    """Parsea, llama al estudio y a la escritura y devuelve 0; por defecto hace la senal."""
    code = cli.main(
        ["market-lag", "--tests", "2024-01-01", "2024-04-01", "--lags", "3", "--rebuild"]
    )
    assert code == 0 and "w" in cli_calls
    assert cli_calls["tests"] == ["2024-01-01", "2024-04-01"]
    assert cli_calls["kwargs"] == {"lags": 3, "run_signal": True, "rebuild": True}


def test_cli_defaults_are_the_twelve_quarters_and_five_lags(cli_calls: dict[str, object]) -> None:
    """Sin argumentos usa los 12 trimestres de D-37 y 5 retardos."""
    assert cli.main(["market-lag"]) == 0
    assert cli_calls["tests"] == list(cli.DEFAULT_TESTS) and len(cli.DEFAULT_TESTS) == 12
    assert cli_calls["kwargs"] == {"lags": 5, "run_signal": True, "rebuild": False}


def test_cli_skip_signal_passes_run_signal_false(cli_calls: dict[str, object]) -> None:
    """`--skip-signal` pasa `run_signal=False`."""
    assert cli.main(["market-lag", "--tests", "2024-01-01", "--skip-signal"]) == 0
    kwargs = cli_calls["kwargs"]
    assert isinstance(kwargs, dict) and kwargs["run_signal"] is False


@pytest.mark.parametrize("error", [ValueError, RuntimeError, OSError, ConfigError])
def test_cli_job_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: type[Exception]
) -> None:
    """Un error esperado del trabajo sale con 1 y sin traceback."""

    def boom(tests: list[str], **kwargs: object) -> MarketLagResult:
        raise error("fallo esperado")

    monkeypatch.setattr(cli, "compare_market_lag", boom)
    assert cli.main(["market-lag", "--tests", "2024-01-01"]) == 1
    assert "Traceback" not in caplog.text
    assert "fallo esperado" in caplog.text


def test_cli_write_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, cli_calls: dict[str, object]
) -> None:
    """Un `OSError` al escribir las tablas tambien sale con 1."""

    def full_disk(result: MarketLagResult) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr(cli, "write_market_lag_tables", full_disk)
    assert cli.main(["market-lag", "--tests", "2024-01-01"]) == 1


def test_cli_unexpected_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un error que no es de los esperados (p. ej. `KeyError`) no se captura."""

    def bug(tests: list[str], **kwargs: object) -> MarketLagResult:
        raise KeyError("bug")

    monkeypatch.setattr(cli, "compare_market_lag", bug)
    with pytest.raises(KeyError):
        cli.main(["market-lag", "--tests", "2024-01-01"])


@pytest.mark.parametrize(
    "argv",
    [
        ["market-lag", "--tests", "2023-02-01"],
        ["market-lag", "--tests", "no-es-fecha"],
        ["market-lag", "--lags", "0"],
        ["market-lag", "--lags", "-1"],
        ["market-lag", "--lags", "abc"],
        ["market-lag", "--no-existe"],
    ],
)
def test_cli_invalid_arguments_exit_2(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Trimestre invalido, `--lags` < 1 o flag desconocida salen con 2 sin ejecutar el estudio."""

    def must_not_run(*args: object, **kwargs: object) -> MarketLagResult:
        raise AssertionError("no debe ejecutarse")

    monkeypatch.setattr(cli, "compare_market_lag", must_not_run)
    assert cli.main(argv) == 2
