"""Tests del estudio del factor sin la propia moneda (MIAX-215): metricas, folds, tablas y CLI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest

import miax.eval.__main__ as cli
import miax.eval.loo_factor as study
from miax.eval.beta_study import FoldData
from miax.eval.loo_factor import (
    FOLD_COLUMNS,
    LOO_MAKERS,
    METRIC_COLUMNS,
    SUMMARY_COLUMNS,
    LooResult,
    _model_metrics,
    compare_loo,
    write_loo_tables,
)
from miax.features.factor import FactorModel, NotFittedError
from miax.features.factor_loo import LooMeanFactor, LooPcaFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.utils.config import ConfigError

N_ROWS, MIN_OBS, N_COINS = 1500, 200, 6
SYMS = [f"S{i}USDT" for i in range(N_COINS)]

MAKERS: dict[str, Callable[[], FactorModel]] = {
    "mean": lambda: MeanFactor(min_periods=100),
    "pca": lambda: PcaFactor(min_periods=100),
    "loo_mean": lambda: LooMeanFactor(min_periods=100),
    "loo_pca": lambda: LooPcaFactor(min_periods=100),
}


def _frame(start: str, seed: int, *, nan_frac: float = 0.0, k: int = N_COINS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=N_ROWS, freq="min", tz="UTC")
    common = rng.normal(0, 0.001, N_ROWS)
    data = common[:, None] * np.linspace(0.5, 1.5, k) + rng.normal(0, 0.001, (N_ROWS, k))
    if nan_frac:
        data[rng.random(data.shape) < nan_frac] = np.nan
    return pd.DataFrame(data, index=idx, columns=SYMS[:k])


def _fold(test_start: pd.Timestamp, *, nan_frac: float = 0.0, dead_coin: bool = False) -> FoldData:
    month = test_start.month
    train = _frame("2023-01-01", seed=month)
    test = _frame("2024-01-01", seed=100 + month, nan_frac=nan_frac)
    if dead_coin:
        test[SYMS[-1]] = np.nan
    return FoldData(
        f"{test_start:%Y-%m-%d}", SYMS, SYMS, pd.Series(1.0, index=SYMS), train, test,
        test_start - pd.Timedelta(days=365), test_start, test_start + pd.Timedelta(days=90),
    )  # fmt: skip


@pytest.fixture
def fake_folds(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Sustituye `load_fold` por folds sinteticos sin red ni cache."""

    def install(**kwargs: object) -> None:
        monkeypatch.setattr(study, "load_fold", lambda start, **kw: _fold(start, **kwargs))

    install()
    return install


TESTS = ["2023-01-01", "2023-04-01"]


def _run(**kwargs: object) -> LooResult:
    base: dict[str, object] = {"universe": pd.DataFrame(), "makers": MAKERS, "min_obs": MIN_OBS}
    return compare_loo(TESTS, **{**base, **kwargs})  # type: ignore[arg-type]


# --- _model_metrics ---


def test_model_metrics_fields_and_floor() -> None:
    """Las metricas salen completas, con floor = -1/(N-1) con el N de monedas utilizables."""
    fold = _fold(pd.Timestamp("2023-04-01", tz="UTC"), dead_coin=True)
    model = MAKERS["loo_mean"]().fit(fold.returns_train)
    metrics, corr = _model_metrics(model, fold.returns_test, min_obs=MIN_OBS)
    assert set(metrics) == set(METRIC_COLUMNS)
    assert metrics["n_symbols"] == N_COINS - 1 == len(corr)
    assert metrics["floor"] == pytest.approx(-1 / (N_COINS - 2))
    assert metrics["n_complete_rows"] == N_ROWS
    assert np.isnan(metrics["matrix_change"])  # lo rellena compare_loo
    for key in ("pair_mean", "pair_abs_mean", "correlation_drop", "min_eigenvalue", "sum_ratio"):
        assert np.isfinite(metrics[key]), key
    assert np.isfinite(metrics["beta_mean"]) and np.isfinite(metrics["market_exposure"])


def test_model_metrics_too_few_symbols_raises() -> None:
    """Con menos de 2 monedas utilizables es ValueError."""
    fold = _fold(pd.Timestamp("2023-04-01", tz="UTC"))
    model = MAKERS["mean"]().fit(fold.returns_train)
    test = fold.returns_test.copy()
    test.iloc[:, 1:] = np.nan
    with pytest.raises(ValueError, match="menos de 2"):
        _model_metrics(model, test, min_obs=MIN_OBS)


def test_model_metrics_few_complete_rows_warns_and_leaves_nan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Con menos de min_obs filas completas, min_eigenvalue y sum_ratio son NaN y hay aviso."""
    fold = _fold(pd.Timestamp("2023-04-01", tz="UTC"), nan_frac=0.3)
    model = MAKERS["mean"]().fit(fold.returns_train)
    with caplog.at_level(logging.WARNING):
        metrics, _ = _model_metrics(model, fold.returns_test, min_obs=MIN_OBS)
    assert metrics["n_complete_rows"] < MIN_OBS
    assert np.isnan(metrics["min_eigenvalue"]) and np.isnan(metrics["sum_ratio"])
    assert np.isfinite(metrics["min_eigenvalue_pairwise"]) and np.isfinite(metrics["pair_mean"])
    assert "filas completas" in caplog.text


# --- compare_loo ---


def test_compare_loo_columns_order_and_matrix_change(fake_folds: Callable[..., None]) -> None:
    """Columnas y orden fijos; matrix_change NaN para estandar y finito para LOO."""
    result = _run()
    assert list(result.folds.columns) == FOLD_COLUMNS
    assert list(result.summary.columns) == SUMMARY_COLUMNS
    assert list(result.folds["fold"]) == ["2023-01-01"] * 4 + ["2023-04-01"] * 4
    assert list(result.folds["model"]) == list(MAKERS) * 2
    assert (result.folds["error"] == "").all()
    assert (result.folds["n_symbols"] == N_COINS).all()
    assert result.folds["floor"].to_numpy() == pytest.approx(-1 / (N_COINS - 1))
    by_model = result.folds.groupby("model")["matrix_change"]
    assert by_model.apply(lambda s: s.isna().all())[["mean", "pca"]].all()
    assert by_model.apply(lambda s: (s > 0).all())[["loo_mean", "loo_pca"]].all()
    assert list(result.summary["model"]) == list(MAKERS)
    assert (result.summary["n_folds"] == 2).all()


def test_compare_loo_summary_is_median_of_valid_rows(fake_folds: Callable[..., None]) -> None:
    """El resumen es la mediana entre folds validos; un fold con error no cuenta en n_folds."""
    calls = {"n": 0}

    class FailsOnce(MeanFactor):
        def fit(self, returns: pd.DataFrame) -> Self:
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("fold roto a proposito")
            return super().fit(returns)

    result = _run(makers={"mean": MAKERS["mean"], "flaky": lambda: FailsOnce(min_periods=100)})
    flaky = result.folds[result.folds["model"] == "flaky"]
    assert list(flaky["error"]) == ["", "fold roto a proposito"]
    assert flaky.loc[flaky["error"] != "", METRIC_COLUMNS].isna().all(axis=None)
    row = result.summary.set_index("model").loc["flaky"]
    assert row["n_folds"] == 1
    assert row["pair_mean"] == pytest.approx(flaky["pair_mean"].iloc[0])
    mean = result.folds[result.folds["model"] == "mean"]
    summary_mean = result.summary.set_index("model").loc["mean"]
    assert summary_mean["n_folds"] == 2
    assert summary_mean["pair_mean"] == pytest.approx(mean["pair_mean"].median())


def test_compare_loo_value_error_is_a_row_and_study_goes_on(
    fake_folds: Callable[..., None],
) -> None:
    """Un modelo que lanza ValueError da fila con `error` y los demas siguen."""

    class Broken(MeanFactor):
        def fit(self, returns: pd.DataFrame) -> Self:
            raise ValueError("no ajusta")

    result = _run(makers={"broken": Broken, "mean": MAKERS["mean"]})
    broken = result.folds[result.folds["model"] == "broken"]
    assert (broken["error"] == "no ajusta").all() and broken["pair_mean"].isna().all()
    assert (result.folds[result.folds["model"] == "mean"]["error"] == "").all()
    summary = result.summary.set_index("model")
    assert summary.loc["broken", "n_folds"] == 0 and np.isnan(summary.loc["broken", "pair_mean"])
    assert summary.loc["mean", "n_folds"] == 2


def test_compare_loo_runtime_error_is_not_caught(fake_folds: Callable[..., None]) -> None:
    """Un RuntimeError (fuga o NotFittedError) tumba el estudio."""

    class Leaky(MeanFactor):
        def fit(self, returns: pd.DataFrame) -> Self:
            raise NotFittedError("simulado")

    with pytest.raises(RuntimeError):
        _run(makers={"leaky": Leaky})


def test_compare_loo_rejects_bad_min_obs(fake_folds: Callable[..., None]) -> None:
    """min_obs < 2 es ValueError."""
    with pytest.raises(ValueError, match="min_obs"):
        _run(min_obs=1)


def test_compare_loo_default_makers_and_tests() -> None:
    """El registro por defecto son los cuatro modelos y los tests por defecto, los 12 de D-37."""
    assert list(LOO_MAKERS) == ["mean", "pca", "loo_mean", "loo_pca"]
    assert len(study.DEFAULT_TESTS) == 12
    assert cli.DEFAULT_TESTS is study.DEFAULT_TESTS
    assert study.DEFAULT_TESTS[0] == "2023-01-01" and study.DEFAULT_TESTS[-1] == "2025-10-01"


def test_compare_loo_is_deterministic(fake_folds: Callable[..., None]) -> None:
    """Dos ejecuciones dan tablas identicas."""
    a, b = _run(), _run()
    pd.testing.assert_frame_equal(a.folds, b.folds)
    pd.testing.assert_frame_equal(a.summary, b.summary)


# --- write_loo_tables ---


def test_write_loo_tables_is_deterministic_and_formatted(
    fake_folds: Callable[..., None], tmp_path: Path
) -> None:
    """Dos ejecuciones dan los mismos bytes; cabeceras fijas y LF."""
    first = write_loo_tables(_run(), out_dir=tmp_path / "a")
    second = write_loo_tables(_run(), out_dir=tmp_path / "b")
    assert [p.name for p in first] == ["loo_factor_folds.csv", "loo_factor_summary.csv"]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes() and b"\r" not in a.read_bytes()
    assert first[0].read_text(encoding="utf-8").splitlines()[0] == ",".join(FOLD_COLUMNS)
    assert first[1].read_text(encoding="utf-8").splitlines()[0] == ",".join(SUMMARY_COLUMNS)


def test_write_loo_tables_number_format(tmp_path: Path) -> None:
    """Los flotantes salen con 10 cifras significativas."""
    folds = pd.DataFrame({c: [np.nan] for c in FOLD_COLUMNS})
    folds["pair_mean"] = 1 / 3
    summary = pd.DataFrame({c: [np.nan] for c in SUMMARY_COLUMNS})
    paths = write_loo_tables(LooResult(folds, summary), out_dir=tmp_path)
    assert "0.3333333333" in paths[0].read_text(encoding="utf-8")
    assert "0.33333333333" not in paths[0].read_text(encoding="utf-8")


# --- CLI ---


def _fake_result() -> LooResult:
    return LooResult(pd.DataFrame(), pd.DataFrame({"model": ["mean"], "n_folds": [1]}))


def test_registry_has_loo_factor() -> None:
    """El registro incluye el subcomando `loo-factor`."""
    assert "loo-factor" in cli._COMMANDS


def test_cli_loo_factor_is_a_thin_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parsea, llama a compare_loo y write_loo_tables y devuelve 0."""
    calls: dict[str, object] = {}

    def fake_compare(tests: list[str], **kwargs: object) -> LooResult:
        calls["tests"], calls["kwargs"] = tests, kwargs
        return _fake_result()

    monkeypatch.setattr(cli, "compare_loo", fake_compare)
    monkeypatch.setattr(cli, "write_loo_tables", lambda result: calls.setdefault("w", result))
    assert cli.main(["loo-factor", "--tests", "2024-01-01", "2024-04-01", "--rebuild"]) == 0
    assert calls["tests"] == ["2024-01-01", "2024-04-01"] and "w" in calls
    assert calls["kwargs"] == {"rebuild": True}


def test_cli_loo_factor_defaults_to_the_twelve_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin --tests usa los 12 trimestres por defecto y sin --rebuild."""
    calls: dict[str, object] = {}

    def fake_compare(tests: list[str], **kwargs: object) -> LooResult:
        calls["tests"], calls["kwargs"] = tests, kwargs
        return _fake_result()

    monkeypatch.setattr(cli, "compare_loo", fake_compare)
    monkeypatch.setattr(cli, "write_loo_tables", lambda result: None)
    assert cli.main(["loo-factor"]) == 0
    assert calls["tests"] == list(study.DEFAULT_TESTS) and calls["kwargs"] == {"rebuild": False}


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("sin cache"),
        RuntimeError("estado cambiado al proyectar"),
        OSError("disco"),
        ConfigError("config rota"),
    ],
)
def test_cli_loo_factor_job_failure_exits_1(
    exc: Exception, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ValueError, RuntimeError, OSError y ConfigError salen con 1 y sin traceback."""

    def boom(tests: list[str], **kwargs: object) -> LooResult:
        raise exc

    monkeypatch.setattr(cli, "compare_loo", boom)
    assert cli.main(["loo-factor", "--tests", "2024-01-01"]) == 1
    assert "Traceback" not in caplog.text


def test_cli_loo_factor_write_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un OSError al escribir las tablas tambien sale con 1."""
    monkeypatch.setattr(cli, "compare_loo", lambda tests, **kw: _fake_result())

    def boom(result: LooResult) -> None:
        raise OSError("sin permisos")

    monkeypatch.setattr(cli, "write_loo_tables", boom)
    assert cli.main(["loo-factor", "--tests", "2024-01-01"]) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["loo-factor", "--tests", "2023-02-01"],  # no es mes de inicio de trimestre
        ["loo-factor", "--tests", "2024-01-01", "2023-05-01"],
        ["loo-factor", "--tests", "no-es-fecha"],
        ["loo-factor", "--tests"],  # falta el valor
        ["loo-factor", "--desconocido"],
    ],
)
def test_cli_loo_factor_invalid_arguments_exit_2(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Argumentos invalidos salen con 2 sin ejecutar el estudio."""

    def must_not_run(*args: object, **kwargs: object) -> LooResult:
        raise AssertionError("no debe ejecutarse")

    monkeypatch.setattr(cli, "compare_loo", must_not_run)
    assert cli.main(argv) == 2
