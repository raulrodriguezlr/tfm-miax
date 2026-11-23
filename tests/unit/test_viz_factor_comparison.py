"""Tests de la comparativa de factores (MIAX-055): split, comparacion, tabla, informe y CLI."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import miax.viz.__main__ as cli
from miax import viz
from miax.features.factor import FactorModel
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.viz import (
    FACTOR_MODELS,
    FactorComparisonPaths,
    build_factor_comparison_report,
    compare_factors,
    factor_comparison_table,
    split_period,
)

# Se pide el modulo por `sys.modules`: como atributo del paquete el nombre puede estar pisado.
fc_module = sys.modules["miax.viz.factor_comparison"]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AAVEUSDT", "XRPUSDT"]


def _returns(start: str, days: int, freq: str, seed: int = 0, beta: float = 0.7) -> pd.DataFrame:
    """Retornos de un factor comun (`beta * f + ruido`) con indice UTC desde `start`."""
    index = pd.date_range(start, periods=days * pd.Timedelta("1D") // pd.Timedelta(freq), freq=freq)
    index = index.tz_localize("UTC")
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((len(index), 1))
    noise = rng.standard_normal((len(index), len(SYMBOLS)))
    data = 0.001 * (beta * factor + np.sqrt(1 - beta**2) * noise)
    return pd.DataFrame(data, index=index, columns=SYMBOLS)


def _small_models() -> dict[str, Callable[[], FactorModel]]:
    """Fabricas con `min_periods` pequeno para tests rapidos."""
    from miax.features.factor_btc import BtcFactor

    return {
        "btc": lambda: BtcFactor(min_periods=100),
        "mean": lambda: MeanFactor(min_periods=100),
        "pca": lambda: PcaFactor(min_periods=100),
    }


@pytest.fixture(scope="module")
def quarter() -> pd.DataFrame:
    """Un trimestre sintetico a 1 hora (2160 filas)."""
    return _returns("2025-01-01", 90, "1h", seed=1)


# --- split_period ---


def test_split_period_train_before_test_without_overlap_and_union_is_input(
    quarter: pd.DataFrame,
) -> None:
    """Train antes que test, sin solape, y juntos reconstruyen la entrada."""
    train, test = split_period(quarter, train_months=2)
    assert train.index.max() < test.index.min()
    assert not train.index.intersection(test.index).size
    assert train.index[0] == quarter.index[0]
    assert train.index.max() < pd.Timestamp("2025-03-01", tz="UTC") <= test.index.min()
    pd.testing.assert_frame_equal(pd.concat([train, test]), quarter)


def test_split_period_invalid_inputs_raise(quarter: pd.DataFrame) -> None:
    """`train_months < 1`, entrada vacia o test vacio dan `ValueError`."""
    with pytest.raises(ValueError, match="train_months"):
        split_period(quarter, train_months=0)
    with pytest.raises(ValueError, match="vacio"):
        split_period(quarter.iloc[:0])
    with pytest.raises(ValueError, match="vacio"):
        split_period(quarter, train_months=6)  # el test queda vacio


# --- compare_factors ---


def test_compare_factors_structure_and_values(quarter: pd.DataFrame) -> None:
    """Indice = factores en orden, columnas esperadas, caida alta y suelo -1/(N-1)."""
    table = compare_factors(quarter, models=_small_models(), min_periods=100)
    assert list(table.index) == ["btc", "mean", "pca"] and table.index.name == "factor"
    assert list(table.columns) == list(fc_module._ROW_COLUMNS)
    train, test = split_period(quarter)
    assert (table["train_rows"] == len(train)).all() and (table["test_rows"] == len(test)).all()
    assert (table["error"] == "").all()
    assert (table["n_symbols"] == 6).all()
    for name in ("mean", "pca"):
        # Con N=6 y un factor, la residual queda en ~ -1/(N-1) = -0.2 frente a 0.49 cruda.
        assert table.loc[name, "correlation_drop"] > 0.5
        assert table.loc[name, "residual_pair_mean"] == pytest.approx(-1 / 5, abs=0.03)
        assert table.loc[name, "signed_floor"] == pytest.approx(-1 / 5)
        assert table.loc[name, "residual_pair_abs_mean"] < table.loc[name, "raw_pair_abs_mean"]
    # BTC sin residuo propio queda fuera de la metrica: N util = 5 y suelo -1/4.
    assert table.loc["btc", "n_symbols_without_data"] == 1
    assert table.loc["btc", "signed_floor"] == pytest.approx(-1 / 4)


def test_compare_factors_is_deterministic(quarter: pd.DataFrame) -> None:
    """Dos ejecuciones con la misma entrada dan la misma tabla."""
    first = compare_factors(quarter, models=_small_models(), min_periods=100)
    second = compare_factors(quarter, models=_small_models(), min_periods=100)
    pd.testing.assert_frame_equal(first, second)


def test_compare_factors_default_models_are_factor_models() -> None:
    """Sin `models` usa `FACTOR_MODELS` (btc, mean, pca, en ese orden)."""
    assert list(FACTOR_MODELS) == ["btc", "mean", "pca"]
    returns = _returns("2025-01-01", 90, "10min", seed=2)
    table = compare_factors(returns)
    assert list(table.index) == ["btc", "mean", "pca"]
    assert (table["error"] == "").all()


def test_compare_factors_failing_model_leaves_error_row_without_stopping_others(
    quarter: pd.DataFrame,
) -> None:
    """BTC sin su columna falla con `error` y NaN; media y PCA salen igual."""
    table = compare_factors(
        quarter.drop(columns="BTCUSDT"), models=_small_models(), min_periods=100
    )
    assert "BTCUSDT" in table.loc["btc", "error"]
    metrics = [c for c in fc_module._METRIC_COLUMNS if c not in ("symbols_without_data",)]
    assert table.loc["btc", metrics].isna().all()
    assert table.loc["btc", "train_rows"] > 0 and table.loc["btc", "test_rows"] > 0
    for name in ("mean", "pca"):
        assert table.loc[name, "error"] == ""
        assert table.loc[name, "correlation_drop"] > 0.4  # N=5: suelo -0.25 frente a 0.49


def test_compare_factors_impossible_split_marks_every_row_with_error(
    quarter: pd.DataFrame,
) -> None:
    """Si el periodo no se puede partir, todas las filas llevan `error` y NaN."""
    table = compare_factors(quarter, train_months=6, models=_small_models(), min_periods=100)
    assert list(table.index) == ["btc", "mean", "pca"]
    assert (table["error"] != "").all()
    assert table["correlation_drop"].isna().all()
    empty = compare_factors(quarter.iloc[:0], models=_small_models())
    assert (empty["error"] != "").all() and list(empty.index) == ["btc", "mean", "pca"]


def test_compare_factors_builds_a_new_model_per_factor_and_period(quarter: pd.DataFrame) -> None:
    """La fabrica se llama una vez por factor y periodo: nunca se reutiliza un modelo ajustado."""
    built: list[FactorModel] = []

    def factory() -> FactorModel:
        model = MeanFactor(min_periods=100)
        built.append(model)
        return model

    compare_factors(quarter, models={"mean": factory}, min_periods=100)
    compare_factors(quarter, models={"mean": factory}, min_periods=100)
    assert len(built) == 2 and built[0] is not built[1]


def test_compare_factors_fits_only_on_train_no_leakage(quarter: pd.DataFrame) -> None:
    """Cambiar el test no cambia las betas ajustadas: el ajuste solo ve el train."""
    seen: list[pd.DataFrame] = []

    class Spy(MeanFactor):
        def fit(self, returns: pd.DataFrame) -> Spy:  # type: ignore[override]
            seen.append(returns)
            super().fit(returns)
            return self

    train, test = split_period(quarter)
    compare_factors(quarter, models={"mean": lambda: Spy(min_periods=100)}, min_periods=100)
    assert len(seen) == 1
    assert seen[0].index.max() < test.index.min()
    pd.testing.assert_frame_equal(seen[0], train)


# --- factor_comparison_table ---

TABLE_COLUMNS = [
    "period_start",
    "factor",
    "error",
    "n_symbols",
    "n_symbols_without_data",
    "correlation_drop",
    "raw_pair_abs_mean",
    "residual_pair_abs_mean",
    "residual_pair_mean",
    "signed_floor",
    "residual_first_eigenvalue_share",
]


def _long(
    rows: list[tuple[str, str, float]],
    overrides: dict[tuple[str, str], dict[str, object]] | None = None,
) -> pd.DataFrame:
    """DataFrame largo (periodo, factor, caida) con 40 monedas y el resto de columnas fijas."""
    records = []
    for start, factor, drop in rows:
        records.append(
            {
                "period_start": start,
                "factor": factor,
                "error": "",
                "n_symbols": 40,
                "n_symbols_without_data": 0,
                "correlation_drop": drop,
                "raw_pair_abs_mean": 0.5,
                "residual_pair_abs_mean": 0.5 * (1 - drop),
                "residual_pair_mean": 0.01,
                "signed_floor": -1 / 39,
                "residual_first_eigenvalue_share": 0.05,
            }
        )
    frame = pd.DataFrame(records, columns=TABLE_COLUMNS)
    for (start, factor), values in (overrides or {}).items():
        mask = (frame["period_start"] == start) & (frame["factor"] == factor)
        for column, value in values.items():
            frame.loc[mask, column] = value
    return frame


QUARTERS = ["2024-01-01", "2024-04-01", "2024-07-01"]
DROPS = {"btc": [0.2, 0.4, 0.9], "mean": [0.8, 0.85, 0.9], "pca": [0.7, 0.9, 0.95]}


def _full_long() -> pd.DataFrame:
    """Tres trimestres x tres factores, con las filas mezcladas."""
    rows = [(q, f, DROPS[f][i]) for i, q in enumerate(QUARTERS) for f in ("pca", "btc", "mean")]
    return _long(rows)


def test_factor_comparison_table_shape_order_and_medians() -> None:
    """Filas y orden exactos, columnas en el orden de `FACTOR_MODELS` y medianas a mano."""
    table = factor_comparison_table(_full_long(), min_symbols=30)
    assert list(table.columns) == list(FACTOR_MODELS)
    assert list(table.index) == [
        "n_periods",
        "correlation_drop_median",
        "correlation_drop_min",
        "correlation_drop_max",
        "raw_pair_abs_mean_median",
        "residual_pair_abs_mean_median",
        "residual_pair_mean_median",
        "signed_floor_median",
        "residual_first_eigenvalue_share_median",
        "first_period",
        "last_period",
    ]
    assert table.loc["correlation_drop_median"].to_dict() == pytest.approx(
        {"btc": 0.4, "mean": 0.85, "pca": 0.9}
    )
    assert table.loc["correlation_drop_min", "btc"] == 0.2
    assert table.loc["correlation_drop_max", "pca"] == 0.95
    assert table.loc["residual_pair_abs_mean_median", "btc"] == pytest.approx(0.5 * 0.6)
    assert table.loc["signed_floor_median", "mean"] == pytest.approx(-1 / 39)
    assert (table.loc["n_periods"] == 3).all()
    assert (table.loc["first_period"] == "2024-01-01").all()
    assert (table.loc["last_period"] == "2024-07-01").all()


def test_factor_comparison_table_excludes_whole_quarter_when_a_factor_has_error() -> None:
    """Si un factor falla en un trimestre, ese trimestre sale para los tres factores."""
    long = _long(
        [(q, f, DROPS[f][i]) for i, q in enumerate(QUARTERS) for f in FACTOR_MODELS],
        {("2024-04-01", "btc"): {"error": "falta BTCUSDT", "correlation_drop": np.nan}},
    )
    table = factor_comparison_table(long)
    assert (table.loc["n_periods"] == 2).all()
    # Sin el trimestre central: btc [0.2, 0.9] -> 0.55; mean [0.8, 0.9] -> 0.85.
    assert table.loc["correlation_drop_median", "btc"] == pytest.approx(0.55)
    assert table.loc["correlation_drop_median", "mean"] == pytest.approx(0.85)
    assert (table.loc["last_period"] == "2024-07-01").all()


def test_factor_comparison_table_excludes_quarter_below_min_symbols_for_all_factors() -> None:
    """Un factor bajo `min_symbols` saca el trimestre entero, aunque los otros lo cumplan."""
    long = _long(
        [(q, f, DROPS[f][i]) for i, q in enumerate(QUARTERS) for f in FACTOR_MODELS],
        {("2024-07-01", "pca"): {"n_symbols": 20}},
    )
    table = factor_comparison_table(long, min_symbols=30)
    assert (table.loc["n_periods"] == 2).all()
    assert (table.loc["last_period"] == "2024-04-01").all()
    assert table.loc["correlation_drop_median", "mean"] == pytest.approx(0.825)


def test_factor_comparison_table_counts_symbols_with_data_not_universe() -> None:
    """`min_symbols` se compara con `n_symbols - n_symbols_without_data`."""
    long = _long(
        [(q, f, DROPS[f][i]) for i, q in enumerate(QUARTERS) for f in FACTOR_MODELS],
        {("2024-01-01", "mean"): {"n_symbols_without_data": 15}},  # 40 - 15 = 25 < 30
    )
    assert (factor_comparison_table(long, min_symbols=30).loc["n_periods"] == 2).all()
    assert (factor_comparison_table(long, min_symbols=25).loc["n_periods"] == 3).all()


def test_factor_comparison_table_no_valid_quarter_raises() -> None:
    """Si ningun trimestre es valido, `ValueError`."""
    with pytest.raises(ValueError, match="ningun periodo valido"):
        factor_comparison_table(_full_long(), min_symbols=100)
    long = _long([(QUARTERS[0], f, 0.5) for f in FACTOR_MODELS])
    long["error"] = "roto"
    with pytest.raises(ValueError, match="ningun periodo valido"):
        factor_comparison_table(long)


def test_factor_comparison_table_treats_nan_error_as_no_error() -> None:
    """Un `error` NaN (CSV releido) cuenta como vacio."""
    long = _full_long()
    long["error"] = np.nan
    assert (factor_comparison_table(long).loc["n_periods"] == 3).all()


# --- build_factor_comparison_report ---


def _universe() -> pd.DataFrame:
    """Dos periodos de 2025 y uno de 2026 con 6 monedas cada uno."""
    periods = [
        ("2025-01-01", "2025-04-01"),
        ("2025-04-01", "2025-07-01"),
        ("2026-01-01", "2026-04-01"),
    ]
    rows = [(s, e, sym, "l1") for s, e in periods for sym in SYMBOLS]
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    return universe


@pytest.fixture
def fake_panels(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    """Sustituye `cached_period_panels` por paneles sinteticos y recuerda el universo recibido."""
    received: list[pd.DataFrame] = []

    def fake(universe: pd.DataFrame, interval: str, **kwargs: object) -> Iterator[tuple]:
        received.append(universe)
        for i, start in enumerate(sorted(universe["period_start"].unique())):
            returns = _returns(f"{start:%Y-%m-%d}", 90, "10min", seed=10 + i)
            yield pd.Timestamp(start), SimpleNamespace(features={"return": returns})

    monkeypatch.setattr(fc_module, "cached_period_panels", fake)
    return received


def test_build_report_writes_both_csv_and_leaves_reserved_period_out(
    tmp_path: Path, fake_panels: list[pd.DataFrame]
) -> None:
    """Escribe los dos CSV, el periodo de 2026 no aparece (D-37) y el resumen usa `%.10g`."""
    tables = tmp_path / "tables"
    paths = build_factor_comparison_report(universe=_universe(), min_symbols=5, tables_dir=tables)
    assert isinstance(paths, FactorComparisonPaths)
    assert paths.by_period_csv == tables / "factor_comparison_by_period.csv"
    assert paths.summary_csv == tables / "factor_comparison.csv"
    assert paths.by_period_csv.is_file() and paths.summary_csv.is_file()

    # Al lector de paneles solo le llegan los periodos que acaban antes de la reserva.
    assert set(fake_panels[0]["period_start"].dt.strftime("%Y-%m-%d")) == {
        "2025-01-01",
        "2025-04-01",
    }

    by_period = pd.read_csv(paths.by_period_csv, keep_default_na=False)
    assert set(by_period["period_start"]) == {"2025-01-01", "2025-04-01"}
    assert not by_period["period_start"].str.startswith("2026").any()
    assert list(by_period["factor"]) == ["btc", "mean", "pca"] * 2
    assert (by_period["error"] == "").all()

    summary = pd.read_csv(paths.summary_csv, dtype=str, keep_default_na=False, index_col="metric")
    assert list(summary.columns) == ["btc", "mean", "pca"]
    assert (summary.loc["n_periods"] == "2").all()
    assert (summary.loc["first_period"] == "2025-01-01").all()
    assert (summary.loc["last_period"] == "2025-04-01").all()
    for text in summary.drop(index=["first_period", "last_period"]).to_numpy().ravel():
        assert f"{float(text):.10g}" == text
    median = by_period[by_period["factor"] == "mean"]["correlation_drop"].median()
    assert float(summary.loc["correlation_drop_median", "mean"]) == pytest.approx(median, abs=1e-8)


def test_build_report_is_reproducible_byte_for_byte(
    tmp_path: Path, fake_panels: list[pd.DataFrame]
) -> None:
    """Misma entrada, mismos CSV bytes a bytes."""
    first = build_factor_comparison_report(
        universe=_universe(), min_symbols=5, tables_dir=tmp_path / "a"
    )
    second = build_factor_comparison_report(
        universe=_universe(), min_symbols=5, tables_dir=tmp_path / "b"
    )
    assert first.by_period_csv.read_bytes() == second.by_period_csv.read_bytes()
    assert first.summary_csv.read_bytes() == second.summary_csv.read_bytes()


def test_build_report_naive_reserved_from_raises(
    tmp_path: Path, fake_panels: list[pd.DataFrame]
) -> None:
    """`reserved_from` sin zona horaria es un `ValueError`."""
    with pytest.raises(ValueError, match="UTC"):
        build_factor_comparison_report(
            universe=_universe(), reserved_from=pd.Timestamp("2026-01-01"), tables_dir=tmp_path
        )


def test_build_report_without_periods_before_reserve_raises(
    tmp_path: Path, fake_panels: list[pd.DataFrame]
) -> None:
    """Un universo sin periodos antes de la reserva da `ValueError`."""
    only_2026 = _universe()
    only_2026 = only_2026[only_2026["period_start"].dt.year == 2026]
    with pytest.raises(ValueError, match="ningun periodo termina antes"):
        build_factor_comparison_report(universe=only_2026, tables_dir=tmp_path)


def test_build_report_period_with_failed_split_keeps_error_rows_and_is_left_out_of_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un periodo imposible de partir no aborta: lleva `error` y no entra al resumen."""

    def fake(universe: pd.DataFrame, interval: str, **kwargs: object) -> Iterator[tuple]:
        for i, start in enumerate(sorted(universe["period_start"].unique())):
            days = 90 if i == 0 else 20  # el segundo no admite 2 meses de train + test
            returns = _returns(f"{start:%Y-%m-%d}", days, "10min", seed=20 + i)
            yield pd.Timestamp(start), SimpleNamespace(features={"return": returns})

    monkeypatch.setattr(fc_module, "cached_period_panels", fake)
    paths = build_factor_comparison_report(universe=_universe(), min_symbols=5, tables_dir=tmp_path)
    by_period = pd.read_csv(paths.by_period_csv, keep_default_na=False)
    bad = by_period[by_period["period_start"] == "2025-04-01"]
    assert len(bad) == 3 and (bad["error"] != "").all()
    summary = pd.read_csv(paths.summary_csv, dtype=str, keep_default_na=False, index_col="metric")
    assert (summary.loc["n_periods"] == "1").all()
    assert (summary.loc["last_period"] == "2025-01-01").all()


# --- CLI ---


@pytest.fixture
def spy_report(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Sustituye el informe por un espia que recuerda los argumentos."""
    calls: dict[str, object] = {}

    def fake(**kwargs: object) -> FactorComparisonPaths:
        calls.update(kwargs)
        return FactorComparisonPaths(Path("a.csv"), Path("b.csv"))

    monkeypatch.setattr(cli, "build_factor_comparison_report", fake)
    return calls


def test_cli_factor_comparison_ok_exits_0_and_passes_arguments(
    spy_report: dict[str, object],
) -> None:
    """Con argumentos validos devuelve 0 y pasa interval, train_months y min_symbols."""
    assert cli.main(["factor-comparison"]) == 0
    assert spy_report == {"interval": "1m", "train_months": 2, "min_symbols": 30}
    assert cli.main(["factor-comparison", "--train-months", "1", "--min-symbols", "10"]) == 0
    assert spy_report["train_months"] == 1 and spy_report["min_symbols"] == 10


@pytest.mark.parametrize(
    "args",
    [
        ["--train-months", "0"],
        ["--min-symbols", "1"],
        ["--interval", "5m"],
        ["--interval", "xx"],
    ],
)
def test_cli_factor_comparison_invalid_arguments_exit_2(
    spy_report: dict[str, object], args: list[str]
) -> None:
    """`--train-months 0`, `--min-symbols 1` o un intervalo invalido dan 2 sin llamar al informe."""
    assert cli.main(["factor-comparison", *args]) == 2
    assert spy_report == {}


def test_cli_factor_comparison_work_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Si el trabajo lanza `ValueError` da 1 y sin traceback."""

    def boom(**kwargs: object) -> FactorComparisonPaths:
        raise ValueError("ningun periodo valido")

    monkeypatch.setattr(cli, "build_factor_comparison_report", boom)
    assert cli.main(["factor-comparison"]) == 1
    assert "Traceback" not in capsys.readouterr().err


# --- exportaciones ---


def test_public_names_are_exported_from_viz() -> None:
    """Todo lo publico del modulo se exporta desde `miax.viz`."""
    for name in (
        "FACTOR_MODELS",
        "FactorComparisonPaths",
        "build_factor_comparison_report",
        "compare_factors",
        "factor_comparison_table",
        "split_period",
    ):
        assert name in viz.__all__ and hasattr(viz, name)
