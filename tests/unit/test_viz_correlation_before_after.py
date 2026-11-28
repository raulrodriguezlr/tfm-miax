"""Tests de la figura de correlacion antes y despues de residualizar (MIAX-060)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import miax.viz.__main__ as cli
from miax.features.factor import FactorModel
from miax.features.factor_btc import BtcFactor
from miax.features.factor_mean import MeanFactor
from miax.features.factor_pca import PcaFactor
from miax.viz import (
    CorrelationFigurePaths,
    build_correlation_before_after_report,
    factor_comparison_by_period,
    factor_comparison_table,
    plot_correlation_before_after,
    raw_correlation,
    residual_correlations,
    split_period,
)
from miax.viz.factor_comparison import compare_factors, usable_quarters
from miax.viz.raw_correlation import MISSING_COLOR, category_groups
from tests.unit.leakage import assert_no_lookahead

# Se piden los modulos por `sys.modules`: como atributo del paquete el nombre puede estar pisado.
cba = sys.modules["miax.viz.correlation_before_after"]
fc_module = sys.modules["miax.viz.factor_comparison"]

CATEGORIES = {
    "BTCUSDT": "l1",
    "ETHUSDT": "l1",
    "SOLUSDT": "l1",
    "DOGEUSDT": "meme",
    "SHIBUSDT": "meme",
    "AAVEUSDT": "defi",
    "UNIUSDT": "defi",
    "XRPUSDT": "pay",
}
SYMBOLS = list(CATEGORIES)
# Las columnas llegan desordenadas a proposito: la salida tiene que respetar el orden de entrada.
INPUT_ORDER = [
    "XRPUSDT",
    "BTCUSDT",
    "UNIUSDT",
    "ETHUSDT",
    "DOGEUSDT",
    "SOLUSDT",
    "AAVEUSDT",
    "SHIBUSDT",
]


def _returns(
    start: str, days: int = 90, freq: str = "1h", seed: int = 0, columns: list[str] | None = None
) -> pd.DataFrame:
    """Retornos de un factor comun (`beta * f + ruido`) con indice UTC desde `start`."""
    cols = INPUT_ORDER if columns is None else columns
    periods = days * pd.Timedelta("1D") // pd.Timedelta(freq)
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((len(index), 1))
    betas = np.linspace(0.4, 0.8, len(cols))
    noise = rng.standard_normal((len(index), len(cols)))
    data = 0.001 * (betas * factor + np.sqrt(1 - betas**2) * noise)
    return pd.DataFrame(data, index=index, columns=cols)


def _small_models() -> dict[str, Callable[[], FactorModel]]:
    """Fabricas con `min_periods` pequeno para tests rapidos."""
    return {
        "btc": lambda: BtcFactor(min_periods=100),
        "mean": lambda: MeanFactor(min_periods=100),
        "pca": lambda: PcaFactor(min_periods=100),
    }


@pytest.fixture(scope="module")
def quarter() -> pd.DataFrame:
    """Un trimestre sintetico a 1 hora (2160 filas) con las columnas desordenadas."""
    return _returns("2025-01-01", seed=1)


@pytest.fixture(scope="module")
def matrices(quarter: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Las cuatro matrices del trimestre sintetico, ya ordenadas por categoria con BTC primero."""
    order = [
        "BTCUSDT",
        *sorted((s for s in SYMBOLS if s != "BTCUSDT"), key=lambda s: (CATEGORIES[s], s)),
    ]
    return residual_correlations(quarter[order], models=_small_models(), min_periods=100)


@pytest.fixture
def miax_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """`caplog` enganchado a los loggers de miax (no propagan al raiz)."""
    names = ("miax.viz.correlation_before_after", "miax.viz.factor_comparison")
    loggers = [logging.getLogger(name) for name in names]
    for module_logger in loggers:
        module_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING, logger=module_logger.name)
    yield caplog
    for module_logger in loggers:
        module_logger.removeHandler(caplog.handler)


# --- residual_correlations ---


def test_residual_correlations_keys_order_and_shape(quarter: pd.DataFrame) -> None:
    """Claves raw/btc/mean/pca; todas con las mismas filas y columnas, en el orden de entrada."""
    result = residual_correlations(quarter, models=_small_models(), min_periods=100)
    assert list(result) == ["raw", "btc", "mean", "pca"]
    for name, matrix in result.items():
        assert list(matrix.index) == INPUT_ORDER, name
        assert list(matrix.columns) == INPUT_ORDER, name


def test_residual_correlations_raw_is_the_correlation_of_the_third_month(
    quarter: pd.DataFrame,
) -> None:
    """(a) es la correlacion de la parte de test (tercer mes), no la del trimestre entero."""
    result = residual_correlations(quarter, models=_small_models(), min_periods=100)
    _, test = split_period(quarter, train_months=2)
    assert test.index[0] == pd.Timestamp("2025-03-01", tz="UTC")
    pd.testing.assert_frame_equal(result["raw"], raw_correlation(test, min_periods=100))
    whole = raw_correlation(quarter, min_periods=100)
    assert not np.allclose(result["raw"].to_numpy(), whole.to_numpy())


def test_residual_correlations_btc_is_nan_in_the_btc_factor_matrix(quarter: pd.DataFrame) -> None:
    """BTC no tiene residuo con su propio factor: su fila, columna y diagonal son NaN (D-34)."""
    result = residual_correlations(quarter, models=_small_models(), min_periods=100)
    btc = result["btc"]
    assert btc.loc["BTCUSDT"].isna().all() and btc["BTCUSDT"].isna().all()
    assert btc.drop(index="BTCUSDT", columns="BTCUSDT").notna().all().all()
    # En media y PCA, BTC si tiene residuo.
    assert result["mean"].loc["BTCUSDT", "BTCUSDT"] == 1.0
    assert result["pca"].loc["BTCUSDT", "BTCUSDT"] == 1.0


def test_residual_correlations_removes_common_factor(quarter: pd.DataFrame) -> None:
    """Quitar el factor comun baja el |rho| medio fuera de la diagonal."""
    result = residual_correlations(quarter, models=_small_models(), min_periods=100)

    def abs_mean(matrix: pd.DataFrame) -> float:
        values = matrix.to_numpy()[~np.eye(len(matrix), dtype=bool)]
        return float(np.nanmean(np.abs(values)))

    for name in ("mean", "pca"):
        assert abs_mean(result[name]) < 0.5 * abs_mean(result["raw"]), name


def test_residual_correlations_factor_without_its_symbol_is_all_nan_and_does_not_raise(
    quarter: pd.DataFrame, miax_warnings: pytest.LogCaptureFixture
) -> None:
    """Sin BTC en los datos, el factor BTC da una matriz toda NaN con aviso; el resto sale."""
    returns = quarter.drop(columns="BTCUSDT")
    result = residual_correlations(returns, models=_small_models(), min_periods=100)
    columns = list(returns.columns)
    assert result["btc"].isna().all().all()
    assert list(result["btc"].index) == columns and list(result["btc"].columns) == columns
    assert result["mean"].notna().all().all() and result["pca"].notna().all().all()
    assert any("Factor btc" in r.getMessage() for r in miax_warnings.records)


class _Boom(FactorModel):
    """Modelo que siempre falla al ajustar."""

    def get_state(self) -> dict[str, object]:
        return {}

    def _fit(self, returns: pd.DataFrame) -> None:
        raise ValueError("no ajusta")

    def _factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("no se llama")

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("no se llama")


class _FewerColumns(MeanFactor):
    """Media cuyo residuo pierde una columna: la validacion de `transform` lo rechaza."""

    def _residuals(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return super()._residuals(returns, factors).iloc[:, :-1]


def test_residual_correlations_failing_models_give_all_nan_matrix_without_raising(
    quarter: pd.DataFrame, miax_warnings: pytest.LogCaptureFixture
) -> None:
    """Un modelo que lanza ValueError, o devuelve menos columnas, deja su matriz toda NaN."""
    models: dict[str, Callable[[], FactorModel]] = {
        "boom": _Boom,
        "short": lambda: _FewerColumns(min_periods=100),
        "mean": lambda: MeanFactor(min_periods=100),
    }
    result = residual_correlations(quarter, models=models, min_periods=100)
    assert list(result) == ["raw", "boom", "short", "mean"]
    for name in ("boom", "short"):
        assert result[name].isna().all().all(), name
        assert list(result[name].index) == INPUT_ORDER
        assert list(result[name].columns) == INPUT_ORDER
    assert result["mean"].notna().all().all()
    assert result["raw"].notna().all().all()
    messages = [r.getMessage() for r in miax_warnings.records]
    assert any("boom" in m for m in messages) and any("short" in m for m in messages)


def test_residual_correlations_model_returning_fewer_columns_after_transform_is_reindexed(
    quarter: pd.DataFrame,
) -> None:
    """Un modelo que devuelve una columna menos en `transform` conserva la forma (esa, NaN)."""

    class Dropper:
        def fit(self, returns: pd.DataFrame) -> Dropper:
            return self

        def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
            return returns.drop(columns="XRPUSDT")

    result = residual_correlations(
        quarter,
        models={"drop": lambda: Dropper()},  # type: ignore[dict-item,return-value]
        min_periods=100,
    )
    matrix = result["drop"]
    assert list(matrix.index) == INPUT_ORDER and list(matrix.columns) == INPUT_ORDER
    assert matrix.loc["XRPUSDT"].isna().all() and matrix["XRPUSDT"].isna().all()
    assert matrix.drop(index="XRPUSDT", columns="XRPUSDT").notna().all().all()


def test_residual_correlations_is_deterministic(quarter: pd.DataFrame) -> None:
    """Misma entrada, mismas matrices."""
    first = residual_correlations(quarter, models=_small_models(), min_periods=100)
    second = residual_correlations(quarter, models=_small_models(), min_periods=100)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_residual_correlations_default_models_and_invalid_train_months(
    quarter: pd.DataFrame,
) -> None:
    """Sin `models` usa los tres factores por defecto; `train_months < 1` es ValueError."""
    result = residual_correlations(_returns("2025-01-01", freq="10min", seed=4), min_periods=1000)
    assert list(result) == ["raw", "btc", "mean", "pca"]
    with pytest.raises(ValueError, match="train_months"):
        residual_correlations(quarter, train_months=0, models=_small_models(), min_periods=100)


# --- no-fuga ---


class _SpyModels:
    """Fabricas que recuerdan los modelos y las filas con las que se ajustaron."""

    def __init__(self) -> None:
        self.built: dict[str, FactorModel] = {}
        self.fitted_on: dict[str, pd.DataFrame] = {}

    def factories(self) -> dict[str, Callable[[], FactorModel]]:
        spy = self

        def make(name: str, base: type[FactorModel]) -> Callable[[], FactorModel]:
            class Spy(base):  # type: ignore[valid-type,misc]
                def fit(self, returns: pd.DataFrame) -> Spy:
                    spy.fitted_on[name] = returns.copy()
                    super().fit(returns)
                    return self

            def factory() -> FactorModel:
                model = Spy(min_periods=100)
                spy.built[name] = model
                return model

            return factory

        return {
            "btc": make("btc", BtcFactor),
            "mean": make("mean", MeanFactor),
            "pca": make("pca", PcaFactor),
        }


def test_residual_correlations_fit_uses_only_the_first_two_months(quarter: pd.DataFrame) -> None:
    """Cada factor se ajusta solo con filas anteriores al tercer mes (fit == train de split)."""
    spy = _SpyModels()
    residual_correlations(quarter, models=spy.factories(), min_periods=100)
    train, test = split_period(quarter, train_months=2)
    assert set(spy.fitted_on) == {"btc", "mean", "pca"}
    for name, fitted in spy.fitted_on.items():
        assert fitted.index.max() < test.index.min(), name
        pd.testing.assert_frame_equal(fitted, train)


def test_residual_correlations_fitted_state_ignores_the_third_month(quarter: pd.DataFrame) -> None:
    """Perturbar desde el inicio del tercer mes no cambia el estado ajustado (MIAX-058)."""
    cut = quarter.index[0] + pd.DateOffset(months=2)

    def run(data: pd.DataFrame) -> object:
        spy = _SpyModels()
        residual_correlations(data, models=spy.factories(), min_periods=100)
        return {name: model.get_state() for name, model in spy.built.items()}

    assert_no_lookahead(run, quarter, cut, what="estado ajustado de los factores")
    assert_no_lookahead(run, quarter, cut, move_nans=True, what="estado con NaN movidos")


def test_residual_correlations_a_leaky_fit_would_be_detected(quarter: pd.DataFrame) -> None:
    """Control positivo: ajustar con todo el trimestre cambia el estado al perturbar el futuro."""
    cut = quarter.index[0] + pd.DateOffset(months=2)

    def leaky(data: pd.DataFrame) -> object:
        return MeanFactor(min_periods=100).fit(data).get_state()

    with pytest.raises(AssertionError, match="fuga de futuro"):
        assert_no_lookahead(leaky, quarter, cut)


def test_residual_correlations_raw_changes_with_the_third_month_only(
    quarter: pd.DataFrame,
) -> None:
    """Lo pintado es del tercer mes: perturbar los dos primeros no cambia el crudo."""
    cut = quarter.index[0] + pd.DateOffset(months=2)
    base = residual_correlations(quarter, models=_small_models(), min_periods=100)["raw"]
    changed = quarter.copy()
    changed.loc[changed.index < cut, "ETHUSDT"] *= 3.0
    after = residual_correlations(changed, models=_small_models(), min_periods=100)["raw"]
    pd.testing.assert_frame_equal(base, after)


# --- plot_correlation_before_after ---


def _by_period(quarters: list[str], raw: list[float] | None = None) -> pd.DataFrame:
    """Formato largo (periodo, factor) con las cifras que pinta (f)."""
    rows = []
    for i, start in enumerate(quarters):
        for factor, residual in (("btc", 0.2), ("mean", 0.05), ("pca", 0.04)):
            rows.append(
                {
                    "period_start": start,
                    "factor": factor,
                    "raw_pair_abs_mean": 0.4 + 0.01 * i if raw is None else raw[i],
                    "residual_pair_abs_mean": residual,
                }
            )
    return pd.DataFrame(rows)


def _figure(matrices: dict[str, pd.DataFrame], by_period: pd.DataFrame, **kwargs: object) -> Figure:
    return plot_correlation_before_after(
        matrices, CATEGORIES, by_period, period_label="2025-01-01", title="prueba", **kwargs
    )  # type: ignore[arg-type]


def _line(ax: object, label: str) -> np.ndarray:
    line = next(line for line in ax.lines if line.get_label() == label)  # type: ignore[attr-defined]
    return np.asarray(line.get_ydata(), dtype="float64")


QUARTERS = ["2024-10-01", "2025-01-01", "2025-04-01"]


def test_plot_has_seven_axes_four_maps_with_fixed_scale_and_all_titles(
    matrices: dict[str, pd.DataFrame],
) -> None:
    """4 mapas + barra de color + (e) + (f); los mapas con clim (-1, 1) y titulos (a)-(f)."""
    figure = _figure(matrices, _by_period(QUARTERS), highlight="2025-01-01")
    assert len(figure.axes) == 7
    maps = [ax for ax in figure.axes if ax.images]
    assert len(maps) == 4
    for ax in maps:
        assert ax.images[0].get_clim() == (-1.0, 1.0)
        assert ax.images[0].get_cmap().name == "RdBu_r"
    titles = [ax.get_title() for ax in figure.axes]
    for letter in "abcdef":
        assert sum(t.startswith(f"({letter})") for t in titles) == 1, (letter, titles)
    assert [ax.get_title().split("\n")[0][:3] for ax in maps] == ["(a)", "(b)", "(c)", "(d)"]
    assert figure.texts and "2 primeros meses" in figure.texts[-1].get_text()


def test_plot_maps_share_order_and_missing_cells_are_grey(
    matrices: dict[str, pd.DataFrame],
) -> None:
    """Los cuatro mapas pintan las mismas casillas en el mismo orden; los NaN van en gris."""
    figure = _figure(matrices, _by_period(QUARTERS), highlight="2025-01-01")
    maps = [ax for ax in figure.axes if ax.images]
    for ax, name in zip(maps, ["raw", "btc", "mean", "pca"], strict=True):
        data = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)
        np.testing.assert_allclose(data, matrices[name].to_numpy(), equal_nan=True)
        assert tuple(ax.images[0].get_cmap().get_bad()) == matplotlib_rgba(MISSING_COLOR)
    # Etiquetas de monedas solo en (a).
    assert maps[0].get_xticks().size == len(matrices["raw"]) and maps[1].get_xticks().size == 0


def matplotlib_rgba(color: str) -> tuple[float, ...]:
    """Color como tupla RGBA."""
    from matplotlib.colors import to_rgba

    return tuple(to_rgba(color))


def test_plot_raw_line_in_f_equals_raw_pair_abs_mean_of_the_mean_row(
    matrices: dict[str, pd.DataFrame],
) -> None:
    """La linea "Crudo" de (f) en el trimestre marcado es el `raw_pair_abs_mean` de la fila mean."""
    by_period = _by_period(QUARTERS, raw=[0.31, 0.47, 0.52])
    # La fila "mean" manda: se contamina el crudo de las otras filas para que no se confundan.
    by_period.loc[by_period["factor"] != "mean", "raw_pair_abs_mean"] = 0.99
    figure = _figure(matrices, by_period, highlight="2025-01-01")
    ax_f = next(ax for ax in figure.axes if ax.get_title().startswith("(f)"))
    raw = _line(ax_f, "Crudo")
    assert raw.tolist() == [0.31, 0.47, 0.52]
    highlighted = QUARTERS.index("2025-01-01")
    mean_row = by_period[
        (by_period["factor"] == "mean") & (by_period["period_start"] == "2025-01-01")
    ]
    assert raw[highlighted] == mean_row["raw_pair_abs_mean"].iloc[0]
    assert _line(ax_f, "Residuo media").tolist() == [0.05] * 3
    assert _line(ax_f, "Residuo BTC").tolist() == [0.2] * 3


def test_plot_titles_show_abs_mean_with_decimal_comma(matrices: dict[str, pd.DataFrame]) -> None:
    """Cada mapa lleva su |rho| medio fuera de la diagonal con coma decimal."""
    figure = _figure(matrices, _by_period(QUARTERS), highlight="2025-01-01")
    maps = [ax for ax in figure.axes if ax.images]
    for ax, name in zip(maps, ["raw", "btc", "mean", "pca"], strict=True):
        values = matrices[name].to_numpy()[~np.eye(len(matrices[name]), dtype=bool)]
        expected = f"{np.nanmean(np.abs(values)):.2f}".replace(".", ",")
        assert ax.get_title().endswith(f"|ρ| medio {expected}"), (name, ax.get_title())


def test_plot_absent_highlight_warns_and_does_not_fail(
    matrices: dict[str, pd.DataFrame], miax_warnings: pytest.LogCaptureFixture
) -> None:
    """Un trimestre marcado que no esta en (f) avisa por el logger y la figura sale."""
    figure = _figure(matrices, _by_period(QUARTERS), highlight="2019-01-01")
    assert len(figure.axes) == 7
    assert any("2019-01-01" in r.getMessage() for r in miax_warnings.records)


def test_plot_empty_by_period_does_not_fail(matrices: dict[str, pd.DataFrame]) -> None:
    """Sin trimestres para (f), la figura sale igual con el aviso en el panel."""
    empty = pd.DataFrame(
        columns=["period_start", "factor", "raw_pair_abs_mean", "residual_pair_abs_mean"]
    )
    figure = _figure(matrices, empty, highlight="2025-01-01")
    assert len(figure.axes) == 7
    ax_f = next(ax for ax in figure.axes if ax.get_title().startswith("(f)"))
    assert any("sin trimestres" in t.get_text() for t in ax_f.texts)


def test_plot_all_nan_matrix_does_not_fail(matrices: dict[str, pd.DataFrame]) -> None:
    """Una matriz toda NaN (factor que fallo) se pinta en gris con "sin dato" en el titulo."""
    broken = {**matrices, "pca": matrices["pca"] * np.nan}
    figure = _figure(broken, _by_period(QUARTERS), highlight="2025-01-01")
    titles = [ax.get_title() for ax in figure.axes if ax.images]
    assert "sin dato" in titles[3]


def test_plot_missing_matrix_raises(matrices: dict[str, pd.DataFrame]) -> None:
    """Faltar una de las cuatro matrices es ValueError."""
    partial = {k: v for k, v in matrices.items() if k != "pca"}
    with pytest.raises(ValueError, match="faltan matrices"):
        _figure(partial, _by_period(QUARTERS))


def test_plot_draws_floor_line_in_e(matrices: dict[str, pd.DataFrame]) -> None:
    """(e) marca el suelo mecanico -1/(N-1) con N las monedas con datos."""
    figure = _figure(matrices, _by_period(QUARTERS), highlight="2025-01-01")
    ax_e = next(ax for ax in figure.axes if ax.get_title().startswith("(e)"))
    n = len(matrices["raw"])
    xs = [line.get_xdata()[0] for line in ax_e.lines]
    assert pytest.approx(-1 / (n - 1)) in xs


# --- build_correlation_before_after_report (extremo a extremo) ---


def _universe() -> pd.DataFrame:
    """Dos trimestres de 2025 y uno de 2026 con 8 monedas cada uno."""
    periods = [
        ("2025-01-01", "2025-04-01"),
        ("2025-04-01", "2025-07-01"),
        ("2026-01-01", "2026-04-01"),
    ]
    rows = [(s, e, sym, CATEGORIES[sym]) for s, e in periods for sym in SYMBOLS]
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    return universe


def _period_returns(start: str, columns: list[str]) -> pd.DataFrame:
    """Retornos del periodo, siempre los mismos para el mismo inicio (semilla por fecha)."""
    seed = int(pd.Timestamp(start).strftime("%y%m%d"))
    return _returns(start, days=90, freq="10min", seed=seed, columns=SYMBOLS)[columns]


@pytest.fixture
def fake_data(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Sustituye `cached_panel` y `cached_period_panels` por paneles sinteticos coherentes."""
    calls: list[tuple] = []

    def fake_panel(
        symbols: list[str], interval: str, start_ms: int, end_ms: int, **kw: object
    ) -> SimpleNamespace:
        calls.append((list(symbols), interval, start_ms, end_ms))
        start = f"{pd.Timestamp(start_ms, unit='ms'):%Y-%m-%d}"
        return SimpleNamespace(features={"return": _period_returns(start, list(symbols))})

    def fake_periods(universe: pd.DataFrame, interval: str, **kwargs: object) -> Iterator[tuple]:
        for start in sorted(universe["period_start"].unique()):
            label = f"{pd.Timestamp(start):%Y-%m-%d}"
            yield (
                pd.Timestamp(start),
                SimpleNamespace(features={"return": _period_returns(label, SYMBOLS)}),
            )

    monkeypatch.setattr(cba, "cached_panel", fake_panel)
    monkeypatch.setattr(fc_module, "cached_period_panels", fake_periods)
    return calls


class _SpyPlot:
    """Sustituto de `plot_correlation_before_after` que guarda las figuras y los argumentos."""

    def __init__(self) -> None:
        self.figures: list[Figure] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> Figure:
        figure = plot_correlation_before_after(*args, **kwargs)  # type: ignore[arg-type]
        self.figures.append(figure)
        self.kwargs.append(kwargs)
        return figure


@pytest.fixture
def spy_plot(monkeypatch: pytest.MonkeyPatch) -> _SpyPlot:
    """Intercepta la figura que dibuja `build_correlation_before_after_report`."""
    spy = _SpyPlot()
    monkeypatch.setattr(cba, "plot_correlation_before_after", spy)
    return spy


def _build(tmp_path: Path, name: str = "out", **kwargs: object) -> CorrelationFigurePaths:
    options: dict[str, object] = {
        "universe": _universe(),
        "min_symbols": 5,
        "base_dir": tmp_path / "klines",
        "cache_dir": tmp_path / "panels",
        "out_dir": tmp_path / name / "figures",
        "tables_dir": tmp_path / name / "tables",
    }
    options.update(kwargs)
    return build_correlation_before_after_report("2025-01-01", **options)  # type: ignore[arg-type]


def test_build_report_writes_png_pdf_and_csv(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """Escribe PNG, PDF y CSV con los nombres esperados en las carpetas pedidas."""
    paths = _build(tmp_path)
    assert isinstance(paths, CorrelationFigurePaths)
    assert paths.png == tmp_path / "out" / "figures" / "correlation_before_after_2025-01-01_1m.png"
    assert paths.pdf == tmp_path / "out" / "figures" / "correlation_before_after_2025-01-01_1m.pdf"
    assert paths.csv == tmp_path / "out" / "tables" / "correlation_before_after_2025-01-01_1m.csv"
    assert paths.png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths.pdf.read_bytes()[:5] == b"%PDF-"
    assert paths.csv.stat().st_size > 0
    # El panel se pide una vez, para el trimestre elegido, con BTC y todas las monedas.
    assert len(fake_data) == 1
    assert set(fake_data[0][0]) == set(SYMBOLS) and fake_data[0][1] == "1m"
    assert len(spy_plot.figures) == 1 and len(spy_plot.figures[0].axes) == 7


def test_build_report_csv_rows_columns_and_number_format(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """Una fila por matriz (raw, btc, mean, pca), columnas esperadas y `%.10g`."""
    paths = _build(tmp_path)
    table = pd.read_csv(paths.csv, dtype=str, keep_default_na=False)
    assert list(table.columns) == list(cba._CSV_COLUMNS)
    assert list(table["matrix"]) == ["raw", "btc", "mean", "pca"]
    assert (table["period_start"] == "2025-01-01").all()
    assert list(table["n_symbols"]) == ["8", "7", "8", "8"]  # BTC sin residuo con su factor
    assert list(table["n_pairs"]) == ["28", "21", "28", "28"]
    for column in ("pair_abs_mean", "pair_mean", "first_eigenvalue_share", "correlation_drop"):
        for text in table[column]:
            if text != "":
                assert f"{float(text):.10g}" == text, (column, text)
    # El crudo no tiene caida; los tres factores si, y es positiva.
    assert table.loc[0, "correlation_drop"] == ""
    assert (table.loc[1:, "correlation_drop"].astype(float) > 0).all()
    assert (table["pair_abs_mean"].astype(float) > 0).all()


def test_build_report_csv_matches_independent_calculation(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """Las cifras del CSV coinciden con un calculo independiente sobre el tercer mes."""
    paths = _build(tmp_path)
    table = pd.read_csv(paths.csv, index_col="matrix")
    returns = _period_returns("2025-01-01", SYMBOLS)
    _, test = split_period(returns, train_months=2)
    corr = test.corr(min_periods=1000).to_numpy()
    upper = corr[np.triu_indices(len(corr), k=1)]
    assert table.loc["raw", "pair_abs_mean"] == pytest.approx(np.abs(upper).mean(), abs=1e-8)
    assert table.loc["raw", "pair_mean"] == pytest.approx(upper.mean(), abs=1e-8)
    # La caida del CSV es la de `compare_factors` del mismo trimestre.
    drops = compare_factors(returns).set_index(pd.Index(["btc", "mean", "pca"]))["correlation_drop"]
    for name in ("btc", "mean", "pca"):
        assert table.loc[name, "correlation_drop"] == pytest.approx(drops[name], abs=1e-8)


def test_build_report_raw_abs_mean_agrees_across_csv_title_and_panel_f(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """El |rho| del crudo es el mismo en el CSV, en el titulo de (a) y en la linea de (f)."""
    paths = _build(tmp_path)
    table = pd.read_csv(paths.csv, index_col="matrix")
    raw_value = float(table.loc["raw", "pair_abs_mean"])
    figure = spy_plot.figures[0]
    maps = [ax for ax in figure.axes if ax.images]
    expected = f"{raw_value:.2f}".replace(".", ",")
    assert maps[0].get_title().endswith(f"|ρ| medio {expected}")
    ax_f = next(ax for ax in figure.axes if ax.get_title().startswith("(f)"))
    raw_line = _line(ax_f, "Crudo")
    # Dos trimestres de 2025 con 8 monedas (>= min_symbols=5); el marcado es el primero.
    assert len(raw_line) == 2
    assert raw_line[0] == pytest.approx(raw_value, abs=1e-8)
    assert spy_plot.kwargs[0]["highlight"] == "2025-01-01"


def test_build_report_is_reproducible_byte_for_byte(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """Misma entrada, mismo CSV (y mismo PNG) bytes a bytes en dos ejecuciones."""
    first = _build(tmp_path, "a")
    second = _build(tmp_path, "b")
    assert first.csv.read_bytes() == second.csv.read_bytes()
    assert first.png.read_bytes() == second.png.read_bytes()


def test_build_report_trims_f_to_quarters_with_enough_symbols(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """Con `min_symbols` mayor que las monedas, (f) queda sin trimestres y la figura sale."""
    _build(tmp_path, min_symbols=30)
    ax_f = next(ax for ax in spy_plot.figures[0].axes if ax.get_title().startswith("(f)"))
    assert any("sin trimestres" in t.get_text() for t in ax_f.texts)


def test_build_report_reserved_2026_period_raises_citing_d37(
    tmp_path: Path, fake_data: list[tuple]
) -> None:
    """Un periodo de 2026 no se pinta: ValueError que cita D-37, y no se toca ningun panel."""
    with pytest.raises(ValueError, match="D-37"):
        build_correlation_before_after_report(
            "2026-01-01", universe=_universe(), tables_dir=tmp_path, out_dir=tmp_path
        )
    assert fake_data == []


def test_build_report_period_ending_after_reserved_from_raises(
    tmp_path: Path, fake_data: list[tuple]
) -> None:
    """Con `reserved_from` anterior al fin del periodo tambien falla, aunque sea de 2025."""
    with pytest.raises(ValueError, match="D-37"):
        _build(tmp_path, reserved_from=pd.Timestamp("2025-03-01", tz="UTC"))
    assert fake_data == []


def test_build_report_period_ending_exactly_at_reserved_from_is_allowed(
    tmp_path: Path, fake_data: list[tuple], spy_plot: _SpyPlot
) -> None:
    """`period_end == reserved_from` vale (el fin es exclusivo)."""
    paths = _build(tmp_path, reserved_from=pd.Timestamp("2025-04-01", tz="UTC"))
    assert paths.csv.is_file()


def test_build_report_naive_period_or_reserved_from_raises(
    tmp_path: Path, fake_data: list[tuple]
) -> None:
    """Un periodo (o `reserved_from`) sin zona horaria es ValueError."""
    with pytest.raises(ValueError, match="UTC"):
        build_correlation_before_after_report(pd.Timestamp("2025-01-01"), universe=_universe())
    with pytest.raises(ValueError, match="UTC"):
        _build(tmp_path, reserved_from=pd.Timestamp("2026-01-01"))


def test_build_report_period_outside_universe_or_without_btc_raises(
    tmp_path: Path, fake_data: list[tuple]
) -> None:
    """Un periodo fuera del universo, o sin BTC, es ValueError."""
    with pytest.raises(ValueError, match="no esta en el universo"):
        build_correlation_before_after_report("1999-01-01", universe=_universe())
    no_btc = _universe()
    no_btc = no_btc[no_btc["symbol"] != "BTCUSDT"]
    with pytest.raises(ValueError, match="BTCUSDT no esta en el universo"):
        _build(tmp_path, universe=no_btc)
    assert fake_data == []


# --- refactors de MIAX-046 y MIAX-055 ---


def test_category_groups_consecutive_spans_with_factor_in_its_own() -> None:
    """Tramos consecutivos (etiqueta, inicio, fin); el factor va en el suyo aunque sea de l1."""
    order = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT", "DOGEUSDT", "SHIBUSDT"]
    assert category_groups(order, CATEGORIES, "BTCUSDT") == [
        ("BTC", 0, 1),
        ("l1", 1, 3),
        ("defi", 3, 4),
        ("meme", 4, 6),
    ]
    assert category_groups(["AAVEUSDT"], CATEGORIES, "BTCUSDT") == [("defi", 0, 1)]


def test_raw_correlation_module_keeps_its_figure_unchanged_after_refactor() -> None:
    """`plot_raw_correlation` sigue pintando con el mismo gris y los mismos tramos."""
    assert MISSING_COLOR == "#d9d9d9"
    raw_module = sys.modules["miax.viz.raw_correlation"]
    assert raw_module.category_groups is category_groups


def _long(drops: dict[str, float], **overrides: dict[str, object]) -> pd.DataFrame:
    """Formato largo de dos trimestres y tres factores con 40 monedas."""
    rows = []
    for start in ("2024-01-01", "2024-04-01"):
        for factor, drop in drops.items():
            rows.append(
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
    frame = pd.DataFrame(rows)
    for key, values in overrides.items():
        start, factor = key.split("|")
        mask = (frame["period_start"] == start) & (frame["factor"] == factor)
        for column, value in values.items():
            frame.loc[mask, column] = value
    return frame


def _old_inline_filter(by_period: pd.DataFrame, min_symbols: int) -> pd.DataFrame:
    """El filtro tal y como estaba dentro de `factor_comparison_table` antes del refactor."""
    errors = by_period["error"].fillna("").astype(str)
    usable = by_period["n_symbols"] - by_period["n_symbols_without_data"]
    row_ok = (errors == "") & (usable >= min_symbols)
    quarter_ok = row_ok.groupby(by_period["period_start"]).transform("all")
    return by_period[quarter_ok]


def test_usable_quarters_matches_the_previous_inline_filter() -> None:
    """`usable_quarters` da las mismas filas que el filtro que habia dentro de la tabla."""
    drops = {"btc": 0.3, "mean": 0.8, "pca": 0.9}
    cases = [
        _long(drops),
        _long(drops, **{"2024-04-01|btc": {"error": "falta BTCUSDT"}}),
        _long(drops, **{"2024-01-01|pca": {"n_symbols": 20}}),
        _long(drops, **{"2024-01-01|mean": {"n_symbols_without_data": 15}}),
    ]
    for frame in cases:
        for min_symbols in (5, 25, 30, 41):
            pd.testing.assert_frame_equal(
                usable_quarters(frame, min_symbols=min_symbols),
                _old_inline_filter(frame, min_symbols),
            )
    assert usable_quarters(cases[1])["period_start"].unique().tolist() == ["2024-01-01"]
    assert usable_quarters(cases[0], min_symbols=41).empty


def test_factor_comparison_table_still_summarises_usable_quarters() -> None:
    """La tabla resumen sale de `usable_quarters`: un trimestre invalido no cuenta en ninguno."""
    long = _long({"btc": 0.3, "mean": 0.8, "pca": 0.9}, **{"2024-04-01|btc": {"error": "x"}})
    table = factor_comparison_table(long)
    assert (table.loc["n_periods"] == 1).all()
    assert (table.loc["last_period"] == "2024-01-01").all()


def test_factor_comparison_by_period_equals_compare_factors_per_period(
    fake_data: list[tuple],
) -> None:
    """Una sola fila por (periodo, factor), en formato largo, igual que `compare_factors` a mano."""
    result = factor_comparison_by_period(_universe())
    expected = []
    for start in ("2025-01-01", "2025-04-01"):
        part = compare_factors(_period_returns(start, SYMBOLS))
        part.insert(0, "period_start", start)
        expected.append(part.reset_index())
    pd.testing.assert_frame_equal(result, pd.concat(expected, ignore_index=True))
    assert list(result["factor"]) == ["btc", "mean", "pca"] * 2


def test_factor_comparison_by_period_leaves_reserved_out_and_validates(
    fake_data: list[tuple],
) -> None:
    """2026 no entra (D-37); `reserved_from` sin zona o sin periodos previos es ValueError."""
    result = factor_comparison_by_period(_universe())
    assert set(result["period_start"]) == {"2025-01-01", "2025-04-01"}
    with pytest.raises(ValueError, match="UTC"):
        factor_comparison_by_period(_universe(), reserved_from=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError, match="ningun periodo"):
        factor_comparison_by_period(_universe(), reserved_from=pd.Timestamp("2020-01-01", tz="UTC"))


# --- CLI ---


def _fake_paths(tmp_path: Path) -> CorrelationFigurePaths:
    return CorrelationFigurePaths(tmp_path / "f.png", tmp_path / "f.pdf", tmp_path / "f.csv")


def test_cli_success_exits_0_and_passes_the_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Con la funcion sustituida el subcomando devuelve 0 y le pasa los argumentos."""
    seen: dict[str, object] = {}

    def fake(period: object, **kwargs: object) -> CorrelationFigurePaths:
        seen["period"] = period
        seen.update(kwargs)
        return _fake_paths(tmp_path)

    monkeypatch.setattr(cli, "build_correlation_before_after_report", fake)
    code = cli.main(
        [
            "correlation-before-after",
            "--period",
            "2024-04-01",
            "--train-months",
            "1",
            "--min-symbols",
            "10",
            "--rebuild",
        ]
    )
    assert code == 0
    assert seen == {
        "period": pd.Timestamp("2024-04-01", tz="UTC"),
        "interval": "1m",
        "train_months": 1,
        "min_symbols": 10,
        "rebuild": True,
    }
    # Por defecto: 2025-01-01, 2 meses, 30 monedas, sin reconstruir.
    assert cli.main(["correlation-before-after"]) == 0
    assert seen["period"] == pd.Timestamp("2025-01-01", tz="UTC")
    assert (seen["train_months"], seen["min_symbols"], seen["rebuild"]) == (2, 30, False)


@pytest.mark.parametrize(
    "argv",
    [
        ["--period", "no-es-fecha"],
        ["--interval", "5m"],
        ["--interval", "xx"],
        ["--train-months", "0"],
        ["--min-symbols", "1"],
    ],
)
def test_cli_invalid_arguments_exit_2_without_running_the_job(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Periodo mal escrito, intervalo distinto de 1m, --train-months 0 o --min-symbols 1: 2."""
    called: list[object] = []
    monkeypatch.setattr(
        cli, "build_correlation_before_after_report", lambda *a, **k: called.append(a)
    )
    assert cli.main(["correlation-before-after", *argv]) == 2
    assert called == []
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("error", [ValueError("boom"), RuntimeError("boom"), OSError("boom")])
def test_cli_job_error_exits_1_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    """Si el trabajo lanza un error esperado, el codigo es 1 y no hay traceback."""

    def fail(*args: object, **kwargs: object) -> CorrelationFigurePaths:
        raise error

    monkeypatch.setattr(cli, "build_correlation_before_after_report", fail)
    assert cli.main(["correlation-before-after"]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err


def test_cli_real_function_with_reserved_period_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    """Un periodo reservado (2026) llega a la funcion real y la CLI lo traduce en 1."""
    assert cli.main(["correlation-before-after", "--period", "2026-01-01"]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
