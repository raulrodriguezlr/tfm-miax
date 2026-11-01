"""Tests de la correlacion cruda (MIAX-046): calculo, estructura de un factor, figura y CLI."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import miax.viz.__main__ as cli
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.viz import (
    ReportPaths,
    build_raw_correlation_report,
    correlation_summary,
    explained_variance,
    order_by_category,
    plot_raw_correlation,
    raw_correlation,
    save_figure,
    symbols_without_data,
)

MIN_MS = 60_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
MINUTES = 1440  # un dia


def _one_factor(seed: int, t: int, n: int, beta: float) -> pd.DataFrame:
    """Retornos `beta * f + sqrt(1 - beta^2) * ruido`: varianza 1 y correlacion `beta^2` por par."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((t, 1))
    noise = rng.standard_normal((t, n))
    data = beta * factor + np.sqrt(1 - beta**2) * noise
    return pd.DataFrame(data, columns=[f"S{i}" for i in range(n)])


def _matrix(values: list[list[float]], names: list[str]) -> pd.DataFrame:
    """Matriz de correlacion pequena con nombres."""
    return pd.DataFrame(values, index=names, columns=names)


# --- raw_correlation ---


def test_raw_correlation_matches_pandas_without_gaps() -> None:
    """Sin huecos coincide con `DataFrame.corr`."""
    returns = _one_factor(0, 500, 5, 0.6)
    pd.testing.assert_frame_equal(raw_correlation(returns, min_periods=10), returns.corr())


def test_raw_correlation_symmetric_unit_diagonal_and_input_untouched() -> None:
    """Simetrica, diagonal 1 y no modifica la entrada."""
    returns = _one_factor(1, 300, 4, 0.5)
    returns.iloc[::7, 1] = np.nan
    before = returns.copy()
    corr = raw_correlation(returns, min_periods=10)
    values = corr.to_numpy()
    assert np.array_equal(values, values.T, equal_nan=True)
    assert np.all(np.diag(values) == 1.0)
    pd.testing.assert_frame_equal(returns, before)


def test_raw_correlation_min_periods_gives_nan_not_zero() -> None:
    """Un par con menos de `min_periods` observaciones comunes es `NaN`, nunca 0."""
    returns = _one_factor(2, 200, 3, 0.7)
    returns.loc[:149, "S2"] = np.nan  # S2 solo comparte 50 filas con el resto
    corr = raw_correlation(returns, min_periods=100)
    assert np.isnan(corr.loc["S0", "S2"]) and np.isnan(corr.loc["S2", "S0"])
    assert np.isfinite(corr.loc["S0", "S1"])
    assert np.isnan(corr.loc["S2", "S2"])  # con 50 datos (< min_periods) no hay diagonal 1
    assert corr.loc["S0", "S0"] == 1.0


def test_raw_correlation_constant_column_has_nan_row_column_and_diagonal() -> None:
    """Una serie constante no tiene diagonal 1: su fila, columna y diagonal son `NaN`."""
    returns = _one_factor(20, 300, 3, 0.5)
    returns["S1"] = 0.0
    corr = raw_correlation(returns, min_periods=10)
    assert corr.loc["S1"].isna().all() and corr["S1"].isna().all()
    assert corr.loc["S0", "S0"] == 1.0 and np.isfinite(corr.loc["S0", "S2"])


def test_raw_correlation_all_nan_column_has_nan_row_column_and_diagonal() -> None:
    """Una serie toda `NaN` tampoco tiene diagonal 1."""
    returns = _one_factor(21, 300, 3, 0.5)
    returns["S2"] = np.nan
    corr = raw_correlation(returns, min_periods=10)
    assert corr.loc["S2"].isna().all() and corr["S2"].isna().all()
    assert corr.loc["S0", "S0"] == 1.0


def test_raw_correlation_two_symbols() -> None:
    """Con 2 simbolos hay un unico par, simetrico, y diagonal 1."""
    returns = _one_factor(22, 300, 2, 0.8)
    corr = raw_correlation(returns, min_periods=10)
    assert corr.shape == (2, 2) and corr.loc["S0", "S1"] == corr.loc["S1", "S0"]
    assert corr.loc["S0", "S1"] == pytest.approx(returns["S0"].corr(returns["S1"]))
    assert np.all(np.diag(corr.to_numpy()) == 1.0)


def test_raw_correlation_min_periods_boundary() -> None:
    """Con exactamente `min_periods` observaciones comunes hay valor; con una menos, `NaN`."""
    returns = _one_factor(23, 200, 2, 0.7)
    returns.loc[:99, "S1"] = np.nan  # 100 filas comunes
    assert np.isfinite(raw_correlation(returns, min_periods=100).loc["S0", "S1"])
    returns.loc[100, "S1"] = np.nan  # 99 filas comunes
    assert np.isnan(raw_correlation(returns, min_periods=100).loc["S0", "S1"])


def test_raw_correlation_uses_common_observations() -> None:
    """Cada par usa solo las filas donde ambos tienen valor."""
    returns = _one_factor(3, 400, 3, 0.6)
    returns.iloc[:50, 0] = np.nan
    corr = raw_correlation(returns, min_periods=10)
    both = returns[["S0", "S1"]].dropna()
    expected = np.corrcoef(both["S0"], both["S1"])[0, 1]
    assert corr.loc["S0", "S1"] == pytest.approx(expected, abs=1e-12)


# --- estructura de un factor ---


def test_one_factor_structure_matches_theory() -> None:
    """Con `beta * f + ruido`: correlacion media `beta^2` y primer autovalor `1 + (N-1) beta^2`."""
    beta, n = 0.7, 10
    corr = raw_correlation(_one_factor(4, 30_000, n, beta), min_periods=1000)
    summary = correlation_summary(corr, factor="S0")
    assert summary["pair_mean"] == pytest.approx(beta**2, abs=0.02)
    assert summary["first_eigenvalue"] == pytest.approx(1 + (n - 1) * beta**2, abs=0.15)
    assert summary["first_eigenvalue_share"] == pytest.approx(
        (1 + (n - 1) * beta**2) / n, abs=0.015
    )
    # Los demas autovalores valen ~ 1 - beta^2: el segundo es mucho menor que el primero.
    assert summary["second_eigenvalue"] == pytest.approx(1 - beta**2, abs=0.15)


def test_independent_noise_has_no_common_factor() -> None:
    """Con ruido independiente: primer autovalor minimo y correlacion media ~ 0."""
    n = 10
    corr = raw_correlation(_one_factor(5, 30_000, n, 0.0), min_periods=1000)
    summary = correlation_summary(corr, factor="S0")
    assert abs(summary["pair_mean"]) < 0.02
    assert summary["first_eigenvalue"] < 1.15
    assert summary["pairs_above_high_pct"] == 0.0


# --- explained_variance ---


def test_explained_variance_sums_to_one_and_is_decreasing() -> None:
    """Suma 1 y esta ordenada de mayor a menor."""
    shares = explained_variance(raw_correlation(_one_factor(6, 2000, 6, 0.5), min_periods=10))
    assert shares.sum() == pytest.approx(1.0)
    assert np.all(np.diff(shares) <= 1e-12)


def test_explained_variance_uses_correlation_not_covariance() -> None:
    """Una escala distinta de un activo no cambia nada: es la matriz de correlacion."""
    returns = _one_factor(7, 3000, 5, 0.6)
    scaled = returns.assign(S0=returns["S0"] * 1000)
    a = explained_variance(raw_correlation(returns, min_periods=10))
    b = explained_variance(raw_correlation(scaled, min_periods=10))
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_explained_variance_nan_pairs_use_mean_not_zero() -> None:
    """Un par sin valor se rellena con la media de los demas, no con 0."""
    names = [f"S{i}" for i in range(6)]
    values = np.full((6, 6), 0.8)
    np.fill_diagonal(values, 1.0)
    full = _matrix(values.tolist(), names)
    holey = full.copy()
    holey.iloc[0, 1] = holey.iloc[1, 0] = np.nan  # 1 de 15 pares: 6,7 %
    np.testing.assert_allclose(
        explained_variance(holey, max_nan_share=0.1), explained_variance(full), atol=1e-12
    )


def test_explained_variance_too_many_nan_raises() -> None:
    """Si faltan mas pares que `max_nan_share`, `ValueError`."""
    names = ["A", "B", "C"]
    corr = _matrix([[1, np.nan, 0.5], [np.nan, 1, 0.5], [0.5, 0.5, 1]], names)
    with pytest.raises(ValueError, match="faltan"):
        explained_variance(corr, max_nan_share=0.05)


def test_explained_variance_counts_constant_column_pairs_and_respects_max_nan_share() -> None:
    """Los pares de una serie constante se rellenan con la media y cuentan como ausentes."""
    returns = _one_factor(24, 500, 6, 0.6)
    returns["S5"] = 1.0
    corr = raw_correlation(returns, min_periods=10)
    assert correlation_summary(corr, factor="S0", max_nan_share=0.4)["n_missing_pairs"] == 5
    assert explained_variance(corr, max_nan_share=0.4).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="faltan"):
        explained_variance(corr, max_nan_share=0.1)


def test_explained_variance_non_psd_matrix_raises() -> None:
    """Una matriz con un autovalor negativo claro es un error, no una cifra."""
    corr = _matrix([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]], ["A", "B", "C"])
    with pytest.raises(ValueError, match="semidefinida"):
        explained_variance(corr)


# --- correlation_summary ---


def test_correlation_summary_counts_missing_pairs() -> None:
    """`n_missing_pairs` cuenta los pares sin valor una sola vez, sin la diagonal."""
    names = ["BTCUSDT", "A", "B", "C"]
    nan = np.nan
    corr = _matrix(
        [[1, 0.5, 0.6, 0.7], [0.5, 1, nan, 0.4], [0.6, nan, 1, 0.3], [0.7, 0.4, 0.3, 1]], names
    )
    summary = correlation_summary(corr, max_nan_share=0.2)
    assert summary["n_missing_pairs"] == 1 and summary["n_pairs"] == 5


def test_correlation_summary_by_hand() -> None:
    """Cifras a mano, con el umbral `> high` estricto y los simbolos de minimo y maximo."""
    names = ["BTCUSDT", "AAA", "BBB", "CCC"]
    corr = _matrix(
        [
            [1.0, 0.5, 0.9, 0.2],
            [0.5, 1.0, 0.6, 0.4],
            [0.9, 0.6, 1.0, 0.3],
            [0.2, 0.4, 0.3, 1.0],
        ],
        names,
    )
    summary = correlation_summary(corr, high=0.5)
    pairs = [0.5, 0.9, 0.2, 0.6, 0.4, 0.3]
    assert summary["n_pairs"] == 6
    assert summary["pair_mean"] == pytest.approx(np.mean(pairs))
    assert summary["pair_median"] == pytest.approx(0.45)
    assert (summary["pair_min"], summary["pair_max"]) == (0.2, 0.9)
    # Solo 0.9 y 0.6 superan 0.5 de forma estricta: el par igual a 0.5 no cuenta.
    assert summary["pairs_above_high_pct"] == pytest.approx(2 / 6 * 100)
    assert summary["factor_median"] == pytest.approx(0.5)
    assert (summary["factor_min"], summary["factor_min_symbol"]) == (0.2, "CCC")
    assert (summary["factor_max"], summary["factor_max_symbol"]) == (0.9, "BBB")


def test_correlation_summary_unknown_factor_raises() -> None:
    """Un factor que no esta en la matriz es un error."""
    corr = _matrix([[1.0, 0.5], [0.5, 1.0]], ["A", "B"])
    with pytest.raises(ValueError, match="no esta"):
        correlation_summary(corr, factor="BTCUSDT")


# --- order_by_category ---


CATEGORIES = {"BTCUSDT": "l1", "ETHUSDT": "l1", "DOGEUSDT": "meme", "AAVEUSDT": "defi", "AAA": "ai"}


def test_order_by_category_first_grouped_and_alphabetical() -> None:
    """BTC el primero; despues por categoria y alfabetico dentro de cada una."""
    order = order_by_category(["DOGEUSDT", "ETHUSDT", "AAVEUSDT", "BTCUSDT", "AAA"], CATEGORIES)
    assert order == ["BTCUSDT", "AAA", "AAVEUSDT", "ETHUSDT", "DOGEUSDT"]


def test_order_by_category_is_deterministic_and_input_order_independent() -> None:
    """Misma entrada, mismo orden; y no depende del orden en que llegan los simbolos."""
    symbols = list(CATEGORIES)
    first = order_by_category(symbols, CATEGORIES)
    assert order_by_category(symbols, CATEGORIES) == first
    assert order_by_category(symbols[::-1], CATEGORIES) == first


def test_order_by_category_sorts_symbols_within_same_category() -> None:
    """Dentro de una categoria con varios simbolos el orden es alfabetico, no el de entrada."""
    cats = {"BTCUSDT": "l1", "ZZZ": "meme", "AAA": "meme", "MMM": "meme", "ETHUSDT": "l1"}
    expected = ["BTCUSDT", "ETHUSDT", "AAA", "MMM", "ZZZ"]
    assert order_by_category(list(cats), cats) == expected
    assert order_by_category(list(cats)[::-1], cats) == expected


def test_order_by_category_single_symbol_category() -> None:
    """Una categoria con un unico simbolo se coloca en su sitio alfabetico."""
    cats = {"BTCUSDT": "l1", "BNBUSDT": "exchange", "AAVEUSDT": "defi", "UNIUSDT": "defi"}
    order = order_by_category(list(cats), cats)
    assert order == ["BTCUSDT", "AAVEUSDT", "UNIUSDT", "BNBUSDT"]


def test_order_by_category_without_category_raises() -> None:
    """Un simbolo sin categoria es un error, no se coloca al final en silencio."""
    with pytest.raises(ValueError, match="sin categoria"):
        order_by_category(["BTCUSDT", "XYZ"], CATEGORIES)


# --- figura ---


def _small_figure() -> tuple:
    """Figura de un universo sintetico de 6 simbolos."""
    returns = _one_factor(8, 3000, 6, 0.6)
    returns.columns = ["BTCUSDT", "ETHUSDT", "AAVEUSDT", "DOGEUSDT", "AAA", "ZZZ"]
    categories = {**CATEGORIES, "ZZZ": "meme"}
    order = order_by_category(list(returns.columns), categories)
    corr = raw_correlation(returns[order], min_periods=10)
    summary = correlation_summary(corr)
    shares = explained_variance(corr)
    return plot_raw_correlation(corr, categories, summary, shares, title="prueba"), corr


def test_plot_missing_factor_correlation_draws_no_zero_bar() -> None:
    """Un simbolo sin correlacion con BTC no dibuja barra de 0 y se anota en el titulo."""
    names = ["BTCUSDT", "AAA", "BBB", "CCC"]
    nan = np.nan
    corr = _matrix(
        [[1, 0.5, nan, 0.7], [0.5, 1, 0.2, 0.4], [nan, 0.2, 1, 0.3], [0.7, 0.4, 0.3, 1]], names
    )
    cats = {"BTCUSDT": "l1", "AAA": "ai", "BBB": "ai", "CCC": "defi"}
    summary = correlation_summary(corr, max_nan_share=0.2)
    figure = plot_raw_correlation(
        corr, cats, summary, explained_variance(corr, max_nan_share=0.2), title="t"
    )
    ax_b = next(ax for ax in figure.axes if ax.get_title().startswith("(b)"))
    heights = [patch.get_height() for patch in ax_b.patches]
    assert len(heights) == 3 and not any(h == 0 for h in heights)
    assert sum(np.isnan(h) for h in heights) == 1
    assert "sin dato (1): BBB" in ax_b.get_title()


def test_save_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    """Crea un PNG (cabecera correcta) y un PDF no vacios en la carpeta pedida."""
    figure, _ = _small_figure()
    png, pdf = save_figure(figure, "fig", out_dir=tmp_path / "out")
    assert png == tmp_path / "out" / "fig.png" and pdf == tmp_path / "out" / "fig.pdf"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert pdf.stat().st_size > 1000


def test_heatmap_scale_is_fixed_to_unit_interval() -> None:
    """La escala de color va de -1 a 1 aunque todos los valores sean altos."""
    figure, _ = _small_figure()
    image = figure.axes[0].images[0]
    assert image.get_clim() == (-1.0, 1.0)


def test_plotting_does_not_import_pyplot() -> None:
    """Dibujar y guardar no importa `pyplot` (sin estado global ni ventanas)."""
    code = (
        "import sys, tempfile\n"
        "import numpy as np, pandas as pd\n"
        "from miax.viz import plot_raw_correlation, save_figure\n"
        "n = ['BTCUSDT', 'AAA', 'BBB']\n"
        "corr = pd.DataFrame(np.full((3, 3), 0.5) + np.eye(3) * 0.5, index=n, columns=n)\n"
        "cats = {'BTCUSDT': 'l1', 'AAA': 'ai', 'BBB': 'ai'}\n"
        "summary = {'factor_median': 0.5}\n"
        "fig = plot_raw_correlation(corr, cats, summary, np.array([0.7, 0.2, 0.1]), title='t')\n"
        "save_figure(fig, 'x', out_dir=tempfile.mkdtemp())\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --- punta a punta ---


SYMBOLS = {
    "BTCUSDT": ("l1", 0.0),
    "ETHUSDT": ("l1", 1.0),
    "SOLUSDT": ("l1", 2.0),
    "DOGEUSDT": ("meme", 3.0),
    "AAVEUSDT": ("defi", 4.0),
}


def _write_klines(base_dir: Path, symbol: str, close: np.ndarray) -> None:
    """Guarda velas de 1 minuto con operaciones en todos los minutos."""
    open_times = [T0 + k * MIN_MS for k in range(len(close))]
    n = len(close)
    frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1.0] * n,
            "close_time": [t + MIN_MS - 1 for t in open_times],
            "quote_asset_volume": [10.0] * n,
            "num_trades": [5] * n,
            "taker_buy_base": [0.5] * n,
            "taker_buy_quote": [5.0] * n,
            "ignore": [0.0] * n,
        },
        columns=KLINE_COLUMNS,
    ).astype(KLINE_DTYPES)
    save_klines(frame, symbol, "1m", base_dir=base_dir)


def _world(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache de un dia con un factor comun y su universo; devuelve el universo y los retornos."""
    rng = np.random.default_rng(11)
    factor = rng.standard_normal(MINUTES)
    returns = {}
    for symbol, (_, beta_shift) in SYMBOLS.items():
        beta = 0.9 - 0.1 * beta_shift
        r = 0.001 * (beta * factor + np.sqrt(1 - beta**2) * rng.standard_normal(MINUTES))
        returns[symbol] = r
        _write_klines(base_dir, symbol, 100.0 * np.exp(np.cumsum(r)))
    rows = [("2024-01-01", "2024-01-02", s, c) for s, (c, _) in SYMBOLS.items()]
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    return universe, pd.DataFrame(returns)


def _build(
    tmp_path: Path, universe: pd.DataFrame, out: str = "out", **kwargs: object
) -> ReportPaths:
    """Ejecuta el informe con todo dentro de `tmp_path`."""
    return build_raw_correlation_report(
        "2024-01-01",
        universe=universe,
        min_periods=100,
        base_dir=tmp_path / "klines",
        cache_dir=tmp_path / "panels",
        out_dir=tmp_path / out / "figures",
        tables_dir=tmp_path / out / "tables",
        **kwargs,
    )


def test_build_report_end_to_end_matches_manual_calculation(tmp_path: Path) -> None:
    """Escribe figura y tablas, y las cifras coinciden con un calculo manual independiente."""
    universe, raw = _world(tmp_path / "klines")
    paths = _build(tmp_path, universe)
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0
    assert paths.png.name == "raw_correlation_2024-01-01_1m.png"
    assert paths.summary_csv.name == "raw_correlation_summary_2024-01-01_1m.csv"
    assert paths.eda_csv.name == "panel_eda_2024-01-01_1m.csv"

    # Calculo manual con np.corrcoef y eigvalsh, sin miax.viz. El retorno de la vela k lleva
    # la hora de su cierre (D-19): la primera no tiene previa y el de la ultima cae fuera del rango.
    log_returns = raw.to_numpy()[1:-1]
    manual = np.corrcoef(log_returns, rowvar=False)
    upper = manual[np.triu_indices(len(manual), k=1)]
    eigenvalues = np.sort(np.linalg.eigvalsh(manual))[::-1]

    summary = pd.read_csv(paths.summary_csv, index_col="metric")["value"]
    assert float(summary["pair_mean"]) == pytest.approx(upper.mean(), abs=1e-6)
    assert float(summary["pair_median"]) == pytest.approx(np.median(upper), abs=1e-6)
    assert float(summary["first_eigenvalue"]) == pytest.approx(eigenvalues[0], abs=1e-6)
    assert int(summary["n_pairs"]) == 10 and int(summary["n_symbols"]) == 5

    eda = pd.read_csv(paths.eda_csv, index_col="symbol")
    # BTC primero, luego categoria (defi, l1, meme) y alfabetico dentro.
    assert list(eda.index) == ["BTCUSDT", "AAVEUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    assert list(eda["category"]) == ["l1", "defi", "l1", "l1", "meme"]
    assert eda.loc["BTCUSDT", "corr_BTCUSDT"] == pytest.approx(1.0)
    assert eda.loc["ETHUSDT", "corr_BTCUSDT"] == pytest.approx(manual[0, 1], abs=1e-6)
    assert (eda["valid_pct"] > 99).all() and (eda["return_nan_pct"] < 1).all()


def test_build_report_is_reproducible_byte_for_byte(tmp_path: Path) -> None:
    """Misma entrada, mismos CSV bytes a bytes (la segunda vez sale de la cache del panel)."""
    universe, _ = _world(tmp_path / "klines")
    first = _build(tmp_path, universe, "a")
    second = _build(tmp_path, universe, "b")
    assert first.summary_csv.read_bytes() == second.summary_csv.read_bytes()
    assert first.eda_csv.read_bytes() == second.eda_csv.read_bytes()


def test_build_report_unknown_period_or_factor_raises(tmp_path: Path) -> None:
    """Un periodo fuera del universo, o un factor ausente, dan `ValueError`."""
    universe, _ = _world(tmp_path / "klines")
    with pytest.raises(ValueError, match="no esta en el universo"):
        build_raw_correlation_report("1999-01-01", universe=universe)
    with pytest.raises(ValueError, match="no esta en el universo"):
        _build(tmp_path, universe, factor="XRPUSDT")


def test_build_report_naive_timestamp_period_raises(tmp_path: Path) -> None:
    """Un `pd.Timestamp` sin zona horaria no vale: tiene que estar en UTC."""
    universe, _ = _world(tmp_path / "klines")
    with pytest.raises(ValueError, match="UTC"):
        build_raw_correlation_report(pd.Timestamp("2024-01-01"), universe=universe)


# --- CLI ---


def test_cli_invalid_arguments_exit_2() -> None:
    """Fecha o intervalo invalidos, o subcomando desconocido, dan codigo 2."""
    assert cli.main(["raw-correlation", "--period", "no-es-fecha"]) == 2
    assert cli.main(["raw-correlation", "--interval", "5m"]) == 2
    assert cli.main(["raw-correlation", "--interval", "xx"]) == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-existe"])
    assert exc.value.code == 2


def test_cli_period_not_in_universe_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    """Un periodo que no esta en el universo da 1 y un mensaje, sin traceback."""
    assert cli.main(["raw-correlation", "--period", "1999-01-01"]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_cli_registry_maps_names_to_functions() -> None:
    """El registro de subcomandos tiene `raw-correlation`."""
    assert set(cli._COMMANDS) == {"raw-correlation"}


# --- monedas sin datos en el periodo (MIAX-213) ---

GREY = (217, 217, 217)  # "#d9d9d9", el color de las casillas sin dato


def _no_data_returns() -> pd.DataFrame:
    """Retornos con una constante (S1), una con pocos datos (S2) y una toda NaN (S3)."""
    returns = _one_factor(30, 400, 5, 0.6)
    returns["S1"] = 0.0
    returns["S2"] = np.where(np.arange(400) < 50, returns["S2"], np.nan)
    returns["S3"] = np.nan
    return returns


def test_symbols_without_data_finds_all_nan_constant_and_short_in_matrix_order() -> None:
    """Columna toda NaN, constante y con menos de `min_periods`: salen, en el orden de la matriz."""
    corr = raw_correlation(_no_data_returns(), min_periods=100)
    assert symbols_without_data(corr) == ["S1", "S2", "S3"]
    order = ["S3", "S0", "S2", "S4", "S1"]
    assert symbols_without_data(corr.loc[order, order]) == ["S3", "S2", "S1"]


def test_symbols_without_data_is_empty_when_every_diagonal_is_valid() -> None:
    """Sin ninguna diagonal NaN devuelve una lista vacia (y es una lista)."""
    corr = raw_correlation(_one_factor(31, 300, 4, 0.5), min_periods=10)
    result = symbols_without_data(corr)
    assert result == [] and isinstance(result, list)


def test_symbols_without_data_ignores_valid_diagonal_without_valid_pairs() -> None:
    """Una moneda con datos propios sin solape NO se aparta (decision documentada)."""
    returns = _one_factor(32, 400, 4, 0.6)
    returns.loc[150:, "S3"] = np.nan  # S3 tiene 150 datos...
    returns.loc[:149, ["S0", "S1", "S2"]] = np.nan  # ...y ninguna fila comun con las demas
    corr = raw_correlation(returns, min_periods=100)
    assert corr.loc["S3", "S3"] == 1.0 and corr.loc["S3"].drop("S3").isna().all()
    assert symbols_without_data(corr) == []


def _write_klines_from(base_dir: Path, symbol: str, close: np.ndarray, first_minute: int) -> None:
    """Como `_write_klines` pero empezando en el minuto `first_minute` del dia."""
    n = len(close)
    open_times = [T0 + (first_minute + k) * MIN_MS for k in range(n)]
    frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1.0] * n,
            "close_time": [t + MIN_MS - 1 for t in open_times],
            "quote_asset_volume": [10.0] * n,
            "num_trades": [5] * n,
            "taker_buy_base": [0.5] * n,
            "taker_buy_quote": [5.0] * n,
            "ignore": [0.0] * n,
        },
        columns=KLINE_COLUMNS,
    ).astype(KLINE_DTYPES)
    save_klines(frame, symbol, "1m", base_dir=base_dir)


EXTRA_CATEGORIES = {
    "NEWUSDT": "ai",
    "ZETAUSDT": "ai",
    "OLDUSDT": "gaming",
    "LASTUSDT": "zk",
    "SHORTUSDT": "ai",
}


def _universe_frame(categories: dict[str, str]) -> pd.DataFrame:
    """Universo de un periodo (2024-01-01 a 2024-01-02) con esos simbolos y categorias."""
    rows = [("2024-01-01", "2024-01-02", s, c) for s, c in categories.items()]
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol", "category"])
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    return universe


def _dead_world(
    base_dir: Path,
    *,
    dead: Sequence[str] = (),
    drop: Sequence[str] = (),
    short: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """El mundo de `_world` mas monedas sin velas (`dead`) y con 50 velas (`short`).

    Las de `drop` estan en el universo pero no tienen velas. Devuelve el universo y los retornos
    de las cinco monedas de `SYMBOLS` (los mismos que `_world`: misma semilla).
    """
    rng = np.random.default_rng(11)
    factor = rng.standard_normal(MINUTES)
    returns = {}
    for symbol, (_, beta_shift) in SYMBOLS.items():
        beta = 0.9 - 0.1 * beta_shift
        r = 0.001 * (beta * factor + np.sqrt(1 - beta**2) * rng.standard_normal(MINUTES))
        returns[symbol] = r
        if symbol not in drop:
            _write_klines_from(base_dir, symbol, 100.0 * np.exp(np.cumsum(r)), 0)
    for symbol in short:
        walk = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(50)))
        _write_klines_from(base_dir, symbol, walk, 0)
    categories = {s: c for s, (c, _) in SYMBOLS.items()}
    categories.update({s: EXTRA_CATEGORIES[s] for s in (*dead, *short)})
    return _universe_frame(categories), pd.DataFrame(returns)


def _report(tmp_path: Path, name: str, universe: pd.DataFrame, **kwargs: object) -> ReportPaths:
    """Informe con base, cache y salidas en `tmp_path / name` (varias ejecuciones por test)."""
    return build_raw_correlation_report(
        "2024-01-01",
        universe=universe,
        min_periods=100,
        base_dir=tmp_path / name / "klines",
        cache_dir=tmp_path / name / "panels",
        out_dir=tmp_path / name / "figures",
        tables_dir=tmp_path / name / "tables",
        **kwargs,
    )


class _SpyPlot:
    """Sustituto de `plot_raw_correlation` que recuerda los argumentos y llama al original."""

    def __init__(self) -> None:
        self.title = ""
        self.without_data: Sequence[str] = ()

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.title = str(kwargs["title"])
        self.without_data = kwargs["without_data"]  # type: ignore[assignment]
        return plot_raw_correlation(*args, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def spy_plot(monkeypatch: pytest.MonkeyPatch) -> _SpyPlot:
    """Intercepta el titulo y las monedas apartadas que `build_raw_correlation_report` dibuja."""
    spy = _SpyPlot()
    # `miax.viz.raw_correlation` como atributo del paquete es la funcion: se pide el modulo.
    module = sys.modules["miax.viz.raw_correlation"]
    monkeypatch.setattr(module, "plot_raw_correlation", spy)
    return spy


@pytest.fixture
def miax_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """`caplog` enganchado al logger del modulo (el logger de miax no propaga al raiz)."""
    module_logger = logging.getLogger("miax.viz.raw_correlation")
    module_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="miax.viz.raw_correlation")
    yield caplog
    module_logger.removeHandler(caplog.handler)


def _read_summary(path: Path) -> pd.Series:
    """Resumen CSV como texto tal cual (sin convertir ni perder los vacios)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, index_col="metric")
    return table["value"]


def test_report_with_a_coin_without_candles_runs_and_summarises_the_others(
    tmp_path: Path, spy_plot: _SpyPlot, miax_warnings: pytest.LogCaptureFixture
) -> None:
    """Una moneda sin velas no hace fallar el informe; las cifras son las de las demas."""
    universe, raw = _dead_world(tmp_path / "klines", dead=["NEWUSDT"])
    paths = _build(tmp_path, universe)
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0

    summary = _read_summary(paths.summary_csv)
    assert summary["n_symbols_in_universe"] == "6"
    assert summary["n_symbols_without_data"] == "1"
    assert summary["symbols_without_data"] == "NEWUSDT"
    assert summary["n_symbols"] == "5"
    assert summary["n_pairs"] == "10" and summary["n_missing_pairs"] == "0"

    # Calculo manual sobre las cinco monedas con datos, sin miax.viz (el retorno de la vela k lleva
    # la hora de su cierre, D-19: se descarta el primero y el ultimo).
    manual = np.corrcoef(raw.to_numpy()[1:-1], rowvar=False)
    upper = manual[np.triu_indices(5, k=1)]
    eigenvalues = np.sort(np.linalg.eigvalsh(manual))[::-1]
    assert float(summary["pair_mean"]) == pytest.approx(upper.mean(), abs=1e-6)
    assert float(summary["pair_median"]) == pytest.approx(np.median(upper), abs=1e-6)
    assert float(summary["pair_min"]) == pytest.approx(upper.min(), abs=1e-6)
    assert float(summary["pair_max"]) == pytest.approx(upper.max(), abs=1e-6)
    assert float(summary["first_eigenvalue"]) == pytest.approx(eigenvalues[0], abs=1e-6)
    assert float(summary["second_eigenvalue"]) == pytest.approx(eigenvalues[1], abs=1e-6)
    assert float(summary["first_eigenvalue_share"]) == pytest.approx(eigenvalues[0] / 5, abs=1e-6)

    # La tabla por simbolo SI incluye la moneda sin datos, con 0 % valido y 100 % de NaN.
    eda = pd.read_csv(paths.eda_csv, index_col="symbol")
    assert list(eda.index) == ["BTCUSDT", "NEWUSDT", "AAVEUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    assert eda.loc["NEWUSDT", "valid_pct"] == 0.0 and eda.loc["NEWUSDT", "return_nan_pct"] == 100.0
    assert np.isnan(eda.loc["NEWUSDT", "corr_BTCUSDT"])
    assert (eda.drop(index="NEWUSDT")["valid_pct"] > 99).all()

    # La figura recibe las monedas apartadas y se avisa una vez, con su nombre.
    assert list(spy_plot.without_data) == ["NEWUSDT"]
    warnings = [
        r.getMessage() for r in miax_warnings.records if r.name == "miax.viz.raw_correlation"
    ]
    assert len(warnings) == 1 and "NEWUSDT" in warnings[0] and "fuera del resumen" in warnings[0]


def test_report_with_a_short_history_coin_sets_it_aside(tmp_path: Path) -> None:
    """Una moneda con menos de `min_periods` velas (50) tambien se aparta, como una sin velas."""
    universe, _ = _dead_world(tmp_path / "klines", short=["SHORTUSDT"])
    summary = _read_summary(_build(tmp_path, universe).summary_csv)
    assert summary["symbols_without_data"] == "SHORTUSDT"
    assert summary["n_symbols_without_data"] == "1" and summary["n_symbols"] == "5"
    assert summary["n_missing_pairs"] == "0"


def test_report_common_rows_equal_the_universe_without_the_dead_coin(tmp_path: Path) -> None:
    """Con o sin la moneda muerta, las filas comunes del resumen son exactamente las mismas."""
    alive_universe, raw = _dead_world(tmp_path / "alive" / "klines")
    dead_universe, _ = _dead_world(tmp_path / "dead" / "klines", dead=["NEWUSDT"])
    alive = _read_summary(_report(tmp_path, "alive", alive_universe).summary_csv)
    dead = _read_summary(_report(tmp_path, "dead", dead_universe).summary_csv)

    # Sin moneda muerta: las 3 filas nuevas valen 5, 0 y vacio.
    assert alive["n_symbols_in_universe"] == "5"
    assert alive["n_symbols_without_data"] == "0"
    assert alive["symbols_without_data"] == ""

    # Calculo directo con las funciones publicas sobre la matriz de las cinco monedas.
    order = ["BTCUSDT", "AAVEUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    corr = raw_correlation(raw.iloc[1:-1][order], min_periods=100)
    for key, value in correlation_summary(corr).items():
        if isinstance(value, str):
            assert alive[key] == value, key
        else:
            assert float(alive[key]) == pytest.approx(value, abs=1e-8), key

    # Con la moneda muerta, todas las filas salvo las 3 del universo son identicas, byte a byte.
    own = {"n_symbols_in_universe", "n_symbols_without_data", "symbols_without_data"}
    assert list(dead.index) == list(alive.index)
    assert all(alive[k] == dead[k] for k in alive.index if k not in own)
    # Las 3 filas nuevas van justo despues de min_periods.
    assert list(alive.index)[3:7] == [
        "min_periods",
        "n_symbols_in_universe",
        "n_symbols_without_data",
        "symbols_without_data",
    ]


def test_report_joins_several_dead_coins_with_semicolon_in_matrix_order(tmp_path: Path) -> None:
    """Varias monedas muertas se unen con `;` en el orden de la matriz, no el de entrada."""
    dead = ["LASTUSDT", "OLDUSDT", "ZETAUSDT"]  # matriz: ZETA (ai), OLD (gaming), LAST (zk)
    universe, _ = _dead_world(tmp_path / "klines", dead=dead)
    summary = _read_summary(_build(tmp_path, universe).summary_csv)
    assert summary["symbols_without_data"] == "ZETAUSDT;OLDUSDT;LASTUSDT"
    assert summary["n_symbols_without_data"] == "3"
    assert summary["n_symbols_in_universe"] == "8" and summary["n_symbols"] == "5"
    assert summary["n_missing_pairs"] == "0"


def test_report_title_counts_symbols_with_data_only_when_some_are_set_aside(
    tmp_path: Path, spy_plot: _SpyPlot
) -> None:
    """El titulo general dice "N criptoactivos (M con datos)" solo si hay monedas apartadas."""
    dead_universe, _ = _dead_world(tmp_path / "dead" / "klines", dead=["NEWUSDT"])
    _report(tmp_path, "dead", dead_universe)
    assert "entre 6 criptoactivos (5 con datos), trimestre desde 2024-01-01" in spy_plot.title

    alive_universe, _ = _dead_world(tmp_path / "alive" / "klines")
    _report(tmp_path, "alive", alive_universe)
    assert "entre 5 criptoactivos, trimestre desde 2024-01-01" in spy_plot.title
    assert "con datos" not in spy_plot.title and list(spy_plot.without_data) == []


def test_report_factor_without_data_raises(tmp_path: Path) -> None:
    """Si el factor (BTC) no tiene datos en el periodo, `ValueError` claro."""
    universe, _ = _dead_world(tmp_path / "klines", drop=["BTCUSDT"])
    with pytest.raises(ValueError, match="no tiene datos"):
        _build(tmp_path, universe)


def test_report_fewer_than_two_symbols_with_data_raises_before_warning(
    tmp_path: Path, miax_warnings: pytest.LogCaptureFixture
) -> None:
    """Con menos de 2 monedas con datos, `ValueError` y sin el aviso de "fuera del resumen"."""
    # Control positivo: el aviso si llega a `miax_warnings` cuando hay una moneda apartada.
    ok_universe, _ = _dead_world(tmp_path / "ok" / "klines", dead=["NEWUSDT"])
    _report(tmp_path, "ok", ok_universe)
    assert any("fuera del resumen" in r.getMessage() for r in miax_warnings.records)
    miax_warnings.clear()

    drop = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "AAVEUSDT"]  # solo BTC tiene datos
    universe, _ = _dead_world(tmp_path / "klines", drop=drop)
    with pytest.raises(ValueError, match="menos de 2 simbolos con datos"):
        _build(tmp_path, universe)
    assert not any("fuera del resumen" in r.getMessage() for r in miax_warnings.records)


def _late_world(base_dir: Path) -> pd.DataFrame:
    """BTC y 4 monedas con velas solo desde el minuto 300; LATE, con 150 velas, solo antes."""
    rng = np.random.default_rng(12)
    categories = {s: c for s, (c, _) in SYMBOLS.items()}
    factor = rng.standard_normal(MINUTES - 300)
    for symbol in SYMBOLS:
        r = 0.001 * (0.7 * factor + 0.7 * rng.standard_normal(MINUTES - 300))
        _write_klines_from(base_dir, symbol, 100.0 * np.exp(np.cumsum(r)), 300)
    late = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(150)))
    _write_klines_from(base_dir, "LATEUSDT", late, 0)
    categories["LATEUSDT"] = "ai"
    return _universe_frame(categories)


def test_report_still_fails_on_missing_pairs_between_coins_with_data(tmp_path: Path) -> None:
    """Una moneda con datos propios y sin solape con nadie sigue haciendo saltar el 5 %."""
    universe = _late_world(tmp_path / "klines")
    with pytest.raises(ValueError, match="faltan"):
        _build(tmp_path, universe)

    # Con el limite subido pasa, y los 5 pares de LATE cuentan como ausentes (no se aparta).
    summary = _read_summary(_build(tmp_path, universe, "wide", max_nan_share=0.4).summary_csv)
    assert summary["n_symbols_without_data"] == "0" and summary["symbols_without_data"] == ""
    assert summary["n_symbols"] == "6" and summary["n_missing_pairs"] == "5"


# --- figura con monedas sin datos ---


def _figure_with_dead(dead: Sequence[str]) -> tuple[Figure, pd.DataFrame]:
    """Figura de 4 monedas con datos (BTC primero) y las de `dead` sin datos."""
    live = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"]
    returns = _one_factor(33, 600, len(live) + len(dead), 0.6)
    returns.columns = [*live, *dead]
    for symbol in dead:
        returns[symbol] = np.nan
    categories = {s: "l1" if s == "BTCUSDT" else "ai" for s in returns.columns}
    corr = raw_correlation(returns, min_periods=10)
    without = symbols_without_data(corr)
    kept = corr.drop(index=without, columns=without)
    figure = plot_raw_correlation(
        corr,
        categories,
        correlation_summary(kept),
        explained_variance(kept),
        title="prueba",
        without_data=without,
    )
    return figure, corr


def _title_c(figure: Figure) -> str:
    """Titulo del panel (c)."""
    return next(ax.get_title() for ax in figure.axes if ax.get_title().startswith("(c)"))


def test_heatmap_does_not_paint_missing_cells_as_zero() -> None:
    """Las casillas NaN van enmascaradas y con el color "bad" gris, no con el 0 (blanco)."""
    before = tuple(matplotlib.colormaps["RdBu_r"].get_bad())
    figure, corr = _figure_with_dead(["XXXUSDT", "YYYUSDT"])
    image = figure.axes[0].images[0]
    array = image.get_array()
    bad = np.ma.getmaskarray(array) | np.isnan(np.ma.getdata(array))
    np.testing.assert_array_equal(bad, corr.isna().to_numpy())
    assert bad.sum() == 2 * 6 * 2 - 4  # 2 filas + 2 columnas de 6, sin contar dos veces el cruce
    assert tuple(image.get_cmap().get_bad()) == matplotlib.colors.to_rgba("#d9d9d9")
    assert tuple(matplotlib.colormaps["RdBu_r"].get_bad()) == before
    assert image.get_cmap() is not matplotlib.colormaps["RdBu_r"]


def test_heatmap_missing_cell_is_rendered_grey_not_white() -> None:
    """En el render, una casilla sin dato sale gris y una con dato no."""
    figure, corr = _figure_with_dead(["XXXUSDT"])
    ax = figure.axes[0]
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())
    height = pixels.shape[0]

    def colour(row: int, col: int) -> tuple[int, int, int]:
        x, y = ax.transData.transform((col, row))
        r, g, b, _ = pixels[round(height - y), round(x)]
        return int(r), int(g), int(b)

    dead = list(corr.index).index("XXXUSDT")
    assert colour(dead, 0) == GREY and colour(0, dead) == GREY and colour(dead, dead) == GREY
    assert colour(1, 2) != GREY and colour(0, 0) != GREY


def test_panel_c_title_lists_the_dropped_coins() -> None:
    """Con 1 apartada: "(1): X"; con exactamente 6 las lista todas; con mas de 6, abrevia."""
    title = _title_c(_figure_with_dead(["XXXUSDT"])[0])
    assert title.startswith("(c) Varianza explicada") and "\n" in title
    assert title.endswith("sin datos, fuera del calculo (1): XXX")

    six = [f"D{i}USDT" for i in range(6)]
    title = _title_c(_figure_with_dead(six)[0])
    assert "(6): D0, D1, D2, D3, D4, D5" in title and "..." not in title

    eight = [f"D{i}USDT" for i in range(8)]
    title = _title_c(_figure_with_dead(eight)[0])
    assert "(8): D0, D1, D2, D3, D4, D5, ..." in title
    assert "D6" not in title and "D7" not in title


def test_panel_c_title_has_no_extra_line_without_dropped_coins() -> None:
    """Sin monedas apartadas el titulo de (c) es el de siempre, de una sola linea."""
    title = _title_c(_figure_with_dead([])[0])
    assert "\n" not in title and "sin datos" not in title


def test_grey_note_only_appears_when_the_map_has_nan() -> None:
    """La nota "Gris = sin dato" sale si el mapa tiene algun NaN, y solo entonces."""

    def has_note(figure: Figure) -> bool:
        return any("Gris = sin dato" in t.get_text() for t in figure.texts)

    assert has_note(_figure_with_dead(["XXXUSDT"])[0])
    assert not has_note(_figure_with_dead([])[0])

    # Tambien con un NaN suelto entre monedas con datos (sin ninguna apartada).
    nan = np.nan
    corr = _matrix(
        [[1, 0.5, 0.6, 0.7], [0.5, 1, nan, 0.4], [0.6, nan, 1, 0.3], [0.7, 0.4, 0.3, 1]],
        ["BTCUSDT", "AAA", "BBB", "CCC"],
    )
    cats = {"BTCUSDT": "l1", "AAA": "ai", "BBB": "ai", "CCC": "defi"}
    figure = plot_raw_correlation(
        corr,
        cats,
        correlation_summary(corr, max_nan_share=0.2),
        explained_variance(corr, max_nan_share=0.2),
        title="t",
    )
    assert has_note(figure)
