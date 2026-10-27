"""Tests de la correlacion cruda (MIAX-046): calculo, estructura de un factor, figura y CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
