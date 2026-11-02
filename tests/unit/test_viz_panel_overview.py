"""Tests del resumen del panel (MIAX-214): tablas por periodo, validez por barra y correlacion."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from miax.features import GapCause, Panel, build_time_grid, panel_store
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.viz import (
    PanelOverview,
    ReportPaths,
    bar_validity_pct,
    build_panel_overview,
    build_raw_correlation_report,
    correlation_by_liquidity,
    correlation_summary,
    explained_variance,
    feature_quantiles,
    period_gap_causes,
    raw_correlation,
    summarize_correlation,
    symbol_quality,
)

MIN_MS = 60_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
DAY = 1440  # minutos por dia
GAP_NAMES = [cause.name for cause in GapCause]

_FEATURES = ("return", "realized_vol", "rel_volume", "rel_trades")


@pytest.fixture
def module_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """`caplog` enganchado a los loggers de miax (no propagan al raiz)."""
    names = ("miax.viz.panel_overview", "miax.viz.raw_correlation")
    loggers = [logging.getLogger(name) for name in names]
    for logger in loggers:
        logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING)
    yield caplog
    for logger in loggers:
        logger.removeHandler(caplog.handler)


def _warnings(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    """Mensajes de aviso (o peor) del logger `name`."""
    return [
        r.getMessage() for r in caplog.records if r.name == name and r.levelno >= logging.WARNING
    ]


# --- paneles y cache sinteticos ---------------------------------------------------------------


def _panel(
    values: dict[str, np.ndarray],
    valid: np.ndarray,
    symbols: Sequence[str],
    *,
    start_min: int = 1,
) -> Panel:
    """Panel con `values` (feature -> matriz T x N) y `valid` (T x N); cierres desde `start_min`."""
    n = valid.shape[0]
    index = build_time_grid(T0 + start_min * MIN_MS, T0 + (start_min + n) * MIN_MS, "1m")
    columns = list(symbols)
    features = {
        name: pd.DataFrame(np.asarray(matrix, dtype="float64"), index=index, columns=columns)
        for name, matrix in values.items()
    }
    return Panel(
        features=features,
        valid=pd.DataFrame(np.asarray(valid, dtype=bool), index=index, columns=columns),
        interval="1m",
        symbols=tuple(columns),
    )


def _valid_panel(valid: np.ndarray, symbols: Sequence[str], *, start_min: int = 1) -> Panel:
    """Panel con solo la feature `return` a cero y la validez dada."""
    return _panel({"return": np.zeros(valid.shape)}, valid, symbols, start_min=start_min)


def _oracle_bar_pct(valid: pd.DataFrame, horizon_min: int, share: float = 0.8) -> dict[str, float]:
    """% de barras validas a mano, con bucles: ultimo minuto valido y `share` de minutos validos."""
    # `valid` va indexado al cierre: la vela de la fila `tau` se abrio en `tau - 1 min`.
    open_min = (valid.index.as_unit("ms").asi8 // MIN_MS - 1).tolist()
    first, last = open_min[0], open_min[-1]
    start = -(-first // horizon_min) * horizon_min  # primer multiplo del horizonte >= first
    result: dict[str, float] = {}
    for symbol in valid.columns:
        flags = dict(zip(open_min, valid[symbol].tolist(), strict=True))
        bars: list[bool] = []
        begin = start
        while begin + horizon_min - 1 <= last:
            minutes = [flags[m] for m in range(begin, begin + horizon_min)]
            bars.append(minutes[-1] and sum(minutes) / horizon_min >= share)
            begin += horizon_min
        result[str(symbol)] = 100 * sum(bars) / len(bars) if bars else float("nan")
    return result


def _klines_frame(open_min: Sequence[int], close: np.ndarray, trades: np.ndarray) -> pd.DataFrame:
    """Velas de 1 minuto abiertas en los minutos dados (desde `T0`)."""
    n = len(open_min)
    open_times = [T0 + int(k) * MIN_MS for k in open_min]
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1.0] * n,
            "close_time": [t + MIN_MS - 1 for t in open_times],
            "quote_asset_volume": [10.0] * n,
            "num_trades": trades,
            "taker_buy_base": [0.5] * n,
            "taker_buy_quote": [5.0] * n,
            "ignore": [0.0] * n,
        },
        columns=KLINE_COLUMNS,
    ).astype(KLINE_DTYPES)


def _universe(rows: Sequence[tuple[str, str, str]]) -> pd.DataFrame:
    """Universo `(period_start, period_end, symbol)` con categoria y primera vela el 2024-01-01."""
    universe = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol"])
    universe["category"] = "l1"
    for column in ("period_start", "period_end"):
        universe[column] = pd.to_datetime(universe[column], utc=True).astype("datetime64[us, UTC]")
    universe["first_candle"] = pd.Timestamp("2024-01-01", tz="UTC")
    return universe


# Tres "trimestres" de un dia. BTC: hueco en [100, 108) y sin operaciones en 200 y 201. ETH: hueco
# en [100, 104). SOL: completo. X: velas solo en los dias 0 y 2 (sin velas en el trimestre 2).
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XUSDT")
PERIODS = (
    pd.Timestamp("2024-01-01", tz="UTC"),
    pd.Timestamp("2024-01-02", tz="UTC"),
    pd.Timestamp("2024-01-03", tz="UTC"),
)


def _write_world(base_dir: Path) -> np.ndarray:
    """Cache de 3 dias; devuelve los log-retornos por minuto (`3 * DAY x 4`, orden de SYMBOLS)."""
    rng = np.random.default_rng(214)
    total = 3 * DAY
    factor = rng.standard_normal(total)
    increments = 0.001 * (0.6 * factor[:, None] + 0.8 * rng.standard_normal((total, 4)))
    close = 100.0 * np.exp(np.cumsum(increments, axis=0))
    holes = {"BTCUSDT": range(100, 108), "ETHUSDT": range(100, 104)}
    for j, symbol in enumerate(SYMBOLS):
        minutes = [k for k in range(total) if k not in holes.get(symbol, ())]
        if symbol == "XUSDT":
            minutes = [k for k in range(total) if k < DAY or k >= 2 * DAY]
        trades = np.full(len(minutes), 5)
        if symbol == "BTCUSDT":
            trades[[minutes.index(200), minutes.index(201)]] = 0
        save_klines(
            _klines_frame(minutes, close[minutes, j], trades), symbol, "1m", base_dir=base_dir
        )
    return increments


def _world_universe() -> pd.DataFrame:
    """Los tres trimestres con las cuatro monedas en todos."""
    rows = [(f"2024-01-0{d}", f"2024-01-0{d + 1}", symbol) for d in (1, 2, 3) for symbol in SYMBOLS]
    return _universe(rows)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    """Cache de velas del mundo sintetico (solo se lee) y sus incrementos."""
    base_dir = tmp_path_factory.mktemp("overview_klines")
    return base_dir, _write_world(base_dir)


def _overview(world: tuple[Path, np.ndarray], cache_dir: Path, **kwargs: object) -> PanelOverview:
    """Resumen del mundo con la cache de paneles en `cache_dir` y umbrales pequenos."""
    params: dict[str, object] = {"min_periods": 100, "illiquid_below": 99.5, **kwargs}
    return build_panel_overview(
        _world_universe(), "1m", base_dir=world[0], cache_dir=cache_dir, **params
    )  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def overview(
    world: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
) -> PanelOverview:
    """Resumen por defecto del mundo sintetico, calculado una sola vez."""
    return _overview(world, tmp_path_factory.mktemp("overview_panels"))


# --- symbol_quality ---------------------------------------------------------------------------


def test_symbol_quality_by_hand() -> None:
    """% valido y % de NaN por feature, calculados a mano, con indice `symbol`."""
    nan = np.nan
    values = {
        "return": np.array([[1.0, nan], [nan, nan], [2.0, nan], [3.0, nan], [4.0, nan]]),
        "realized_vol": np.array([[1.0, 1.0]] * 5),
    }
    valid = np.array([[1, 1], [0, 1], [1, 1], [1, 1], [1, 0]], dtype=bool)
    table = symbol_quality(_panel(values, valid, ["A", "B"]))
    assert list(table.index) == ["A", "B"] and table.index.name == "symbol"
    assert list(table.columns) == ["valid_pct", "return_nan_pct", "realized_vol_nan_pct"]
    assert table.loc["A", "valid_pct"] == pytest.approx(80.0)
    assert table.loc["B", "valid_pct"] == pytest.approx(80.0)
    assert table.loc["A", "return_nan_pct"] == pytest.approx(20.0)
    assert table.loc["B", "return_nan_pct"] == pytest.approx(100.0)
    assert (table["realized_vol_nan_pct"] == 0.0).all()


def test_symbol_quality_keeps_panel_symbol_order_and_does_not_mutate() -> None:
    """Respeta el orden de columnas del panel y no toca el panel."""
    valid = np.array([[1, 0, 1]] * 4, dtype=bool)
    panel = _valid_panel(valid, ["Z", "A", "M"])
    before = panel.valid.copy()
    assert list(symbol_quality(panel).index) == ["Z", "A", "M"]
    pd.testing.assert_frame_equal(panel.valid, before)


# --- feature_quantiles ------------------------------------------------------------------------


def test_feature_quantiles_ignores_nan_and_uses_all_symbols() -> None:
    """Los NaN no cuentan como 0 y se mezclan todos los simbolos: 0..10 da cuantiles exactos."""
    nan = np.nan
    ret = np.array(
        [[0.0, 6.0], [1.0, 7.0], [2.0, nan], [3.0, 8.0], [4.0, nan], [5.0, 9.0], [10.0, nan]]
    )
    panel = _panel({"return": ret}, np.ones(ret.shape, dtype=bool), ["A", "B"])
    table = feature_quantiles(panel, quantiles=(0.0, 0.5, 1.0))
    values = [0, 6, 1, 7, 2, 3, 8, 4, 5, 9, 10]  # los 11 valores no NaN
    assert list(table.columns) == ["q0", "q0.5", "q1"]
    assert table.loc["return", "q0"] == 0.0 and table.loc["return", "q1"] == 10.0
    assert table.loc["return", "q0.5"] == pytest.approx(np.median(values)) == 5.0


def test_feature_quantiles_quantiles_are_a_parameter() -> None:
    """Los cuantiles pedidos son las columnas, en el orden dado; por defecto son siete."""
    ret = np.arange(40, dtype="float64").reshape(20, 2)
    panel = _panel({"return": ret}, np.ones(ret.shape, dtype=bool), ["A", "B"])
    table = feature_quantiles(panel, quantiles=(0.25, 0.75))
    assert list(table.columns) == ["q0.25", "q0.75"]
    assert table.loc["return", "q0.25"] == pytest.approx(np.quantile(ret, 0.25))
    assert table.loc["return", "q0.75"] == pytest.approx(np.quantile(ret, 0.75))
    default = feature_quantiles(panel)
    assert list(default.columns) == ["q0.001", "q0.01", "q0.25", "q0.5", "q0.75", "q0.99", "q0.999"]


def test_feature_quantiles_all_nan_feature_gives_nan_not_zero() -> None:
    """Una feature sin ningun valor da NaN en todos sus cuantiles, nunca 0 ni error."""
    values = {"return": np.arange(8.0).reshape(4, 2), "rel_volume": np.full((4, 2), np.nan)}
    panel = _panel(values, np.ones((4, 2), dtype=bool), ["A", "B"])
    table = feature_quantiles(panel, quantiles=(0.1, 0.9))
    assert table.loc["rel_volume"].isna().all()
    assert table.loc["return"].notna().all()


def test_feature_quantiles_keeps_one_row_per_feature_in_panel_order() -> None:
    """Una fila por feature, en el orden del panel."""
    values = {name: np.arange(6.0).reshape(3, 2) for name in _FEATURES}
    panel = _panel(values, np.ones((3, 2), dtype=bool), ["A", "B"])
    assert list(feature_quantiles(panel).index) == list(_FEATURES)


@pytest.mark.parametrize("bad", [(), (-0.1, 0.5), (0.5, 1.5)])
def test_feature_quantiles_rejects_invalid_quantiles(bad: tuple[float, ...]) -> None:
    """Cuantiles vacios o fuera de [0, 1] son `ValueError`."""
    panel = _valid_panel(np.ones((3, 2), dtype=bool), ["A", "B"])
    with pytest.raises(ValueError, match="quantiles"):
        feature_quantiles(panel, quantiles=bad)


# --- period_gap_causes ------------------------------------------------------------------------


def _save_slots(
    base_dir: Path, symbol: str, slots: Sequence[int], trades: dict[int, int] | None = None
) -> None:
    """Guarda velas de `symbol` solo en los minutos `slots`, con operaciones salvo en `trades`."""
    counts = np.array([(trades or {}).get(k, 5) for k in slots])
    close = np.full(len(slots), 100.0)
    save_klines(_klines_frame(slots, close, counts), symbol, "1m", base_dir=base_dir)


def test_period_gap_causes_by_hand(tmp_path: Path) -> None:
    """Cuenta por simbolo y causa: huecos, caida, sin operaciones y NOT_LISTED con `listed_ms`."""
    n = 20
    _save_slots(tmp_path, "A", range(n))
    _save_slots(tmp_path, "B", [k for k in range(n) if not 5 <= k < 10])
    _save_slots(tmp_path, "C", [k for k in range(n) if k not in (5, 6)])
    _save_slots(tmp_path, "D", range(3, n), trades={12: 0})
    table = period_gap_causes(
        ["A", "B", "C", "D"],
        "1m",
        T0,
        T0 + n * MIN_MS,
        listed_ms={"D": T0},
        base_dir=tmp_path,
    )
    assert list(table.columns) == GAP_NAMES
    assert list(table.index) == ["A", "B", "C", "D"] and table.index.name == "symbol"
    expected = {
        "A": {"VALID": 20},
        # En 5 y 6 faltan B y C de 4 vivos (50 %): caida. En 7..9 solo B: hueco.
        "B": {"VALID": 15, "OUTAGE": 2, "HALTED": 3},
        "C": {"VALID": 18, "OUTAGE": 2},
        # La primera vela de D es el slot 3 y su listado cae ese mismo dia: 0..2 son NOT_LISTED.
        "D": {"VALID": 16, "NOT_LISTED": 3, "NO_TRADES": 1},
    }
    for symbol, counts in expected.items():
        row = table.loc[symbol]
        assert {k: int(v) for k, v in row.items() if v} == counts, symbol
    assert (table.sum(axis=1) == n).all()


def test_period_gap_causes_without_listed_ms_is_no_data(tmp_path: Path) -> None:
    """Sin `listed_ms` los slots anteriores a la primera vela no son NOT_LISTED sino NO_DATA."""
    n = 20
    _save_slots(tmp_path, "A", range(n))
    _save_slots(tmp_path, "D", range(3, n))
    table = period_gap_causes(["A", "D"], "1m", T0, T0 + n * MIN_MS, base_dir=tmp_path)
    assert table.loc["D", "NO_DATA"] == 3 and table.loc["D", "NOT_LISTED"] == 0


def test_period_gap_causes_data_end_marks_delisted_only_until_the_download_end(
    tmp_path: Path,
) -> None:
    """Hasta `data_end_ms` es DELISTED; mas alla no hay descarga y es NO_DATA."""
    n = 20
    _save_slots(tmp_path, "A", range(n))
    _save_slots(tmp_path, "E", range(15))
    table = period_gap_causes(
        ["A", "E"],
        "1m",
        T0,
        T0 + n * MIN_MS,
        data_end_ms={"A": T0 + 19 * MIN_MS, "E": T0 + 16 * MIN_MS},
        base_dir=tmp_path,
    )
    row = table.loc["E"]
    assert (row["VALID"], row["DELISTED"], row["NO_DATA"]) == (15, 2, 3)


# --- correlation_by_liquidity -----------------------------------------------------------------


def _corr(values: list[list[float]], names: list[str]) -> pd.DataFrame:
    """Matriz de correlacion pequena con nombres."""
    return pd.DataFrame(values, index=names, columns=names)


NAN = np.nan


def test_correlation_by_liquidity_by_hand_and_dead_coin_is_not_illiquid() -> None:
    """Pares entre liquidas y con iliquidas; la moneda sin datos (diagonal NaN) no es iliquida."""
    corr = _corr(
        [
            [1.0, 0.8, 0.6, NAN],
            [0.8, 1.0, 0.2, NAN],
            [0.6, 0.2, 1.0, NAN],
            [NAN, NAN, NAN, NAN],
        ],
        ["A", "B", "C", "D"],
    )
    pct = pd.Series({"A": 100.0, "B": 99.0, "C": 90.0, "D": 0.0})
    result = correlation_by_liquidity(corr, pct)
    assert result["n_illiquid"] == 1  # C; D tiene 0 % pero no tiene datos
    assert result["pair_mean_liquid"] == pytest.approx(0.8)  # solo A-B
    assert result["pair_mean_with_illiquid"] == pytest.approx((0.6 + 0.2) / 2)


def test_correlation_by_liquidity_threshold_is_strict_and_a_parameter() -> None:
    """Es iliquida la que baja ESTRICTAMENTE de `illiquid_below`; el umbral es un parametro."""
    corr = _corr([[1.0, 0.5, 0.3], [0.5, 1.0, 0.1], [0.3, 0.1, 1.0]], ["A", "B", "C"])
    pct = pd.Series({"A": 97.0, "B": 96.99, "C": 100.0})
    assert correlation_by_liquidity(corr, pct)["n_illiquid"] == 1  # 97.0 no, 96.99 si
    assert correlation_by_liquidity(corr, pct, illiquid_below=50.0)["n_illiquid"] == 0
    assert correlation_by_liquidity(corr, pct, illiquid_below=100.0)["n_illiquid"] == 2
    assert correlation_by_liquidity(corr, pct, illiquid_below=100.1)["n_illiquid"] == 3


def test_correlation_by_liquidity_means_without_pairs_are_nan_never_zero() -> None:
    """Sin pares de un tipo la media es NaN, no 0: ni sin iliquidas, ni sin liquidas."""
    corr = _corr([[1.0, 0.4], [0.4, 1.0]], ["A", "B"])
    none_illiquid = correlation_by_liquidity(corr, pd.Series({"A": 100.0, "B": 100.0}))
    assert none_illiquid["pair_mean_liquid"] == pytest.approx(0.4)
    assert np.isnan(none_illiquid["pair_mean_with_illiquid"])
    all_illiquid = correlation_by_liquidity(corr, pd.Series({"A": 10.0, "B": 20.0}))
    assert np.isnan(all_illiquid["pair_mean_liquid"])
    assert all_illiquid["pair_mean_with_illiquid"] == pytest.approx(0.4)
    assert all_illiquid["n_illiquid"] == 2


def test_correlation_by_liquidity_skips_missing_pairs_between_coins_with_data() -> None:
    """Un par NaN entre dos monedas con datos no entra en ninguna media (ni como 0)."""
    corr = _corr(
        [[1.0, 0.8, NAN], [0.8, 1.0, 0.4], [NAN, 0.4, 1.0]],
        ["A", "B", "C"],
    )
    pct = pd.Series({"A": 100.0, "B": 100.0, "C": 50.0})
    result = correlation_by_liquidity(corr, pct)
    assert result["pair_mean_liquid"] == pytest.approx(0.8)
    assert result["pair_mean_with_illiquid"] == pytest.approx(0.4)  # A-C (NaN) no cuenta


def test_correlation_by_liquidity_all_dead_gives_nan_means() -> None:
    """Todas sin datos: nada iliquido y las dos medias NaN."""
    corr = _corr([[NAN, NAN], [NAN, NAN]], ["A", "B"])
    result = correlation_by_liquidity(corr, pd.Series({"A": 0.0, "B": 0.0}))
    assert result["n_illiquid"] == 0
    assert np.isnan(result["pair_mean_liquid"]) and np.isnan(result["pair_mean_with_illiquid"])


def test_correlation_by_liquidity_aligns_valid_pct_by_symbol_and_checks_coverage() -> None:
    """`valid_pct` se alinea por nombre (no por posicion) y debe tener todas las monedas."""
    corr = _corr([[1.0, 0.5, 0.3], [0.5, 1.0, 0.1], [0.3, 0.1, 1.0]], ["A", "B", "C"])
    shuffled = pd.Series({"C": 100.0, "A": 100.0, "B": 10.0, "EXTRA": 0.0})
    result = correlation_by_liquidity(corr, shuffled)
    assert result["n_illiquid"] == 1  # B
    assert result["pair_mean_liquid"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="valid_pct"):
        correlation_by_liquidity(corr, pd.Series({"A": 100.0, "B": 100.0}))


# --- bar_validity_pct -------------------------------------------------------------------------


@pytest.mark.parametrize("start_min", [7, 60, 1])
def test_bar_validity_matches_independent_oracle_at_5_15_60(start_min: int) -> None:
    """Barras a 5/15/60 min contra un oraculo con bucles: ultimo minuto valido y 80 %."""
    rng = np.random.default_rng(start_min)
    probs = np.array([0.99, 0.92, 0.7])
    n = 8 * 60 + 13
    valid = rng.random((n, 3)) < probs
    panel = _valid_panel(valid, ["HI", "MID", "LOW"], start_min=start_min)
    result = bar_validity_pct(panel)
    assert list(result.columns) == ["1m", "5m", "15m", "60m"]
    assert list(result.index) == ["HI", "MID", "LOW"] and result.index.name == "symbol"
    for horizon, minutes in (("5m", 5), ("15m", 15), ("60m", 60)):
        expected = _oracle_bar_pct(panel.valid, minutes)
        for symbol in panel.symbols:
            assert result.loc[symbol, horizon] == pytest.approx(expected[symbol]), (horizon, symbol)
    # Que el oraculo no sea trivial: hay barras validas e invalidas.
    assert 0 < result.loc["MID", "5m"] < 100 and 0 < result.loc["LOW", "5m"] < 100
    np.testing.assert_allclose(result["1m"], panel.valid.mean().to_numpy() * 100)


def test_bar_validity_oracle_with_strict_share() -> None:
    """Con `min_valid_share=1.0` una sola vela invalida invalida la barra, contra el oraculo."""
    rng = np.random.default_rng(5)
    panel = _valid_panel(rng.random((300, 2)) < 0.97, ["A", "B"], start_min=60)
    result = bar_validity_pct(panel, ("5m", "15m"), min_valid_share=1.0)
    for horizon, minutes in (("5m", 5), ("15m", 15)):
        expected = _oracle_bar_pct(panel.valid, minutes, share=1.0)
        for symbol in panel.symbols:
            assert result.loc[symbol, horizon] == pytest.approx(expected[symbol])


def test_bar_validity_shifts_valid_from_close_to_open_d19() -> None:
    """`valid` va al cierre (D-19): la fila `tau` es la vela abierta en `tau - 1 min`."""
    # Cierres 1..10 = aperturas 0..9: dos barras de 5m completas (opens 0-4 y 5-9).
    valid = np.ones((10, 3), dtype=bool)
    valid[4, 0] = (
        False  # A: cierre 5 = apertura 4 = ultimo minuto de la barra 1 -> barra 1 invalida
    )
    valid[5, 1] = False  # B: cierre 6 = apertura 5 = primer minuto de la barra 2 -> 4/5, valida
    valid[[1, 2], 2] = False  # C: cierres 2 y 3 -> 3/5 de la barra 1 -> invalida
    panel = _valid_panel(valid, ["A", "B", "C"], start_min=1)
    result = bar_validity_pct(panel, ("5m", "10m"))
    assert result["5m"].to_dict() == {"A": 50.0, "B": 100.0, "C": 50.0}
    # 10m: una barra con 9/10, 9/10 y 8/10 (borde: 0.8 cuenta) y el ultimo minuto valido.
    assert result["10m"].to_dict() == {"A": 100.0, "B": 100.0, "C": 100.0}
    assert result["1m"].to_dict() == pytest.approx({"A": 90.0, "B": 90.0, "C": 80.0})
    strict = bar_validity_pct(panel, ("10m",), min_valid_share=1.0)
    assert strict["10m"].to_dict() == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_bar_validity_trims_to_complete_bars_aligned_to_the_clock() -> None:
    """La primera fila (vela de la barra anterior) y la ultima barra incompleta no cuentan."""
    # Cierres 0..13 = aperturas -1..12: la fila 0 es de la barra anterior; 10, 11 y 12, de una
    # barra de 5m incompleta. Las barras completas son las aperturas 0-4 y 5-9.
    valid = np.ones((14, 2), dtype=bool)
    valid[0, :] = False  # apertura -1
    valid[11:, 0] = False  # aperturas 10..12, solo en la moneda A
    valid[11:, 1] = True
    panel = _valid_panel(valid, ["A", "B"], start_min=0)
    result = bar_validity_pct(panel, ("5m",))
    assert result["5m"].to_dict() == {"A": 100.0, "B": 100.0}
    # Y al reves: lo invalido DENTRO de una barra completa si cuenta.
    inside = valid.copy()
    inside[5, 1] = False  # apertura 4 = ultimo minuto de la barra 1
    assert bar_validity_pct(_valid_panel(inside, ["A", "B"], start_min=0), ("5m",))["5m"][
        "B"
    ] == pytest.approx(50.0)
    # `1m` es el % del panel entero, sin recortar.
    assert result["1m"]["A"] == pytest.approx(100 * 10 / 14)


@pytest.mark.parametrize(
    ("start_min", "rows", "horizon"),
    [
        (3, 10, "15m"),  # aperturas 2..11: 10 velas, ninguna barra de 15m
        (1, 4, "5m"),  # aperturas 0..3: empieza una barra alineada pero no se completa
        (6, 4, "5m"),  # aperturas 5..8: idem, la barra 5..9 queda a medias
        (2, 4, "5m"),  # aperturas 1..4: ninguna apertura alineada al reloj
    ],
)
def test_bar_validity_panel_shorter_than_a_bar_raises(
    start_min: int, rows: int, horizon: str
) -> None:
    """Un panel sin ninguna barra completa de `horizon` es `ValueError` (no un 0 ni un NaN)."""
    panel = _valid_panel(np.ones((rows, 1), dtype=bool), ["A"], start_min=start_min)
    with pytest.raises(ValueError, match="panel mas corto que una barra"):
        bar_validity_pct(panel, (horizon,))


@pytest.mark.parametrize(("start_min", "rows"), [(3, 8), (6, 5)])
def test_bar_validity_panel_with_exactly_one_complete_bar_does_not_raise(
    start_min: int, rows: int
) -> None:
    """Control positivo: con una sola barra completa (aperturas 5..9) no falla."""
    panel = _valid_panel(np.ones((rows, 1), dtype=bool), ["A"], start_min=start_min)
    assert bar_validity_pct(panel, ("5m",)).loc["A", "5m"] == 100.0


def test_bar_validity_error_message_names_the_horizon() -> None:
    """El mensaje de error dice el horizonte que no cabe."""
    panel = _valid_panel(np.ones((10, 1), dtype=bool), ["A"], start_min=3)
    with pytest.raises(ValueError, match="panel mas corto que una barra de 15m"):
        bar_validity_pct(panel, ("5m", "15m"))


def test_bar_validity_rejects_non_multiple_horizon() -> None:
    """Un horizonte que no es multiplo del intervalo es `ValueError`."""
    panel = _valid_panel(np.ones((20, 1), dtype=bool), ["A"], start_min=1)
    with pytest.raises(ValueError):
        bar_validity_pct(panel, ("30s",))


@pytest.mark.parametrize("head_valid", [True, False])
def test_bar_validity_no_leak_later_minutes_do_not_change_an_earlier_bar(head_valid: bool) -> None:
    """Cambiar la validez de minutos POSTERIORES a una barra no cambia esa barra."""
    # Barra 1 = aperturas 0..4 (cierres 1..5): valida si todo True, invalida si falla el ultimo.
    head = np.ones(5, dtype=bool)
    head[-1] = head_valid
    rng = np.random.default_rng(7)
    for _ in range(25):
        tail = rng.random((15, 4)) < rng.choice([0.5, 0.85, 1.0], size=4)
        valid = np.vstack([np.tile(head[:, None], (1, 4)), tail])
        panel = _valid_panel(valid, list("ABCD"), start_min=1)
        result = bar_validity_pct(panel, ("5m",))["5m"]
        # La cola son 3 barras de 5m; el oraculo da cuantas son validas y la barra 1 aporta 0 o 1.
        tail_panel = _valid_panel(tail, list("ABCD"), start_min=6)
        tail_bars = _oracle_bar_pct(tail_panel.valid, 5)
        for symbol in "ABCD":
            valid_bars = result[symbol] * 4 / 100 - tail_bars[symbol] * 3 / 100
            assert valid_bars == pytest.approx(1.0 if head_valid else 0.0), symbol


def test_bar_validity_is_deterministic_and_does_not_mutate_the_panel() -> None:
    """Misma entrada, mismo resultado, y el panel queda intacto."""
    rng = np.random.default_rng(3)
    panel = _valid_panel(rng.random((200, 3)) < 0.9, ["A", "B", "C"], start_min=11)
    before = panel.valid.copy()
    first = bar_validity_pct(panel)
    pd.testing.assert_frame_equal(first, bar_validity_pct(panel))
    pd.testing.assert_frame_equal(panel.valid, before)


# --- build_panel_overview: de punta a punta ---------------------------------------------------


def test_overview_returns_every_table_with_one_row_per_period(overview: PanelOverview) -> None:
    """Todas las tablas del NamedTuple, una fila por trimestre donde toca, contrato cumplido."""
    assert overview._fields == (
        "periods",
        "quality",
        "feature_quantiles",
        "tails",
        "bar_validity",
        "gap_causes",
        "correlation",
    )
    assert all(isinstance(table, pd.DataFrame) and len(table) for table in overview)
    assert list(overview.periods.index) == list(PERIODS)
    assert overview.periods.index.name == "period_start"
    assert (overview.periods["n_rows"] == DAY).all()
    assert (overview.periods["n_symbols"] == 4).all() and (
        overview.periods["n_features"] == 4
    ).all()
    assert (
        overview.periods["contract_ok"].all() and (overview.periods["contract_error"] == "").all()
    )
    assert list(overview.correlation.index) == list(PERIODS)
    for table, level, per_period in (
        (overview.quality, "symbol", 4),
        (overview.tails, "symbol", 4),
        (overview.bar_validity, "symbol", 4),
        (overview.gap_causes, "symbol", 4),
        (overview.feature_quantiles, "feature", 4),
    ):
        assert list(table.index.names) == ["period_start", level]
        assert len(table) == 3 * per_period
        assert sorted(set(table.index.get_level_values(0))) == list(PERIODS)
    assert list(overview.quality.loc[PERIODS[0]].index) == list(SYMBOLS)


def test_overview_quality_by_hand(overview: PanelOverview) -> None:
    """% de minutos validos de cada simbolo y trimestre, contados a mano sobre la cache."""
    pct = overview.quality["valid_pct"]
    expected = {
        # La primera fila del trimestre 1 es el minuto -1 (NOT_LISTED): el cierre en el minuto 0.
        (0, "BTCUSDT"): 1429,  # 1440 - 1 (-1) - 8 (hueco) - 2 (sin operaciones)
        (0, "ETHUSDT"): 1435,  # 1440 - 1 - 4
        (0, "SOLUSDT"): 1439,
        (0, "XUSDT"): 1439,
        (1, "BTCUSDT"): 1440,
        (1, "ETHUSDT"): 1440,
        (1, "SOLUSDT"): 1440,
        (1, "XUSDT"): 1,  # solo el ultimo minuto del dia 0
        (2, "XUSDT"): 1439,  # el primer slot (ultimo minuto del dia 1) no tiene vela
        (2, "SOLUSDT"): 1440,
    }
    for (p, symbol), count in expected.items():
        assert pct[(PERIODS[p], symbol)] == pytest.approx(100 * count / DAY), (p, symbol)
    # Las features pierden NaN donde no hay validez: el % de NaN del retorno es >= 100 - valid_pct.
    nan_pct = overview.quality["return_nan_pct"]
    assert (nan_pct >= 100 - pct - 1e-9).all()
    # X en el trimestre 2: solo su primera fila tiene retorno (la vela 1439 y la 1438 son validas).
    assert nan_pct[(PERIODS[1], "XUSDT")] == pytest.approx(100 * 1439 / DAY)


def test_overview_feature_quantiles_match_pooled_returns_and_nan_features(
    overview: PanelOverview, world: tuple[Path, np.ndarray]
) -> None:
    """Cuantiles del retorno = cuantiles de los retornos del trimestre 2; rel_volume sin ventana."""
    increments = world[1]
    # Trimestre 2: filas = cierres de las velas 1439..2878. BTC, ETH y SOL tienen los 1440 retornos
    # y X solo el de la primera fila (la vela 1439 y la anterior son validas).
    pooled = np.append(increments[1439:2879, :3].ravel(), increments[1439, 3])
    table = overview.feature_quantiles.loc[PERIODS[1]]
    assert list(table.index) == list(_FEATURES)
    assert list(table.columns) == ["q0.001", "q0.01", "q0.25", "q0.5", "q0.75", "q0.99", "q0.999"]
    for column, q in zip(table.columns, (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999), strict=True):
        assert table.loc["return", column] == pytest.approx(np.quantile(pooled, q), abs=1e-9)
    # La ventana de actividad (10080 min) no cabe en 3 dias: todo NaN, y eso es NaN, no 0.
    for period in PERIODS:
        assert overview.feature_quantiles.loc[(period, "rel_volume")].isna().all()
        assert overview.feature_quantiles.loc[(period, "rel_trades")].isna().all()
        assert overview.feature_quantiles.loc[(period, "realized_vol")].notna().all()


def test_overview_tails_have_one_row_per_symbol_and_period(
    overview: PanelOverview, world: tuple[Path, np.ndarray]
) -> None:
    """La tabla de colas cuenta los retornos reales: SOL todos y X solo uno en el trimestre 2."""
    tails = overview.tails
    assert tails.loc[(PERIODS[1], "XUSDT"), "n"] == 1
    assert tails.loc[(PERIODS[1], "SOLUSDT"), "n"] == DAY
    increments = world[1]
    sol = increments[1439:2879, 2]
    assert tails.loc[(PERIODS[1], "SOLUSDT"), "std"] == pytest.approx(sol.std(ddof=1), rel=1e-9)


def test_overview_bar_validity_by_hand(overview: PanelOverview) -> None:
    """Barras validas a 5/15/60 min contadas a mano (hueco de BTC/ETH, sin operaciones de BTC)."""
    bars = overview.bar_validity
    assert list(bars.columns) == ["1m", "5m", "15m", "60m"]
    p1 = bars.loc[PERIODS[0]]
    # Barras completas del trimestre 1 (aperturas 0..1438): 287 de 5m, 95 de 15m y 23 de 60m.
    # BTC, 5m: invalidas 100-104 (ultimo invalido), 105-109 (2/5) y 200-204 (3/5).
    assert p1.loc["BTCUSDT", "5m"] == pytest.approx(100 * 284 / 287)
    assert p1.loc["ETHUSDT", "5m"] == pytest.approx(100 * 286 / 287)  # solo 100-104 (1/5)
    # 15m: la barra 90-104 falla en los dos; la 105-119 de BTC tiene 12/15 = 0.8, que cuenta.
    assert p1.loc["BTCUSDT", "15m"] == pytest.approx(100 * 94 / 95)
    assert p1.loc["ETHUSDT", "15m"] == pytest.approx(100 * 94 / 95)
    assert p1.loc["BTCUSDT", "60m"] == 100.0 and p1.loc["ETHUSDT", "60m"] == 100.0
    # La primera fila (minuto -1, NOT_LISTED) no pertenece a ninguna barra completa.
    assert (p1.loc["SOLUSDT", ["5m", "15m", "60m"]] == 100.0).all()
    assert p1.loc["SOLUSDT", "1m"] == pytest.approx(100 * 1439 / DAY)
    # Trimestre 2: X solo tiene valida la primera fila (fuera de las barras completas).
    p2 = bars.loc[PERIODS[1]]
    assert (p2.loc["XUSDT", ["5m", "15m", "60m"]] == 0.0).all()
    assert (p2.loc["BTCUSDT"] == 100.0).all()
    # Trimestre 3: la fila sin vela de X queda fuera de las barras; el resto es completo.
    assert (bars.loc[(PERIODS[2], "XUSDT"), ["5m", "15m", "60m"]] == 100.0).all()


def test_overview_gap_causes_match_the_valid_slots(overview: PanelOverview) -> None:
    """Las causas se piden en `[start - paso, ...)`: mismos slots que `valid` (VALID = validos)."""
    causes = overview.gap_causes
    assert list(causes.columns) == GAP_NAMES
    assert (causes.sum(axis=1) == DAY).all()
    valid_slots = (overview.quality["valid_pct"] * DAY / 100).round().astype(int)
    assert (causes["VALID"].to_numpy() == valid_slots.to_numpy()).all()
    p1 = causes.loc[PERIODS[0]]
    # El primer slot del trimestre 1 es el minuto -1, antes de la primera vela y del listado.
    assert (p1["NOT_LISTED"] == 1).all()
    assert p1.loc["BTCUSDT", ["VALID", "HALTED", "OUTAGE", "NO_TRADES"]].tolist() == [1429, 4, 4, 2]
    # En 100..103 faltan BTC y ETH de 4 vivas (50 %): caida; en 104..107 solo BTC: hueco.
    assert p1.loc["ETHUSDT", ["VALID", "HALTED", "OUTAGE"]].tolist() == [1435, 0, 4]
    assert p1.loc["SOLUSDT", "VALID"] == 1439 and p1.loc["XUSDT", "VALID"] == 1439
    # Trimestre 2: X sin velas en medio de su vida (dias 0 y 2): HALTED salvo el slot del dia 0.
    p2 = causes.loc[PERIODS[1]]
    assert p2.loc["XUSDT", ["VALID", "HALTED"]].tolist() == [1, 1439]
    assert (p2.loc[["BTCUSDT", "ETHUSDT", "SOLUSDT"], "VALID"] == DAY).all()
    # Trimestre 3: el primer slot (ultimo minuto del dia 1) es de X sin vela.
    assert causes.loc[(PERIODS[2], "XUSDT"), ["VALID", "HALTED"]].tolist() == [1439, 1]


def test_overview_correlation_sets_the_dead_coin_aside(
    overview: PanelOverview, world: tuple[Path, np.ndarray]
) -> None:
    """El trimestre 2 aparta X y su resumen coincide con np.corrcoef de las otras tres."""
    row = overview.correlation.loc[PERIODS[1]]
    assert row["error"] == ""
    assert row["n_symbols_in_universe"] == 4 and row["n_symbols_without_data"] == 1
    assert row["symbols_without_data"] == "XUSDT" and row["n_symbols"] == 3
    live = world[1][1439:2879, :3]
    manual = np.corrcoef(live, rowvar=False)
    upper = manual[np.triu_indices(3, k=1)]
    assert row["pair_mean"] == pytest.approx(upper.mean(), abs=1e-9)
    assert row["first_eigenvalue"] == pytest.approx(np.linalg.eigvalsh(manual)[-1], abs=1e-9)
    # Todas liquidas (100 %): ninguna iliquida y la media con iliquidas es NaN, no 0.
    assert row["n_illiquid"] == 0
    assert row["pair_mean_liquid"] == pytest.approx(upper.mean(), abs=1e-9)
    assert np.isnan(row["pair_mean_with_illiquid"])
    # Trimestre 1 y 3: nadie apartado; BTC es iliquida en el 1 (99,24 % < 99,5 %).
    for period in (PERIODS[0], PERIODS[2]):
        other = overview.correlation.loc[period]
        assert other["error"] == "" and other["n_symbols_without_data"] == 0
        assert other["symbols_without_data"] == "" and other["n_symbols"] == 4
    assert overview.correlation.loc[PERIODS[0], "n_illiquid"] == 1
    assert overview.correlation.loc[PERIODS[2], "n_illiquid"] == 0
    assert np.isfinite(overview.correlation.loc[PERIODS[0], "pair_mean_with_illiquid"])
    assert (overview.correlation["error"] == "").all()


def test_overview_without_gap_causes_leaves_the_table_empty_with_all_columns(
    world: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """`include_gap_causes=False` deja `gap_causes` vacia con las columnas de `GapCause`."""
    result = _overview(world, tmp_path / "panels", include_gap_causes=False)
    assert result.gap_causes.empty and list(result.gap_causes.columns) == GAP_NAMES
    assert len(result.periods) == 3 and len(result.quality) == 12


class _LoadSpy:
    """Cuenta las llamadas a `cached_panel`, `build_panel` y `load_panel` de la cache."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: dict[str, list[tuple[object, ...]]] = {
            "cached_panel": [],
            "build_panel": [],
            "load_panel": [],
        }
        for name in self.calls:
            original = getattr(panel_store, name)
            monkeypatch.setattr(panel_store, name, self._wrap(name, original))

    def _wrap(self, name: str, original: Callable[..., object]) -> Callable[..., object]:
        """Envuelve `original` registrando sus argumentos posicionales."""

        def spy(*args: object, **kwargs: object) -> object:
            self.calls[name].append(args)
            return original(*args, **kwargs)

        return spy


def test_overview_loads_each_panel_exactly_once(
    world: tuple[Path, np.ndarray], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cada panel se pide una vez: se construye la primera vez y se carga la segunda, nunca dos."""
    spy = _LoadSpy(monkeypatch)
    first = _overview(world, tmp_path / "panels")
    assert len(spy.calls["cached_panel"]) == 3
    starts = [call[2] for call in spy.calls["cached_panel"]]
    assert len(set(starts)) == 3  # tres periodos distintos, uno por llamada
    assert len(spy.calls["build_panel"]) == 3 and len(spy.calls["load_panel"]) == 0

    second = _overview(world, tmp_path / "panels")
    assert len(spy.calls["cached_panel"]) == 6
    assert len(spy.calls["build_panel"]) == 3 and len(spy.calls["load_panel"]) == 3
    # Cargado de la cache o construido, el resumen es el mismo.
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_overview_is_deterministic(
    world: tuple[Path, np.ndarray], tmp_path: Path, overview: PanelOverview
) -> None:
    """Otra cache de paneles, mismo resultado."""
    again = _overview(world, tmp_path / "other")
    for a, b in zip(overview, again, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_overview_correlation_failure_leaves_an_error_row_and_a_warning(
    world: tuple[Path, np.ndarray],
    tmp_path: Path,
    module_warnings: pytest.LogCaptureFixture,
) -> None:
    """Si el resumen de un trimestre falla (factor X sin datos en el 2): fila `error` y aviso."""
    result = _overview(world, tmp_path / "panels", factor="XUSDT")
    bad = result.correlation.loc[PERIODS[1]]
    assert "XUSDT no tiene datos en el periodo 2024-01-02" in bad["error"]
    assert bad["n_symbols_in_universe"] == 4
    assert pd.isna(bad["pair_mean"]) and pd.isna(bad["n_illiquid"])
    # No aborta: los otros trimestres tienen su resumen y todas las demas tablas estan completas.
    for period in (PERIODS[0], PERIODS[2]):
        good = result.correlation.loc[period]
        assert good["error"] == "" and np.isfinite(good["pair_mean"])
        assert good["factor"] == "XUSDT"
    assert len(result.periods) == 3 and len(result.quality) == 12 and len(result.gap_causes) == 12
    messages = _warnings(module_warnings, "miax.viz.panel_overview")
    assert len(messages) == 1
    assert "Periodo 2024-01-02" in messages[0] and "XUSDT no tiene datos" in messages[0]


def test_overview_correlation_failure_is_isolated_with_a_patched_summary(
    world: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_warnings: pytest.LogCaptureFixture,
) -> None:
    """Un `ValueError` cualquiera en el resumen de un periodo solo marca ese periodo."""
    import miax.viz.panel_overview as module

    original = module.summarize_correlation

    def flaky(corr: pd.DataFrame, **kwargs: object) -> object:
        if kwargs["label"] == "2024-01-03":
            raise ValueError("fallo de prueba")
        return original(corr, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "summarize_correlation", flaky)
    result = _overview(world, tmp_path / "panels")
    assert result.correlation["error"].tolist() == ["", "", "fallo de prueba"]
    assert result.correlation["n_symbols_in_universe"].tolist() == [4, 4, 4]
    assert pd.isna(result.correlation.loc[PERIODS[2], "pair_mean"])
    messages = _warnings(module_warnings, "miax.viz.panel_overview")
    assert len(messages) == 1 and "fallo de prueba" in messages[0] and "2024-01-03" in messages[0]


def test_overview_rejects_interval_without_fixed_step(
    world: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """Un intervalo sin paso fijo (`1M`) es `ValueError` antes de tocar la cache."""
    with pytest.raises(ValueError, match="paso fijo"):
        build_panel_overview(_world_universe(), "1M", base_dir=world[0], cache_dir=tmp_path)


# --- summarize_correlation --------------------------------------------------------------------


def _returns(names: Sequence[str], dead: Sequence[str] = (), seed: int = 41) -> pd.DataFrame:
    """Retornos de un factor comun con las monedas `dead` toda NaN."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((500, 1))
    data = 0.6 * factor + 0.8 * rng.standard_normal((500, len(names)))
    returns = pd.DataFrame(data, columns=list(names))
    returns[list(dead)] = np.nan
    return returns


def test_summarize_correlation_sets_dead_coins_aside_and_matches_the_public_functions() -> None:
    """Aparta las monedas sin datos y da lo mismo que `correlation_summary` sobre las demas."""
    corr = raw_correlation(_returns(["BTCUSDT", "A", "DEAD", "B"], dead=["DEAD"]), min_periods=10)
    summary, eigenvalues, without = summarize_correlation(corr, label="2024-01-01")
    assert without == ["DEAD"]
    kept = corr.drop(index="DEAD", columns="DEAD")
    assert summary == correlation_summary(kept)
    assert summary["n_symbols"] == 3
    np.testing.assert_array_equal(eigenvalues, explained_variance(kept))
    assert eigenvalues.shape == (3,) and eigenvalues.sum() == pytest.approx(1.0)


def test_summarize_correlation_without_dead_coins_returns_an_empty_list_and_does_not_warn(
    module_warnings: pytest.LogCaptureFixture,
) -> None:
    """Sin monedas apartadas la lista es vacia y no hay aviso."""
    corr = raw_correlation(_returns(["BTCUSDT", "A", "B"]), min_periods=10)
    _, _, without = summarize_correlation(corr, label="2024-01-01")
    assert without == []
    assert _warnings(module_warnings, "miax.viz.raw_correlation") == []


def test_summarize_correlation_warns_once_with_label_count_and_names(
    module_warnings: pytest.LogCaptureFixture,
) -> None:
    """Con monedas apartadas, un solo aviso con el periodo, cuantas y cuales."""
    corr = raw_correlation(
        _returns(["BTCUSDT", "A", "D1", "D2", "B"], dead=["D1", "D2"]), min_periods=10
    )
    summarize_correlation(corr, label="2024-04-01")
    messages = _warnings(module_warnings, "miax.viz.raw_correlation")
    assert messages == [
        "Periodo 2024-04-01: 2 simbolos sin datos, fuera del resumen y de los componentes: D1, D2"
    ]


def test_summarize_correlation_factor_without_data_raises_before_warning(
    module_warnings: pytest.LogCaptureFixture,
) -> None:
    """Si el factor no tiene datos: `ValueError` con el periodo y sin aviso."""
    corr = raw_correlation(
        _returns(["BTCUSDT", "A", "B", "D"], dead=["BTCUSDT", "D"]), min_periods=10
    )
    with pytest.raises(ValueError, match="BTCUSDT no tiene datos en el periodo 2024-01-01$"):
        summarize_correlation(corr, label="2024-01-01")
    assert _warnings(module_warnings, "miax.viz.raw_correlation") == []


def test_summarize_correlation_custom_factor_without_data_raises() -> None:
    """El factor es un parametro: uno distinto de BTC sin datos tambien falla."""
    corr = raw_correlation(_returns(["BTCUSDT", "A", "B"], dead=["A"]), min_periods=10)
    with pytest.raises(ValueError, match="A no tiene datos"):
        summarize_correlation(corr, factor="A", label="x")
    assert summarize_correlation(corr, factor="BTCUSDT", label="x")[2] == ["A"]


def test_summarize_correlation_fewer_than_two_with_data_raises_before_warning(
    module_warnings: pytest.LogCaptureFixture,
) -> None:
    """Con menos de 2 monedas con datos: `ValueError` con las apartadas y sin aviso."""
    corr = raw_correlation(_returns(["BTCUSDT", "D1", "D2"], dead=["D1", "D2"]), min_periods=10)
    with pytest.raises(
        ValueError,
        match=r"menos de 2 simbolos con datos en el periodo P; sin datos: \['D1', 'D2'\]",
    ):
        summarize_correlation(corr, label="P")
    assert _warnings(module_warnings, "miax.viz.raw_correlation") == []


def test_summarize_correlation_parameters_are_passed_through() -> None:
    """`high` y `max_nan_share` llegan a `correlation_summary`; un par ausente cuenta."""
    names = ["BTCUSDT", "A", "B", "C"]
    corr = _corr(
        [
            [1.0, 0.5, 0.6, 0.7],
            [0.5, 1.0, NAN, 0.4],
            [0.6, NAN, 1.0, 0.3],
            [0.7, 0.4, 0.3, 1.0],
        ],
        names,
    )
    with pytest.raises(ValueError, match="faltan"):
        summarize_correlation(corr, label="x")  # 1 de 6 pares = 16,7 % > 5 %
    summary, _, without = summarize_correlation(corr, max_nan_share=0.2, high=0.55, label="x")
    assert without == [] and summary["n_missing_pairs"] == 1
    assert summary["high_threshold"] == 0.55
    assert summary["pairs_above_high_pct"] == pytest.approx(2 / 5 * 100)  # 0.6 y 0.7 de 5 pares


def test_summarize_correlation_does_not_modify_the_matrix() -> None:
    """La matriz de entrada queda intacta (la mapa y la tabla por simbolo la usan completa)."""
    corr = raw_correlation(_returns(["BTCUSDT", "A", "D", "B"], dead=["D"]), min_periods=10)
    before = corr.copy()
    summarize_correlation(corr, label="x")
    pd.testing.assert_frame_equal(corr, before)


# --- build_raw_correlation_report tras el refactor --------------------------------------------


def _report_world(base_dir: Path, *, dead: bool = False, skip: Sequence[str] = ()) -> pd.DataFrame:
    """Un dia con BTC, ETH y SOL (menos `skip`) y, si `dead`, NEWUSDT sin velas; da el universo."""
    rng = np.random.default_rng(99)
    factor = rng.standard_normal(DAY)
    names = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for name in names:
        step = 0.001 * (0.7 * factor + 0.7 * rng.standard_normal(DAY))
        if name in skip:
            continue
        close = 100.0 * np.exp(np.cumsum(step))
        save_klines(
            _klines_frame(range(DAY), close, np.full(DAY, 5)), name, "1m", base_dir=base_dir
        )
    symbols = [*names, *(["NEWUSDT"] if dead else [])]
    return _universe([("2024-01-01", "2024-01-02", s) for s in symbols])


def _report(tmp_path: Path, universe: pd.DataFrame) -> ReportPaths:
    """Informe del 2024-01-01 con todo dentro de `tmp_path`."""
    return build_raw_correlation_report(
        "2024-01-01",
        universe=universe,
        min_periods=100,
        base_dir=tmp_path / "klines",
        cache_dir=tmp_path / "panels",
        out_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
    )


def test_report_messages_after_the_refactor_carry_the_period_label(
    tmp_path: Path, module_warnings: pytest.LogCaptureFixture
) -> None:
    """El informe sigue avisando con el texto de siempre y con la fecha del periodo."""
    _report(tmp_path, _report_world(tmp_path / "klines", dead=True))
    assert _warnings(module_warnings, "miax.viz.raw_correlation") == [
        "Periodo 2024-01-01: 1 simbolos sin datos, fuera del resumen y de los componentes: NEWUSDT"
    ]


def test_report_factor_without_data_error_carries_the_period_label(tmp_path: Path) -> None:
    """Sin velas de BTC el `ValueError` del informe dice el factor y la fecha del periodo."""
    universe = _report_world(tmp_path / "klines", skip=["BTCUSDT"])
    with pytest.raises(ValueError, match="BTCUSDT no tiene datos en el periodo 2024-01-01$"):
        _report(tmp_path, universe)


def test_report_fewer_than_two_coins_error_carries_the_period_label(tmp_path: Path) -> None:
    """Con solo BTC con velas el `ValueError` dice el periodo y las monedas sin datos."""
    universe = _report_world(tmp_path / "klines", skip=["ETHUSDT", "SOLUSDT"])
    with pytest.raises(
        ValueError,
        match=r"menos de 2 simbolos con datos en el periodo 2024-01-01; "
        r"sin datos: \['ETHUSDT', 'SOLUSDT'\]",
    ):
        _report(tmp_path, universe)


def test_report_summary_equals_summarize_correlation_on_the_same_matrix(tmp_path: Path) -> None:
    """Las cifras del CSV del informe son las de `summarize_correlation` sobre la misma matriz."""
    universe = _report_world(tmp_path / "klines", dead=True)
    paths = _report(tmp_path, universe)
    table = pd.read_csv(paths.summary_csv, dtype=str, keep_default_na=False, index_col="metric")
    summary = table["value"]
    order = list(pd.read_csv(paths.eda_csv, index_col="symbol").index)
    panel = panel_store.cached_panel(
        order,
        "1m",
        T0,
        T0 + DAY * MIN_MS,
        base_dir=tmp_path / "klines",
        cache_dir=tmp_path / "panels",
    )
    corr = raw_correlation(panel.features["return"][order], min_periods=100)
    direct, _, without = summarize_correlation(corr, label="2024-01-01")
    assert without == ["NEWUSDT"] and summary["symbols_without_data"] == "NEWUSDT"
    for key, value in direct.items():
        if isinstance(value, str):
            assert summary[key] == value, key
        else:
            assert float(summary[key]) == pytest.approx(value, abs=1e-7), key  # CSV a 8 decimales
