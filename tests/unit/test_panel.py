"""Tests del panel maestro: identidad con cada `*_from_cache`, invariantes, no-fuga y periodos."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import miax.features.panel as panel_module
from miax.features import (
    FEATURE_NAMES,
    Panel,
    build_panel,
    build_time_grid,
    log_relative_to_past,
    period_panels,
    realized_volatility,
    realized_volatility_from_cache,
    relative_trades_from_cache,
    relative_volume_from_cache,
    returns_from_cache,
)
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
VOL_W = 5
ACT_W = 20
SMALL = {"vol_window": VOL_W, "activity_window": ACT_W}


def _ms(minute: int) -> int:
    """Instante en ms de `minute` minutos despues de T0."""
    return T0 + minute * MIN_MS


def _at(minute: int) -> pd.Timestamp:
    """Marca de tiempo UTC de `minute` minutos despues de T0."""
    return pd.Timestamp(_ms(minute), unit="ms", tz="UTC")


def _data(seed: int, n: int) -> dict[str, np.ndarray]:
    """Cierre, volumen y operaciones sinteticos, positivos y con semilla fija."""
    rng = np.random.default_rng(seed)
    return {
        "close": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n))),
        "quote": np.exp(rng.normal(5.0, 1.0, n)),
        "trades": rng.integers(1, 100, n),
    }


def _cache(
    base_dir: Path, symbol: str, data: dict[str, np.ndarray], minutes: Iterable[int]
) -> None:
    """Guarda en `base_dir` las velas de `symbol` solo en los minutos dados."""
    rows = list(minutes)
    open_times = [_ms(k) for k in rows]
    n = len(rows)
    frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": data["close"][rows],
            "high": data["close"][rows],
            "low": data["close"][rows],
            "close": data["close"][rows],
            "volume": [1.0] * n,
            "close_time": [t + MIN_MS - 1 for t in open_times],
            "quote_asset_volume": data["quote"][rows],
            "num_trades": data["trades"][rows],
            "taker_buy_base": [0.5] * n,
            "taker_buy_quote": [5.0] * n,
            "ignore": [0.0] * n,
        },
        columns=KLINE_COLUMNS,
    ).astype(KLINE_DTYPES)
    save_klines(frame, symbol, "1m", base_dir=base_dir)


def _world(base_dir: Path, n: int = 240) -> None:
    """Tres simbolos: A con hueco, B con minutos sin operaciones, C con un tramo de precio plano."""
    a, b, c = _data(1, n), _data(2, n), _data(3, n)
    _cache(base_dir, "A", a, [k for k in range(n) if not 100 <= k < 108])
    b["trades"] = b["trades"].copy()
    b["trades"][[90, 91, 150]] = 0
    _cache(base_dir, "B", b, range(n))
    c["close"] = c["close"].copy()
    c["close"][120:131] = c["close"][120]
    _cache(base_dir, "C", c, range(n))


def _panel(base_dir: Path, start_min: int = 60, end_min: int = 200, **kwargs: object) -> Panel:
    """Panel de A, B y C en `[start_min, end_min)` con ventanas pequenas."""
    params = {**SMALL, **kwargs}
    return build_panel(
        ["A", "B", "C"], "1m", _ms(start_min), _ms(end_min), base_dir=base_dir, **params
    )


def _expected(base_dir: Path, start_min: int, end_min: int) -> dict[str, pd.DataFrame]:
    """Las cuatro features calculadas con sus funciones `*_from_cache` por separado."""
    args = (["A", "B", "C"], "1m", _ms(start_min), _ms(end_min))
    return {
        "return": returns_from_cache(*args, base_dir=base_dir),
        "realized_vol": realized_volatility_from_cache(*args, window=VOL_W, base_dir=base_dir),
        "rel_volume": relative_volume_from_cache(*args, window=ACT_W, base_dir=base_dir),
        "rel_trades": relative_trades_from_cache(*args, window=ACT_W, base_dir=base_dir),
    }


# --- identidad con cada *_from_cache ----------------------------------------------------------


@pytest.mark.parametrize(("start_min", "end_min"), [(60, 200), (0, 100), (5, 30), (230, 240)])
def test_each_feature_is_exactly_its_from_cache(
    tmp_path: Path, start_min: int, end_min: int
) -> None:
    """Cada feature del panel es la salida exacta de su `*_from_cache` (atol=0), con huecos."""
    _world(tmp_path)
    panel = _panel(tmp_path, start_min, end_min)
    for name, frame in _expected(tmp_path, start_min, end_min).items():
        assert_frame_equal(panel.features[name], frame, check_exact=True, atol=0)


def test_identity_with_default_windows(tmp_path: Path) -> None:
    """Con los parametros por defecto tambien coincide con las funciones por defecto."""
    _world(tmp_path)
    args = (["A", "B", "C"], "1m", _ms(100), _ms(140))
    panel = build_panel(*args, base_dir=tmp_path)
    assert_frame_equal(
        panel.features["realized_vol"],
        realized_volatility_from_cache(*args, base_dir=tmp_path),
        check_exact=True,
        atol=0,
    )
    assert_frame_equal(
        panel.features["rel_volume"],
        relative_volume_from_cache(*args, base_dir=tmp_path),
        check_exact=True,
        atol=0,
    )


# --- invariantes -------------------------------------------------------------------------------


def test_invariants_of_the_panel(tmp_path: Path) -> None:
    """Mismo indice (la rejilla), columnas en orden pedido, float64 y `valid` bool."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    grid = build_time_grid(_ms(60), _ms(200), "1m")
    assert tuple(panel.features) == FEATURE_NAMES
    assert panel.symbols == ("A", "B", "C")
    assert panel.interval == "1m"
    assert panel.index.equals(grid)
    assert panel.shape == (140, 3, 4)
    for frame in panel.features.values():
        assert frame.index.equals(grid)
        assert list(frame.columns) == ["A", "B", "C"]
        assert all(dtype == "float64" for dtype in frame.dtypes)
    assert panel.valid.index.equals(grid)
    assert all(dtype == bool for dtype in panel.valid.dtypes)


def test_valid_is_the_validity_of_the_candle_closing_at_tau(tmp_path: Path) -> None:
    """La fila `tau` de `valid` es la de la vela abierta en `tau - 1 min`."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    # A no tiene velas en 100..107 (cierran en 101..108); B tiene NO_TRADES en 90 y 91 (cierran
    # en 91 y 92); C esta siempre valido.
    assert not panel.valid.loc[_at(101) : _at(108), "A"].any()
    assert panel.valid.loc[_at(100), "A"]
    assert panel.valid.loc[_at(109), "A"]
    assert not panel.valid.loc[_at(91), "B"]
    assert not panel.valid.loc[_at(92), "B"]
    assert panel.valid.loc[_at(93), "B"]
    assert panel.valid["C"].all()


def test_invalid_candle_has_nan_features_never_zero(tmp_path: Path) -> None:
    """En un slot invalido la actividad es `NaN` (D-02): no hay 0 ni relleno."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    invalid = ~panel.valid.to_numpy()
    assert invalid.any()
    for name in ("rel_volume", "rel_trades"):
        assert np.isnan(panel.features[name].to_numpy()[invalid]).all()
    assert np.isnan(panel.features["return"].to_numpy()[invalid]).all()


def test_panel_is_immutable(tmp_path: Path) -> None:
    """No se puede reasignar ningun campo del panel."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.interval = "5m"  # type: ignore[misc]


def test_duplicate_symbols_are_collapsed_in_order(tmp_path: Path) -> None:
    """Un simbolo repetido no duplica columnas: se conserva el orden de primera aparicion."""
    _world(tmp_path)
    panel = build_panel(["B", "A", "B"], "1m", _ms(60), _ms(80), base_dir=tmp_path, **SMALL)
    assert panel.symbols == ("B", "A")
    assert list(panel.valid.columns) == ["B", "A"]


# --- to_array ----------------------------------------------------------------------------------


def test_to_array_has_axes_time_asset_feature(tmp_path: Path) -> None:
    """`array[t, n, f]` coincide con `features[f].iloc[t, n]`, en float64 y con forma (T, N, F)."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    array = panel.to_array()
    assert array.shape == (140, 3, 4)
    assert array.dtype == np.float64
    for f, name in enumerate(FEATURE_NAMES):
        np.testing.assert_array_equal(array[:, :, f], panel.features[name].to_numpy())
    # Un valor concreto: el eje 2 es la feature (realized_vol = 1) y el 1 el simbolo (C = 2).
    assert array[50, 2, 1] == panel.features["realized_vol"].iloc[50, 2]


def test_to_array_keeps_nan_and_adds_no_zero(tmp_path: Path) -> None:
    """Los `NaN` y los ceros reales pasan intactos: no se crea ningun cero nuevo."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    array = panel.to_array()
    stacked = [frame.to_numpy() for frame in panel.features.values()]
    assert int(np.isnan(array).sum()) == sum(int(np.isnan(m).sum()) for m in stacked)
    assert int((array == 0).sum()) == sum(int((m == 0).sum()) for m in stacked)
    # C tiene el precio plano en 120..130: retornos 0 reales que se conservan.
    assert (panel.features["return"]["C"] == 0).any()
    assert np.isnan(array).any()


def test_to_array_respects_order(tmp_path: Path) -> None:
    """`order` reordena y elige las features; un nombre desconocido lanza `ValueError`."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    array = panel.to_array(("rel_trades", "return"))
    assert array.shape == (140, 3, 2)
    np.testing.assert_array_equal(array[:, :, 0], panel.features["rel_trades"].to_numpy())
    np.testing.assert_array_equal(array[:, :, 1], panel.features["return"].to_numpy())
    with pytest.raises(ValueError, match="desconocidas"):
        panel.to_array(("nope",))


# --- defaults sincronizados --------------------------------------------------------------------


def test_defaults_match_the_real_functions() -> None:
    """Los defaults de `build_panel` son los de `realized_volatility` y `log_relative_to_past`."""
    own = inspect.signature(build_panel).parameters
    vol = inspect.signature(realized_volatility).parameters
    act = inspect.signature(log_relative_to_past).parameters
    assert own["vol_window"].default == vol["window"].default
    assert own["vol_min_valid_share"].default == vol["min_valid_share"].default
    assert own["activity_window"].default == act["window"].default
    assert own["activity_min_valid_share"].default == act["min_valid_share"].default
    period = inspect.signature(period_panels).parameters
    for name in (
        "vol_window",
        "vol_min_valid_share",
        "activity_window",
        "activity_min_valid_share",
    ):
        assert period[name].default == own[name].default


# --- sin fuga ----------------------------------------------------------------------------------


def _world_with_future(base_dir: Path, from_min: int, n: int = 240) -> None:
    """Como `_world`, pero con el precio, volumen y operaciones muy distintos desde `from_min`."""
    for symbol, seed in (("A", 1), ("B", 2), ("C", 3)):
        data = _data(seed, n)
        data["close"][from_min:] *= 7.0
        data["quote"][from_min:] *= 1000.0
        data["trades"][from_min:] = data["trades"][from_min:] + 500
        _cache(base_dir, symbol, data, range(n))


def test_no_leak_changing_candles_after_end_does_not_change_the_panel(tmp_path: Path) -> None:
    """Las velas desde `end_ms` en adelante no alteran ningun valor del panel."""
    _world_with_future(tmp_path / "x", from_min=10_000)
    _world_with_future(tmp_path / "y", from_min=200)
    first = _panel(tmp_path / "x")
    second = _panel(tmp_path / "y")
    for name in FEATURE_NAMES:
        assert_frame_equal(first.features[name], second.features[name], check_exact=True, atol=0)
    assert_frame_equal(first.valid, second.valid)


def test_no_leak_future_candles_do_not_change_earlier_rows(tmp_path: Path) -> None:
    """Un panel con un `end` posterior da las mismas filas anteriores que uno con `end` menor."""
    _world(tmp_path)
    short = _panel(tmp_path, 60, 150)
    long = _panel(tmp_path, 60, 220)
    for name in FEATURE_NAMES:
        assert_frame_equal(
            long.features[name].iloc[:90], short.features[name], check_exact=True, atol=0
        )
    assert_frame_equal(long.valid.iloc[:90], short.valid)


def test_no_leak_the_first_row_only_uses_the_past(tmp_path: Path) -> None:
    """Cambiar el precio de la vela que abre en `start` no cambia la primera fila."""
    _world_with_future(tmp_path / "x", from_min=10_000)
    _world_with_future(tmp_path / "y", from_min=60)
    first = _panel(tmp_path / "x", 60, 61)
    second = _panel(tmp_path / "y", 60, 61)
    for name in FEATURE_NAMES:
        assert_frame_equal(first.features[name], second.features[name], check_exact=True, atol=0)


# --- hueco largo (LUNAUSDT no se parte) --------------------------------------------------------


def test_long_gap_leaves_nan_until_coverage_recovers(tmp_path: Path) -> None:
    """Un hueco mas largo que la ventana deja `NaN` durante y despues, sin partir la serie."""
    n = 240
    _cache(tmp_path, "A", _data(1, n), [k for k in range(n) if not 80 <= k < 120])
    panel = build_panel(["A"], "1m", _ms(60), _ms(200), base_dir=tmp_path, **SMALL)
    feats = panel.features
    # Velas 80..119 ausentes: cierran en 81..120.
    assert not panel.valid.loc[_at(81) : _at(120), "A"].any()
    assert feats["return"].loc[_at(81) : _at(121), "A"].isna().all()
    assert feats["rel_volume"].loc[_at(81) : _at(130), "A"].isna().all()
    assert feats["rel_trades"].loc[_at(81) : _at(130), "A"].isna().all()
    assert feats["realized_vol"].loc[_at(82) : _at(124), "A"].isna().all()
    # La cobertura vuelve: 10 velas validas de 20 (50 %) y 4 retornos validos de 5 (80 %).
    assert not np.isnan(feats["rel_volume"].loc[_at(131), "A"])
    assert not np.isnan(feats["realized_vol"].loc[_at(125), "A"])
    # Antes del hueco hay valores y la serie sigue siendo una sola.
    assert not np.isnan(feats["rel_volume"].loc[_at(79), "A"])


# --- simbolo sin cache -------------------------------------------------------------------------


def test_symbol_without_cache_is_a_nan_column(tmp_path: Path) -> None:
    """Un simbolo sin cache da una columna `NaN` en las features y `False` en `valid`."""
    _world(tmp_path)
    panel = build_panel(["A", "ZZZ"], "1m", _ms(60), _ms(100), base_dir=tmp_path, **SMALL)
    assert panel.symbols == ("A", "ZZZ")
    for frame in panel.features.values():
        assert frame["ZZZ"].isna().all()
    assert not panel.valid["ZZZ"].any()
    assert panel.valid["A"].any()


# --- period_panels -----------------------------------------------------------------------------


def _universe() -> pd.DataFrame:
    """Universo sintetico de dos periodos de un dia con simbolos distintos."""
    rows = [
        ("2024-01-01", "2024-01-02", "A"),
        ("2024-01-01", "2024-01-02", "B"),
        ("2024-01-02", "2024-01-03", "B"),
        ("2024-01-02", "2024-01-03", "C"),
    ]
    frame = pd.DataFrame(rows, columns=["period_start", "period_end", "symbol"])
    for column in ("period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[us, UTC]")
    return frame


def test_period_panels_yields_one_panel_per_period_in_order(tmp_path: Path) -> None:
    """Un panel por periodo, en orden, con sus simbolos y su rango semiabierto exacto."""
    for symbol, seed in (("A", 1), ("B", 2), ("C", 3)):
        _cache(tmp_path, symbol, _data(seed, 3000), range(3000))
    out = list(period_panels(_universe(), "1m", base_dir=tmp_path, **SMALL))
    assert [start for start, _ in out] == [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-02", tz="UTC"),
    ]
    first, second = out[0][1], out[1][1]
    assert first.symbols == ("A", "B")
    assert second.symbols == ("B", "C")
    assert first.index.equals(build_time_grid(_ms(0), _ms(1440), "1m"))
    assert second.index.equals(build_time_grid(_ms(1440), _ms(2880), "1m"))
    assert first.index[-1] == pd.Timestamp("2024-01-01 23:59", tz="UTC")
    direct = build_panel(["B", "C"], "1m", _ms(1440), _ms(2880), base_dir=tmp_path, **SMALL)
    for name in FEATURE_NAMES:
        assert_frame_equal(second.features[name], direct.features[name], check_exact=True, atol=0)


def test_period_panels_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consumir un panel no calcula el siguiente."""
    calls: list[tuple[list[str], int, int]] = []

    def fake(symbols: list[str], interval: str, start_ms: int, end_ms: int, **_: object) -> str:
        calls.append((list(symbols), start_ms, end_ms))
        return f"panel{len(calls)}"

    monkeypatch.setattr(panel_module, "build_panel", fake)
    generator = period_panels(_universe(), "1m")
    assert calls == []
    first = next(generator)
    assert first[1] == "panel1"
    assert calls == [(["A", "B"], _ms(0), _ms(1440))]
    next(generator)
    assert calls[1] == (["B", "C"], _ms(1440), _ms(2880))
    with pytest.raises(StopIteration):
        next(generator)


def test_period_panels_with_empty_universe_yields_nothing() -> None:
    """Un universo sin filas no rinde ningun panel."""
    empty = _universe().iloc[0:0]
    assert list(period_panels(empty, "1m")) == []


# --- errores -----------------------------------------------------------------------------------


def test_empty_symbols_raise() -> None:
    """Sin simbolos, `ValueError`."""
    with pytest.raises(ValueError, match="symbols"):
        build_panel([], "1m", _ms(0), _ms(10))


def test_empty_range_raises() -> None:
    """Un rango sin ningun slot lanza `ValueError`."""
    with pytest.raises(ValueError, match="vacio"):
        build_panel(["A"], "1m", _ms(10), _ms(10))


def test_interval_without_fixed_step_raises() -> None:
    """Un intervalo sin paso fijo lanza `ValueError`."""
    with pytest.raises(ValueError, match="duracion fija"):
        build_panel(["A"], "1M", _ms(0), _ms(10))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vol_window": 0},
        {"vol_min_valid_share": 0.0},
        {"activity_window": 0},
        {"activity_min_valid_share": 1.5},
    ],
)
def test_bad_window_params_raise_before_reading(kwargs: dict[str, float]) -> None:
    """Parametros de ventana invalidos lanzan `ValueError` antes de tocar la cache."""
    with pytest.raises(ValueError):
        build_panel(["A"], "1m", _ms(0), _ms(10), **kwargs)


def _frame(index: pd.DatetimeIndex, columns: list[str], value: object = 1.0) -> pd.DataFrame:
    """DataFrame constante con ese indice y columnas."""
    return pd.DataFrame(value, index=index, columns=columns)


def test_panel_rejects_mismatched_frames() -> None:
    """`Panel` lanza `ValueError` si las formas, el dtype o `valid` no cuadran."""
    index = build_time_grid(_ms(0), _ms(3), "1m")
    good = _frame(index, ["A", "B"])
    valid = _frame(index, ["A", "B"], True).astype(bool)
    Panel(features={"x": good}, valid=valid, interval="1m", symbols=("A", "B"))
    with pytest.raises(ValueError, match="columnas"):
        Panel(features={"x": good[["A"]]}, valid=valid, interval="1m", symbols=("A", "B"))
    with pytest.raises(ValueError, match="indice"):
        Panel(features={"x": good.iloc[:2]}, valid=valid, interval="1m", symbols=("A", "B"))
    with pytest.raises(ValueError, match="float64"):
        Panel(
            features={"x": good.astype("float32")}, valid=valid, interval="1m", symbols=("A", "B")
        )
    with pytest.raises(ValueError, match="bool"):
        Panel(features={"x": good}, valid=good, interval="1m", symbols=("A", "B"))
    with pytest.raises(ValueError, match="vacio"):
        Panel(features={}, valid=valid, interval="1m", symbols=("A", "B"))
    with pytest.raises(ValueError, match="columnas"):
        Panel(features={"x": good}, valid=valid, interval="1m", symbols=("B", "A"))


def test_final_index_mismatch_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el indice de salida no es la rejilla, `RuntimeError` (error del codigo, no del dato)."""
    _world(tmp_path)
    monkeypatch.setattr(panel_module, "shift_to_close", lambda index, interval: index)
    with pytest.raises(RuntimeError, match="rejilla"):
        _panel(tmp_path)


# --- refuerzos tras la auditoria (MIAX-041) ----------------------------------------------------


def _good_parts() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """Devuelve un indice UTC de 3 minutos, una feature float64 y un `valid` bool (A y B)."""
    index = build_time_grid(_ms(0), _ms(3), "1m")
    return index, _frame(index, ["A", "B"]), _frame(index, ["A", "B"], True).astype(bool)


def test_panel_structure_is_frozen(tmp_path: Path) -> None:
    """`features` no admite asignacion, `symbols` es tupla y reasignar campos falla."""
    _world(tmp_path)
    panel = _panel(tmp_path)
    with pytest.raises(TypeError):
        panel.features["x"] = panel.valid.astype("float64")  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.symbols = ("A",)  # type: ignore[misc]
    _, good, valid = _good_parts()
    symbols: Any = ["A", "B"]
    from_list = Panel(features={"x": good}, valid=valid, interval="1m", symbols=symbols)
    assert from_list.symbols == ("A", "B")
    assert isinstance(from_list.symbols, tuple)
    assert from_list.to_array().shape == (3, 2, 1)


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("naive", "UTC"),
        ("madrid", "UTC"),
        ("range", "DatetimeIndex"),
        ("reversed", "creciente"),
        ("duplicated", "duplicados"),
    ],
)
def test_panel_rejects_bad_index(kind: str, match: str) -> None:
    """Indice no datetime, sin UTC, desordenado o con duplicados lanza `ValueError`."""
    index, good, valid = _good_parts()
    bad: pd.Index = {
        "naive": index.tz_localize(None),
        "madrid": index.tz_convert("Europe/Madrid"),
        "range": pd.RangeIndex(3),
        "reversed": index[::-1],
        "duplicated": index[[0, 0, 1]],
    }[kind]
    with pytest.raises(ValueError, match=match):
        Panel(
            features={"x": good.set_axis(bad, axis=0)},
            valid=valid.set_axis(bad, axis=0),
            interval="1m",
            symbols=("A", "B"),
        )


def test_panel_rejects_interval_without_fixed_step() -> None:
    """Un `interval` sin paso fijo lanza `ValueError`."""
    _, good, valid = _good_parts()
    with pytest.raises(ValueError, match="paso fijo"):
        Panel(features={"x": good}, valid=valid, interval="1M", symbols=("A", "B"))


def test_panel_rejects_duplicate_symbols() -> None:
    """Nombres de `symbols` repetidos lanzan `ValueError`."""
    index = build_time_grid(_ms(0), _ms(3), "1m")
    good = _frame(index, ["A", "A"])
    valid = _frame(index, ["A", "A"], True).astype(bool)
    with pytest.raises(ValueError, match="duplicados"):
        Panel(features={"x": good}, valid=valid, interval="1m", symbols=("A", "A"))


def test_panel_rejects_valid_with_other_columns() -> None:
    """Un `valid` con columnas distintas de `symbols` lanza `ValueError`."""
    _, good, valid = _good_parts()
    with pytest.raises(ValueError, match="valid: las columnas"):
        Panel(features={"x": good}, valid=valid[["B", "A"]], interval="1m", symbols=("A", "B"))


def test_panel_accepts_fixed_step_interval_other_than_1m() -> None:
    """`Panel` acepta un intervalo de paso fijo distinto de 1m (lo usara MIAX-102)."""
    index = build_time_grid(_ms(0), _ms(15), "5m")
    good = _frame(index, ["A"])
    valid = _frame(index, ["A"], True).astype(bool)
    panel = Panel(features={"x": good}, valid=valid, interval="5m", symbols=("A",))
    assert panel.shape == (3, 1, 1)


def test_build_panel_only_accepts_1m() -> None:
    """`build_panel` con 5m lanza `ValueError` que remite a MIAX-102."""
    with pytest.raises(ValueError, match="MIAX-102"):
        build_panel(["A"], "5m", _ms(0), _ms(30))


@pytest.mark.parametrize(
    ("order", "match"),
    [(("return", "return"), "repetidas"), ((), "vacio"), (("nope", "return"), "desconocidas")],
)
def test_to_array_rejects_bad_order(tmp_path: Path, order: tuple[str, ...], match: str) -> None:
    """Un `order` repetido, vacio o con features desconocidas lanza `ValueError` propio."""
    _world(tmp_path)
    with pytest.raises(ValueError, match=match):
        _panel(tmp_path).to_array(order)


def test_period_panels_rejects_several_period_ends() -> None:
    """Un mismo `period_start` con `period_end` distintos lanza `ValueError`."""
    universe = _universe()
    universe.loc[1, "period_end"] = pd.Timestamp("2024-01-05", tz="UTC")
    with pytest.raises(ValueError, match="period_end"):
        list(period_panels(universe, "1m"))


def test_period_panels_rejects_missing_columns() -> None:
    """Si faltan columnas del universo, `ValueError` que las nombra (no `KeyError`)."""
    with pytest.raises(ValueError, match="period_end"):
        list(period_panels(_universe().drop(columns=["period_end"]), "1m"))


def test_period_panels_rejects_naive_timestamps() -> None:
    """Un `period_start` sin zona horaria lanza `ValueError` claro."""
    universe = _universe()
    universe["period_start"] = universe["period_start"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        list(period_panels(universe, "1m"))


def test_identity_with_a_symbol_without_cache(tmp_path: Path) -> None:
    """Con un simbolo sin cache en la lista, cada feature sigue siendo su `*_from_cache`."""
    _world(tmp_path)
    args = (["A", "ZZZ", "C"], "1m", _ms(60), _ms(200))
    panel = build_panel(*args, base_dir=tmp_path, **SMALL)
    expected = {
        "return": returns_from_cache(*args, base_dir=tmp_path),
        "realized_vol": realized_volatility_from_cache(*args, window=VOL_W, base_dir=tmp_path),
        "rel_volume": relative_volume_from_cache(*args, window=ACT_W, base_dir=tmp_path),
        "rel_trades": relative_trades_from_cache(*args, window=ACT_W, base_dir=tmp_path),
    }
    for name, frame in expected.items():
        assert_frame_equal(panel.features[name], frame, check_exact=True, atol=0)


def test_identity_with_a_single_row_range(tmp_path: Path) -> None:
    """Un rango de una sola fila coincide con los `*_from_cache`."""
    _world(tmp_path)
    panel = _panel(tmp_path, 150, 151)
    assert panel.shape[0] == 1
    for name, frame in _expected(tmp_path, 150, 151).items():
        assert_frame_equal(panel.features[name], frame, check_exact=True, atol=0)


def test_identity_of_rel_trades_with_default_windows(tmp_path: Path) -> None:
    """`rel_trades` con los parametros por defecto coincide con `relative_trades_from_cache`."""
    _world(tmp_path)
    args = (["A", "B", "C"], "1m", _ms(100), _ms(140))
    panel = build_panel(*args, base_dir=tmp_path)
    assert_frame_equal(
        panel.features["rel_trades"],
        relative_trades_from_cache(*args, base_dir=tmp_path),
        check_exact=True,
        atol=0,
    )


def test_all_invalid_candles_give_all_nan(tmp_path: Path) -> None:
    """Si todas las velas tienen `num_trades=0`, `valid` es todo `False` y las features `NaN`."""
    data = _data(1, 240)
    data["trades"] = np.zeros(240, dtype=int)
    _cache(tmp_path, "A", data, range(240))
    panel = build_panel(["A"], "1m", _ms(60), _ms(200), base_dir=tmp_path, **SMALL)
    assert not panel.valid.to_numpy().any()
    assert np.isnan(panel.to_array()).all()


def test_identity_when_end_is_not_a_multiple_of_the_step(tmp_path: Path) -> None:
    """Un `end_ms` que no es multiplo del paso da lo mismo que los `*_from_cache`."""
    _world(tmp_path)
    args = (["A", "B", "C"], "1m", _ms(60), _ms(200) + 30_000)
    panel = build_panel(*args, base_dir=tmp_path, **SMALL)
    assert_frame_equal(
        panel.features["return"],
        returns_from_cache(*args, base_dir=tmp_path),
        check_exact=True,
        atol=0,
    )
    assert_frame_equal(
        panel.features["rel_volume"],
        relative_volume_from_cache(*args, window=ACT_W, base_dir=tmp_path),
        check_exact=True,
        atol=0,
    )


def test_modifying_a_panel_does_not_affect_a_second_build(tmp_path: Path) -> None:
    """Modificar un DataFrame interno de un panel no afecta a otra llamada con la misma cache."""
    _world(tmp_path)
    first = _panel(tmp_path)
    reference = _panel(tmp_path)
    first.features["return"].iloc[:, :] = 123.0
    first.valid.iloc[:, :] = False
    second = _panel(tmp_path)
    for name in FEATURE_NAMES:
        assert_frame_equal(second.features[name], reference.features[name], check_exact=True)
    assert_frame_equal(second.valid, reference.valid)


def test_realized_vol_can_have_value_where_valid_is_false(tmp_path: Path) -> None:
    """Una ventana con cobertura suficiente da `realized_vol` aun con `valid=False`."""
    _cache(tmp_path, "A", _data(1, 240), [k for k in range(240) if k != 100])
    panel = build_panel(["A"], "1m", _ms(60), _ms(200), base_dir=tmp_path, **SMALL)
    tau = _at(101)  # la vela que falta abre en el minuto 100 y cierra en 101
    assert not panel.valid.loc[tau, "A"]
    assert np.isnan(panel.features["return"].loc[tau, "A"])
    assert not np.isnan(panel.features["realized_vol"].loc[tau, "A"])
