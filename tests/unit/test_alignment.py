"""Tests de la alineacion a rejilla comun: caches sinteticas en `tmp_path`, sin red."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from miax.features import align_frame, align_klines, build_time_grid
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
DAY_MS = 86_400_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z (jueves)
MONDAY = T0 + 4 * DAY_MS  # 2026-01-05T00:00:00Z, apertura de vela semanal


def _candles(open_times: list[int], close: list[float] | None = None) -> pd.DataFrame:
    """Klines de 1m con esos `open_time` exactos; `close` por defecto es 100 + posicion."""
    n = len(open_times)
    prices = [100.0 + i for i in range(n)] if close is None else close
    data = {
        "open_time": open_times,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1.0] * n,
        "close_time": [t + MIN_MS - 1 for t in open_times],
        "quote_asset_volume": [10.0] * n,
        "num_trades": [7] * n,
        "taker_buy_base": [0.5] * n,
        "taker_buy_quote": [5.0] * n,
        "ignore": [0.0] * n,
    }
    return pd.DataFrame(data, columns=KLINE_COLUMNS).astype(KLINE_DTYPES)


def _minutes(first: int, last: int) -> list[int]:
    """`open_time` de los minutos `first..last` (inclusive) contados desde T0."""
    return [T0 + k * MIN_MS for k in range(first, last + 1)]


def _cache(base_dir: Path, symbol: str, open_times: list[int], **kwargs: list[float]) -> None:
    """Guarda en `base_dir` una cache de 1m de `symbol` con esos `open_time`."""
    save_klines(_candles(open_times, **kwargs), symbol, "1m", base_dir=base_dir)


# --- Rejilla -----------------------------------------------------------------------------------


def test_grid_length_bounds_and_step() -> None:
    """La rejilla de 10 minutos tiene 10 slots, de T0 al minuto 9, a paso de 60 000 ms."""
    grid = build_time_grid(T0, T0 + 10 * MIN_MS, "1m")
    assert len(grid) == 10
    assert grid[0] == pd.Timestamp(T0, unit="ms", tz="UTC")
    assert grid[-1] == pd.Timestamp(T0 + 9 * MIN_MS, unit="ms", tz="UTC")
    assert (np.diff(grid.asi8) == MIN_MS).all()


def test_grid_is_utc_ms_and_has_no_freq() -> None:
    """El indice es UTC, `datetime64[ms, UTC]` y no infiere `freq`."""
    grid = build_time_grid(T0, T0 + 10 * MIN_MS, "1m")
    assert str(grid.dtype) == "datetime64[ms, UTC]"
    assert grid.tz is not None
    assert str(grid.tz) == "UTC"
    assert grid.freq is None


def test_grid_excludes_end() -> None:
    """El extremo final no entra: `[inicio, fin)`."""
    grid = build_time_grid(T0, T0 + 3 * MIN_MS, "1m")
    assert pd.Timestamp(T0 + 3 * MIN_MS, unit="ms", tz="UTC") not in grid
    assert len(grid) == 3


def test_unaligned_start_moves_to_next_slot() -> None:
    """Un inicio entre dos slots arranca en el siguiente, nunca en uno anterior."""
    grid = build_time_grid(T0 + 1, T0 + 5 * MIN_MS, "1m")
    assert grid[0] == pd.Timestamp(T0 + MIN_MS, unit="ms", tz="UTC")
    assert len(grid) == 4


def test_weekly_grid_uses_monday_offset() -> None:
    """Las velas semanales abren en lunes, no en jueves como la epoca."""
    grid = build_time_grid(T0, MONDAY + 2 * 7 * DAY_MS, "1w")
    assert list(grid.asi8) == [MONDAY, MONDAY + 7 * DAY_MS]
    assert grid[0].dayofweek == 0


def test_range_without_slot_gives_empty_grid() -> None:
    """Un rango no vacio que no contiene ningun slot da un indice vacio, sin error."""
    grid = build_time_grid(T0 + 1, T0 + MIN_MS - 1, "1m")
    assert len(grid) == 0
    assert str(grid.dtype) == "datetime64[ms, UTC]"


@pytest.mark.parametrize(
    ("start", "end", "interval"),
    [
        (T0, T0 + MIN_MS, "1M"),
        (T0, T0, "1m"),
        (T0 + MIN_MS, T0, "1m"),
    ],
)
def test_invalid_grid_raises(start: int, end: int, interval: str) -> None:
    """Un intervalo sin paso fijo, un rango vacio o uno invertido dan `ValueError`."""
    with pytest.raises(ValueError):
        build_time_grid(start, end, interval)


# --- Alineacion --------------------------------------------------------------------------------


def test_gap_stays_nan_not_zero_nor_previous_value(tmp_path: Path) -> None:
    """D-02: un hueco queda NaN, no 0 ni el valor anterior, y hay tantos NaN como velas faltan."""
    present = [t for t in _minutes(0, 9) if t not in (T0 + 4 * MIN_MS, T0 + 5 * MIN_MS)]
    _cache(tmp_path, "AAAUSDT", present, close=[10.0 + i for i in range(len(present))])

    wide = align_klines(["AAAUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)["close"]
    column = wide["AAAUSDT"]

    assert column.isna().sum() == 2
    assert column.isna().tolist() == [False] * 4 + [True, True] + [False] * 4
    assert column.iloc[3] == 13.0  # el valor anterior al hueco no se arrastra
    assert column.iloc[6] == 14.0
    assert (column.dropna() != 0).all()


def test_two_symbols_share_exact_index(tmp_path: Path) -> None:
    """Dos simbolos con inicios distintos comparten indice; el tardio tiene NaN antes de cotizar."""
    _cache(tmp_path, "EARLYUSDT", _minutes(0, 9))
    _cache(tmp_path, "LATEUSDT", _minutes(6, 9))

    wide = align_klines(["EARLYUSDT", "LATEUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)[
        "close"
    ]

    assert list(wide.columns) == ["EARLYUSDT", "LATEUSDT"]
    assert wide["LATEUSDT"].iloc[:6].isna().all()
    assert wide["LATEUSDT"].iloc[6:].notna().all()
    assert wide["EARLYUSDT"].notna().all()
    assert_index_equal(wide.index, build_time_grid(T0, T0 + 10 * MIN_MS, "1m"))


def test_index_is_full_grid_even_if_no_symbol_covers_the_edges(tmp_path: Path) -> None:
    """La rejilla sale del rango, no de las velas: sin velas en los extremos hay NaN en todos."""
    _cache(tmp_path, "AAAUSDT", _minutes(2, 7))
    _cache(tmp_path, "BBBUSDT", _minutes(3, 6))

    wide = align_klines(["AAAUSDT", "BBBUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)[
        "close"
    ]

    assert_index_equal(wide.index, build_time_grid(T0, T0 + 10 * MIN_MS, "1m"))
    assert len(wide) == 10
    assert wide.iloc[:2].isna().all().all()
    assert wide.iloc[8:].isna().all().all()
    assert wide["AAAUSDT"].iloc[2:8].notna().all()
    assert wide["BBBUSDT"].iloc[3:7].notna().all()


def test_weekly_interval_aligns_on_monday_offset(tmp_path: Path) -> None:
    """Con velas semanales (abren en lunes) cada una cae en su slot y la que falta es NaN."""
    weeks = [MONDAY, MONDAY + 2 * 7 * DAY_MS, MONDAY + 3 * 7 * DAY_MS]
    save_klines(_candles(weeks, close=[1.0, 3.0, 4.0]), "WKUSDT", "1w", base_dir=tmp_path)

    wide = align_klines(["WKUSDT"], "1w", T0, MONDAY + 4 * 7 * DAY_MS, base_dir=tmp_path)["close"]

    assert list(wide.index.asi8) == [MONDAY + k * 7 * DAY_MS for k in range(4)]
    assert wide["WKUSDT"].isna().tolist() == [False, True, False, False]
    assert wide["WKUSDT"].dropna().tolist() == [1.0, 3.0, 4.0]


def test_candle_at_end_does_not_leak(tmp_path: Path) -> None:
    """Una vela justo en `end_ms` no entra en el resultado."""
    _cache(tmp_path, "AAAUSDT", _minutes(0, 10))  # el minuto 10 es end_ms

    wide = align_klines(["AAAUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)["close"]

    assert len(wide) == 10
    assert wide["AAAUSDT"].notna().all()
    assert wide.index[-1] == pd.Timestamp(T0 + 9 * MIN_MS, unit="ms", tz="UTC")


def test_candles_outside_range_are_ignored(tmp_path: Path) -> None:
    """Las velas antes del inicio y despues del fin no alteran el resultado."""
    _cache(tmp_path, "AAAUSDT", _minutes(0, 30))

    wide = align_klines(["AAAUSDT"], "1m", T0 + 5 * MIN_MS, T0 + 10 * MIN_MS, base_dir=tmp_path)

    assert wide["close"]["AAAUSDT"].tolist() == [105.0, 106.0, 107.0, 108.0, 109.0]


def test_duplicate_open_time_fails_naming_symbol(tmp_path: Path) -> None:
    """Una `open_time` duplicada lanza `ValueError` con el simbolo y la pista de `validate`."""
    _cache(tmp_path, "DUPUSDT", [*_minutes(0, 4), T0 + 2 * MIN_MS])

    with pytest.raises(ValueError, match="DUPUSDT.*duplicado") as info:
        align_klines(["DUPUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)
    assert "validate" in str(info.value)


def test_misaligned_open_time_fails_naming_symbol(tmp_path: Path) -> None:
    """Una `open_time` fuera de la rejilla lanza `ValueError` con el simbolo."""
    _cache(tmp_path, "BADUSDT", [*_minutes(0, 4), T0 + 5 * MIN_MS + 1_000])

    with pytest.raises(ValueError, match="BADUSDT.*desalineado"):
        align_klines(["BADUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)


def test_empty_grid_fails_before_loading(tmp_path: Path) -> None:
    """Un rango sin ningun slot lanza `ValueError` de rejilla vacia, sin leer del disco."""
    with pytest.raises(ValueError, match="rejilla vacia"):
        align_klines(["AAAUSDT"], "1m", T0 + 1, T0 + MIN_MS - 1, base_dir=tmp_path)


def test_empty_symbols_raises(tmp_path: Path) -> None:
    """Sin simbolos no hay nada que alinear: `ValueError`."""
    with pytest.raises(ValueError, match="symbols vacio"):
        align_klines([], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)


def test_empty_fields_raises(tmp_path: Path) -> None:
    """Sin campos no hay nada que devolver: `ValueError`."""
    with pytest.raises(ValueError, match="fields vacio"):
        align_klines(["AAAUSDT"], "1m", T0, T0 + 10 * MIN_MS, fields=[], base_dir=tmp_path)


def test_symbol_without_cache_is_all_nan_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Un simbolo sin cache da una columna entera de NaN y un aviso, sin cambiar la forma."""
    _cache(tmp_path, "AAAUSDT", _minutes(0, 9))

    with caplog.at_level(logging.WARNING):
        wide = align_klines(["AAAUSDT", "NONEUSDT"], "1m", T0, T0 + 10 * MIN_MS, base_dir=tmp_path)[
            "close"
        ]

    assert wide.shape == (10, 2)
    assert wide["NONEUSDT"].isna().all()
    assert wide["AAAUSDT"].notna().all()
    assert any("NONEUSDT" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("field", ["open_time", "close_time", "ignore", "nope"])
def test_invalid_field_raises(tmp_path: Path, field: str) -> None:
    """Los tiempos, `ignore` y los campos inexistentes no se pueden alinear."""
    with pytest.raises(ValueError, match="Campos no alineables"):
        align_klines(["AAAUSDT"], "1m", T0, T0 + MIN_MS, fields=[field], base_dir=tmp_path)


def test_several_fields_share_index_and_trades_are_float(tmp_path: Path) -> None:
    """Varios campos devuelven un dict con el mismo indice; `num_trades` sale `float64`."""
    _cache(tmp_path, "AAAUSDT", _minutes(0, 4))

    result = align_klines(
        ["AAAUSDT"],
        "1m",
        T0,
        T0 + 10 * MIN_MS,
        fields=("close", "volume", "num_trades"),
        base_dir=tmp_path,
    )

    assert list(result) == ["close", "volume", "num_trades"]
    assert_index_equal(result["close"].index, result["volume"].index)
    assert_index_equal(result["close"].index, result["num_trades"].index)
    assert (result["num_trades"].dtypes == "float64").all()
    assert result["num_trades"]["AAAUSDT"].iloc[:5].tolist() == [7.0] * 5
    assert result["num_trades"]["AAAUSDT"].iloc[5:].isna().all()


def test_align_frame_is_pure_and_float64() -> None:
    """`align_frame` reindexa sin I/O y devuelve `float64` con el indice de la rejilla."""
    grid = build_time_grid(T0, T0 + 5 * MIN_MS, "1m")
    aligned = align_frame(_candles([T0, T0 + 3 * MIN_MS]), grid, ["close"])

    assert_index_equal(aligned.index, grid)
    assert aligned["close"].dtype == "float64"
    assert aligned["close"].tolist()[0] == 100.0
    assert aligned["close"].isna().tolist() == [False, True, True, False, True]
