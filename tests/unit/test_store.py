"""Tests del cacheo de klines en Parquet por mes; usan `tmp_path`, sin red y sin tocar `data/`."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from miax.ingest import load_klines, save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

HOUR_MS = 3_600_000
SYMBOL = "BTCUSDT"
INTERVAL = "1h"


def _ms(timestamp: str) -> int:
    """Convierte una fecha UTC en milisegundos desde epoch."""
    return int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1000)


# Cruza enero, febrero y marzo de 2024 (2024 es bisiesto).
T_START = _ms("2024-01-31 22:00")


def _frame(n: int = 100, start: int = T_START, step: int = 24 * HOUR_MS) -> pd.DataFrame:
    """DataFrame sintetico con las columnas y dtypes crudos de Binance, desordenado."""
    rng = np.random.default_rng(42)
    open_time = start + np.arange(n, dtype="int64") * step
    data = {
        "open_time": open_time,
        "open": rng.uniform(100, 200, n),
        "high": rng.uniform(100, 200, n),
        "low": rng.uniform(100, 200, n),
        "close": rng.uniform(100, 200, n),
        "volume": rng.uniform(0, 10, n),
        "close_time": open_time + HOUR_MS - 1,
        "quote_asset_volume": rng.uniform(0, 1000, n),
        "num_trades": rng.integers(0, 500, n),
        "taker_buy_base": rng.uniform(0, 5, n),
        "taker_buy_quote": rng.uniform(0, 500, n),
        "ignore": np.zeros(n),
    }
    frame = pd.DataFrame(data).astype(KLINE_DTYPES)
    return frame.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_one_file_per_month(tmp_path: Path) -> None:
    """Cada mes UTC que aparece en los datos genera su propio `YYYY-MM.parquet`."""
    df = _frame(n=35)  # 35 dias desde el 2024-01-31 22:00: enero, febrero y marzo
    paths = save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    directory = tmp_path / SYMBOL / INTERVAL
    assert [p.name for p in paths] == ["2024-01.parquet", "2024-02.parquet", "2024-03.parquet"]
    assert sorted(p.name for p in directory.iterdir()) == [p.name for p in paths]
    assert all(p.parent == directory for p in paths)


def test_month_boundary_is_utc(tmp_path: Path) -> None:
    """La vela de las 23:00 UTC del 31 de enero va a enero y la de las 00:00 UTC, a febrero."""
    df = _frame(n=2, start=_ms("2024-01-31 23:00"), step=HOUR_MS)
    paths = save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    assert [p.name for p in paths] == ["2024-01.parquet", "2024-02.parquet"]
    assert (
        len(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path, end_ms=_ms("2024-01-31 23:00"))) == 1
    )


def test_round_trip_is_identical(tmp_path: Path) -> None:
    """Guardar y releer devuelve exactamente lo mismo, con los mismos dtypes."""
    df = _frame(n=35)
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    expected = df.sort_values("open_time").reset_index(drop=True)

    assert_frame_equal(loaded, expected)
    assert list(loaded.columns) == KLINE_COLUMNS
    assert loaded.dtypes.astype(str).to_dict() == KLINE_DTYPES


def test_save_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    """Guardar dos veces lo mismo, en cualquier orden de entrada, deja el mismo resultado."""
    df = _frame(n=35)
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)
    first = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)

    save_klines(df.iloc[::-1], SYMBOL, INTERVAL, base_dir=tmp_path)
    second = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)

    assert_frame_equal(first, second)


def test_range_reads_only_months_in_range(tmp_path: Path) -> None:
    """Con rango solo se leen los meses que lo solapan y las filas fuera de el se descartan."""
    df = _frame(n=35)
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)
    (tmp_path / SYMBOL / INTERVAL / "2024-02.parquet").unlink()

    start_ms, end_ms = _ms("2024-03-01 00:00"), _ms("2024-03-31 23:00")
    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path, start_ms=start_ms, end_ms=end_ms)

    expected = df.sort_values("open_time")
    expected = expected[(expected["open_time"] >= start_ms) & (expected["open_time"] <= end_ms)]
    assert_frame_equal(loaded, expected.reset_index(drop=True))
    assert not loaded.empty

    # Febrero ya no existe: pedirlo todo solo devuelve enero y marzo, sin error.
    everything = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    months = pd.to_datetime(everything["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m")
    assert sorted(months.unique()) == ["2024-01", "2024-03"]


def test_range_is_inclusive_on_both_ends(tmp_path: Path) -> None:
    """`start_ms` y `end_ms` son inclusivos, igual que en la descarga."""
    df = _frame(n=10, start=_ms("2024-02-01 00:00"), step=HOUR_MS)
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    start_ms, end_ms = _ms("2024-02-01 02:00"), _ms("2024-02-01 04:00")
    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path, start_ms=start_ms, end_ms=end_ms)

    assert loaded["open_time"].tolist() == [start_ms, start_ms + HOUR_MS, end_ms]


def test_missing_symbol_returns_empty_frame_with_columns(tmp_path: Path) -> None:
    """Sin datos para el simbolo se devuelve un DataFrame vacio con columnas y dtypes."""
    save_klines(_frame(n=5), SYMBOL, INTERVAL, base_dir=tmp_path)

    loaded = load_klines("ETHUSDT", INTERVAL, base_dir=tmp_path)

    assert loaded.empty
    assert list(loaded.columns) == KLINE_COLUMNS
    assert loaded.dtypes.astype(str).to_dict() == KLINE_DTYPES


def test_range_outside_data_returns_empty_frame(tmp_path: Path) -> None:
    """Un rango sin ningun mes guardado devuelve el DataFrame vacio con las columnas correctas."""
    save_klines(_frame(n=5), SYMBOL, INTERVAL, base_dir=tmp_path)

    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path, start_ms=_ms("2025-01-01 00:00"))

    assert loaded.empty
    assert list(loaded.columns) == KLINE_COLUMNS


def test_saving_empty_frame_writes_nothing(tmp_path: Path) -> None:
    """Guardar un DataFrame vacio no crea ficheros ni directorios."""
    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in KLINE_DTYPES.items()})

    assert save_klines(empty, SYMBOL, INTERVAL, base_dir=tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_range_crossing_months_midway_returns_exact_candles(tmp_path: Path) -> None:
    """Un rango que empieza a mitad de un mes y acaba a mitad de otro devuelve justo esas velas."""
    df = _frame(n=35)
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    start_ms, end_ms = _ms("2024-01-31 22:00") + 20 * 24 * HOUR_MS, _ms("2024-03-03 22:00")
    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path, start_ms=start_ms, end_ms=end_ms)

    expected = df.sort_values("open_time")
    expected = expected[(expected["open_time"] >= start_ms) & (expected["open_time"] <= end_ms)]
    assert_frame_equal(loaded, expected.reset_index(drop=True))
    months = pd.to_datetime(loaded["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m")
    assert sorted(months.unique()) == ["2024-02", "2024-03"]
    assert loaded["open_time"].min() == start_ms
    assert loaded["open_time"].max() == end_ms


def test_saving_existing_month_overwrites_it_entirely(tmp_path: Path) -> None:
    """Guardar de nuevo un mes ya existente lo sustituye entero, sin mezclar velas antiguas."""
    first = _frame(n=10, start=_ms("2024-02-01 00:00"), step=HOUR_MS)
    second = _frame(n=3, start=_ms("2024-02-01 05:00"), step=HOUR_MS)
    save_klines(first, SYMBOL, INTERVAL, base_dir=tmp_path)
    save_klines(second, SYMBOL, INTERVAL, base_dir=tmp_path)

    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)

    assert_frame_equal(loaded, second.sort_values("open_time").reset_index(drop=True))
    assert len(loaded) == 3


def test_foreign_file_in_directory_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Un fichero ajeno a `YYYY-MM.parquet` (incluido un mes imposible) se ignora con warning."""
    # El logger del proyecto no propaga; se activa solo en este test para capturarlo.
    monkeypatch.setattr(logging.getLogger("miax.ingest.store"), "propagate", True)
    df = _frame(n=35)
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)
    directory = tmp_path / SYMBOL / INTERVAL
    (directory / "notas.parquet").write_bytes(b"no es parquet")
    (directory / "2024-1.parquet").write_bytes(b"no es parquet")
    (directory / "2024-13.parquet").write_bytes(b"no es parquet")

    with caplog.at_level(logging.WARNING, logger="miax.ingest.store"):
        loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)

    assert_frame_equal(loaded, df.sort_values("open_time").reset_index(drop=True))
    assert any("2024-13.parquet" in r.getMessage() for r in caplog.records)


def test_last_ms_of_month_stays_in_that_month(tmp_path: Path) -> None:
    """La vela del ultimo ms de enero (UTC) va a enero y el rango la excluye de febrero."""
    last_ms_january = _ms("2024-02-01 00:00") - 1
    df = _frame(n=2, start=last_ms_january, step=1)
    paths = save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    assert [p.name for p in paths] == ["2024-01.parquet", "2024-02.parquet"]
    february = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path, start_ms=last_ms_january + 1)
    assert february["open_time"].tolist() == [last_ms_january + 1]


def test_extra_columns_are_dropped_on_save(tmp_path: Path) -> None:
    """Las columnas extra se descartan al guardar: releer devuelve solo las canonicas."""
    df = _frame(n=5)
    df["extra"] = 1.0
    save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)

    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)

    assert list(loaded.columns) == KLINE_COLUMNS
    assert_frame_equal(loaded, df[KLINE_COLUMNS].sort_values("open_time").reset_index(drop=True))


def test_missing_columns_fail_on_save(tmp_path: Path) -> None:
    """Si faltan columnas canonicas, el error salta al guardar y no se escribe nada."""
    df = _frame(n=5).drop(columns=["close"])

    with pytest.raises(ValueError, match="close"):
        save_klines(df, SYMBOL, INTERVAL, base_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad",
    ["..", ".", "", "a/b", "a\\b", "..\\escape", "C:\\escape", "C:", "../escape", "/abs", "a\x00b"],
)
def test_invalid_path_segments_are_rejected(tmp_path: Path, bad: str) -> None:
    """`symbol` e `interval` con separadores, `..` o vacios se rechazan y nada se escribe."""
    base = tmp_path / "base"
    df = _frame(n=3)

    for kwargs in ({"symbol": bad, "interval": INTERVAL}, {"symbol": SYMBOL, "interval": bad}):
        with pytest.raises(ValueError):
            save_klines(df, base_dir=base, **kwargs)
        with pytest.raises(ValueError):
            load_klines(base_dir=base, **kwargs)

    assert list(tmp_path.iterdir()) == []


def test_absolute_path_segment_is_rejected(tmp_path: Path) -> None:
    """Una ruta absoluta como `symbol` se rechaza y no se escribe fuera de `base_dir`."""
    outside = tmp_path / "outside"
    base = tmp_path / "base"

    with pytest.raises(ValueError):
        save_klines(_frame(n=3), str(outside), INTERVAL, base_dir=base)
    with pytest.raises(ValueError):
        load_klines(str(outside), INTERVAL, base_dir=base)

    assert list(tmp_path.iterdir()) == []
