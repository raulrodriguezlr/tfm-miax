"""Tests de la reanudacion de descargas con un cliente espia: sin red y sobre `tmp_path`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from miax.ingest import cached_last_open_time, load_klines, save_klines, update_symbol
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.utils.io import write_parquet

HOUR_MS = 3_600_000
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
# Arranca el 2024-01-31 00:00 UTC: 72 horas cruzan de enero a febrero.
T0 = int(pd.Timestamp("2024-01-31", tz="UTC").timestamp() * 1000)
N = 72


def _candle(open_time: int) -> list[Any]:
    """Una vela cruda de Binance determinista a partir de su `open_time`."""
    price = str(100.0 + open_time % 7)
    return [open_time, price, price, price, price, "1.5", open_time + HOUR_MS - 1, "150.0", 10]


def _row(open_time: int) -> list[Any]:
    """La vela cruda completada con las dos ultimas columnas de Binance."""
    return [*_candle(open_time), "0.7", "70.0", "0"]


class _FakeResponse:
    """Respuesta simulada que solo expone `json()`."""

    def __init__(self, payload: list[list[Any]]) -> None:
        """Guarda el cuerpo que devolvera `json()`."""
        self._payload = payload

    def json(self) -> list[list[Any]]:
        """Devuelve el cuerpo simulado."""
        return self._payload


class _SpyBinance:
    """Cliente falso que sirve velas horarias (startTime/endTime/limit) y cuenta las llamadas."""

    def __init__(self, n_candles: int = N) -> None:
        """Genera `n_candles` velas horarias desde T0."""
        self.candles = [_row(T0 + i * HOUR_MS) for i in range(n_candles)]
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        """Registra la llamada y devuelve el bloque que Binance serviria."""
        assert params is not None
        self.calls.append(dict(params))
        selected = [c for c in self.candles if params["startTime"] <= c[0] <= params["endTime"]]
        return _FakeResponse(selected[: params["limit"]])

    def close(self) -> None:
        """No hace nada: el espia no tiene recursos que liberar."""


def _end_of(n_candles: int) -> int:
    """`open_time` de la ultima vela de una serie de `n_candles`."""
    return T0 + (n_candles - 1) * HOUR_MS


def _update(spy: _SpyBinance, base_dir: Path, start_ms: int = T0, end_ms: int | None = None) -> int:
    """Ejecuta `update_symbol` con el espia y `base_dir` dados."""
    return update_symbol(
        SYMBOL,
        INTERVAL,
        start_ms,
        _end_of(N) if end_ms is None else end_ms,
        client=spy,  # type: ignore[arg-type]
        base_dir=base_dir,
    )


def _seed_cache(base_dir: Path, n_candles: int) -> None:
    """Guarda en la cache las primeras `n_candles` velas, como si una descarga previa se cortara."""
    frame = pd.DataFrame([_row(T0 + i * HOUR_MS) for i in range(n_candles)], columns=KLINE_COLUMNS)
    save_klines(frame.astype(KLINE_DTYPES), SYMBOL, INTERVAL, base_dir=base_dir)


def test_cached_last_open_time_is_none_without_cache(tmp_path: Path) -> None:
    """Sin ficheros guardados no hay ultima vela cacheada."""
    assert cached_last_open_time(SYMBOL, INTERVAL, base_dir=tmp_path) is None


def test_cached_last_open_time_spans_several_months(tmp_path: Path) -> None:
    """Con varios meses guardados devuelve el `open_time` maximo del mes mas reciente."""
    _seed_cache(tmp_path, N)  # enero (31) y febrero (1 y 2)
    assert (tmp_path / SYMBOL / INTERVAL / "2024-01.parquet").exists()
    assert (tmp_path / SYMBOL / INTERVAL / "2024-02.parquet").exists()
    assert cached_last_open_time(SYMBOL, INTERVAL, base_dir=tmp_path) == _end_of(N)


def _empty_frame() -> pd.DataFrame:
    """Un DataFrame sin filas con las columnas y dtypes de las klines."""
    return pd.DataFrame({c: pd.Series(dtype=KLINE_DTYPES[c]) for c in KLINE_COLUMNS})


def test_cached_last_open_time_ignores_empty_latest_month(tmp_path: Path) -> None:
    """Si el mes mas reciente esta vacio se ignora y se usa el ultimo mes con velas."""
    _seed_cache(tmp_path, 24)  # solo enero (el 31)
    directory = tmp_path / SYMBOL / INTERVAL
    write_parquet(_empty_frame(), directory / "2024-03.parquet")

    assert cached_last_open_time(SYMBOL, INTERVAL, base_dir=tmp_path) == _end_of(24)


def test_cached_last_open_time_ignores_intruder_entries(tmp_path: Path) -> None:
    """Ficheros con nombre no mensual y directorios ajenos no cuentan como meses cacheados."""
    _seed_cache(tmp_path, 24)  # solo enero (el 31)
    directory = tmp_path / SYMBOL / INTERVAL
    # Los intrusos llevan velas posteriores: si se leyeran, el maximo cambiaria.
    later = pd.DataFrame([_row(_end_of(N) + 100 * HOUR_MS)], columns=KLINE_COLUMNS)
    later = later.astype(KLINE_DTYPES)
    write_parquet(later, directory / "notes.parquet")
    write_parquet(later, directory / "2024-13.parquet")
    (directory / "extra").mkdir()
    (directory / "old.parquet").mkdir()

    assert cached_last_open_time(SYMBOL, INTERVAL, base_dir=tmp_path) == _end_of(24)


def test_cached_last_open_time_rejects_unsafe_names(tmp_path: Path) -> None:
    """Un simbolo con separadores de ruta se rechaza igual que en el almacen."""
    with pytest.raises(ValueError):
        cached_last_open_time("../x", INTERVAL, base_dir=tmp_path)


def test_fully_cached_range_makes_no_calls(tmp_path: Path) -> None:
    """Pedir un rango ya cubierto no llama al cliente y no cambia los datos."""
    _seed_cache(tmp_path, N)
    before = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    spy = _SpyBinance()

    assert _update(spy, tmp_path) == 0
    assert spy.calls == []
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), before)


def test_end_before_last_cached_makes_no_calls(tmp_path: Path) -> None:
    """Si la cache va mas alla de `end_ms` no se llama al cliente y se devuelve 0."""
    _seed_cache(tmp_path, N)
    spy = _SpyBinance()

    assert _update(spy, tmp_path, end_ms=_end_of(10)) == 0
    assert spy.calls == []


def test_start_after_end_without_cache_makes_no_calls(tmp_path: Path) -> None:
    """Un rango invertido (`start_ms > end_ms`) sin cache no llama al cliente y devuelve 0."""
    spy = _SpyBinance()

    assert _update(spy, tmp_path, start_ms=_end_of(10), end_ms=T0) == 0
    assert spy.calls == []


def test_nothing_cached_downloads_full_range(tmp_path: Path) -> None:
    """Sin cache se descarga el rango completo desde `start_ms` y se guarda."""
    spy = _SpyBinance()

    assert _update(spy, tmp_path) == N
    assert spy.calls[0]["startTime"] == T0
    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    assert loaded["open_time"].tolist() == [T0 + i * HOUR_MS for i in range(N)]


def test_partial_cache_downloads_only_the_tail_without_truncating(tmp_path: Path) -> None:
    """Con cache parcial solo se pide desde la ultima + 1 y el mes parcial conserva sus velas."""
    cached_n = 30  # enero entero (24 h del 31) mas 6 h de febrero: febrero queda parcial
    _seed_cache(tmp_path, cached_n)
    spy = _SpyBinance()

    assert _update(spy, tmp_path) == N - cached_n
    assert [c["startTime"] for c in spy.calls] == [_end_of(cached_n) + 1]
    loaded = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    assert loaded["open_time"].tolist() == [T0 + i * HOUR_MS for i in range(N)]
    assert loaded["open_time"].is_unique


def test_rerun_after_full_download_makes_no_calls(tmp_path: Path) -> None:
    """Relanzar tras una descarga completa es idempotente: cero llamadas."""
    assert _update(_SpyBinance(), tmp_path) == N
    first = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    second_spy = _SpyBinance()

    assert _update(second_spy, tmp_path) == 0
    assert second_spy.calls == []
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), first)


def test_start_after_cache_resumes_from_start(tmp_path: Path) -> None:
    """Si `start_ms` es posterior a lo cacheado, la descarga arranca en `start_ms`."""
    _seed_cache(tmp_path, 10)
    spy = _SpyBinance()
    start = T0 + 20 * HOUR_MS

    assert _update(spy, tmp_path, start_ms=start) == N - 20
    assert spy.calls[0]["startTime"] == start


def test_empty_download_returns_zero_and_keeps_cache(tmp_path: Path) -> None:
    """Si Binance no devuelve velas nuevas no se guarda nada y se devuelve 0."""
    _seed_cache(tmp_path, 10)
    before = load_klines(SYMBOL, INTERVAL, base_dir=tmp_path)
    spy = _SpyBinance(n_candles=10)  # no tiene nada posterior a lo cacheado

    assert _update(spy, tmp_path) == 0
    assert len(spy.calls) == 1
    pd.testing.assert_frame_equal(load_klines(SYMBOL, INTERVAL, base_dir=tmp_path), before)
