"""Tests de la descarga paginada de klines con un Binance simulado, sin tocar la red."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from miax.ingest import download_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

STEP_MS = 60_000
T0 = 1_700_000_000_000


def _candle(open_time: int) -> list[Any]:
    """Una vela con el formato crudo de Binance: precios como texto, tiempos y trades como int."""
    price = str(100.0 + open_time % 7)
    return [
        open_time,
        price,
        price,
        price,
        price,
        "1.5",
        open_time + STEP_MS - 1,
        "150.0",
        10,
        "0.7",
        "70.0",
        "0",
    ]


class _FakeResponse:
    """Respuesta simulada que solo expone `json()`."""

    def __init__(self, payload: list[list[Any]]) -> None:
        """Guarda el cuerpo que devolvera `json()`."""
        self._payload = payload

    def json(self) -> list[list[Any]]:
        """Devuelve el cuerpo simulado."""
        return self._payload


class _FakeBinance:
    """Cliente falso: sirve velas de 1 minuto respetando startTime, endTime y limit."""

    def __init__(self, n_candles: int, gap: range | None = None) -> None:
        """Genera `n_candles` velas a partir de T0, saltando los indices de `gap` si se da."""
        skipped = set(gap) if gap is not None else set()
        self.candles = [_candle(T0 + i * STEP_MS) for i in range(n_candles) if i not in skipped]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        """Registra la llamada y devuelve el bloque que Binance serviria."""
        assert params is not None
        self.calls.append((url, dict(params)))
        selected = [c for c in self.candles if params["startTime"] <= c[0] <= params["endTime"]]
        return _FakeResponse(selected[: params["limit"]])

    def close(self) -> None:
        """Marca el cliente como cerrado."""
        self.closed = True


def _end_of(n_candles: int) -> int:
    """`open_time` de la ultima vela de una serie de `n_candles`."""
    return T0 + (n_candles - 1) * STEP_MS


def test_range_over_1000_candles_is_downloaded_in_full() -> None:
    """1000 + 1000 + 250 velas dan 2250 filas en orden y sin duplicar la vela frontera."""
    fake = _FakeBinance(2250)
    df = download_klines("BTCUSDT", "1m", T0, _end_of(2250), client=fake)  # type: ignore[arg-type]
    assert len(fake.calls) == 3
    assert len(df) == 2250
    assert df["open_time"].is_monotonic_increasing
    assert df["open_time"].is_unique
    assert df["open_time"].iloc[0] == T0
    assert df["open_time"].iloc[-1] == _end_of(2250)


def test_cursor_advances_to_last_open_time_plus_one() -> None:
    """Cada llamada pide desde la `open_time` de la ultima vela anterior + 1 ms."""
    fake = _FakeBinance(2250)
    end = _end_of(2250)
    download_klines("BTCUSDT", "1m", T0, end, client=fake)  # type: ignore[arg-type]
    starts = [params["startTime"] for _, params in fake.calls]
    assert starts == [T0, _end_of(1000) + 1, _end_of(2000) + 1]
    for url, params in fake.calls:
        assert url == "https://api.binance.com/api/v3/klines"
        assert params["symbol"] == "BTCUSDT"
        assert params["interval"] == "1m"
        assert params["endTime"] == end
        assert params["limit"] == 1000


def test_stops_on_empty_block() -> None:
    """Si tras un bloque completo el siguiente viene vacio, se corta sin mas llamadas."""
    fake = _FakeBinance(1000)
    df = download_klines("BTCUSDT", "1m", T0, T0 + 5000 * STEP_MS, client=fake)  # type: ignore[arg-type]
    assert len(fake.calls) == 2
    assert len(df) == 1000


def test_short_block_does_not_stop_pagination() -> None:
    """Un bloque corto no corta: se pide otro y solo el bloque vacio cierra la descarga."""
    fake = _FakeBinance(1250)
    df = download_klines("BTCUSDT", "1m", T0, T0 + 5000 * STEP_MS, client=fake)  # type: ignore[arg-type]
    assert len(fake.calls) == 3
    assert len(df) == 1250


def test_gaps_are_downloaded_in_full_without_duplicates() -> None:
    """Con `open_time` salteados a mitad de rango se descargan todas las velas, en orden."""
    n_total = 1600
    fake = _FakeBinance(n_total, gap=range(700, 800))
    df = download_klines("BTCUSDT", "1m", T0, _end_of(n_total), client=fake, limit=500)  # type: ignore[arg-type]
    expected = [c[0] for c in fake.candles]
    assert len(expected) == 1500
    assert len(fake.calls) == 3
    assert df["open_time"].tolist() == expected
    assert df["open_time"].is_unique


def test_stops_when_cursor_passes_end_without_extra_call() -> None:
    """Un bloque lleno que acaba justo en `end_ms` cierra la descarga sin pedir otro bloque."""
    fake = _FakeBinance(2000)
    df = download_klines("BTCUSDT", "1m", T0, _end_of(2000), client=fake)  # type: ignore[arg-type]
    assert len(fake.calls) == 2
    assert len(df) == 2000
    assert df["open_time"].iloc[-1] == _end_of(2000)


def test_single_block_makes_a_single_call() -> None:
    """Un rango con menos velas que `limit` se descarga en una sola llamada."""
    fake = _FakeBinance(300)
    df = download_klines("ETHUSDT", "1m", T0, _end_of(300), client=fake)  # type: ignore[arg-type]
    assert len(fake.calls) == 1
    assert len(df) == 300


def test_stops_when_cursor_does_not_advance() -> None:
    """Un bloque que no avanza el cursor no se anade y la paginacion termina en vez de ciclar."""

    class _Stuck:
        """Cliente que devuelve siempre el mismo bloque lleno, anterior al cursor."""

        calls = 0

        def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
            """Devuelve `limit` velas cuya ultima es anterior a startTime."""
            type(self).calls += 1
            assert params is not None
            return _FakeResponse([_candle(T0 + i * STEP_MS) for i in range(params["limit"])])

    start = T0 + 10_000 * STEP_MS
    df = download_klines("BTCUSDT", "1m", start, start + 20_000 * STEP_MS, client=_Stuck())  # type: ignore[arg-type]
    assert _Stuck.calls == 1
    assert df.empty


def test_empty_range_returns_empty_frame_with_columns() -> None:
    """Sin datos en el rango se devuelve un DataFrame vacio con las columnas y dtypes."""
    fake = _FakeBinance(0)
    df = download_klines("BTCUSDT", "1m", T0, T0 + STEP_MS, client=fake)  # type: ignore[arg-type]
    assert df.empty
    assert list(df.columns) == KLINE_COLUMNS
    assert {c: str(t) for c, t in df.dtypes.items()} == KLINE_DTYPES


def test_start_after_end_makes_no_calls() -> None:
    """Un rango invertido no llama a la API y devuelve un DataFrame vacio."""
    fake = _FakeBinance(10)
    df = download_klines("BTCUSDT", "1m", T0 + STEP_MS, T0, client=fake)  # type: ignore[arg-type]
    assert fake.calls == []
    assert df.empty


def test_columns_and_dtypes() -> None:
    """El DataFrame lleva las columnas crudas de Binance con sus dtypes numericos."""
    fake = _FakeBinance(5)
    df = download_klines("BTCUSDT", "1m", T0, _end_of(5), client=fake)  # type: ignore[arg-type]
    assert list(df.columns) == KLINE_COLUMNS
    assert {c: str(t) for c, t in df.dtypes.items()} == KLINE_DTYPES
    assert df["close"].iloc[0] == pytest.approx(float(fake.candles[0][4]))
    assert df["num_trades"].iloc[0] == 10
    assert isinstance(df, pd.DataFrame)


def test_invalid_limit_raises() -> None:
    """Un `limit` fuera de 1..1000 es un error de uso."""
    fake = _FakeBinance(1)
    with pytest.raises(ValueError, match="limit"):
        download_klines("BTCUSDT", "1m", T0, T0, client=fake, limit=1001)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        download_klines("BTCUSDT", "1m", T0, T0, client=fake, limit=0)  # type: ignore[arg-type]


def test_injected_client_is_not_closed_but_default_one_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El cliente inyectado sigue abierto; el creado por defecto se cierra al terminar."""
    injected = _FakeBinance(3)
    download_klines("BTCUSDT", "1m", T0, _end_of(3), client=injected)  # type: ignore[arg-type]
    assert injected.closed is False

    created = _FakeBinance(3)
    monkeypatch.setattr("miax.ingest.klines.HttpClient", lambda: created)
    df = download_klines("BTCUSDT", "1m", T0, _end_of(3))
    assert created.closed is True
    assert len(df) == 3
