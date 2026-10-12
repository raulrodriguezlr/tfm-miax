"""Tests del filtro de universo por volumen con un Binance simulado, sin tocar la red."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest

from miax.ingest import universe
from miax.ingest.universe import filter_by_volume, median_daily_quote_volume

DAY_MS = 86_400_000
DAY0 = 1_700_000_000_000 - (1_700_000_000_000 % DAY_MS)  # medianoche UTC


def _candle(open_time: int, quote_volume: float) -> list[Any]:
    """Una vela diaria con el formato crudo de Binance y el volumen en USDT dado."""
    return [
        open_time,
        "100.0",
        "100.0",
        "100.0",
        "100.0",
        "1.5",
        open_time + DAY_MS - 1,
        str(quote_volume),
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
    """Cliente falso: velas diarias por simbolo respetando startTime, endTime y limit."""

    def __init__(self, volumes: Mapping[str, Sequence[float]]) -> None:
        """Cada simbolo tiene una vela por dia desde DAY0 con los volumenes dados."""
        self.candles = {
            sym: [_candle(DAY0 + i * DAY_MS, v) for i, v in enumerate(vols)]
            for sym, vols in volumes.items()
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> _FakeResponse:
        """Registra la llamada y devuelve el bloque que Binance serviria."""
        assert params is not None
        self.calls.append((url, dict(params)))
        rows = self.candles.get(params["symbol"], [])
        selected = [c for c in rows if params["startTime"] <= c[0] <= params["endTime"]]
        return _FakeResponse(selected[: params["limit"]])

    def close(self) -> None:
        """Marca el cliente como cerrado."""
        self.closed = True


def _end_of_day(day_index: int) -> int:
    """Ultimo milisegundo del dia `day_index` (cuenta desde DAY0)."""
    return DAY0 + (day_index + 1) * DAY_MS - 1


def test_median_is_correct_per_symbol() -> None:
    """La mediana se calcula por simbolo sobre los dias de la ventana."""
    fake = _FakeBinance({"AAA": [1.0, 5.0, 3.0], "BBB": [10.0, 20.0, 30.0, 40.0]})
    result = median_daily_quote_volume(["AAA", "BBB"], DAY0, _end_of_day(3), client=fake)  # type: ignore[arg-type]
    assert result["AAA"] == 3.0
    assert result["BBB"] == 25.0


def test_pump_day_does_not_move_median() -> None:
    """Un dia de pump enorme no altera la mediana."""
    fake = _FakeBinance({"AAA": [10.0, 11.0, 12.0, 13.0, 1e12]})
    result = median_daily_quote_volume(["AAA"], DAY0, _end_of_day(4), client=fake)  # type: ignore[arg-type]
    assert result["AAA"] == 12.0


def test_symbol_without_candles_is_nan() -> None:
    """Un simbolo sin velas en la ventana da NaN, sin romper el resto."""
    fake = _FakeBinance({"AAA": [1.0, 2.0, 3.0]})
    result = median_daily_quote_volume(["AAA", "ZZZ"], DAY0, _end_of_day(2), client=fake)  # type: ignore[arg-type]
    assert math.isnan(result["ZZZ"])
    assert result["AAA"] == 2.0


def test_series_shape_name_and_dtype() -> None:
    """Serie indexada por simbolo, con nombre y dtype float64."""
    fake = _FakeBinance({"AAA": [1.0], "BBB": [2.0]})
    result = median_daily_quote_volume(["AAA", "BBB"], DAY0, _end_of_day(0), client=fake)  # type: ignore[arg-type]
    assert isinstance(result, pd.Series)
    assert list(result.index) == ["AAA", "BBB"]
    assert result.name == "median_quote_volume"
    assert result.dtype == "float64"


def test_params_request_daily_interval_and_window() -> None:
    """Las peticiones piden intervalo 1d y la ventana dada."""
    fake = _FakeBinance({"AAA": [1.0, 2.0]})
    start, end = DAY0, _end_of_day(1)
    median_daily_quote_volume(["AAA"], start, end, client=fake)  # type: ignore[arg-type]
    assert fake.calls
    for _, params in fake.calls:
        assert params["interval"] == "1d"
        assert params["symbol"] == "AAA"
        assert params["startTime"] >= start
        assert params["endTime"] == end


def test_start_after_end_raises() -> None:
    """Una ventana invertida es un error."""
    fake = _FakeBinance({"AAA": [1.0]})
    with pytest.raises(ValueError, match="start_ms"):
        median_daily_quote_volume(["AAA"], DAY0 + 1, DAY0, client=fake)  # type: ignore[arg-type]


def test_injected_client_is_not_closed() -> None:
    """Un cliente inyectado lo cierra quien lo creo, no la funcion."""
    fake = _FakeBinance({"AAA": [1.0]})
    median_daily_quote_volume(["AAA"], DAY0, _end_of_day(0), client=fake)  # type: ignore[arg-type]
    assert fake.closed is False


def test_own_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un cliente creado por la funcion se cierra siempre."""
    created: list[_FakeBinance] = []

    def factory(*args: object, **kwargs: object) -> _FakeBinance:
        fake = _FakeBinance({"AAA": [1.0, 2.0]})
        created.append(fake)
        return fake

    monkeypatch.setattr(universe, "HttpClient", factory)
    result = median_daily_quote_volume(["AAA"], DAY0, _end_of_day(1))
    assert len(created) == 1
    assert created[0].closed is True
    assert result["AAA"] == 1.5


def test_own_client_is_closed_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la descarga falla, el cliente propio tambien se cierra."""
    fake = _FakeBinance({"AAA": [1.0]})

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(universe, "HttpClient", lambda *a, **k: fake)
    monkeypatch.setattr(universe, "download_klines", boom)
    with pytest.raises(RuntimeError):
        median_daily_quote_volume(["AAA"], DAY0, _end_of_day(0))
    assert fake.closed is True


def test_duplicate_symbols_download_once() -> None:
    """Un simbolo repetido se descarga una sola vez."""
    fake = _FakeBinance({"AAA": [1.0, 2.0, 3.0]})
    result = median_daily_quote_volume(["AAA", "AAA", "AAA"], DAY0, _end_of_day(2), client=fake)  # type: ignore[arg-type]
    first_pages = [p for _, p in fake.calls if p["startTime"] == DAY0]
    assert len(first_pages) == 1  # una descarga; la paginacion anade una pagina vacia final
    assert list(result.index) == ["AAA"]


def test_no_leak_candle_closing_after_end_is_ignored() -> None:
    """ANTIFUGA: con end_ms a mitad de dia, la vela de ese dia (aun abierta) no cuenta."""
    base = [10.0, 20.0, 30.0]
    midday = DAY0 + 3 * DAY_MS + 12 * 3_600_000  # 12:00 UTC del cuarto dia
    clean = _FakeBinance({"AAA": base})
    with_future = _FakeBinance({"AAA": [*base, 1e15]})  # el cuarto dia cierra tras end_ms
    expected = median_daily_quote_volume(["AAA"], DAY0, midday, client=clean)  # type: ignore[arg-type]
    got = median_daily_quote_volume(["AAA"], DAY0, midday, client=with_future)  # type: ignore[arg-type]
    assert got["AAA"] == expected["AAA"] == 20.0


def test_candle_closing_exactly_at_end_counts() -> None:
    """Con end_ms en el ultimo ms del dia, la vela de ese dia ya esta cerrada y cuenta."""
    fake = _FakeBinance({"AAA": [10.0, 20.0, 30.0, 1e15]})
    result = median_daily_quote_volume(["AAA"], DAY0, _end_of_day(3), client=fake)  # type: ignore[arg-type]
    assert result["AAA"] == (20.0 + 30.0) / 2


def _pairs() -> pd.DataFrame:
    """Catalogo de pares minimo."""
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "base_asset": ["A", "B", "C", "D"],
        }
    )


def test_filter_threshold_is_inclusive_and_sorted_desc() -> None:
    """El umbral es >= y el resultado va de mayor a menor con indice 0..n-1."""
    volumes = pd.Series({"AAA": 100.0, "BBB": 500.0, "CCC": 99.9, "DDD": 300.0})
    out = filter_by_volume(_pairs(), volumes, min_quote_volume=100.0)
    assert list(out["symbol"]) == ["BBB", "DDD", "AAA"]
    assert list(out.index) == [0, 1, 2]
    assert list(out["median_quote_volume"]) == [500.0, 300.0, 100.0]


def test_filter_keeps_columns_and_adds_volume() -> None:
    """Conserva las columnas de pairs y anade median_quote_volume."""
    volumes = pd.Series({"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0})
    out = filter_by_volume(_pairs(), volumes, min_quote_volume=0.0)
    assert list(out.columns) == ["symbol", "base_asset", "median_quote_volume"]
    assert out["median_quote_volume"].dtype == "float64"


def test_filter_drops_nan_and_missing_symbols() -> None:
    """NaN y simbolos ausentes de volumes se descartan."""
    volumes = pd.Series({"AAA": float("nan"), "BBB": 50.0, "CCC": 60.0})  # DDD ausente
    out = filter_by_volume(_pairs(), volumes, min_quote_volume=0.0)
    assert list(out["symbol"]) == ["CCC", "BBB"]


def test_filter_negative_threshold_raises() -> None:
    """Un umbral negativo es un error."""
    with pytest.raises(ValueError, match="negativo"):
        filter_by_volume(_pairs(), pd.Series({"AAA": 1.0}), min_quote_volume=-1.0)


def test_filter_without_symbol_column_raises() -> None:
    """Sin columna symbol no se puede cruzar."""
    with pytest.raises(ValueError, match="symbol"):
        filter_by_volume(pd.DataFrame({"x": [1]}), pd.Series({"AAA": 1.0}))


def test_filter_nothing_passes_returns_empty_with_columns() -> None:
    """Si nada pasa el umbral, DataFrame vacio pero con columnas."""
    volumes = pd.Series({"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0})
    out = filter_by_volume(_pairs(), volumes, min_quote_volume=1e9)
    assert out.empty
    assert list(out.columns) == ["symbol", "base_asset", "median_quote_volume"]


def test_filter_does_not_mutate_input() -> None:
    """La entrada no se modifica."""
    pairs = _pairs()
    filter_by_volume(pairs, pd.Series({"AAA": 1.0}), min_quote_volume=0.0)
    assert list(pairs.columns) == ["symbol", "base_asset"]
