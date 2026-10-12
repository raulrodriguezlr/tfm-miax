"""Tests del filtro de universo por volumen con un Binance simulado, sin tocar la red."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest

from miax.ingest import universe
from miax.ingest.universe import (
    filter_by_listing_age,
    filter_by_volume,
    first_candle_time,
    median_daily_quote_volume,
)

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

    def __init__(
        self,
        volumes: Mapping[str, Sequence[float]],
        start_days: Mapping[str, int] | None = None,
    ) -> None:
        """Una vela por dia con los volumenes dados, desde DAY0 + start_days[simbolo] dias."""
        offsets = start_days or {}
        self.candles = {
            sym: [_candle(DAY0 + (offsets.get(sym, 0) + i) * DAY_MS, v) for i, v in enumerate(vols)]
            for sym, vols in volumes.items()
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.weights: list[int] = []
        self.closed = False

    def get(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> _FakeResponse:
        """Registra la llamada y devuelve el bloque que Binance serviria."""
        assert params is not None
        self.calls.append((url, dict(params)))
        self.weights.append(weight)
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


# --- MIAX-026: filtro por antiguedad de listado ---

REFERENCE_MS = DAY0 + 200 * DAY_MS
CUTOFF_MS = REFERENCE_MS - 90 * DAY_MS  # corte con los 90 dias por defecto


def test_first_candle_returns_first_open_time() -> None:
    """Devuelve la apertura de la primera vela de cada simbolo, no la ultima."""
    fake = _FakeBinance({"AAA": [1.0, 2.0, 3.0], "BBB": [1.0, 2.0]}, start_days={"BBB": 5})
    result = first_candle_time(["AAA", "BBB"], DAY0 + 50 * DAY_MS, client=fake)  # type: ignore[arg-type]
    assert result["AAA"] == DAY0
    assert result["BBB"] == DAY0 + 5 * DAY_MS


def test_first_candle_na_when_unknown_or_listed_after_reference() -> None:
    """Un simbolo inexistente o listado tras la referencia da NA, sin romper el resto."""
    fake = _FakeBinance({"AAA": [1.0], "LATE": [1.0]}, start_days={"LATE": 20})
    result = first_candle_time(["AAA", "ZZZ", "LATE"], DAY0 + 10 * DAY_MS, client=fake)  # type: ignore[arg-type]
    assert result["AAA"] == DAY0
    assert pd.isna(result["ZZZ"])
    assert pd.isna(result["LATE"])


def test_first_candle_series_shape_name_and_dtype() -> None:
    """Serie indexada por simbolo, con nombre first_candle_ms y dtype Int64."""
    fake = _FakeBinance({"AAA": [1.0], "BBB": [1.0]})
    result = first_candle_time(["AAA", "BBB", "ZZZ"], DAY0 + DAY_MS, client=fake)  # type: ignore[arg-type]
    assert isinstance(result, pd.Series)
    assert list(result.index) == ["AAA", "BBB", "ZZZ"]
    assert result.name == "first_candle_ms"
    assert result.dtype == "Int64"


def test_first_candle_single_call_with_expected_params_and_weight() -> None:
    """ANTIFUGA: una llamada por simbolo con startTime=0, endTime=referencia, limit=1 y peso 2."""
    fake = _FakeBinance({"AAA": [1.0, 2.0], "BBB": [1.0]})
    reference = DAY0 + 30 * DAY_MS
    first_candle_time(["AAA", "BBB"], reference, client=fake)  # type: ignore[arg-type]
    assert [p["symbol"] for _, p in fake.calls] == ["AAA", "BBB"]
    for url, params in fake.calls:
        assert url.endswith("/api/v3/klines")
        assert params == {
            "symbol": params["symbol"],
            "interval": "1d",
            "startTime": 0,
            "endTime": reference,
            "limit": 1,
        }
    assert fake.weights == [2, 2]


def test_first_candle_never_requests_beyond_reference() -> None:
    """ANTIFUGA: una vela que abre un ms despues de la referencia no se ve; en la referencia si."""
    reference = DAY0 + 3 * DAY_MS
    on_reference = _FakeBinance({"AAA": [1.0]}, start_days={"AAA": 3})
    after_reference = _FakeBinance({"AAA": [1.0]}, start_days={"AAA": 3})
    assert first_candle_time(["AAA"], reference, client=on_reference)["AAA"] == reference  # type: ignore[arg-type]
    assert pd.isna(first_candle_time(["AAA"], reference - 1, client=after_reference)["AAA"])  # type: ignore[arg-type]
    assert all(p["endTime"] <= reference for _, p in on_reference.calls)
    assert all(p["endTime"] <= reference for _, p in after_reference.calls)


def test_first_candle_duplicates_one_call_and_order_kept() -> None:
    """Simbolos repetidos hacen una sola llamada y se conserva el orden de primera aparicion."""
    fake = _FakeBinance({"AAA": [1.0], "BBB": [1.0]})
    result = first_candle_time(["BBB", "AAA", "BBB", "AAA"], DAY0 + DAY_MS, client=fake)  # type: ignore[arg-type]
    assert list(result.index) == ["BBB", "AAA"]
    assert [p["symbol"] for _, p in fake.calls] == ["BBB", "AAA"]


def test_first_candle_injected_client_is_not_closed() -> None:
    """Un cliente inyectado lo cierra quien lo creo, no la funcion."""
    fake = _FakeBinance({"AAA": [1.0]})
    first_candle_time(["AAA"], DAY0 + DAY_MS, client=fake)  # type: ignore[arg-type]
    assert fake.closed is False


def test_first_candle_own_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un cliente creado por la funcion se cierra siempre."""
    created: list[_FakeBinance] = []

    def factory(*args: object, **kwargs: object) -> _FakeBinance:
        fake = _FakeBinance({"AAA": [1.0]})
        created.append(fake)
        return fake

    monkeypatch.setattr(universe, "HttpClient", factory)
    result = first_candle_time(["AAA"], DAY0 + DAY_MS)
    assert len(created) == 1
    assert created[0].closed is True
    assert result["AAA"] == DAY0


def test_first_candle_own_client_is_closed_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la peticion falla, el error se propaga y el cliente propio tambien se cierra."""
    fake = _FakeBinance({"AAA": [1.0]})

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(universe, "HttpClient", lambda *a, **k: fake)
    monkeypatch.setattr(fake, "get", boom)
    with pytest.raises(RuntimeError):
        first_candle_time(["AAA"], DAY0 + DAY_MS)
    assert fake.closed is True


def test_first_candle_negative_reference_raises() -> None:
    """Una referencia negativa es un error y no hace ninguna llamada."""
    fake = _FakeBinance({"AAA": [1.0]})
    with pytest.raises(ValueError, match="reference_ms"):
        first_candle_time(["AAA"], -1, client=fake)  # type: ignore[arg-type]
    assert fake.calls == []


def test_age_filter_cutoff_is_inclusive_and_one_ms_later_is_out() -> None:
    """ANTIFUGA: primera vela == corte entra; un ms despues queda fuera."""
    first = pd.Series({"AAA": CUTOFF_MS, "BBB": CUTOFF_MS + 1, "CCC": CUTOFF_MS - 1}, dtype="Int64")
    out = filter_by_listing_age(_pairs(), first, REFERENCE_MS)
    assert list(out["symbol"]) == ["AAA", "CCC"]


def test_age_filter_drops_na_and_missing_symbols() -> None:
    """Las primeras velas NA y los simbolos ausentes de first_candles se descartan."""
    first = pd.Series({"AAA": pd.NA, "BBB": DAY0}, dtype="Int64")  # CCC y DDD ausentes
    out = filter_by_listing_age(_pairs(), first, REFERENCE_MS)
    assert list(out["symbol"]) == ["BBB"]


def test_age_filter_keeps_input_order_columns_and_dtype() -> None:
    """Conserva el orden de entrada, indice 0..n-1, columnas de pairs y first_candle_ms Int64."""
    pairs = _pairs().iloc[::-1].reset_index(drop=True)  # DDD, CCC, BBB, AAA
    first = pd.Series({"AAA": DAY0, "BBB": DAY0, "CCC": DAY0, "DDD": DAY0}, dtype="Int64")
    out = filter_by_listing_age(pairs, first, REFERENCE_MS)
    assert list(out["symbol"]) == ["DDD", "CCC", "BBB", "AAA"]
    assert list(out.index) == [0, 1, 2, 3]
    assert list(out.columns) == ["symbol", "base_asset", "first_candle_ms"]
    assert out["first_candle_ms"].dtype == "Int64"


def test_age_filter_zero_days_lets_everything_listed_by_reference_through() -> None:
    """Con min_listing_days=0 pasa todo lo listado hasta la referencia (incluida)."""
    first = pd.Series(
        {"AAA": DAY0, "BBB": REFERENCE_MS, "CCC": REFERENCE_MS + 1, "DDD": pd.NA}, dtype="Int64"
    )
    out = filter_by_listing_age(_pairs(), first, REFERENCE_MS, min_listing_days=0)
    assert list(out["symbol"]) == ["AAA", "BBB"]


def test_age_filter_negative_days_raises() -> None:
    """Un minimo de dias negativo es un error."""
    with pytest.raises(ValueError, match="min_listing_days"):
        filter_by_listing_age(_pairs(), pd.Series({"AAA": DAY0}, dtype="Int64"), REFERENCE_MS, -1)


def test_age_filter_without_symbol_column_raises() -> None:
    """Sin columna symbol no se puede cruzar."""
    with pytest.raises(ValueError, match="symbol"):
        filter_by_listing_age(
            pd.DataFrame({"x": [1]}), pd.Series({"AAA": DAY0}, dtype="Int64"), REFERENCE_MS
        )


def test_age_filter_duplicate_index_in_first_candles_raises() -> None:
    """Un indice repetido en first_candles es un error en vez de resolverse en silencio."""
    first = pd.Series([DAY0, DAY0 + DAY_MS], index=["AAA", "AAA"], dtype="Int64")
    with pytest.raises(ValueError, match="repetidos"):
        filter_by_listing_age(_pairs(), first, REFERENCE_MS)


def test_age_filter_does_not_mutate_input() -> None:
    """La entrada no se modifica."""
    pairs = _pairs()
    first = pd.Series({"AAA": DAY0}, dtype="Int64")
    filter_by_listing_age(pairs, first, REFERENCE_MS)
    assert list(pairs.columns) == ["symbol", "base_asset"]


def test_age_and_volume_filters_commute_on_symbol_set() -> None:
    """Aplicar antiguedad y volumen en cualquier orden da el mismo conjunto de simbolos."""
    first = pd.Series({"AAA": DAY0, "BBB": CUTOFF_MS + 1, "CCC": DAY0, "DDD": pd.NA}, dtype="Int64")
    volumes = pd.Series({"AAA": 500.0, "BBB": 900.0, "CCC": 1.0, "DDD": 900.0})
    age_first = filter_by_volume(
        filter_by_listing_age(_pairs(), first, REFERENCE_MS), volumes, min_quote_volume=100.0
    )
    volume_first = filter_by_listing_age(
        filter_by_volume(_pairs(), volumes, min_quote_volume=100.0), first, REFERENCE_MS
    )
    assert set(age_first["symbol"]) == set(volume_first["symbol"]) == {"AAA"}


def test_arb_like_token_listed_inside_volume_window_is_dropped() -> None:
    """Un token listado dentro de la ventana de volumen (caso ARB) se descarta de punta a punta."""
    window_start = DAY0 + 200 * DAY_MS  # inicio de la ventana de volumen = referencia
    fake = _FakeBinance(
        {"OLD": [1.0], "ARB": [1.0], "NEW": [1.0]},
        start_days={"OLD": 80, "ARB": 205, "NEW": 150},  # OLD: 120 dias antes; NEW: 50 dias antes
    )
    pairs = pd.DataFrame({"symbol": ["OLD", "ARB", "NEW"], "base_asset": ["O", "A", "N"]})
    first = first_candle_time(list(pairs["symbol"]), window_start, client=fake)  # type: ignore[arg-type]
    assert pd.isna(first["ARB"])  # abre tras la referencia: ni se ve
    out = filter_by_listing_age(pairs, first, window_start)
    assert list(out["symbol"]) == ["OLD"]
