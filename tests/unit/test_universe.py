"""Tests del filtro de universo por volumen con un Binance simulado, sin tocar la red."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from miax.ingest import universe
from miax.ingest.universe import (
    MAX_LOGGED_UNCATEGORIZED,
    UNIVERSE_HEADER,
    UniverseParams,
    UniversePeriod,
    diversify_universe,
    filter_by_listing_age,
    filter_by_volume,
    first_candle_time,
    load_categories,
    load_universe,
    load_universe_params,
    median_daily_quote_volume,
    period_candidates,
    prefilter_by_category,
    select_period_universe,
    universe_periods,
    write_universe,
)
from miax.utils.config import CONFIGS_DIR, ConfigError

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


# --- MIAX-027: diversificacion del universo por categoria ---

CATEGORIES = {
    "A": "c1",
    "B": "c1",
    "C": "c1",
    "D": "c2",
    "E": "c3",
    "F": "c4",
    "USDC": "stablecoin",
    "WBTC": "pegged",
}


def _cat_pairs(assets: Sequence[Any]) -> pd.DataFrame:
    """Pares con volumen decreciente en el orden dado, como saldria de filter_by_volume."""
    return pd.DataFrame(
        {
            "symbol": [f"{a}USDT" for a in assets],
            "base_asset": list(assets),
            "median_quote_volume": [float(1000 - i) for i in range(len(assets))],
        }
    )


def _write_yaml(tmp_path: Path, text: str) -> Path:
    """Escribe un YAML de categorias en tmp_path y devuelve su ruta."""
    path = tmp_path / "cats.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def universe_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Engancha caplog al logger del modulo (tiene propagate=False) y lo desengancha al acabar."""
    universe.logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=universe.logger.name)
    yield caplog
    universe.logger.removeHandler(caplog.handler)


# load_categories


def test_real_categories_yaml_loads_with_expected_categories() -> None:
    """El YAML real carga y contiene las categorias y activos clave."""
    mapping = load_categories()
    assert set(mapping.values()) >= {"l1", "l2", "defi", "meme", "stablecoin", "pegged"}
    assert mapping["BTC"] == "l1"
    assert mapping["USDC"] == "stablecoin"
    assert mapping["WBTC"] == "pegged"
    assert mapping["TIA"] == "infra"
    assert mapping["RNDR"] == "ai"
    assert mapping["RENDER"] == "ai"


def test_real_categories_yaml_has_no_repeated_assets() -> None:
    """Ningun activo aparece dos veces en el YAML real, ni en la misma categoria ni entre ellas."""
    raw = yaml.safe_load((CONFIGS_DIR / "categories.yaml").read_text(encoding="utf-8"))
    flat = [str(asset).strip().upper() for assets in raw.values() for asset in assets]
    assert len(flat) == len(set(flat))
    assert len(load_categories()) == len(flat)


def test_load_categories_uppercases_and_strips_assets(tmp_path: Path) -> None:
    """Los activos se normalizan a mayusculas y sin espacios."""
    path = _write_yaml(tmp_path, "l1: [btc, ' eth ']\n")
    assert load_categories(path) == {"BTC": "l1", "ETH": "l1"}


def test_load_categories_keeps_order_of_appearance(tmp_path: Path) -> None:
    """El mapa conserva el orden de aparicion en el YAML."""
    path = _write_yaml(tmp_path, "b: [Z, A]\na: [M]\n")
    assert list(load_categories(path)) == ["Z", "A", "M"]


def test_load_categories_tolerates_repeat_in_same_category(tmp_path: Path) -> None:
    """Un activo repetido dentro de la misma categoria se deduplica sin error."""
    path = _write_yaml(tmp_path, "l1: [BTC, btc, ETH]\n")
    assert load_categories(path) == {"BTC": "l1", "ETH": "l1"}


def test_load_categories_asset_in_two_categories_raises(tmp_path: Path) -> None:
    """Un activo en dos categorias es un error que nombra el activo y ambas categorias."""
    path = _write_yaml(tmp_path, "l1: [BTC]\ndefi: [btc]\n")
    with pytest.raises(ConfigError, match="BTC.*l1.*defi"):
        load_categories(path)


@pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n", ""])
def test_load_categories_non_mapping_raises(tmp_path: Path, text: str) -> None:
    """Un YAML que no es un mapeo es un error."""
    with pytest.raises(ConfigError, match="mapeo"):
        load_categories(_write_yaml(tmp_path, text))


def test_load_categories_non_list_value_raises(tmp_path: Path) -> None:
    """Un valor que no es lista es un error."""
    with pytest.raises(ConfigError, match="lista"):
        load_categories(_write_yaml(tmp_path, "l1: BTC\n"))


def test_load_categories_non_str_category_raises(tmp_path: Path) -> None:
    """Una categoria que no es texto es un error."""
    with pytest.raises(ConfigError, match="texto"):
        load_categories(_write_yaml(tmp_path, "1: [BTC]\n"))


@pytest.mark.parametrize("assets", ["[1]", "[true]", "[null]", "['']", "['  ']", "[[a]]"])
def test_load_categories_invalid_asset_raises(tmp_path: Path, assets: str) -> None:
    """Un activo no-str o vacio es un error."""
    with pytest.raises(ConfigError, match="activo invalido"):
        load_categories(_write_yaml(tmp_path, f"l1: {assets}\n"))


def test_load_categories_missing_file_raises(tmp_path: Path) -> None:
    """Un fichero inexistente es un error."""
    with pytest.raises(ConfigError, match="No se encuentra"):
        load_categories(tmp_path / "nope.yaml")


def test_load_categories_relative_path_resolves_against_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una ruta relativa se resuelve contra CONFIGS_DIR."""
    _write_yaml(tmp_path, "l1: [BTC]\n")
    monkeypatch.setattr(universe, "CONFIGS_DIR", tmp_path)
    assert load_categories("cats.yaml") == {"BTC": "l1"}


def test_load_categories_absolute_path_is_used_as_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una ruta absoluta no se resuelve contra CONFIGS_DIR."""
    path = _write_yaml(tmp_path, "l1: [ETH]\n")
    monkeypatch.setattr(universe, "CONFIGS_DIR", tmp_path / "otro")
    assert load_categories(path) == {"ETH": "l1"}


# diversify_universe


def test_diversify_category_cap_keeps_first_in_input_order() -> None:
    """El tope por categoria conserva las primeras filas en orden de entrada, no por nombre."""
    pairs = _cat_pairs(["C", "A", "B", "D", "E", "F"])  # c1 = C, A, B
    out = diversify_universe(pairs, CATEGORIES, max_per_category=2)
    assert list(out["base_asset"]) == ["C", "A", "D", "E", "F"]


def test_diversify_total_cap_keeps_first_in_input_order() -> None:
    """El tope total conserva las primeras filas en orden de entrada."""
    pairs = _cat_pairs(["F", "E", "D", "A", "B", "C"])
    out = diversify_universe(pairs, CATEGORIES, max_pairs=3, min_categories=1)
    assert list(out["base_asset"]) == ["F", "E", "D"]


def test_diversify_total_cap_applies_after_category_cap() -> None:
    """Primero se aplica el tope por categoria y despues el total."""
    pairs = _cat_pairs(["A", "B", "C", "D", "E", "F"])
    out = diversify_universe(pairs, CATEGORIES, max_per_category=1, max_pairs=3, min_categories=1)
    assert list(out["base_asset"]) == ["A", "D", "E"]


def test_diversify_excludes_stablecoin_and_pegged_by_default() -> None:
    """Stablecoins y pegged se excluyen por defecto."""
    pairs = _cat_pairs(["USDC", "A", "WBTC", "D", "E", "F"])
    out = diversify_universe(pairs, CATEGORIES)
    assert list(out["base_asset"]) == ["A", "D", "E", "F"]


def test_diversify_empty_excluded_categories_lets_stablecoins_in() -> None:
    """Con excluded_categories vacio entran las stablecoins y los pegged."""
    pairs = _cat_pairs(["USDC", "A", "WBTC", "D", "E", "F"])
    out = diversify_universe(pairs, CATEGORIES, excluded_categories=())
    assert list(out["base_asset"]) == ["USDC", "A", "WBTC", "D", "E", "F"]
    assert set(out["category"]) == {"stablecoin", "pegged", "c1", "c2", "c3", "c4"}


def test_diversify_drops_uncategorized() -> None:
    """Un activo sin categoria se descarta."""
    pairs = _cat_pairs(["A", "XXX", "D", "E", "F"])
    out = diversify_universe(pairs, CATEGORIES)
    assert list(out["base_asset"]) == ["A", "D", "E", "F"]


def test_diversify_nan_base_asset_is_dropped_as_uncategorized() -> None:
    """Un base_asset NaN se descarta como sin categoria."""
    pairs = _cat_pairs(["A", "D", "E", "F", "X"])
    pairs["base_asset"] = pairs["base_asset"].astype(object)
    pairs.loc[4, "base_asset"] = float("nan")
    out = diversify_universe(pairs, CATEGORIES)
    assert list(out["base_asset"]) == ["A", "D", "E", "F"]


def test_diversify_adds_category_column_and_resets_index() -> None:
    """Anade la columna category, conserva las demas y deja el indice 0..n-1."""
    pairs = _cat_pairs(["XXX", "A", "D", "E", "F"])
    out = diversify_universe(pairs, CATEGORIES)
    assert list(out.index) == [0, 1, 2, 3]
    assert list(out.columns) == ["symbol", "base_asset", "median_quote_volume", "category"]
    assert list(out["category"]) == ["c1", "c2", "c3", "c4"]


def test_diversify_does_not_mutate_input() -> None:
    """La entrada no se modifica."""
    pairs = _cat_pairs(["A", "D", "E", "F"])
    before = pairs.copy()
    diversify_universe(pairs, CATEGORIES)
    pd.testing.assert_frame_equal(pairs, before)


def test_diversify_lowercase_base_asset_maps_the_same() -> None:
    """Un base_asset en minusculas casa con la categoria, y el valor original se conserva."""
    pairs = _cat_pairs(["a", "d", "e", "f"])
    out = diversify_universe(pairs, CATEGORIES)
    assert list(out["base_asset"]) == ["a", "d", "e", "f"]
    assert list(out["category"]) == ["c1", "c2", "c3", "c4"]


def test_diversify_normalizes_category_keys() -> None:
    """Las claves de categories se normalizan: minusculas y espacios casan igual."""
    categories = {" a ": "c1", "d": "c2", "E ": "c3", "f": "c4"}
    out = diversify_universe(_cat_pairs(["A", "D", "E", "F"]), categories)
    assert list(out["category"]) == ["c1", "c2", "c3", "c4"]


def test_diversify_too_few_categories_raises_only_l1_case() -> None:
    """Un universo solo L1 incumple el minimo de categorias y falla con un mensaje claro."""
    categories = {"A": "l1", "B": "l1", "C": "l1", "D": "l1"}
    with pytest.raises(ValueError, match=r"1 categorias.*l1.*minimo.*4"):
        diversify_universe(_cat_pairs(["A", "B", "C", "D"]), categories)


def test_diversify_min_categories_boundary() -> None:
    """Con exactamente min_categories categorias no falla; con una menos si."""
    pairs = _cat_pairs(["A", "D", "E"])
    assert len(diversify_universe(pairs, CATEGORIES, min_categories=3)) == 3
    with pytest.raises(ValueError, match="minimo"):
        diversify_universe(pairs, CATEGORIES, min_categories=4)


def test_diversify_empty_input_raises_for_min_categories() -> None:
    """Un catalogo vacio no llega al minimo de categorias."""
    with pytest.raises(ValueError, match="0 categorias"):
        diversify_universe(_cat_pairs([]), CATEGORIES)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"max_per_category": 0}, "max_per_category"),
        ({"max_pairs": 0}, "max_pairs"),
        ({"min_categories": 0}, "min_categories"),
    ],
)
def test_diversify_invalid_parameters_raise(kwargs: dict[str, int], name: str) -> None:
    """Los parametros menores que 1 son un error."""
    with pytest.raises(ValueError, match=name):
        diversify_universe(_cat_pairs(["A"]), CATEGORIES, **kwargs)


def test_diversify_without_base_asset_column_raises() -> None:
    """Sin columna base_asset no se puede clasificar."""
    with pytest.raises(ValueError, match="base_asset"):
        diversify_universe(pd.DataFrame({"symbol": ["AAA"]}), CATEGORIES)


def test_diversify_logs_at_most_max_uncategorized_and_the_remainder(
    universe_log: pytest.LogCaptureFixture,
) -> None:
    """El log lista como mucho MAX_LOGGED_UNCATEGORIZED sin categoria y anade 'y N mas'."""
    extra = [f"UNK{i:02d}" for i in range(MAX_LOGGED_UNCATEGORIZED + 5)]
    pairs = _cat_pairs(["A", "D", "E", "F", *extra])
    diversify_universe(pairs, CATEGORIES)
    message = universe_log.text
    assert "y 5 mas" in message
    assert f"UNK{MAX_LOGGED_UNCATEGORIZED - 1:02d}" in message
    assert f"UNK{MAX_LOGGED_UNCATEGORIZED:02d}" not in message


def test_diversify_log_without_overflow_has_no_remainder(
    universe_log: pytest.LogCaptureFixture,
) -> None:
    """Con pocos sin categoria se listan todos y no aparece 'y N mas'."""
    pairs = _cat_pairs(["A", "D", "E", "F", "UNK1", "UNK2"])
    diversify_universe(pairs, CATEGORIES)
    assert "UNK1" in universe_log.text
    assert "UNK2" in universe_log.text
    assert " mas" not in universe_log.text


def test_diversify_is_deterministic() -> None:
    """La misma entrada da siempre la misma salida."""
    pairs = _cat_pairs(["C", "A", "B", "D", "E", "F", "XXX"])
    first = diversify_universe(pairs, CATEGORIES, max_per_category=2)
    second = diversify_universe(pairs, CATEGORIES, max_per_category=2)
    pd.testing.assert_frame_equal(first, second)


def test_volume_filter_then_diversify_keeps_volume_order() -> None:
    """Integracion pura: tras filter_by_volume, diversificar conserva el orden de volumen."""
    assets = ["A", "B", "D", "E", "F", "USDC"]
    pairs = pd.DataFrame({"symbol": [f"{a}USDT" for a in assets], "base_asset": assets})
    volumes = pd.Series(
        {
            "AUSDT": 10.0,
            "BUSDT": 90.0,
            "DUSDT": 50.0,
            "EUSDT": 70.0,
            "FUSDT": 30.0,
            "USDCUSDT": 99.0,
        }
    )
    by_volume = filter_by_volume(pairs, volumes, min_quote_volume=0.0)
    out = diversify_universe(by_volume, CATEGORIES)
    assert list(out["base_asset"]) == ["B", "E", "D", "F", "A"]
    assert list(out["median_quote_volume"]) == [90.0, 70.0, 50.0, 30.0, 10.0]


# --- MIAX-028: universo por periodos congelado en un fichero versionado ---

PARAMS = UniverseParams(
    quote_asset="USDT",
    first_period_start="2018-01-01",
    last_period_end="2026-10-01",
    period_months=12,
    volume_window_months=3,
    min_listing_days=30,
    min_quote_volume=10_000_000.0,
    max_per_category=10,
    max_pairs=50,
    min_categories=4,
    categories_file="categories.yaml",
)
WINDOW_START = DAY0 + 200 * DAY_MS
PERIOD = UniversePeriod(
    window_start_ms=WINDOW_START,
    window_end_ms=WINDOW_START + 90 * DAY_MS - 1,
    period_start_ms=WINDOW_START + 90 * DAY_MS,
    period_end_ms=WINDOW_START + 455 * DAY_MS,
)
AGE_CUTOFF = WINDOW_START - PARAMS.min_listing_days * DAY_MS
UNIVERSE_COLUMNS = [
    "period_start",
    "period_end",
    "symbol",
    "category",
    "median_quote_volume",
    "first_candle",
]


def _ms(text: str) -> int:
    """Milisegundos UTC de una fecha ISO."""
    return int((pd.Timestamp(text, tz="UTC") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms"))


def _params_file(tmp_path: Path, **overrides: object) -> Path:
    """Escribe un universe_params.yaml en tmp_path con los valores base y los cambios dados."""
    values = {**dataclasses.asdict(PARAMS), **overrides}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _selection_inputs(
    ages: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Pares A..F + USDC, primeras velas (todas muy antiguas salvo `ages`) y volumen decreciente."""
    assets = ["A", "B", "C", "D", "E", "F", "USDC"]
    pairs = pd.DataFrame({"symbol": [f"{a}USDT" for a in assets], "base_asset": assets})
    first = pd.Series({f"{a}USDT": DAY0 for a in assets}, dtype="Int64")
    for symbol, value in (ages or {}).items():
        first[symbol] = value
    volumes = pd.Series({f"{a}USDT": 1e9 - i * 1e6 for i, a in enumerate(assets)})
    return pairs, first, volumes


def _universe_frame(rows: Sequence[tuple[str, str, float, int]]) -> pd.DataFrame:
    """DataFrame de un universo con las columnas que escribe write_universe."""
    return pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "category": [r[1] for r in rows],
            "median_quote_volume": [r[2] for r in rows],
            "first_candle_ms": pd.array([r[3] for r in rows], dtype="Int64"),
        }
    )


# load_universe_params


def test_real_universe_params_load() -> None:
    """El universe_params.yaml real carga con los valores acordados (trimestral, 5 M y 40 pares)."""
    expected = dataclasses.replace(
        PARAMS, period_months=3, min_quote_volume=5_000_000.0, max_pairs=40
    )
    assert load_universe_params() == expected


def test_load_universe_params_absolute_path(tmp_path: Path) -> None:
    """Una ruta absoluta se lee tal cual y los tipos son los del esquema."""
    params = load_universe_params(_params_file(tmp_path, max_pairs=7, min_quote_volume=5))
    assert params.max_pairs == 7
    assert params.min_quote_volume == 5
    assert isinstance(params, UniverseParams)


def test_load_universe_params_missing_file_raises(tmp_path: Path) -> None:
    """Un fichero inexistente es ConfigError."""
    with pytest.raises(ConfigError):
        load_universe_params(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_period_start", "2018/01/01"),
        ("first_period_start", "2018-1-1"),
        ("first_period_start", "20180101"),
        ("first_period_start", "2018-13-01"),
        ("first_period_start", "2018-02-30"),
        ("last_period_end", "01-10-2026"),
        ("last_period_end", "2026-10-01T00:00"),
        ("first_period_start", "2026-10-01"),  # igual a last_period_end
        ("first_period_start", "2027-01-01"),  # posterior a last_period_end
        ("period_months", 0),
        ("period_months", -3),
        ("volume_window_months", 0),
        ("min_listing_days", -1),
        ("min_quote_volume", -0.01),
        ("max_per_category", 0),
        ("max_pairs", 0),
        ("min_categories", 0),
    ],
)
def test_load_universe_params_invalid_value_raises(
    tmp_path: Path, field: str, value: object
) -> None:
    """Cada validacion de load_universe_params lanza ConfigError."""
    with pytest.raises(ConfigError):
        load_universe_params(_params_file(tmp_path, **{field: value}))


@pytest.mark.parametrize("literal", [".nan", ".inf", "-.inf"])
def test_load_universe_params_non_finite_min_quote_volume_raises(
    tmp_path: Path, literal: str
) -> None:
    """min_quote_volume nan o inf se rechaza (nan esquivaria la comparacion con el minimo)."""
    path = _params_file(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "min_quote_volume: 10000000.0", f"min_quote_volume: {literal}"
    )
    assert literal in text
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="min_quote_volume"):
        load_universe_params(path)


def test_load_universe_params_boundary_values_are_accepted(tmp_path: Path) -> None:
    """Los minimos exactos (0 dias, volumen 0, 1 mes...) son validos."""
    path = _params_file(
        tmp_path,
        period_months=1,
        volume_window_months=1,
        min_listing_days=0,
        min_quote_volume=0.0,
        max_per_category=1,
        max_pairs=1,
        min_categories=1,
        first_period_start="2018-01-01",
        last_period_end="2018-01-02",
    )
    assert load_universe_params(path).min_listing_days == 0


def test_load_universe_params_wrong_type_or_unknown_field_raises(tmp_path: Path) -> None:
    """Un tipo equivocado o un campo desconocido son ConfigError (los valida load_config)."""
    with pytest.raises(ConfigError):
        load_universe_params(_params_file(tmp_path, period_months="doce"))
    with pytest.raises(ConfigError):
        load_universe_params(_params_file(tmp_path, extra_field=1))


# universe_periods


def test_universe_periods_real_params_give_nine_periods() -> None:
    """Con los parametros reales hay 9 periodos (2018..2026) y el ultimo acaba el 2026-10-01."""
    periods = universe_periods(PARAMS)
    assert len(periods) == 9
    assert [p.period_start_ms for p in periods] == [_ms(f"{y}-01-01") for y in range(2018, 2027)]
    assert periods[-1].period_end_ms == _ms("2026-10-01")  # recortado
    assert periods[0].period_end_ms == _ms("2019-01-01")


def test_universe_periods_window_ends_one_ms_before_period_and_starts_three_months_before() -> None:
    """ANTIFUGA: cada ventana acaba 1 ms antes de su periodo y empieza 3 meses antes."""
    periods = universe_periods(PARAMS)
    for period in periods:
        assert period.window_end_ms == period.period_start_ms - 1
        start = pd.Timestamp(period.period_start_ms, unit="ms", tz="UTC")
        expected = start - pd.DateOffset(months=3)
        assert period.window_start_ms == _ms(expected.strftime("%Y-%m-%d"))
    assert periods[0].window_start_ms == _ms("2017-10-01")
    assert periods[0].window_end_ms == _ms("2018-01-01") - 1


def test_universe_periods_are_contiguous_and_ordered() -> None:
    """El fin de cada periodo es el inicio del siguiente y el orden es creciente."""
    periods = universe_periods(PARAMS)
    for previous, current in pairwise(periods):
        assert previous.period_end_ms == current.period_start_ms
        assert previous.period_start_ms < current.period_start_ms
    assert all(p.period_start_ms < p.period_end_ms for p in periods)


def test_universe_periods_single_period_when_months_cover_the_range() -> None:
    """Un period_months que cubre todo el rango da un unico periodo recortado (universo fijo)."""
    periods = universe_periods(dataclasses.replace(PARAMS, period_months=120))
    assert len(periods) == 1
    assert periods[0].period_start_ms == _ms("2018-01-01")
    assert periods[0].period_end_ms == _ms("2026-10-01")


def test_universe_periods_last_period_is_trimmed() -> None:
    """El ultimo periodo se recorta a last_period_end y no lo supera."""
    params = dataclasses.replace(PARAMS, last_period_end="2019-04-01")
    periods = universe_periods(params)
    assert [p.period_end_ms for p in periods] == [_ms("2019-01-01"), _ms("2019-04-01")]


def test_universe_periods_exact_multiple_has_no_empty_extra_period() -> None:
    """Si el rango es multiplo exacto, no aparece un periodo vacio al final."""
    periods = universe_periods(dataclasses.replace(PARAMS, last_period_end="2020-01-01"))
    assert len(periods) == 2
    assert periods[-1].period_end_ms == _ms("2020-01-01")


def test_universe_periods_month_end_start_stays_contiguous() -> None:
    """Un inicio en dia 31 con pasos mensuales no deja huecos entre periodos."""
    params = dataclasses.replace(
        PARAMS, first_period_start="2018-01-31", last_period_end="2018-06-30", period_months=1
    )
    periods = universe_periods(params)
    for previous, current in pairwise(periods):
        assert previous.period_end_ms == current.period_start_ms


def test_universe_periods_are_deterministic() -> None:
    """La misma entrada da la misma salida."""
    assert universe_periods(PARAMS) == universe_periods(PARAMS)


@pytest.mark.parametrize("field", ["period_months", "volume_window_months"])
@pytest.mark.parametrize("value", [0, -1])
def test_universe_periods_invalid_months_raise_value_error(field: str, value: int) -> None:
    """Con parametros construidos a mano, meses < 1 lanzan ValueError."""
    with pytest.raises(ValueError, match=field):
        universe_periods(dataclasses.replace(PARAMS, **{field: value}))


# period_candidates


def test_period_candidates_cutoff_is_inclusive_and_one_ms_later_is_out() -> None:
    """ANTIFUGA: primera vela == corte entra; un ms despues queda fuera; un ms antes entra."""
    first = pd.Series(
        {"ON": AGE_CUTOFF, "AFTER": AGE_CUTOFF + 1, "BEFORE": AGE_CUTOFF - 1}, dtype="Int64"
    )
    assert period_candidates(first, PERIOD, PARAMS) == ["ON", "BEFORE"]


def test_period_candidates_drops_na_and_keeps_order() -> None:
    """Los NA se descartan y el orden de entrada se conserva."""
    first = pd.Series({"Z": DAY0, "M": pd.NA, "A": DAY0, "K": DAY0}, dtype="Int64")
    assert period_candidates(first, PERIOD, PARAMS) == ["Z", "A", "K"]


def test_period_candidates_empty_series() -> None:
    """Una serie vacia da lista vacia."""
    assert period_candidates(pd.Series({}, dtype="Int64"), PERIOD, PARAMS) == []


def test_period_candidates_uses_window_start_not_period_start() -> None:
    """La referencia es el inicio de la ventana: listado dentro de la ventana no es candidato."""
    first = pd.Series({"IN_WINDOW": PERIOD.window_start_ms + DAY_MS}, dtype="Int64")
    assert period_candidates(first, PERIOD, PARAMS) == []


def test_period_candidates_matches_the_age_filter_of_select_period_universe() -> None:
    """Los candidatos coinciden con los simbolos que pasa el filtro de antiguedad del periodo."""
    pairs, first, _ = _selection_inputs({"AUSDT": AGE_CUTOFF + 1, "BUSDT": AGE_CUTOFF})
    aged = filter_by_listing_age(pairs, first, PERIOD.window_start_ms, PARAMS.min_listing_days)
    assert period_candidates(first, PERIOD, PARAMS) == list(aged["symbol"])


# prefilter_by_category


def test_prefilter_excludes_stablecoin_pegged_and_uncategorized() -> None:
    """Quedan fuera stablecoins, pegged y activos sin categoria; el resto se conserva."""
    out = prefilter_by_category(_cat_pairs(["A", "USDC", "WBTC", "ZZZ", "D"]), CATEGORIES)
    assert list(out["base_asset"]) == ["A", "D"]


def test_prefilter_normalizes_case_keeps_order_and_resets_index() -> None:
    """Normaliza mayusculas (base y claves), conserva el orden y deja indice 0..n-1."""
    pairs = pd.DataFrame(
        {"symbol": ["dUSDT", "aUSDT", "eUSDT", "zUSDT"], "base_asset": ["d", "a", "E", "z"]}
    )
    out = prefilter_by_category(pairs, {" a ": "c1", "d": "c2", "E": "c3"})
    assert list(out["symbol"]) == ["dUSDT", "aUSDT", "eUSDT"]
    assert list(out.index) == [0, 1, 2]
    assert list(out.columns) == ["symbol", "base_asset"]


def test_prefilter_custom_excluded_categories() -> None:
    """Con excluded_categories vacio, las stablecoins pasan."""
    out = prefilter_by_category(_cat_pairs(["A", "USDC"]), CATEGORIES, excluded_categories=())
    assert list(out["base_asset"]) == ["A", "USDC"]


def test_prefilter_does_not_mutate_input_and_empty_ok() -> None:
    """No muta la entrada y un DataFrame vacio da vacio."""
    pairs = _cat_pairs(["A", "USDC"])
    before = pairs.copy()
    prefilter_by_category(pairs, CATEGORIES)
    pd.testing.assert_frame_equal(pairs, before)
    assert prefilter_by_category(_cat_pairs([]), CATEGORIES).empty


def test_prefilter_without_base_asset_raises() -> None:
    """Sin columna base_asset es ValueError."""
    with pytest.raises(ValueError, match="base_asset"):
        prefilter_by_category(pd.DataFrame({"symbol": ["AUSDT"]}), CATEGORIES)


# select_period_universe


def test_select_applies_listing_age_with_window_start_reference_at_the_edge() -> None:
    """ANTIFUGA: antiguedad medida desde el inicio de la ventana; limite inclusivo a +-1 ms."""
    pairs, first, volumes = _selection_inputs(
        {"AUSDT": AGE_CUTOFF, "BUSDT": AGE_CUTOFF + 1, "CUSDT": AGE_CUTOFF - 1}
    )
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    symbols = set(out["symbol"])
    assert {"AUSDT", "CUSDT"} <= symbols
    assert "BUSDT" not in symbols


def test_select_token_listed_inside_volume_window_is_dropped_despite_huge_volume() -> None:
    """Un token lanzado dentro de la ventana no entra aunque tenga muchisimo volumen."""
    pairs, first, volumes = _selection_inputs({"AUSDT": PERIOD.window_start_ms + 10 * DAY_MS})
    volumes["AUSDT"] = 1e15
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert "AUSDT" not in set(out["symbol"])


def test_select_first_candle_after_the_period_is_dropped() -> None:
    """Una primera vela posterior al periodo (consulta 'de hoy') tampoco entra."""
    pairs, first, volumes = _selection_inputs({"AUSDT": PERIOD.period_end_ms + 5 * DAY_MS})
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert "AUSDT" not in set(out["symbol"])


def test_select_applies_volume_threshold_and_orders_by_volume() -> None:
    """Descarta bajo el umbral (>= entra), ordena por volumen y excluye stablecoins."""
    pairs, first, _ = _selection_inputs()
    volumes = pd.Series(
        {
            "AUSDT": 20e6,
            "BUSDT": PARAMS.min_quote_volume,
            "CUSDT": PARAMS.min_quote_volume - 1,
            "DUSDT": 90e6,
            "EUSDT": 50e6,
            "FUSDT": float("nan"),
            "USDCUSDT": 99e9,
        }
    )
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert list(out["symbol"]) == ["DUSDT", "EUSDT", "AUSDT", "BUSDT"]
    assert list(out.index) == [0, 1, 2, 3]
    for column in ("symbol", "base_asset", "category", "median_quote_volume", "first_candle_ms"):
        assert column in out.columns


def test_select_applies_category_and_total_caps() -> None:
    """Respeta max_per_category y max_pairs conservando el orden de volumen."""
    pairs, first, volumes = _selection_inputs()
    capped = dataclasses.replace(PARAMS, max_per_category=1)
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, capped)
    assert list(out["symbol"]) == ["AUSDT", "DUSDT", "EUSDT", "FUSDT"]
    total = dataclasses.replace(PARAMS, max_pairs=2)
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, total)
    assert list(out["symbol"]) == ["AUSDT", "BUSDT"]


def test_select_warns_but_does_not_raise_below_min_categories(
    universe_log: pytest.LogCaptureFixture,
) -> None:
    """Con menos categorias que min_categories avisa (con el periodo) y devuelve igualmente."""
    assets = ["A", "B", "C"]
    pairs = pd.DataFrame({"symbol": [f"{a}USDT" for a in assets], "base_asset": assets})
    first = pd.Series({f"{a}USDT": DAY0 for a in assets}, dtype="Int64")
    volumes = pd.Series({f"{a}USDT": 5e7 for a in assets})
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert len(out) == 3
    warnings = [r for r in universe_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "c1" in message
    label = pd.Timestamp(PERIOD.period_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    assert label in message


def test_select_does_not_warn_when_min_categories_is_met(
    universe_log: pytest.LogCaptureFixture,
) -> None:
    """Con categorias suficientes no hay avisos."""
    pairs, first, volumes = _selection_inputs()
    select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert [r for r in universe_log.records if r.levelno == logging.WARNING] == []


def test_select_returns_empty_when_nothing_survives(
    universe_log: pytest.LogCaptureFixture,
) -> None:
    """Si ningun par supera los filtros devuelve vacio con columnas, avisa y no lanza."""
    pairs, first, _ = _selection_inputs()
    volumes = pd.Series({f"{a}USDT": 1.0 for a in "ABCDEF"})
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert out.empty
    assert "symbol" in out.columns
    assert "category" in out.columns
    assert any(r.levelno == logging.WARNING for r in universe_log.records)


def test_select_returns_empty_when_everything_is_uncategorized_or_excluded(
    universe_log: pytest.LogCaptureFixture,
) -> None:
    """Todo sin categoria o stablecoin da vacio con aviso, sin lanzar."""
    pairs = pd.DataFrame({"symbol": ["ZZZUSDT", "USDCUSDT"], "base_asset": ["ZZZ", "USDC"]})
    first = pd.Series({"ZZZUSDT": DAY0, "USDCUSDT": DAY0}, dtype="Int64")
    volumes = pd.Series({"ZZZUSDT": 5e8, "USDCUSDT": 5e8})
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    assert out.empty
    assert "category" in out.columns
    assert any(r.levelno == logging.WARNING for r in universe_log.records)


def test_select_empty_pairs_returns_empty_without_raising() -> None:
    """Un catalogo vacio da universo vacio."""
    pairs = pd.DataFrame({"symbol": [], "base_asset": []})
    first = pd.Series({}, dtype="Int64")
    volumes = pd.Series({}, dtype="float64")
    assert select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS).empty


def test_select_does_not_mutate_inputs_and_is_deterministic() -> None:
    """No muta las entradas y repetir da el mismo resultado."""
    pairs, first, volumes = _selection_inputs()
    before = (pairs.copy(), first.copy(), volumes.copy())
    one = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    two = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    pd.testing.assert_frame_equal(one, two)
    pd.testing.assert_frame_equal(pairs, before[0])
    pd.testing.assert_series_equal(first, before[1])
    pd.testing.assert_series_equal(volumes, before[2])


def test_leak_no_selected_pair_is_listed_after_age_cutoff_in_any_real_period() -> None:
    """FUGA: en todos los periodos reales, ningun par elegido cotiza tras el corte de antiguedad."""
    assets = ["A", "B", "C", "D", "E", "F"]
    pairs = pd.DataFrame({"symbol": [f"{a}USDT" for a in assets], "base_asset": assets})
    # Primeras velas repartidas entre 2016 y 2026, incluidas las posteriores a cada periodo.
    first = pd.Series(
        {f"{a}USDT": _ms(f"{2016 + 2 * i}-06-15") for i, a in enumerate(assets)}, dtype="Int64"
    )
    volumes = pd.Series({f"{a}USDT": 5e7 for a in assets})
    for period in universe_periods(PARAMS):
        cutoff = period.window_start_ms - PARAMS.min_listing_days * DAY_MS
        out = select_period_universe(pairs, first, volumes, CATEGORIES, period, PARAMS)
        assert (out["first_candle_ms"] <= cutoff).all()
        assert (out["first_candle_ms"] < period.period_start_ms).all()
        assert all(first[s] <= cutoff for s in period_candidates(first, period, PARAMS))


# write_universe / load_universe


def _two_periods() -> list[tuple[UniversePeriod, pd.DataFrame]]:
    """Dos periodos reales: el primero con tres pares y el segundo con uno."""
    periods = universe_periods(PARAMS)
    first = _universe_frame(
        [
            ("BTCUSDT", "l1", 123456789.123456, _ms("2017-08-17")),
            ("ETHUSDT", "l1", 98765432.1, _ms("2017-08-17")),
            ("UNIUSDT", "defi", 55555555.557, _ms("2020-09-17")),
        ]
    )
    second = _universe_frame([("SOLUSDT", "l1", 77000000.0, _ms("2020-08-11"))])
    return [(periods[0], first), (periods[1], second)]


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lee el YAML escrito por write_universe."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_write_returns_path_and_has_header_and_structure(tmp_path: Path) -> None:
    """Escribe la cabecera, la estructura del plan y devuelve la ruta."""
    target = tmp_path / "u.yaml"
    written = write_universe(_two_periods(), PARAMS, target, generated_at="2026-10-12T18:00:00Z")
    assert written == target
    assert target.read_text(encoding="utf-8").startswith(UNIVERSE_HEADER)
    doc = _read_yaml(target)
    assert list(doc) == ["generated_at", "params", "periods"]
    assert doc["generated_at"] == "2026-10-12T18:00:00Z"
    assert doc["params"] == dataclasses.asdict(PARAMS)
    first = doc["periods"][0]
    assert first["period_start"] == "2018-01-01"
    assert first["period_end"] == "2019-01-01"
    assert first["volume_window"] == ["2017-10-01", "2017-12-31"]
    assert first["n_pairs"] == 3
    assert first["categories"] == {"l1": 2, "defi": 1}
    assert first["pairs"][0] == {
        "symbol": "BTCUSDT",
        "category": "l1",
        "median_quote_volume": 123456789.12,
        "first_candle": "2017-08-17",
    }
    assert doc["periods"][1]["n_pairs"] == 1


def test_write_default_generated_at_is_utc_timestamp(tmp_path: Path) -> None:
    """Sin generated_at se usa la hora actual en UTC con formato ISO terminado en Z."""
    write_universe(_two_periods(), PARAMS, tmp_path / "u.yaml")
    stamp = _read_yaml(tmp_path / "u.yaml")["generated_at"]
    assert stamp.endswith("Z")
    assert pd.Timestamp(stamp).tzinfo is not None


def test_write_yaml_uses_native_python_types(tmp_path: Path) -> None:
    """El YAML tiene tipos nativos (float, int, str), sin objetos numpy."""
    write_universe(_two_periods(), PARAMS, tmp_path / "u.yaml")
    text = (tmp_path / "u.yaml").read_text(encoding="utf-8")
    assert "!!python" not in text
    assert "numpy" not in text
    doc = yaml.safe_load(text)
    pair = doc["periods"][0]["pairs"][0]
    assert type(pair["median_quote_volume"]) is float
    assert type(pair["symbol"]) is str
    assert type(pair["category"]) is str
    assert type(pair["first_candle"]) is str
    assert type(doc["periods"][0]["n_pairs"]) is int
    assert type(doc["params"]["period_months"]) is int
    assert type(doc["params"]["min_quote_volume"]) is float


def test_write_rounds_volume_to_two_decimals(tmp_path: Path) -> None:
    """median_quote_volume se redondea a 2 decimales."""
    write_universe(_two_periods(), PARAMS, tmp_path / "u.yaml")
    pairs = _read_yaml(tmp_path / "u.yaml")["periods"][0]["pairs"]
    assert [p["median_quote_volume"] for p in pairs] == [123456789.12, 98765432.1, 55555555.56]


def test_write_first_candle_is_utc_date_of_the_instant(tmp_path: Path) -> None:
    """La primera vela se escribe como fecha UTC (el ultimo ms del dia sigue siendo ese dia)."""
    frame = _universe_frame([("AUSDT", "l1", 1e7, _ms("2017-08-17") + DAY_MS - 1)])
    write_universe([(universe_periods(PARAMS)[0], frame)], PARAMS, tmp_path / "u.yaml")
    doc = _read_yaml(tmp_path / "u.yaml")
    assert doc["periods"][0]["pairs"][0]["first_candle"] == "2017-08-17"


def test_write_is_atomic_leaves_no_tmp_file(tmp_path: Path) -> None:
    """Tras escribir con exito no queda ningun .tmp."""
    write_universe(_two_periods(), PARAMS, tmp_path / "u.yaml")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["u.yaml"]


def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    """Escribir sobre un fichero existente lo reemplaza."""
    target = tmp_path / "u.yaml"
    target.write_text("viejo\n", encoding="utf-8")
    write_universe(_two_periods(), PARAMS, target)
    assert "viejo" not in target.read_text(encoding="utf-8")


def test_write_failed_replace_leaves_no_tmp_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si os.replace falla no queda .tmp, el fichero previo sigue intacto y el error se propaga."""
    target = tmp_path / "u.yaml"
    target.write_text("previo\n", encoding="utf-8")

    def boom(src: object, dst: object) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr(universe.os, "replace", boom)
    with pytest.raises(OSError, match="disco lleno"):
        write_universe(_two_periods(), PARAMS, target)
    assert target.read_text(encoding="utf-8") == "previo\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["u.yaml"]


def test_write_relative_path_resolves_against_configs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una ruta relativa se escribe bajo CONFIGS_DIR (parcheado a tmp_path)."""
    monkeypatch.setattr(universe, "CONFIGS_DIR", tmp_path / "configs")
    written = write_universe(_two_periods(), PARAMS, "u.yaml")
    assert written == tmp_path / "configs" / "u.yaml"
    assert written.is_file()
    assert len(load_universe("u.yaml")) == 4


def test_write_empty_period_and_no_periods(tmp_path: Path) -> None:
    """Un periodo vacio se escribe con n_pairs 0 y pairs []; sin periodos tambien es valido."""
    period = universe_periods(PARAMS)[0]
    write_universe([(period, _universe_frame([]))], PARAMS, tmp_path / "a.yaml")
    doc = _read_yaml(tmp_path / "a.yaml")
    assert doc["periods"][0]["n_pairs"] == 0
    assert doc["periods"][0]["pairs"] == []
    assert doc["periods"][0]["categories"] == {}
    write_universe([], PARAMS, tmp_path / "b.yaml")
    assert _read_yaml(tmp_path / "b.yaml")["periods"] == []


def test_roundtrip_write_then_load_is_lossless(tmp_path: Path) -> None:
    """write_universe -> load_universe conserva filas, orden, valores y tipos."""
    write_universe(_two_periods(), PARAMS, tmp_path / "u.yaml")
    loaded = load_universe(tmp_path / "u.yaml")
    assert list(loaded.columns) == UNIVERSE_COLUMNS
    assert list(loaded["symbol"]) == ["BTCUSDT", "ETHUSDT", "UNIUSDT", "SOLUSDT"]
    assert list(loaded["category"]) == ["l1", "l1", "defi", "l1"]
    assert list(loaded["median_quote_volume"]) == [123456789.12, 98765432.1, 55555555.56, 77e6]
    assert list(loaded.index) == [0, 1, 2, 3]
    for column in ("period_start", "period_end", "first_candle"):
        assert str(loaded[column].dtype) == "datetime64[us, UTC]"
    assert loaded["period_start"].iloc[0] == pd.Timestamp("2018-01-01", tz="UTC")
    assert loaded["period_end"].iloc[0] == pd.Timestamp("2019-01-01", tz="UTC")
    assert loaded["period_start"].iloc[3] == pd.Timestamp("2019-01-01", tz="UTC")
    assert loaded["first_candle"].iloc[2] == pd.Timestamp("2020-09-17", tz="UTC")
    assert loaded["median_quote_volume"].dtype == "float64"
    assert pd.api.types.is_string_dtype(loaded["symbol"])
    assert pd.api.types.is_string_dtype(loaded["category"])


def test_load_universe_period_with_empty_pairs_adds_no_rows(tmp_path: Path) -> None:
    """Un periodo con `pairs: []` no aporta filas pero el resto si."""
    periods = universe_periods(PARAMS)
    full = _universe_frame([("BTCUSDT", "l1", 1e8, _ms("2017-08-17"))])
    write_universe(
        [(periods[0], full), (periods[1], _universe_frame([])), (periods[2], full)],
        PARAMS,
        tmp_path / "u.yaml",
    )
    loaded = load_universe(tmp_path / "u.yaml")
    assert list(loaded["period_start"]) == [
        pd.Timestamp("2018-01-01", tz="UTC"),
        pd.Timestamp("2020-01-01", tz="UTC"),
    ]


EMPTY_PERIOD_YAML = (
    "periods:\n  - period_start: '2018-01-01'\n    period_end: '2019-01-01'\n    pairs: []\n"
)


@pytest.mark.parametrize("text", ["periods: []\n", EMPTY_PERIOD_YAML])
def test_load_universe_without_rows_keeps_columns_and_utc_dtypes(tmp_path: Path, text: str) -> None:
    """Sin ninguna fila, las columnas existen y las fechas son datetime64[us, UTC]."""
    path = tmp_path / "u.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = load_universe(path)
    assert loaded.empty
    assert list(loaded.columns) == UNIVERSE_COLUMNS
    for column in ("period_start", "period_end", "first_candle"):
        assert str(loaded[column].dtype) == "datetime64[us, UTC]"
    assert loaded["median_quote_volume"].dtype == "float64"


def test_load_universe_missing_file_raises(tmp_path: Path) -> None:
    """Un fichero inexistente es ConfigError."""
    with pytest.raises(ConfigError, match="No se encuentra"):
        load_universe(tmp_path / "nope.yaml")


def test_load_universe_relative_path_resolves_against_configs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una ruta relativa se busca bajo CONFIGS_DIR; una inexistente alli es ConfigError."""
    monkeypatch.setattr(universe, "CONFIGS_DIR", tmp_path)
    write_universe(_two_periods(), PARAMS, tmp_path / "u.yaml")
    assert len(load_universe("u.yaml")) == 4
    with pytest.raises(ConfigError):
        load_universe("otro.yaml")


def _pair_yaml(**overrides: str) -> str:
    """YAML de un periodo con un par, con los campos del par sobrescribibles."""
    pair = {
        "symbol": "A",
        "category": "c",
        "median_quote_volume": "1.0",
        "first_candle": "'2017-01-01'",
        **overrides,
    }
    body = ", ".join(f"{key}: {value}" for key, value in pair.items())
    head = "period_start: '2018-01-01', period_end: '2019-01-01'"
    return f"periods:\n  - {{{head}, pairs: [{{{body}}}]}}\n"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "- a\n- b\n",
        "just text\n",
        "key: [unclosed\n",
        "periods: 3\n",
        "periods:\n  - period_start: '2018-01-01'\n",
        "periods:\n  - 7\n",
        "periods:\n  - {period_start: '2018-01-01', period_end: '2019-01-01', pairs: 5}\n",
        "periods:\n  - {period_start: '2018-01-01', period_end: '2019-01-01', pairs: [{symbol: A}]}\n",  # noqa: E501
        _pair_yaml(median_quote_volume="abc"),
        _pair_yaml(first_candle="'ayer'"),
        _pair_yaml().replace("'2018-01-01'", "'ayer'", 1),
    ],
)
def test_load_universe_malformed_raises_config_error(tmp_path: Path, text: str) -> None:
    """Un YAML mal formado o con campos invalidos es ConfigError."""
    path = tmp_path / "u.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_universe(path)


def test_roundtrip_of_select_period_universe_output(tmp_path: Path) -> None:
    """La salida real de select_period_universe se puede escribir y volver a leer."""
    pairs, first, volumes = _selection_inputs()
    out = select_period_universe(pairs, first, volumes, CATEGORIES, PERIOD, PARAMS)
    write_universe([(PERIOD, out)], PARAMS, tmp_path / "u.yaml")
    loaded = load_universe(tmp_path / "u.yaml")
    assert list(loaded["symbol"]) == list(out["symbol"])
    assert list(loaded["category"]) == list(out["category"])


# build_universe (orquestacion con la red parcheada)


def test_build_universe_orchestrates_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_universe encadena catalogo, prefiltro, primeras velas y ventanas por periodo."""
    categories = {"A": "l1", "B": "l1", "D": "l2", "USDC": "stablecoin"}
    params = dataclasses.replace(
        PARAMS,
        first_period_start="2020-01-01",
        last_period_end="2021-07-01",
        min_categories=1,
    )
    client = object()
    now_ms = _ms("2026-10-13")
    base_url = "https://example.invalid"
    calls: dict[str, list[Any]] = {"pairs": [], "first": [], "volume": []}

    def fake_list_pairs(**kwargs: object) -> pd.DataFrame:
        calls["pairs"].append(kwargs)
        assets = ["A", "B", "D", "USDC", "ZZZ"]
        return pd.DataFrame({"symbol": [f"{a}USDT" for a in assets], "base_asset": assets})

    def fake_first(symbols: Sequence[str], reference_ms: int, **kwargs: object) -> pd.Series:
        calls["first"].append((list(symbols), reference_ms, kwargs))
        dates = {"AUSDT": "2019-01-01", "BUSDT": "2020-12-01", "DUSDT": "2018-05-01"}
        return pd.Series({s: _ms(dates[s]) for s in symbols}, dtype="Int64")

    def fake_volume(symbols: Sequence[str], start: int, end: int, **kwargs: object) -> pd.Series:
        calls["volume"].append((list(symbols), start, end, kwargs))
        return pd.Series({s: 5e7 for s in symbols}, dtype="float64")

    monkeypatch.setattr(universe, "list_pairs", fake_list_pairs)
    monkeypatch.setattr(universe, "first_candle_time", fake_first)
    monkeypatch.setattr(universe, "median_daily_quote_volume", fake_volume)

    result = universe.build_universe(
        params, categories, client=client, now_ms=now_ms, base_url=base_url
    )

    assert len(calls["pairs"]) == 1
    assert calls["pairs"][0] == {"quote_asset": "USDT", "client": client, "base_url": base_url}
    # Prefiltro: sin USDC (stablecoin) ni ZZZ (sin categoria); primeras velas una sola vez.
    assert len(calls["first"]) == 1
    symbols, reference_ms, kwargs = calls["first"][0]
    assert symbols == ["AUSDT", "BUSDT", "DUSDT"]
    assert reference_ms == now_ms
    assert kwargs == {"client": client, "base_url": base_url}
    # BUSDT cotiza despues del corte de antiguedad de ambos periodos.
    assert len(calls["volume"]) == 2
    (symbols1, start1, end1, kwargs1), (symbols2, start2, end2, kwargs2) = calls["volume"]
    assert symbols1 == ["AUSDT", "DUSDT"]
    assert symbols2 == ["AUSDT", "DUSDT"]
    assert (start1, end1) == (_ms("2019-10-01"), _ms("2020-01-01") - 1)
    assert (start2, end2) == (_ms("2020-10-01"), _ms("2021-01-01") - 1)
    assert kwargs1 == kwargs2 == {"client": client, "base_url": base_url}
    # Resultado: un par (periodo, universo) por periodo, en orden.
    assert [period for period, _ in result] == universe_periods(params)
    assert [list(frame["symbol"]) for _, frame in result] == [["AUSDT", "DUSDT"]] * 2
    assert all(frame["category"].notna().all() for _, frame in result)


def test_build_universe_does_not_write_any_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_universe solo devuelve datos: la escritura es cosa del llamador."""
    monkeypatch.setattr(universe, "CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(
        universe,
        "list_pairs",
        lambda **kwargs: pd.DataFrame({"symbol": ["AUSDT"], "base_asset": ["A"]}),
    )
    monkeypatch.setattr(
        universe,
        "first_candle_time",
        lambda symbols, reference_ms, **kwargs: pd.Series(
            {s: _ms("2017-01-01") for s in symbols}, dtype="Int64"
        ),
    )
    monkeypatch.setattr(
        universe,
        "median_daily_quote_volume",
        lambda symbols, start, end, **kwargs: pd.Series({s: 5e7 for s in symbols}),
    )
    params = dataclasses.replace(PARAMS, min_categories=1)
    result = universe.build_universe(params, {"A": "l1"}, client=object(), now_ms=_ms("2026-10-13"))
    assert len(result) == len(universe_periods(params))
    assert list(tmp_path.iterdir()) == []
