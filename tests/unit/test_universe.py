"""Tests del filtro de universo por volumen con un Binance simulado, sin tocar la red."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from miax.ingest import universe
from miax.ingest.universe import (
    MAX_LOGGED_UNCATEGORIZED,
    diversify_universe,
    filter_by_listing_age,
    filter_by_volume,
    first_candle_time,
    load_categories,
    median_daily_quote_volume,
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
