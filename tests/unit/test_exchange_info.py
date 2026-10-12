"""Tests del catalogo de pares de Binance con un cliente simulado, sin tocar la red."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from miax.ingest import exchange_info as module
from miax.ingest.exchange_info import (
    EXCHANGE_INFO_WEIGHT,
    PAIR_COLUMNS,
    fetch_exchange_info,
    list_pairs,
    pairs_from_exchange_info,
)
from miax.ingest.http import ClientError

BASE = "https://api.binance.com"
URL = "https://api.binance.com/api/v3/exchangeInfo"


def _symbol(
    symbol: str,
    base: str,
    quote: str,
    status: str = "TRADING",
    margin: bool | None = True,
) -> dict[str, Any]:
    """Un simbolo con el formato de Binance; `margin=None` omite el flag de margen."""
    item: dict[str, Any] = {
        "symbol": symbol,
        "status": status,
        "baseAsset": base,
        "quoteAsset": quote,
        "isSpotTradingAllowed": True,
    }
    if margin is not None:
        item["isMarginTradingAllowed"] = margin
    return item


def _payload() -> dict[str, Any]:
    """Payload pequeno y desordenado: mezcla USDT y BTC, TRADING y BREAK, un flag ausente."""
    return {
        "timezone": "UTC",
        "symbols": [
            _symbol("ETHUSDT", "ETH", "USDT"),
            _symbol("BNBBTC", "BNB", "BTC", margin=None),
            _symbol("ADAUSDT", "ADA", "USDT", status="BREAK", margin=False),
            _symbol("ETHBTC", "ETH", "BTC"),
            _symbol("BTCUSDT", "BTC", "USDT"),
        ],
    }


class _FakeResponse:
    """Respuesta simulada que solo expone `json()`."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Guarda el cuerpo que devolvera `json()`."""
        self._payload = payload

    def json(self) -> dict[str, Any]:
        """Devuelve el cuerpo simulado."""
        return self._payload


class _FakeClient:
    """Cliente falso que registra (url, params, weight) y si se cerro."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        """Prepara el payload a servir."""
        self.payload = _payload() if payload is None else payload
        self.calls: list[tuple[str, dict[str, Any] | None, int]] = []
        self.closed = False

    def get(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> _FakeResponse:
        """Registra la llamada y devuelve el payload."""
        self.calls.append((url, params, weight))
        return _FakeResponse(self.payload)

    def close(self) -> None:
        """Marca el cliente como cerrado."""
        self.closed = True


class _RaisingClient(_FakeClient):
    """Cliente falso cuyo `get` lanza un ClientError."""

    def get(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> _FakeResponse:
        """Lanza un 400 como haria el cliente real."""
        raise ClientError(400, url)


# --- pairs_from_exchange_info ---


def test_columns_are_exactly_pair_columns_in_order() -> None:
    """Las columnas coinciden con PAIR_COLUMNS y en el mismo orden."""
    df = pairs_from_exchange_info(_payload())
    assert list(df.columns) == PAIR_COLUMNS


def test_dtypes_flags_are_bool_and_texts_are_string() -> None:
    """Los flags son bool y los textos tienen dtype de texto."""
    df = pairs_from_exchange_info(_payload())
    for col in ("is_spot_trading_allowed", "is_margin_trading_allowed"):
        assert df[col].dtype == bool
    expected = pd.Series(["x"]).astype("str").dtype
    for col in ("symbol", "status", "base_asset", "quote_asset"):
        assert df[col].dtype == expected
        assert pd.api.types.is_string_dtype(df[col])


def test_sorted_by_symbol_with_clean_index() -> None:
    """El resultado esta ordenado por symbol con indice 0..n-1."""
    df = pairs_from_exchange_info(_payload())
    assert df["symbol"].tolist() == ["ADAUSDT", "BNBBTC", "BTCUSDT", "ETHBTC", "ETHUSDT"]
    assert df.index.tolist() == list(range(5))


def test_missing_margin_flag_defaults_to_false() -> None:
    """Un simbolo sin isMarginTradingAllowed queda con el flag en False."""
    df = pairs_from_exchange_info(_payload()).set_index("symbol")
    assert not df.loc["BNBBTC", "is_margin_trading_allowed"]
    assert df.loc["ETHBTC", "is_margin_trading_allowed"]


def test_empty_symbols_gives_empty_frame_with_columns() -> None:
    """Con `symbols: []` sale un DataFrame vacio con las columnas del catalogo."""
    df = pairs_from_exchange_info({"symbols": []})
    assert df.empty
    assert list(df.columns) == PAIR_COLUMNS


def test_payload_without_symbols_key_raises() -> None:
    """Sin la clave `symbols` se lanza ValueError que la nombra."""
    with pytest.raises(ValueError, match="symbols"):
        pairs_from_exchange_info({"timezone": "UTC"})


@pytest.mark.parametrize(
    ("missing", "field"),
    [("status", "status"), ("baseAsset", "baseAsset"), ("quoteAsset", "quoteAsset")],
)
def test_symbol_missing_required_field_names_it(missing: str, field: str) -> None:
    """Un simbolo sin un campo obligatorio lanza ValueError que nombra el campo."""
    bad = _symbol("ETHUSDT", "ETH", "USDT")
    del bad[missing]
    with pytest.raises(ValueError, match=field):
        pairs_from_exchange_info({"symbols": [bad]})


# --- list_pairs ---


def test_list_pairs_without_filters_returns_all_including_break() -> None:
    """Sin filtros salen todos los pares, tambien los BREAK."""
    df = list_pairs(client=_FakeClient())  # type: ignore[arg-type]
    assert len(df) == 5
    assert "BREAK" in df["status"].tolist()


def test_list_pairs_filters_by_quote_asset() -> None:
    """El filtro quote_asset deja solo ese activo cotizado."""
    df = list_pairs(quote_asset="USDT", client=_FakeClient())  # type: ignore[arg-type]
    assert df["symbol"].tolist() == ["ADAUSDT", "BTCUSDT", "ETHUSDT"]
    assert df.index.tolist() == [0, 1, 2]


def test_list_pairs_filters_by_status() -> None:
    """El filtro status excluye los pares en BREAK."""
    df = list_pairs(status="TRADING", client=_FakeClient())  # type: ignore[arg-type]
    assert df["symbol"].tolist() == ["BNBBTC", "BTCUSDT", "ETHBTC", "ETHUSDT"]


def test_list_pairs_combines_filters() -> None:
    """Los dos filtros se aplican a la vez."""
    df = list_pairs(
        quote_asset="USDT",
        status="TRADING",
        client=_FakeClient(),  # type: ignore[arg-type]
    )
    assert df["symbol"].tolist() == ["BTCUSDT", "ETHUSDT"]


def test_list_pairs_filter_without_matches_is_empty_with_columns() -> None:
    """Un filtro sin coincidencias da un DataFrame vacio con las columnas."""
    df = list_pairs(quote_asset="EUR", client=_FakeClient())  # type: ignore[arg-type]
    assert df.empty
    assert list(df.columns) == PAIR_COLUMNS


# --- fetch_exchange_info ---


@pytest.mark.parametrize("base_url", [BASE, BASE + "/"])
def test_fetch_builds_url_and_uses_weight(base_url: str) -> None:
    """La URL se forma con o sin barra final y la llamada pesa 20."""
    fake = _FakeClient()
    result = fetch_exchange_info(client=fake, base_url=base_url)  # type: ignore[arg-type]
    assert result == fake.payload
    assert len(fake.calls) == 1
    url, _, weight = fake.calls[0]
    assert url == URL
    assert weight == EXCHANGE_INFO_WEIGHT == 20


def test_injected_client_is_not_closed() -> None:
    """Un cliente inyectado no lo cierra la funcion."""
    fake = _FakeClient()
    fetch_exchange_info(client=fake)  # type: ignore[arg-type]
    assert fake.closed is False


def test_own_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """El cliente creado por la funcion se cierra al terminar."""
    created: list[_FakeClient] = []

    def factory() -> _FakeClient:
        fake = _FakeClient()
        created.append(fake)
        return fake

    monkeypatch.setattr(module, "HttpClient", factory)
    fetch_exchange_info()
    assert len(created) == 1
    assert created[0].closed is True
    assert created[0].calls[0][0] == URL


def test_own_client_is_closed_even_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la llamada falla, el cliente propio se cierra igualmente."""
    created: list[_RaisingClient] = []

    def factory() -> _RaisingClient:
        fake = _RaisingClient()
        created.append(fake)
        return fake

    monkeypatch.setattr(module, "HttpClient", factory)
    with pytest.raises(ClientError):
        fetch_exchange_info()
    assert created[0].closed is True


def test_client_error_propagates() -> None:
    """Un ClientError del cliente sube hasta quien llama."""
    with pytest.raises(ClientError):
        fetch_exchange_info(client=_RaisingClient())  # type: ignore[arg-type]
    with pytest.raises(ClientError):
        list_pairs(client=_RaisingClient())  # type: ignore[arg-type]
