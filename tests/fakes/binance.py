"""Binance falso para tests sin red: sirve los fixtures grabados como lo haria la API (MIAX-032).

Vive en `tests/fakes/` y no dentro de un test porque lo puede reutilizar cualquier test de
integracion. Se engancha a `requests.Session.get`, asi el codigo de produccion no cambia.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import requests
from requests.structures import CaseInsensitiveDict

Json = dict[str, Any] | list[Any]

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "binance"

KLINES_PATH = "/api/v3/klines"
EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
KLINES_WEIGHT = 2
EXCHANGE_INFO_WEIGHT = 20
DEFAULT_LIMIT = 500  # el que aplica Binance si no se pasa `limit`
MAX_LIMIT = 1000
USED_WEIGHT_HEADER = "X-MBX-USED-WEIGHT-1M"


class FakeResponse:
    """Respuesta simulada con `status_code`, `headers` y `json()`, como la de `requests`."""

    def __init__(self, status_code: int, payload: Json, headers: dict[str, str]) -> None:
        """Guarda el codigo, el cuerpo ya decodificado y las cabeceras."""
        self.status_code = status_code
        self.headers: CaseInsensitiveDict[str] = CaseInsensitiveDict(headers)
        self._payload = payload

    def json(self) -> Json:
        """Devuelve una copia del cuerpo, para que quien lo reciba pueda modificarlo."""
        return copy.deepcopy(self._payload)

    @property
    def text(self) -> str:
        """Cuerpo serializado como texto."""
        return json.dumps(self._payload)


@dataclass(frozen=True)
class RecordedRequest:
    """Una peticion recibida por el falso, con su resultado."""

    path: str
    params: dict[str, Any]
    status: int
    n_rows: int | None  # filas devueltas por /klines; None en el resto


def _load(fixtures_dir: Path, name: str) -> Json:
    """Lee un fixture JSON de la carpeta dada."""
    return json.loads((fixtures_dir / name).read_text(encoding="utf-8"))


def _error_body(status: int) -> dict[str, Any]:
    """Cuerpo de error al estilo de Binance (`code` y `msg`)."""
    return {"code": -1003 if status in (418, 429) else -1100, "msg": f"Error simulado {status}"}


class FakeBinance:
    """Responde a `/api/v3/klines` y `/api/v3/exchangeInfo` con los fixtures grabados."""

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR) -> None:
        """Carga los fixtures; las velas se sirven ordenadas por `open_time`."""
        raw: dict[tuple[str, str], list[list[Any]]] = {
            ("BTCUSDT", "1m"): _load(fixtures_dir, "klines_BTCUSDT_1m.json"),
            ("LUNAUSDT", "1m"): _load(fixtures_dir, "klines_LUNAUSDT_1m.json"),
        }
        for symbol, rows in _load(fixtures_dir, "klines_1d.json").items():
            raw[(symbol, "1d")] = rows
        self._klines = {key: sorted(rows, key=lambda row: int(row[0])) for key, rows in raw.items()}
        self.exchange_info: dict[str, Any] = _load(fixtures_dir, "exchange_info.json")
        self.requests: list[RecordedRequest] = []
        self.used_weight = 0
        self._scheduled: deque[tuple[int, dict[str, str]]] = deque()

    @property
    def kline_requests(self) -> list[RecordedRequest]:
        """Las peticiones recibidas a `/klines`, en orden."""
        return [item for item in self.requests if item.path == KLINES_PATH]

    def schedule_error(
        self, status: int, *, retry_after: int | None = None, times: int = 1
    ) -> None:
        """Hace que las proximas `times` peticiones, sean cuales sean, respondan con ese error."""
        headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
        self._scheduled.extend([(status, headers)] * times)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sustituye `requests.Session.get` por este falso hasta que acabe el test."""

        def _get(
            _session: requests.Session,
            url: str,
            params: Mapping[str, Any] | None = None,
            **_kwargs: object,
        ) -> FakeResponse:
            return self.handle(url, params)

        monkeypatch.setattr(requests.Session, "get", _get)

    def _respond(
        self, status: int, payload: Json, weight: int, headers: dict[str, str] | None = None
    ) -> FakeResponse:
        """Construye la respuesta con el peso acumulado en la cabecera de Binance."""
        self.used_weight += weight
        merged = {USED_WEIGHT_HEADER: str(self.used_weight), **(headers or {})}
        return FakeResponse(status, payload, merged)

    def handle(self, url: str, params: Mapping[str, Any] | None = None) -> FakeResponse:
        """Atiende un GET como lo haria Binance y lo anota en `requests`."""
        path = urlparse(url).path
        query = dict(params or {})
        if self._scheduled:
            status, headers = self._scheduled.popleft()
            response = self._respond(status, _error_body(status), 1, headers)
            self.requests.append(RecordedRequest(path, query, status, None))
            return response
        if path == KLINES_PATH:
            response = self._klines_response(query)
        elif path == EXCHANGE_INFO_PATH:
            response = self._respond(200, self.exchange_info, EXCHANGE_INFO_WEIGHT)
        else:
            response = self._respond(404, {"code": 0, "msg": f"Ruta desconocida {path}"}, 1)
        n_rows = (
            len(response.json()) if path == KLINES_PATH and response.status_code == 200 else None
        )
        self.requests.append(RecordedRequest(path, query, response.status_code, n_rows))
        return response

    def _klines_response(self, query: dict[str, Any]) -> FakeResponse:
        """Filtra las velas por simbolo, intervalo, rango (inclusivo) y `limit`."""
        symbol, interval = query.get("symbol"), query.get("interval")
        if symbol is None or interval is None:
            return self._respond(400, {"code": -1102, "msg": "Falta symbol o interval"}, 1)
        rows = self._klines.get((str(symbol), str(interval)))
        if rows is None:
            # Un simbolo sin fixture es un fallo del test, no una respuesta vacia.
            return self._respond(400, {"code": -1121, "msg": "Invalid symbol."}, 1)
        start = int(query.get("startTime", 0))
        end = int(query["endTime"]) if "endTime" in query else None
        limit = min(int(query.get("limit", DEFAULT_LIMIT)), MAX_LIMIT)
        selected = [
            row for row in rows if int(row[0]) >= start and (end is None or int(row[0]) <= end)
        ]
        return self._respond(200, selected[:limit], KLINES_WEIGHT)
