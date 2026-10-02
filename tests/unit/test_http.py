"""Tests del cliente HTTP base: reintentos, 4xx y timeout, sin tocar la red."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest
import requests

from miax.ingest import ClientError, HttpClient, RetriesExhaustedError
from miax.ingest.http import _retry_after_seconds

URL = "https://example.invalid/api/v3/ping"


def _response(status: int, headers: dict[str, str] | None = None) -> requests.Response:
    """Construye una respuesta de requests con ese codigo y cabeceras, sin red."""
    resp = requests.Response()
    resp.status_code = status
    if headers:
        resp.headers.update(headers)
    return resp


class _FakeGet:
    """Sustituto de `Session.get`: devuelve o lanza lo que dicte la lista de resultados."""

    def __init__(self, outcomes: list[requests.Response | Exception]) -> None:
        """Guarda los resultados que se irán sirviendo, uno por llamada."""
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, **kwargs: object) -> requests.Response:
        """Registra la llamada y devuelve o lanza el siguiente resultado."""
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Sustituye time.sleep por un registro de las esperas pedidas."""
    recorded: list[float] = []
    monkeypatch.setattr("miax.ingest.http.time.sleep", recorded.append)
    return recorded


def _install(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[requests.Response | Exception]
) -> _FakeGet:
    """Parchea `requests.Session.get` con un doble que sirve `outcomes` en orden."""
    fake = _FakeGet(outcomes)
    # staticmethod evita que Python enlace el doble como metodo y le pase `self`.
    monkeypatch.setattr(requests.Session, "get", staticmethod(fake))
    return fake


def test_success_on_first_attempt_does_not_sleep(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Con exito al primer intento hay una sola llamada y ninguna espera."""
    fake = _install(monkeypatch, [_response(200)])
    resp = HttpClient().get(URL)
    assert resp.status_code == 200
    assert len(fake.calls) == 1
    assert sleeps == []


def test_connection_error_then_success_is_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un ConnectionError seguido de exito se reintenta y acaba bien con 2 intentos."""
    fake = _install(monkeypatch, [requests.exceptions.ConnectionError("boom"), _response(200)])
    resp = HttpClient(backoff_base=0.5).get(URL)
    assert resp.status_code == 200
    assert len(fake.calls) == 2
    assert sleeps == [0.5]


def test_5xx_then_success_is_retried(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    """Dos 5xx seguidos de exito se reintentan con espera exponencial."""
    fake = _install(monkeypatch, [_response(500), _response(503), _response(200)])
    resp = HttpClient(backoff_base=1.0).get(URL)
    assert resp.status_code == 200
    assert len(fake.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_timeout_exception_is_retried(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    """Un Timeout de requests es transitorio y se reintenta."""
    fake = _install(monkeypatch, [requests.exceptions.Timeout("slow"), _response(200)])
    HttpClient().get(URL)
    assert len(fake.calls) == 2
    assert len(sleeps) == 1


def test_permanent_failure_exhausts_retries_and_raises(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un 5xx permanente agota los reintentos (4 intentos) y lanza."""
    fake = _install(monkeypatch, [_response(502)] * 4)
    with pytest.raises(RetriesExhaustedError):
        HttpClient(max_retries=3, backoff_base=1.0).get(URL)
    assert len(fake.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_permanent_connection_error_raises_with_cause(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un ConnectionError permanente lanza RetriesExhaustedError encadenada con la causa."""
    fake = _install(monkeypatch, [requests.exceptions.ConnectionError("down")] * 3)
    with pytest.raises(RetriesExhaustedError) as info:
        HttpClient(max_retries=2).get(URL)
    assert isinstance(info.value.__cause__, requests.exceptions.ConnectionError)
    assert len(fake.calls) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("status", [400, 404, 499])
def test_4xx_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status: int
) -> None:
    """Un 4xx duro (no 418/429) lanza ClientError en un solo intento y sin esperar."""
    fake = _install(monkeypatch, [_response(status), _response(200)])
    with pytest.raises(ClientError) as info:
        HttpClient().get(URL)
    assert info.value.status_code == status
    assert len(fake.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [429, 418])
def test_rate_limit_with_seconds_waits_exactly_retry_after(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status: int
) -> None:
    """429 y 418 con Retry-After en segundos esperan esos segundos y luego reintentan."""
    fake = _install(monkeypatch, [_response(status, {"Retry-After": "120"}), _response(200)])
    resp = HttpClient(backoff_base=0.5, max_retry_after=300.0).get(URL)
    assert resp.status_code == 200
    assert len(fake.calls) == 2
    assert sleeps == [120.0]


def test_rate_limit_with_http_date_waits_seconds_until_that_date(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un Retry-After en fecha HTTP futura espera los segundos que faltan hasta ella."""
    target = datetime.now(tz=UTC) + timedelta(seconds=120)
    header = format_datetime(target, usegmt=True)
    fake = _install(monkeypatch, [_response(429, {"Retry-After": header}), _response(200)])
    HttpClient().get(URL)
    assert len(fake.calls) == 2
    # La fecha HTTP trunca a segundos y el reloj avanza algo entre medias.
    assert len(sleeps) == 1
    assert 118.0 <= sleeps[0] <= 120.0


@pytest.mark.parametrize("status", [429, 418])
def test_rate_limit_without_retry_after_uses_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status: int
) -> None:
    """Sin Retry-After, 429 y 418 usan el backoff exponencial de fallback."""
    fake = _install(monkeypatch, [_response(status), _response(status), _response(200)])
    HttpClient(backoff_base=1.0).get(URL)
    assert len(fake.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_rate_limit_with_unparsable_retry_after_falls_back_to_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un Retry-After ilegible se ignora y se usa el backoff exponencial."""
    _install(monkeypatch, [_response(429, {"Retry-After": "pronto"}), _response(200)])
    HttpClient(backoff_base=0.5).get(URL)
    assert sleeps == [0.5]


def test_persistent_429_exhausts_retries_and_raises(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un 429 permanente cuenta contra max_retries y acaba en RetriesExhaustedError."""
    fake = _install(monkeypatch, [_response(429, {"Retry-After": "2"})] * 3)
    with pytest.raises(RetriesExhaustedError, match="HTTP 429"):
        HttpClient(max_retries=2).get(URL)
    assert len(fake.calls) == 3
    assert sleeps == [2.0, 2.0]


def test_retry_after_above_cap_is_clamped(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un Retry-After por encima de max_retry_after se recorta a ese tope."""
    _install(monkeypatch, [_response(418, {"Retry-After": "86400"}), _response(200)])
    HttpClient(max_retry_after=60.0).get(URL)
    assert sleeps == [60.0]


def test_max_retry_after_does_not_clamp_fallback_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """El tope max_retry_after solo recorta Retry-After, no el backoff de fallback."""
    _install(monkeypatch, [_response(429), _response(200)])
    HttpClient(backoff_base=100.0, max_retry_after=60.0).get(URL)
    assert sleeps == [100.0]


def test_http_date_retry_after_above_cap_is_clamped(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Una fecha HTTP lejana tambien se recorta a max_retry_after."""
    target = datetime.now(tz=UTC) + timedelta(days=1)
    header = format_datetime(target, usegmt=True)
    _install(monkeypatch, [_response(418, {"Retry-After": header}), _response(200)])
    HttpClient(max_retry_after=60.0).get(URL)
    assert sleeps == [60.0]


def test_rate_limit_then_5xx_shares_attempt_counter_for_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un 429 consume un reintento: el 5xx siguiente usa el backoff del intento 2."""
    fake = _install(
        monkeypatch, [_response(429, {"Retry-After": "7"}), _response(503), _response(200)]
    )
    HttpClient(backoff_base=1.0, max_retries=2).get(URL)
    assert len(fake.calls) == 3
    assert sleeps == [7.0, 2.0]


def test_rate_limit_with_retry_after_zero_sleeps_zero_not_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Un Retry-After de 0 es valido: espera 0.0 s y no cae al backoff."""
    fake = _install(monkeypatch, [_response(429, {"Retry-After": "0"}), _response(200)])
    HttpClient(backoff_base=5.0).get(URL)
    assert len(fake.calls) == 2
    assert sleeps == [0.0]


def test_rate_limit_retry_is_logged_with_status_and_wait(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """El reintento por 429 se loguea con el codigo y los segundos de espera."""
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    # El logger de miax no propaga y escribe en el stderr original: se engancha un handler propio.
    handler = _Capture()
    http_logger = logging.getLogger("miax.ingest.http")
    http_logger.addHandler(handler)
    try:
        _install(monkeypatch, [_response(429, {"Retry-After": "7"}), _response(200)])
        HttpClient().get(URL)
    finally:
        http_logger.removeHandler(handler)
    assert len(messages) == 1
    assert "HTTP 429" in messages[0]
    assert "7.00s" in messages[0]


def test_retry_after_seconds_parses_integer_seconds() -> None:
    """El helper entiende Retry-After en segundos enteros."""
    assert _retry_after_seconds(_response(429, {"Retry-After": "120"})) == 120.0
    assert _retry_after_seconds(_response(429, {"Retry-After": " 5 "})) == 5.0


def test_retry_after_seconds_parses_http_date_against_fixed_now() -> None:
    """El helper calcula los segundos hasta una fecha HTTP respecto a un 'ahora' fijo."""
    now = datetime(2026, 10, 21, 7, 26, 0, tzinfo=UTC)
    resp = _response(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert _retry_after_seconds(resp, now=now) == 120.0


@pytest.mark.parametrize(
    "header",
    ["Wed, 21 Oct 2026 07:28:00", "Wed, 21 Oct 2026 07:28:00 -0000"],
)
def test_retry_after_seconds_http_date_without_timezone_is_utc(header: str) -> None:
    """Una fecha HTTP sin zona horaria se interpreta como UTC."""
    now = datetime(2026, 10, 21, 7, 26, 0, tzinfo=UTC)
    assert _retry_after_seconds(_response(429, {"Retry-After": header}), now=now) == 120.0


def test_retry_after_seconds_past_http_date_is_zero() -> None:
    """Una fecha HTTP ya pasada da 0 segundos, nunca un valor negativo."""
    now = datetime(2026, 10, 21, 8, 0, 0, tzinfo=UTC)
    resp = _response(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert _retry_after_seconds(resp, now=now) == 0.0


def test_retry_after_seconds_absent_is_none() -> None:
    """Sin cabecera Retry-After el helper devuelve None."""
    assert _retry_after_seconds(_response(429)) is None


@pytest.mark.parametrize("value", ["pronto", "", "-5", "1.5", "12abc"])
def test_retry_after_seconds_garbage_is_none(value: str) -> None:
    """Un valor que no es ni entero ni fecha HTTP devuelve None."""
    assert _retry_after_seconds(_response(429, {"Retry-After": value})) is None


def test_non_transient_requests_exception_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Una excepcion de requests no transitoria se propaga tal cual en un solo intento."""
    fake = _install(monkeypatch, [requests.exceptions.InvalidURL("mala"), _response(200)])
    with pytest.raises(requests.exceptions.InvalidURL):
        HttpClient().get(URL)
    assert len(fake.calls) == 1
    assert sleeps == []


def test_timeout_is_passed_to_requests(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """El timeout y los params configurados llegan a la llamada de requests."""
    fake = _install(monkeypatch, [_response(200)])
    HttpClient(timeout=3.5).get(URL, params={"symbol": "BTCUSDT"})
    assert fake.calls[0]["timeout"] == 3.5
    assert fake.calls[0]["params"] == {"symbol": "BTCUSDT"}


def test_max_retries_zero_means_single_attempt(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Con max_retries=0 hay un unico intento y no se espera."""
    fake = _install(monkeypatch, [_response(500)])
    with pytest.raises(RetriesExhaustedError):
        HttpClient(max_retries=0).get(URL)
    assert len(fake.calls) == 1
    assert sleeps == []


def test_backoff_is_deterministic_exponential() -> None:
    """La espera es base * 2**(reintento-1), sin jitter."""
    client = HttpClient(backoff_base=0.5)
    assert [client.backoff_delay(n) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]


def test_negative_max_retries_is_rejected() -> None:
    """max_retries negativo es un error de configuracion."""
    with pytest.raises(ValueError, match="max_retries"):
        HttpClient(max_retries=-1)


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_non_positive_timeout_is_rejected(timeout: float) -> None:
    """El timeout debe ser > 0."""
    with pytest.raises(ValueError, match="timeout"):
        HttpClient(timeout=timeout)


def test_negative_backoff_base_is_rejected() -> None:
    """backoff_base negativo es un error de configuracion."""
    with pytest.raises(ValueError, match="backoff_base"):
        HttpClient(backoff_base=-0.1)


@pytest.mark.parametrize("cap", [0, -1.0])
def test_non_positive_max_retry_after_is_rejected(cap: float) -> None:
    """max_retry_after debe ser > 0."""
    with pytest.raises(ValueError, match="max_retry_after"):
        HttpClient(max_retry_after=cap)


def test_context_manager_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Al salir del bloque `with` se cierra la sesion de requests."""
    closed: list[bool] = []
    monkeypatch.setattr(requests.Session, "close", lambda self: closed.append(True))
    with HttpClient() as client:
        assert isinstance(client, HttpClient)
        assert closed == []
    assert closed == [True]


def test_context_manager_closes_session_when_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si el cuerpo del `with` lanza, la sesion se cierra igualmente y la excepcion sale."""
    closed: list[bool] = []
    monkeypatch.setattr(requests.Session, "close", lambda self: closed.append(True))
    with pytest.raises(RuntimeError, match="fallo del cuerpo"), HttpClient():
        raise RuntimeError("fallo del cuerpo")
    assert closed == [True]
