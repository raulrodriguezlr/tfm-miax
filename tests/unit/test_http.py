"""Tests del cliente HTTP base: reintentos, 4xx y timeout, sin tocar la red."""

from __future__ import annotations

import pytest
import requests

from miax.ingest import ClientError, HttpClient, RetriesExhaustedError

URL = "https://example.invalid/api/v3/ping"


def _response(status: int) -> requests.Response:
    """Construye una respuesta de requests con ese codigo, sin red."""
    resp = requests.Response()
    resp.status_code = status
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


@pytest.mark.parametrize("status", [400, 404, 418, 429, 499])
def test_4xx_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status: int
) -> None:
    """Todo 4xx (incluidos 418 y 429, hasta MIAX-017) lanza ClientError en un solo intento."""
    fake = _install(monkeypatch, [_response(status), _response(200)])
    with pytest.raises(ClientError) as info:
        HttpClient().get(URL)
    assert info.value.status_code == status
    assert len(fake.calls) == 1
    assert sleeps == []


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
