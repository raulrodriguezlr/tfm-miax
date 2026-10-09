"""Tests de la descarga masiva de simbolos, con `update_symbol` parcheado: sin red ni disco."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
import requests

from miax.ingest import ClientError, HttpError, RetriesExhaustedError
from miax.ingest import __main__ as cli

NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)


class _Recorder:
    """Sustituto de `update_symbol` que anota las llamadas y falla en simbolos concretos."""

    def __init__(self, failing: dict[str, Exception] | None = None) -> None:
        self.failing = failing or {}
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> int:
        """Anota la llamada; lanza el error del simbolo si lo tiene, si no devuelve 3."""
        self.calls.append((args, kwargs))
        symbol = str(args[0])
        if symbol in self.failing:
            raise self.failing[symbol]
        return 3

    @property
    def symbols(self) -> list[object]:
        """Simbolos pedidos, en el orden de las llamadas."""
        return [args[0] for args, _ in self.calls]


class _ListHandler(logging.Handler):
    """Handler que guarda los mensajes formateados para poder inspeccionarlos."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Guarda el mensaje ya formateado."""
        self.messages.append(record.getMessage())


@pytest.fixture
def logs() -> Iterator[_ListHandler]:
    """Engancha un handler al logger de la CLI mientras dura el test."""
    handler = _ListHandler()
    cli.logger.addHandler(handler)
    yield handler
    cli.logger.removeHandler(handler)


def _install(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    """Parchea `update_symbol` y el reloj de la CLI."""
    monkeypatch.setattr(cli, "update_symbol", recorder)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)


def test_symbols_are_downloaded_in_order_with_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cada simbolo llama una vez a `update_symbol`, en orden y con el mismo cliente."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT"]) == 0

    assert recorder.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    clients = [kwargs["client"] for _, kwargs in recorder.calls]
    assert isinstance(clients[0], cli.HttpClient)
    assert clients[0].rate_limiter is not None
    assert all(client is clients[0] for client in clients)


def test_client_is_closed_once_for_the_whole_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """El cliente se crea una vez y se cierra una vez, no por simbolo."""
    closed: list[bool] = []
    monkeypatch.setattr(cli.HttpClient, "close", lambda self: closed.append(True))
    _install(monkeypatch, _Recorder())

    cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT"])

    assert closed == [True]


@pytest.mark.parametrize("error", [HttpError("sin red"), ValueError("sin red"), OSError("sin red")])
def test_failing_symbol_does_not_abort_the_batch(
    monkeypatch: pytest.MonkeyPatch, logs: _ListHandler, error: Exception
) -> None:
    """Si un simbolo falla se procesan los demas, el resumen lo refleja y el codigo es 1."""
    recorder = _Recorder(failing={"ETHUSDT": error})
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT"]) == 1

    assert recorder.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert any("Fallo la descarga de ETHUSDT" in m and "sin red" in m for m in logs.messages)
    assert logs.messages[-1] == "Resumen: 2 simbolos OK, 1 fallidos (ETHUSDT)"


def test_client_is_closed_once_even_if_a_symbol_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aunque un simbolo falle, el cliente se cierra una sola vez."""
    closed: list[bool] = []
    monkeypatch.setattr(cli.HttpClient, "close", lambda self: closed.append(True))
    _install(monkeypatch, _Recorder(failing={"ETHUSDT": OSError("disco")}))

    assert cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT"]) == 1

    assert closed == [True]


def test_same_clipped_range_reaches_every_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """El mismo `start_ms`/`end_ms` (ya recortado a la ultima vela cerrada) llega a todos."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT", "--interval", "1h"]) == 0

    ranges = {(args[2], args[3]) for args, _ in recorder.calls}
    assert len(recorder.calls) == 3
    assert len(ranges) == 1
    # NOW son las 12:00 en punto: la vela de 1h en curso abre ahora, la cerrada acaba 1 ms antes.
    assert ranges.pop()[1] == cli._to_ms(NOW) - 1


def test_circuit_breaker_stops_the_batch_after_consecutive_systemic_failures(
    monkeypatch: pytest.MonkeyPatch, logs: _ListHandler
) -> None:
    """Con N RetriesExhaustedError seguidos se corta: el resto no se intenta y el codigo es 1."""
    failing: dict[str, Exception] = {
        s: RetriesExhaustedError("418") for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    }
    recorder = _Recorder(failing=failing)
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,XRPUSDT"]) == 1

    assert recorder.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert any("Tanda cortada" in m and "2 simbolos" in m for m in logs.messages)
    assert logs.messages[-1] == (
        "Resumen: 0 simbolos OK, 3 fallidos (BTCUSDT, ETHUSDT, SOLUSDT), "
        "2 sin procesar (ADAUSDT, XRPUSDT)"
    )


def test_circuit_breaker_marks_pending_symbols_as_not_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Los simbolos tras el corte quedan como NOT_PROCESSED, distintos de los fallidos (None)."""
    failing: dict[str, Exception] = {
        "A": RetriesExhaustedError("x"),
        "B": RetriesExhaustedError("x"),
    }
    recorder = _Recorder(failing=failing)
    monkeypatch.setattr(cli, "update_symbol", recorder)

    with cli.HttpClient() as client:
        results = cli.download_symbols(
            ["A", "B", "C", "D"], "1m", 0, 1, client=client, max_consecutive_failures=2
        )

    assert results == {"A": None, "B": None, "C": cli.NOT_PROCESSED, "D": cli.NOT_PROCESSED}
    assert recorder.symbols == ["A", "B"]


def test_client_is_closed_once_when_the_circuit_breaker_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aunque el cortacircuitos corte la tanda, el cliente compartido se cierra una sola vez."""
    closed: list[bool] = []
    monkeypatch.setattr(cli.HttpClient, "close", lambda self: closed.append(True))
    failing: dict[str, Exception] = {s: RetriesExhaustedError("429") for s in ("A", "B", "C")}
    recorder = _Recorder(failing=failing)
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", "A,B,C,D,E"]) == 1

    assert recorder.symbols == ["A", "B", "C"]
    assert closed == [True]


def test_success_in_between_resets_the_systemic_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un exito intercalado resetea el contador: fallos separados por un exito no cortan."""
    failing: dict[str, Exception] = {s: RetriesExhaustedError("429") for s in ("A", "B", "D", "E")}
    recorder = _Recorder(failing=failing)
    monkeypatch.setattr(cli, "update_symbol", recorder)

    with cli.HttpClient() as client:
        results = cli.download_symbols(
            ["A", "B", "C", "D", "E", "F"], "1m", 0, 1, client=client, max_consecutive_failures=3
        )

    assert recorder.symbols == ["A", "B", "C", "D", "E", "F"]
    assert results == {"A": None, "B": None, "C": 3, "D": None, "E": None, "F": 3}


def test_symbol_own_failure_resets_the_systemic_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un fallo propio del simbolo (ClientError) tambien resetea el contador sistemico."""
    failing: dict[str, Exception] = {
        "A": RetriesExhaustedError("x"),
        "B": RetriesExhaustedError("x"),
        "C": ClientError(400, "http://x"),
        "D": RetriesExhaustedError("x"),
    }
    recorder = _Recorder(failing=failing)
    monkeypatch.setattr(cli, "update_symbol", recorder)

    with cli.HttpClient() as client:
        results = cli.download_symbols(
            ["A", "B", "C", "D", "E"], "1m", 0, 1, client=client, max_consecutive_failures=3
        )

    assert recorder.symbols == ["A", "B", "C", "D", "E"]
    assert results["E"] == 3


def test_requests_error_is_isolated_and_does_not_count_as_systemic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un error de `requests` no reintentado aisla al simbolo y la tanda sigue."""
    error = requests.exceptions.ChunkedEncodingError("corte")
    recorder = _Recorder(failing={"B": error})
    monkeypatch.setattr(cli, "update_symbol", recorder)

    with cli.HttpClient() as client:
        results = cli.download_symbols(
            ["A", "B", "C"], "1m", 0, 1, client=client, max_consecutive_failures=1
        )

    assert results == {"A": 3, "B": None, "C": 3}
    assert recorder.symbols == ["A", "B", "C"]


@pytest.mark.parametrize("error", [HttpError("x"), ValueError("x"), OSError("x")])
def test_download_symbols_records_failure_as_none(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """`download_symbols` devuelve None para el simbolo fallido y las velas nuevas para el resto."""
    monkeypatch.setattr(cli, "update_symbol", _Recorder(failing={"B": error}))

    with cli.HttpClient() as client:
        results = cli.download_symbols(["A", "B", "C"], "1m", 0, 1, client=client)

    assert results == {"A": 3, "B": None, "C": 3}
    assert list(results) == ["A", "B", "C"]


def test_all_ok_returns_zero_and_summarises(
    monkeypatch: pytest.MonkeyPatch, logs: _ListHandler
) -> None:
    """Con todos los simbolos bien, el codigo es 0 y el resumen no lista fallos."""
    _install(monkeypatch, _Recorder())

    assert cli.main(["--symbol", "BTCUSDT,ETHUSDT"]) == 0

    assert logs.messages[-1] == "Resumen: 2 simbolos OK, 0 fallidos"


def test_progress_is_logged_per_symbol(monkeypatch: pytest.MonkeyPatch, logs: _ListHandler) -> None:
    """Se registra el progreso `[i/N] SIMBOLO: K velas nuevas` de cada simbolo."""
    _install(monkeypatch, _Recorder())

    cli.main(["--symbol", "BTCUSDT,ETHUSDT,SOLUSDT"])

    for position, symbol in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT"], start=1):
        assert f"[{position}/3] {symbol}: 3 velas nuevas" in logs.messages


def test_single_symbol_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un solo simbolo (sin comas) sigue funcionando como antes."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", "BTCUSDT"]) == 0
    assert recorder.symbols == ["BTCUSDT"]


def test_symbol_list_is_uppercased_and_deduplicated_after_normalising() -> None:
    """`btcusdt` y `BTCUSDT` son el mismo simbolo: se pasa a mayusculas antes de deduplicar."""
    assert cli._parse_symbols("btcusdt,BTCUSDT, ethusdt") == ["BTCUSDT", "ETHUSDT"]


def test_symbol_list_is_trimmed_and_cleaned() -> None:
    """Se quitan espacios y vacios, se mantiene el orden y se descartan repetidos."""
    assert cli._parse_symbols(" BTCUSDT, ETHUSDT ,,BTCUSDT,SOLUSDT,") == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]


def test_empty_symbol_list_fails_without_downloading(
    monkeypatch: pytest.MonkeyPatch, logs: _ListHandler
) -> None:
    """Una lista sin simbolos reales da codigo 2 y no descarga."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    assert cli.main(["--symbol", " , ,"]) == 2

    assert recorder.calls == []
    assert any("ningun simbolo" in m for m in logs.messages)
