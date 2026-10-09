"""Tests de la CLI de ingesta, con `update_symbol` parcheado: no tocan la red ni el disco."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from miax.ingest import ClientError, HttpError
from miax.ingest import __main__ as cli

NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)
NOW_MS = 1_791_547_200_000  # 2026-10-09T12:00:00Z
DAY_MS = 86_400_000
START_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z


class _Spy:
    """Sustituto de `update_symbol` que registra la llamada y no descarga nada."""

    def __init__(self, result: int = 7) -> None:
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> int:
        """Anota los argumentos y devuelve un numero fijo de velas nuevas."""
        self.calls.append((args, kwargs))
        return self.result


class _ListHandler(logging.Handler):
    """Handler que guarda los mensajes formateados para poder inspeccionarlos."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Guarda el mensaje ya formateado."""
        self.messages.append(record.getMessage())


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Parchea `update_symbol` y fija el reloj de la CLI."""
    fake = _Spy()
    monkeypatch.setattr(cli, "update_symbol", fake)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    return fake


@pytest.fixture
def logs() -> Iterator[_ListHandler]:
    """Engancha un handler al logger de la CLI mientras dura el test."""
    handler = _ListHandler()
    cli.logger.addHandler(handler)
    yield handler
    cli.logger.removeHandler(handler)


def test_known_dates_match_expected_ms() -> None:
    """Las constantes de los tests y la conversion de la CLI coinciden."""
    assert int(NOW.timestamp() * 1000) == NOW_MS
    assert cli._day_start_ms("2026-01-01") == START_MS


def test_defaults_call_update_symbol_with_last_30_days(spy: _Spy) -> None:
    """Sin fechas, el rango va de hace 30 dias a la ultima vela cerrada, con intervalo 1m."""
    assert cli.main(["--symbol", "BTCUSDT"]) == 0

    (args, kwargs) = spy.calls[0]
    # NOW cae justo en el cambio de minuto: la vela de las 12:00 aun esta en curso.
    assert args == ("BTCUSDT", "1m", NOW_MS - 30 * DAY_MS, NOW_MS - 1)
    assert kwargs["base_dir"] == "data/klines"
    assert kwargs["base_url"] == "https://api.binance.com"


def test_explicit_interval_start_end_are_parsed_to_utc_ms(spy: _Spy) -> None:
    """Intervalo y fechas explicitos llegan a `update_symbol` como ms UTC exactos."""
    argv = [
        "--symbol",
        "ETHUSDT",
        "--interval",
        "5m",
        "--start",
        "2026-01-01",
        "--end",
        "2026-01-02",
    ]
    assert cli.main(argv) == 0

    (args, _) = spy.calls[0]
    # --start es 00:00 UTC del dia; --end es inclusivo: llega al ultimo ms de ese dia.
    assert args == ("ETHUSDT", "5m", START_MS, START_MS + 2 * DAY_MS - 1)


def test_start_and_end_same_day_is_valid(spy: _Spy) -> None:
    """Un rango de un solo dia (start == end) es valido y cubre el dia entero."""
    assert cli.main(["--symbol", "BTCUSDT", "--start", "2026-01-01", "--end", "2026-01-01"]) == 0
    assert spy.calls[0][0][2:] == (START_MS, START_MS + DAY_MS - 1)


def test_base_dir_and_base_url_are_forwarded(spy: _Spy) -> None:
    """`--base-dir` y `--base-url` se pasan tal cual a `update_symbol`."""
    cli.main(["--symbol", "BTCUSDT", "--base-dir", "somewhere", "--base-url", "http://x"])

    kwargs = spy.calls[0][1]
    assert kwargs["base_dir"] == "somewhere"
    assert kwargs["base_url"] == "http://x"


def test_client_has_rate_limiter_and_is_closed(spy: _Spy, monkeypatch: pytest.MonkeyPatch) -> None:
    """El cliente lleva `RateLimiter` y se cierra al terminar."""
    closed: list[bool] = []
    monkeypatch.setattr(cli.HttpClient, "close", lambda self: closed.append(True))

    cli.main(["--symbol", "BTCUSDT"])

    client = spy.calls[0][1]["client"]
    assert isinstance(client, cli.HttpClient)
    assert isinstance(client.rate_limiter, cli.RateLimiter)
    assert closed == [True]


def test_summary_reports_new_candles(spy: _Spy, logs: _ListHandler) -> None:
    """El resumen final nombra el simbolo y cuantas velas nuevas se bajaron."""
    cli.main(["--symbol", "BTCUSDT"])

    summary = logs.messages[-1]
    assert "BTCUSDT" in summary
    assert "7 velas nuevas" in summary


@pytest.mark.parametrize("bad", ["no-es-fecha", "2026-13-01", "2026/01/01", ""])
def test_invalid_start_fails_without_downloading(spy: _Spy, logs: _ListHandler, bad: str) -> None:
    """Una fecha `--start` invalida da codigo distinto de 0, avisa y no descarga."""
    assert cli.main(["--symbol", "BTCUSDT", "--start", bad]) != 0

    assert spy.calls == []
    assert any("Fecha invalida" in message for message in logs.messages)


def test_invalid_end_fails_without_downloading(spy: _Spy) -> None:
    """Una fecha `--end` invalida da codigo distinto de 0 y no descarga."""
    assert cli.main(["--symbol", "BTCUSDT", "--end", "ayer"]) != 0
    assert spy.calls == []


def test_start_after_end_fails_without_downloading(spy: _Spy, logs: _ListHandler) -> None:
    """Con `start > end` devuelve codigo distinto de 0 y no descarga."""
    assert cli.main(["--symbol", "BTCUSDT", "--start", "2026-02-01", "--end", "2026-01-01"]) != 0

    assert spy.calls == []
    assert any("posterior" in message for message in logs.messages)


def test_default_start_after_explicit_past_end_fails(spy: _Spy) -> None:
    """El inicio por defecto (hace 30 dias) cae despues de un `--end` muy antiguo: error."""
    assert cli.main(["--symbol", "BTCUSDT", "--end", "2020-01-01"]) != 0
    assert spy.calls == []


def test_download_error_returns_nonzero_without_traceback(
    monkeypatch: pytest.MonkeyPatch, logs: _ListHandler
) -> None:
    """Un fallo HTTP se loguea y devuelve 1 en vez de propagar la excepcion."""

    def boom(*args: object, **kwargs: object) -> int:
        raise HttpError("sin conexion")

    monkeypatch.setattr(cli, "update_symbol", boom)

    assert cli.main(["--symbol", "BTCUSDT"]) == 1
    assert any("sin conexion" in message for message in logs.messages)


@pytest.mark.parametrize(
    "error",
    [HttpError("sin red"), ClientError(400, "http://x"), ValueError("raro"), OSError("disco")],
)
def test_download_failures_return_one(
    monkeypatch: pytest.MonkeyPatch, logs: _ListHandler, error: Exception
) -> None:
    """HttpError, ClientError, ValueError y OSError de la descarga devuelven 1 y se loguean."""

    def boom(*args: object, **kwargs: object) -> int:
        raise error

    monkeypatch.setattr(cli, "update_symbol", boom)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)

    assert cli.main(["--symbol", "BTCUSDT"]) == 1
    assert any("Fallo la descarga" in message for message in logs.messages)


def test_client_is_closed_even_if_download_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """El cliente HTTP se cierra aunque `update_symbol` lance."""
    closed: list[bool] = []

    def boom(*args: object, **kwargs: object) -> int:
        raise HttpError("sin red")

    monkeypatch.setattr(cli, "update_symbol", boom)
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW)
    monkeypatch.setattr(cli.HttpClient, "close", lambda self: closed.append(True))

    assert cli.main(["--symbol", "BTCUSDT"]) == 1
    assert closed == [True]


def test_start_after_now_without_end_fails(spy: _Spy, logs: _ListHandler) -> None:
    """Un `--start` futuro sin `--end` queda despues de la ultima vela cerrada: rc 2."""
    assert cli.main(["--symbol", "BTCUSDT", "--start", "2026-12-01"]) == 2

    assert spy.calls == []
    assert any("posterior" in message for message in logs.messages)


def test_end_is_clipped_to_last_closed_candle(spy: _Spy, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mitad de un minuto, el fin por defecto es el ultimo ms de la vela 1m ya cerrada."""
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW + timedelta(seconds=30))

    assert cli.main(["--symbol", "BTCUSDT", "--interval", "1m"]) == 0

    end_ms = spy.calls[0][0][3]
    # A las 12:00:30 la vela de las 12:00 sigue abierta: la ultima cerrada acaba a las 11:59:59.999.
    assert end_ms == NOW_MS - 1


def test_future_end_is_clipped_to_last_closed_candle(
    spy: _Spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un `--end` de hoy o futuro tambien se recorta a la ultima vela cerrada."""
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW + timedelta(seconds=30))

    assert cli.main(["--symbol", "BTCUSDT", "--end", "2099-01-01"]) == 0
    assert cli.main(["--symbol", "BTCUSDT", "--end", "2026-10-09"]) == 0

    assert [call[0][3] for call in spy.calls] == [NOW_MS - 1, NOW_MS - 1]


def test_past_end_is_not_clipped(spy: _Spy) -> None:
    """Un `--end` ya cerrado por completo no se toca."""
    assert cli.main(["--symbol", "BTCUSDT", "--start", "2026-01-01", "--end", "2026-01-02"]) == 0
    assert spy.calls[0][0][3] == START_MS + 2 * DAY_MS - 1


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1m", NOW_MS - 1),
        ("15m", NOW_MS - 1),  # las 12:00 es multiplo de 15 min
        ("1h", NOW_MS - 1),
        ("4h", NOW_MS - 1),
        ("1d", NOW_MS - 12 * 60 * 60 * 1000 - 1),  # la vela diaria de hoy sigue abierta
        ("1w", START_MS + 277 * DAY_MS - 1),  # lunes 2026-10-05 00:00 UTC, menos 1 ms
    ],
)
def test_last_closed_ms_per_interval(interval: str, expected: int) -> None:
    """El limite de la ultima vela cerrada respeta la duracion y la alineacion de cada intervalo."""
    assert cli._last_closed_ms(interval, NOW_MS) == expected


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1m", NOW_MS + 419_999),  # 12:06:59.999
        ("5m", NOW_MS + 299_999),  # 12:04:59.999
        ("15m", NOW_MS - 1),  # 11:59:59.999
        ("1h", NOW_MS - 1),
        ("1d", NOW_MS - 12 * 60 * 60 * 1000 - 1),  # 2026-10-08T23:59:59.999
        ("1w", START_MS + 277 * DAY_MS - 1),  # domingo 2026-10-04T23:59:59.999
    ],
)
def test_main_end_is_last_closed_candle_mid_interval(
    spy: _Spy, monkeypatch: pytest.MonkeyPatch, interval: str, expected: int
) -> None:
    """Con un ahora a mitad de vela (12:07:31.456), `update_symbol` recibe la ultima cerrada."""
    now = NOW + timedelta(minutes=7, seconds=31, milliseconds=456)
    monkeypatch.setattr(cli, "_utc_now", lambda: now)

    assert cli.main(["--symbol", "BTCUSDT", "--interval", interval]) == 0

    (args, _) = spy.calls[0]
    assert args[3] == expected
    # El inicio por defecto conserva el ms exacto de "ahora" menos 30 dias.
    assert args[2] == NOW_MS + 451_456 - 30 * DAY_MS


def test_variable_length_interval_is_not_clipped_and_warns(
    spy: _Spy, logs: _ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un intervalo sin duracion fija (1M) no se recorta, avisa y no falla."""
    monkeypatch.setattr(cli, "_utc_now", lambda: NOW + timedelta(seconds=30))

    assert cli.main(["--symbol", "BTCUSDT", "--interval", "1M"]) == 0

    assert spy.calls[0][0][3] == NOW_MS + 30_000
    assert any("No se recorta" in message for message in logs.messages)


def test_to_ms_keeps_exact_milliseconds_with_microseconds() -> None:
    """La conversion a ms no pierde 1 ms por el redondeo de coma flotante."""
    moment = datetime(2026, 10, 9, 12, 0, 0, 123_000, tzinfo=UTC)

    assert cli._to_ms(moment) == NOW_MS + 123


def test_symbol_is_required(spy: _Spy) -> None:
    """Sin `--symbol` argparse aborta con codigo distinto de 0 y no descarga."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code != 0
    assert spy.calls == []


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """`--help` muestra la ayuda con el formato de fecha y sale con codigo 0."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--symbol" in out
    assert "YYYY-MM-DD" in out
