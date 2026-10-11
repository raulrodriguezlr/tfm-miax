"""Tests de miax.utils.logging: formato comun, nivel configurable e idempotencia."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import pytest

from miax.utils import get_logger
from miax.utils import logging as miax_logging

LINE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \| (?P<level>[A-Z]+) \| (?P<name>\S+) \| (?P<msg>.*)$"
)


def miax_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Devuelve solo los handlers de miax: pytest cuelga los suyos en loggers sin propagacion."""
    return [h for h in logger.handlers if getattr(h, miax_logging._HANDLER_FLAG, False)]


@pytest.fixture
def logger_name() -> Iterator[str]:
    """Da un nombre de logger unico y limpia sus handlers al acabar el test."""
    name = "miax.test_logging.unit"
    yield name
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_module_does_not_shadow_stdlib() -> None:
    """El modulo miax.utils.logging es distinto de la stdlib y esta usa la stdlib."""
    assert miax_logging is not logging
    assert miax_logging.logging is logging


def test_format_has_utc_timestamp_level_name_and_message(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """La linea sale con timestamp UTC, nivel, nombre y mensaje en el formato comun."""
    logger = get_logger(logger_name)

    logger.info("hola %s", "mundo")

    line = capsys.readouterr().err.strip()
    match = LINE_PATTERN.match(line)
    assert match is not None, line
    assert match["level"] == "INFO"
    assert match["name"] == logger_name
    assert match["msg"] == "hola mundo"


def test_timestamp_is_utc(logger_name: str) -> None:
    """El formateador convierte a UTC, no a la hora local."""
    logger = get_logger(logger_name)
    formatter = miax_handlers(logger)[0].formatter
    assert formatter is not None
    record = logging.LogRecord(logger_name, logging.INFO, __file__, 1, "x", None, None)
    record.created = 86400.0  # 1970-01-02 00:00:00 UTC

    assert formatter.format(record).startswith("1970-01-02T00:00:00Z")


def test_default_level_is_info(logger_name: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Por defecto se ignora DEBUG y se emite INFO."""
    logger = get_logger(logger_name)

    logger.debug("oculto")
    logger.info("visible")

    err = capsys.readouterr().err
    assert "oculto" not in err
    assert "visible" in err


@pytest.mark.parametrize("level", ["WARNING", logging.WARNING])
def test_level_is_respected_as_name_or_int(
    logger_name: str, capsys: pytest.CaptureFixture[str], level: int | str
) -> None:
    """El nivel se acepta como texto o entero y filtra los mensajes por debajo."""
    logger = get_logger(logger_name, level=level)

    logger.info("oculto")
    logger.warning("visible")

    err = capsys.readouterr().err
    assert "oculto" not in err
    assert "visible" in err
    assert logger.level == logging.WARNING


def test_debug_level_emits_debug(logger_name: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Con nivel DEBUG los mensajes DEBUG si salen."""
    logger = get_logger(logger_name, level="DEBUG")

    logger.debug("detalle")

    assert "detalle" in capsys.readouterr().err


def test_second_call_does_not_duplicate_handlers(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Llamar dos veces con el mismo nombre deja un solo handler y cada mensaje sale una vez."""
    first = get_logger(logger_name)
    second = get_logger(logger_name)

    second.info("una vez")

    assert first is second
    assert len(miax_handlers(second)) == 1
    assert capsys.readouterr().err.count("una vez") == 1


def test_second_call_updates_level(logger_name: str) -> None:
    """Una segunda llamada con otro nivel cambia el nivel sin anadir handlers."""
    get_logger(logger_name, level="INFO")
    logger = get_logger(logger_name, level="ERROR")

    assert logger.level == logging.ERROR
    assert len(miax_handlers(logger)) == 1
