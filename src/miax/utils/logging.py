"""Logging comun del proyecto: mismo formato y nivel configurable en todos los modulos."""

from __future__ import annotations

# Import absoluto (el unico posible en Python 3): trae la stdlib, no este mismo fichero.
import logging
import time

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Marca que distingue el handler de miax de cualquier otro que se cuelgue al logger.
_HANDLER_FLAG = "_miax_handler"


def _make_formatter() -> logging.Formatter:
    """Crea el formateador comun, con la hora en UTC."""
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    formatter.converter = time.gmtime
    return formatter


def get_logger(name: str, level: int | str = "INFO") -> logging.Logger:
    """Devuelve el logger `name` con el formato comun y el nivel dado, sin duplicar handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Llamar de nuevo sobre el mismo logger solo actualiza el nivel: no apila handlers.
    if not any(getattr(handler, _HANDLER_FLAG, False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(_make_formatter())
        setattr(handler, _HANDLER_FLAG, True)
        logger.addHandler(handler)

    # Evita que el mensaje suba al logger raiz y salga dos veces si este tiene handlers.
    logger.propagate = False
    return logger
