"""Fixtures compartidas por toda la suite: aqui vive la guardia de red."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

_NETWORK_RULE_MESSAGE = (
    "Ningun test toca la red (CLAUDE.md): socket.connect bloqueado por la guardia "
    "de tests/conftest.py. Usa un fixture que simule la respuesta en vez de conectar."
)


def _blocked_connect(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(_NETWORK_RULE_MESSAGE)


def _blocked_connect_ex(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(_NETWORK_RULE_MESSAGE)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Bloquea socket.connect/connect_ex en cada test; monkeypatch restaura al acabar."""
    # Solo se bloquea la conexion saliente, no la creacion de sockets: tests como
    # test_package.py y test_ruff_config.py lanzan subprocesos y eso debe seguir
    # funcionando.
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect_ex)
    # Limitacion conocida: un subproceso es otro proceso y no hereda este parche,
    # asi que la red sigue accesible dentro de subprocesos propios (MIAX-032
    # tendra que cuidarlo con sus propios fixtures).
    yield
