"""Fixtures compartidas por toda la suite: aqui vive la guardia de red."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

_NETWORK_RULE_MESSAGE = (
    "Ningun test toca la red (CLAUDE.md): socket.connect bloqueado por la guardia "
    "de tests/conftest.py. Usa un fixture que simule la respuesta en vez de conectar."
)

_UNREACHABLE_PROXY = "http://127.0.0.1:9"  # el puerto 9 (discard) rechaza la conexion
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def _blocked_connect(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(_NETWORK_RULE_MESSAGE)


def _blocked_connect_ex(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(_NETWORK_RULE_MESSAGE)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Bloquea socket.connect/connect_ex y los proxies en cada test; monkeypatch lo restaura."""
    # Solo se bloquea la conexion saliente, no la creacion de sockets: tests como
    # test_package.py y test_ruff_config.py lanzan subprocesos y eso debe seguir
    # funcionando.
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect_ex)
    # Un subproceso es otro proceso y no hereda el parche de sockets. Las variables de proxy
    # si se heredan: apuntarlas a una direccion inalcanzable hace que un subproceso que use
    # `requests` falle al instante en local en vez de salir a internet (MIAX-032).
    # Limitacion conocida: no cubre sockets crudos (sin `requests`) dentro de subprocesos.
    for name in _PROXY_VARIABLES:
        monkeypatch.setenv(name, _UNREACHABLE_PROXY)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)  # sin excepciones que salten el proxy
    yield
