"""Comprueba que la guardia de red de tests/conftest.py bloquea conexiones salientes."""

from __future__ import annotations

import socket

import pytest


def test_connect_is_blocked() -> None:
    """socket.socket().connect(...) lanza RuntimeError y nombra la regla de red."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Ningun test toca la red"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


def test_connect_ex_is_blocked() -> None:
    """socket.socket().connect_ex(...) tambien lanza RuntimeError, no solo connect."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Ningun test toca la red"):
            sock.connect_ex(("127.0.0.1", 9))
    finally:
        sock.close()


def test_creating_socket_without_connecting_still_works() -> None:
    """Crear un socket sin conectarlo sigue funcionando: solo se bloquea la conexion."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()
