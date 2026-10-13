"""Comprueba que la guardia de red de tests/conftest.py bloquea conexiones salientes."""

from __future__ import annotations

import os
import socket
import subprocess
import sys

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


def test_proxy_variables_point_to_unreachable_address() -> None:
    """Dentro de un test, los proxies apuntan a una direccion local inalcanzable, sin NO_PROXY."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert os.environ[name] == "http://127.0.0.1:9"
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ


def test_subprocess_using_requests_cannot_reach_the_internet() -> None:
    """Un subproceso con `requests` falla contra el proxy inalcanzable, sin llegar a internet."""
    code = (
        "import sys, requests\n"
        "try:\n"
        "    requests.get('https://api.binance.com/api/v3/ping', timeout=2)\n"
        "except requests.exceptions.ProxyError:\n"
        "    sys.exit(42)\n"
        "except requests.exceptions.RequestException as error:\n"
        "    sys.exit(42 if '127.0.0.1:9' in str(error) else 43)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )

    # 42 = fallo contra el proxy 127.0.0.1:9 (se aplica); 0 = la peticion salio; 43 u otro = fallo
    # ajeno al proxy (p. ej. DNS), que no demuestra nada.
    assert result.returncode == 42, result.stderr
