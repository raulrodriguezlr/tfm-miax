"""Cliente HTTP base de la ingesta: timeout y reintentos con backoff ante fallos transitorios."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Self

import requests

from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5

# Fallos de red que merece la pena reintentar; el resto de excepciones se propagan.
_TRANSIENT_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


class HttpError(Exception):
    """Error base del cliente HTTP de la ingesta."""


class ClientError(HttpError):
    """Respuesta 4xx: error del cliente, no recuperable, no se reintenta."""

    def __init__(self, status_code: int, url: str) -> None:
        """Guarda el codigo y la URL que fallaron."""
        super().__init__(f"HTTP {status_code} en {url}: error no recuperable, no se reintenta")
        self.status_code = status_code
        self.url = url


class RetriesExhaustedError(HttpError):
    """Se agotaron los reintentos ante un fallo transitorio."""


class HttpClient:
    """Cliente GET sobre `requests` con timeout y reintentos con backoff exponencial."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        """Configura timeout (s), reintentos tras el primer intento y base del backoff (s)."""
        if timeout <= 0:
            raise ValueError(f"timeout debe ser > 0, recibido {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries debe ser >= 0, recibido {max_retries}")
        if backoff_base < 0:
            raise ValueError(f"backoff_base debe ser >= 0, recibido {backoff_base}")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._session = requests.Session()

    def close(self) -> None:
        """Cierra la sesion de requests y libera sus conexiones."""
        self._session.close()

    def __enter__(self) -> Self:
        """Permite usar el cliente como context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Cierra la sesion al salir del bloque `with`."""
        self.close()

    def backoff_delay(self, retry: int) -> float:
        """Espera en segundos antes del reintento `retry` (1, 2, ...): base * 2**(retry - 1)."""
        # Determinista y sin jitter: depende solo del numero de reintento, no del reloj.
        return self.backoff_base * 2 ** (retry - 1)

    def get(self, url: str, params: Mapping[str, Any] | None = None) -> requests.Response:
        """Hace un GET y devuelve la respuesta; reintenta 5xx, timeouts y fallos de conexion."""
        total_attempts = self.max_retries + 1
        last_failure = ""
        for attempt in range(1, total_attempts + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except _TRANSIENT_EXCEPTIONS as exc:
                last_failure = f"{type(exc).__name__}: {exc}"
                cause: Exception = exc
            else:
                status = response.status_code
                if 400 <= status < 500:
                    # 429/418 con Retry-After se tratan aparte en MIAX-017.
                    raise ClientError(status, url)
                if status < 500:
                    return response
                last_failure = f"HTTP {status}"
                cause = HttpError(last_failure)

            if attempt == total_attempts:
                raise RetriesExhaustedError(
                    f"GET {url} fallo tras {total_attempts} intentos: {last_failure}"
                ) from cause

            delay = self.backoff_delay(attempt)
            logger.warning(
                "GET %s fallo (%s); intento %d/%d, reintento en %.2fs",
                url,
                last_failure,
                attempt,
                total_attempts,
                delay,
            )
            time.sleep(delay)

        raise AssertionError("inalcanzable: el bucle devuelve o lanza")  # pragma: no cover
