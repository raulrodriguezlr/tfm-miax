"""Cliente HTTP base de la ingesta: timeout y reintentos con backoff ante fallos transitorios."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Self

import requests

from miax.ingest.ratelimit import RateLimiter
from miax.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_MAX_RETRY_AFTER = 300.0

# Fallos de red que merece la pena reintentar; el resto de excepciones se propagan.
_TRANSIENT_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

# 429 = limite de peso excedido; 418 = IP baneada temporalmente. Ambos exigen esperar.
_RATE_LIMIT_STATUSES = frozenset({429, 418})

# Peso usado en el ultimo minuto segun Binance; sirve para resincronizar el limitador.
_USED_WEIGHT_HEADER = "X-MBX-USED-WEIGHT-1M"


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


def _retry_after_seconds(response: requests.Response, now: datetime | None = None) -> float | None:
    """Segundos de `Retry-After` (entero o fecha HTTP), o None si falta o no se entiende."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        return float(value)
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        # Una fecha HTTP sin zona se interpreta como UTC, igual que GMT.
        target = target.replace(tzinfo=UTC)
    # Solo mira el reloj actual para esperar hacia delante; no usa datos de mercado.
    reference = now if now is not None else datetime.now(tz=UTC)
    return max(0.0, (target - reference).total_seconds())


class HttpClient:
    """Cliente GET sobre `requests` con timeout y reintentos con backoff exponencial."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Configura timeout, reintentos, backoff, tope de Retry-After y limitador opcional."""
        if timeout <= 0:
            raise ValueError(f"timeout debe ser > 0, recibido {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries debe ser >= 0, recibido {max_retries}")
        if backoff_base < 0:
            raise ValueError(f"backoff_base debe ser >= 0, recibido {backoff_base}")
        if max_retry_after <= 0:
            raise ValueError(f"max_retry_after debe ser > 0, recibido {max_retry_after}")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_retry_after = max_retry_after
        self.rate_limiter = rate_limiter
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

    def _sync_rate_limiter(self, response: requests.Response) -> None:
        """Pasa al limitador el peso usado que informa Binance, si la cabecera es valida."""
        if self.rate_limiter is None:
            return
        value = response.headers.get(_USED_WEIGHT_HEADER)
        if value is None:
            return
        try:
            used = int(value)
        except ValueError:
            return
        if used >= 0:
            self.rate_limiter.sync_used_weight(used)

    def get(
        self, url: str, params: Mapping[str, Any] | None = None, weight: int = 1
    ) -> requests.Response:
        """Hace un GET y devuelve la respuesta; reintenta 5xx, 429/418, timeouts y fallos de red."""
        total_attempts = self.max_retries + 1
        last_failure = ""
        for attempt in range(1, total_attempts + 1):
            retry_after: float | None = None
            if self.rate_limiter is not None:
                # Cada intento, reintentos incluidos, es una peticion que consume peso.
                self.rate_limiter.acquire(weight)
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except _TRANSIENT_EXCEPTIONS as exc:
                last_failure = f"{type(exc).__name__}: {exc}"
                cause: Exception = exc
            else:
                self._sync_rate_limiter(response)
                status = response.status_code
                if status in _RATE_LIMIT_STATUSES:
                    # Respetar Retry-After es obligatorio: ignorarlo en un 418 alarga el baneo.
                    retry_after = _retry_after_seconds(response)
                    last_failure = f"HTTP {status}"
                    cause = HttpError(last_failure)
                elif 400 <= status < 500:
                    raise ClientError(status, url)
                elif status < 500:
                    return response
                else:
                    last_failure = f"HTTP {status}"
                    cause = HttpError(last_failure)

            if attempt == total_attempts:
                raise RetriesExhaustedError(
                    f"GET {url} fallo tras {total_attempts} intentos: {last_failure}"
                ) from cause

            if retry_after is not None:
                delay = min(retry_after, self.max_retry_after)
            else:
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
