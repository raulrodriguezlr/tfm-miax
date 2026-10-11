"""Limitador proactivo por peso de peticion: frena antes de que Binance responda 429/418."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from miax.utils.logging import get_logger

logger = get_logger(__name__)

# Conservador: Binance spot da 6000 de peso por minuto e IP (1200 en versiones antiguas de la API).
# Ajustable desde configs/ en el futuro.
DEFAULT_MAX_WEIGHT = 1200
DEFAULT_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Ventana deslizante por peso: espera lo justo para no superar `max_weight` por ventana."""

    def __init__(
        self,
        max_weight: int = DEFAULT_MAX_WEIGHT,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configura el presupuesto de peso, la ventana en segundos y el reloj (inyectable)."""
        if max_weight <= 0:
            raise ValueError(f"max_weight debe ser > 0, recibido {max_weight}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds debe ser > 0, recibido {window_seconds}")
        self.max_weight = max_weight
        self.window_seconds = window_seconds
        self._clock = clock
        self._entries: deque[tuple[float, int]] = deque()
        self._used = 0

    @property
    def used_weight(self) -> int:
        """Peso consumido dentro de la ventana actual segun la cuenta interna."""
        self._prune(self._clock())
        return self._used

    def _prune(self, now: float) -> None:
        """Descarta los consumos que ya han salido de la ventana en el instante `now`."""
        # Solo mira el pasado: un consumo sale cuando ts + window <= now, nunca se adelanta.
        # Se compara contra el instante de expiracion (el mismo que calcula `_expiry_target`)
        # para que `now == ts + window` pode siempre, sin error de coma flotante al restar.
        while self._entries and self._entries[0][0] + self.window_seconds <= now:
            _, weight = self._entries.popleft()
            self._used -= weight

    def _record(self, now: float, weight: int) -> None:
        """Anota un consumo de `weight` en `now`, sin fecharlo antes que la ultima entrada."""
        # Mantiene la deque ordenada por timestamp (`_prune` y `_expiry_target` lo asumen): una
        # entrada previa puede estar fechada por delante del reloj si el sleep desperto antes de
        # `target`. Fechar tarde es conservador: nunca se infra-cuenta el peso de la ventana.
        if self._entries:
            now = max(now, self._entries[-1][0])
        self._entries.append((now, weight))
        self._used += weight

    def _expiry_target(self, weight: int) -> float:
        """Instante exacto en que, al salir consumos antiguos, cabe `weight` en el presupuesto."""
        remaining = self._used
        for ts, entry_weight in self._entries:
            remaining -= entry_weight
            if remaining + weight <= self.max_weight:
                return ts + self.window_seconds
        raise AssertionError("weight <= max_weight garantiza que se sale antes")  # pragma: no cover

    def acquire(self, weight: int = 1) -> None:
        """Reserva `weight` antes de una peticion; espera si la ventana no tiene hueco."""
        if weight <= 0:
            raise ValueError(f"weight debe ser > 0, recibido {weight}")
        if weight > self.max_weight:
            raise ValueError(f"weight {weight} supera max_weight {self.max_weight}")
        now = self._clock()
        self._prune(now)
        if self._used + weight > self.max_weight:
            target = self._expiry_target(weight)
            wait = max(0.0, target - now)
            logger.info(
                "Limitador: peso %d + %d supera el presupuesto %d por %.0fs; espero %.2fs",
                self._used,
                weight,
                self.max_weight,
                self.window_seconds,
                wait,
            )
            time.sleep(wait)
            # Se avanza al menos hasta el instante exacto de expiracion (sin sumar la espera, que
            # puede quedar 1 ulp por debajo) aunque el reloj no se haya movido.
            now = max(self._clock(), target)
            self._prune(now)
        self._record(now, weight)

    def sync_used_weight(self, used: int) -> None:
        """Alinea la cuenta con el peso que reporta Binance, sin bajar nunca de la cuenta propia."""
        if used < 0:
            raise ValueError(f"used debe ser >= 0, recibido {used}")
        now = self._clock()
        self._prune(now)
        # Solo se sube: la cuenta propia ya incluye peticiones en vuelo que el servidor aun no
        # ha contado, asi que bajarla podria permitir pasarse. El exceso se fecha ahora, lo que
        # es prudente (tarda como mucho una ventana completa en salir).
        if used > self._used:
            self._record(now, used - self._used)
