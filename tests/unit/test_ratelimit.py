"""Tests del limitador por peso: ventana deslizante, espera exacta y sincronizacion, sin esperas."""

from __future__ import annotations

import logging
import math

import pytest

from miax.ingest import RateLimiter


class _Clock:
    """Reloj falso que solo avanza cuando el test (o un sleep falso) lo mueve."""

    def __init__(self) -> None:
        """Empieza en un instante arbitrario distinto de cero."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Devuelve el instante actual."""
        return self.now


@pytest.fixture
def clock() -> _Clock:
    """Reloj controlado por el test."""
    return _Clock()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> list[float]:
    """Sustituye time.sleep: registra la espera y adelanta el reloj falso."""
    recorded: list[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)
        clock.now += seconds

    monkeypatch.setattr("miax.ingest.ratelimit.time.sleep", fake_sleep)
    return recorded


def test_within_budget_does_not_sleep(clock: _Clock, sleeps: list[float]) -> None:
    """Mientras el peso total quepa en la ventana no hay esperas."""
    limiter = RateLimiter(max_weight=10, window_seconds=60.0, clock=clock)
    for _ in range(5):
        limiter.acquire(2)
    assert sleeps == []
    assert limiter.used_weight == 10


def test_exceeding_budget_sleeps_exactly_what_is_missing(
    clock: _Clock, sleeps: list[float]
) -> None:
    """Al pasarse espera hasta que el consumo mas antiguo necesario sale de la ventana."""
    limiter = RateLimiter(max_weight=10, window_seconds=60.0, clock=clock)
    limiter.acquire(6)  # t=1000
    clock.now += 10
    limiter.acquire(4)  # t=1010, ya al limite
    clock.now += 5  # t=1015
    limiter.acquire(5)  # hace falta que salga el primer consumo (peso 6) en t=1060
    assert sleeps == [pytest.approx(45.0)]
    assert limiter.used_weight == 9  # quedan el de peso 4 y el nuevo de peso 5


def test_sleep_waits_for_enough_entries_to_expire(clock: _Clock, sleeps: list[float]) -> None:
    """Si un solo consumo antiguo no libera bastante, espera al siguiente."""
    limiter = RateLimiter(max_weight=10, window_seconds=60.0, clock=clock)
    limiter.acquire(3)  # t=1000
    clock.now += 10
    limiter.acquire(3)  # t=1010
    clock.now += 10
    limiter.acquire(4)  # t=1020, total 10
    clock.now += 10  # t=1030
    limiter.acquire(5)  # liberar 3 no basta (7+5>10): hay que esperar a que salga el de t=1010
    assert sleeps == [pytest.approx(40.0)]  # hasta t=1070


def test_window_frees_when_clock_advances(clock: _Clock, sleeps: list[float]) -> None:
    """Al avanzar el reloj una ventana entera el peso vuelve a quedar libre."""
    limiter = RateLimiter(max_weight=10, window_seconds=60.0, clock=clock)
    limiter.acquire(10)
    assert limiter.used_weight == 10
    # Valores binarios exactos (59.5 + 0.5 == 60.0 sin error de coma flotante): borde exacto.
    clock.now += 59.5
    assert limiter.used_weight == 10
    clock.now += 0.5
    assert limiter.used_weight == 0
    limiter.acquire(10)
    assert sleeps == []


def test_sleep_that_does_not_move_the_clock_does_not_hang(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """Con un sleep que no adelanta el reloj, acquire termina igualmente tras una espera."""
    recorded: list[float] = []
    monkeypatch.setattr("miax.ingest.ratelimit.time.sleep", recorded.append)
    limiter = RateLimiter(max_weight=5, window_seconds=60.0, clock=clock)
    limiter.acquire(5)
    limiter.acquire(3)
    assert recorded == [pytest.approx(60.0)]
    assert limiter.used_weight == 3


def test_acquire_never_exceeds_budget_with_float_rounding(
    clock: _Clock, sleeps: list[float]
) -> None:
    """Tras esperar, el consumo expirado se poda aunque ts + window no sea exacto en flotante."""
    clock.now = 926473.2240554473
    limiter = RateLimiter(max_weight=19, window_seconds=0.1, clock=clock)
    limiter.acquire(14)
    limiter.acquire(11)  # 14 + 11 > 19: hay que esperar a que expire el primero
    assert len(sleeps) == 1
    # Se lee la cuenta interna: `used_weight` podaria de nuevo y enmascararia el fallo.
    assert limiter._used <= limiter.max_weight
    assert limiter._used == 11


def test_entries_stay_ordered_when_sleep_wakes_one_ulp_early(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """Si el sleep despierta antes de `target`, la deque sigue ordenada y no se excede el peso."""

    def early_sleep(seconds: float) -> None:
        # Despierta 1 ulp antes del instante de expiracion (el reloj queda por debajo de target).
        clock.now = math.nextafter(clock.now + seconds, 0.0)

    monkeypatch.setattr("miax.ingest.ratelimit.time.sleep", early_sleep)
    limiter = RateLimiter(max_weight=4, window_seconds=10.0, clock=clock)
    limiter.acquire(4)  # t=1000
    clock.now = 1005.0
    limiter.acquire(1)  # espera hasta target=1010; el reloj queda en 1009.999...
    limiter.acquire(2)  # cabe, no duerme: no debe fecharse antes que la entrada anterior
    limiter.acquire(3)  # duerme; debe podar las dos entradas anteriores
    timestamps = [ts for ts, _ in limiter._entries]
    assert timestamps == sorted(timestamps)
    # Se lee la cuenta interna: `used_weight` podaria de nuevo y enmascararia el fallo.
    assert limiter._used <= limiter.max_weight
    assert limiter._used == 3


def test_sync_raises_count_to_server_value(clock: _Clock, sleeps: list[float]) -> None:
    """Si el servidor informa de mas peso del contado, la cuenta sube y obliga a esperar antes."""
    limiter = RateLimiter(max_weight=10, window_seconds=60.0, clock=clock)
    limiter.acquire(2)
    limiter.sync_used_weight(9)
    assert limiter.used_weight == 9
    limiter.acquire(2)
    assert sleeps == [pytest.approx(60.0)]


def test_sync_never_lowers_count(clock: _Clock) -> None:
    """Un valor del servidor menor que la cuenta propia no la reduce."""
    limiter = RateLimiter(max_weight=10, window_seconds=60.0, clock=clock)
    limiter.acquire(8)
    limiter.sync_used_weight(3)
    assert limiter.used_weight == 8


def test_sync_rejects_negative(clock: _Clock) -> None:
    """Un peso negativo es un error de uso."""
    with pytest.raises(ValueError, match="used"):
        RateLimiter(clock=clock).sync_used_weight(-1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_weight": 0}, "max_weight"),
        ({"max_weight": -5}, "max_weight"),
        ({"window_seconds": 0}, "window_seconds"),
        ({"window_seconds": -1.0}, "window_seconds"),
    ],
)
def test_constructor_validation(kwargs: dict[str, float], match: str) -> None:
    """Presupuesto y ventana deben ser positivos."""
    with pytest.raises(ValueError, match=match):
        RateLimiter(**kwargs)  # type: ignore[arg-type]


def test_acquire_validates_weight(clock: _Clock) -> None:
    """Peso no positivo o mayor que el presupuesto lanza ValueError en vez de bloquearse."""
    limiter = RateLimiter(max_weight=5, clock=clock)
    with pytest.raises(ValueError, match="weight"):
        limiter.acquire(0)
    with pytest.raises(ValueError, match="supera"):
        limiter.acquire(6)


def test_wait_is_logged(clock: _Clock, sleeps: list[float]) -> None:
    """Cuando espera, el limitador lo avisa con el motivo."""
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    # El logger de miax no propaga y escribe en el stderr original: se engancha un handler propio.
    handler = _Capture()
    ratelimit_logger = logging.getLogger("miax.ingest.ratelimit")
    ratelimit_logger.addHandler(handler)
    try:
        limiter = RateLimiter(max_weight=2, window_seconds=60.0, clock=clock)
        limiter.acquire(2)
        limiter.acquire(1)
    finally:
        ratelimit_logger.removeHandler(handler)
    assert len(messages) == 1
    assert "supera el presupuesto" in messages[0]
    assert "60.00s" in messages[0]
