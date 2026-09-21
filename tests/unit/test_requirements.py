"""Comprueba que requirements.txt fija todas las versiones con == exacto, sin rangos."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REQUIREMENTS_PATH = Path(__file__).resolve().parents[2] / "requirements.txt"

# Nombre de paquete, extras opcionales entre corchetes, == exacto y version que
# empieza por digito (admite etiqueta local como +pt212cpu), con marcador de
# entorno opcional tras ';'. Se usa con fullmatch, sin anclas ^/$ propias.
EXACT_PIN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<extras>\[[A-Za-z0-9_,-]+\])?"
    r"=="
    r"(?P<version>\d[A-Za-z0-9.+_-]*)"
    r"(?:\s*;.*)?"
)


def read_requirement_lines(path: Path) -> list[str]:
    """Lee un requirements.txt y devuelve solo las lineas de paquete, sin comentarios ni opciones de pip."""
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        lines.append(line)
    return lines


VALID_PINS: list[str] = [
    "pandas==3.0.6",
    "torch-scatter==2.1.2+pt212cpu",
    'pyyaml==6.0.3; sys_platform == "win32"',
    "uvicorn[standard]==0.30.0",
]

INVALID_PINS: list[str] = [
    "pandas>=3.0",
    "pandas",
    "numpy~=2.5",
    "ruff==0.16.*",
    "pytest===9.1.1",
    "scipy!=1.18.0",
    "numpy==2.5.3,<3",
    "matplotlib<=3.11.2",
]


@pytest.mark.parametrize("pin", VALID_PINS)
def test_exact_pin_accepts_valid_pins(pin: str) -> None:
    """EXACT_PIN reconoce pines con version exacta, con o sin extras y marcador de entorno."""
    assert EXACT_PIN.fullmatch(pin)


@pytest.mark.parametrize("pin", INVALID_PINS)
def test_exact_pin_rejects_ranges_and_missing_version(pin: str) -> None:
    """EXACT_PIN rechaza rangos, operadores distintos de == y nombres sin version."""
    assert not EXACT_PIN.fullmatch(pin)


def test_requirements_file_has_at_least_one_requirement() -> None:
    """El requirements.txt real declara al menos una dependencia."""
    lines = read_requirement_lines(REQUIREMENTS_PATH)

    assert len(lines) > 0


def test_all_requirements_are_exact_pins() -> None:
    """Ninguna linea real de requirements.txt debe tener rango: todas van con == exacto."""
    lines = read_requirement_lines(REQUIREMENTS_PATH)

    offending = [line for line in lines if not EXACT_PIN.fullmatch(line)]

    assert not offending, f"Lineas sin version exacta en requirements.txt: {offending}"
