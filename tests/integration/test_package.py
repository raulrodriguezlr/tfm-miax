"""Comprueba que el paquete miax es instalable y se importa desde fuera del repo."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBPACKAGES: tuple[str, ...] = (
    "ingest",
    "features",
    "graph",
    "models",
    "eval",
    "viz",
    "utils",
)


def test_import_miax_from_outside_repo(tmp_path: Path) -> None:
    """import miax funciona desde un directorio fuera del repo tras pip install -e ."""
    result = subprocess.run(
        [sys.executable, "-c", "import miax; print(miax.__file__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "No se pudo importar miax desde fuera del repo; ejecuta 'pip install -e .' "
        f"primero.\nstderr:\n{result.stderr}"
    )

    printed_path = Path(result.stdout.strip()).resolve()
    expected_root = REPO_ROOT / "src" / "miax"

    assert printed_path.is_relative_to(expected_root), (
        f"{printed_path} no esta dentro de {expected_root}: el modo editable "
        "deberia apuntar al codigo del repo, no a una copia."
    )


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name: str) -> None:
    """Cada subpaquete de miax se importa sin errores y conserva su nombre completo."""
    module = importlib.import_module(f"miax.{name}")

    assert module.__name__ == f"miax.{name}"
