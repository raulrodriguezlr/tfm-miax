"""Comprueba que la configuracion de pytest en pyproject.toml es la esperada."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def read_pytest_ini_options() -> dict[str, object]:
    """Devuelve la tabla [tool.pytest.ini_options] del pyproject.toml del repo."""
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data["tool"]["pytest"]["ini_options"]


def test_testpaths_covers_tests_directory() -> None:
    """La clave testpaths apunta a tests/, de donde cuelgan unit e integration."""
    options = read_pytest_ini_options()

    assert "tests" in options["testpaths"]


def test_addopts_enforces_strict_markers() -> None:
    """La clave addopts trae --strict-markers, para que un marcador mal escrito falle."""
    options = read_pytest_ini_options()

    assert "--strict-markers" in options["addopts"]


def test_collects_unit_and_integration_tests() -> None:
    """El comando pytest --collect-only encuentra tests en tests/unit y tests/integration."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"La coleccion de tests fallo.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "tests/unit" in result.stdout or "tests\\unit" in result.stdout
    assert "tests/integration" in result.stdout or "tests\\integration" in result.stdout
