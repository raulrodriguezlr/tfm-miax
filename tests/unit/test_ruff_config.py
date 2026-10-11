"""Comprueba que la configuracion de ruff en pyproject.toml aplica las reglas esperadas."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sin docstring de funcion y sin type hints: dispara ANN, D y NPY, y DTZ005 por
# usar datetime.now() sin zona horaria. No se guarda en disco: entra por stdin.
PROBE_SNIPPET = """\
import datetime

import numpy as np


def f(x):
    np.random.seed(0)
    return datetime.datetime.now()
"""


def lint_codes(code: str, filename: str) -> set[str]:
    """Ejecuta ruff check sobre code por stdin y devuelve el conjunto de codigos de regla."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format",
            "json",
            "--stdin-filename",
            filename,
            "-",
        ],
        input=code,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    try:
        violations = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"Salida de ruff no es JSON valido.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from None

    return {item["code"] for item in violations}


def test_project_rules_are_active() -> None:
    """Bajo src/miax se exigen type hints, docstrings, default_rng y fechas con zona horaria."""
    # Sin guion bajo inicial: ruff/pydocstyle tratan un modulo "_probe.py" como
    # privado y cambian ANN201 por ANN202 y suprimen D103, lo que falsearia el test.
    codes = lint_codes(PROBE_SNIPPET, "src/miax/probe.py")

    assert {"ANN001", "ANN201", "D103", "NPY002", "DTZ005"} <= codes


def test_notebooks_skip_types_and_docstrings() -> None:
    """Bajo notebooks no se exigen type hints ni docstrings, pero si default_rng."""
    codes = lint_codes(PROBE_SNIPPET, "notebooks/probe.py")

    # DTZ (fechas naive) empieza por "D" pero no es una regla de pydocstyle: se
    # descarta filtrando solo "D" seguida de digito, que es como nombra ruff a D.
    ann_or_doc_codes = {
        code for code in codes if code.startswith("ANN") or (code[0] == "D" and code[1].isdigit())
    }

    assert not ann_or_doc_codes
    assert "NPY002" in codes
