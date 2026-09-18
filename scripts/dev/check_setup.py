"""Comprueba que el entorno y las skills del equipo estan listos. No instala nada sin permiso."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Skills obligatorias segun CLAUDE.md, con el comando que las instala.
REQUIRED_SKILLS: dict[str, str] = {
    "senior-engineer": "claude plugin install anthropic-skills",
    "frontend-design": "claude plugin install anthropic-skills",
}

REQUIRED_AGENTS = ["miax-coder", "miax-auditor", "miax-fixer", "miax-tester"]

CORE_PACKAGES = ["pandas", "numpy", "statsmodels", "pyarrow", "pytest"]


def _skill_dirs() -> list[Path]:
    """Devuelve las rutas donde Claude Code busca skills, existan o no."""
    home = Path.home() / ".claude"
    return [home / "skills", home / "plugins", REPO / ".claude" / "skills"]


def find_missing_skills() -> list[str]:
    """Busca cada skill obligatoria por nombre de carpeta en las rutas conocidas."""
    missing = []
    for skill in REQUIRED_SKILLS:
        found = any(
            root.exists() and any(p.name == skill for p in root.rglob(skill))
            for root in _skill_dirs()
        )
        if not found:
            missing.append(skill)
    return missing


def find_missing_agents() -> list[str]:
    """Comprueba que los cuatro subagentes del flujo estan definidos en el repo."""
    agents_dir = REPO / ".claude" / "agents"
    return [a for a in REQUIRED_AGENTS if not (agents_dir / f"{a}.md").exists()]


def find_missing_packages() -> list[str]:
    """Comprueba las dependencias minimas para arrancar el pipeline."""
    return [p for p in CORE_PACKAGES if importlib.util.find_spec(p) is None]


def current_branch() -> str:
    """Devuelve la rama actual, o cadena vacia si git no esta disponible."""
    if shutil.which("git") is None:
        return ""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def main() -> int:
    problems: list[str] = []

    missing_pkgs = find_missing_packages()
    if missing_pkgs:
        problems.append(
            f"Faltan paquetes: {', '.join(missing_pkgs)}\n"
            f"   Instalalos con:  pip install -r requirements.txt"
        )

    missing_skills = find_missing_skills()
    if missing_skills:
        cmds = sorted({REQUIRED_SKILLS[s] for s in missing_skills})
        problems.append(
            f"Faltan skills obligatorias: {', '.join(missing_skills)}\n"
            f"   Instalalas con:  {' && '.join(cmds)}\n"
            f"   Sin ellas el codigo de los tres no se parecera. Ver CLAUDE.md seccion 3."
        )

    missing_agents = find_missing_agents()
    if missing_agents:
        problems.append(
            f"Faltan subagentes en .claude/agents: {', '.join(missing_agents)}\n"
            f"   Haz git pull: vienen versionados en el repo."
        )

    branch = current_branch()
    if branch == "main":
        problems.append(
            "Estas en main. Nunca se escribe en main.\n"
            "   Crea la rama del ticket:  git checkout -b feature/MIAX-XXX"
        )

    print("\n" + "=" * 64)
    print("  TFM MIAX - comprobacion de entorno")
    print("=" * 64)

    if not problems:
        print("  Todo listo. Coge un ticket de docs/project/BACKLOG.md.")
        print("  Recuerda: plan en Opus, ejecucion con los subagentes Sonnet.")
        print("=" * 64 + "\n")
        return 0

    for i, p in enumerate(problems, 1):
        print(f"\n  [{i}] {p}")
    print("\n" + "=" * 64 + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
