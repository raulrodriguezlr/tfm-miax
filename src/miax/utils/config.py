"""Carga tipada de configuracion desde YAML, validada contra un dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import get_type_hints

import yaml

# Raiz del repo: src/miax/utils/config.py esta a tres niveles de profundidad.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


class ConfigError(Exception):
    """Error al localizar, leer o validar un fichero de configuracion."""


def _resolve_config_path(path: str | Path) -> Path:
    """Busca el YAML como ruta absoluta, relativa a la raiz del repo o relativa a configs/."""
    candidate = Path(path)
    search_order = (
        [candidate] if candidate.is_absolute() else [REPO_ROOT / candidate, CONFIGS_DIR / candidate]
    )
    for option in search_order:
        if option.is_file():
            return option
    tried = ", ".join(str(option) for option in search_order)
    raise ConfigError(
        f"No se encuentra el fichero de configuracion '{path}'. Rutas probadas: {tried}."
    )


_SUPPORTED_FIELD_TYPES = (str, int, float, bool)


def _check_field_type(field_name: str, value: object, expected_type: type) -> None:
    """Comprueba que value sea del tipo esperado; solo admite str, int, float y bool."""
    if not isinstance(expected_type, type) or expected_type not in _SUPPORTED_FIELD_TYPES:
        raise ConfigError(
            f"El campo '{field_name}' usa el tipo '{expected_type}', que el validador "
            "todavia no soporta (solo str, int, float y bool)."
        )
    if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
        return  # YAML permite escribir un float como entero literal (p.ej. 1 en vez de 1.0).
    if isinstance(value, bool) and expected_type is not bool:
        raise ConfigError(
            f"El campo '{field_name}' deberia ser de tipo {expected_type.__name__}, pero es bool."
        )
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"El campo '{field_name}' deberia ser de tipo {expected_type.__name__}, "
            f"pero es {type(value).__name__}."
        )


def load_config[SchemaT](path: str | Path, schema: type[SchemaT]) -> SchemaT:
    """Lee un YAML tipado contra un dataclass; falla claro si falta o sobra un campo."""
    if not dataclasses.is_dataclass(schema):
        raise ConfigError(f"El esquema '{schema}' no es un dataclass.")

    resolved_path = _resolve_config_path(path)
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    raw = raw if raw is not None else {}
    if not isinstance(raw, dict):
        raise ConfigError(f"El YAML '{resolved_path}' no representa un mapeo de campos.")

    fields = {field.name: field for field in dataclasses.fields(schema)}
    type_hints = get_type_hints(schema)

    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(f"Campos desconocidos en '{resolved_path}': {unknown}.")

    missing = sorted(
        name
        for name, field in fields.items()
        if name not in raw
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"Faltan campos obligatorios en '{resolved_path}': {missing}.")

    for name, value in raw.items():
        _check_field_type(name, value, type_hints[name])

    return schema(**raw)


@dataclasses.dataclass
class ExampleConfig:
    """Esquema minimo de ejemplo para comprobar la carga tipada de YAML."""

    name: str
    max_iterations: int
    tolerance: float
    verbose: bool = False
