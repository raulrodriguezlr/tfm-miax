"""Comprueba que load_config lee un YAML tipado y falla claro ante configuraciones invalidas."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from miax.utils.config import ConfigError, ExampleConfig, load_config

VALID_YAML = """\
name: "ejemplo"
max_iterations: 100
tolerance: 0.001
verbose: true
"""


def write_yaml(tmp_path: Path, content: str, filename: str = "config.yaml") -> Path:
    """Escribe content en un fichero YAML dentro de tmp_path y devuelve su ruta."""
    config_path = tmp_path / filename
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_config_returns_typed_instance(tmp_path: Path) -> None:
    """Un YAML valido se lee como una instancia tipada del dataclass indicado."""
    config_path = write_yaml(tmp_path, VALID_YAML)

    config = load_config(config_path, ExampleConfig)

    assert config == ExampleConfig(
        name="ejemplo", max_iterations=100, tolerance=0.001, verbose=True
    )


def test_load_config_uses_default_for_missing_optional_field(tmp_path: Path) -> None:
    """Un campo con valor por defecto en el dataclass puede omitirse en el YAML."""
    content = 'name: "ejemplo"\nmax_iterations: 5\ntolerance: 0.1\n'
    config_path = write_yaml(tmp_path, content)

    config = load_config(config_path, ExampleConfig)

    assert config.verbose is False


def test_load_config_missing_required_field_raises_clear_error(tmp_path: Path) -> None:
    """Si falta un campo obligatorio, el error nombra ese campo."""
    content = 'name: "ejemplo"\ntolerance: 0.1\n'
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(config_path, ExampleConfig)


def test_load_config_wrong_type_raises_clear_error(tmp_path: Path) -> None:
    """Si un campo tiene un tipo distinto del declarado, el error nombra ese campo."""
    content = 'name: "ejemplo"\nmax_iterations: "cien"\ntolerance: 0.1\n'
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(config_path, ExampleConfig)


def test_load_config_unknown_field_raises_clear_error(tmp_path: Path) -> None:
    """Un campo del YAML que no existe en el dataclass se rechaza, para pillar erratas."""
    content = 'name: "ejemplo"\nmax_iterations: 5\ntolerance: 0.1\nmax_iteratoins: 5\n'
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="max_iteratoins"):
        load_config(config_path, ExampleConfig)


def test_load_config_missing_file_raises_clear_error(tmp_path: Path) -> None:
    """Una ruta que no existe produce un error claro en vez de una excepcion de bajo nivel."""
    missing_path = tmp_path / "no_existe.yaml"

    with pytest.raises(ConfigError, match="no_existe.yaml"):
        load_config(missing_path, ExampleConfig)


def test_example_config_from_configs_directory() -> None:
    """El YAML de ejemplo versionado en configs/ se lee tipado, sin tocar la red."""
    config = load_config("example.yaml", ExampleConfig)

    assert isinstance(config, ExampleConfig)
    assert config.name == "ejemplo"


def test_load_config_non_dict_root_raises_clear_error(tmp_path: Path) -> None:
    """Un YAML cuya raiz es una lista, y no un mapeo, produce un error claro."""
    content = "- 1\n- 2\n"
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="mapeo"):
        load_config(config_path, ExampleConfig)


def test_load_config_scalar_root_raises_clear_error(tmp_path: Path) -> None:
    """Un YAML cuya raiz es un escalar, y no un mapeo, produce un error claro."""
    config_path = write_yaml(tmp_path, "42\n")

    with pytest.raises(ConfigError, match="mapeo"):
        load_config(config_path, ExampleConfig)


def test_load_config_non_dataclass_schema_raises_clear_error(tmp_path: Path) -> None:
    """Un schema que no es un dataclass produce el error explicito ya contemplado en el codigo."""
    config_path = write_yaml(tmp_path, VALID_YAML)

    class NotADataclass:
        """Clase de prueba que deliberadamente no es un dataclass."""

    with pytest.raises(ConfigError, match="no es un dataclass"):
        load_config(config_path, NotADataclass)


def test_load_config_float_field_accepts_integer_literal(tmp_path: Path) -> None:
    """Un campo float puede escribirse como entero literal en el YAML (p.ej. 1 en vez de 1.0)."""
    content = 'name: "ejemplo"\nmax_iterations: 5\ntolerance: 1\n'
    config_path = write_yaml(tmp_path, content)

    config = load_config(config_path, ExampleConfig)

    assert config.tolerance == 1
    assert not isinstance(config.tolerance, bool)


def test_load_config_bool_value_for_numeric_field_raises_clear_error(tmp_path: Path) -> None:
    """Un bool no se acepta para un campo int, aunque bool sea subclase de int en Python."""
    content = 'name: "ejemplo"\nmax_iterations: true\ntolerance: 0.1\n'
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(config_path, ExampleConfig)


def test_load_config_int_value_for_bool_field_raises_clear_error(tmp_path: Path) -> None:
    """Un entero no se acepta para un campo bool, aunque parezca un 0/1 valido."""
    content = 'name: "ejemplo"\nmax_iterations: 5\ntolerance: 0.1\nverbose: 1\n'
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="verbose"):
        load_config(config_path, ExampleConfig)


def test_load_config_empty_yaml_raises_missing_fields_error(tmp_path: Path) -> None:
    """Un YAML vacio se trata como sin campos, y falla nombrando todos los obligatorios."""
    config_path = write_yaml(tmp_path, "")

    with pytest.raises(ConfigError, match="name"):
        load_config(config_path, ExampleConfig)


def test_load_config_parameterized_generic_field_raises_config_error(tmp_path: Path) -> None:
    """Un campo con un generico parametrizado (list[str]) falla con ConfigError, no TypeError."""

    @dataclasses.dataclass
    class GenericFieldConfig:
        """Esquema de prueba con un campo de tipo generico parametrizado."""

        tags: list[str]

    content = "tags:\n  - a\n  - b\n"
    config_path = write_yaml(tmp_path, content)

    with pytest.raises(ConfigError, match="tags"):
        load_config(config_path, GenericFieldConfig)
