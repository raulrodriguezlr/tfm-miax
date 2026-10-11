"""Lectura y escritura de Parquet con rutas relativas a la raiz del repo."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from miax.utils.config import REPO_ROOT


def resolve_repo_path(path: str | Path) -> Path:
    """Devuelve la ruta tal cual si es absoluta, y si no, resuelta desde la raiz del repo."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def write_parquet(df: pd.DataFrame, path: str | Path, compression: str = "snappy") -> Path:
    """Escribe un DataFrame en Parquet de forma atomica y devuelve la ruta final."""
    target = resolve_repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        df.to_parquet(tmp, engine="pyarrow", compression=compression)
        # os.replace es atomico: una escritura cortada no deja un Parquet corrupto con ese nombre.
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Lee un Parquet; todo vuelve igual salvo `DatetimeIndex.freq`, que Parquet no guarda."""
    return pd.read_parquet(resolve_repo_path(path), engine="pyarrow", columns=columns)
