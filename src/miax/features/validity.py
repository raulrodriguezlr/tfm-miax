"""Mascara de validez de los slots y de los retornos a 1 paso, sin rellenar huecos (D-02)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from miax.features.gaps import GapCause

# Validez = estado de liquidez del slot (hay vela con operaciones), nunca el valor del retorno:
# descartar los retornos 0 con operaciones seria filtrar por la variable que se quiere explicar.


def validity_mask(causes: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un `bool` por slot: `True` solo donde la causa es `GapCause.VALID`."""
    if any(pd.api.types.is_bool_dtype(dtype) for dtype in causes.dtypes):
        raise ValueError("causes no puede ser bool: parece una mascara, no causas de GapCause")
    values = causes.to_numpy()
    known = np.isin(values, [int(cause) for cause in GapCause])
    if not known.all():
        raise ValueError("causes tiene valores que no son un GapCause conocido")
    return pd.DataFrame(values == GapCause.VALID, index=causes.index, columns=causes.columns)


def return_validity(valid: pd.DataFrame) -> pd.DataFrame:
    """Marca los retornos a 1 paso validos: solo si lo son el slot `t` y el `t-1`."""
    # shift(1) solo trae el pasado (t-1) a la fila t: no hay fuga; la primera fila no tiene t-1.
    previous = valid.shift(1, fill_value=False).astype(bool)
    return (valid.astype(bool) & previous).astype(bool)


def mask_invalid(frame: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Copia de `frame` con `NaN` donde `valid` es `False`; nunca 0 ni relleno."""
    if not frame.index.equals(valid.index) or not frame.columns.equals(valid.columns):
        raise ValueError("frame y valid deben tener el mismo indice y las mismas columnas")
    if not all(pd.api.types.is_float_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("frame debe ser float: un entero no admite NaN")
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in valid.dtypes):
        raise ValueError("valid debe ser bool en todas sus columnas: un NaN no es False")
    return frame.where(valid.to_numpy(dtype=bool))
