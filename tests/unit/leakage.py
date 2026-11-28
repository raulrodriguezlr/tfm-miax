"""Ayudante para tests antifuga: inyecta futuro y comprueba que el pasado no cambia."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from miax.features.factor import _same

SIGNAL_SDS = 50.0  # tamano de la senal de futuro inyectada, en desviaciones tipicas de la moneda
NEW_NAN_SHARE = 0.5  # con `move_nans`, fraccion de celdas del futuro que quedan en NaN


def perturb_after(
    df: pd.DataFrame, cut: pd.Timestamp, *, seed: int = 58, move_nans: bool = False
) -> pd.DataFrame:
    """Cambia todas las filas con indice >= `cut` por ruido nuevo mas una senal grande; antes, nada.

    Mantiene los NaN de cada celda salvo con `move_nans`, que los recoloca al azar.
    """
    rng = np.random.default_rng(seed)
    positions = np.flatnonzero(df.index >= cut)
    if positions.size == 0:
        raise ValueError(f"no hay filas con indice >= {cut}: no hay futuro que perturbar")
    scale = df.std().fillna(0.0).to_numpy()
    block = df.iloc[positions]
    values = rng.normal(0.0, 1.0, size=block.shape) * scale
    present = block.notna().to_numpy()
    if move_nans:
        present = rng.random(block.shape) >= NEW_NAN_SHARE
    # La senal va en la primera moneda: un futuro evidente que un modelo honesto no debe ver.
    values[:, 0] += SIGNAL_SDS * scale[0]
    out = df.copy()
    out.iloc[positions] = np.where(present, values, np.nan)
    return out


def _compare(before: object, after: object, cut: pd.Timestamp) -> None:
    """Compara dos salidas: filas < `cut` si tienen indice temporal, si no enteras."""
    if isinstance(before, pd.DataFrame | pd.Series) and isinstance(before.index, pd.DatetimeIndex):
        assert isinstance(after, type(before))
        old, new = before[before.index < cut], after[after.index < cut]
        if old.empty:
            raise ValueError(f"la salida no tiene filas anteriores a {cut}: no se compara nada")
        if old.isna().to_numpy().all():
            raise ValueError("la salida anterior al corte es toda NaN: la comprobacion seria vacia")
        if isinstance(old, pd.DataFrame):
            pd.testing.assert_frame_equal(old, new, check_exact=True)  # type: ignore[arg-type]
        else:
            pd.testing.assert_series_equal(old, new, check_exact=True)  # type: ignore[arg-type]
        return
    assert _same(before, after), "el estado cambio"


def assert_no_lookahead(
    run: Callable[[pd.DataFrame], object],
    data: pd.DataFrame,
    cut: pd.Timestamp,
    *,
    seed: int = 58,
    move_nans: bool = False,
    what: str = "",
) -> None:
    """Falla (AssertionError) si cambiar los datos desde `cut` cambia `run` antes de `cut`.

    Si la salida no tiene indice temporal (betas, cargas), se compara entera.
    """
    perturbed = perturb_after(data, cut, seed=seed, move_nans=move_nans)
    try:
        _compare(run(data), run(perturbed), cut)
    except AssertionError as exc:
        raise AssertionError(
            f"fuga de futuro en {what or 'run'} (corte {cut}): el resultado anterior al corte "
            f"cambia al cambiar los datos posteriores (D-03, CLAUDE.md 6.2)\n{exc}"
        ) from exc
