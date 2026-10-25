"""Contrato de la interfaz A -> B: que significa cada feature del panel y como se comprueba."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from miax.ingest.klines import interval_ms

if TYPE_CHECKING:
    # Solo para los tipos: `panel.py` importa este modulo, y un import real seria circular.
    from miax.features.panel import Panel

_INDEX_DTYPE = "datetime64[ms, UTC]"


@dataclass(frozen=True)
class FeatureSpec:
    """Significado de una feature del panel: unidad, definicion y que valores admite."""

    name: str
    unit: str
    description: str
    nan_if_invalid: bool
    non_negative: bool


# Mismo orden que `FEATURE_NAMES` de `panel.py`.
FEATURE_SPECS: Mapping[str, FeatureSpec] = MappingProxyType(
    {
        "return": FeatureSpec(
            name="return",
            unit="log-retorno, unidades naturales (no %)",
            description="ln P_t - ln P_{t-1} del cierre; solo existe si t y t-1 son validos",
            nan_if_invalid=True,
            non_negative=False,
        ),
        "realized_vol": FeatureSpec(
            name="realized_vol",
            unit="desviacion tipica del retorno de 1 paso, sin anualizar",
            description=(
                "sqrt((W/n) * sum(r^2)) de los W retornos anteriores incluido el actual, con n "
                "los no NaN (D-20); NaN si n/W baja del minimo (por defecto W = 60, minimo 80 %)"
            ),
            nan_if_invalid=False,
            non_negative=True,
        ),
        "rel_volume": FeatureSpec(
            name="rel_volume",
            unit="diferencia de logaritmos, sin unidad",
            description=(
                "ln(volumen en quote) menos la media de ln V de las velas validas de la ventana "
                "anterior, sin la actual (D-21); NaN si la cobertura valida baja del minimo "
                "(por defecto ventana de 10080 velas, minimo 50 %)"
            ),
            nan_if_invalid=True,
            non_negative=False,
        ),
        "rel_trades": FeatureSpec(
            name="rel_trades",
            unit="diferencia de logaritmos, sin unidad",
            description=(
                "ln(numero de operaciones) menos la media de ln N de las velas validas de la "
                "ventana anterior, sin la actual (D-21); NaN si la cobertura valida baja del "
                "minimo (por defecto ventana de 10080 velas, minimo 50 %)"
            ),
            nan_if_invalid=True,
            non_negative=False,
        ),
    }
)


def _violation(label: str, mask: np.ndarray, panel: Panel) -> str | None:
    """Texto de la violacion (celdas y primer instante), o None si `mask` es toda False."""
    count = int(mask.sum())
    if not count:
        return None
    first_row = int(mask.any(axis=1).argmax())
    first_col = int(mask[first_row].argmax())
    return (
        f"{label}: {count} celdas; primer instante {panel.index[first_row]} "
        f"({panel.symbols[first_col]})"
    )


def validate_panel(panel: Panel) -> None:
    """Comprueba el significado del panel (contrato A -> B) y lanza un `ValueError` con todo."""
    problems: list[str] = []

    unknown = [name for name in panel.features if name not in FEATURE_SPECS]
    if unknown:
        problems.append(f"features fuera del contrato: {unknown}; hay {list(FEATURE_SPECS)}")

    index = panel.index
    if str(index.dtype) != _INDEX_DTYPE:
        problems.append(f"el indice es {index.dtype}, debe ser {_INDEX_DTYPE}")
    step = interval_ms(panel.interval)
    if len(index) > 1 and step is not None:
        # Solo se miran diferencias entre filas consecutivas (pasado): no hay ventana hacia delante.
        gaps = np.asarray((index[1:] - index[:-1]) != pd.Timedelta(milliseconds=step))
        if gaps.any():
            problems.append(
                f"indice: {int(gaps.sum())} saltos distintos de {panel.interval}; "
                f"primero tras {index[int(gaps.argmax())]}"
            )

    valid = panel.valid.to_numpy(dtype=bool)
    for name, frame in panel.features.items():
        values = frame.to_numpy(dtype="float64")
        checks = [(f"{name}: valores infinitos", np.isinf(values))]
        spec = FEATURE_SPECS.get(name)
        if spec is not None:
            if spec.nan_if_invalid:
                checks.append(
                    (f"{name}: valor no NaN en velas no validas", ~np.isnan(values) & ~valid)
                )
            if spec.non_negative:
                checks.append((f"{name}: valores negativos", values < 0))
        for label, mask in checks:
            message = _violation(label, mask, panel)
            if message is not None:
                problems.append(message)

    if "return" in panel.features and len(index):
        # `shift(1)` mira a la fila anterior (pasado), no hay fuga; la primera fila no se comprueba
        # porque su vela anterior queda fuera del panel (`fill_value=True`).
        previous_valid = panel.valid.shift(1, fill_value=True).to_numpy(dtype=bool)
        returns = panel.features["return"].to_numpy(dtype="float64")
        message = _violation(
            "return: puentea un hueco (retorno con la vela anterior no valida)",
            ~np.isnan(returns) & ~previous_valid,
            panel,
        )
        if message is not None:
            problems.append(message)

    if problems:
        raise ValueError("el panel no cumple el contrato A -> B:\n- " + "\n- ".join(problems))
