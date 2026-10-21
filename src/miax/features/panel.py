"""Panel maestro `tiempo x activo x feature`: el objeto que se entrega al resto del proyecto."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from miax.features.activity import log_relative_to_past
from miax.features.alignment import align_klines, build_time_grid, shift_to_close
from miax.features.gaps import classify_klines
from miax.features.returns import log_returns
from miax.features.validity import validity_mask
from miax.features.volatility import check_window_params, realized_volatility
from miax.ingest.klines import interval_ms
from miax.ingest.store import DEFAULT_BASE_DIR
from miax.ingest.universe import to_ms

FEATURE_NAMES: tuple[str, ...] = ("return", "realized_vol", "rel_volume", "rel_trades")


@dataclass(frozen=True, eq=False)
class Panel:
    """Features `tiempo x simbolo` y validez por vela.

    La estructura no se reasigna, pero los DataFrames internos no estan protegidos:
    copia antes de modificar.
    """

    features: Mapping[str, pd.DataFrame]
    valid: pd.DataFrame
    interval: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        """Valida indice, columnas y dtypes, y congela `features` y `symbols`."""
        if not self.features:
            raise ValueError("features vacio: el panel necesita al menos una feature")
        symbols = tuple(self.symbols)
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"symbols tiene nombres duplicados: {list(symbols)}")
        if interval_ms(self.interval) is None:
            raise ValueError(f"El intervalo {self.interval!r} no tiene paso fijo")
        index = self.valid.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
            raise ValueError("el indice debe ser un DatetimeIndex con zona UTC")
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("el indice debe ser monotono creciente y sin duplicados")
        columns = list(symbols)
        if not all(pd.api.types.is_bool_dtype(dtype) for dtype in self.valid.dtypes):
            raise ValueError("valid debe ser bool en todas sus columnas")
        frames = {**self.features, "valid": self.valid}
        for name, frame in frames.items():
            if list(frame.columns) != columns:
                raise ValueError(f"{name}: las columnas no coinciden con symbols")
            if not frame.index.equals(self.valid.index):
                raise ValueError(f"{name}: el indice no coincide con el de valid")
        for name, frame in self.features.items():
            if not all(dtype == "float64" for dtype in frame.dtypes):
                raise ValueError(f"{name}: debe ser float64 en todas las columnas")
        # `frozen` impide asignar: se usa `object.__setattr__` solo para congelar la entrada.
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    @property
    def index(self) -> pd.DatetimeIndex:
        """Indice comun: el instante en que se conoce cada fila."""
        return pd.DatetimeIndex(self.valid.index)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Forma `(T, N, F)` de `to_array()`."""
        return (len(self.valid.index), len(self.symbols), len(self.features))

    def to_array(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Apila las features en un array `float64` `(T, N, F)` conservando los `NaN`."""
        names = list(self.features) if order is None else list(order)
        if not names:
            raise ValueError("order vacio: hace falta al menos una feature")
        if len(set(names)) != len(names):
            raise ValueError(f"order tiene features repetidas: {names}")
        unknown = [name for name in names if name not in self.features]
        if unknown:
            raise ValueError(f"features desconocidas: {unknown}; hay {list(self.features)}")
        return np.stack([self.features[name].to_numpy(dtype="float64") for name in names], axis=2)


def build_panel(
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    vol_window: int = 60,
    vol_min_valid_share: float = 0.8,
    activity_window: int = 10080,
    activity_min_valid_share: float = 0.5,
) -> Panel:
    """Panel de las cuatro features a 1m en `[start_ms, end_ms)`.

    Solo `return`, `rel_volume` y `rel_trades` son `NaN` en un slot invalido;
    `realized_vol` puede tener valor con `valid=False`.
    """
    # Cada feature es identica a la salida de su `*_from_cache`; la causa del hueco no entra (D-17).
    if not symbols:
        raise ValueError("symbols vacio: no hay nada que construir")
    check_window_params(vol_window, vol_min_valid_share)
    check_window_params(activity_window, activity_min_valid_share)
    step = interval_ms(interval)
    if step is None:
        raise ValueError(f"El intervalo {interval!r} no tiene duracion fija: no hay rejilla")
    if interval != "1m":
        raise ValueError(
            f"build_panel solo admite 1m, no {interval!r}: las ventanas por defecto son en filas; "
            "las features a 5/15/60 min son MIAX-102"
        )
    grid = build_time_grid(start_ms, end_ms, interval)
    n_out = len(grid)
    if not n_out:
        raise ValueError("rejilla vacia: el rango no contiene ningun slot")
    names = tuple(dict.fromkeys(symbols))

    # Se ensancha solo hacia el pasado, lo justo para la ventana mas larga: retornos (2 pasos),
    # volatilidad (W + 1) y actividad (window + 1) piden velas ANTERIORES a start_ms, y el final
    # es end_ms - paso (la vela que cierra en end_ms - paso), asi que ninguna fila ve el futuro.
    back = max(2, vol_window + 1, activity_window + 1)
    wide_start = start_ms - back * step
    wide_end = end_ms - step
    wide_len = len(build_time_grid(wide_start, wide_end, interval))
    fields = ("close", "quote_asset_volume", "num_trades")
    aligned = align_klines(names, interval, wide_start, wide_end, fields=fields, base_dir=base_dir)
    causes = classify_klines(names, interval, wide_start, wide_end, base_dir=base_dir)
    valid_wide = validity_mask(causes)
    del causes

    def tail(frame: pd.DataFrame, first_ms: int) -> pd.DataFrame:
        # Todos los recortes acaban en wide_end: solo se quita pasado lejano, nunca se ve el futuro.
        return frame.iloc[wide_len - len(build_time_grid(first_ms, wide_end, interval)) :]

    # Retornos de [start - 2*paso, end - paso) sin su primera fila, como `returns_from_cache`;
    # el diff de una fila solo usa t-1, asi que calcularlos sobre el rango ancho da lo mismo.
    ret_first = start_ms - 2 * step
    returns = log_returns(
        tail(aligned["close"], ret_first), tail(valid_wide, ret_first), interval
    ).iloc[1:]
    # Volatilidad: ventana sobre los retornos desde start - (W - 1)*paso (pasado), como
    # `realized_volatility_from_cache`; se quitan las `lead` filas que solo llenan la ventana.
    lead = vol_window - 1
    vol_first = start_ms - lead * step - 2 * step
    vol_returns = log_returns(
        tail(aligned["close"], vol_first), tail(valid_wide, vol_first), interval
    ).iloc[1:]
    vol = realized_volatility(
        vol_returns, window=vol_window, min_valid_share=vol_min_valid_share
    ).iloc[lead:]
    del vol_returns

    # Actividad: base de `window` velas anteriores desde start - (window + 1)*paso (pasado), como
    # `relative_activity_from_cache`; se quitan las `window` filas que solo llenan la base.
    act_first = start_ms - (activity_window + 1) * step
    act_valid = tail(valid_wide, act_first)
    relative: dict[str, pd.DataFrame] = {}
    for key, field in (("rel_volume", "quote_asset_volume"), ("rel_trades", "num_trades")):
        relative[key] = log_relative_to_past(
            tail(aligned[field], act_first),
            act_valid,
            interval,
            window=activity_window,
            min_valid_share=activity_min_valid_share,
        ).iloc[activity_window:]
    del aligned

    features = {
        "return": returns,
        "realized_vol": vol,
        "rel_volume": relative["rel_volume"],
        "rel_trades": relative["rel_trades"],
    }
    # Validez de la vela que cierra en tau: la vela abierta en tau - paso, indexada al cierre.
    valid = valid_wide.iloc[-n_out:]
    valid = valid.set_axis(shift_to_close(valid.index, interval), axis=0)
    for name, frame in {**features, "valid": valid}.items():
        if not frame.index.equals(grid):
            raise RuntimeError(f"el indice de {name} no coincide con la rejilla del rango")
    return Panel(features=features, valid=valid, interval=interval, symbols=names)


def period_panels(
    universe: pd.DataFrame,
    interval: str,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    vol_window: int = 60,
    vol_min_valid_share: float = 0.8,
    activity_window: int = 10080,
    activity_min_valid_share: float = 0.5,
) -> Iterator[tuple[pd.Timestamp, Panel]]:
    """Rinde un panel por periodo del universo, de uno en uno y en orden de `period_start`."""
    missing = [c for c in ("period_start", "period_end", "symbol") if c not in universe.columns]
    if missing:
        raise ValueError(f"el universo no tiene las columnas {missing}")
    ordered = universe.sort_values("period_start", kind="stable")
    for period_start, group in ordered.groupby("period_start", sort=True):
        symbols = list(dict.fromkeys(group["symbol"]))
        if group["period_end"].nunique() != 1:
            raise ValueError(f"el periodo {period_start} tiene varios period_end distintos")
        if (
            pd.Timestamp(period_start).tzinfo is None
            or pd.Timestamp(group["period_end"].iloc[0]).tzinfo is None
        ):
            raise ValueError(
                f"el periodo {period_start} debe tener period_start y period_end en UTC"
            )
        # `period_end` es exclusivo; `to_ms` evita `Timestamp.value`, que en pandas 3 es ns.
        start_ms = to_ms(pd.Timestamp(period_start))
        end_ms = to_ms(pd.Timestamp(group["period_end"].iloc[0]))
        yield (
            pd.Timestamp(period_start),
            build_panel(
                symbols,
                interval,
                start_ms,
                end_ms,
                base_dir=base_dir,
                vol_window=vol_window,
                vol_min_valid_share=vol_min_valid_share,
                activity_window=activity_window,
                activity_min_valid_share=activity_min_valid_share,
            ),
        )
