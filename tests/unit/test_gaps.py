"""Tests de la clasificacion de huecos por causa: tablas sinteticas y caches en `tmp_path`, sin red.

Las series se escriben como cadenas, un caracter por slot de 1 minuto: `v` = vela con operaciones,
`z` = vela sin operaciones (`num_trades == 0`) y `.` = sin vela. Las causas se leen igual:
V valid, T no_trades, L not_listed, H halted, O outage, D delisted, N no_data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from miax.features import (
    GapCause,
    build_time_grid,
    cause_counts,
    classify_klines,
    classify_slots,
    download_ends_from_universe,
    gap_runs,
    listings_from_universe,
)
from miax.features.alignment import grid_ms
from miax.ingest import save_klines
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
DAY_MS = 86_400_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
# El slot 6 de la rejilla es la medianoche del 2026-01-02: los 6 primeros son del dia 1.
START = T0 + DAY_MS - 6 * MIN_MS
LETTERS = {
    GapCause.VALID: "V",
    GapCause.NO_TRADES: "T",
    GapCause.NOT_LISTED: "L",
    GapCause.HALTED: "H",
    GapCause.OUTAGE: "O",
    GapCause.DELISTED: "D",
    GapCause.NO_DATA: "N",
}


def _slot(k: int) -> int:
    """`open_time` (ms) del slot `k` de la rejilla de los tests."""
    return START + k * MIN_MS


def _trades(spec: str) -> list[float]:
    """Pasa una cadena `v`/`z`/`.` a `num_trades` (NaN = sin vela)."""
    values = {"v": 5.0, "z": 0.0, ".": float("nan")}
    return [values[c] for c in spec]


def _wide(specs: Mapping[str, str]) -> pd.DataFrame:
    """Ancho `tiempo x simbolo` de `num_trades` con la rejilla de 1m desde `START`."""
    n = len(next(iter(specs.values())))
    index = build_time_grid(START, START + n * MIN_MS, "1m")
    return pd.DataFrame({s: _trades(spec) for s, spec in specs.items()}, index=index)


def _life(specs: Mapping[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    """Primera y ultima vela de cada serie (la vida en la cache); omite las series sin velas."""
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for symbol, spec in specs.items():
        present = [k for k, c in enumerate(spec) if c != "."]
        if present:
            first[symbol], last[symbol] = _slot(present[0]), _slot(present[-1])
    return first, last


def _classify(specs: Mapping[str, str], **kwargs: object) -> pd.DataFrame:
    """`classify_slots` con la vida sacada de las propias series, salvo que se pase otra."""
    first, last = _life(specs)
    kwargs.setdefault("first_ms", first)
    kwargs.setdefault("last_ms", last)
    return classify_slots(_wide(specs), **kwargs)  # type: ignore[arg-type]


def _word(causes: pd.DataFrame, symbol: str) -> str:
    """Las causas de un simbolo como cadena de letras, una por slot."""
    return "".join(LETTERS[GapCause(int(c))] for c in causes[symbol])


# --- Cada causa y sus bordes --------------------------------------------------------------------


def test_every_cause_in_one_series() -> None:
    """Una serie con velas, una sin operaciones, un hueco, listado, y deslistado y fin de datos."""
    specs = {"AAAUSDT": "..vzv..vv..."}
    out = _classify(specs, listed_ms={"AAAUSDT": _slot(1)}, data_end_ms=_slot(10))

    assert _word(out, "AAAUSDT") == "LLVTVHHVVDDN"


def test_output_shape_dtype_index_and_columns() -> None:
    """Sale el mismo ancho que entra, en `int8`, con el mismo indice y orden de columnas."""
    specs = {"BBBUSDT": "vvvv", "AAAUSDT": "vv.v"}
    wide = _wide(specs)
    out = classify_slots(wide, first_ms=_life(specs)[0], last_ms=_life(specs)[1])

    assert out.shape == wide.shape
    assert (out.dtypes == "int8").all()
    assert out.index.equals(wide.index)
    assert list(out.columns) == ["BBBUSDT", "AAAUSDT"]


def test_no_trades_is_a_candle_with_zero_trades_not_a_gap() -> None:
    """Una vela con 0 operaciones es `NO_TRADES`, no un hueco; el resto de velas es `VALID`."""
    out = _classify({"AAAUSDT": "vzzv"})
    assert _word(out, "AAAUSDT") == "VTTV"


def test_first_and_last_slot_of_life_are_valid_and_neighbours_are_not() -> None:
    """El primer y el ultimo slot de la vida son velas; antes y despues ya es otra causa."""
    specs = {"AAAUSDT": "..vvvv..", "BBBUSDT": "vvvvvvvv"}
    out = _classify(specs)

    assert _word(out, "AAAUSDT") == "NNVVVVDD"
    assert _word(out, "BBBUSDT") == "VVVVVVVV"


@pytest.mark.parametrize(
    ("data_end_slot", "expected"),
    [
        (4, "VVVVVNNNNNNN"),  # data_end_ms == ultima vela: no hay tramo deslistado
        (5, "VVVVVDNNNNNN"),  # un slot despues de la ultima vela: justo ese es DELISTED
        (8, "VVVVVDDDDNNN"),  # data_end_ms es inclusivo: el slot 8 entra y el 9 no
        (11, "VVVVVDDDDDDD"),
    ],
)
def test_data_end_is_inclusive(data_end_slot: int, expected: str) -> None:
    """`data_end_ms` es una `open_time` inclusiva: hasta ahi `DELISTED`, mas alla `NO_DATA`."""
    out = _classify({"AAAUSDT": "vvvvv......."}, data_end_ms=_slot(data_end_slot))
    assert _word(out, "AAAUSDT") == expected


def test_data_end_defaults_to_max_last_ms() -> None:
    """Sin `data_end_ms`, `DELISTED` llega hasta la ultima vela del simbolo que mas dura."""
    out = _classify({"AAAUSDT": "vvv.........", "BBBUSDT": "vvvvvvv....."})

    assert _word(out, "AAAUSDT") == "VVVDDDDNNNNN"
    assert _word(out, "BBBUSDT") == "VVVVVVVNNNNN"


def test_data_end_before_last_candle_does_not_hide_candles() -> None:
    """Un `data_end_ms` anterior a la ultima vela no cambia las velas: siguen siendo `VALID`."""
    out = _classify({"AAAUSDT": "vvvvvvvv...."}, data_end_ms=_slot(2))
    assert _word(out, "AAAUSDT") == "VVVVVVVVNNNN"


def test_no_listing_known_means_no_data_before_first_candle() -> None:
    """Sin listado conocido, lo anterior a la primera vela cacheada es `NO_DATA`."""
    out = _classify({"AAAUSDT": "....vvvv"})
    assert _word(out, "AAAUSDT") == "NNNNVVVV"


def test_listing_on_a_different_day_leaves_no_data_between_listing_and_first_candle() -> None:
    """Listado otro dia: `NOT_LISTED` hasta el listado y `NO_DATA` hasta la primera vela."""
    # Listado en el slot 2 (dia 1); primera vela en el slot 6 (medianoche del dia 2).
    out = _classify({"AAAUSDT": "......vvvv.."}, listed_ms={"AAAUSDT": _slot(2)})
    assert _word(out, "AAAUSDT") == "LLNNNNVVVVNN"


def test_listing_on_the_same_day_makes_first_candle_the_listing() -> None:
    """Si la primera vela cae el dia del listado, el listado efectivo es esa vela."""
    out = _classify({"AAAUSDT": "....vvvv...."}, listed_ms={"AAAUSDT": _slot(1)})
    assert _word(out, "AAAUSDT") == "LLLLVVVVNNNN"


def test_listing_at_midnight_with_first_candle_the_same_day() -> None:
    """Listado a las 00:00 del dia 2 y primera vela a las 00:01: el listado efectivo es la vela."""
    out = _classify({"AAAUSDT": ".......vvvvv"}, listed_ms={"AAAUSDT": _slot(6)})
    assert _word(out, "AAAUSDT") == "LLLLLLLVVVVV"


def test_listing_after_first_candle_on_another_day_marks_nothing() -> None:
    """Un listado posterior a la primera vela y de otro dia no inventa `NOT_LISTED`."""
    out = _classify({"AAAUSDT": "vvvvvv......"}, listed_ms={"AAAUSDT": _slot(8) + DAY_MS})
    assert "L" not in _word(out, "AAAUSDT")


def test_symbol_without_cache_is_all_no_data_except_before_known_listing() -> None:
    """Sin cache: todo `NO_DATA`, y `NOT_LISTED` solo antes del listado; nunca `DELISTED`."""
    specs = {"AAAUSDT": "vvvvvvvvvvvv", "NONEUSDT": "............"}
    first, last = _life(specs)
    assert "NONEUSDT" not in first

    plain = classify_slots(_wide(specs), first_ms=first, last_ms=last, data_end_ms=_slot(11))
    listed = classify_slots(
        _wide(specs),
        first_ms=first,
        last_ms=last,
        data_end_ms=_slot(11),
        listed_ms={"NONEUSDT": _slot(4)},
    )

    assert _word(plain, "NONEUSDT") == "NNNNNNNNNNNN"
    assert _word(listed, "NONEUSDT") == "LLLLNNNNNNNN"
    assert _word(plain, "AAAUSDT") == "VVVVVVVVVVVV"


def test_gap_at_edge_of_life_next_to_a_candle_is_halted() -> None:
    """Un hueco pegado a la primera y a la ultima vela, dentro de la vida, es `HALTED`."""
    out = _classify({"AAAUSDT": "v..vv..v", "BBBUSDT": "vvvvvvvv"})
    assert _word(out, "AAAUSDT") == "VHHVVHHV"


def test_listed_entries_for_other_symbols_are_ignored() -> None:
    """Un listado de un simbolo que no esta en la tabla no hace nada."""
    out = _classify({"AAAUSDT": "..vv"}, listed_ms={"ZZZUSDT": _slot(0)})
    assert _word(out, "AAAUSDT") == "NNVV"


def test_classification_is_deterministic_and_does_not_mutate_input() -> None:
    """Dos llamadas con lo mismo dan lo mismo y la entrada no cambia."""
    specs = {"AAAUSDT": "vv.vz.v.", "BBBUSDT": "vv.vvvvv"}
    wide = _wide(specs)
    snapshot = wide.copy()
    first, last = _life(specs)

    one = classify_slots(wide, first_ms=first, last_ms=last)
    two = classify_slots(wide, first_ms=first, last_ms=last)

    pd.testing.assert_frame_equal(one, two)
    pd.testing.assert_frame_equal(wide, snapshot)


# --- OUTAGE -------------------------------------------------------------------------------------


def test_outage_exactly_at_the_share_threshold_is_inclusive() -> None:
    """4 vivos: faltar 2 (0,5) ya es `OUTAGE`; faltar 1 (0,25), `HALTED`; faltar 3, `OUTAGE`."""
    specs = {
        "AAAUSDT": "vv.v.v.v",
        "BBBUSDT": "vv.vvv.v",
        "CCCUSDT": "vvvvvv.v",
        "DDDUSDT": "vvvvvvvv",
    }
    out = _classify(specs)

    assert _word(out, "AAAUSDT") == "VVOVHVOV"
    assert _word(out, "BBBUSDT") == "VVOVVVOV"
    assert _word(out, "CCCUSDT") == "VVVVVVOV"
    assert _word(out, "DDDUSDT") == "VVVVVVVV"


def test_outage_below_the_share_threshold_is_halted() -> None:
    """Con 5 vivos, 2 que faltan (0,4) no llegan al 0,5: `HALTED`; 3 (0,6) si: `OUTAGE`."""
    specs = {
        "AAAUSDT": "vv.v.v.v",
        "BBBUSDT": "vv.vvv.v",
        "CCCUSDT": "vvvvvv.v",
        "DDDUSDT": "vvvvvvvv",
        "EEEUSDT": "vvvvvvvv",
    }
    out = _classify(specs)

    assert _word(out, "AAAUSDT") == "VVHVHVOV"
    assert _word(out, "BBBUSDT") == "VVHVVVOV"


def test_outage_min_share_is_a_parameter() -> None:
    """Bajar `outage_min_share` a 0,4 convierte en `OUTAGE` el caso de 2 de 5."""
    specs = {
        "AAAUSDT": "vv.vv",
        "BBBUSDT": "vv.vv",
        "CCCUSDT": "vvvvv",
        "DDDUSDT": "vvvvv",
        "EEEUSDT": "vvvvv",
    }
    assert _word(_classify(specs), "AAAUSDT") == "VVHVV"
    assert _word(_classify(specs, outage_min_share=0.4), "AAAUSDT") == "VVOVV"
    assert _word(_classify(specs, outage_min_share=0.41), "AAAUSDT") == "VVHVV"


def test_two_symbols_one_missing_is_halted_both_missing_is_outage() -> None:
    """Con solo BTC y ETH: si falta uno es `HALTED`; si faltan los dos a la vez, `OUTAGE`."""
    one = _classify({"BTCUSDT": "vv.vv", "ETHUSDT": "vvvvv"})
    both = _classify({"BTCUSDT": "vv.vv", "ETHUSDT": "vv.vv"})

    assert _word(one, "BTCUSDT") == "VVHVV"
    assert _word(one, "ETHUSDT") == "VVVVV"
    assert _word(both, "BTCUSDT") == "VVOVV"
    assert _word(both, "ETHUSDT") == "VVOVV"


def test_single_symbol_never_has_outage() -> None:
    """Un solo simbolo con hueco no puede ser `OUTAGE` (faltan menos de `outage_min_symbols`)."""
    out = _classify({"AAAUSDT": "vv...vv"})
    assert _word(out, "AAAUSDT") == "VVHHHVV"


def test_outage_min_symbols_is_a_parameter() -> None:
    """Con `outage_min_symbols=1` un solo simbolo que falta (1 de 2, 0,5) ya es `OUTAGE`."""
    specs = {"AAAUSDT": "v.v", "BBBUSDT": "vvv"}
    assert _word(_classify(specs), "AAAUSDT") == "VHV"
    assert _word(_classify(specs, outage_min_symbols=1), "AAAUSDT") == "VOV"


def test_symbols_not_alive_do_not_count_in_the_denominator() -> None:
    """En el slot 3 viven 3 simbolos y faltan 2 (0,67): `OUTAGE`; con los 6 contados seria 0,33."""
    specs = {
        "AAAUSDT": "vvv.vvvv",  # vivo, falta
        "BBBUSDT": "vvv.vvvv",  # vivo, falta
        "CCCUSDT": "vvvvvvvv",  # vivo, con vela
        "DDDUSDT": "....vvvv",  # aun sin cotizar
        "EEEUSDT": "vv......",  # ya deslistado
        "FFFUSDT": "......vv",  # aun sin cotizar
    }
    out = _classify(specs)

    assert _word(out, "AAAUSDT") == "VVVOVVVV"
    assert _word(out, "BBBUSDT") == "VVVOVVVV"
    # Los no vivos no cuentan como "faltan": conservan su propia causa.
    assert _word(out, "DDDUSDT") == "NNNNVVVV"
    assert _word(out, "EEEUSDT") == "VVDDDDDD"
    assert _word(out, "FFFUSDT") == "NNNNNNVV"


def test_candles_without_trades_do_not_count_as_missing_for_outage() -> None:
    """Las velas con 0 operaciones existen: no hacen `OUTAGE` aunque sean todas a la vez."""
    out = _classify({"AAAUSDT": "vzv", "BBBUSDT": "vzv"})
    assert _word(out, "AAAUSDT") == "VTV"
    assert _word(out, "BBBUSDT") == "VTV"


def test_outage_does_not_touch_non_interior_slots() -> None:
    """`OUTAGE` solo afecta a huecos internos: `DELISTED` y `NO_DATA` simultaneos se quedan."""
    specs = {"AAAUSDT": "vv.vvvv", "BBBUSDT": "vv.vvvv", "CCCUSDT": "vv....."}
    out = _classify(specs)

    assert _word(out, "AAAUSDT") == "VVOVVVV"
    assert _word(out, "CCCUSDT") == "VVDDDDD"


# --- Validaciones -------------------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
def test_invalid_outage_min_share_raises(share: float) -> None:
    """Una proporcion fuera de (0, 1] da `ValueError`."""
    with pytest.raises(ValueError, match="outage_min_share"):
        _classify({"AAAUSDT": "vvv"}, outage_min_share=share)


def test_outage_min_share_one_is_valid() -> None:
    """El 1 es un valor valido: solo hay `OUTAGE` si faltan todos los vivos."""
    out = _classify({"AAAUSDT": "v.v", "BBBUSDT": "v.v"}, outage_min_share=1.0)
    assert _word(out, "AAAUSDT") == "VOV"


def test_invalid_outage_min_symbols_raises() -> None:
    """Un minimo de simbolos menor que 1 da `ValueError`."""
    with pytest.raises(ValueError, match="outage_min_symbols"):
        _classify({"AAAUSDT": "vvv"}, outage_min_symbols=0)


def test_duplicate_columns_raise() -> None:
    """Columnas repetidas dan `ValueError`."""
    wide = _wide({"AAAUSDT": "vvv"})
    wide = pd.concat([wide, wide], axis=1)
    with pytest.raises(ValueError, match="repetidos"):
        classify_slots(wide, first_ms={"AAAUSDT": _slot(0)}, last_ms={"AAAUSDT": _slot(2)})


def test_first_without_last_raises_naming_symbol() -> None:
    """Dar `first_ms` sin `last_ms` (o al reves) da `ValueError` con el simbolo."""
    wide = _wide({"AAAUSDT": "vvv"})
    with pytest.raises(ValueError, match="AAAUSDT.*a la vez"):
        classify_slots(wide, first_ms={"AAAUSDT": _slot(0)}, last_ms={})
    with pytest.raises(ValueError, match="AAAUSDT.*a la vez"):
        classify_slots(wide, first_ms={}, last_ms={"AAAUSDT": _slot(2)})


def test_first_after_last_raises_naming_symbol() -> None:
    """Un `first_ms` posterior a `last_ms` da `ValueError` con el simbolo."""
    wide = _wide({"AAAUSDT": "vvv"})
    with pytest.raises(ValueError, match="AAAUSDT.*posterior"):
        classify_slots(wide, first_ms={"AAAUSDT": _slot(2)}, last_ms={"AAAUSDT": _slot(0)})


@pytest.mark.parametrize(
    ("first_slot", "last_slot"),
    [(1, 2), (0, 1)],  # una vela antes de la primera, una vela despues de la ultima
)
def test_candle_outside_declared_life_raises(first_slot: int, last_slot: int) -> None:
    """Una vela fuera de `[first_ms, last_ms]` es incoherente y da `ValueError` con el simbolo."""
    wide = _wide({"AAAUSDT": "vvv"})
    with pytest.raises(ValueError, match="AAAUSDT"):
        classify_slots(
            wide, first_ms={"AAAUSDT": _slot(first_slot)}, last_ms={"AAAUSDT": _slot(last_slot)}
        )


def test_candle_of_symbol_without_bounds_raises() -> None:
    """Una vela de un simbolo sin limites de cache tambien es incoherente."""
    wide = _wide({"AAAUSDT": "vvv", "BBBUSDT": "vvv"})
    with pytest.raises(ValueError, match="BBBUSDT"):
        classify_slots(wide, first_ms={"AAAUSDT": _slot(0)}, last_ms={"AAAUSDT": _slot(2)})


# --- gap_runs y cause_counts -------------------------------------------------------------------


def _synthetic_causes() -> pd.DataFrame:
    """Tabla de causas con tramos al principio, en medio y al final, y dos simbolos."""
    specs = {"AAAUSDT": "..vzzv..vv...", "BBBUSDT": "vvvvvvvvvvvvv"}
    return _classify(specs, listed_ms={"AAAUSDT": _slot(0)}, data_end_ms=_slot(11))


def test_synthetic_table_is_what_the_runs_expect() -> None:
    """Punto de partida de los tests de tramos: la tabla sintetica da estas causas."""
    causes = _synthetic_causes()
    assert _word(causes, "AAAUSDT") == "LLVTTVHHVVDDN"
    assert _word(causes, "BBBUSDT") == "VVVVVVVVVVVVV"


def test_gap_runs_lists_runs_without_valid() -> None:
    """Cada tramo de una misma causa sale una vez, sin `VALID`, con `start`, `end` y `n_slots`."""
    runs = gap_runs(_synthetic_causes())

    assert list(runs.columns) == ["symbol", "cause", "start", "end", "n_slots"]
    assert runs["symbol"].tolist() == ["AAAUSDT"] * 5
    assert runs["cause"].tolist() == ["NOT_LISTED", "NO_TRADES", "HALTED", "DELISTED", "NO_DATA"]
    assert runs["n_slots"].tolist() == [2, 2, 2, 2, 1]
    assert runs["start"].tolist() == [
        pd.Timestamp(_slot(k), unit="ms", tz="UTC") for k in (0, 3, 6, 10, 12)
    ]
    assert runs["end"].tolist() == [
        pd.Timestamp(_slot(k), unit="ms", tz="UTC") for k in (1, 4, 7, 11, 12)
    ]


def test_gap_runs_dtypes() -> None:
    """`n_slots` es `int64`, `start`/`end` son fechas UTC y simbolo/causa son texto."""
    runs = gap_runs(_synthetic_causes())

    assert runs["n_slots"].dtype == "int64"
    assert str(runs["start"].dtype).endswith(", UTC]")
    assert str(runs["end"].dtype).endswith(", UTC]")
    assert runs["symbol"].dtype == object
    assert runs["cause"].dtype == object


def test_gap_runs_do_not_merge_different_causes_and_split_around_valid() -> None:
    """Dos huecos separados por una vela son dos tramos; causas distintas pegadas tambien."""
    causes = _classify({"AAAUSDT": "vv..v.vv", "BBBUSDT": "vv..v.vv"})
    runs = gap_runs(causes)
    assert runs["cause"].tolist() == ["OUTAGE", "OUTAGE"] * 2
    assert runs["n_slots"].tolist() == [2, 1, 2, 1]

    adjacent = _classify({"AAAUSDT": "vz..vv"})
    assert gap_runs(adjacent)["cause"].tolist() == ["NO_TRADES", "HALTED"]


def test_gap_runs_without_gaps_is_empty_with_columns() -> None:
    """Sin ningun tramo (todo `VALID`) vuelve un vacio con columnas y `n_slots` `int64`."""
    runs = gap_runs(_classify({"AAAUSDT": "vvvv"}))

    assert runs.empty
    assert list(runs.columns) == ["symbol", "cause", "start", "end", "n_slots"]
    assert runs["n_slots"].dtype == "int64"


def test_gap_runs_with_no_rows_is_empty_with_columns() -> None:
    """Una tabla de causas sin filas tambien da un resultado vacio con las columnas."""
    causes = _synthetic_causes().iloc[:0]
    runs = gap_runs(causes)

    assert runs.empty
    assert list(runs.columns) == ["symbol", "cause", "start", "end", "n_slots"]


def test_gap_runs_run_slots_add_up_to_non_valid_slots() -> None:
    """La suma de `n_slots` por simbolo es el numero de slots que no son `VALID`."""
    causes = _synthetic_causes()
    runs = gap_runs(causes)
    totals = runs.groupby("symbol")["n_slots"].sum()

    assert totals["AAAUSDT"] == (causes["AAAUSDT"] != GapCause.VALID).sum()
    assert "BBBUSDT" not in totals.index


def test_cause_counts_one_column_per_cause_and_rows_add_up() -> None:
    """Hay una columna por cada `GapCause` (aunque valga 0), `int64`, y cada fila suma los slots."""
    causes = _synthetic_causes()
    counts = cause_counts(causes)

    assert list(counts.columns) == [c.name for c in GapCause]
    assert counts.index.name == "symbol"
    assert list(counts.index) == ["AAAUSDT", "BBBUSDT"]
    assert (counts.dtypes == "int64").all()
    assert counts.sum(axis=1).tolist() == [len(causes)] * 2
    assert counts.loc["AAAUSDT"].to_dict() == {
        "VALID": 4,
        "NO_TRADES": 2,
        "NOT_LISTED": 2,
        "HALTED": 2,
        "OUTAGE": 0,
        "DELISTED": 2,
        "NO_DATA": 1,
    }
    assert counts.loc["BBBUSDT", "VALID"] == len(causes)


# --- listings_from_universe ---------------------------------------------------------------------


def _universe(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Universo minimo con `symbol` y `first_candle` como los de `load_universe`."""
    frame = pd.DataFrame(rows, columns=["symbol", "first_candle"])
    frame["first_candle"] = pd.to_datetime(frame["first_candle"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    return frame


def test_listings_take_the_minimum_first_candle_per_symbol() -> None:
    """Un simbolo en varios periodos se queda con el minimo `first_candle`, en ms UTC."""
    universe = _universe(
        [
            ("AAAUSDT", "2022-03-05"),
            ("BBBUSDT", "2021-01-01"),
            ("AAAUSDT", "2021-12-31"),
            ("AAAUSDT", "2023-01-01"),
        ]
    )
    listings = listings_from_universe(universe)

    assert listings == {
        "AAAUSDT": int(pd.Timestamp("2021-12-31", tz="UTC").timestamp() * 1000),
        "BBBUSDT": int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000),
    }
    assert all(isinstance(v, int) for v in listings.values())


def test_listings_from_empty_universe_is_empty() -> None:
    """Un universo sin filas da un dict vacio."""
    assert listings_from_universe(_universe([])) == {}


def test_listings_missing_columns_raise() -> None:
    """Un universo sin `symbol` o sin `first_candle` da `ValueError` con las que faltan."""
    with pytest.raises(ValueError, match="first_candle"):
        listings_from_universe(pd.DataFrame({"symbol": ["AAAUSDT"]}))
    with pytest.raises(ValueError, match="symbol"):
        listings_from_universe(pd.DataFrame({"first_candle": []}))


def test_grid_ms_is_int64_milliseconds() -> None:
    """`grid_ms` (publica) devuelve la rejilla como `int64` en ms desde la epoca."""
    grid = build_time_grid(START, START + 3 * MIN_MS, "1m")
    values = grid_ms(grid)

    assert values.dtype == np.int64
    assert values.tolist() == [_slot(0), _slot(1), _slot(2)]


# --- data_end_ms por simbolo --------------------------------------------------------------------


def test_data_end_mapping_gives_each_symbol_its_own_end() -> None:
    """Con un mapeo, un mismo slot es `DELISTED` para un simbolo y `NO_DATA` para otro."""
    specs = {"AAAUSDT": "vvv.........", "BBBUSDT": "vvv........."}
    out = _classify(specs, data_end_ms={"AAAUSDT": _slot(8), "BBBUSDT": _slot(4)})

    assert _word(out, "AAAUSDT") == "VVVDDDDDDNNN"
    assert _word(out, "BBBUSDT") == "VVVDDNNNNNNN"
    # Slot 6: el mismo instante, dos causas segun el final de descarga de cada uno.
    assert _word(out, "AAAUSDT")[6] == "D"
    assert _word(out, "BBBUSDT")[6] == "N"


def test_data_end_mapping_missing_symbol_falls_back_to_max_last_ms() -> None:
    """Un simbolo que no esta en el mapeo usa el maximo de `last_ms` de los pasados."""
    specs = {"AAAUSDT": "vvv.........", "BBBUSDT": "vvvvvvv.....", "CCCUSDT": "vvvvv......."}
    out = _classify(specs, data_end_ms={"AAAUSDT": _slot(9)})

    assert _word(out, "AAAUSDT") == "VVVDDDDDDDNN"  # el suyo: hasta el slot 9
    assert _word(out, "BBBUSDT") == "VVVVVVVNNNNN"  # por defecto: ultima vela de todos (slot 6)
    assert _word(out, "CCCUSDT") == "VVVVVDDNNNNN"  # por defecto: DELISTED hasta el slot 6


def test_data_end_mapping_empty_or_with_foreign_keys_equals_the_default() -> None:
    """Un mapeo vacio, o con simbolos que no estan en la tabla, da lo mismo que `None`."""
    specs = {"AAAUSDT": "vvv.........", "BBBUSDT": "vvvvvvv....."}
    default = _classify(specs)

    pd.testing.assert_frame_equal(_classify(specs, data_end_ms={}), default)
    pd.testing.assert_frame_equal(_classify(specs, data_end_ms={"ZZZUSDT": _slot(11)}), default)


def test_data_end_int_equals_a_mapping_with_the_same_value_for_everyone() -> None:
    """Un entero se comporta como antes: es el final de todos los simbolos."""
    specs = {"AAAUSDT": "vvv.........", "BBBUSDT": "vvvvvvv.....", "CCCUSDT": "vvvvv......."}
    as_int = _classify(specs, data_end_ms=_slot(9))
    as_mapping = _classify(specs, data_end_ms={s: _slot(9) for s in specs})

    pd.testing.assert_frame_equal(as_int, as_mapping)
    assert _word(as_int, "AAAUSDT") == "VVVDDDDDDDNN"
    assert _word(as_int, "BBBUSDT") == "VVVVVVVDDDNN"
    assert _word(as_int, "CCCUSDT") == "VVVVVDDDDDNN"


@pytest.mark.parametrize(
    ("end_a", "expected_a"),
    [
        (2, "VVVNNNNNNNNN"),  # igual que antes de su ultima vela: nada deslistado
        (3, "VVVDNNNNNNNN"),  # un slot despues de la ultima vela: solo ese
        (6, "VVVDDDDNNNNN"),  # inclusivo: el slot 6 entra y el 7 no
    ],
)
def test_data_end_mapping_is_inclusive_per_symbol(end_a: int, expected_a: str) -> None:
    """El borde es inclusivo para cada simbolo, y el final de uno no mueve el del otro."""
    specs = {"AAAUSDT": "vvv.........", "BBBUSDT": "vvvvv......."}
    out = _classify(specs, data_end_ms={"AAAUSDT": _slot(end_a), "BBBUSDT": _slot(8)})

    assert _word(out, "AAAUSDT") == expected_a
    assert _word(out, "BBBUSDT") == "VVVVVDDDDNNN"


def test_data_end_mapping_before_last_candle_does_not_hide_candles() -> None:
    """Un final por simbolo anterior a su ultima vela no cambia las velas."""
    out = _classify({"AAAUSDT": "vvvvvvvv...."}, data_end_ms={"AAAUSDT": _slot(2)})
    assert _word(out, "AAAUSDT") == "VVVVVVVVNNNN"


def test_data_end_mapping_does_not_turn_a_series_without_cache_into_delisted() -> None:
    """Un simbolo sin cache con final en el mapeo sigue siendo `NO_DATA`, nunca `DELISTED`."""
    specs = {"AAAUSDT": "vvvvvvvv", "NONEUSDT": "........"}
    out = _classify(specs, data_end_ms={"AAAUSDT": _slot(7), "NONEUSDT": _slot(7)})

    assert _word(out, "NONEUSDT") == "NNNNNNNN"


def test_data_end_mapping_does_not_alter_outage() -> None:
    """Los finales por simbolo solo tocan `DELISTED`/`NO_DATA`; `OUTAGE` no cambia."""
    specs = {"AAAUSDT": "vv.vv...", "BBBUSDT": "vv.vv...", "CCCUSDT": "vvvvvvvv"}
    plain = _classify(specs, data_end_ms=_slot(7))
    mapped = _classify(specs, data_end_ms={s: _slot(7) for s in specs})

    pd.testing.assert_frame_equal(plain, mapped)
    assert _word(mapped, "AAAUSDT") == "VVOVVDDD"


# --- download_ends_from_universe ----------------------------------------------------------------


def _ends_universe(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Universo minimo con `symbol` y `period_end` como los de `load_universe`."""
    frame = pd.DataFrame(rows, columns=["symbol", "period_end"])
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    return frame


def _ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


def test_download_ends_take_the_latest_period_end_per_symbol_minus_one_ms() -> None:
    """El final de cada simbolo es su `period_end` maximo menos 1 ms, en ms UTC y como `int`."""
    universe = _ends_universe(
        [
            ("AAAUSDT", "2022-01-01"),
            ("BBBUSDT", "2023-01-01"),
            ("AAAUSDT", "2024-01-01"),
            ("AAAUSDT", "2023-01-01"),
        ]
    )
    ends = download_ends_from_universe(universe)

    assert ends == {"AAAUSDT": _ms("2024-01-01") - 1, "BBBUSDT": _ms("2023-01-01") - 1}
    assert all(type(v) is int for v in ends.values())


def test_download_ends_is_exactly_the_last_ms_before_the_period_end() -> None:
    """Menos 1 ms de las 00:00 del 1 de enero es 23:59:59.999 del 31-dic: nada del dia siguiente."""
    ends = download_ends_from_universe(_ends_universe([("AAAUSDT", "2023-01-01")]))

    assert ends["AAAUSDT"] == _ms("2022-12-31 23:59:59.999")
    assert ends["AAAUSDT"] == _ms("2023-01-01") - 1
    # La ultima vela de 1m del periodo (23:59) cabe; la de las 00:00 del dia siguiente no.
    assert _ms("2022-12-31 23:59") <= ends["AAAUSDT"] < _ms("2023-01-01")


def test_download_ends_from_empty_universe_is_empty() -> None:
    """Un universo sin filas da un dict vacio."""
    assert download_ends_from_universe(_ends_universe([])) == {}


def test_download_ends_missing_columns_raise() -> None:
    """Sin `symbol` o sin `period_end` da `ValueError` con las que faltan."""
    with pytest.raises(ValueError, match="period_end"):
        download_ends_from_universe(pd.DataFrame({"symbol": ["AAAUSDT"]}))
    with pytest.raises(ValueError, match="symbol"):
        download_ends_from_universe(pd.DataFrame({"period_end": []}))
    with pytest.raises(ValueError, match="period_end"):
        download_ends_from_universe(pd.DataFrame({"symbol": [], "first_candle": []}))


def test_download_ends_feed_classify_slots_as_a_mapping() -> None:
    """Lo que sale de `download_ends_from_universe` se puede pasar tal cual como `data_end_ms`."""
    universe = pd.DataFrame(
        {
            "symbol": ["AAAUSDT", "BBBUSDT"],
            "period_end": [
                pd.Timestamp(_slot(8) + 1, unit="ms", tz="UTC"),
                pd.Timestamp(_slot(4) + 1, unit="ms", tz="UTC"),
            ],
        }
    )
    ends = download_ends_from_universe(universe)
    out = _classify({"AAAUSDT": "vvv.........", "BBBUSDT": "vvv........."}, data_end_ms=ends)

    assert ends == {"AAAUSDT": _slot(8), "BBBUSDT": _slot(4)}
    assert _word(out, "AAAUSDT") == "VVVDDDDDDNNN"
    assert _word(out, "BBBUSDT") == "VVVDDNNNNNNN"


# --- No-fuga ------------------------------------------------------------------------------------
#
# Recortar el RANGO por la derecha, con la vida de la serie medida en toda la cache, no cambia
# ninguna causa. Recortar la CACHE (lo unico que se sabria en `t`) si cambia `HALTED`/`DELISTED`:
# es la naturaleza ex post de la clasificacion. `VALID`, `NO_TRADES` y `NOT_LISTED` solo dependen
# del propio slot (y del listado), y `OUTAGE` de los simbolos vivos en ese mismo slot.

LEAK_SPECS = {
    # A: sin operaciones en el slot 2, hueco en 10-11 (solo A: HALTED) y acaba en el 24.
    "AAAUSDT": "vvzvvvvvvv..vvvvvvvvvvvvv.....",
    # B y C: caen a la vez en 14-15 (2 de 4 vivos: OUTAGE) y siguen.
    "BBBUSDT": "vvvvvvvvvvvvvv..vvvvvvvvvvvvvv",
    "CCCUSDT": "vvvvvvvvvvvvvv..vvvvvvvvvvvvvv",
    # D: aparece en el slot 5 (listado el mismo dia: antes, NOT_LISTED).
    "DDDUSDT": ".....vvvvvvvvvvvvvvvvvvvvvvvvv",
}
LEAK_LISTED = {"DDDUSDT": _slot(2)}


def _classify_visible(cut: int) -> pd.DataFrame:
    """Clasifica solo lo visible hasta el slot `cut` (excluido), con la vida de lo recortado."""
    visible = {s: spec[:cut] for s, spec in LEAK_SPECS.items()}
    return _classify(visible, listed_ms=LEAK_LISTED, data_end_ms=_slot(cut - 1))


def _classify_full() -> pd.DataFrame:
    return _classify(LEAK_SPECS, listed_ms=LEAK_LISTED, data_end_ms=_slot(29))


def test_leak_full_table_has_all_the_causes_we_test() -> None:
    """Punto de partida: la tabla completa tiene HALTED, OUTAGE, NOT_LISTED y DELISTED."""
    full = _classify_full()

    assert _word(full, "AAAUSDT") == "VVTVVVVVVVHHVVVVVVVVVVVVVDDDDD"
    assert _word(full, "BBBUSDT") == "VVVVVVVVVVVVVVOOVVVVVVVVVVVVVV"
    assert _word(full, "DDDUSDT") == "LLLLLVVVVVVVVVVVVVVVVVVVVVVVVV"


@pytest.mark.parametrize("cut", [8, 12, 15, 20, 28])
def test_leak_stable_causes_do_not_change_when_the_cache_is_cut_on_the_right(cut: int) -> None:
    """Con la cache cortada, `VALID`, `NO_TRADES` y `NOT_LISTED` de los slots comunes no cambian."""
    full = _classify_full().iloc[:cut]
    visible = _classify_visible(cut)
    stable = [GapCause.VALID, GapCause.NO_TRADES, GapCause.NOT_LISTED]

    for cause in stable:
        pd.testing.assert_frame_equal(visible == cause, full == cause)


@pytest.mark.parametrize("cut", [20, 28])
def test_leak_outage_does_not_change_if_the_cut_falls_after_the_outage(cut: int) -> None:
    """`OUTAGE` no cambia si los simbolos ya habian vuelto antes del recorte."""
    full = _classify_full().iloc[:cut]
    visible = _classify_visible(cut)

    assert (full == GapCause.OUTAGE).to_numpy().sum() == 4
    pd.testing.assert_frame_equal(visible == GapCause.OUTAGE, full == GapCause.OUTAGE)


def test_leak_halted_and_delisted_do_change_when_the_cache_is_cut() -> None:
    """Documenta lo ex post: con la cache cortada dentro del hueco, `HALTED` pasa a `DELISTED`."""
    full = _classify_full()
    visible = _classify_visible(12)  # el hueco de A (10-11) llega al final de lo visible

    assert _word(full, "AAAUSDT")[10:12] == "HH"
    assert _word(visible, "AAAUSDT")[10:12] == "DD"
    # Y un slot que en la cache completa es DELISTED (A, 25-29) aqui ni existe: no se sabia.
    assert _word(full, "AAAUSDT")[25:] == "DDDDD"


def test_leak_outage_cut_inside_the_outage_becomes_delisted() -> None:
    """Si el recorte cae dentro de la caida, esos slots eran `DELISTED` para quien miraba en `t`."""
    full = _classify_full()
    visible = _classify_visible(15)  # B y C paran en el slot 13; el 14 es el ultimo visible

    assert _word(full, "BBBUSDT")[14] == "O"
    assert _word(visible, "BBBUSDT")[14] == "D"
    # La causa observable en `t` (hay vela o no) si es la misma: lo que cambia es la etiqueta.
    assert full["BBBUSDT"].iloc[14] != GapCause.VALID
    assert visible["BBBUSDT"].iloc[14] != GapCause.VALID


def test_leak_trimming_only_the_range_with_fixed_cache_bounds_changes_nothing() -> None:
    """Con vida y fin de datos fijos (toda la cache), recortar filas no cambia ninguna causa."""
    first, last = _life(LEAK_SPECS)
    full = classify_slots(
        _wide(LEAK_SPECS),
        first_ms=first,
        last_ms=last,
        listed_ms=LEAK_LISTED,
        data_end_ms=_slot(29),
    )
    for cut in (8, 12, 15, 20, 28):
        trimmed = classify_slots(
            _wide(LEAK_SPECS).iloc[:cut],
            first_ms=first,
            last_ms=last,
            listed_ms=LEAK_LISTED,
            data_end_ms=_slot(29),
        )
        pd.testing.assert_frame_equal(trimmed, full.iloc[:cut])


# --- classify_klines de punta a punta ---------------------------------------------------------


def _write_cache(base_dir: Path, symbol: str, spec: str) -> None:
    """Escribe en `base_dir` una cache de 1m de `symbol` con las velas que marca `spec`."""
    open_times = [_slot(k) for k, c in enumerate(spec) if c != "."]
    trades = [0 if c == "z" else 7 for c in spec if c != "."]
    n = len(open_times)
    data = {
        "open_time": open_times,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1.0] * n,
        "close_time": [t + MIN_MS - 1 for t in open_times],
        "quote_asset_volume": [100.0] * n,
        "num_trades": trades,
        "taker_buy_base": [0.5] * n,
        "taker_buy_quote": [50.0] * n,
        "ignore": [0.0] * n,
    }
    frame = pd.DataFrame(data, columns=KLINE_COLUMNS).astype(KLINE_DTYPES)
    save_klines(frame, symbol, "1m", base_dir=base_dir)


E2E_SPECS = {
    "AAAUSDT": "..vzv..vvv..vvv",  # vida: slots 2-14
    "BBBUSDT": "vvvvvvvvvvvvvvvvvvvv",  # vida: slots 0-19 (el fin de datos de la cache)
}


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """Cache temporal con AAA y BBB; `NONEUSDT` no tiene."""
    for symbol, spec in E2E_SPECS.items():
        _write_cache(tmp_path, symbol, spec)
    return tmp_path


def test_classify_klines_end_to_end(cache: Path, caplog: pytest.LogCaptureFixture) -> None:
    """De la cache a las causas: velas, huecos, listado, deslistado y simbolo sin cache."""
    with caplog.at_level(logging.WARNING):
        out = classify_klines(
            ["AAAUSDT", "BBBUSDT", "NONEUSDT"],
            "1m",
            _slot(0),
            _slot(16),
            base_dir=cache,
            listed_ms={"AAAUSDT": _slot(1), "NONEUSDT": _slot(3)},
        )

    assert out.shape == (16, 3)
    assert (out.dtypes == "int8").all()
    assert out.index.equals(build_time_grid(_slot(0), _slot(16), "1m"))
    assert list(out.columns) == ["AAAUSDT", "BBBUSDT", "NONEUSDT"]
    assert _word(out, "AAAUSDT") == "LLVTVHHVVVHHVVVD"
    assert _word(out, "BBBUSDT") == "V" * 16
    assert _word(out, "NONEUSDT") == "LLLNNNNNNNNNNNNN"
    assert any("NONEUSDT" in record.getMessage() for record in caplog.records)


def test_classify_klines_uses_whole_cache_life_not_the_requested_range(cache: Path) -> None:
    """Un rango dentro del hueco de A lo ve `HALTED`; uno tras su ultima vela, `DELISTED`."""
    inside = classify_klines(["AAAUSDT", "BBBUSDT"], "1m", _slot(5), _slot(11), base_dir=cache)
    after = classify_klines(["AAAUSDT", "BBBUSDT"], "1m", _slot(15), _slot(20), base_dir=cache)

    assert _word(inside, "AAAUSDT") == "HHVVVH"
    assert _word(after, "AAAUSDT") == "DDDDD"


def test_classify_klines_data_end_default_and_explicit(cache: Path) -> None:
    """Por defecto el fin de datos es la ultima vela de los simbolos pasados; explicito, el dado."""
    only_a = classify_klines(["AAAUSDT"], "1m", _slot(10), _slot(20), base_dir=cache)
    explicit = classify_klines(
        ["AAAUSDT"], "1m", _slot(10), _slot(20), base_dir=cache, data_end_ms=_slot(19)
    )
    with_b = classify_klines(["AAAUSDT", "BBBUSDT"], "1m", _slot(10), _slot(20), base_dir=cache)

    assert _word(only_a, "AAAUSDT") == "HHVVVNNNNN"
    assert _word(explicit, "AAAUSDT") == "HHVVVDDDDD"
    assert _word(with_b, "AAAUSDT") == "HHVVVDDDDD"


def test_classify_klines_detects_outage_across_cached_symbols(tmp_path: Path) -> None:
    """Con tres simbolos que dejan de cotizar a la vez en la cache, el hueco es `OUTAGE`."""
    for symbol, spec in {"AAAUSDT": "vv..vv", "BBBUSDT": "vv..vv", "CCCUSDT": "vvvvvv"}.items():
        _write_cache(tmp_path, symbol, spec)
    out = classify_klines(
        ["AAAUSDT", "BBBUSDT", "CCCUSDT"], "1m", _slot(0), _slot(6), base_dir=tmp_path
    )

    assert _word(out, "AAAUSDT") == "VVOOVV"
    assert _word(out, "CCCUSDT") == "VVVVVV"


def test_classify_klines_deduplicates_symbols(cache: Path) -> None:
    """Un simbolo repetido en la lista sale una sola vez, sin error de columnas duplicadas."""
    out = classify_klines(["AAAUSDT", "AAAUSDT"], "1m", _slot(0), _slot(4), base_dir=cache)
    assert list(out.columns) == ["AAAUSDT"]


def test_classify_klines_trimming_the_range_changes_no_cause(cache: Path) -> None:
    """No-fuga: con la misma cache, recortar el rango por la derecha no cambia ninguna causa."""
    symbols = ["AAAUSDT", "BBBUSDT", "NONEUSDT"]
    listed = {"AAAUSDT": _slot(1), "NONEUSDT": _slot(3)}
    wide = classify_klines(symbols, "1m", _slot(0), _slot(20), base_dir=cache, listed_ms=listed)

    for end in (5, 8, 12, 15):
        short = classify_klines(
            symbols, "1m", _slot(0), _slot(end), base_dir=cache, listed_ms=listed
        )
        pd.testing.assert_frame_equal(short, wide.iloc[:end])


def test_classify_klines_data_end_mapping_per_symbol(cache: Path) -> None:
    """Con un mapeo, cada simbolo de la cache usa su final; el que falta, el por defecto."""
    ends = {"AAAUSDT": _slot(17)}  # BBB no esta: su por defecto es la ultima vela de los pasados
    both = classify_klines(
        ["AAAUSDT", "BBBUSDT"], "1m", _slot(10), _slot(20), base_dir=cache, data_end_ms=ends
    )
    only_a = classify_klines(
        ["AAAUSDT", "NONEUSDT"], "1m", _slot(10), _slot(20), base_dir=cache, data_end_ms=ends
    )
    default = classify_klines(["AAAUSDT", "BBBUSDT"], "1m", _slot(10), _slot(20), base_dir=cache)

    assert _word(both, "AAAUSDT") == "HHVVVDDDNN"
    assert _word(both, "BBBUSDT") == "V" * 10
    # Sin mapeo (por defecto = slot 19 de BBB) el mismo simbolo seria DELISTED hasta el final.
    assert _word(default, "AAAUSDT") == "HHVVVDDDDD"
    assert _word(only_a, "AAAUSDT") == "HHVVVDDDNN"
    assert _word(only_a, "NONEUSDT") == "N" * 10


def test_classify_klines_data_end_mapping_from_universe(cache: Path) -> None:
    """Del universo a `classify_klines`, con un final previo a la ultima vela cacheada."""
    universe = pd.DataFrame(
        {
            "symbol": ["AAAUSDT", "BBBUSDT"],
            "period_end": [
                pd.Timestamp(_slot(13) + 1, unit="ms", tz="UTC"),  # AAA: su ultima vela es el 14
                pd.Timestamp(_slot(19) + 1, unit="ms", tz="UTC"),
            ],
        }
    )
    out = classify_klines(
        ["AAAUSDT", "BBBUSDT"],
        "1m",
        _slot(10),
        _slot(20),
        base_dir=cache,
        data_end_ms=download_ends_from_universe(universe),
    )

    # El final de AAA (slot 13) es anterior a su ultima vela (14): la vela sigue siendo VALID y
    # despues, sin descarga, todo es NO_DATA; no se inventa un deslistado.
    assert _word(out, "AAAUSDT") == "HHVVVNNNNN"
    assert _word(out, "BBBUSDT") == "V" * 10
