"""Tests del informe de cobertura temporal de la cache de klines, con caches en `tmp_path`."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from miax.ingest import coverage_from_frame, coverage_report, save_klines, universe_coverage
from miax.ingest.coverage import COVERAGE_COLUMNS
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES

MIN_MS = 60_000
HOUR_MS = 3_600_000
WEEK_MS = 7 * 86_400_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z (jueves)
MONDAY = T0 + 4 * 86_400_000  # 2026-01-05T00:00:00Z, apertura de vela semanal
STATS_KEYS = [c for c in COVERAGE_COLUMNS if c != "symbol"]


def _frame(
    open_times: list[int], step: int = HOUR_MS, trades: list[int] | None = None
) -> pd.DataFrame:
    """Klines coherentes y con dtypes canonicos para esos `open_time`; `trades` fija num_trades."""
    n = len(open_times)
    data = {
        "open_time": open_times,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1.5] * n,
        "close_time": [t + step - 1 for t in open_times],
        "quote_asset_volume": [150.0] * n,
        "num_trades": [10] * n if trades is None else trades,
        "taker_buy_base": [0.7] * n,
        "taker_buy_quote": [70.0] * n,
        "ignore": [0.0] * n,
    }
    return pd.DataFrame(data).astype(KLINE_DTYPES)[KLINE_COLUMNS]


def _minutes(slots: Iterable[int], base: int = T0) -> list[int]:
    """`open_time` de velas de 1m en las ranuras indicadas a partir de `base`."""
    return [base + s * MIN_MS for s in slots]


def _hours(slots: Iterable[int], base: int = T0) -> list[int]:
    """`open_time` de velas de 1h en las ranuras indicadas a partir de `base`."""
    return [base + s * HOUR_MS for s in slots]


# --- coverage_from_frame: sin rango ---


def test_no_range_complete_series() -> None:
    """Una serie completa tiene cobertura 100 %, sin huecos ni bordes perdidos."""
    result = coverage_from_frame(_frame(_hours(range(5))), "1h")
    assert result["expected"] == 5
    assert result["present"] == 5
    assert result["coverage_pct"] == pytest.approx(100.0)
    assert result["traded_pct"] == pytest.approx(100.0)
    assert result["n_gaps"] == 0
    assert result["max_gap_candles"] == 0
    assert result["missing_before"] == 0
    assert result["missing_after"] == 0
    assert result["first_open"] == pd.Timestamp(T0, unit="ms", tz="UTC")
    assert result["last_open"] == pd.Timestamp(T0 + 4 * HOUR_MS, unit="ms", tz="UTC")


def test_result_keys_match_columns_without_symbol() -> None:
    """El diccionario devuelto tiene exactamente las claves de COVERAGE_COLUMNS menos symbol."""
    result = coverage_from_frame(_frame(_hours([0, 1])), "1h")
    assert list(result) == STATS_KEYS


def test_no_range_internal_gaps() -> None:
    """Los huecos internos se cuentan y el mayor se mide en velas."""
    # Ranuras 0,1,2, hueco de 2 (3,4), 5, hueco de 3 (6,7,8), 9 -> 10 esperadas, 5 presentes.
    result = coverage_from_frame(_frame(_hours([0, 1, 2, 5, 9])), "1h")
    assert result["expected"] == 10
    assert result["present"] == 5
    assert result["coverage_pct"] == pytest.approx(50.0)
    assert result["n_gaps"] == 2
    assert result["max_gap_candles"] == 3
    assert result["missing_before"] == 0
    assert result["missing_after"] == 0


def test_no_range_unaligned_candles_ignored_and_present_never_exceeds_expected() -> None:
    """Sin rango, las velas fuera de la rejilla se ignoran: present <= expected."""
    result = coverage_from_frame(_frame([0, 30_000, 60_000], step=MIN_MS), "1m")
    assert result["present"] == 2
    assert result["expected"] == 2
    assert result["present"] <= result["expected"]
    assert result["coverage_pct"] == pytest.approx(100.0)
    assert result["n_gaps"] == 0


def test_no_range_unaligned_candle_at_the_edges_does_not_extend_the_range() -> None:
    """Una vela desalineada antes de la primera alineada no ensancha el rango esperado."""
    result = coverage_from_frame(_frame([30_000, 60_000, 120_000], step=MIN_MS), "1m")
    assert result["expected"] == 2
    assert result["present"] == 2
    assert result["first_open"] == pd.Timestamp(60_000, unit="ms", tz="UTC")


def test_no_range_only_unaligned_candles_is_empty() -> None:
    """Si ninguna vela esta alineada, no hay nada esperado ni presente y los porcentajes son NaN."""
    result = coverage_from_frame(_frame([30_000, 90_000], step=MIN_MS), "1m")
    assert result["expected"] == 0
    assert result["present"] == 0
    assert math.isnan(result["coverage_pct"])
    assert math.isnan(result["traded_pct"])
    assert result["first_open"] is pd.NaT
    assert result["last_open"] is pd.NaT


def test_no_range_empty_frame() -> None:
    """Un DataFrame vacio da expected 0, present 0, porcentajes NaN y fechas NaT."""
    result = coverage_from_frame(_frame([]), "1h")
    assert result["expected"] == 0
    assert result["present"] == 0
    assert math.isnan(result["coverage_pct"])
    assert math.isnan(result["traded_pct"])
    assert result["first_open"] is pd.NaT
    assert result["last_open"] is pd.NaT
    assert result["n_gaps"] == 0
    assert result["max_gap_candles"] == 0
    assert result["missing_before"] == 0
    assert result["missing_after"] == 0


def test_no_range_single_candle() -> None:
    """Una sola vela: 1 esperada, 1 presente."""
    result = coverage_from_frame(_frame(_hours([3])), "1h")
    assert result["expected"] == 1
    assert result["present"] == 1
    assert result["coverage_pct"] == pytest.approx(100.0)
    assert result["first_open"] == result["last_open"]


def test_duplicates_are_counted_once() -> None:
    """Un `open_time` repetido cuenta como una sola vela."""
    frame = pd.concat([_frame(_hours([0, 1, 2])), _frame(_hours([1, 2]))], ignore_index=True)
    result = coverage_from_frame(frame, "1h")
    assert result["present"] == 3
    assert result["expected"] == 3
    assert result["coverage_pct"] == pytest.approx(100.0)


def test_unsorted_input_is_handled() -> None:
    """El orden de entrada no cambia el resultado."""
    ordered = coverage_from_frame(_frame(_hours([0, 1, 4, 5])), "1h")
    shuffled = coverage_from_frame(_frame(_hours([5, 0, 4, 1])), "1h")
    assert ordered == shuffled


def test_traded_pct_counts_candles_with_trades() -> None:
    """`traded_pct` es el % de velas esperadas con num_trades > 0."""
    frame = _frame(_hours(range(5)), trades=[10, 0, 3, 0, 7])
    result = coverage_from_frame(frame, "1h")
    assert result["present"] == 5
    assert result["coverage_pct"] == pytest.approx(100.0)
    assert result["traded_pct"] == pytest.approx(60.0)


def test_traded_pct_is_relative_to_expected_not_present() -> None:
    """Con huecos, `traded_pct` se divide por las esperadas, no por las presentes."""
    # Esperadas 0..3 (4); presentes 0,1,3; con operaciones 0 y 3 -> 2/4 = 50 %.
    frame = _frame(_hours([0, 1, 3]), trades=[5, 0, 5])
    result = coverage_from_frame(frame, "1h")
    assert result["expected"] == 4
    assert result["coverage_pct"] == pytest.approx(75.0)
    assert result["traded_pct"] == pytest.approx(50.0)


def test_traded_pct_all_zero_trades() -> None:
    """Si ninguna vela tiene operaciones, traded_pct es 0 aunque la cobertura sea 100 %."""
    result = coverage_from_frame(_frame(_hours(range(3)), trades=[0, 0, 0]), "1h")
    assert result["coverage_pct"] == pytest.approx(100.0)
    assert result["traded_pct"] == pytest.approx(0.0)


# --- coverage_from_frame: con rango ---


def test_range_cut_series_reports_missing_before_and_after() -> None:
    """Serie cortada tipo LUNA: empieza tarde y acaba pronto dentro del periodo."""
    # Rango de 10 ranuras de 1m; velas en las ranuras 2..6.
    frame = _frame(_minutes(range(2, 7)), step=MIN_MS)
    result = coverage_from_frame(frame, "1m", T0, T0 + 10 * MIN_MS)
    assert result["expected"] == 10
    assert result["present"] == 5
    assert result["coverage_pct"] == pytest.approx(50.0)
    assert result["missing_before"] == 2
    assert result["missing_after"] == 3
    assert result["n_gaps"] == 0
    assert result["first_open"] == pd.Timestamp(T0 + 2 * MIN_MS, unit="ms", tz="UTC")
    assert result["last_open"] == pd.Timestamp(T0 + 6 * MIN_MS, unit="ms", tz="UTC")


def test_range_end_is_exclusive_and_start_inclusive() -> None:
    """El rango es semiabierto: la vela de `start_ms` entra y la de `end_ms` no."""
    frame = _frame(_minutes(range(6)), step=MIN_MS)
    result = coverage_from_frame(frame, "1m", T0 + MIN_MS, T0 + 5 * MIN_MS)
    assert result["expected"] == 4
    assert result["present"] == 4
    assert result["first_open"] == pd.Timestamp(T0 + MIN_MS, unit="ms", tz="UTC")
    assert result["last_open"] == pd.Timestamp(T0 + 4 * MIN_MS, unit="ms", tz="UTC")
    assert result["missing_before"] == 0
    assert result["missing_after"] == 0


def test_range_ignores_candles_outside() -> None:
    """Las velas anteriores o posteriores al rango no cuentan como presentes."""
    frame = _frame(_minutes([-3, -1, 0, 1, 2, 3, 4, 7]), step=MIN_MS)
    result = coverage_from_frame(frame, "1m", T0, T0 + 5 * MIN_MS)
    assert result["expected"] == 5
    assert result["present"] == 5
    assert result["coverage_pct"] == pytest.approx(100.0)


def test_range_with_unaligned_bounds_counts_grid_slots() -> None:
    """Los bordes del rango no alineados solo cuentan las ranuras de la rejilla que caen dentro."""
    frame = _frame(_minutes(range(8)), step=MIN_MS)
    # [30 s, 270 s) contiene las ranuras de 60, 120, 180 y 240 s.
    result = coverage_from_frame(frame, "1m", T0 + 30_000, T0 + 270_000)
    assert result["expected"] == 4
    assert result["present"] == 4


def test_range_unaligned_candles_ignored() -> None:
    """Con rango, una vela desalineada dentro del rango no cuenta."""
    frame = _frame([T0, T0 + 30_000, T0 + MIN_MS], step=MIN_MS)
    result = coverage_from_frame(frame, "1m", T0, T0 + 2 * MIN_MS)
    assert result["expected"] == 2
    assert result["present"] == 2
    assert result["present"] <= result["expected"]


def test_range_empty_frame_all_missing_before() -> None:
    """Sin velas y con rango, todo lo esperado queda 'antes' y nada 'despues'."""
    result = coverage_from_frame(_frame([]), "1m", T0, T0 + 10 * MIN_MS)
    assert result["expected"] == 10
    assert result["present"] == 0
    assert result["coverage_pct"] == pytest.approx(0.0)
    assert result["traded_pct"] == pytest.approx(0.0)
    assert result["missing_before"] == 10
    assert result["missing_after"] == 0
    assert result["first_open"] is pd.NaT
    assert result["last_open"] is pd.NaT


def test_range_all_candles_outside_behaves_like_empty() -> None:
    """Si todas las velas caen fuera del rango, el resultado es el de una serie vacia."""
    frame = _frame(_minutes(range(20, 25)), step=MIN_MS)
    result = coverage_from_frame(frame, "1m", T0, T0 + 10 * MIN_MS)
    assert result["present"] == 0
    assert result["expected"] == 10
    assert result["missing_before"] == 10
    assert result["missing_after"] == 0


def test_range_internal_gaps_measured_inside_range_only() -> None:
    """Los huecos internos solo miran las velas dentro del rango."""
    # Velas 0, 1, 5, 6 dentro; 20 fuera del rango no crea un hueco.
    frame = _frame(_minutes([0, 1, 5, 6, 20]), step=MIN_MS)
    result = coverage_from_frame(frame, "1m", T0, T0 + 8 * MIN_MS)
    assert result["expected"] == 8
    assert result["present"] == 4
    assert result["n_gaps"] == 1
    assert result["max_gap_candles"] == 3
    assert result["missing_before"] == 0
    assert result["missing_after"] == 1


def test_range_traded_pct_with_zero_trade_candles() -> None:
    """Con rango, traded_pct cuenta solo velas del rango con num_trades > 0."""
    frame = _frame(_minutes(range(4)), step=MIN_MS, trades=[0, 5, 0, 9])
    result = coverage_from_frame(frame, "1m", T0, T0 + 4 * MIN_MS)
    assert result["traded_pct"] == pytest.approx(50.0)
    assert result["coverage_pct"] == pytest.approx(100.0)


def test_range_duplicates_counted_once() -> None:
    """Con rango, los duplicados tampoco inflan present por encima de expected."""
    frame = pd.concat(
        [_frame(_minutes([0, 1]), step=MIN_MS), _frame(_minutes([0, 1]), step=MIN_MS)],
        ignore_index=True,
    )
    result = coverage_from_frame(frame, "1m", T0, T0 + 2 * MIN_MS)
    assert result["present"] == 2
    assert result["coverage_pct"] == pytest.approx(100.0)


# --- coverage_from_frame: intervalo semanal ---


def test_weekly_interval_uses_monday_offset() -> None:
    """Las velas semanales se alinean a lunes 00:00 UTC, no al jueves de la epoca."""
    times = [MONDAY, MONDAY + WEEK_MS, MONDAY + 3 * WEEK_MS]
    result = coverage_from_frame(_frame(times, step=WEEK_MS), "1w")
    assert result["expected"] == 4
    assert result["present"] == 3
    assert result["n_gaps"] == 1
    assert result["max_gap_candles"] == 1


def test_weekly_range_counts_mondays() -> None:
    """Con rango semanal solo cuentan las ranuras de lunes dentro del rango."""
    times = [MONDAY, MONDAY + WEEK_MS]
    # [jueves 1-ene, lunes + 2 semanas) contiene dos aperturas de lunes.
    result = coverage_from_frame(_frame(times, step=WEEK_MS), "1w", T0, MONDAY + 2 * WEEK_MS)
    assert result["expected"] == 2
    assert result["present"] == 2


def test_weekly_candle_on_epoch_thursday_is_unaligned() -> None:
    """Una vela semanal en jueves (rejilla de la epoca) no esta alineada y se ignora."""
    result = coverage_from_frame(_frame([T0, MONDAY], step=WEEK_MS), "1w")
    assert result["present"] == 1
    assert result["expected"] == 1
    assert result["first_open"] == pd.Timestamp(MONDAY, unit="ms", tz="UTC")


# --- coverage_from_frame: errores ---


def test_monthly_interval_raises() -> None:
    """`1M` no tiene duracion fija: ValueError."""
    with pytest.raises(ValueError, match="duracion fija"):
        coverage_from_frame(_frame(_hours([0, 1])), "1M")


def test_invalid_interval_raises() -> None:
    """Un intervalo que no se reconoce da ValueError."""
    with pytest.raises(ValueError):
        coverage_from_frame(_frame(_hours([0, 1])), "foo")


def test_missing_columns_raise() -> None:
    """Si faltan `open_time` o `num_trades`, ValueError."""
    with pytest.raises(ValueError, match="num_trades"):
        coverage_from_frame(_frame(_hours([0, 1])).drop(columns="num_trades"), "1h")
    with pytest.raises(ValueError, match="open_time"):
        coverage_from_frame(_frame(_hours([0, 1])).drop(columns="open_time"), "1h")


@pytest.mark.parametrize(("start", "end"), [(T0, None), (None, T0 + HOUR_MS)])
def test_half_range_raises(start: int | None, end: int | None) -> None:
    """Dar solo uno de `start_ms` y `end_ms` es ValueError."""
    with pytest.raises(ValueError, match="juntos"):
        coverage_from_frame(_frame(_hours([0, 1])), "1h", start, end)


@pytest.mark.parametrize("end", [T0, T0 - HOUR_MS])
def test_empty_or_inverted_range_raises(end: int) -> None:
    """Un rango vacio (start == end) o invertido es ValueError."""
    with pytest.raises(ValueError, match="Rango"):
        coverage_from_frame(_frame(_hours([0, 1])), "1h", T0, end)


def test_input_frame_is_not_modified() -> None:
    """La funcion no altera el DataFrame de entrada."""
    frame = pd.concat([_frame(_hours([3, 1, 1, 0]))], ignore_index=True)
    before = frame.copy()
    coverage_from_frame(frame, "1h", T0, T0 + 5 * HOUR_MS)
    pd.testing.assert_frame_equal(frame, before)


# --- coverage_report ---


def test_report_has_one_row_per_symbol_in_input_order(tmp_path: Path) -> None:
    """Una fila por simbolo, en el orden de entrada y con las columnas canonicas."""
    save_klines(_frame(_hours(range(4))), "BTCUSDT", "1h", base_dir=tmp_path)
    save_klines(_frame(_hours(range(2, 6))), "ETHUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(["ETHUSDT", "BTCUSDT"], "1h", base_dir=tmp_path)
    assert list(report.columns) == COVERAGE_COLUMNS
    assert list(report["symbol"]) == ["ETHUSDT", "BTCUSDT"]
    assert list(report["present"]) == [4, 4]
    assert report["first_open"].dt.tz is not None
    assert report.loc[1, "first_open"] == pd.Timestamp(T0, unit="ms", tz="UTC")


def test_report_symbol_without_cache_has_zero_present_row(tmp_path: Path) -> None:
    """Un simbolo sin cache no se omite: fila con present 0 y fechas NaT."""
    save_klines(_frame(_hours(range(3))), "BTCUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(["BTCUSDT", "ETHUSDT"], "1h", base_dir=tmp_path)
    assert list(report["symbol"]) == ["BTCUSDT", "ETHUSDT"]
    eth = report.loc[report["symbol"] == "ETHUSDT"].iloc[0]
    assert eth["present"] == 0
    assert eth["expected"] == 0
    assert pd.isna(eth["first_open"])
    assert pd.isna(eth["last_open"])


def test_report_without_cache_and_with_range_counts_expected_slots(tmp_path: Path) -> None:
    """Sin cache y con rango, todo el rango queda esperado y ausente."""
    report = coverage_report(
        ["ETHUSDT"], "1h", base_dir=tmp_path, start_ms=T0, end_ms=T0 + 6 * HOUR_MS
    )
    assert report.loc[0, "present"] == 0
    assert report.loc[0, "expected"] == 6
    assert report.loc[0, "missing_before"] == 6


def test_report_duplicated_symbols_appear_once(tmp_path: Path) -> None:
    """Un simbolo repetido en la entrada produce una sola fila."""
    save_klines(_frame(_hours(range(3))), "BTCUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(["BTCUSDT", "BTCUSDT"], "1h", base_dir=tmp_path)
    assert list(report["symbol"]) == ["BTCUSDT"]


def test_report_empty_symbol_list_returns_empty_frame_with_columns(tmp_path: Path) -> None:
    """Una lista vacia da un DataFrame vacio con las columnas del informe."""
    report = coverage_report([], "1h", base_dir=tmp_path)
    assert report.empty
    assert list(report.columns) == COVERAGE_COLUMNS
    assert isinstance(report["first_open"].dtype, pd.DatetimeTZDtype)


def test_report_range_end_is_exclusive(tmp_path: Path) -> None:
    """Con rango la vela de `end_ms` no cuenta aunque este en la cache."""
    save_klines(_frame(_hours(range(10))), "BTCUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(
        ["BTCUSDT"], "1h", base_dir=tmp_path, start_ms=T0 + 2 * HOUR_MS, end_ms=T0 + 6 * HOUR_MS
    )
    row = report.iloc[0]
    assert row["expected"] == 4
    assert row["present"] == 4
    assert row["last_open"] == pd.Timestamp(T0 + 5 * HOUR_MS, unit="ms", tz="UTC")
    assert row["missing_after"] == 0


def test_report_range_detects_cut_series(tmp_path: Path) -> None:
    """Con rango, una serie que acaba pronto marca missing_after."""
    save_klines(_frame(_hours(range(4))), "LUNAUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(
        ["LUNAUSDT"], "1h", base_dir=tmp_path, start_ms=T0, end_ms=T0 + 10 * HOUR_MS
    )
    row = report.iloc[0]
    assert row["present"] == 4
    assert row["expected"] == 10
    assert row["missing_after"] == 6
    assert row["coverage_pct"] == pytest.approx(40.0)


def test_report_traded_pct_from_cache(tmp_path: Path) -> None:
    """El informe lee num_trades de la cache para traded_pct."""
    save_klines(_frame(_hours(range(4)), trades=[0, 0, 5, 5]), "BTCUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(["BTCUSDT"], "1h", base_dir=tmp_path)
    assert report.loc[0, "traded_pct"] == pytest.approx(50.0)


def test_report_across_month_files(tmp_path: Path) -> None:
    """Una serie repartida en varios ficheros mensuales se mide como una sola."""
    feb = 1_769_904_000_000  # 2026-02-01T00:00:00Z
    times = [feb - 2 * HOUR_MS, feb - HOUR_MS, feb, feb + HOUR_MS]
    save_klines(_frame(times), "BTCUSDT", "1h", base_dir=tmp_path)
    report = coverage_report(["BTCUSDT"], "1h", base_dir=tmp_path)
    assert report.loc[0, "present"] == 4
    assert report.loc[0, "expected"] == 4
    assert report.loc[0, "n_gaps"] == 0


def test_report_invalid_symbol_raises(tmp_path: Path) -> None:
    """Un simbolo con caracteres no validos lanza ValueError."""
    with pytest.raises(ValueError):
        coverage_report(["../x"], "1h", base_dir=tmp_path)


def test_report_monthly_interval_raises(tmp_path: Path) -> None:
    """`1M` lanza ValueError aunque no haya cache."""
    with pytest.raises(ValueError):
        coverage_report(["BTCUSDT"], "1M", base_dir=tmp_path)


# --- universe_coverage ---


def _universe(rows: list[tuple[int, int, str, str]]) -> pd.DataFrame:
    """Universo con las columnas de `load_universe` para filas (inicio, fin, simbolo, categoria)."""
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "period_end": pd.to_datetime([r[1] for r in rows], unit="ms", utc=True),
            "symbol": [r[2] for r in rows],
            "category": [r[3] for r in rows],
        }
    )


def test_universe_coverage_columns_and_order(tmp_path: Path) -> None:
    """Columnas period_start, period_end, category + COVERAGE_COLUMNS, en el orden del universo."""
    save_klines(_frame(_hours(range(10))), "AUSDT", "1h", base_dir=tmp_path)
    save_klines(_frame(_hours(range(10))), "BUSDT", "1h", base_dir=tmp_path)
    universe = _universe(
        [
            (T0, T0 + 10 * HOUR_MS, "BUSDT", "l2"),
            (T0, T0 + 10 * HOUR_MS, "AUSDT", "l1"),
        ]
    )
    report = universe_coverage(universe, "1h", base_dir=tmp_path)
    assert list(report.columns) == ["period_start", "period_end", "category", *COVERAGE_COLUMNS]
    assert list(report["symbol"]) == ["BUSDT", "AUSDT"]
    assert list(report["category"]) == ["l2", "l1"]
    assert list(report["present"]) == [10, 10]
    assert report["period_start"].dt.tz is not None
    assert report.loc[0, "period_start"] == pd.Timestamp(T0, unit="ms", tz="UTC")


def test_universe_coverage_measures_each_pair_in_its_period(tmp_path: Path) -> None:
    """Cada par se mide solo dentro de su periodo."""
    save_klines(_frame(_hours(range(20))), "AUSDT", "1h", base_dir=tmp_path)
    universe = _universe(
        [
            (T0, T0 + 10 * HOUR_MS, "AUSDT", "l1"),
            (T0 + 10 * HOUR_MS, T0 + 20 * HOUR_MS, "AUSDT", "l1"),
        ]
    )
    report = universe_coverage(universe, "1h", base_dir=tmp_path)
    assert list(report["expected"]) == [10, 10]
    assert list(report["present"]) == [10, 10]


def test_universe_coverage_current_period_is_cut_by_now(tmp_path: Path) -> None:
    """Un periodo en curso se mide solo hasta `now_ms`."""
    save_klines(_frame(_hours(range(10))), "AUSDT", "1h", base_dir=tmp_path)
    universe = _universe([(T0, T0 + 10 * HOUR_MS, "AUSDT", "l1")])
    report = universe_coverage(universe, "1h", base_dir=tmp_path, now_ms=T0 + 4 * HOUR_MS)
    assert report.loc[0, "expected"] == 4
    assert report.loc[0, "present"] == 4
    assert report.loc[0, "coverage_pct"] == pytest.approx(100.0)
    assert report.loc[0, "missing_after"] == 0


def test_universe_coverage_without_now_uses_full_period(tmp_path: Path) -> None:
    """Sin `now_ms` se usa el periodo entero, aunque este por llenar."""
    save_klines(_frame(_hours(range(4))), "AUSDT", "1h", base_dir=tmp_path)
    universe = _universe([(T0, T0 + 10 * HOUR_MS, "AUSDT", "l1")])
    report = universe_coverage(universe, "1h", base_dir=tmp_path)
    assert report.loc[0, "expected"] == 10
    assert report.loc[0, "present"] == 4
    assert report.loc[0, "missing_after"] == 6


def test_universe_coverage_now_after_period_end_does_not_extend_it(tmp_path: Path) -> None:
    """Un `now_ms` posterior al fin del periodo no lo alarga."""
    save_klines(_frame(_hours(range(20))), "AUSDT", "1h", base_dir=tmp_path)
    universe = _universe([(T0, T0 + 10 * HOUR_MS, "AUSDT", "l1")])
    report = universe_coverage(universe, "1h", base_dir=tmp_path, now_ms=T0 + 15 * HOUR_MS)
    assert report.loc[0, "expected"] == 10


def test_universe_coverage_future_period_is_omitted(tmp_path: Path) -> None:
    """Un periodo que empieza en o despues de `now_ms` no genera fila."""
    save_klines(_frame(_hours(range(30))), "AUSDT", "1h", base_dir=tmp_path)
    universe = _universe(
        [
            (T0, T0 + 10 * HOUR_MS, "AUSDT", "l1"),
            (T0 + 10 * HOUR_MS, T0 + 20 * HOUR_MS, "AUSDT", "l1"),
            (T0 + 20 * HOUR_MS, T0 + 30 * HOUR_MS, "AUSDT", "l1"),
        ]
    )
    report = universe_coverage(universe, "1h", base_dir=tmp_path, now_ms=T0 + 10 * HOUR_MS)
    assert len(report) == 1
    assert report.loc[0, "period_start"] == pd.Timestamp(T0, unit="ms", tz="UTC")


def test_universe_coverage_all_future_gives_empty_frame_with_columns(tmp_path: Path) -> None:
    """Si todos los periodos son futuros, el informe esta vacio pero con sus columnas."""
    universe = _universe([(T0 + 10 * HOUR_MS, T0 + 20 * HOUR_MS, "AUSDT", "l1")])
    report = universe_coverage(universe, "1h", base_dir=tmp_path, now_ms=T0)
    assert report.empty
    assert list(report.columns) == ["period_start", "period_end", "category", *COVERAGE_COLUMNS]


def test_universe_coverage_empty_universe(tmp_path: Path) -> None:
    """Un universo sin filas da un informe vacio con columnas."""
    universe = _universe([])
    report = universe_coverage(universe, "1h", base_dir=tmp_path)
    assert report.empty
    assert list(report.columns) == ["period_start", "period_end", "category", *COVERAGE_COLUMNS]


def test_universe_coverage_pair_without_cache_has_zero_present(tmp_path: Path) -> None:
    """Un par sin cache aparece con present 0 y todo lo esperado como ausente."""
    universe = _universe([(T0, T0 + 10 * HOUR_MS, "AUSDT", "l1")])
    report = universe_coverage(universe, "1h", base_dir=tmp_path)
    assert report.loc[0, "present"] == 0
    assert report.loc[0, "expected"] == 10
    assert report.loc[0, "missing_before"] == 10


def test_universe_coverage_detects_late_listing_and_zero_trades(tmp_path: Path) -> None:
    """Un par que lista tarde y con minutos sin operaciones se refleja en el informe."""
    save_klines(_frame(_hours(range(6, 10)), trades=[0, 0, 4, 4]), "AUSDT", "1h", base_dir=tmp_path)
    universe = _universe([(T0, T0 + 10 * HOUR_MS, "AUSDT", "l1")])
    report = universe_coverage(universe, "1h", base_dir=tmp_path)
    row = report.iloc[0]
    assert row["missing_before"] == 6
    assert row["present"] == 4
    assert row["coverage_pct"] == pytest.approx(40.0)
    assert row["traded_pct"] == pytest.approx(20.0)


def test_universe_coverage_does_not_use_candles_after_period_end(tmp_path: Path) -> None:
    """Las velas posteriores al fin del periodo no influyen en el informe (sin mirar al futuro)."""
    short = _frame(_hours(range(10)))
    save_klines(short, "AUSDT", "1h", base_dir=tmp_path / "a")
    save_klines(_frame(_hours(range(40))), "AUSDT", "1h", base_dir=tmp_path / "b")
    universe = _universe([(T0, T0 + 10 * HOUR_MS, "AUSDT", "l1")])
    only_period = universe_coverage(universe, "1h", base_dir=tmp_path / "a")
    with_future = universe_coverage(universe, "1h", base_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(only_period, with_future)
