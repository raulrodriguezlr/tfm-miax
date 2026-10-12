"""Tests del validador de integridad de la cache de klines, con caches en `tmp_path`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from miax.ingest import find_gaps, save_klines, validate_klines
from miax.ingest.integrity import (
    ERROR,
    ISSUE_COLUMNS,
    WARNING,
    check_klines_frame,
    count_month_files,
)
from miax.ingest.klines import KLINE_COLUMNS, KLINE_DTYPES
from miax.utils.io import write_parquet

HOUR_MS = 3_600_000
T0 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
FEB = 1_769_904_000_000  # 2026-02-01T00:00:00Z
MAR = 1_772_323_200_000  # 2026-03-01T00:00:00Z
SYMBOL = "BTCUSDT"


def _frame(open_times: list[int], step: int = HOUR_MS) -> pd.DataFrame:
    """Construye klines coherentes y con los dtypes canonicos para esos `open_time`."""
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
        "num_trades": [10] * n,
        "taker_buy_base": [0.7] * n,
        "taker_buy_quote": [70.0] * n,
        "ignore": [0.0] * n,
    }
    return pd.DataFrame(data).astype(KLINE_DTYPES)[KLINE_COLUMNS]


def _hours(start: int, count: int) -> list[int]:
    """`count` open_time horarios consecutivos desde `start`."""
    return [start + i * HOUR_MS for i in range(count)]


def _write(root: Path, month: str, frame: pd.DataFrame, interval: str = "1h") -> Path:
    """Escribe `frame` tal cual como el fichero del mes dado."""
    return write_parquet(frame, root / SYMBOL / interval / f"{month}.parquet")


def _checks(issues: pd.DataFrame) -> list[str]:
    """Lista los nombres de check de las incidencias."""
    return list(issues["check"])


def _healthy_cache(root: Path) -> None:
    """Cache de tres meses consecutivos y sin huecos (enero a marzo de 2026, horaria)."""
    times = _hours(T0, (FEB - T0) // HOUR_MS) + _hours(FEB, (MAR - FEB) // HOUR_MS)
    times += _hours(MAR, 48)
    save_klines(_frame(times), SYMBOL, "1h", base_dir=root)


def test_healthy_multi_month_cache_has_no_issues(tmp_path: Path) -> None:
    """Una cache sana de varios meses no da incidencias."""
    _healthy_cache(tmp_path)
    assert count_month_files(SYMBOL, "1h", base_dir=tmp_path) == 3
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert issues.empty
    assert list(issues.columns) == ISSUE_COLUMNS


def test_unreadable_file(tmp_path: Path) -> None:
    """Bytes basura con extension .parquet dan un error `unreadable`."""
    path = tmp_path / SYMBOL / "1h" / "2026-01.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"esto no es un parquet")
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["unreadable"]
    assert issues.iloc[0]["severity"] == ERROR
    assert issues.iloc[0]["file"] == "2026-01.parquet"


def test_unreadable_file_does_not_stop_other_files(tmp_path: Path) -> None:
    """Un fichero ilegible no impide revisar los demas."""
    bad = tmp_path / SYMBOL / "1h" / "2026-01.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"basura")
    frame = _frame(_hours(FEB, 3))
    frame.loc[1, "high"] = 50.0  # high < close
    _write(tmp_path, "2026-02", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert set(_checks(issues)) == {"unreadable", "ohlc"}
    ohlc = issues[issues["check"] == "ohlc"].iloc[0]
    assert ohlc["file"] == "2026-02.parquet"


def test_missing_column_is_schema_error(tmp_path: Path) -> None:
    """Falta una columna: error `schema` y no se aplican el resto de comprobaciones."""
    frame = _frame(_hours(T0, 3)).drop(columns=["volume"])
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["schema"]
    assert issues.iloc[0]["severity"] == ERROR
    assert "volume" in issues.iloc[0]["detail"]


def test_wrong_dtype_is_schema_error(tmp_path: Path) -> None:
    """Un dtype distinto al canonico es un error `schema`."""
    frame = _frame(_hours(T0, 3))
    frame["num_trades"] = frame["num_trades"].astype("float64")
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["schema"]
    assert "num_trades" in issues.iloc[0]["detail"]


def test_duplicates_within_a_file(tmp_path: Path) -> None:
    """Un `open_time` repetido en el mismo fichero da `duplicates` en ese fichero."""
    times = _hours(T0, 4)
    frame = _frame([*times, times[1]]).sort_values("open_time").reset_index(drop=True)
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["duplicates"]
    assert issues.iloc[0]["count"] == 1
    assert issues.iloc[0]["file"] == "2026-01.parquet"


def test_duplicates_between_two_months(tmp_path: Path) -> None:
    """La misma vela en dos ficheros de mes da `duplicates` global del simbolo."""
    last_jan = FEB - HOUR_MS  # 2026-01-31T23:00Z
    _write(tmp_path, "2026-01", _frame([last_jan - HOUR_MS, last_jan]))
    # Vela repetida (de enero) dentro del fichero de febrero.
    _write(tmp_path, "2026-02", _frame([last_jan, last_jan + HOUR_MS]))
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert set(_checks(issues)) == {"duplicates", "wrong_month"}
    dup = issues[issues["check"] == "duplicates"].iloc[0]
    assert dup["file"] == f"{SYMBOL}/1h"
    assert dup["count"] == 1
    assert dup["severity"] == ERROR


def test_unordered_open_time(tmp_path: Path) -> None:
    """`open_time` que decrece dentro del fichero da `unordered`."""
    times = _hours(T0, 4)
    frame = _frame([times[0], times[2], times[1], times[3]])
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["unordered"]
    assert issues.iloc[0]["count"] == 1


def test_misaligned_open_time(tmp_path: Path) -> None:
    """Una vela que no abre en la rejilla del intervalo da `misaligned`."""
    frame = _frame(_hours(T0, 3))
    frame.loc[1, "open_time"] += 1_000
    frame["close_time"] = frame["open_time"] + HOUR_MS - 1
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    misaligned = issues[issues["check"] == "misaligned"]
    assert len(misaligned) == 1
    assert misaligned.iloc[0]["count"] == 1
    assert misaligned.iloc[0]["severity"] == ERROR


def test_bad_close_time(tmp_path: Path) -> None:
    """`close_time` distinto de open_time + paso - 1 da `close_time`."""
    frame = _frame(_hours(T0, 3))
    frame.loc[2, "close_time"] += 5
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["close_time"]
    assert issues.iloc[0]["count"] == 1


def test_candle_stored_in_wrong_month(tmp_path: Path) -> None:
    """Una vela de febrero guardada en el fichero de enero da `wrong_month`."""
    frame = _frame([T0, T0 + HOUR_MS, FEB])
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    wrong = issues[issues["check"] == "wrong_month"]
    assert len(wrong) == 1
    assert wrong.iloc[0]["count"] == 1
    assert wrong.iloc[0]["severity"] == ERROR


def test_high_below_close_is_ohlc_error(tmp_path: Path) -> None:
    """`high < close` da `ohlc`."""
    frame = _frame(_hours(T0, 3))
    frame.loc[0, "high"] = 100.0  # menor que close (100.5)
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["ohlc"]
    assert issues.iloc[0]["count"] == 1


def test_negative_volume_is_ohlc_error(tmp_path: Path) -> None:
    """Un volumen negativo da `ohlc`."""
    frame = _frame(_hours(T0, 3))
    frame.loc[1, "volume"] = -1.0
    _write(tmp_path, "2026-01", frame)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["ohlc"]


def test_gap_is_a_warning_with_missing_count(tmp_path: Path) -> None:
    """Un hueco de N velas es un aviso (no un error) con count == N."""
    n_missing = 5
    times = _hours(T0, 4) + _hours(T0 + (4 + n_missing) * HOUR_MS, 4)
    save_klines(_frame(times), SYMBOL, "1h", base_dir=tmp_path)
    issues = validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    assert _checks(issues) == ["gaps"]
    row = issues.iloc[0]
    assert row["severity"] == WARNING
    assert row["count"] == n_missing
    assert (issues["severity"] == ERROR).sum() == 0


def test_validator_does_not_modify_files(tmp_path: Path) -> None:
    """Validar una cache con problemas deja bytes, mtime y lista de ficheros intactos."""
    frame = _frame(_hours(T0, 4))
    frame.loc[1, "high"] = 1.0
    good = _write(tmp_path, "2026-01", frame)
    bad = tmp_path / SYMBOL / "1h" / "2026-02.parquet"
    bad.write_bytes(b"basura")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (good, bad)}
    files_before = sorted((tmp_path / SYMBOL / "1h").iterdir())
    validate_klines(SYMBOL, "1h", base_dir=tmp_path)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (good, bad)}
    assert before == after
    assert sorted((tmp_path / SYMBOL / "1h").iterdir()) == files_before


def test_non_month_files_are_ignored(tmp_path: Path) -> None:
    """Un .parquet cuyo nombre no es un mes no se revisa ni se cuenta."""
    _write(tmp_path, "2026-01", _frame(_hours(T0, 3)))
    (tmp_path / SYMBOL / "1h" / "notas.parquet").write_bytes(b"basura")
    assert count_month_files(SYMBOL, "1h", base_dir=tmp_path) == 1
    assert validate_klines(SYMBOL, "1h", base_dir=tmp_path).empty


def test_find_gaps_reports_range_and_count() -> None:
    """Un hueco devuelve el primer y ultimo open_time ausentes y cuantas velas faltan."""
    times = pd.Series([T0, T0 + HOUR_MS, T0 + 5 * HOUR_MS])
    gaps = find_gaps(times, "1h")
    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row["gap_start_ms"] == T0 + 2 * HOUR_MS
    assert row["gap_end_ms"] == T0 + 4 * HOUR_MS
    assert row["missing_candles"] == 3


def test_find_gaps_unsorted_and_duplicated_input() -> None:
    """La entrada se ordena y se deduplica antes de buscar huecos."""
    times = pd.Series([T0 + 3 * HOUR_MS, T0, T0, T0 + HOUR_MS, T0 + 2 * HOUR_MS])
    assert find_gaps(times, "1h").empty


def test_find_gaps_empty_and_single() -> None:
    """Serie vacia o con un solo open_time no tiene huecos."""
    assert find_gaps(pd.Series([], dtype="int64"), "1h").empty
    assert find_gaps(pd.Series([T0]), "1h").empty


def test_find_gaps_misaligned_jump_below_two_steps_is_not_a_gap() -> None:
    """Un salto entre step y 2*step (desalineado) no deja velas completas sin cubrir."""
    times = pd.Series([T0, T0 + HOUR_MS + 1_800_000])
    assert find_gaps(times, "1h").empty


def test_find_gaps_columns_when_empty() -> None:
    """El resultado sin huecos conserva las columnas esperadas."""
    gaps = find_gaps(pd.Series([T0]), "1h")
    assert list(gaps.columns) == ["gap_start_ms", "gap_end_ms", "missing_candles"]


def test_variable_length_interval_skips_gaps_and_alignment(tmp_path: Path) -> None:
    """Con `1M` (sin duracion fija) no se comprueban huecos, alineacion ni close_time."""
    assert find_gaps(pd.Series([T0, T0 + 10 * HOUR_MS]), "1M").empty
    frame = _frame([T0 + 1, T0 + 99_999_999])
    assert check_klines_frame(frame, "1M").empty
    _write(tmp_path, "2026-01", frame, interval="1M")
    assert validate_klines(SYMBOL, "1M", base_dir=tmp_path).empty


def test_check_klines_frame_is_pure_and_leaves_file_empty() -> None:
    """La funcion pura devuelve `file` vacio y no toca el DataFrame de entrada."""
    frame = _frame(_hours(T0, 3))
    frame.loc[0, "volume"] = -1.0
    snapshot = frame.copy()
    issues = check_klines_frame(frame, "1h", month="2026-01")
    assert list(issues["file"]) == [""]
    pd.testing.assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize("create_dir", [True, False])
def test_empty_or_missing_directory(tmp_path: Path, create_dir: bool) -> None:
    """Sin ficheros (carpeta vacia o inexistente) se devuelve un DataFrame vacio con columnas."""
    root = tmp_path / "cache"
    if create_dir:
        (root / SYMBOL / "1h").mkdir(parents=True)
    issues = validate_klines(SYMBOL, "1h", base_dir=root)
    assert issues.empty
    assert list(issues.columns) == ISSUE_COLUMNS
    assert count_month_files(SYMBOL, "1h", base_dir=root) == 0
