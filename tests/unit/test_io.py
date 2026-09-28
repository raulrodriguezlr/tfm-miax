"""Comprueba que write_parquet y read_parquet hacen ida y vuelta exacta y de forma atomica."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pyarrow
import pytest

from miax.utils.config import REPO_ROOT
from miax.utils.io import read_parquet, resolve_repo_path, write_parquet


def make_basic_df() -> pd.DataFrame:
    """Construye un DataFrame con tipos basicos e indice UTC de 5 minutos sin freq."""
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
        freq=None,
        name="open_time",
    )
    return pd.DataFrame(
        {
            "close": [1.0, 2.0, float("nan"), 4.0, 5.0],
            "volume": [1, 2, 3, 4, 5],
            "is_up": [True, False, True, False, True],
            "symbol": ["BTC", "ETH", "BTC", "ETH", "BTC"],
        },
        index=index,
    )


def make_categorical_df() -> pd.DataFrame:
    """Construye el DataFrame basico con una columna categorica anadida."""
    df = make_basic_df()
    df["grade"] = pd.Categorical(["a", "b", "a", "c", "b"])
    return df


def make_nullable_int_df() -> pd.DataFrame:
    """Construye el DataFrame basico con una columna Int64 nullable con un nulo."""
    df = make_basic_df()
    df["count"] = pd.array([1, 2, None, 4, 5], dtype="Int64")
    return df


def make_empty_df() -> pd.DataFrame:
    """Construye un DataFrame vacio que conserva las columnas y tipos del basico."""
    return make_basic_df().iloc[:0]


def make_multiindex_columns_df() -> pd.DataFrame:
    """Construye un DataFrame con columnas MultiIndex de activo por feature."""
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
        freq=None,
        name="open_time",
    )
    columns = pd.MultiIndex.from_product(
        [["BTC", "ETH"], ["close", "volume"]], names=["asset", "feature"]
    )
    return pd.DataFrame(
        [[1.0, 100.0, 2.0, 200.0], [1.1, 110.0, 2.1, 210.0]],
        columns=columns,
        index=index,
    )


def make_multiindex_rows_df() -> pd.DataFrame:
    """Construye un DataFrame con indice MultiIndex de open_time por simbolo."""
    open_times = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
        freq=None,
    )
    index = pd.MultiIndex.from_product([open_times, ["BTC", "ETH"]], names=["open_time", "sym"])
    return pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=index)


ROUND_TRIP_CASES = [
    pytest.param(make_basic_df, id="tipos_basicos"),
    pytest.param(make_categorical_df, id="categorica"),
    pytest.param(make_nullable_int_df, id="int64_nullable"),
    pytest.param(make_empty_df, id="vacio_con_columnas"),
    pytest.param(make_multiindex_columns_df, id="multiindex_columnas"),
    pytest.param(make_multiindex_rows_df, id="multiindex_filas"),
]


@pytest.mark.parametrize("build_df", ROUND_TRIP_CASES)
def test_write_then_read_round_trip(tmp_path: Path, build_df: Callable[[], pd.DataFrame]) -> None:
    """Escribir y releer un Parquet devuelve exactamente el mismo DataFrame."""
    original = build_df()
    path = tmp_path / "round_trip.parquet"

    write_parquet(original, path)
    reread = read_parquet(path)

    pdt.assert_frame_equal(original, reread)


def test_freq_is_not_preserved(tmp_path: Path) -> None:
    """El indice recupera freq=None tras el Parquet, aunque el original tuviera una frecuencia."""
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC"),
        name="open_time",
    )
    original = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    path = tmp_path / "freq.parquet"

    write_parquet(original, path)
    reread = read_parquet(path)

    assert reread.index.freq is None
    pdt.assert_frame_equal(original, reread, check_freq=False)


def test_read_parquet_columns_subset(tmp_path: Path) -> None:
    """Pedir columns=[...] al leer devuelve solo esas columnas."""
    original = pd.DataFrame({"close": [1.0, 2.0], "open": [3.0, 4.0], "volume": [1, 2]})
    path = tmp_path / "subset.parquet"
    write_parquet(original, path)

    reread = read_parquet(path, columns=["close"])

    assert reread.columns.tolist() == ["close"]


def test_resolve_repo_path_relative() -> None:
    """Una ruta relativa se resuelve desde la raiz del repo. No escribe nada en disco."""
    resolved = resolve_repo_path("data/raw/x.parquet")

    assert resolved == REPO_ROOT / "data/raw/x.parquet"


def test_resolve_repo_path_absolute(tmp_path: Path) -> None:
    """Una ruta absoluta vuelve tal cual, sin resolverla contra la raiz del repo."""
    absolute_path = tmp_path / "a.parquet"

    resolved = resolve_repo_path(absolute_path)

    assert resolved == absolute_path


def test_write_parquet_creates_parent_dirs(tmp_path: Path) -> None:
    """write_parquet crea las carpetas padre que no existan y devuelve la ruta final."""
    path = tmp_path / "a" / "b" / "c.parquet"
    original = pd.DataFrame({"close": [1.0, 2.0]})

    result = write_parquet(original, path)

    assert result == path
    assert path.exists()


def test_write_parquet_leaves_no_tmp_file(tmp_path: Path) -> None:
    """Tras una escritura correcta no queda ningun fichero .tmp en la carpeta destino."""
    path = tmp_path / "clean.parquet"
    original = pd.DataFrame({"close": [1.0, 2.0]})

    write_parquet(original, path)

    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_write_keeps_previous_file_and_no_tmp(tmp_path: Path) -> None:
    """Si la escritura falla, el fichero previo sigue intacto y no queda ningun .tmp."""
    path = tmp_path / "target.parquet"
    good = pd.DataFrame({"close": [1.0, 2.0]})
    write_parquet(good, path)

    bad = pd.DataFrame({"close": [1, "a"]})
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        write_parquet(bad, path)

    reread = read_parquet(path)
    pdt.assert_frame_equal(reread, good)
    assert list(tmp_path.glob("*.tmp")) == []
