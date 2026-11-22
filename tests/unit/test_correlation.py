"""Tests de `miax.features.correlation`: exportaciones, refactor sin duplicar y `require_psd`."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from miax import features
from miax.features import correlation

# `miax.viz` re-exporta una funcion con el mismo nombre que el modulo: se importa por ruta.
viz_raw = importlib.import_module("miax.viz.raw_correlation")

NAMES = (
    "raw_correlation",
    "symbols_without_data",
    "off_diagonal",
    "count_missing_pairs",
    "explained_variance",
)


def _corr(values: list[list[float]]) -> pd.DataFrame:
    labels = [f"S{i}" for i in range(len(values))]
    return pd.DataFrame(values, index=labels, columns=labels, dtype="float64")


@pytest.mark.parametrize("name", NAMES)
def test_exported_from_features_and_same_object_as_viz(name: str) -> None:
    """Las funciones se exportan desde `miax.features` y la viz usa el mismo objeto."""
    assert name in features.__all__
    assert getattr(features, name) is getattr(correlation, name)
    assert getattr(viz_raw, name) is getattr(correlation, name)


def test_require_psd_false_returns_eigenvalues_of_non_psd_matrix() -> None:
    """Con `require_psd=False` una matriz no semidefinida da sus autovalores; por defecto falla."""
    corr = _corr([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(ValueError, match="semidefinida"):
        correlation.explained_variance(corr)
    with pytest.raises(ValueError, match="semidefinida"):
        correlation.explained_variance(corr, require_psd=True)

    out = correlation.explained_variance(corr, require_psd=False)

    expected = np.linalg.eigvalsh(corr.to_numpy())[::-1] / 3
    np.testing.assert_allclose(out, expected)
    assert out[-1] < 0
    assert out.sum() == pytest.approx(1.0)


def test_require_psd_flag_is_irrelevant_on_valid_matrix() -> None:
    """En una matriz valida los dos modos dan exactamente lo mismo."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("ABCD"))
    corr = correlation.raw_correlation(data, min_periods=10)

    np.testing.assert_array_equal(
        correlation.explained_variance(corr),
        correlation.explained_variance(corr, require_psd=False),
    )
