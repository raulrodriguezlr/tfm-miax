"""Comprueba que set_global_seed y make_rng dan resultados deterministas."""

from __future__ import annotations

import random

import numpy as np
import pytest

from miax.utils.seeds import make_rng, set_global_seed


def test_same_seed_same_random_sequence() -> None:
    """Dos ejecuciones con la misma semilla dan la misma secuencia en random."""
    set_global_seed(42)
    first = [random.random() for _ in range(5)]

    set_global_seed(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_different_seeds_different_random_sequence() -> None:
    """Dos semillas distintas dan secuencias distintas en random."""
    set_global_seed(1)
    first = [random.random() for _ in range(5)]

    set_global_seed(2)
    second = [random.random() for _ in range(5)]

    assert first != second


def test_same_seed_same_numpy_global_sequence() -> None:
    """Dos ejecuciones con la misma semilla dan la misma secuencia en el estado global de numpy."""
    set_global_seed(42)
    first = np.random.rand(5)  # noqa: NPY002

    set_global_seed(42)
    second = np.random.rand(5)  # noqa: NPY002

    assert np.array_equal(first, second)


def test_different_seeds_different_numpy_global_sequence() -> None:
    """Dos semillas distintas dan secuencias distintas en el estado global de numpy."""
    set_global_seed(1)
    first = np.random.rand(5)  # noqa: NPY002

    set_global_seed(2)
    second = np.random.rand(5)  # noqa: NPY002

    assert not np.array_equal(first, second)


def test_make_rng_same_seed_same_sequence() -> None:
    """make_rng con la misma semilla devuelve generadores que producen la misma secuencia."""
    first = make_rng(42).random(5)
    second = make_rng(42).random(5)

    assert np.array_equal(first, second)


def test_make_rng_different_seeds_different_sequence() -> None:
    """make_rng con semillas distintas devuelve generadores que producen secuencias distintas."""
    first = make_rng(1).random(5)
    second = make_rng(2).random(5)

    assert not np.array_equal(first, second)


def test_make_rng_independent_of_numpy_global_state() -> None:
    """make_rng(42) da la misma secuencia aunque se altere antes el estado global de numpy."""
    baseline = make_rng(42).random(5)

    set_global_seed(999)
    np.random.rand(3)  # noqa: NPY002 - altera el estado legacy para comprobar que no afecta a make_rng

    after_global_change = make_rng(42).random(5)

    assert np.array_equal(baseline, after_global_change)


def test_make_rng_returns_numpy_generator() -> None:
    """make_rng devuelve una instancia de numpy.random.Generator, no el estado legacy."""
    rng = make_rng(42)

    assert isinstance(rng, np.random.Generator)


def test_set_global_seed_sets_torch_seed() -> None:
    """Con torch instalado, la misma semilla da la misma secuencia de torch.rand."""
    torch = pytest.importorskip("torch")

    set_global_seed(42)
    first = torch.rand(5)

    set_global_seed(42)
    second = torch.rand(5)

    assert torch.equal(first, second)
