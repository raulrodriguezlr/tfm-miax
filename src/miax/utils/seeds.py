"""Semillas fijas para random, numpy y torch, y generadores numpy independientes."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fija la semilla global de random, numpy y torch (si esta instalado)."""
    random.seed(seed)
    # Semilla del estado global legacy de numpy: hace falta ademas de make_rng()
    # porque librerias externas (sklearn, etc.) llaman a np.random.* directamente.
    np.random.seed(seed)  # noqa: NPY002

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int = 42) -> np.random.Generator:
    """Crea un generador numpy independiente del estado global, con semilla fija."""
    return np.random.default_rng(seed)
