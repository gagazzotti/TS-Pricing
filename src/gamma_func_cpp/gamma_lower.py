"""TBD"""

import numpy.typing as npt

# pylint: disable=all
from src.gamma_func_cpp.gamma_incomp import (
    gamma_lower_incomplete_non_normalized,
)

# pylint: enable=all


def gamma_lower(a: npt.NDArray, z: npt.NDArray, N: int = 100, tolerance: float = 1e-10):
    """
    TBD

    Args:
        a (npt.NDArray): _description_
        z (npt.NDArray): _description_
        N (int, optional): _description_. Defaults to 100.
        tolerance (float, optional): _description_. Defaults to 1e-10.

    Returns:
        _type_: _description_
    """
    return gamma_lower_incomplete_non_normalized(a, z, N, tolerance)
