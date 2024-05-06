# Implements the Lambert function, code took from gmgeorg github lambertw
"""A torch implementation of the Lambert W function.

In special module to follow the style of scipy.special.* and tfp.special.*

This implementation is a direct translation from TensorFlow Probability
https://www.tensorflow.org/probability/api_docs/python/tfp/math/lambertw

Vectorized, except for while loop. TODO to make this work in vectorized/GPU version
(equivalent of tf.while_loop() in TensorFlow).
"""

from typing import Optional, Tuple

import numpy as np
import torch

_EPS = 1e-6
# Constant for - 1 / e.  This is the lowest 'z' for which principal / non-principal W
# is real valued (W(-1/e) = -1).  For any z < -1 / exp(1), W(z) = NA.
_EXP_INV = np.exp(-1)
_M_EXP_INV = -1 * _EXP_INV

_MAX_ITER = 100


def _lambertw_winitzki_approx(z: torch.Tensor) -> torch.Tensor:
    """
    Computes Winitzki approximation to Lambert W function at z >= -1/exp(1).

    Args:
        z: Value for which W(z) should be computed. Expected z >= -1/exp(1).

    Returns:
        lambertw_winitzki_approx: Approximation for W(z) for z >= -1/exp(1).
    """
    log1pz = torch.log1p(z)
    return log1pz * (1.0 - torch.log1p(log1pz) / (2.0 + log1pz))


def _halley_iteration(
    w: torch.Tensor, z: torch.Tensor, tol: float, iteration_count: int
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Halley's method on root finding of w for the equation w * exp(w) = z.

    Args:
        w (torch.Tensor): Current value of w.
        z (torch.Tensor): Value for which the root is being found.
        tol (float): Tolerance for convergence.
        iteration_count (int): Current iteration count.

    Returns:
        Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            - bool: Whether the iteration should stop.
            - torch.Tensor: Updated value of w.
            - torch.Tensor: Input value z.
            - torch.Tensor: Delta value.
            - int: Updated iteration count.
    """
    f = w - z * torch.exp(-w)
    delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))
    w_next = w - delta
    converged = torch.abs(delta) <= tol
    converged = converged | torch.isnan(w_next)
    should_stop_next = torch.all(converged) or (iteration_count >= _MAX_ITER)
    return should_stop_next, w_next, z, delta, iteration_count + 1


def _lambertw_principal_branch_nonna(z: torch.Tensor) -> torch.Tensor:
    """Computes principal branch of Lambert W function for input with nonna output."""
    # check if z > -1 (vectorized)
    w = torch.where(z >= _M_EXP_INV, _lambertw_winitzki_approx(z), z)
    stop_condition = False
    counter = 0
    while not stop_condition:
        counter += 1
        stop_condition, w, z, _, _ = _halley_iteration(w, z, _EPS, counter)

    # if z = _M_EXP_INV, return exactly -1. If z = Inf, return Inf
    return torch.where(torch.abs(z - _M_EXP_INV) < _EPS, -1 * torch.ones_like(z), w)


def lambertw(z: torch.Tensor) -> torch.Tensor:
    """Computes principal branch for z.

    For z < -1/exp(1), it returns nan; for z = inf, it returns inf.
    """
    w = torch.where(z >= _M_EXP_INV, _lambertw_principal_branch_nonna(z), torch.nan)
    return torch.where(torch.isposinf(z), torch.inf, w)


def lambertw_gradient(
    z: torch.tensor, k: int = 0, w: Optional[torch.tensor] = None
) -> torch.tensor:
    """Computes the gradience of W(z)."""

    if w is None:
        w = lambertw(z, k=k)
    dw_dz = w / (z * (1 + w))
    return torch.where(torch.abs(z) < _EPS, torch.ones_like(z), dw_dz)
