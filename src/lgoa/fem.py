r"""Temporal finite-element operators for the discrete :math:`H^1` metric.

The control :math:`\theta_\tau` is a continuous piecewise-linear (CG(1))
function of time, represented by its nodal coefficient matrix
:math:`\Theta_\tau \in \mathbb{R}^{(K+1) \times n}`.  On a partition with
intervals :math:`h_k` the mass and stiffness matrices are tridiagonal,

.. math::

    M^\tau_{kk} = \tfrac{h_k + h_{k+1}}{3}, \quad
    M^\tau_{k,k+1} = \tfrac{h_k}{6}, \quad
    A^\tau_{kk} = \tfrac{1}{h_k} + \tfrac{1}{h_{k+1}}, \quad
    A^\tau_{k,k+1} = -\tfrac{1}{h_k},

and the discrete Riesz map is :math:`B^\tau = M^\tau + A^\tau`.  The metric
gradient :math:`G_\tau` solves :math:`B^\tau G_\tau = R_\tau`, so every control
component is one right-hand side of the same tridiagonal system and the solve
costs :math:`O(K)` per component.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.linalg import cholesky

__all__ = ["FemOperators", "assemble", "apply", "solve", "dense_riesz", "cholesky_factor"]


@dataclass(frozen=True)
class FemOperators:
    """Tridiagonal bands of :math:`M^\\tau`, :math:`A^\\tau` and :math:`B^\\tau`."""

    mass_lower: jnp.ndarray
    mass_diag: jnp.ndarray
    mass_upper: jnp.ndarray
    stiff_lower: jnp.ndarray
    stiff_diag: jnp.ndarray
    stiff_upper: jnp.ndarray
    riesz_lower: jnp.ndarray
    riesz_diag: jnp.ndarray
    riesz_upper: jnp.ndarray

    @property
    def mass(self):
        return self.mass_lower, self.mass_diag, self.mass_upper

    @property
    def stiffness(self):
        return self.stiff_lower, self.stiff_diag, self.stiff_upper

    @property
    def riesz(self):
        return self.riesz_lower, self.riesz_diag, self.riesz_upper


def assemble(time_grid) -> FemOperators:
    """Assemble the CG(1) mass, stiffness and Riesz bands on ``time_grid``."""
    h = jnp.diff(time_grid)
    zero = jnp.zeros((1,), dtype=time_grid.dtype)

    mass_diag = jnp.concatenate([h[:1] / 3.0, (h[:-1] + h[1:]) / 3.0, h[-1:] / 3.0])
    mass_off = h / 6.0
    mass_lower = jnp.concatenate([zero, mass_off])
    mass_upper = jnp.concatenate([mass_off, zero])

    stiff_diag = jnp.concatenate(
        [1.0 / h[:1], 1.0 / h[:-1] + 1.0 / h[1:], 1.0 / h[-1:]]
    )
    stiff_off = -1.0 / h
    stiff_lower = jnp.concatenate([zero, stiff_off])
    stiff_upper = jnp.concatenate([stiff_off, zero])

    return FemOperators(
        mass_lower=mass_lower,
        mass_diag=mass_diag,
        mass_upper=mass_upper,
        stiff_lower=stiff_lower,
        stiff_diag=stiff_diag,
        stiff_upper=stiff_upper,
        riesz_lower=mass_lower + stiff_lower,
        riesz_diag=mass_diag + stiff_diag,
        riesz_upper=mass_upper + stiff_upper,
    )


def apply(bands, X):
    """Multiply a tridiagonal operator given as ``(lower, diag, upper)``."""
    lower, diagonal, upper = bands
    Y = diagonal[:, None] * X
    Y = Y.at[1:].add(lower[1:, None] * X[:-1])
    Y = Y.at[:-1].add(upper[:-1, None] * X[1:])
    return Y


def solve(bands, rhs):
    """Solve a tridiagonal system for all control components at once."""
    lower, diagonal, upper = bands
    return lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs)


def dense_riesz(operators: FemOperators) -> np.ndarray:
    """Dense, symmetrized :math:`B^\\tau`; used only to whiten BFGS."""
    diagonal = np.asarray(operators.riesz_diag, dtype=float)
    lower = np.asarray(operators.riesz_lower, dtype=float)
    upper = np.asarray(operators.riesz_upper, dtype=float)
    matrix = np.diag(diagonal)
    matrix += np.diag(upper[:-1], k=1)
    matrix += np.diag(lower[1:], k=-1)
    return 0.5 * (matrix + matrix.T)


def cholesky_factor(operators: FemOperators) -> np.ndarray:
    r"""Lower Cholesky factor :math:`L^\tau` of :math:`B^\tau = L^\tau (L^\tau)^\top`."""
    return cholesky(dense_riesz(operators), lower=True, check_finite=False)
