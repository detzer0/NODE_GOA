r"""Implementation self-checks.

Five independent checks guard the hand-written numerics:

1. the JAX tridiagonal solve against SciPy's banded solve;
2. the manually assembled control derivative :math:`R_\tau` against
   ``jax.grad`` of the same discrete objective;
3. the endpoint-map derivative against ``jax.grad``;
4. invariance of the frozen endpoint maps under a control-only Adam step;
5. the closed-form linear/quadratic pairing against dense quadrature of the
   very same polynomial integrand;
6. equality of Dörfler refinement with a single marked interval and plain
   single-interval insertion, bit for bit.

:func:`self_check` returns a table and raises :class:`AssertionError` on the
first violated tolerance, so it doubles as a notebook cell and a unit test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

from . import fem as fem_module
from . import neural_ode as node
from . import optimizers
from . import refinement as refinement_module
from .backend import dtype, x64_enabled

__all__ = ["self_check", "check_single_interval_refinement_equivalence"]


def _banded_from_operators(operators, n_nodes):
    ab = np.zeros((3, n_nodes))
    ab[0, 1:] = np.asarray(operators.riesz_upper[:-1])
    ab[1] = np.asarray(operators.riesz_diag)
    ab[2, :-1] = np.asarray(operators.riesz_lower[1:])
    return ab


def check_single_interval_refinement_equivalence(hidden_dim: int = 3) -> None:
    """Refining one marked interval must reproduce single-interval insertion."""
    float_dtype = dtype()
    test_grid = jnp.asarray([0.0, 0.3, 0.9, 1.5], dtype=float_dtype)
    n_nodes = int(test_grid.shape[0])
    packed_width = hidden_dim * hidden_dim + hidden_dim

    test_params = {
        "W_in": jnp.arange(2 * hidden_dim, dtype=float_dtype).reshape(2, hidden_dim),
        "A": jnp.arange(n_nodes * hidden_dim**2, dtype=float_dtype).reshape(
            n_nodes, hidden_dim, hidden_dim
        ),
        "b": jnp.arange(n_nodes * hidden_dim, dtype=float_dtype).reshape(
            n_nodes, hidden_dim
        ),
        "W_out": jnp.arange(2 * hidden_dim, dtype=float_dtype).reshape(2, hidden_dim),
    }
    shape = (n_nodes, packed_width)
    test_state = {
        "count": jnp.asarray(7, dtype=jnp.int32),
        "m": jnp.arange(n_nodes * packed_width, dtype=float_dtype).reshape(shape),
        "v": (1.0 + jnp.arange(n_nodes * packed_width, dtype=float_dtype)).reshape(
            shape
        ),
    }

    for k in range(n_nodes - 1):
        reference_params, reference_grid, reference_state = (
            refinement_module.insert_single_interval(
                test_params, test_grid, k, test_state
            )
        )
        generalized = refinement_module.refine(
            test_params, test_grid, test_state, [k], transfer_adam_state=True
        )

        if not np.array_equal(
            np.asarray(generalized.time_grid), np.asarray(reference_grid)
        ):
            raise AssertionError("Time-grid transfer differs from single insertion.")
        for name in ("A", "b"):
            if not np.array_equal(
                np.asarray(generalized.params[name]),
                np.asarray(reference_params[name]),
            ):
                raise AssertionError(
                    f"Control transfer for {name} differs from single insertion."
                )
        for name in ("m", "v"):
            if not np.array_equal(
                np.asarray(generalized.optimizer_state[name]),
                np.asarray(reference_state[name]),
            ):
                raise AssertionError(
                    f"Adam-state transfer for {name} differs from single insertion."
                )


def self_check(
    X=None,
    y=None,
    *,
    hidden_dim: int = 4,
    num_classes: int = 5,
    reg_lambda: float = 1.0e-4,
    seed: int = 0,
    raise_on_failure: bool = True,
) -> pd.DataFrame:
    """Run every implementation check and return the relative errors."""
    float_dtype = dtype()
    rng = np.random.default_rng(seed + 5)

    if X is None or y is None:
        X = jnp.asarray(rng.normal(size=(12, 2)), dtype=float_dtype)
        y = jnp.asarray(rng.integers(0, num_classes, size=12), dtype=jnp.int32)
    else:
        X = jnp.asarray(X[:12], dtype=float_dtype)
        y = jnp.asarray(y[:12], dtype=jnp.int32)
        num_classes = int(np.max(np.asarray(y)) + 1) if num_classes is None else num_classes

    tiny_grid = jnp.asarray([0.0, 0.7, 1.4, 2.0], dtype=float_dtype)
    params = node.initialize_params(
        jax.random.PRNGKey(seed + 999), len(tiny_grid), hidden_dim, num_classes
    )
    operators = fem_module.assemble(tiny_grid)

    # 1. tridiagonal solve ------------------------------------------------
    theta = node.pack_controls(params)
    rhs = jnp.asarray(rng.normal(size=theta.shape), dtype=float_dtype)
    jax_solution = np.asarray(fem_module.solve(operators.riesz, rhs))
    scipy_solution = solve_banded(
        (1, 1), _banded_from_operators(operators, len(tiny_grid)), np.asarray(rhs)
    )
    solver_error = np.linalg.norm(jax_solution - scipy_solution) / (
        np.linalg.norm(scipy_solution) + 1.0e-15
    )

    # 2./3. derivatives against jax.grad ----------------------------------
    metrics, grads = node.objective_and_gradients(
        params, X, y, tiny_grid, operators, reg_lambda, True
    )

    def autodiff_objective(tree):
        packed = node.pack_controls(tree)
        B_packed = fem_module.apply(operators.riesz, packed)
        return node.data_loss_only(
            tree, X, y, tiny_grid
        ) + 0.5 * reg_lambda * jnp.sum(packed * B_packed)

    autodiff = jax.grad(autodiff_objective)(params)
    manual_dual = np.asarray(metrics["control_dual"])
    auto_dual = np.asarray(node.pack_controls(autodiff))
    control_error = np.linalg.norm(manual_dual - auto_dual) / (
        np.linalg.norm(auto_dual) + 1.0e-15
    )

    endpoint_error = (
        np.linalg.norm(np.asarray(grads["W_in"] - autodiff["W_in"]))
        + np.linalg.norm(np.asarray(grads["W_out"] - autodiff["W_out"]))
    ) / (
        np.linalg.norm(np.asarray(autodiff["W_in"]))
        + np.linalg.norm(np.asarray(autodiff["W_out"]))
        + 1.0e-15
    )

    # 4. frozen endpoint maps --------------------------------------------
    _, frozen_grads = node.objective_and_gradients(
        params, X, y, tiny_grid, operators, reg_lambda, False
    )
    stepped, _ = optimizers.adam_update(
        params,
        frozen_grads,
        optimizers.adam_init(params),
        jnp.asarray(1.0e-3, dtype=float_dtype),
    )
    frozen_change = float(
        np.linalg.norm(np.asarray(stepped["W_in"] - params["W_in"]))
        + np.linalg.norm(np.asarray(stepped["W_out"] - params["W_out"]))
    )

    # 5. closed-form pairing against dense quadrature ---------------------
    m = 17
    theta0 = rng.normal(size=m)
    theta1 = rng.normal(size=m)
    q0 = rng.normal(size=m)
    q1 = rng.normal(size=m)
    dt = 0.73
    B = rng.normal(size=m)
    A = theta1 - theta0 - B
    C = theta0
    from .indicators import linear_quadratic_pairing_coefficients

    alpha, beta = linear_quadratic_pairing_coefficients(A, B, C)
    analytical = dt * (np.dot(q0, alpha) + np.dot(q1, beta))
    s = np.linspace(0.0, 1.0, 20001)
    q_s = (1.0 - s[:, None]) * q0 + s[:, None] * q1
    v_s = A * s[:, None] ** 2 + B * s[:, None] + C
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    dense = dt * trapezoid(np.sum(q_s * v_s, axis=1), s)
    pairing_error = abs(analytical - dense) / (abs(dense) + 1.0e-15)

    # 6. refinement equivalence ------------------------------------------
    try:
        check_single_interval_refinement_equivalence()
        refinement_error = 0.0
    except AssertionError:
        refinement_error = np.inf

    solver_tolerance = 1.0e-9 if x64_enabled() else 2.0e-6
    gradient_tolerance = 5.0e-8 if x64_enabled() else 2.0e-5
    frozen_tolerance = 1.0e-14 if x64_enabled() else 1.0e-7

    rows = [
        ("JAX tridiagonal solve vs SciPy", solver_error, solver_tolerance),
        ("control derivative vs jax.grad", control_error, gradient_tolerance),
        ("endpoint derivative vs jax.grad", endpoint_error, gradient_tolerance),
        ("frozen endpoint maps unchanged", frozen_change, frozen_tolerance),
        ("closed-form control pairing", pairing_error, 2.0e-8),
        ("Dorfler(|M|=1) == single insertion", refinement_error, 0.0),
    ]
    table = pd.DataFrame(
        [
            {
                "check": name,
                "relative_error": float(error),
                "tolerance": float(tolerance),
                "passed": bool(error <= tolerance),
            }
            for name, error, tolerance in rows
        ]
    )

    if raise_on_failure and not table["passed"].all():
        failed = table.loc[~table["passed"], "check"].tolist()
        raise AssertionError(f"Implementation check(s) failed: {failed}")
    return table
