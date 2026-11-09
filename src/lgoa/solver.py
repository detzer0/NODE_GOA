r"""Algorithm 1 -- the fixed-depth discretized neural ODE solver.

On a frozen temporal grid :math:`\mathcal{K}` the control problem is solved by
full-batch :math:`H^1`-Adam, optionally finished by BFGS:

1. **Adam phase.** Each step consumes the metric gradient :math:`G_\tau` from
   :math:`B^\tau G_\tau = R_\tau`.  The learning rate is reduced on plateaus and
   the iterate with the smallest stationarity measure is checkpointed.
2. **Switch.** Once :math:`s_\tau(\Theta_\tau) \le \kappa_s
   \varepsilon_{\mathrm{stat}}` the solver hands the warm start to BFGS.
3. **BFGS phase.** BFGS runs in the whitened coordinates
   :math:`\hat\Theta_\tau = (L^\tau)^\top \Theta_\tau`, where :math:`B^\tau =
   L^\tau (L^\tau)^\top`.  That change of variables is an isometry, so the
   Euclidean gradient norm in those coordinates *is* the discrete :math:`H^1`
   stationarity measure and a standard SciPy implementation respects the
   :math:`H^1` geometry.

The returned iterate is chosen feasibility-first: among iterates meeting
:math:`\varepsilon_{\mathrm{stat}}` the one with the lowest objective wins;
if none is feasible, the one with the smallest stationarity measure wins.
Setting ``use_bfgs_polish=False`` recovers the pure Adam algorithm.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from . import fem as fem_module
from . import neural_ode as node
from . import optimizers
from .backend import dtype
from .config import ProblemConfig, SolverConfig
from .datasets import Dataset

__all__ = ["ControlProblem", "PhaseResult", "make_control_problem", "solve_fixed_depth"]


# --------------------------------------------------------------------------- #
# Compiled problem on a fixed grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ControlProblem:
    """Compiled closures for one fixed temporal grid."""

    time_grid: jnp.ndarray
    operators: fem_module.FemOperators
    evaluate: Callable
    adam_chunk: Callable
    solution_fn: Callable
    accuracy_fn: Callable
    loss_fn: Callable
    margin_fn: Callable

    @property
    def depth(self) -> int:
        return int(self.time_grid.shape[0]) - 1


def make_control_problem(
    time_grid,
    data: Dataset,
    problem: ProblemConfig,
    solver: SolverConfig,
    *,
    fixed_W_in,
    fixed_W_out,
) -> ControlProblem:
    """Build the jitted objective, gradient and Adam kernel for ``time_grid``."""
    operators = fem_module.assemble(time_grid)
    hidden_dim = int(problem.hidden_dim)
    reg_lambda = float(solver.reg_lambda)
    X_train, y_train = data.X_train, data.y_train

    @jax.jit
    def evaluate(theta_matrix):
        params = node.params_from_control_matrix(
            theta_matrix, fixed_W_in, fixed_W_out, hidden_dim
        )
        metrics, _ = node.objective_and_gradients(
            params, X_train, y_train, time_grid, operators, reg_lambda, False
        )
        return metrics

    @partial(jax.jit, static_argnames=("n_steps",))
    def adam_chunk(theta_matrix, state, learning_rate, n_steps):
        def body(_, carry):
            theta, current_state = carry
            values = evaluate(theta)
            gradient = values["h1_gradient_matrix"]
            clip = jnp.minimum(
                1.0, solver.grad_clip / (values["stationarity_norm"] + 1.0e-12)
            )
            gradient = clip * gradient

            count = current_state["count"] + 1
            m = (
                solver.adam_beta1 * current_state["m"]
                + (1.0 - solver.adam_beta1) * gradient
            )
            v = (
                solver.adam_beta2 * current_state["v"]
                + (1.0 - solver.adam_beta2) * gradient * gradient
            )
            bias1 = 1.0 - solver.adam_beta1**count
            bias2 = 1.0 - solver.adam_beta2**count
            theta = theta - learning_rate * (m / bias1) / (
                jnp.sqrt(v / bias2) + solver.adam_eps
            )
            return theta, {"count": count, "m": m, "v": v}

        theta_matrix, state = lax.fori_loop(
            0, int(n_steps), body, (theta_matrix, state)
        )
        return theta_matrix, state, evaluate(theta_matrix)

    solution_fn = jax.jit(
        lambda params, X, y: node.forward_adjoint_and_euclidean_grads(
            params, X, y, time_grid
        )[:3]
    )
    accuracy_fn = jax.jit(
        lambda params, X, y: node.multiclass_accuracy(params, X, y, time_grid)
    )
    loss_fn = jax.jit(lambda params, X, y: node.data_loss_only(params, X, y, time_grid))
    margin_fn = jax.jit(
        lambda params, X, y: node.classification_margins(params, X, y, time_grid)
    )

    return ControlProblem(
        time_grid=time_grid,
        operators=operators,
        evaluate=evaluate,
        adam_chunk=adam_chunk,
        solution_fn=solution_fn,
        accuracy_fn=accuracy_fn,
        loss_fn=loss_fn,
        margin_fn=margin_fn,
    )


# --------------------------------------------------------------------------- #
# BFGS phase
# --------------------------------------------------------------------------- #
class _StationarityReached(RuntimeError):
    """Internal signal used to stop SciPy after an accepted feasible iterate."""


def _run_bfgs_polish(
    theta_start,
    control: ControlProblem,
    solver: SolverConfig,
    *,
    tolerance: float,
    label: str,
    verbose: bool,
) -> dict[str, Any]:
    r"""Polish a control in :math:`H^1`-whitened coordinates.

    With :math:`B^\tau = L L^\top` and :math:`\hat\Theta = L^\top \Theta`, the
    gradient is :math:`\hat G = L^{-1} R_\tau` and
    :math:`\lVert \hat G \rVert_F = s_\tau(\Theta_\tau)`.
    """
    start = time.perf_counter()
    theta_start_np = np.asarray(theta_start, dtype=float)
    theta_shape = theta_start_np.shape

    L = fem_module.cholesky_factor(control.operators)

    def theta_to_z(theta_matrix):
        return (L.T @ np.asarray(theta_matrix, dtype=float)).reshape(-1)

    def z_to_theta(z_flat):
        z_matrix = np.asarray(z_flat, dtype=float).reshape(theta_shape)
        return solve_triangular(L.T, z_matrix, lower=False, check_finite=False)

    cache: dict[str, Any] = {"z": None, "record": None}
    actual_evaluations = 0

    def evaluate_z(z_flat):
        nonlocal actual_evaluations
        z_array = np.asarray(z_flat, dtype=float)
        if cache["z"] is not None and np.array_equal(z_array, cache["z"]):
            return cache["record"]

        theta_np = z_to_theta(z_array)
        values = control.evaluate(jnp.asarray(theta_np, dtype=dtype()))
        objective = float(values["objective"])
        control_dual = np.asarray(values["control_dual"], dtype=float)
        gradient_z = solve_triangular(L, control_dual, lower=True, check_finite=False)
        record = {
            "z": np.array(z_array, copy=True),
            "theta": np.array(theta_np, copy=True),
            "objective": objective,
            "gradient": np.asarray(gradient_z, dtype=float).reshape(-1),
            "stationarity": float(np.linalg.norm(gradient_z.ravel())),
        }
        cache["z"] = np.array(z_array, copy=True)
        cache["record"] = record
        actual_evaluations += 1
        return record

    accepted_records: list[dict[str, Any]] = []

    def append_accepted(z_flat, iteration):
        record = dict(evaluate_z(z_flat))
        record["iteration"] = int(iteration)
        if not accepted_records or not np.array_equal(
            record["z"], accepted_records[-1]["z"]
        ):
            accepted_records.append(record)
        return record

    z0 = theta_to_z(theta_start_np)
    initial_record = append_accepted(z0, 0)
    iteration_count = 0
    stopping_reason = "BFGS maximum iterations reached"

    if initial_record["stationarity"] <= tolerance:
        stopping_reason = "Adam warm start already satisfies stationarity tolerance"
    else:

        def objective_and_gradient(z_flat):
            record = evaluate_z(z_flat)
            return record["objective"], record["gradient"]

        def callback(z_accepted):
            nonlocal iteration_count, stopping_reason
            iteration_count += 1
            record = append_accepted(z_accepted, iteration_count)
            if verbose and (
                iteration_count == 1
                or iteration_count % 10 == 0
                or record["stationarity"] <= tolerance
            ):
                print(
                    f"  {label}: {solver.bfgs_method} iter={iteration_count:4d}, "
                    f"J={record['objective']:.8f}, "
                    f"||G||_H1={record['stationarity']:.3e}"
                )
            if record["stationarity"] <= tolerance:
                stopping_reason = "BFGS stationarity tolerance reached"
                raise _StationarityReached()

        if solver.bfgs_method == "BFGS":
            options = {
                "maxiter": int(solver.bfgs_max_iterations),
                "gtol": float(solver.bfgs_internal_gtol),
                "disp": False,
            }
        else:
            # L-BFGS-B in recent SciPy versions rejects the legacy "disp" key.
            options = {
                "maxiter": int(solver.bfgs_max_iterations),
                "maxls": int(solver.bfgs_max_line_search_steps),
                "maxcor": int(solver.bfgs_memory),
                "ftol": float(solver.bfgs_ftol),
                "gtol": float(solver.bfgs_internal_gtol),
            }

        try:
            scipy_result = minimize(
                objective_and_gradient,
                z0,
                method=solver.bfgs_method,
                jac=True,
                callback=callback,
                options=options,
            )
            iteration_count = max(iteration_count, int(getattr(scipy_result, "nit", 0)))
            append_accepted(scipy_result.x, iteration_count)
            stopping_reason = str(getattr(scipy_result, "message", stopping_reason))
        except _StationarityReached:
            pass

    feasible = [r for r in accepted_records if r["stationarity"] <= tolerance]
    if feasible:
        selected = min(feasible, key=lambda r: r["objective"])
        selected_reason = "lowest-objective accepted BFGS iterate satisfying the tolerance"
    else:
        selected = min(
            accepted_records, key=lambda r: (r["stationarity"], r["objective"])
        )
        selected_reason = "accepted BFGS iterate with the smallest H1 stationarity norm"

    history = pd.DataFrame(
        [
            {
                "iteration": r["iteration"],
                "objective": r["objective"],
                "stationarity": r["stationarity"],
            }
            for r in accepted_records
        ]
    )

    return {
        "theta": jnp.asarray(selected["theta"], dtype=dtype()),
        "objective": float(selected["objective"]),
        "stationarity_norm": float(selected["stationarity"]),
        "converged": bool(selected["stationarity"] <= tolerance),
        "selected_iterate_reason": selected_reason,
        "method": solver.bfgs_method,
        "iterations": int(iteration_count),
        "function_evaluations": int(actual_evaluations),
        "stopping_reason": stopping_reason,
        "history": history,
        "elapsed_seconds": float(time.perf_counter() - start),
    }


def _candidate_key(objective, stationarity, tolerance):
    """Feasibility first, then objective; otherwise stationarity first."""
    if stationarity <= tolerance:
        return (0, objective, stationarity)
    return (1, stationarity, objective)


# --------------------------------------------------------------------------- #
# Result of one fixed-depth solve
# --------------------------------------------------------------------------- #
@dataclass
class PhaseResult:
    """Outcome of Algorithm 1 on one grid."""

    params: dict
    time_grid: jnp.ndarray
    optimizer_state: dict | None
    control: ControlProblem

    objective: float
    data_loss: float
    regularizer: float
    mass_energy: float
    stiffness_energy: float
    h1_energy: float

    stationarity_norm: float
    stationarity_tolerance: float
    stationarity_ratio: float
    converged: bool

    initial_objective: float
    initial_stationarity_norm: float
    best_stationarity_seen: float
    best_stationarity_objective: float

    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    margin_median: float
    margin_q10: float

    solver_mode: str
    selected_optimizer: str
    selected_iterate_reason: str

    adam_steps: int
    adam_final_learning_rate: float
    adam_lr_reductions: int
    adam_stopping_reason: str
    adam_history: pd.DataFrame

    bfgs_enabled: bool
    bfgs_triggered: bool
    bfgs_method: str
    bfgs_iterations: int
    bfgs_function_evaluations: int
    bfgs_stopping_reason: str
    bfgs_elapsed_seconds: float
    bfgs_history: pd.DataFrame

    elapsed_seconds: float

    @property
    def depth(self) -> int:
        return int(self.time_grid.shape[0]) - 1

    def metrics(self) -> dict[str, Any]:
        """Flat dictionary of the scalar fields, for the phase table."""
        skip = {
            "params",
            "time_grid",
            "optimizer_state",
            "control",
            "adam_history",
            "bfgs_history",
        }
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in skip
        }


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #
def solve_fixed_depth(
    initial_params,
    time_grid,
    data: Dataset,
    problem: ProblemConfig,
    solver: SolverConfig,
    *,
    fixed_W_in,
    fixed_W_out,
    label: str = "phase",
    verbose: bool = True,
    initial_optimizer_state=None,
) -> PhaseResult:
    """Run Algorithm 1 on one fixed temporal grid."""
    start = time.perf_counter()
    tolerance = float(solver.stationarity_tolerance)
    control = make_control_problem(
        time_grid,
        data,
        problem,
        solver,
        fixed_W_in=fixed_W_in,
        fixed_W_out=fixed_W_out,
    )

    theta = node.pack_controls(initial_params)
    state = (
        optimizers.control_state_copy(initial_optimizer_state)
        if initial_optimizer_state is not None
        else optimizers.control_state_init(theta)
    )

    values = control.evaluate(theta)
    initial_objective = float(values["objective"])
    initial_stationarity = float(values["stationarity_norm"])
    current_lr = float(solver.adam_learning_rate)

    best_stationarity = initial_stationarity
    best_stationarity_objective = initial_objective
    best_stationarity_theta = jnp.array(theta, copy=True)
    best_stationarity_state = optimizers.control_state_copy(state)

    best_feasible_objective = np.inf
    best_feasible_theta = None
    best_feasible_state = None

    history = [
        {
            "step": 0,
            "objective": initial_objective,
            "stationarity": initial_stationarity,
            "learning_rate": current_lr,
        }
    ]
    steps_used = 0
    lr_reductions = 0
    checks_without_improvement = 0
    consecutive_feasible_checks = 0
    stopping_reason = "maximum Adam steps reached"
    bfgs_triggered = False

    # ---------------------------------------------------------------- Adam
    while steps_used < int(solver.adam_max_steps):
        n_steps = min(
            int(solver.adam_check_every), int(solver.adam_max_steps) - steps_used
        )
        theta, state, values = control.adam_chunk(
            theta, state, jnp.asarray(current_lr, dtype=dtype()), n_steps=int(n_steps)
        )
        steps_used += int(n_steps)
        objective = float(values["objective"])
        stationarity = float(values["stationarity_norm"])

        if not np.isfinite(objective) or not np.isfinite(stationarity):
            stopping_reason = "non-finite objective or stationarity norm"
            break

        history.append(
            {
                "step": steps_used,
                "objective": objective,
                "stationarity": stationarity,
                "learning_rate": current_lr,
            }
        )

        relative_improvement = (best_stationarity - stationarity) / (
            abs(best_stationarity) + 1.0e-30
        )
        if stationarity < best_stationarity:
            best_stationarity = stationarity
            best_stationarity_objective = objective
            best_stationarity_theta = jnp.array(theta, copy=True)
            best_stationarity_state = optimizers.control_state_copy(state)

        if relative_improvement >= solver.adam_relative_improvement:
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        if stationarity <= tolerance and objective < best_feasible_objective:
            best_feasible_objective = objective
            best_feasible_theta = jnp.array(theta, copy=True)
            best_feasible_state = optimizers.control_state_copy(state)

        if steps_used >= int(solver.adam_min_steps) and stationarity <= tolerance:
            consecutive_feasible_checks += 1
        else:
            consecutive_feasible_checks = 0

        if verbose:
            print(
                f"  {label}: Adam step={steps_used:5d}, "
                f"J={objective:.8f}, ||G||_H1={stationarity:.3e}, "
                f"lr={current_lr:.2e}, best={best_stationarity:.3e}"
            )

        if (
            solver.use_bfgs_polish
            and steps_used >= int(solver.adam_min_steps)
            and stationarity <= solver.bfgs_switch_ratio * tolerance
        ):
            bfgs_triggered = True
            stopping_reason = (
                f"entered BFGS switch region "
                f"({solver.bfgs_switch_ratio:.2f} x tolerance)"
            )
            break

        if consecutive_feasible_checks >= int(solver.adam_convergence_patience):
            stopping_reason = (
                f"stationarity tolerance reached for "
                f"{solver.adam_convergence_patience} consecutive checks"
            )
            break

        if checks_without_improvement >= int(solver.adam_plateau_patience):
            if current_lr > solver.adam_min_learning_rate * (1.0 + 1.0e-12):
                current_lr = max(
                    solver.adam_min_learning_rate, current_lr * solver.adam_lr_decay
                )
                lr_reductions += 1
                checks_without_improvement = 0
                if solver.adam_reset_moments_on_lr_decay:
                    state = optimizers.control_state_init(theta)
                if verbose:
                    print(f"  {label}: reducing Adam learning rate to {current_lr:.2e}")
            else:
                stopping_reason = "Adam stationarity plateau at the minimum learning rate"
                break

    if best_feasible_theta is not None:
        adam_theta = best_feasible_theta
        adam_state = best_feasible_state
        adam_selected_reason = (
            "lowest-objective Adam checkpoint satisfying the common tolerance"
        )
    else:
        adam_theta = best_stationarity_theta
        adam_state = best_stationarity_state
        adam_selected_reason = (
            "Adam checkpoint with the smallest observed H1 stationarity norm"
        )

    adam_values = control.evaluate(adam_theta)
    adam_objective = float(adam_values["objective"])
    adam_stationarity = float(adam_values["stationarity_norm"])

    # ---------------------------------------------------------------- BFGS
    bfgs_result = None
    selected_optimizer = "adam"
    selected_reason = adam_selected_reason
    theta = adam_theta
    state = adam_state

    if bfgs_triggered:
        if verbose:
            print(
                f"  {label}: switching from Adam to {solver.bfgs_method} "
                f"at ||G||_H1={adam_stationarity:.3e}"
            )
        bfgs_result = _run_bfgs_polish(
            adam_theta,
            control,
            solver,
            tolerance=tolerance,
            label=label,
            verbose=verbose,
        )
        bfgs_key = _candidate_key(
            bfgs_result["objective"], bfgs_result["stationarity_norm"], tolerance
        )
        adam_key = _candidate_key(adam_objective, adam_stationarity, tolerance)
        if bfgs_key < adam_key:
            theta = bfgs_result["theta"]
            # BFGS changes the control without updating Adam's moments, so a
            # fresh state is the only consistent thing to transfer onward.
            state = optimizers.control_state_init(theta)
            selected_optimizer = solver.bfgs_method.lower()
            selected_reason = bfgs_result["selected_iterate_reason"]

    # ------------------------------------------------------------- Report
    final_values = control.evaluate(theta)
    params = node.params_from_control_matrix(
        theta, fixed_W_in, fixed_W_out, problem.hidden_dim
    )
    stationarity = float(final_values["stationarity_norm"])
    train_accuracy = float(control.accuracy_fn(params, data.X_train, data.y_train))
    validation_loss = float(control.loss_fn(params, data.X_val, data.y_val))
    validation_accuracy = float(control.accuracy_fn(params, data.X_val, data.y_val))
    margins = np.asarray(
        control.margin_fn(params, data.X_val, data.y_val), dtype=float
    )
    params["A"].block_until_ready()
    elapsed = time.perf_counter() - start

    empty_history = pd.DataFrame(columns=["iteration", "objective", "stationarity"])
    result = PhaseResult(
        params=params,
        time_grid=jnp.asarray(time_grid, dtype=dtype()),
        optimizer_state=optimizers.control_state_copy(state),
        control=control,
        objective=float(final_values["objective"]),
        data_loss=float(final_values["data_loss"]),
        regularizer=float(final_values["regularizer"]),
        mass_energy=float(final_values["mass_energy"]),
        stiffness_energy=float(final_values["stiffness_energy"]),
        h1_energy=float(final_values["h1_energy"]),
        stationarity_norm=stationarity,
        stationarity_tolerance=tolerance,
        stationarity_ratio=stationarity / tolerance,
        converged=bool(stationarity <= tolerance),
        initial_objective=initial_objective,
        initial_stationarity_norm=initial_stationarity,
        best_stationarity_seen=float(best_stationarity),
        best_stationarity_objective=float(best_stationarity_objective),
        train_accuracy=train_accuracy,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        margin_median=float(np.median(margins)),
        margin_q10=float(np.quantile(margins, 0.10)),
        solver_mode=(
            f"adam+{solver.bfgs_method.lower()}" if bfgs_triggered else "adam"
        ),
        selected_optimizer=selected_optimizer,
        selected_iterate_reason=selected_reason,
        adam_steps=int(steps_used),
        adam_final_learning_rate=float(current_lr),
        adam_lr_reductions=int(lr_reductions),
        adam_stopping_reason=stopping_reason,
        adam_history=pd.DataFrame(history),
        bfgs_enabled=bool(solver.use_bfgs_polish),
        bfgs_triggered=bool(bfgs_triggered),
        bfgs_method=solver.bfgs_method,
        bfgs_iterations=int(bfgs_result["iterations"]) if bfgs_result else 0,
        bfgs_function_evaluations=(
            int(bfgs_result["function_evaluations"]) if bfgs_result else 0
        ),
        bfgs_stopping_reason=(
            str(bfgs_result["stopping_reason"])
            if bfgs_result
            else "BFGS polish not triggered"
        ),
        bfgs_elapsed_seconds=(
            float(bfgs_result["elapsed_seconds"]) if bfgs_result else 0.0
        ),
        bfgs_history=bfgs_result["history"] if bfgs_result else empty_history,
        elapsed_seconds=elapsed,
    )

    if verbose:
        print(
            f"{label:>22s} | K={result.depth:3d} | "
            f"J={result.objective:.7f} | val={validation_loss:.6f} "
            f"(acc={validation_accuracy:.4f}) | "
            f"||G||_H1={stationarity:.3e} ({result.stationarity_ratio:.2f} x tol) | "
            f"Adam={steps_used:5d} | BFGS={result.bfgs_iterations:4d} | "
            f"stationary={result.converged}"
        )
    return result
