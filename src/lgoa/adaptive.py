r"""Algorithm 2 -- goal-oriented adaptive training.

The outer loop alternates

.. math::

    \text{solve (Alg. 1)} \;\to\; \text{estimate } \{\eta_k\}
    \;\to\; \mathcal{M}_K = \mathrm{Mark}(\mathcal{K}, \{\eta_k\})
    \;\to\; \mathcal{K}^{+} = \mathrm{Refine}(\mathcal{K}, \mathcal{M}_K)
    \;\to\; \Theta^{(0)}_{\tau^{+}} = \Pi_{\tau \to \tau^{+}} \Theta^{\star}_{\tau},

until the layer budget :math:`K_{\max}` is exhausted.  Every refinement round
also records the local indicators immediately after insertion, before
re-optimization, which is what makes the parent/child diagnostics possible.

Non-stationary phases are recorded and, when
``allow_nonstationary_refinement`` is set, refinement continues with a warning;
otherwise the loop stops at the stationarity gate.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from . import indicators as indicator_module
from . import marking as marking_module
from . import refinement as refinement_module
from . import solver as solver_module
from .backend import dtype
from .config import ExperimentConfig
from .datasets import Dataset
from .datasets import load as load_dataset
from .history import History, Phase, RefinementEvent
from .initialization import Calibration, calibrate_endpoint_maps
from .neural_ode import initial_time_grid
from .trees import copy_tree

__all__ = ["run"]


def _indicators_on_grid(params, time_grid, data, problem, solver, adaptive, *, keys):
    control = solver_module.make_control_problem(
        time_grid, data, problem, solver, **keys
    )
    return indicator_module.compute_local_indicators(
        params,
        time_grid,
        data.X_indicator,
        data.y_indicator,
        control.solution_fn,
        reg_lambda=solver.reg_lambda,
        direct_quadrature_initial_order=adaptive.direct_quadrature_initial_order,
        direct_quadrature_max_order=adaptive.direct_quadrature_max_order,
        direct_quadrature_rtol=adaptive.direct_quadrature_rtol,
        direct_quadrature_atol=adaptive.direct_quadrature_atol,
        control_pullback_quadrature=adaptive.control_pullback_quadrature,
    )


def run(
    config: ExperimentConfig,
    data: Dataset | None = None,
    calibration: Calibration | None = None,
    *,
    progress: bool = True,
) -> History:
    """Execute Algorithm 2 for one configuration and return its history.

    Parameters
    ----------
    config:
        Experiment configuration, e.g. from :func:`lgoa.load_preset`.
    data:
        Pre-built data set.  Loaded from ``config.problem`` when omitted.
    calibration:
        Pre-computed endpoint maps.  Reuse one across several marking rules to
        make the comparison paired; computed here when omitted.
    progress:
        Print one line per phase and per refinement round.  Inner Adam/BFGS
        logging is controlled separately by ``config.verbose``.
    """
    problem, solver = config.problem, config.solver
    adaptive = config.adaptive
    data = load_dataset(problem) if data is None else data

    grid = initial_time_grid(problem.final_time, adaptive.initial_depth)

    if calibration is None:
        if progress:
            print(f"Calibrating endpoint maps on K = {adaptive.initial_depth} ...")
        calibration = calibrate_endpoint_maps(
            grid,
            data,
            problem,
            solver,
            config.initialization,
            reg_lambda=config.initialization_reg_lambda,
            verbose=False,
        )
        if progress:
            print(calibration.summary())

    keys = {
        "fixed_W_in": calibration.fixed_W_in,
        "fixed_W_out": calibration.fixed_W_out,
    }
    params = copy_tree(calibration.params)
    time_grid = jnp.asarray(grid, dtype=dtype())
    optimizer_state = None

    history = History(config=config, calibration=calibration)
    pending_event: RefinementEvent | None = None
    phase_index = 0
    start = time.perf_counter()

    while True:
        phase_index += 1
        phase_result = solver_module.solve_fixed_depth(
            params,
            time_grid,
            data,
            problem,
            solver,
            label=f"{adaptive.marking} phase {phase_index}",
            verbose=config.verbose,
            initial_optimizer_state=optimizer_state,
            **keys,
        )
        params = phase_result.params
        time_grid = phase_result.time_grid
        optimizer_state = phase_result.optimizer_state

        eta = indicator_module.compute_local_indicators(
            params,
            time_grid,
            data.X_indicator,
            data.y_indicator,
            phase_result.control.solution_fn,
            reg_lambda=solver.reg_lambda,
            direct_quadrature_initial_order=adaptive.direct_quadrature_initial_order,
            direct_quadrature_max_order=adaptive.direct_quadrature_max_order,
            direct_quadrature_rtol=adaptive.direct_quadrature_rtol,
            direct_quadrature_atol=adaptive.direct_quadrature_atol,
            control_pullback_quadrature=adaptive.control_pullback_quadrature,
        )

        near_stationary = bool(
            (not phase_result.converged)
            and phase_result.stationarity_ratio <= adaptive.near_stationary_ratio
        )
        phase = Phase(
            index=phase_index,
            result=phase_result,
            indicators=eta,
            near_stationary=near_stationary,
            continued_without_stationarity=bool(
                (not phase_result.converged)
                and (not near_stationary)
                and adaptive.allow_nonstationary_refinement
            ),
        )
        history.phases.append(phase)

        if pending_event is not None and pending_event.after_reoptimized is None:
            pending_event.after_reoptimized = phase
            pending_event = None

        if progress and not config.verbose:
            print(
                f"  phase {phase_index:2d} | K={phase.depth:3d} | "
                f"J={phase_result.objective:.7f} | "
                f"val={phase_result.validation_loss:.6f} "
                f"(acc={phase_result.validation_accuracy:.4f}) | "
                f"||G||_H1={phase_result.stationarity_norm:.3e} "
                f"({phase.stationarity_status})"
            )

        # ------------------------------------------------- stationarity gate
        if not phase_result.converged:
            if near_stationary:
                if progress:
                    print(
                        "  continuing with a near-stationary phase: "
                        f"||G||_H1/tol = {phase_result.stationarity_ratio:.2f}"
                    )
            elif adaptive.allow_nonstationary_refinement:
                if progress:
                    print(
                        "  WARNING: refining without strict or near stationarity; "
                        f"||G||_H1/tol = {phase_result.stationarity_ratio:.2f}"
                    )
            else:
                history.stop_reason = "stationarity gate not met"
                break

        # ----------------------------------------------- outer stopping rules
        if phase.depth >= adaptive.max_depth:
            history.stop_reason = f"layer budget K_max = {adaptive.max_depth} reached"
            break
        if (
            adaptive.indicator_tolerance is not None
            and np.max(phase.eta) <= adaptive.indicator_tolerance
        ):
            history.stop_reason = "local-indicator tolerance reached"
            break

        # ------------------------------------------------------------- mark
        remaining_budget = int(adaptive.max_depth - phase.depth)
        result = marking_module.mark(
            phase.eta,
            adaptive.marking,
            max_new_layers=remaining_budget,
            dorfler_fraction=adaptive.dorfler_fraction,
        )
        phase.marking = result

        # ----------------------------------------------------------- refine
        old_grid = np.asarray(time_grid, dtype=float)
        refined = refinement_module.refine(
            params,
            time_grid,
            optimizer_state,
            result.marked_intervals,
            transfer_adam_state=adaptive.transfer_adam_state,
        )
        params = refined.params
        time_grid = refined.time_grid
        optimizer_state = refined.optimizer_state

        immediate = _indicators_on_grid(
            params, time_grid, data, problem, solver, adaptive, keys=keys
        )

        event = RefinementEvent(
            index=len(history.events) + 1,
            depth_before=phase.depth,
            depth_after=int(len(time_grid) - 1),
            marking=result,
            marked_bounds=tuple(
                (float(old_grid[k]), float(old_grid[k + 1]))
                for k in result.marked_intervals
            ),
            inserted_times=refined.inserted_times,
            parent_to_children=refined.parent_to_children,
            before=phase,
            after_immediate=immediate,
        )
        history.events.append(event)
        pending_event = event

        if progress:
            marked_text = ", ".join(str(k) for k in result.marked_intervals)
            print(
                f"  marked {result.count} interval(s) [{marked_text}] | "
                f"indicator share = {result.achieved_share:.1%} | "
                f"K: {phase.depth} -> {len(time_grid) - 1}"
                + (" | budget cap active" if result.budget_capped else "")
            )

        # Defensive guard against accidental non-growth of the partition.
        if phase_index > adaptive.layer_budget + 2:
            raise RuntimeError("The refinement loop exceeded the layer budget.")

    history.elapsed_seconds = float(time.perf_counter() - start)
    if progress:
        print(f"\nStopped: {history.stop_reason}")
    return history
