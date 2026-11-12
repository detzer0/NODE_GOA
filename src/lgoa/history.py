"""Records produced by one adaptive run.

A :class:`History` is the single object the notebooks pass to
:mod:`lgoa.plots`: it holds one :class:`Phase` per fixed-depth solve, one
:class:`RefinementEvent` per layer insertion, and tidy
:class:`pandas.DataFrame` views for tables and CSV export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .indicators import LocalIndicators
from .initialization import Calibration
from .marking import MarkingResult
from .solver import PhaseResult

__all__ = ["Phase", "RefinementEvent", "History"]

_SCALAR_PHASE_FIELDS = (
    "objective",
    "data_loss",
    "regularizer",
    "mass_energy",
    "stiffness_energy",
    "h1_energy",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "margin_median",
    "margin_q10",
    "stationarity_norm",
    "stationarity_tolerance",
    "stationarity_ratio",
    "converged",
    "solver_mode",
    "selected_optimizer",
    "selected_iterate_reason",
    "adam_steps",
    "adam_final_learning_rate",
    "adam_lr_reductions",
    "adam_stopping_reason",
    "bfgs_triggered",
    "bfgs_iterations",
    "bfgs_function_evaluations",
    "bfgs_stopping_reason",
    "bfgs_elapsed_seconds",
    "elapsed_seconds",
)

_INDICATOR_PHASE_FIELDS = (
    "global_state_indicator",
    "global_adjoint_indicator",
    "global_control_indicator",
    "global_practical_indicator",  # retained name for existing consumers
    "global_direct_indicator",
    "direct_quadrature_order",
    "direct_quadrature_error_estimate",
    "direct_quadrature_converged",
    "reconstruction_jump_trace_error",
    "control_fraction",
    "max_abs_control_stiffness_term",
)


@dataclass
class Phase:
    """One accepted fixed-depth solution together with its local indicators."""

    index: int
    result: PhaseResult
    indicators: LocalIndicators
    marking: MarkingResult | None = None
    near_stationary: bool = False
    continued_without_stationarity: bool = False

    @property
    def depth(self) -> int:
        return self.result.depth

    @property
    def grid(self) -> np.ndarray:
        return np.asarray(self.result.time_grid, dtype=float)

    @property
    def params(self) -> dict:
        return self.result.params

    @property
    def eta(self) -> np.ndarray:
        """Local indicators :math:`\\eta_k` on this grid."""
        return np.asarray(self.indicators.total, dtype=float)

    @property
    def stationarity_status(self) -> str:
        if self.result.converged:
            return "strict"
        if self.near_stationary:
            return "near"
        if self.continued_without_stationarity:
            return "continued"
        return "failed"

    def row(self) -> dict[str, Any]:
        """Flat record for the phase table."""
        record: dict[str, Any] = {"phase": self.index, "depth": self.depth}
        record.update(
            {name: getattr(self.result, name) for name in _SCALAR_PHASE_FIELDS}
        )
        record.update(
            {
                name: getattr(self.indicators, name)
                for name in _INDICATOR_PHASE_FIELDS
            }
        )
        record["max_local_indicator"] = float(np.max(self.eta))
        record["stationarity_status"] = self.stationarity_status
        if self.marking is None:
            record.update(
                {
                    "marked_intervals": "",
                    "marked_count": 0,
                    "marked_indicator_share": np.nan,
                    "budget_capped": False,
                }
            )
        else:
            record.update(
                {
                    "marked_intervals": ",".join(
                        str(index) for index in self.marking.marked_intervals
                    ),
                    "marked_count": self.marking.count,
                    "marked_indicator_share": self.marking.achieved_share,
                    "budget_capped": self.marking.budget_capped,
                }
            )
        return record


@dataclass
class RefinementEvent:
    """One layer-insertion round, with indicators before, during and after."""

    index: int
    depth_before: int
    depth_after: int
    marking: MarkingResult
    marked_bounds: tuple[tuple[float, float], ...]
    inserted_times: tuple[float, ...]
    parent_to_children: tuple[dict[str, Any], ...]
    before: Phase
    after_immediate: LocalIndicators
    after_reoptimized: Phase | None = None

    @property
    def marked_intervals(self) -> tuple[int, ...]:
        return self.marking.marked_intervals

    @property
    def inserted_layers(self) -> int:
        return self.marking.count

    @property
    def child_intervals(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    int(child)
                    for mapping in self.parent_to_children
                    for child in mapping["child_intervals"]
                }
            )
        )

    @property
    def completed(self) -> bool:
        return self.after_reoptimized is not None

    def parent_child_ratio(self) -> float:
        """Summed child indicators over summed parent indicators, after re-optimization.

        A value below one means re-optimization reduced the indicator mass on
        the refined region, which is the behaviour the marking rule intends.
        """
        if self.after_reoptimized is None:
            return float("nan")
        parents = self.before.eta[list(self.marked_intervals)]
        children = self.after_reoptimized.eta[list(self.child_intervals)]
        denominator = float(np.sum(parents))
        if denominator <= 0.0:
            return float("nan")
        return float(np.sum(children) / denominator)

    def row(self) -> dict[str, Any]:
        return {
            "event": self.index,
            "depth_before": self.depth_before,
            "depth_after_insertion": self.depth_after,
            "inserted_layers": self.inserted_layers,
            "marked_intervals": ",".join(str(k) for k in self.marked_intervals),
            "target_indicator_share": self.marking.target_share,
            "achieved_indicator_share": self.marking.achieved_share,
            "budget_capped": self.marking.budget_capped,
            "inserted_times": ",".join(f"{t:.6f}" for t in self.inserted_times),
            "parent_indicator_sum": float(
                np.sum(self.before.eta[list(self.marked_intervals)])
            ),
            "post_insertion_child_sum": float(
                np.sum(
                    np.asarray(self.after_immediate.total, dtype=float)[
                        list(self.child_intervals)
                    ]
                )
            ),
            "post_reoptimization_child_sum": (
                float(
                    np.sum(
                        self.after_reoptimized.eta[list(self.child_intervals)]
                    )
                )
                if self.after_reoptimized is not None
                else np.nan
            ),
            "children_to_parent_ratio": self.parent_child_ratio(),
            "post_insertion_direct_quadrature_order": (
                self.after_immediate.direct_quadrature_order
            ),
            "post_insertion_direct_quadrature_error_estimate": (
                self.after_immediate.direct_quadrature_error_estimate
            ),
            "post_insertion_direct_quadrature_converged": (
                self.after_immediate.direct_quadrature_converged
            ),
        }


@dataclass
class History:
    """Everything one adaptive run produced."""

    config: ExperimentConfig
    calibration: Calibration
    phases: list[Phase] = field(default_factory=list)
    events: list[RefinementEvent] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    stop_reason: str = ""

    # ------------------------------------------------------------------ views
    @property
    def phase_table(self) -> pd.DataFrame:
        return pd.DataFrame([phase.row() for phase in self.phases])

    @property
    def event_table(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame([event.row() for event in self.events])

    @property
    def final_phase(self) -> Phase:
        return self.phases[-1]

    @property
    def best_phase(self) -> Phase:
        return min(
            self.phases,
            key=lambda phase: (
                phase.result.validation_loss,
                -phase.result.validation_accuracy,
            ),
        )

    @property
    def completed_events(self) -> list[RefinementEvent]:
        return [event for event in self.events if event.completed]

    @property
    def depths(self) -> np.ndarray:
        return np.asarray([phase.depth for phase in self.phases], dtype=int)

    @property
    def inserted_layers(self) -> int:
        return int(sum(event.inserted_layers for event in self.events))

    @property
    def stationary_fraction(self) -> float:
        if not self.phases:
            return float("nan")
        return float(
            np.mean([phase.result.converged for phase in self.phases])
        )

    def totals(self) -> dict[str, Any]:
        """Headline numbers of the run, for a one-line notebook summary."""
        final = self.final_phase
        return {
            "marking": self.config.adaptive.marking,
            "dorfler_fraction": (
                self.config.adaptive.dorfler_fraction
                if self.config.adaptive.marking == "dorfler"
                else np.nan
            ),
            "reg_lambda": self.config.solver.reg_lambda,
            "final_depth": final.depth,
            "refinement_rounds": len(self.events),
            "inserted_layers": self.inserted_layers,
            "final_objective": final.result.objective,
            "final_validation_loss": final.result.validation_loss,
            "final_validation_accuracy": final.result.validation_accuracy,
            "final_stationarity_norm": final.result.stationarity_norm,
            "stationary_phase_fraction": self.stationary_fraction,
            "total_adam_steps": int(
                sum(phase.result.adam_steps for phase in self.phases)
            ),
            "total_bfgs_iterations": int(
                sum(phase.result.bfgs_iterations for phase in self.phases)
            ),
            "wall_time_seconds": self.elapsed_seconds,
        }

    def summary(self) -> str:
        totals = self.totals()
        return "\n".join(
            [
                f"final depth            : K = {totals['final_depth']}"
                f" after {totals['refinement_rounds']} refinement rounds",
                f"final objective        : {totals['final_objective']:.8f}",
                f"final validation loss  : {totals['final_validation_loss']:.6f}"
                f"  (accuracy {totals['final_validation_accuracy']:.4f})",
                f"final ||G||_H1         : {totals['final_stationarity_norm']:.3e}",
                f"stationary phases      : {totals['stationary_phase_fraction']:.0%}",
                f"optimizer work         : {totals['total_adam_steps']:,} Adam steps,"
                f" {totals['total_bfgs_iterations']:,} BFGS iterations",
                f"wall time              : {totals['wall_time_seconds']:.1f} s",
            ]
        )

    # ------------------------------------------------------------------ export
    def to_csv(self, directory: str | Path, prefix: str | None = None) -> list[Path]:
        """Write the phase table, event table and final grid as CSV files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = prefix or f"{self.config.problem.name}_{self.config.adaptive.marking}"

        written = []
        targets = {
            f"{stem}_phases.csv": self.phase_table,
            f"{stem}_events.csv": self.event_table,
            f"{stem}_final_grid.csv": pd.DataFrame(
                {
                    "node": np.arange(self.final_phase.grid.size),
                    "time": self.final_phase.grid,
                }
            ),
            f"{stem}_final_indicators.csv": pd.DataFrame(
                {
                    "interval": np.arange(self.final_phase.eta.size),
                    "left": self.final_phase.grid[:-1],
                    "right": self.final_phase.grid[1:],
                    "eta_state": self.final_phase.indicators.state,
                    "eta_adjoint": self.final_phase.indicators.adjoint,
                    "eta_control": self.final_phase.indicators.control,
                    "eta_total": self.final_phase.eta,
                    "rho_state_signed": self.final_phase.indicators.state_signed,
                    "rho_adjoint_signed": self.final_phase.indicators.adjoint_signed,
                    "rho_control_signed": self.final_phase.indicators.control_signed,
                    "rho_control_mass_signed": (
                        self.final_phase.indicators.control_mass_signed
                    ),
                    "rho_control_stiffness_signed": (
                        self.final_phase.indicators.control_stiffness_signed
                    ),
                    "rho_control_dynamics_signed": (
                        self.final_phase.indicators.control_dynamics_signed
                    ),
                    "direct_quadrature_order": (
                        self.final_phase.indicators.direct_quadrature_order
                    ),
                    "direct_quadrature_error_estimate": (
                        self.final_phase.indicators.direct_quadrature_error_estimate
                    ),
                    "direct_quadrature_converged": (
                        self.final_phase.indicators.direct_quadrature_converged
                    ),
                }
            ),
        }
        for name, frame in targets.items():
            if frame is None or frame.empty:
                continue
            path = directory / name
            frame.to_csv(path, index=False)
            written.append(path)
        return written
