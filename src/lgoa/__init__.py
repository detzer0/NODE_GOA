"""Layerwise goal-oriented adaptivity for neural ODEs.

Reference implementation of the solver of

    M. Hintermüller, M. Hinze and D. Korolev,
    *Layerwise goal-oriented adaptivity for neural ODEs: an optimal control
    perspective*, Journal of Numerical Mathematics (2026).
    https://doi.org/10.1515/jnma-2026-0008

Typical use::

    import lgoa

    cfg = lgoa.load_preset("swiss_roll", marking="dorfler", dorfler_fraction=0.5)
    data = lgoa.load_dataset(cfg.problem)
    history = lgoa.run(cfg, data)
    lgoa.plots.learning_curves(history)

Module map:

==========================  ====================================================
:mod:`lgoa.fem`             temporal CG(1) operators, discrete :math:`H^1` metric
:mod:`lgoa.neural_ode`      forward pass, discrete adjoint, exact derivative
:mod:`lgoa.solver`          Algorithm 1 (fixed-depth Adam + BFGS polish)
:mod:`lgoa.indicators`      local goal-oriented indicator :math:`\\eta_k`
:mod:`lgoa.marking`         conservative and Dörfler marking
:mod:`lgoa.refinement`      layer insertion and midpoint prolongation
:mod:`lgoa.adaptive`        Algorithm 2 (outer adaptive loop)
:mod:`lgoa.history`         phase and event records, CSV export
:mod:`lgoa.plots`           the figures of Section 8
:mod:`lgoa.verification`    implementation self-checks
==========================  ====================================================
"""

from __future__ import annotations

from . import (
    adaptive,
    backend,
    datasets,
    fem,
    history,
    indicators,
    marking,
    neural_ode,
    optimizers,
    plots,
    refinement,
    solver,
    verification,
)
from .adaptive import run
from .backend import backend_report, configure
from .config import (
    AdaptiveConfig,
    ExperimentConfig,
    InitializationConfig,
    ProblemConfig,
    SolverConfig,
    available_presets,
    config_path,
    load_preset,
    load_yaml,
)
from .datasets import Dataset
from .datasets import load as load_dataset
from .history import History, Phase, RefinementEvent
from .indicators import LocalIndicators
from .initialization import Calibration, calibrate_endpoint_maps
from .verification import self_check

__version__ = "1.0.0"

__all__ = [
    "AdaptiveConfig",
    "Calibration",
    "Dataset",
    "ExperimentConfig",
    "History",
    "InitializationConfig",
    "LocalIndicators",
    "Phase",
    "ProblemConfig",
    "RefinementEvent",
    "SolverConfig",
    "adaptive",
    "available_presets",
    "backend",
    "backend_report",
    "calibrate_endpoint_maps",
    "config_path",
    "configure",
    "datasets",
    "fem",
    "history",
    "indicators",
    "load_dataset",
    "load_preset",
    "load_yaml",
    "marking",
    "neural_ode",
    "optimizers",
    "plots",
    "refinement",
    "run",
    "self_check",
    "solver",
    "verification",
    "__version__",
]
