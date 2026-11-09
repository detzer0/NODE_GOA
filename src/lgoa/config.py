"""Configuration objects and the YAML presets for the two published examples.

The hyperparameter bulk lives in ``configs/<name>.yaml``; a notebook loads a
preset and overrides only the few parameters it wants to study::

    cfg = lgoa.load_preset("peaks", stationarity_tolerance=1e-3,
                           marking="dorfler", dorfler_fraction=0.5)

Override keywords are flat: each one is routed to whichever section declares a
field of that name, so the notebook never has to know the section layout.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from . import backend

__all__ = [
    "ProblemConfig",
    "InitializationConfig",
    "SolverConfig",
    "AdaptiveConfig",
    "ExperimentConfig",
    "available_presets",
    "config_path",
    "load_preset",
    "load_yaml",
]

MARKING_RULES = ("conservative", "dorfler")

_CONFIG_DIR_CANDIDATES = (
    Path(__file__).resolve().parent / "configs",       # shipped with the package
    Path(__file__).resolve().parents[2] / "configs",   # repo-root copy, if any
    Path.cwd() / "configs",                            # local override
)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProblemConfig:
    """Data set, network architecture and the horizon :math:`[0, T]`."""

    name: str
    num_classes: int
    hidden_dim: int
    final_time: float
    samples_per_class: int
    train_fraction: float = 0.60
    seed: int = 0
    grid_size: int | None = None          # Peaks only: 256 x 256 sampling grid
    spiral_noise: float = 0.0             # Swiss roll only
    peaks_thresholds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive.")
        if self.samples_per_class < 2:
            raise ValueError("samples_per_class must be at least two.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1.")
        if self.spiral_noise < 0.0:
            raise ValueError("spiral_noise must be nonnegative.")

    @property
    def validation_fraction(self) -> float:
        return 1.0 - self.train_fraction


@dataclass(frozen=True)
class InitializationConfig:
    """One-time endpoint-map calibration shared by every refinement path."""

    search_max_epochs: int = 1000
    search_min_epochs: int = 150
    check_every: int = 25
    validation_patience: int = 8
    validation_min_delta: float = 1.0e-4
    learning_rate: float = 2.0e-3
    reg_lambda: float | None = None       # None reuses SolverConfig.reg_lambda
    control_init_noise: float = 0.0
    model_seed: int = 0
    n_candidates: int = 1                 # >1 keeps the best by validation loss
    seed_stride: int = 1000

    def __post_init__(self) -> None:
        if self.search_max_epochs < 1:
            raise ValueError("search_max_epochs must be positive.")
        if self.check_every < 1:
            raise ValueError("check_every must be positive.")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be positive.")

    @property
    def candidate_seeds(self) -> tuple[int, ...]:
        return tuple(
            int(self.model_seed + self.seed_stride * index)
            for index in range(int(self.n_candidates))
        )


@dataclass(frozen=True)
class SolverConfig:
    r"""Algorithm 1: full-batch :math:`H^1`-Adam warm start plus BFGS polish.

    ``bfgs_switch_ratio`` is the factor :math:`\kappa_s > 1` of the switching
    region :math:`s_\tau \le \kappa_s \varepsilon_{\mathrm{stat}}`.
    """

    reg_lambda: float = 1.0e-3
    stationarity_tolerance: float = 1.0e-3

    adam_learning_rate: float = 1.0e-3
    adam_min_learning_rate: float = 1.0e-4
    adam_lr_decay: float = 0.75
    adam_plateau_patience: int = 4
    adam_convergence_patience: int = 2
    adam_relative_improvement: float = 1.0e-3
    adam_check_every: int = 100
    adam_min_steps: int = 500
    adam_max_steps: int = 20000
    adam_reset_moments_on_lr_decay: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    grad_clip: float = 50.0

    use_bfgs_polish: bool = True
    bfgs_method: str = "L-BFGS-B"
    bfgs_switch_ratio: float = 5.0
    bfgs_max_iterations: int = 10000
    bfgs_ftol: float = 1.0e-13
    bfgs_internal_gtol: float = 1.0e-12
    bfgs_max_line_search_steps: int = 40
    bfgs_memory: int = 20

    def __post_init__(self) -> None:
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be nonnegative.")
        if self.stationarity_tolerance <= 0.0:
            raise ValueError("stationarity_tolerance must be positive.")
        if self.adam_max_steps < 1:
            raise ValueError("adam_max_steps must be positive.")
        if self.bfgs_method.upper() not in {"BFGS", "L-BFGS-B"}:
            raise ValueError("bfgs_method must be 'BFGS' or 'L-BFGS-B'.")
        if self.bfgs_switch_ratio < 1.0:
            raise ValueError("bfgs_switch_ratio must be at least one.")
        if self.bfgs_max_iterations < 1:
            raise ValueError("bfgs_max_iterations must be positive.")
        object.__setattr__(self, "bfgs_method", self.bfgs_method.strip().upper())
        object.__setattr__(
            self, "adam_min_steps", min(self.adam_min_steps, self.adam_max_steps)
        )


@dataclass(frozen=True)
class AdaptiveConfig:
    """Algorithm 2: the outer refinement loop and its marking rule."""

    initial_depth: int = 2
    max_depth: int = 12
    marking: str = "conservative"
    dorfler_fraction: float = 0.5
    transfer_adam_state: bool = True
    allow_nonstationary_refinement: bool = True
    near_stationary_ratio: float = 1.5
    indicator_tolerance: float | None = None
    indicator_sup_samples: int = 50

    def __post_init__(self) -> None:
        if not 1 <= self.initial_depth <= self.max_depth:
            raise ValueError("Require 1 <= initial_depth <= max_depth.")
        if self.marking not in MARKING_RULES:
            raise ValueError(
                f"marking must be one of {MARKING_RULES}, got {self.marking!r}."
            )
        if not 0.0 < self.dorfler_fraction <= 1.0:
            raise ValueError("dorfler_fraction must lie in (0, 1].")
        if self.near_stationary_ratio < 1.0:
            raise ValueError("near_stationary_ratio must be at least one.")
        if self.indicator_sup_samples < 2:
            raise ValueError("indicator_sup_samples must be at least two.")

    @property
    def layer_budget(self) -> int:
        return self.max_depth - self.initial_depth


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything one adaptive run needs."""

    problem: ProblemConfig
    initialization: InitializationConfig
    solver: SolverConfig
    adaptive: AdaptiveConfig
    use_x64: bool = True
    verbose: bool = True

    @property
    def initialization_reg_lambda(self) -> float:
        if self.initialization.reg_lambda is None:
            return self.solver.reg_lambda
        return float(self.initialization.reg_lambda)

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        """Return a copy with flat keyword overrides routed to their sections."""
        return _apply_overrides(self, overrides)

    def summary(self) -> str:
        marking = self.adaptive.marking
        if marking == "dorfler":
            marking = f"Dorfler (bulk fraction {self.adaptive.dorfler_fraction:g})"
        return "\n".join(
            [
                f"problem                 : {self.problem.name} "
                f"(d={self.problem.hidden_dim}, T={self.problem.final_time:g}, "
                f"{self.problem.num_classes} classes)",
                f"regularization lambda   : {self.solver.reg_lambda:g}",
                f"stationarity tolerance  : {self.solver.stationarity_tolerance:g} "
                f"(kappa_s = {self.solver.bfgs_switch_ratio:g})",
                f"depth budget            : K = {self.adaptive.initial_depth} "
                f"-> {self.adaptive.max_depth}",
                f"marking rule            : {marking}",
                f"Adam moment transfer    : "
                f"{'on' if self.adaptive.transfer_adam_state else 'off'}",
                f"BFGS polish             : "
                f"{self.solver.bfgs_method if self.solver.use_bfgs_polish else 'off'}",
            ]
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_SECTIONS = {
    "problem": ProblemConfig,
    "initialization": InitializationConfig,
    "solver": SolverConfig,
    "adaptive": AdaptiveConfig,
}


def _config_dir() -> Path:
    for candidate in _CONFIG_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the 'configs' directory. Install the package from a "
        "clone of the repository, or pass an explicit path to load_yaml()."
    )


def available_presets() -> list[str]:
    return sorted(path.stem for path in _config_dir().glob("*.yaml"))


def config_path(name: str) -> Path:
    """Absolute path of a preset's YAML file, for inspection or editing."""
    path = _config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown preset {name!r}. Available: {available_presets()}")
    return path


def _coerce(cls, payload: Mapping[str, Any]):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - names
    if unknown:
        raise ValueError(
            f"{cls.__name__} received unknown keys: {sorted(unknown)}"
        )
    payload = dict(payload)
    for key, value in payload.items():
        if isinstance(value, list):
            payload[key] = tuple(value)
    return cls(**payload)


def load_yaml(path: str | Path) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    unknown = set(raw) - set(_SECTIONS) - {"use_x64", "verbose"}
    if unknown:
        raise ValueError(f"{path.name} has unknown top-level keys: {sorted(unknown)}")

    sections = {
        name: _coerce(cls, raw.get(name, {})) for name, cls in _SECTIONS.items()
    }
    return ExperimentConfig(
        **sections,
        use_x64=bool(raw.get("use_x64", True)),
        verbose=bool(raw.get("verbose", True)),
    )


def _apply_overrides(config: ExperimentConfig, overrides: Mapping[str, Any]):
    if not overrides:
        return config

    top_level = {f.name for f in dataclasses.fields(ExperimentConfig)} - set(_SECTIONS)
    section_fields = {
        name: {f.name for f in dataclasses.fields(cls)}
        for name, cls in _SECTIONS.items()
    }

    pending: dict[str, dict[str, Any]] = {name: {} for name in _SECTIONS}
    root: dict[str, Any] = {}

    for key, value in overrides.items():
        if key in top_level:
            root[key] = value
            continue
        owners = [name for name, names in section_fields.items() if key in names]
        if not owners:
            raise ValueError(
                f"Unknown configuration parameter {key!r}. Valid names: "
                + ", ".join(
                    sorted(top_level | set().union(*section_fields.values()))
                )
            )
        if len(owners) > 1:
            # reg_lambda exists in two sections; the solver one is the headline.
            owners = ["solver"] if "solver" in owners else owners[:1]
        pending[owners[0]][key] = value

    sections = {
        name: (
            replace(getattr(config, name), **values)
            if values
            else getattr(config, name)
        )
        for name, values in pending.items()
    }
    return replace(config, **sections, **root)


def load_preset(name: str = "swiss_roll", /, **overrides: Any) -> ExperimentConfig:
    """Load ``configs/<name>.yaml``, apply overrides and configure precision.

    Parameters
    ----------
    name:
        Preset stem, e.g. ``"swiss_roll"`` or ``"peaks"``.
    **overrides:
        Flat parameter overrides such as ``stationarity_tolerance=1e-3``,
        ``marking="dorfler"``, ``reg_lambda=1e-2`` or ``max_depth=10``.
    """
    presets = available_presets()
    if name not in presets:
        raise ValueError(f"Unknown preset {name!r}. Available: {presets}")
    config = load_yaml(_config_dir() / f"{name}.yaml").with_overrides(**overrides)
    backend.configure(config.use_x64)
    return config
