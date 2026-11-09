r"""The two classification problems of the paper.

* **Swiss roll** (Section 8.1) -- two concentric spirals in :math:`\mathbb{R}^2`,

  .. math::
      f_{\mathrm{blue}}(r, \varphi) = r(\cos\varphi, \sin\varphi)^\top, \qquad
      f_{\mathrm{red}}(r, \varphi) = (r + 0.2)(\cos\varphi, \sin\varphi)^\top,

  for :math:`r \in [0, 1]` and :math:`\varphi \in [0, 4\pi]`.

* **Peaks** (Section 8.2) -- five classes obtained by thresholding the Peaks
  function on a regular grid over :math:`[-3, 3]^2`.

Both are split once with a stratified train/validation split; the input scaler
is fitted on the training part only.  The objective, state, adjoint,
:math:`H^1` gradient and the local indicators all use the full training set;
the validation set is used for initialization selection and diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .backend import dtype
from .config import ProblemConfig

__all__ = ["Dataset", "load", "peaks_function", "swiss_roll", "peaks"]


@dataclass(frozen=True)
class Dataset:
    """Scaled training/validation tensors plus raw arrays for plotting."""

    name: str
    X_train: jnp.ndarray
    y_train: jnp.ndarray
    X_val: jnp.ndarray
    y_val: jnp.ndarray
    num_classes: int
    scaler: StandardScaler
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.X_val.shape[0])

    # The indicator and every objective evaluation use the complete
    # training set; the aliases make that explicit at the call sites.
    @property
    def X_indicator(self):
        return self.X_train

    @property
    def y_indicator(self):
        return self.y_train

    def summary(self) -> str:
        return (
            f"{self.name}: train {tuple(self.X_train.shape)}, "
            f"validation {tuple(self.X_val.shape)}, "
            f"{self.num_classes} classes"
        )


def peaks_function(X, Y):
    """MATLAB-style Peaks function used to define the five classes."""
    return (
        3.0 * (1.0 - X) ** 2 * np.exp(-(X**2) - (Y + 1.0) ** 2)
        - 10.0 * (X / 5.0 - X**3 - Y**5) * np.exp(-(X**2) - Y**2)
        - (1.0 / 3.0) * np.exp(-((X + 1.0) ** 2) - Y**2)
    )


def swiss_roll(problem: ProblemConfig) -> Dataset:
    """Two-class concentric-spiral data of Section 8.1."""
    n = int(problem.samples_per_class)
    rng = np.random.default_rng(problem.seed + 42)

    phi = np.linspace(0.1, 4.0 * np.pi, n)
    radius_inner = np.linspace(0.0, 1.0, n)
    radius_outer = radius_inner + 0.2

    class_0 = np.stack([radius_inner * np.cos(phi), radius_inner * np.sin(phi)], axis=1)
    class_1 = np.stack([radius_outer * np.cos(phi), radius_outer * np.sin(phi)], axis=1)

    X_raw = np.vstack([class_0, class_1])
    if problem.spiral_noise > 0.0:
        X_raw = X_raw + problem.spiral_noise * rng.normal(size=X_raw.shape)
    y = np.concatenate([np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.int32)])

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw,
        y,
        test_size=problem.validation_fraction,
        stratify=y,
        random_state=problem.seed + 11,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    span = np.ptp(X_raw, axis=0)
    margin = 0.12 * np.maximum(span, 1.0e-12)

    return Dataset(
        name="Swiss roll",
        X_train=jnp.asarray(X_train, dtype=dtype()),
        y_train=jnp.asarray(y_train, dtype=jnp.int32),
        X_val=jnp.asarray(X_val, dtype=dtype()),
        y_val=jnp.asarray(y_val, dtype=jnp.int32),
        num_classes=2,
        scaler=scaler,
        extras={
            "X_all_raw": X_raw,
            "y_all": y,
            "X_train_raw": X_train_raw,
            "X_val_raw": X_val_raw,
            "y_train_np": y_train,
            "y_val_np": y_val,
            "class_0_raw": class_0,
            "class_1_raw": class_1,
            "plot_lower": np.min(X_raw, axis=0) - margin,
            "plot_upper": np.max(X_raw, axis=0) + margin,
        },
    )


def peaks(problem: ProblemConfig) -> Dataset:
    """Five-class Peaks data of Section 8.2."""
    if problem.grid_size is None:
        raise ValueError("The Peaks problem requires grid_size in the configuration.")

    rng = np.random.default_rng(problem.seed + 42)
    axis = np.linspace(-3.0, 3.0, int(problem.grid_size))
    X_grid, Y_grid = np.meshgrid(axis, axis)
    Z_grid = peaks_function(X_grid, Y_grid)

    thresholds = np.asarray(
        problem.peaks_thresholds or (-2.2, 0.55, 1.75, 3.2), dtype=float
    )
    if len(thresholds) != problem.num_classes - 1:
        raise ValueError(
            "peaks_thresholds must contain num_classes - 1 entries "
            f"({problem.num_classes - 1}), got {len(thresholds)}."
        )

    classes = np.digitize(Z_grid.ravel(), thresholds)
    points = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

    X_parts, y_parts = [], []
    for cls in range(problem.num_classes):
        candidates = np.flatnonzero(classes == cls)
        count = min(int(problem.samples_per_class), len(candidates))
        selected = rng.choice(candidates, size=count, replace=False)
        X_parts.append(points[selected])
        y_parts.append(np.full(count, cls, dtype=np.int32))

    X_raw = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw,
        y,
        train_size=problem.train_fraction,
        stratify=y,
        random_state=problem.seed + 17,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    return Dataset(
        name="Peaks",
        X_train=jnp.asarray(X_train, dtype=dtype()),
        y_train=jnp.asarray(y_train, dtype=jnp.int32),
        X_val=jnp.asarray(X_val, dtype=dtype()),
        y_val=jnp.asarray(y_val, dtype=jnp.int32),
        num_classes=int(problem.num_classes),
        scaler=scaler,
        extras={
            "X_all_raw": X_raw,
            "y_all": y,
            "X_train_raw": X_train_raw,
            "X_val_raw": X_val_raw,
            "y_train_np": y_train,
            "y_val_np": y_val,
            "X_grid": X_grid,
            "Y_grid": Y_grid,
            "Z_grid": Z_grid,
            "thresholds": thresholds,
            "plot_lower": np.array([-3.0, -3.0]),
            "plot_upper": np.array([3.0, 3.0]),
        },
    )


_LOADERS = {"swiss_roll": swiss_roll, "peaks": peaks}


def load(problem: ProblemConfig) -> Dataset:
    """Dispatch on ``problem.name``."""
    try:
        loader = _LOADERS[problem.name]
    except KeyError as error:
        raise ValueError(
            f"Unknown problem {problem.name!r}. Available: {sorted(_LOADERS)}"
        ) from error
    return loader(problem)
