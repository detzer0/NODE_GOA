r"""Layer insertion: bisect the marked intervals and transfer the control.

Each marked interval :math:`I_k` is bisected at its midpoint
:math:`t_{k-1/2}`.  The optimized control is carried to the refined grid by the
midpoint prolongation :math:`\theta_{\tau^+} = \Pi_{\tau \to \tau^+}
\theta^\star_\tau`, which keeps every existing nodal value and assigns

.. math::

    \bigl(\Pi_{\tau \to \tau^+}\theta^\star_\tau\bigr)(t_{k-1/2})
    = \tfrac{1}{2}\bigl(\theta^\star_\tau(t_{k-1}) + \theta^\star_\tau(t_k)\bigr).

The same interpolation prolongs the first and second Adam moments, so the
refined problem can resume from a warm optimizer state instead of restarting it.

Existing nodes are *copied*, never recomputed, which makes the one-mark case
bit-for-bit identical to single-interval insertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

__all__ = ["Refinement", "refine", "insert_single_interval", "uniform_subdivision"]


@dataclass(frozen=True)
class Refinement:
    """Refined grid, prolongated control and the parent/child bookkeeping."""

    params: dict
    time_grid: jnp.ndarray
    optimizer_state: dict | None
    inserted_times: tuple[float, ...]
    inserted_node_indices: tuple[int, ...]
    parent_to_children: tuple[dict[str, Any], ...] = field(default_factory=tuple)

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


def _insert_nodal_midpoints(values, marked_intervals):
    """Refine a nodal CG(1) array, copying all old nodes verbatim."""
    marked = {int(index) for index in marked_intervals}
    pieces = []
    for k in range(int(values.shape[0]) - 1):
        pieces.append(values[k : k + 1])
        if k in marked:
            pieces.append(0.5 * (values[k : k + 1] + values[k + 1 : k + 2]))
    pieces.append(values[-1:])
    return jnp.concatenate(pieces, axis=0)


def _insert_grid_midpoints(time_grid, marked_intervals):
    marked = {int(index) for index in marked_intervals}
    pieces = []
    inserted_times = []
    for k in range(int(time_grid.shape[0]) - 1):
        pieces.append(time_grid[k : k + 1])
        if k in marked:
            midpoint = 0.5 * (time_grid[k : k + 1] + time_grid[k + 1 : k + 2])
            pieces.append(midpoint)
            inserted_times.append(float(np.asarray(midpoint[0])))
    pieces.append(time_grid[-1:])
    return jnp.concatenate(pieces, axis=0), inserted_times


def refine(
    params,
    time_grid,
    optimizer_state,
    marked_intervals,
    *,
    transfer_adam_state: bool,
) -> Refinement:
    """Bisect every marked interval simultaneously by exact CG(1) transfer."""
    marked = sorted({int(index) for index in marked_intervals})
    number_of_intervals = int(time_grid.shape[0]) - 1

    if not marked:
        raise ValueError("At least one interval must be marked.")
    if marked[0] < 0 or marked[-1] >= number_of_intervals:
        raise IndexError("A marked interval lies outside the temporal grid.")

    new_grid, inserted_times = _insert_grid_midpoints(time_grid, marked)

    refined_params = dict(params)
    refined_params["A"] = _insert_nodal_midpoints(params["A"], marked)
    refined_params["b"] = _insert_nodal_midpoints(params["b"], marked)

    if transfer_adam_state and optimizer_state is not None:
        refined_state = {
            "count": optimizer_state["count"],
            "m": _insert_nodal_midpoints(optimizer_state["m"], marked),
            "v": _insert_nodal_midpoints(optimizer_state["v"], marked),
        }
    else:
        refined_state = None

    parent_to_children = []
    inserted_node_indices = []
    for position, parent_index in enumerate(marked):
        # Exactly `position` inserted nodes precede this parent interval.
        left_child = int(parent_index + position)
        right_child = left_child + 1
        parent_to_children.append(
            {
                "parent_interval": int(parent_index),
                "child_intervals": (left_child, right_child),
                "inserted_node_index": right_child,
                "inserted_time": float(inserted_times[position]),
            }
        )
        inserted_node_indices.append(right_child)

    return Refinement(
        params=refined_params,
        time_grid=new_grid,
        optimizer_state=refined_state,
        inserted_times=tuple(inserted_times),
        inserted_node_indices=tuple(inserted_node_indices),
        parent_to_children=tuple(parent_to_children),
    )


def insert_single_interval(params, time_grid, interval_index, optimizer_state=None):
    """Reference single-interval insertion, kept for the equivalence check."""
    k = int(interval_index)
    new_time = 0.5 * (time_grid[k] + time_grid[k + 1])
    refined_grid = jnp.concatenate(
        [
            time_grid[: k + 1],
            jnp.asarray([new_time], dtype=time_grid.dtype),
            time_grid[k + 1 :],
        ]
    )

    def insert(values):
        midpoint = 0.5 * (values[k : k + 1] + values[k + 1 : k + 2])
        return jnp.concatenate([values[: k + 1], midpoint, values[k + 1 :]], axis=0)

    refined_params = dict(params)
    refined_params["A"] = insert(params["A"])
    refined_params["b"] = insert(params["b"])

    refined_state = None
    if optimizer_state is not None:
        refined_state = {
            "count": optimizer_state["count"],
            "m": insert(optimizer_state["m"]),
            "v": insert(optimizer_state["v"]),
        }
    return refined_params, refined_grid, refined_state


def uniform_subdivision(time_grid, subdivisions, dtype=None):
    """Split every interval into ``subdivisions`` equal parts."""
    grid = np.asarray(time_grid, dtype=float)
    pieces = []
    for left, right in zip(grid[:-1], grid[1:], strict=True):
        pieces.extend(np.linspace(left, right, int(subdivisions) + 1)[:-1])
    pieces.append(grid[-1])
    return jnp.asarray(np.asarray(pieces), dtype=dtype or time_grid.dtype)
