r"""Figures for a single adaptive run.

Every function takes a :class:`lgoa.history.History` (plus the data set where
the figure needs it), returns a :class:`matplotlib.figure.Figure`, and draws
nothing else -- so the same call works in a notebook, in a script and in a
test.  The figures correspond to those of Section 8 of the paper that a single
seed can reproduce:

====================================  =========================================
:func:`dataset`                       Fig. 2(A), 2(C) -- the data
:func:`decision_boundary`             Fig. 2(B), 2(D) -- the trained prediction
:func:`learning_curves`               Fig. 3 -- quantities along the hierarchy
:func:`trajectories`                  Fig. 4 -- state/control/adjoint triplet
:func:`refinement_events`             Figs. 5, 7, 9 -- marking mechanics
:func:`indicator_components`          Fig. 6(A) -- indicator decomposition
:func:`grid_history`                  Figs. 8, 10, 11 -- the adaptive grids
====================================  =========================================

The palette is Okabe--Ito, which stays distinguishable under the common forms
of colour-vision deficiency; line styles and markers carry the same
information as a second cue.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator

from .backend import dtype
from .datasets import Dataset
from .history import History, Phase
from .marking import describe as describe_marking
from .neural_ode import forward_states, pack_controls

__all__ = [
    "PALETTE",
    "use_style",
    "dataset",
    "decision_boundary",
    "learning_curves",
    "trajectories",
    "refinement_events",
    "indicator_components",
    "grid_history",
    "save_all",
]

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
    "light_grey": "#D9D9D9",
}

_CLASS_COLORS = [
    PALETTE["blue"],
    PALETTE["vermillion"],
    PALETTE["green"],
    PALETTE["orange"],
    PALETTE["purple"],
]
_CLASS_MARKERS = ["o", "^", "s", "D", "v"]
_MARKED_COLOR = PALETTE["vermillion"]


def use_style() -> None:
    """Apply the publication-ish Matplotlib defaults used by these figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "axes.grid": False,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
        }
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _finish(ax, *, xlabel=None, ylabel=None, title=None, legend=False, logy=False):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", linestyle=":", color=PALETTE["grey"], alpha=0.35)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return ax


_DISPLAY_NAMES = {"swiss_roll": "Swiss roll", "peaks": "Peaks"}


def _run_label(history: History) -> str:
    adaptive = history.config.adaptive
    rule = describe_marking(
        adaptive.marking,
        adaptive.dorfler_fraction if adaptive.marking == "dorfler" else None,
    )
    name = _DISPLAY_NAMES.get(
        history.config.problem.name, history.config.problem.name
    )
    return f"{name}: {rule}, " rf"$\lambda = ${history.config.solver.reg_lambda:g}"


def _class_colormap(num_classes: int) -> ListedColormap:
    return ListedColormap(_CLASS_COLORS[:num_classes])


def _predict_classes(phase: Phase, data: Dataset, X_raw: np.ndarray) -> np.ndarray:
    """Predicted labels for raw (unscaled) inputs, using one phase's solution."""
    X_scaled = jnp.asarray(data.scaler.transform(X_raw), dtype=dtype())
    params = phase.params
    states = forward_states(params, X_scaled, jnp.asarray(phase.result.time_grid))
    logits = states[-1] @ params["W_out"].T
    return np.asarray(jnp.argmax(logits, axis=1))


def _positive_limits(values_list) -> tuple[float, float]:
    blocks = []
    for values in values_list:
        values = np.asarray(values, dtype=float)
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size:
            blocks.append(positive)
    if not blocks:
        return 1.0e-15, 1.0e-14
    values = np.concatenate(blocks)
    y_min = max(float(np.min(values)) / 2.0, np.finfo(float).tiny)
    y_max = max(1.20 * float(np.max(values)), 10.0 * y_min)
    return y_min, y_max


def _representative_indices(count: int, requested: int) -> list[int]:
    if count == 0:
        return []
    requested = min(int(requested), count)
    return np.unique(
        np.rint(np.linspace(0, count - 1, requested)).astype(int)
    ).tolist()


# --------------------------------------------------------------------------- #
# Fig. 2 -- data and prediction
# --------------------------------------------------------------------------- #
def dataset(data: Dataset, *, max_points: int = 4000) -> Figure:
    """Scatter the raw training data, coloured by class."""
    use_style()
    X = np.asarray(data.extras["X_all_raw"])
    y = np.asarray(data.extras["y_all"])
    if X.shape[0] > max_points:
        stride = int(np.ceil(X.shape[0] / max_points))
        X, y = X[::stride], y[::stride]

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for cls in range(data.num_classes):
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=8,
            color=_CLASS_COLORS[cls % len(_CLASS_COLORS)],
            marker=_CLASS_MARKERS[cls % len(_CLASS_MARKERS)],
            linewidths=0,
            alpha=0.85,
            label=f"class {cls}",
        )
    ax.set_aspect("equal", adjustable="box")
    _finish(
        ax,
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=f"{data.name} data ({data.n_train} train / {data.n_val} validation)",
        legend=data.num_classes <= 3,
    )
    fig.tight_layout()
    return fig


def decision_boundary(
    history: History,
    data: Dataset,
    *,
    phase: Phase | None = None,
    resolution: int = 320,
    max_points: int = 3000,
) -> Figure:
    """Predicted class regions of one phase, with the data overlaid."""
    use_style()
    phase = phase or history.final_phase
    lower = np.asarray(data.extras["plot_lower"], dtype=float)
    upper = np.asarray(data.extras["plot_upper"], dtype=float)

    xs = np.linspace(lower[0], upper[0], int(resolution))
    ys = np.linspace(lower[1], upper[1], int(resolution))
    XX, YY = np.meshgrid(xs, ys)
    grid_points = np.stack([XX.ravel(), YY.ravel()], axis=1)
    predicted = _predict_classes(phase, data, grid_points).reshape(XX.shape)

    X = np.asarray(data.extras["X_all_raw"])
    y = np.asarray(data.extras["y_all"])
    if X.shape[0] > max_points:
        stride = int(np.ceil(X.shape[0] / max_points))
        X, y = X[::stride], y[::stride]

    fig, ax = plt.subplots(figsize=(4.9, 4.5))
    ax.pcolormesh(
        XX,
        YY,
        predicted,
        cmap=_class_colormap(data.num_classes),
        vmin=-0.5,
        vmax=data.num_classes - 0.5,
        alpha=0.30,
        shading="auto",
    )
    for cls in range(data.num_classes):
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=7,
            color=_CLASS_COLORS[cls % len(_CLASS_COLORS)],
            marker=_CLASS_MARKERS[cls % len(_CLASS_MARKERS)],
            linewidths=0,
        )
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    _finish(
        ax,
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=(
            f"prediction at $K = {phase.depth}$ "
            f"(validation accuracy {phase.result.validation_accuracy:.3f})"
        ),
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig. 3 -- learning curves along the hierarchy
# --------------------------------------------------------------------------- #
def learning_curves(history: History) -> Figure:
    r"""Stationarity measure, objective, losses and accuracy versus depth."""
    use_style()
    table = history.phase_table
    depth = table["depth"].to_numpy()
    tolerance = float(history.config.solver.stationarity_tolerance)
    strict = table["converged"].to_numpy(dtype=bool)

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.5))

    ax = axes[0]
    ax.plot(depth, table["stationarity_norm"], color=PALETTE["blue"], marker="o")
    if np.any(~strict):
        ax.scatter(
            depth[~strict],
            table["stationarity_norm"].to_numpy()[~strict],
            facecolors="white",
            edgecolors=_MARKED_COLOR,
            zorder=5,
            s=46,
            linewidths=1.4,
            label="tolerance missed",
        )
    ax.axhline(
        tolerance,
        color=PALETTE["grey"],
        linestyle="--",
        label=rf"$\varepsilon_{{\mathrm{{stat}}}} = {tolerance:g}$",
    )
    _finish(
        ax,
        xlabel=r"depth $K$",
        ylabel=r"$s_\tau(\Theta_\tau) = \|g_\tau\|_{H^1}$",
        title="(A) stationarity measure",
        legend=True,
        logy=True,
    )

    ax = axes[1]
    ax.plot(
        depth,
        table["objective"],
        color=PALETTE["black"],
        marker="o",
        label=r"$J_\tau$",
    )
    ax.plot(
        depth,
        table["data_loss"],
        color=PALETTE["sky"],
        marker="s",
        linestyle="--",
        label="training loss",
    )
    _finish(
        ax,
        xlabel=r"depth $K$",
        ylabel="objective",
        title="(B) optimized objective",
        legend=True,
    )

    ax = axes[2]
    ax.plot(
        depth,
        table["data_loss"],
        color=PALETTE["sky"],
        marker="s",
        linestyle="--",
        label="training",
    )
    ax.plot(
        depth,
        table["validation_loss"],
        color=PALETTE["vermillion"],
        marker="o",
        label="validation",
    )
    _finish(
        ax,
        xlabel=r"depth $K$",
        ylabel="cross-entropy",
        title="(C) losses",
        legend=True,
    )

    ax = axes[3]
    ax.plot(
        depth,
        100.0 * table["train_accuracy"],
        color=PALETTE["sky"],
        marker="s",
        linestyle="--",
        label="training",
    )
    ax.plot(
        depth,
        100.0 * table["validation_accuracy"],
        color=PALETTE["green"],
        marker="o",
        label="validation",
    )
    _finish(
        ax,
        xlabel=r"depth $K$",
        ylabel="accuracy (%)",
        title="(D) accuracy",
        legend=True,
    )

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(_run_label(history), y=1.03)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig. 4 -- state / control / adjoint triplet
# --------------------------------------------------------------------------- #
def trajectories(
    history: History,
    data: Dataset,
    *,
    phase: Phase | None = None,
    samples_per_class: int = 3,
    components: tuple[int, ...] = (0, 1),
    control_components: int = 4,
) -> Figure:
    r"""Representative state, control and adjoint trajectories in time."""
    use_style()
    phase = phase or history.final_phase
    grid = phase.grid
    params = phase.params

    y_train = np.asarray(data.y_train)
    chosen = []
    for cls in range(data.num_classes):
        candidates = np.flatnonzero(y_train == cls)[: int(samples_per_class)]
        chosen.extend(candidates.tolist())
    chosen = np.asarray(chosen, dtype=int)

    X_subset = jnp.asarray(np.asarray(data.X_train)[chosen], dtype=dtype())
    y_subset = jnp.asarray(y_train[chosen], dtype=jnp.int32)
    _, states, adjoints = phase.result.control.solution_fn(params, X_subset, y_subset)
    states = np.asarray(states)
    adjoints = np.asarray(adjoints)
    theta = np.asarray(pack_controls(params))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.6))

    ax = axes[0]
    for position, index in enumerate(chosen):
        color = _CLASS_COLORS[int(y_train[index]) % len(_CLASS_COLORS)]
        for component in components:
            ax.plot(
                grid,
                states[:, position, component],
                color=color,
                alpha=0.8,
                linewidth=1.2,
            )
    for cls in range(data.num_classes):
        ax.plot(
            [],
            [],
            color=_CLASS_COLORS[cls % len(_CLASS_COLORS)],
            label=f"class {cls}",
        )
    _finish(
        ax,
        xlabel=r"$t$",
        ylabel=r"$x^i_\tau(t)$",
        title=f"(A) state components {list(components)}",
        legend=data.num_classes <= 3,
    )

    ax = axes[1]
    n_show = min(int(control_components), theta.shape[1])
    for component in range(n_show):
        ax.plot(
            grid,
            theta[:, component],
            marker="o",
            markersize=3.0,
            linewidth=1.3,
            color=_CLASS_COLORS[component % len(_CLASS_COLORS)],
        )
    for node_time in grid:
        ax.axvline(node_time, color=PALETTE["light_grey"], linewidth=0.7, zorder=0)
    _finish(
        ax,
        xlabel=r"$t$",
        ylabel=r"$\theta_\tau(t)$",
        title=f"(B) control, first {n_show} components (nodes marked)",
    )

    ax = axes[2]
    for position, index in enumerate(chosen):
        color = _CLASS_COLORS[int(y_train[index]) % len(_CLASS_COLORS)]
        for component in components:
            ax.plot(
                grid,
                adjoints[:, position, component],
                color=color,
                alpha=0.8,
                linewidth=1.2,
            )
    _finish(
        ax,
        xlabel=r"$t$",
        ylabel=r"$p^i_\tau(t)$",
        title=f"(C) adjoint components {list(components)}",
    )

    fig.suptitle(f"{_run_label(history)}; $K = {phase.depth}$", y=1.04)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figs. 5, 7, 9 -- refinement events
# --------------------------------------------------------------------------- #
def _indicator_histogram(ax, grid, values, highlighted, y_min, y_max):
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    widths = np.diff(grid)
    highlighted = {int(index) for index in highlighted}

    colors = [
        _MARKED_COLOR if index in highlighted else "0.80"
        for index in range(len(values))
    ]
    displayed = np.where(np.isfinite(values) & (values > 0.0), values, y_min)
    displayed = np.maximum(displayed, y_min)

    ax.bar(
        grid[:-1],
        displayed - y_min,
        bottom=y_min,
        width=widths,
        align="edge",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlim(grid[0], grid[-1])
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", which="both", linestyle=":", color=PALETTE["grey"], alpha=0.3)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.get_offset_text().set_visible(False)
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def refinement_events(history: History, *, n_events: int = 4) -> Figure:
    r"""Local indicators before insertion, immediately after, and re-optimized.

    Red bars are the marked parents in the left column and their children in
    the other two.  The annotation reports the depth change, the number of
    inserted layers, the achieved bulk fraction :math:`\beta^\vartheta_K`, and
    the children-to-parent indicator ratio after re-optimization.
    """
    use_style()
    completed = history.completed_events
    indices = _representative_indices(len(completed), n_events)
    if not indices:
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.text(
            0.5,
            0.5,
            "no completed refinement events in this run",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return fig

    selected = [completed[index] for index in indices]
    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(11.2, 2.4 * n_rows), squeeze=False
    )
    column_titles = ("before insertion", "after insertion", "after re-optimization")

    for row, event in enumerate(selected):
        before = event.before
        immediate = event.after_immediate
        after = event.after_reoptimized

        panels = (
            (before.grid, before.eta, event.marked_intervals),
            (
                np.asarray(immediate.grid, dtype=float),
                np.asarray(immediate.total, dtype=float),
                event.child_intervals,
            ),
            (after.grid, after.eta, event.child_intervals),
        )
        y_min, y_max = _positive_limits([values for _, values, _ in panels])

        for column, (grid, values, highlighted) in enumerate(panels):
            ax = axes[row, column]
            _indicator_histogram(ax, grid, values, highlighted, y_min, y_max)
            if row == 0:
                ax.set_title(column_titles[column], pad=6)
            if row == n_rows - 1:
                ax.set_xlabel(r"$t$")
            else:
                ax.tick_params(axis="x", labelbottom=False)

        axes[row, 0].set_ylabel(f"event {event.index}\n" + r"$\eta_k$")
        bulk = event.marking.achieved_share
        axes[row, 0].text(
            0.03,
            0.94,
            rf"$K: {event.depth_before} \to {event.depth_after}$"
            + "\n"
            + rf"marked = {event.inserted_layers}, bulk = {bulk:.0%}",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
        axes[row, 2].text(
            0.97,
            0.94,
            f"children/parents = {event.parent_child_ratio():.2f}",
            transform=axes[row, 2].transAxes,
            ha="right",
            va="top",
            fontsize=9.0,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    fig.suptitle(_run_label(history), y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig. 6(A) -- indicator decomposition
# --------------------------------------------------------------------------- #
def indicator_components(history: History) -> Figure:
    r"""Global indicator components and the final local distribution."""
    use_style()
    table = history.phase_table
    depth = table["depth"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.5))

    ax = axes[0]
    series = (
        ("global_adjoint_indicator", "adjoint term", PALETTE["blue"], "o", "-"),
        ("global_control_indicator", "control term", PALETTE["orange"], "s", "--"),
        ("global_state_indicator", "state term", PALETTE["green"], "^", "-."),
        ("global_practical_indicator", r"total $\eta_K$", PALETTE["black"], "D", "-"),
    )
    for column, label, color, marker, linestyle in series:
        ax.plot(
            depth,
            table[column],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    _finish(
        ax,
        xlabel=r"depth $K$",
        ylabel="indicator",
        title="(A) global indicator components",
        legend=True,
        logy=True,
    )

    ax = axes[1]
    share = 100.0 * table["control_fraction"].to_numpy(dtype=float)
    ax.plot(depth, share, color=PALETTE["orange"], marker="s")
    # Keep the axis honest but readable when the control share is tiny.
    ax.set_ylim(0.0, min(100.0, max(1.0, 1.25 * float(np.max(share)))))
    _finish(
        ax,
        xlabel=r"depth $K$",
        ylabel="share of $\\eta_K$ (%)",
        title="(B) control contribution",
    )

    ax = axes[2]
    final = history.final_phase
    y_min, y_max = _positive_limits([final.eta])
    _indicator_histogram(
        ax, final.grid, final.eta, (), y_min, y_max
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\eta_k$")
    ax.set_title(rf"(C) local indicators at $K = {final.depth}$")

    for ax in axes[:2]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(_run_label(history), y=1.03)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figs. 8, 10, 11 -- the adaptive grids
# --------------------------------------------------------------------------- #
def grid_history(history: History, *, show_all_phases: bool = True) -> Figure:
    """Every temporal grid of the hierarchy, newest at the bottom."""
    use_style()
    phases = history.phases if show_all_phases else [history.final_phase]
    final_time = float(history.config.problem.final_time)

    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(phases) + 1.1))
    for row, phase in enumerate(phases):
        level = -row
        grid = phase.grid
        ax.hlines(level, 0.0, final_time, color=PALETTE["light_grey"], linewidth=1.0)
        ax.plot(
            grid,
            np.full_like(grid, level),
            linestyle="none",
            marker="|",
            markersize=11,
            markeredgewidth=1.6,
            color=PALETTE["blue"],
        )
        if phase.marking is not None:
            inserted = [
                mapping["inserted_time"]
                for event in history.events
                if event.before is phase
                for mapping in event.parent_to_children
            ]
            if inserted:
                ax.plot(
                    inserted,
                    np.full(len(inserted), level),
                    linestyle="none",
                    marker="v",
                    markersize=5.0,
                    color=_MARKED_COLOR,
                )
        ax.text(
            -0.012 * final_time,
            level,
            f"$K={phase.depth}$",
            ha="right",
            va="center",
            fontsize=9.0,
        )

    ax.plot([], [], marker="|", linestyle="none", color=PALETTE["blue"], label="nodes")
    ax.plot(
        [],
        [],
        marker="v",
        linestyle="none",
        color=_MARKED_COLOR,
        label="next insertion",
    )
    ax.set_xlim(-0.08 * final_time, final_time * 1.02)
    ax.set_ylim(-len(phases) + 0.4, 0.6)
    ax.set_yticks([])
    ax.set_xlabel(r"$t$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2)
    ax.grid(True, axis="x", linestyle=":", color=PALETTE["grey"], alpha=0.35)
    ax.set_title(_run_label(history))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def save_all(
    history: History,
    data: Dataset,
    directory: str | Path,
    *,
    prefix: str | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    close: bool = True,
) -> list[Path]:
    """Render every figure of this run and write it to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"{history.config.problem.name}_{history.config.adaptive.marking}"

    figures = {
        "data": lambda: dataset(data),
        "prediction": lambda: decision_boundary(history, data),
        "learning_curves": lambda: learning_curves(history),
        "trajectories": lambda: trajectories(history, data),
        "refinement_events": lambda: refinement_events(history),
        "indicator_components": lambda: indicator_components(history),
        "grid_history": lambda: grid_history(history),
    }

    written: list[Path] = []
    for name, builder in figures.items():
        figure = builder()
        for extension in formats:
            path = directory / f"{stem}_{name}.{extension}"
            figure.savefig(path)
            written.append(path)
        if close:
            plt.close(figure)
    return written
