"""Generate the two Colab notebooks from one shared template.

Keeping the notebooks generated guarantees that both stay in step: they differ
only in the preset name, the headline defaults and a few words of prose.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/REPLACE-ME/layerwise-adaptive-neural-odes"
ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").rstrip().split("\n"),
    }


def _with_newlines(cell: dict) -> dict:
    cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [
        cell["source"][-1]
    ]
    return cell


def build(spec: dict) -> dict:
    cells = [
        markdown(
            f"""
# {spec['title']}

Reference implementation for

> M. Hintermüller, M. Hinze and D. Korolev,
> *Layerwise goal-oriented adaptivity for neural ODEs: an optimal control perspective*,
> Journal of Numerical Mathematics (2026).
> [doi:10.1515/jnma-2026-0008](https://doi.org/10.1515/jnma-2026-0008)

{spec['intro']}

All numerics live in the `lgoa` package; this notebook only chooses parameters,
runs the solver and draws the figures. The two algorithms of the paper are

* **Algorithm 1** &mdash; `lgoa.solver.solve_fixed_depth`: full-batch $H^1$-Adam
  warm start, then BFGS in $H^1$-whitened coordinates, on a fixed temporal grid;
* **Algorithm 2** &mdash; `lgoa.adaptive.run`: solve &rarr; estimate $\\eta_k$
  &rarr; mark &rarr; bisect &rarr; prolongate, until the layer budget is spent.
"""
        ),
        markdown("## 1. Install"),
        code(
            f"""
# On Colab this installs the package from GitHub. A local clone is used as-is.
try:
    import lgoa
except ImportError:
    !pip install -q "git+{REPO_URL}.git"
    import lgoa

import lgoa
from lgoa import plots

print(lgoa.backend_report())
"""
        ),
        markdown(
            """
## 2. Choose the experiment

The bulk of the hyperparameters (network size, Adam/BFGS budgets, early
stopping) lives in the YAML preset printed below &mdash; edit it only if you
want to depart from the published setup. The parameters worth varying from a
notebook are collected here.
"""
        ),
        code(
            f"""
# ----------------------------------------------------------------- main knobs
MARKING                = "{spec['marking']}"  # "conservative" or "dorfler"
DOERFLER_FRACTION      = 0.5           # bulk fraction, used when MARKING == "dorfler"
STATIONARITY_TOLERANCE = {spec['tolerance']}          # eps_stat of Algorithm 1
REGULARIZATION_LAMBDA  = {spec['reg_lambda']}          # lambda of the H^1 regularizer
INITIAL_DEPTH          = {spec['initial_depth']}             # K_0
MAX_DEPTH              = {spec['max_depth']}            # K_max, the outer stopping rule
TRANSFER_ADAM_STATE    = True          # prolongate the Adam moments on refinement

# A reduced profile that finishes in a few minutes on a Colab CPU.
# Set to False for the setup reported in the paper (considerably slower).
QUICK = True

QUICK_PROFILE = dict(
{spec['quick_profile']}
)
# ---------------------------------------------------------------------------

overrides = dict(
    marking=MARKING,
    dorfler_fraction=DOERFLER_FRACTION,
    stationarity_tolerance=STATIONARITY_TOLERANCE,
    reg_lambda=REGULARIZATION_LAMBDA,
    initial_depth=INITIAL_DEPTH,
    max_depth=MAX_DEPTH,
    transfer_adam_state=TRANSFER_ADAM_STATE,
)
if QUICK:
    overrides.update(QUICK_PROFILE)

cfg = lgoa.load_preset("{spec['preset']}", **overrides)
print(cfg.summary())
print("\\npreset file:", lgoa.config_path("{spec['preset']}"))
"""
        ),
        markdown(
            """
## 3. Implementation checks

Six independent checks: the tridiagonal solve against SciPy, the hand-assembled
control and endpoint derivatives against `jax.grad`, invariance of the frozen
endpoint maps, the closed-form indicator pairing against dense quadrature, and
the equality of Dörfler refinement with one marked interval and plain
single-interval insertion.
"""
        ),
        code("lgoa.self_check()"),
        markdown("## 4. Data"),
        code(
            """
data = lgoa.load_dataset(cfg.problem)
print(data.summary())
plots.dataset(data);
"""
        ),
        markdown(
            """
## 5. Run the adaptive hierarchy

The endpoint maps are calibrated once on the initial partition and then frozen,
so every refinement round changes only *where* layers are inserted.
"""
        ),
        code("history = lgoa.run(cfg, data)"),
        code(
            """
print(history.summary())

columns = [
    "phase", "depth", "objective", "data_loss", "validation_loss",
    "validation_accuracy", "stationarity_norm", "stationarity_status",
    "marked_intervals", "marked_indicator_share", "adam_steps", "bfgs_iterations",
]
history.phase_table[columns].round(6)
"""
        ),
        markdown(
            """
## 6. Figures

### 6.1 Quantities along the hierarchy

Stationarity measure against $\\varepsilon_{\\mathrm{stat}}$, the optimized
objective, both losses, and accuracy &mdash; each against the depth $K$.
"""
        ),
        code("plots.learning_curves(history);"),
        markdown(
            """
### 6.2 The adaptive grids

One row per phase, with the nodes marked; the triangles show where the next
layers are about to be inserted.
"""
        ),
        code("plots.grid_history(history);"),
        markdown(
            """
### 6.3 Refinement events

Local indicators $\\eta_k$ before insertion, immediately after insertion, and
after re-optimization. Red bars are the marked parents on the left and their
children in the other two columns. A children-to-parent ratio below one means
re-optimization removed indicator mass from the refined region.
"""
        ),
        code("plots.refinement_events(history, n_events=4);"),
        markdown(
            """
### 6.4 Indicator components

How the adjoint, control and state contributions to $\\eta_K$ evolve with depth,
and the local distribution on the final grid.
"""
        ),
        code("plots.indicator_components(history);"),
        markdown(
            """
### 6.5 State, control and adjoint

The triplet on the final grid. Weak regularization gives visibly more
oscillatory trajectories, i.e. a richer feature transformation.
"""
        ),
        code("plots.trajectories(history, data);"),
        markdown("### 6.6 Prediction of the trained neural ODE"),
        code("plots.decision_boundary(history, data);"),
        markdown(
            """
## 7. Export

Figures as PDF and PNG, tables as CSV. On Colab the files appear in the file
browser on the left; the last line zips them for download.
"""
        ),
        code(
            """
figure_paths = plots.save_all(history, data, "figures")
table_paths = history.to_csv("results")
print(f"{len(figure_paths)} figure files, {len(table_paths)} tables")

# Colab only: uncomment to download everything as a single archive.
# import shutil; from google.colab import files
# shutil.make_archive("lgoa_output", "zip", ".", "figures")
# files.download("lgoa_output.zip")
"""
        ),
        markdown(
            f"""
## 8. Conservative against Dörfler marking

Re-running from the same calibration keeps the comparison paired: identical
data, identical endpoint maps, identical initial control and the same layer
budget &mdash; only the marked set differs. Conservative marking inserts one
layer per round and so re-optimizes after every single insertion; Dörfler
marking inserts a whole bulk at once, which buys fewer rounds of global
re-optimization at the price of less selective placement.

```python
import pandas as pd

calibration = history.calibration          # reuse the frozen endpoint maps
runs = {{"conservative": cfg.with_overrides(marking="conservative")}}
for fraction in (0.25, 0.5, 0.75):
    runs[f"dorfler {{fraction}}"] = cfg.with_overrides(
        marking="dorfler", dorfler_fraction=fraction
    )

results = {{
    name: lgoa.run(config, data, calibration, progress=False)
    for name, config in runs.items()
}}
pd.DataFrame([r.totals() for r in results.values()], index=list(results))
```

{spec['closing']}
"""
        ),
    ]

    return {
        "cells": [_with_newlines(cell) for cell in cells],
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


SPECS = [
    {
        "filename": "01_swiss_roll.ipynb",
        "preset": "swiss_roll",
        "title": "Swiss roll: layerwise goal-oriented adaptivity",
        "intro": (
            "Binary classification of two concentric spirals (Section 8.1). "
            "This is the controlled setting in which to watch the local "
            "indicators respond to layer insertion and re-optimization, and to "
            "see how the regularization parameter $\\lambda$ governs the "
            "stability of the adaptive hierarchy."
        ),
        "marking": "conservative",
        "tolerance": "1e-3",
        "reg_lambda": "1e-3",
        "initial_depth": 2,
        "max_depth": 12,
        "quick_profile": (
            "    samples_per_class=400,     # paper: 1200\n"
            "    n_candidates=3,            # paper: 10 initializations\n"
            "    search_max_epochs=200,     # paper: 1000\n"
            "    search_min_epochs=60,\n"
            "    adam_max_steps=4000,       # paper: 20000\n"
            "    bfgs_max_iterations=500,   # paper: 10000\n"
            "    indicator_sup_samples=20,  # paper: 50\n"
            "    max_depth=9,               # paper: 12"
        ),
        "closing": (
            "Table 1 of the paper reports that $\\lambda = 10^{-3}$ is the most "
            "robust compromise: stronger regularization stabilizes the "
            "hierarchy but limits expressivity, while $\\lambda \\le 10^{-5}$ "
            "makes stationarity progressively harder to reach. Re-run section 2 "
            "with `REGULARIZATION_LAMBDA = 1e-2` or `1e-6` to see both failure "
            "modes."
        ),
    },
    {
        "filename": "02_peaks.ipynb",
        "preset": "peaks",
        "title": "Peaks: layerwise goal-oriented adaptivity",
        "intro": (
            "Five-class classification of the Peaks function (Section 8.2). "
            "The harder multiclass problem on which the paper isolates the role "
            "of the marking rule, comparing conservative and Dörfler layer "
            "insertion under a common depth budget."
        ),
        "marking": "conservative",
        "tolerance": "5e-3",
        "reg_lambda": "1e-3",
        "initial_depth": 2,
        "max_depth": 15,
        "quick_profile": (
            "    samples_per_class=300,     # paper: 1500\n"
            "    grid_size=128,             # paper: 256\n"
            "    search_max_epochs=200,     # paper: 1000\n"
            "    search_min_epochs=60,\n"
            "    adam_max_steps=4000,       # paper: 20000\n"
            "    bfgs_max_iterations=500,   # paper: 5000\n"
            "    indicator_sup_samples=20,  # paper: 50\n"
            "    max_depth=10,              # paper: 15"
        ),
        "closing": (
            "The published ablation of Table 2 averages ten model seeds; a "
            "single seed shows the mechanism but not the statistics, so treat "
            "the ordering of two runs here as indicative rather than as a "
            "reproduction of the table. Change `initialization.model_seed` in "
            "the preset to move to another seed."
        ),
    },
]


def main() -> None:
    target = ROOT / "notebooks"
    target.mkdir(exist_ok=True)
    for spec in SPECS:
        path = target / spec["filename"]
        path.write_text(json.dumps(build(spec), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
