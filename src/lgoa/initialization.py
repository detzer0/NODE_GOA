r"""One-time endpoint-map calibration.

Before any refinement starts, a shallow network on the initial partition
:math:`\mathcal{K}_0` is trained with the endpoint maps
:math:`W_{\mathrm{in}}`, :math:`W_{\mathrm{out}}` *active*, using early
stopping on the validation loss.  The resulting endpoint maps are then frozen
and shared by every refinement path, so a comparison of marking rules differs
only in **where** layers are inserted, never in the initialization.

With ``n_candidates > 1`` several initialization seeds are tried and the one
with the lowest validation loss is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import fem as fem_module
from . import neural_ode as node
from . import optimizers
from .backend import dtype
from .config import InitializationConfig, ProblemConfig, SolverConfig
from .datasets import Dataset
from .trees import copy_tree, scale_tree

__all__ = ["Calibration", "calibrate_endpoint_maps"]


@dataclass(frozen=True)
class Calibration:
    """Frozen endpoint maps and the initial control they came with."""

    params: dict
    fixed_W_in: jnp.ndarray
    fixed_W_out: jnp.ndarray
    seed: int
    best_epoch: int
    epochs_run: int
    validation_loss: float
    validation_accuracy: float
    history: pd.DataFrame
    candidates: pd.DataFrame

    def summary(self) -> str:
        return (
            f"endpoint calibration | seed={self.seed} | "
            f"best epoch={self.best_epoch} | "
            f"validation loss={self.validation_loss:.6f} | "
            f"validation accuracy={self.validation_accuracy:.4f}"
        )


def _make_training_functions(time_grid, data, problem, reg_lambda, solver):
    """Jitted training step with the endpoint maps active."""
    operators = fem_module.assemble(time_grid)

    @jax.jit
    def train_step(params, opt_state, X, y, learning_rate):
        metrics, grads = node.objective_and_gradients(
            params, X, y, time_grid, operators, reg_lambda, True
        )
        clip_scale = jnp.minimum(
            1.0, solver.grad_clip / (metrics["stationarity_norm"] + 1.0e-12)
        )
        params, opt_state = optimizers.adam_update(
            params,
            scale_tree(grads, clip_scale),
            opt_state,
            learning_rate,
            beta1=solver.adam_beta1,
            beta2=solver.adam_beta2,
            eps=solver.adam_eps,
        )
        return params, opt_state, metrics

    @jax.jit
    def evaluate(params, X, y):
        metrics, _ = node.objective_and_gradients(
            params, X, y, time_grid, operators, reg_lambda, True
        )
        return metrics

    accuracy = jax.jit(
        lambda params, X, y: node.multiclass_accuracy(params, X, y, time_grid)
    )
    loss = jax.jit(lambda params, X, y: node.data_loss_only(params, X, y, time_grid))
    return train_step, evaluate, accuracy, loss


def _train_one_candidate(
    seed,
    time_grid,
    functions,
    data: Dataset,
    problem: ProblemConfig,
    initialization: InitializationConfig,
    verbose: bool,
):
    train_step, evaluate, accuracy, loss_fn = functions
    params = node.initialize_params(
        jax.random.PRNGKey(int(seed)),
        len(time_grid),
        problem.hidden_dim,
        problem.num_classes,
        initialization.control_init_noise,
    )
    opt_state = optimizers.adam_init(params)
    best_params = copy_tree(params)
    best_val_loss = np.inf
    best_epoch = 0
    checks_without_improvement = 0
    history = []
    epoch = 0

    for epoch in range(1, int(initialization.search_max_epochs) + 1):
        params, opt_state, _ = train_step(
            params,
            opt_state,
            data.X_train,
            data.y_train,
            jnp.asarray(initialization.learning_rate, dtype=dtype()),
        )
        if not (epoch == 1 or epoch % int(initialization.check_every) == 0):
            continue

        raw = evaluate(params, data.X_train, data.y_train)
        objective = float(raw["objective"])
        stationarity = float(raw["stationarity_norm"])
        val_loss = float(loss_fn(params, data.X_val, data.y_val))
        history.append(
            {
                "epoch": epoch,
                "train_objective": objective,
                "train_stationarity": stationarity,
                "validation_loss": val_loss,
            }
        )

        if val_loss < best_val_loss - initialization.validation_min_delta:
            best_val_loss = val_loss
            best_params = copy_tree(params)
            best_epoch = epoch
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        if verbose:
            print(
                f"  seed={seed:5d} epoch={epoch:4d} J={objective:.5f} "
                f"val={val_loss:.5f} stat={stationarity:.2e}"
            )

        if (
            epoch >= int(initialization.search_min_epochs)
            and checks_without_improvement >= int(initialization.validation_patience)
        ):
            break

    return {
        "seed": int(seed),
        "params": best_params,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "validation_loss": float(loss_fn(best_params, data.X_val, data.y_val)),
        "validation_accuracy": float(accuracy(best_params, data.X_val, data.y_val)),
        "history": pd.DataFrame(history),
    }


def calibrate_endpoint_maps(
    time_grid,
    data: Dataset,
    problem: ProblemConfig,
    solver: SolverConfig,
    initialization: InitializationConfig,
    *,
    reg_lambda: float | None = None,
    verbose: bool = False,
) -> Calibration:
    """Train on the initial partition and freeze the endpoint maps."""
    reg_lambda = float(
        solver.reg_lambda if reg_lambda is None else reg_lambda
    )
    functions = _make_training_functions(
        time_grid, data, problem, reg_lambda, solver
    )

    candidates = [
        _train_one_candidate(
            seed, time_grid, functions, data, problem, initialization, verbose
        )
        for seed in initialization.candidate_seeds
    ]
    table = pd.DataFrame(
        [
            {
                "candidate": index,
                "seed": candidate["seed"],
                "best_epoch": candidate["best_epoch"],
                "epochs_run": candidate["epochs_run"],
                "validation_loss": candidate["validation_loss"],
                "validation_accuracy": candidate["validation_accuracy"],
            }
            for index, candidate in enumerate(candidates)
        ]
    ).sort_values("validation_loss")

    best = candidates[int(table.index[0])]
    params = copy_tree(best["params"])
    return Calibration(
        params=params,
        fixed_W_in=jnp.array(params["W_in"], copy=True),
        fixed_W_out=jnp.array(params["W_out"], copy=True),
        seed=best["seed"],
        best_epoch=best["best_epoch"],
        epochs_run=best["epochs_run"],
        validation_loss=best["validation_loss"],
        validation_accuracy=best["validation_accuracy"],
        history=best["history"],
        candidates=table.reset_index(drop=True),
    )
