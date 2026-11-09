r"""Adam in the discrete :math:`H^1` metric.

Two flavours are needed.  The pytree version drives the endpoint-map
calibration, where :math:`W_{\mathrm{in}}` and :math:`W_{\mathrm{out}}` are
still trained.  The matrix version lives inside the ``jax.jit``-compiled
fixed-depth control solver, where the iterate is the coefficient matrix
:math:`\Theta_\tau` alone.  Both consume the metric gradient
:math:`G_\tau`, not the Euclidean one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .trees import zeros_like

__all__ = [
    "adam_init",
    "adam_update",
    "control_state_init",
    "control_state_copy",
    "control_state_like",
]


def adam_init(params):
    zeros = zeros_like(params)
    return {"count": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def adam_update(
    params, grads, state, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8
):
    count = state["count"] + 1
    m = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * grad * grad, state["v"], grads
    )
    bias1 = 1.0 - beta1**count
    bias2 = 1.0 - beta2**count
    new_params = jax.tree_util.tree_map(
        lambda p, mm, vv: p
        - learning_rate * (mm / bias1) / (jnp.sqrt(vv / bias2) + eps),
        params,
        m,
        v,
    )
    return new_params, {"count": count, "m": m, "v": v}


def control_state_init(theta):
    """Fresh Adam moments for a control coefficient matrix."""
    return {
        "count": jnp.asarray(0, dtype=jnp.int32),
        "m": jnp.zeros_like(theta),
        "v": jnp.zeros_like(theta),
    }


def control_state_copy(state):
    if state is None:
        return None
    return {
        "count": jnp.array(state["count"], copy=True),
        "m": jnp.array(state["m"], copy=True),
        "v": jnp.array(state["v"], copy=True),
    }


def control_state_like(state, m, v):
    """Rebuild a control Adam state with new moments and the same step count."""
    return {"count": state["count"], "m": m, "v": v}
