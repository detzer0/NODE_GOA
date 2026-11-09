r"""Discretized neural ODE, discrete adjoint and the exact control derivative.

The state is advanced by the explicit Euler scheme with midpoint controls,

.. math::

    \theta_\tau^{k-1/2} = \tfrac{1}{2}(\theta_\tau^{k-1} + \theta_\tau^{k}),
    \qquad
    x_\tau^{k} = x_\tau^{k-1} + h_k F(x_\tau^{k-1}, \theta_\tau^{k-1/2}),

with :math:`F(x, (A, b)) = \tanh(xA + b)`.  The adjoint runs backwards from
:math:`p_\tau^K = l'(x_\tau^K)`,

.. math::

    p_\tau^{k-1} = p_\tau^{k}
    + h_k D_1 F(x_\tau^{k-1}, \theta_\tau^{k-1/2})^{*} p_\tau^{k},

and the nodal load :math:`Q_\tau` is assembled from the control pullback.  The
metric gradient then solves :math:`B^\tau G_\tau = \lambda B^\tau \Theta_\tau +
C^\tau Q_\tau`, and the stationarity measure is

.. math::

    s_\tau(\Theta_\tau) = \lVert g_\tau \rVert_{H^1}
    = \bigl[\operatorname{trace}(G_\tau^\top B^\tau G_\tau)\bigr]^{1/2}.

``objective_and_gradients`` returns that gradient together with every quantity
the solver and the indicator need, and is validated against ``jax.grad`` in
:mod:`lgoa.verification`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from . import fem as fem_module
from .backend import dtype

__all__ = [
    "initialize_params",
    "pack_controls",
    "unpack_controls",
    "params_from_control_matrix",
    "vector_field",
    "vector_field_pullback",
    "forward_states",
    "forward_adjoint_and_euclidean_grads",
    "objective_and_gradients",
    "multiclass_accuracy",
    "data_loss_only",
    "classification_margins",
    "initial_time_grid",
]


def initial_time_grid(final_time: float, intervals: int):
    """Uniform partition of :math:`[0, T]` into ``intervals`` intervals."""
    return jnp.linspace(0.0, float(final_time), int(intervals) + 1, dtype=dtype())


def initialize_params(key, n_nodes, hidden_dim, num_classes, control_noise=0.0):
    """Initialize endpoint maps and a depth-constant initial control."""
    k_in, k_base, k_noise, k_out = jax.random.split(key, 4)
    float_dtype = dtype()
    W_in = jax.random.normal(k_in, (2, hidden_dim), dtype=float_dtype) * jnp.sqrt(
        2.0 / (2 + hidden_dim)
    )
    A_base = jax.random.normal(
        k_base, (hidden_dim, hidden_dim), dtype=float_dtype
    ) / jnp.sqrt(hidden_dim)
    A = jnp.broadcast_to(A_base, (n_nodes, hidden_dim, hidden_dim))
    if control_noise:
        A = A + control_noise * jax.random.normal(k_noise, A.shape, dtype=float_dtype)
    b = jnp.zeros((n_nodes, hidden_dim), dtype=float_dtype)
    W_out = 0.1 * jax.random.normal(k_out, (num_classes, hidden_dim), dtype=float_dtype)
    return {"W_in": W_in, "A": A, "b": b, "W_out": W_out}


def pack_controls(params):
    r"""Stack :math:`(A, b)` into the nodal coefficient matrix :math:`\Theta_\tau`."""
    n_nodes, hidden_dim = params["b"].shape
    return jnp.concatenate(
        [params["A"].reshape(n_nodes, hidden_dim * hidden_dim), params["b"]], axis=1
    )


def unpack_controls(matrix, hidden_dim):
    """Inverse of :func:`pack_controls`."""
    n_nodes = matrix.shape[0]
    q = hidden_dim * hidden_dim
    return {
        "A": matrix[:, :q].reshape(n_nodes, hidden_dim, hidden_dim),
        "b": matrix[:, q:],
    }


def params_from_control_matrix(theta_matrix, fixed_W_in, fixed_W_out, hidden_dim):
    """Rebuild a parameter tree from a control matrix and frozen endpoint maps."""
    control = unpack_controls(theta_matrix, hidden_dim)
    return {
        "W_in": fixed_W_in,
        "A": control["A"],
        "b": control["b"],
        "W_out": fixed_W_out,
    }


def vector_field(X, A, b):
    return jnp.tanh(X @ A + b)


def vector_field_pullback(X, A, b, P):
    r"""Return :math:`D_1F^{*}P`, :math:`D_AF^{*}P` and :math:`D_bF^{*}P`."""
    H = jnp.tanh(X @ A + b)
    S = P * (1.0 - H * H)
    return S @ A.T, X.T @ S, jnp.sum(S, axis=0)


def forward_states(params, X, time_grid):
    """Explicit Euler forward pass; returns the states at all :math:`K+1` nodes."""
    dts = jnp.diff(time_grid)
    Z0 = X @ params["W_in"]
    A_mid = 0.5 * (params["A"][:-1] + params["A"][1:])
    b_mid = 0.5 * (params["b"][:-1] + params["b"][1:])

    def step(Z, interval_data):
        dt, A, b = interval_data
        Z_next = Z + dt * vector_field(Z, A, b)
        return Z_next, Z_next

    _, tail = lax.scan(step, Z0, (dts, A_mid, b_mid))
    return jnp.concatenate([Z0[None, ...], tail], axis=0)


def forward_adjoint_and_euclidean_grads(params, X, y, time_grid):
    """Forward pass, backward adjoint pass and the Euclidean (load) derivative."""
    states = forward_states(params, X, time_grid)
    logits = states[-1] @ params["W_out"].T
    log_probabilities = jax.nn.log_softmax(logits)
    data_loss = -jnp.mean(jnp.take_along_axis(log_probabilities, y[:, None], axis=1))

    residual = (
        jax.nn.softmax(logits)
        - jax.nn.one_hot(y, params["W_out"].shape[0], dtype=states.dtype)
    ) / X.shape[0]
    grad_W_out = residual.T @ states[-1]
    P_terminal = residual @ params["W_out"]

    dts = jnp.diff(time_grid)
    A_mid = 0.5 * (params["A"][:-1] + params["A"][1:])
    b_mid = 0.5 * (params["b"][:-1] + params["b"][1:])

    def backward_step(P_next, interval_data):
        X_k, A, b, dt = interval_data
        dxT_P, qA, qb = vector_field_pullback(X_k, A, b, P_next)
        P_prev = P_next + dt * dxT_P
        return P_prev, (P_prev, qA, qb)

    _, reverse_output = lax.scan(
        backward_step,
        P_terminal,
        (states[:-1][::-1], A_mid[::-1], b_mid[::-1], dts[::-1]),
    )
    P_prev_reverse, qA_reverse, qb_reverse = reverse_output
    adjoints = jnp.concatenate([P_prev_reverse[::-1], P_terminal[None, ...]], axis=0)
    qA = qA_reverse[::-1]
    qb = qb_reverse[::-1]

    grad_W_in = X.T @ adjoints[0]
    contrib_A = 0.5 * dts[:, None, None] * qA
    contrib_b = 0.5 * dts[:, None] * qb
    load_A = jnp.zeros_like(params["A"]).at[:-1].add(contrib_A).at[1:].add(contrib_A)
    load_b = jnp.zeros_like(params["b"]).at[:-1].add(contrib_b).at[1:].add(contrib_b)

    grads = {
        "W_in": grad_W_in,
        "A": load_A,
        "b": load_b,
        "W_out": grad_W_out,
    }
    return data_loss, states, adjoints, grads


def objective_and_gradients(
    params,
    X,
    y,
    time_grid,
    operators: fem_module.FemOperators,
    reg_lambda,
    train_endpoint_maps,
):
    r"""Objective, :math:`H^1` gradient and the stationarity measure.

    With ``train_endpoint_maps=False`` the endpoint maps :math:`W_{\mathrm{in}}`
    and :math:`W_{\mathrm{out}}` are frozen, which is the setting of every
    fixed-depth control problem in Algorithm 1.
    """
    data_loss, _, _, data_grads = forward_adjoint_and_euclidean_grads(
        params, X, y, time_grid
    )
    theta = pack_controls(params)
    M_theta = fem_module.apply(operators.mass, theta)
    A_theta = fem_module.apply(operators.stiffness, theta)
    mass_energy = jnp.sum(theta * M_theta)
    stiffness_energy = jnp.sum(theta * A_theta)
    h1_energy = mass_energy + stiffness_energy
    regularizer_dual = reg_lambda * (M_theta + A_theta)
    regularizer = 0.5 * reg_lambda * h1_energy

    control_dual = regularizer_dual + pack_controls(data_grads)
    h1_gradient_matrix = fem_module.solve(operators.riesz, control_dual)
    h1_control = unpack_controls(h1_gradient_matrix, params["b"].shape[1])

    B_gradient = fem_module.apply(operators.riesz, h1_gradient_matrix)
    control_norm_sq = jnp.sum(h1_gradient_matrix * B_gradient)
    endpoint_sensitivity_sq = jnp.sum(data_grads["W_in"] ** 2) + jnp.sum(
        data_grads["W_out"] ** 2
    )

    if train_endpoint_maps:
        grad_W_in = data_grads["W_in"]
        grad_W_out = data_grads["W_out"]
        active_endpoint_sq = endpoint_sensitivity_sq
    else:
        grad_W_in = jnp.zeros_like(data_grads["W_in"])
        grad_W_out = jnp.zeros_like(data_grads["W_out"])
        active_endpoint_sq = jnp.asarray(0.0, dtype=theta.dtype)

    grads = {
        "W_in": grad_W_in,
        "A": h1_control["A"],
        "b": h1_control["b"],
        "W_out": grad_W_out,
    }
    control_norm = jnp.sqrt(jnp.maximum(control_norm_sq, 0.0))
    endpoint_sensitivity = jnp.sqrt(jnp.maximum(endpoint_sensitivity_sq, 0.0))
    active_stationarity = jnp.sqrt(
        jnp.maximum(control_norm_sq + active_endpoint_sq, 0.0)
    )
    metrics = {
        "objective": data_loss + regularizer,
        "data_loss": data_loss,
        "regularizer": regularizer,
        "mass_energy": mass_energy,
        "stiffness_energy": stiffness_energy,
        "h1_energy": h1_energy,
        "stationarity_norm": active_stationarity,
        "control_stationarity_norm": control_norm,
        "endpoint_sensitivity_norm": endpoint_sensitivity,
        "control_dual": control_dual,
        "h1_gradient_matrix": h1_gradient_matrix,
    }
    return metrics, grads


def multiclass_accuracy(params, X, y, time_grid):
    logits = forward_states(params, X, time_grid)[-1] @ params["W_out"].T
    return jnp.mean(jnp.argmax(logits, axis=1) == y)


def data_loss_only(params, X, y, time_grid):
    states = forward_states(params, X, time_grid)
    logits = states[-1] @ params["W_out"].T
    log_probabilities = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(log_probabilities, y[:, None], axis=1))


def classification_margins(params, X, y, time_grid):
    """Correct-class logit minus the strongest competitor, per sample."""
    logits = forward_states(params, X, time_grid)[-1] @ params["W_out"].T
    correct = jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]
    masked = logits.at[jnp.arange(logits.shape[0]), y].set(-jnp.inf)
    return correct - jnp.max(masked, axis=1)
