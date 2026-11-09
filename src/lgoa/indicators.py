r"""The computable local goal-oriented indicator.

For every interval :math:`I_k` the practical indicator of the paper (eq. (90))
is the sum of three residual-weight pairings,

.. math::

    \eta_k = R^k_{p}\,\omega^k_{x}
           + \lvert \rho^k_{\theta} \rvert
           + R^k_{x}\,\omega^k_{p},

and :math:`\eta_K = \tfrac12 \sum_{k=1}^{K} \eta_k` approximates the global
quantity.  The state and adjoint residual factors involve supremum norms, which
are sampled uniformly on each interval.  The control residual is evaluated in
closed form: the quadratic reconstruction
:math:`\vartheta_\tau = A_k s^2 + B_k s + C_k` is built analytically, the
pullback :math:`q = D_2 N_F(x_\tau, \theta_\tau)^{*} p_\tau` is evaluated at both
endpoints and reconstructed linearly, and each polynomial pairing is integrated
exactly using

.. math::

    \alpha_k = \tfrac{A_k}{12} + \tfrac{B_k}{6} + \tfrac{C_k}{2}, \qquad
    \beta_k  = \tfrac{A_k}{4}  + \tfrac{B_k}{3} + \tfrac{C_k}{2}.

The derivative part of the regularization pairing cancels analytically, because
the reconstruction matches both endpoints: :math:`A_k + B_k = \Delta\theta_k`.
Its largest surviving magnitude is reported as
``max_abs_control_stiffness_term``, which therefore monitors that cancellation
at run time; the closed-form pairing itself is checked against dense quadrature
in :mod:`lgoa.verification`.

:class:`LocalIndicators` carries exactly the quantities the rest of the package
consumes -- marking reads ``total``, the figures and the CSV export read the
three components and the global aggregates.  The intermediate residual and jump
factors are local to :func:`compute_local_indicators`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neural_ode import pack_controls

__all__ = [
    "LocalIndicators",
    "compute_local_indicators",
    "quadratic_reconstruction_coefficients",
    "linear_quadratic_pairing_coefficients",
    "exact_linear_linear_pairing",
]


@dataclass(frozen=True)
class LocalIndicators:
    r"""Indicator values on one temporal grid.

    Attributes
    ----------
    grid:
        The :math:`K + 1` temporal nodes the indicators belong to.
    total:
        :math:`\eta_k` per interval; the quantity marking acts on.
    state, adjoint, control:
        The three contributions whose sum is ``total``.
    global_state_indicator, global_adjoint_indicator, global_control_indicator:
        Halved sums of the corresponding local arrays.
    global_practical_indicator:
        :math:`\eta_K`, the sum of the three global contributions.
    control_fraction:
        Share of :math:`\eta_K` carried by the control term.
    max_abs_control_stiffness_term:
        Largest magnitude of the analytically cancelling derivative pairing;
        a roundoff monitor, expected to stay near machine precision.
    """

    grid: np.ndarray
    total: np.ndarray
    state: np.ndarray
    adjoint: np.ndarray
    control: np.ndarray
    global_state_indicator: float
    global_adjoint_indicator: float
    global_control_indicator: float
    global_practical_indicator: float
    control_fraction: float
    max_abs_control_stiffness_term: float

    @property
    def depth(self) -> int:
        return int(self.total.size)


def _packed_control_pullback(X, A, b, P):
    r"""Return :math:`D_\theta N_F(X, \theta)^{*} P` in packed :math:`(A, b)` order."""
    H = np.tanh(X @ A + b)
    S = P * (1.0 - H * H)
    qA = X.T @ S
    qb = np.sum(S, axis=0)
    return np.concatenate([qA.ravel(), qb])


def quadratic_reconstruction_coefficients(theta, time_grid):
    r"""Coefficients :math:`A_k, B_k, C_k` of the quadratic control reconstruction."""
    theta = np.asarray(theta)
    grid = np.asarray(time_grid)
    dts = np.diff(grid)

    slopes = np.zeros_like(theta)
    slopes[1:] = (theta[1:] - theta[:-1]) / dts[:, None]
    C = theta[:-1]
    B = slopes[:-1] * dts[:, None]
    A = theta[1:] - theta[:-1] - B
    return A, B, C


def linear_quadratic_pairing_coefficients(A, B, C):
    r"""Exact coefficients of :math:`\int_0^1 (u_0(1-s) + u_1 s)(As^2 + Bs + C)\,ds`."""
    alpha = A / 12.0 + B / 6.0 + C / 2.0
    beta = A / 4.0 + B / 3.0 + C / 2.0
    return alpha, beta


def exact_linear_linear_pairing(u0, u1, v0, v1, dt):
    """Exact integral of two linearly interpolated vector functions."""
    return dt * (
        np.dot(u0, v0) / 3.0
        + np.dot(u0, v1) / 6.0
        + np.dot(u1, v0) / 6.0
        + np.dot(u1, v1) / 3.0
    )


def compute_local_indicators(
    params,
    time_grid,
    X,
    y,
    solution_fn,
    *,
    reg_lambda,
    supremum_samples=50,
) -> LocalIndicators:
    r"""Evaluate :math:`\{\eta_k\}_{k=1}^{K}` and its global aggregates.

    Parameters
    ----------
    solution_fn:
        Compiled ``(params, X, y) -> (data_loss, states, adjoints)`` closure for
        the current grid, as produced by :func:`lgoa.solver.make_control_problem`.
    """
    _, states, adjoints = solution_fn(params, X, y)
    states = np.asarray(states)
    adjoints = np.asarray(adjoints)
    theta = np.asarray(pack_controls(params))
    grid = np.asarray(time_grid)
    dts = np.diff(grid)
    K = len(dts)
    hidden_dim = params["b"].shape[1]
    qA_size = hidden_dim * hidden_dim

    # Jump terms of the piecewise reconstructions.
    omega_state = np.linalg.norm((states[1:] - states[:-1]).reshape(K, -1), axis=1)
    omega_adjoint = np.linalg.norm(
        (adjoints[1:] - adjoints[:-1]).reshape(K, -1), axis=1
    )

    sup_points = np.linspace(0.0, 1.0, int(supremum_samples))

    A_quad, B_quad, C_quad = quadratic_reconstruction_coefficients(theta, grid)
    alpha, beta = linear_quadratic_pairing_coefficients(A_quad, B_quad, C_quad)

    R_state = np.zeros(K)
    R_adjoint = np.zeros(K)
    rho_mass = np.zeros(K)
    rho_stiffness = np.zeros(K)
    rho_dynamics = np.zeros(K)

    for k in range(K):
        X_const = states[k]
        P_const = adjoints[k + 1]
        dt = dts[k]

        field_norms = []
        adjoint_operator_norms = []
        for s in sup_points:
            theta_s = (1.0 - s) * theta[k] + s * theta[k + 1]
            A_s = theta_s[:qA_size].reshape(hidden_dim, hidden_dim)
            b_s = theta_s[qA_size:]
            H = np.tanh(X_const @ A_s + b_s)
            field_norms.append(np.linalg.norm(H))
            adjoint_operator = (P_const * (1.0 - H * H)) @ A_s.T
            adjoint_operator_norms.append(np.linalg.norm(adjoint_operator))

        next_dt = dts[k + 1] if k < K - 1 else 0.0
        R_state[k] = dt * max(field_norms)
        R_state[k] += dt / (dt + next_dt) * omega_state[k]
        if k > 0:
            # The interval also receives the preceding state jump contribution.
            R_state[k] += dt / (dt + dts[k - 1]) * omega_state[k - 1]

        previous_dt = dts[k - 1] if k > 0 else 0.0
        R_adjoint[k] = dt * max(adjoint_operator_norms)
        R_adjoint[k] += dt / (previous_dt + dt) * omega_adjoint[k]
        if k < K - 1:
            R_adjoint[k] += dt / (dt + dts[k + 1]) * omega_adjoint[k + 1]

        theta_left = theta[k]
        theta_right = theta[k + 1]

        # Mass part of the regularization pairing.
        int_theta_vartheta = dt * (
            np.dot(theta_left, alpha[k]) + np.dot(theta_right, beta[k])
        )
        int_theta_theta = exact_linear_linear_pairing(
            theta_left, theta_right, theta_left, theta_right, dt
        )
        rho_mass[k] = reg_lambda * (int_theta_vartheta - int_theta_theta)

        # Derivative part: cancels analytically up to roundoff.
        delta = theta_right - theta_left
        int_dtheta_dvartheta = np.dot(delta, A_quad[k] + B_quad[k]) / dt
        int_dtheta_dtheta = np.dot(delta, delta) / dt
        rho_stiffness[k] = reg_lambda * (int_dtheta_dvartheta - int_dtheta_dtheta)

        # Dynamics part: endpoint pullbacks reconstructed linearly.
        A_left = theta_left[:qA_size].reshape(hidden_dim, hidden_dim)
        b_left = theta_left[qA_size:]
        A_right = theta_right[:qA_size].reshape(hidden_dim, hidden_dim)
        b_right = theta_right[qA_size:]
        q_left = _packed_control_pullback(X_const, A_left, b_left, P_const)
        q_right = _packed_control_pullback(X_const, A_right, b_right, P_const)

        int_q_vartheta = dt * (np.dot(q_left, alpha[k]) + np.dot(q_right, beta[k]))
        int_q_theta = exact_linear_linear_pairing(
            q_left, q_right, theta_left, theta_right, dt
        )
        rho_dynamics[k] = int_q_vartheta - int_q_theta

    eta_state = R_state * omega_adjoint
    eta_adjoint = R_adjoint * omega_state
    eta_control = np.abs(rho_mass + rho_stiffness + rho_dynamics)

    global_state = float(0.5 * np.sum(eta_state))
    global_adjoint = float(0.5 * np.sum(eta_adjoint))
    global_control = float(0.5 * np.sum(eta_control))
    global_total = global_state + global_adjoint + global_control

    return LocalIndicators(
        grid=grid,
        total=eta_state + eta_adjoint + eta_control,
        state=eta_state,
        adjoint=eta_adjoint,
        control=eta_control,
        global_state_indicator=global_state,
        global_adjoint_indicator=global_adjoint,
        global_control_indicator=global_control,
        global_practical_indicator=global_total,
        control_fraction=global_control / (global_total + 1.0e-30),
        max_abs_control_stiffness_term=float(np.max(np.abs(rho_stiffness))),
    )
