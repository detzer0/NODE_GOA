r"""Direct reconstructed local goal-oriented indicator.

On interval I_k, keep the DG(0) representatives X_{k-1} and P_k fixed,
interpolate the control linearly, and integrate the signed pairings

    rho_adjoint^k = -h_k int_0^1 <D_x F(X_{k-1}, theta(s))^* P_k,
                                 s (X_k - X_{k-1})> ds,
    rho_state^k   =  h_k int_0^1 <F(X_{k-1}, theta(s)),
                                 (1-s) (P_k - P_{k-1})> ds.

The one-sided reconstruction errors vanish at the traces used in the nodal
residuals, so the corresponding jump pairings are zero. These are NOT products
of residual bounds and reconstruction norms. Absolute values are taken only
AFTER integration:

    eta_k = |rho_adjoint^k| + |rho_control^k| + |rho_state^k|,
    eta_K = 0.5 * sum_k eta_k.

The state/adjoint volume integrals use the two reference notebooks' global
Gauss--Legendre order-doubling rule (8, 16, ..., 128 by default). All local
signed integrals must satisfy atol + rtol * abs(high_order_value). As in the
notebooks, the highest-order value is retained silently if that test fails;
``direct_quadrature_converged`` records this, and the difference between the
last two rules is an empirical estimate, not a certified integration bound.

The control residual follows eq. (89). Its mass pairing is integrated in closed
form through the coefficients alpha_k, beta_k and the local mass matrix, and its
derivative pairing cancels analytically because the reconstruction matches both
endpoints. The remaining nonlinear pullback term

    rho_dynamics^k = h_k int_0^1 <D_theta N_F(X_{k-1}, theta(s))^* P_k,
                                  theta*[theta](s) - theta(s)> ds,

with ``theta*[theta](s) - theta(s) = A_k (s^2 - s)``, is evaluated by the same
Gauss--Legendre order-doubling rule as the state and adjoint residuals, which is
what the paper specifies. Setting ``control_pullback_quadrature`` to
``"endpoint_linear"`` instead reconstructs that pullback linearly from its two
endpoint values, reproducing the earlier notebook implementation.

Existing component/aggregate field names remain valid. Signed components and
quadrature diagnostics are available for inspection and CSV export.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .neural_ode import pack_controls

__all__ = [
    "LocalIndicators",
    "compute_local_indicators",
    "direct_reconstructed_state_adjoint_residuals",
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
    state_signed, adjoint_signed, control_signed:
        Local residual pairings before taking absolute values.
    direct_quadrature_order, direct_quadrature_error_estimate:
        Last GL order and maximum absolute difference from the preceding rule,
        over all local state and adjoint integrals.
    direct_quadrature_converged:
        Whether all those differences met the absolute/relative tolerance.
        False means the highest-order fallback was used, not a failed solve.
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
    # Defaults preserve construction with the original public fields.
    state_signed: np.ndarray | None = None
    adjoint_signed: np.ndarray | None = None
    control_signed: np.ndarray | None = None
    control_mass_signed: np.ndarray | None = None
    control_stiffness_signed: np.ndarray | None = None
    control_dynamics_signed: np.ndarray | None = None
    direct_quadrature_order: int = 0
    direct_quadrature_error_estimate: float = float("nan")
    direct_quadrature_converged: bool = False
    reconstruction_jump_trace_error: float = float("nan")

    @property
    def global_direct_indicator(self) -> float:
        """Notebook name for the retained ``global_practical_indicator`` field."""
        return self.global_practical_indicator

    @property
    def state_volume_signed(self):
        return self.state_signed

    @property
    def adjoint_volume_signed(self):
        return self.adjoint_signed

    @property
    def state_jump_signed(self) -> np.ndarray:
        return np.zeros_like(self.state)

    @property
    def adjoint_jump_signed(self) -> np.ndarray:
        return np.zeros_like(self.adjoint)

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


def _validate_quadrature_settings(initial_order, maximum_order, rtol, atol):
    """Reject invalid settings before entering the order-doubling loop."""
    for name, value in (
        ("initial_order", initial_order),
        ("maximum_order", maximum_order),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"Quadrature {name} must be an integer.")
    if initial_order < 2:
        raise ValueError("Quadrature initial_order must be at least two.")
    if maximum_order < 2 * initial_order:
        raise ValueError("Quadrature maximum_order must be at least 2 * initial_order.")
    if not (np.isfinite(rtol) and np.isfinite(atol) and rtol > 0.0 and atol > 0.0):
        raise ValueError("Quadrature tolerances must be finite and positive.")


def _direct_reconstructed_volume_terms_at_order(
    states,
    adjoints,
    theta,
    dts,
    hidden_dim,
    order,
    quadratic_coefficients=None,
):
    r"""
    Evaluate the direct state and adjoint residual pairings with one
    Gauss--Legendre rule.

    On interval I_k, the DG(0) representatives are x_tau^{k-1} and p_tau^k,
    and the piecewise-linear reconstruction errors are

        e_x^R(s) = s (x_tau^k - x_tau^{k-1}),
        e_p^R(s) = -(1-s) (p_tau^k - p_tau^{k-1}).
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    s = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    K = len(dts)
    qA_size = hidden_dim * hidden_dim
    rho_adjoint_volume = np.zeros(K, dtype=float)  # rho_x
    rho_state_volume = np.zeros(K, dtype=float)    # rho_p
    # Control pullback term of eq. (89), on the same rule, when requested.
    want_dynamics = quadratic_coefficients is not None
    rho_dynamics = np.zeros(K, dtype=float) if want_dynamics else None
    control_weight = s * s - s

    for k in range(K):
        X_const = states[k]
        P_const = adjoints[k + 1]
        delta_x = states[k + 1] - states[k]
        delta_p = adjoints[k + 1] - adjoints[k]

        theta_q = (
            (1.0 - s[:, None]) * theta[k]
            + s[:, None] * theta[k + 1]
        )
        A_q = theta_q[:, :qA_size].reshape(
            len(s), hidden_dim, hidden_dim
        )
        b_q = theta_q[:, qA_size:]

        preactivation = (
            np.einsum("nd,qde->qne", X_const, A_q)
            + b_q[:, None, :]
        )
        field = np.tanh(preactivation)

        weighted_adjoint = (
            P_const[None, :, :]
            * (1.0 - field * field)
        )
        adjoint_pullback = np.einsum(
            "qni,qji->qnj",
            weighted_adjoint,
            A_q,
        )

        state_error = s[:, None, None] * delta_x[None, :, :]
        adjoint_error = -(1.0 - s)[:, None, None] * delta_p[None, :, :]

        adjoint_integrand = -np.einsum(
            "qnd,qnd->q",
            adjoint_pullback,
            state_error,
        )
        state_integrand = -np.einsum(
            "qnd,qnd->q",
            field,
            adjoint_error,
        )

        rho_adjoint_volume[k] = dts[k] * np.dot(
            weights,
            adjoint_integrand,
        )
        rho_state_volume[k] = dts[k] * np.dot(
            weights,
            state_integrand,
        )

        if want_dynamics:
            # D_theta N_F(X, theta(s))^* P in packed (A, b) order, reusing the
            # weighted adjoint already formed above.
            qA = np.einsum("nd,qne->qde", X_const, weighted_adjoint)
            qb = np.einsum("qne->qe", weighted_adjoint)
            pullback = np.concatenate(
                [qA.reshape(len(s), qA_size), qb], axis=1
            )
            dynamics_integrand = (
                pullback @ quadratic_coefficients[k]
            ) * control_weight
            rho_dynamics[k] = dts[k] * np.dot(weights, dynamics_integrand)

    return rho_adjoint_volume, rho_state_volume, rho_dynamics


def direct_reconstructed_state_adjoint_residuals(
    states,
    adjoints,
    theta,
    dts,
    hidden_dim,
    cfg,
    quadratic_coefficients=None,
):
    r"""
    Compute rho_x^k, rho_p^k and, when requested, the control pullback term
    rho_dynamics^k, by Gauss--Legendre order doubling.

    The piecewise-linear reconstructions interpolate the one-sided traces
    occurring in the nodal residuals:

        (R x_tau - x_tau)(t_{k-1}^+) = 0,
        (R p_tau - p_tau)(t_k^-) = 0.

    Hence both reconstructed jump pairings are exactly zero. If the requested
    order-doubling tolerance is not reached, the maximum-order value is
    accepted silently.
    """
    _validate_quadrature_settings(
        cfg["direct_quadrature_initial_order"], cfg["direct_quadrature_max_order"],
        cfg["direct_quadrature_rtol"], cfg["direct_quadrature_atol"],
    )
    initial_order = int(cfg["direct_quadrature_initial_order"])
    maximum_order = int(cfg["direct_quadrature_max_order"])
    rtol = float(cfg["direct_quadrature_rtol"])
    atol = float(cfg["direct_quadrature_atol"])

    def evaluate(order):
        values = _direct_reconstructed_volume_terms_at_order(
            states, adjoints, theta, dts, hidden_dim, order, quadratic_coefficients
        )
        for array in values:
            if array is not None and not np.all(np.isfinite(array)):
                raise FloatingPointError(
                    "Non-finite reconstructed state/adjoint/control residual."
                )
        return values

    low_order = initial_order
    low = evaluate(low_order)

    accepted = False
    maximum_error = np.inf
    used_order = low_order
    high = low

    while low_order < maximum_order:
        high_order = min(2 * low_order, maximum_order)
        high = evaluate(high_order)
        used_order = high_order

        errors = [
            (np.abs(h - l), atol + rtol * np.abs(h))
            for h, l in zip(high, low, strict=True)
            if h is not None
        ]
        maximum_error = float(max(np.max(error) for error, _ in errors))

        if all(np.all(error <= tol) for error, tol in errors):
            accepted = True
            break

        low_order = high_order
        low = high

    high_adjoint, high_state, high_dynamics = high

    K = len(dts)
    rho_adjoint_jump = np.zeros(K, dtype=float)
    rho_state_jump = np.zeros(K, dtype=float)

    return {
        "rho_adjoint_volume_signed": high_adjoint,
        "rho_adjoint_jump_signed": rho_adjoint_jump,
        "rho_adjoint_signed": high_adjoint + rho_adjoint_jump,
        "rho_state_volume_signed": high_state,
        "rho_state_jump_signed": rho_state_jump,
        "rho_state_signed": high_state + rho_state_jump,
        "rho_dynamics_signed": high_dynamics,
        "quadrature_order": int(used_order),
        "quadrature_error_estimate": maximum_error,
        "quadrature_converged": bool(accepted),
        "reconstruction_jump_trace_error": 0.0,
    }


def compute_local_indicators(
    params,
    time_grid,
    X,
    y,
    solution_fn,
    *,
    reg_lambda,
    supremum_samples=None,
    direct_quadrature_initial_order=8,
    direct_quadrature_max_order=128,
    direct_quadrature_rtol=1.0e-7,
    direct_quadrature_atol=1.0e-9,
    control_pullback_quadrature="gauss",
) -> LocalIndicators:
    r"""Evaluate :math:`\{\eta_k\}_{k=1}^{K}` and its global aggregates.

    Parameters
    ----------
    solution_fn:
        Compiled ``(params, X, y) -> (data_loss, states, adjoints)`` closure for
        the current grid, as produced by :func:`lgoa.solver.make_control_problem`.
    supremum_samples:
        Deprecated compatibility argument. Accepted but ignored: no sampled
        supremum enters the direct reconstructed indicator.
    direct_quadrature_initial_order, direct_quadrature_max_order:
        State/adjoint Gauss--Legendre orders; defaults match both notebooks.
    direct_quadrature_rtol, direct_quadrature_atol:
        Componentwise tolerances between consecutive signed integral estimates.
        At the order cap, retain the last estimate and record nonconvergence.
    control_pullback_quadrature:
        ``"gauss"`` evaluates the nonlinear control pullback of eq. (89) on the
        same Gauss--Legendre rule as the state and adjoint residuals, as the
        paper specifies. ``"endpoint_linear"`` reconstructs it linearly from its
        two endpoint values, reproducing the earlier notebook implementation.
    """
    if control_pullback_quadrature not in ("gauss", "endpoint_linear"):
        raise ValueError(
            "control_pullback_quadrature must be 'gauss' or 'endpoint_linear', "
            f"got {control_pullback_quadrature!r}."
        )
    _validate_quadrature_settings(
        direct_quadrature_initial_order, direct_quadrature_max_order,
        direct_quadrature_rtol, direct_quadrature_atol,
    )
    grid = np.asarray(time_grid)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError(
            "time_grid must be a finite one-dimensional array with >= 2 nodes."
        )
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    if not np.isfinite(reg_lambda) or reg_lambda < 0.0:
        raise ValueError("reg_lambda must be finite and nonnegative.")
    _, states, adjoints = solution_fn(params, X, y)
    states = np.asarray(states)
    adjoints = np.asarray(adjoints)
    theta = np.asarray(pack_controls(params))
    grid = np.asarray(time_grid)
    dts = np.diff(grid)
    K = len(dts)
    hidden_dim = params["b"].shape[1]
    qA_size = hidden_dim * hidden_dim

    if states.ndim != 3 or states.shape[0] != K + 1 or states.shape[2] != hidden_dim:
        raise ValueError("states must have shape (K+1, n_samples, hidden_dim).")
    if adjoints.shape != states.shape:
        raise ValueError("adjoints and states must have the same shape.")
    if theta.shape != (K + 1, qA_size + hidden_dim):
        raise ValueError("The nodal controls must match time_grid and hidden_dim.")
    if not all(np.all(np.isfinite(a)) for a in (states, adjoints, theta)):
        raise FloatingPointError("States, adjoints and controls must be finite.")

    A_quad, B_quad, C_quad = quadratic_reconstruction_coefficients(theta, grid)
    alpha, beta = linear_quadratic_pairing_coefficients(A_quad, B_quad, C_quad)
    use_endpoint_linear = control_pullback_quadrature == "endpoint_linear"

    rho_mass = np.zeros(K)
    rho_stiffness = np.zeros(K)
    rho_dynamics = np.zeros(K)

    for k in range(K):
        X_const = states[k]
        P_const = adjoints[k + 1]
        dt = dts[k]

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

        if use_endpoint_linear:
            # Legacy path: endpoint pullbacks reconstructed linearly.
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

    direct = direct_reconstructed_state_adjoint_residuals(
        states, adjoints, theta, dts, hidden_dim,
        {
            "direct_quadrature_initial_order": direct_quadrature_initial_order,
            "direct_quadrature_max_order": direct_quadrature_max_order,
            "direct_quadrature_rtol": direct_quadrature_rtol,
            "direct_quadrature_atol": direct_quadrature_atol,
        },
        None if use_endpoint_linear else A_quad,
    )
    if not use_endpoint_linear:
        rho_dynamics = direct["rho_dynamics_signed"]
    rho_control = rho_mass + rho_stiffness + rho_dynamics
    eta_state = np.abs(direct["rho_state_signed"])
    eta_adjoint = np.abs(direct["rho_adjoint_signed"])
    eta_control = np.abs(rho_control)
    if not np.all(np.isfinite(rho_control)):
        raise FloatingPointError("Non-finite control residual.")

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
        state_signed=direct["rho_state_signed"],
        adjoint_signed=direct["rho_adjoint_signed"],
        control_signed=rho_control,
        control_mass_signed=rho_mass,
        control_stiffness_signed=rho_stiffness,
        control_dynamics_signed=rho_dynamics,
        direct_quadrature_order=direct["quadrature_order"],
        direct_quadrature_error_estimate=direct["quadrature_error_estimate"],
        direct_quadrature_converged=direct["quadrature_converged"],
        reconstruction_jump_trace_error=direct["reconstruction_jump_trace_error"],
    )
