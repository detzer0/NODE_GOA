"""Regression tests for the notebook-matched direct residual indicator.

The six fixtures store values obtained by executing the indicator definitions
from BOTH supplied notebooks. See docs/notebook_agreement.json for provenance.
They do not require the original notebooks or a long training run.
"""
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lgoa
from lgoa.config import (
    AdaptiveConfig,
    ExperimentConfig,
    InitializationConfig,
    ProblemConfig,
    SolverConfig,
)
from lgoa.indicators import compute_local_indicators

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CASES = ("single", "uniform", "nonuniform", "constant", "sharp", "zero")
ARRAY_FIELDS = (
    "total", "state", "adjoint", "control", "state_signed", "adjoint_signed",
    "control_signed", "state_volume_signed", "adjoint_volume_signed",
    "state_jump_signed", "adjoint_jump_signed", "control_mass_signed",
    "control_stiffness_signed", "control_dynamics_signed",
)
SCALAR_FIELDS = (
    "global_state_indicator", "global_adjoint_indicator", "global_control_indicator",
    "global_direct_indicator", "control_fraction", "max_abs_control_stiffness_term",
    "direct_quadrature_order", "direct_quadrature_error_estimate",
    "direct_quadrature_converged", "reconstruction_jump_trace_error",
)


@pytest.fixture(autouse=True)
def _double_precision():
    old = lgoa.backend.x64_enabled()
    lgoa.configure(True)
    yield
    lgoa.configure(old)


def _case(name):
    with np.load(FIXTURE_DIR / f"indicators_{name}.npz", allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    params = {key: jnp.asarray(values[key]) for key in ("A", "b", "W_in", "W_out")}
    grid, X, y = (jnp.asarray(values[key]) for key in ("grid", "X", "y"))

    def solution_fn(params, X, y):
        return lgoa.neural_ode.forward_adjoint_and_euclidean_grads(
            params, X, y, grid
        )[:3]

    return values, params, grid, X, y, solution_fn


def _evaluate_case(name, **settings):
    values, params, grid, X, y, solution = _case(name)
    # These fixtures record the reference notebooks, whose control pullback is
    # reconstructed linearly from its endpoints. The package now defaults to the
    # Gauss-Legendre pullback of eq. (89), so pin the legacy path here: this
    # suite's job is to show the notebook path is still reproduced exactly.
    settings.setdefault("control_pullback_quadrature", "endpoint_linear")
    actual = compute_local_indicators(
        params, grid, X, y, solution, reg_lambda=float(values["reg_lambda"]), **settings
    )
    return values, actual


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("reference", ("swiss", "peaks"))
def test_original_notebook_indicator_regression(case, reference):
    expected, actual = _evaluate_case(case)
    for field in ARRAY_FIELDS + SCALAR_FIELDS:
        np.testing.assert_allclose(
            getattr(actual, field), expected[f"{reference}__{field}"],
            rtol=2.0e-12, atol=2.0e-13, err_msg=f"{case}: {reference}: {field}",
        )
    np.testing.assert_allclose(actual.total, actual.state + actual.adjoint + actual.control)
    assert actual.global_practical_indicator == actual.global_direct_indicator
    np.testing.assert_allclose(actual.global_direct_indicator, 0.5 * np.sum(actual.total))
    for fraction in (0.25, 0.5, 0.9, 1.0):
        a = lgoa.marking.mark(actual.total, "dorfler", max_new_layers=2,
                              dorfler_fraction=fraction)
        b = lgoa.marking.mark(expected[f"{reference}__total"], "dorfler",
                              max_new_layers=2, dorfler_fraction=fraction)
        # The marked set and the budget decision must agree exactly; the
        # derived weights are sums of floats that already differ in the last
        # ulp, so comparing the dataclasses with == would be a bit-equality
        # test on quantities the assertion above only pins to 2e-12.
        assert a.marked_intervals == b.marked_intervals, f"{case}: {reference}"
        assert a.budget_capped == b.budget_capped
        assert a.proposed_count == b.proposed_count
        np.testing.assert_allclose(a.achieved_share, b.achieved_share, rtol=1.0e-11)
        np.testing.assert_allclose(a.total_weight, b.total_weight, rtol=1.0e-11)


def test_supremum_argument_no_longer_changes_the_indicator():
    _, a = _evaluate_case("nonuniform", supremum_samples=2)
    _, b = _evaluate_case("nonuniform", supremum_samples=1000)
    np.testing.assert_array_equal(a.total, b.total)


def test_maximum_order_fallback_matches_the_notebooks_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, actual = _evaluate_case("sharp")
    assert not caught
    assert actual.direct_quadrature_order == 128
    assert not actual.direct_quadrature_converged
    assert np.isfinite(actual.total).all()
    assert actual.direct_quadrature_error_estimate > 0.0


def test_signed_pairings_are_not_norm_bounds():
    # Opposite fields cancel in the signed pairing. A product of norms cannot
    # reproduce this cancellation, despite nonzero state/adjoint differences.
    params = {"A": jnp.ones((2, 1, 1)), "b": jnp.zeros((2, 1))}
    states = np.array([[[-1.0], [1.0]], [[-0.5], [1.5]]])
    adjoints = np.array([[[0.0], [0.0]], [[1.0], [1.0]]])
    def solution(*_):
        return 0.0, states, adjoints
    actual = compute_local_indicators(
        params, np.array([0.0, 0.7]), None, None, solution, reg_lambda=0.0
    )
    np.testing.assert_allclose(actual.state_signed, [0.0], atol=1e-15)
    np.testing.assert_allclose(actual.state, [0.0], atol=1e-15)
    assert np.linalg.norm(np.tanh(states[0])) * np.linalg.norm(adjoints[1]) > 0.0
    assert actual.adjoint_signed[0] < 0.0
    np.testing.assert_allclose(actual.adjoint, -actual.adjoint_signed)
    np.testing.assert_array_equal(actual.state_jump_signed, [0.0])
    np.testing.assert_array_equal(actual.adjoint_jump_signed, [0.0])


def test_constant_controls_have_analytic_state_and_adjoint_integrals():
    values, params, grid, X, y, solution = _case("constant")
    actual = compute_local_indicators(
        params, grid, X, y, solution, reg_lambda=float(values["reg_lambda"])
    )
    _, states, adjoints = solution(params, X, y)
    states, adjoints = np.asarray(states), np.asarray(adjoints)
    for k, dt in enumerate(np.diff(grid)):
        f = np.tanh(states[k] @ np.asarray(params["A"][k]) + np.asarray(params["b"][k]))
        dxp = (adjoints[k + 1] * (1 - f * f)) @ np.asarray(params["A"][k]).T
        ref_adj = -0.5 * dt * np.sum(dxp * (states[k + 1] - states[k]))
        ref_state = 0.5 * dt * np.sum(f * (adjoints[k + 1] - adjoints[k]))
        np.testing.assert_allclose(
            actual.adjoint_signed[k], ref_adj, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            actual.state_signed[k], ref_state, rtol=1e-12, atol=1e-14
        )
    np.testing.assert_allclose(actual.control_signed, 0.0, atol=1e-14)


@pytest.mark.parametrize("settings", [
    {"direct_quadrature_initial_order": 1},
    {"direct_quadrature_initial_order": 2.5},
    {"direct_quadrature_initial_order": True},
    {"direct_quadrature_max_order": 8},
    {"direct_quadrature_max_order": 32.5},
    {"direct_quadrature_rtol": 0.0},
    {"direct_quadrature_rtol": float("nan")},
    {"direct_quadrature_atol": -1.0},
    {"direct_quadrature_atol": float("inf")},
])
def test_invalid_quadrature_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        AdaptiveConfig(**settings)
    with pytest.raises(ValueError):
        _evaluate_case("single", **settings)


@pytest.mark.parametrize("grid", [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                  np.array([0.0, np.nan]), np.array([0.0])])
def test_invalid_time_grid_is_rejected_before_solution_evaluation(grid):
    def must_not_run(*_):
        raise AssertionError("Solution should not be evaluated on an invalid grid.")
    with pytest.raises(ValueError):
        compute_local_indicators({}, grid, None, None, must_not_run, reg_lambda=0.01)


def test_nonfinite_solution_is_not_silently_accepted():
    params = {"A": jnp.ones((2, 1, 1)), "b": jnp.zeros((2, 1))}
    bad_states = np.full((2, 1, 1), np.nan)
    def solution(*_):
        return 0.0, bad_states, np.zeros_like(bad_states)
    with pytest.raises(FloatingPointError):
        compute_local_indicators(params, [0.0, 1.0], None, None, solution, reg_lambda=0.0)


def test_old_yaml_loads_and_new_overrides_are_routed(tmp_path):
    payload = {
        "problem": dict(name="swiss_roll", num_classes=2, hidden_dim=3,
                        final_time=1.0, samples_per_class=10),
        "adaptive": dict(indicator_sup_samples=50),
    }
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = lgoa.load_yaml(path)
    assert cfg.adaptive.indicator_sup_samples == 50
    assert cfg.adaptive.direct_quadrature_initial_order == 8
    assert cfg.adaptive.direct_quadrature_max_order == 128
    assert cfg.adaptive.direct_quadrature_rtol == 1e-7
    assert cfg.adaptive.direct_quadrature_atol == 1e-9
    cfg = cfg.with_overrides(direct_quadrature_initial_order=4,
                             direct_quadrature_max_order=32)
    assert cfg.adaptive.direct_quadrature_initial_order == 4
    assert cfg.adaptive.direct_quadrature_max_order == 32


def test_self_check_includes_the_residual_checks():
    table = lgoa.self_check()
    assert len(table) == 9
    assert table.passed.all()
    assert table["check"].str.contains("direct state/adjoint").any()
    assert table["check"].str.contains("Gauss control residual").any()
    assert table["check"].str.contains("endpoint-linear control residual").any()


@pytest.mark.parametrize("problem_name", ("swiss_roll", "peaks"))
@pytest.mark.parametrize("rule", ("conservative", "dorfler"))
def test_small_adaptive_hierarchy_and_diagnostics(
    problem_name, rule, tmp_path, monkeypatch
):
    problem = ProblemConfig(
        name=problem_name, num_classes=2 if problem_name == "swiss_roll" else 5,
        hidden_dim=3, final_time=1.0, samples_per_class=12,
        grid_size=24 if problem_name == "peaks" else None,
    )
    cfg = ExperimentConfig(
        problem=problem,
        initialization=InitializationConfig(search_max_epochs=2, search_min_epochs=1,
                                            check_every=1),
        solver=SolverConfig(adam_max_steps=4, adam_min_steps=1, adam_check_every=2,
                            use_bfgs_polish=False),
        adaptive=AdaptiveConfig(initial_depth=2, max_depth=4, marking=rule,
                                direct_quadrature_initial_order=4,
                                direct_quadrature_max_order=32,
                                direct_quadrature_rtol=1e-8, direct_quadrature_atol=1e-10),
        verbose=False,
    )
    real_compute = lgoa.indicators.compute_local_indicators
    calls = []

    def record_calls(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(lgoa.indicators, "compute_local_indicators", record_calls)
    data = lgoa.load_dataset(problem)
    history = lgoa.run(cfg, data, progress=False)
    assert history.final_phase.depth == 4
    assert len(calls) == len(history.phases) + len(history.events)
    for kwargs in calls:
        assert kwargs["direct_quadrature_initial_order"] == 4
        assert kwargs["direct_quadrature_max_order"] == 32
        assert kwargs["direct_quadrature_rtol"] == 1e-8
        assert kwargs["direct_quadrature_atol"] == 1e-10
        assert "supremum_samples" not in kwargs
    for phase in history.phases:
        assert phase.indicators.direct_quadrature_order in (8, 16, 32)
        assert np.isfinite(phase.eta).all()
        assert (phase.eta >= 0).all()
    assert "direct_quadrature_converged" in history.phase_table
    assert "post_insertion_direct_quadrature_converged" in history.event_table
    written = history.to_csv(tmp_path)
    indicators_csv = next(
        path for path in written if path.name.endswith("_final_indicators.csv")
    )
    table = pd.read_csv(indicators_csv)
    assert {"rho_state_signed", "rho_adjoint_signed", "rho_control_signed",
            "direct_quadrature_order", "direct_quadrature_converged"} <= set(table)
    if problem_name == "swiss_roll" and rule == "conservative":
        builders = [
            lambda: lgoa.plots.dataset(data),
            lambda: lgoa.plots.decision_boundary(history, data, resolution=20),
            lambda: lgoa.plots.learning_curves(history),
            lambda: lgoa.plots.trajectories(history, data),
            lambda: lgoa.plots.refinement_events(history),
            lambda: lgoa.plots.indicator_components(history),
            lambda: lgoa.plots.grid_history(history),
        ]
        for build in builders:
            figure = build()
            figure.canvas.draw()
            plt.close(figure)
