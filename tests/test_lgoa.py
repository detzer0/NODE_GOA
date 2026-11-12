"""Fast regression tests. Run with ``pytest``; the whole file is CPU-only."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import pytest

import lgoa
from lgoa import fem, marking, plots, refinement, verification
from lgoa.backend import dtype


@pytest.fixture(scope="module")
def tiny_config():
    """A configuration small enough to run a full hierarchy in seconds."""
    return lgoa.load_preset(
        "swiss_roll",
        samples_per_class=60,
        hidden_dim=3,
        n_candidates=1,
        search_max_epochs=20,
        search_min_epochs=5,
        check_every=5,
        initial_depth=2,
        max_depth=5,
        adam_max_steps=120,
        adam_min_steps=40,
        adam_check_every=40,
        stationarity_tolerance=1.0e-1,
        bfgs_max_iterations=20,
        indicator_sup_samples=5,
    )


@pytest.fixture(scope="module")
def tiny_run(tiny_config):
    data = lgoa.load_dataset(tiny_config.problem)
    history = lgoa.run(tiny_config, data, progress=False)
    return tiny_config, data, history


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def test_self_check_passes():
    table = verification.self_check()
    assert table["passed"].all(), table


def test_dorfler_single_mark_equals_single_insertion():
    verification.check_single_interval_refinement_equivalence()


def test_riesz_matrix_is_symmetric_positive_definite():
    grid = jnp.asarray([0.0, 0.4, 0.9, 1.0, 2.5], dtype=dtype())
    dense = fem.dense_riesz(fem.assemble(grid))
    assert np.allclose(dense, dense.T)
    assert np.all(np.linalg.eigvalsh(dense) > 0.0)


def test_tridiagonal_apply_matches_dense():
    grid = jnp.asarray([0.0, 0.3, 1.1, 2.0], dtype=dtype())
    operators = fem.assemble(grid)
    X = jnp.asarray(np.random.default_rng(0).normal(size=(4, 6)), dtype=dtype())
    applied = np.asarray(fem.apply(operators.riesz, X))
    assert np.allclose(applied, fem.dense_riesz(operators) @ np.asarray(X))


def test_refinement_preserves_existing_nodes():
    grid = jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=dtype())
    params = {
        "W_in": jnp.zeros((2, 2), dtype=dtype()),
        "A": jnp.arange(4 * 4, dtype=dtype()).reshape(4, 2, 2),
        "b": jnp.arange(4 * 2, dtype=dtype()).reshape(4, 2),
        "W_out": jnp.zeros((2, 2), dtype=dtype()),
    }
    result = refinement.refine(params, grid, None, [0, 2], transfer_adam_state=False)
    new_grid = np.asarray(result.time_grid)
    assert new_grid.size == 6
    # Every original node survives unchanged.
    assert set(np.asarray(grid).tolist()) <= set(new_grid.tolist())
    assert np.allclose(np.asarray(result.params["b"])[0], np.asarray(params["b"])[0])


# --------------------------------------------------------------------------- #
# Marking
# --------------------------------------------------------------------------- #
def test_conservative_marks_the_largest_interval():
    eta = np.array([0.1, 0.9, 0.3])
    result = marking.mark(eta, "conservative", max_new_layers=3)
    assert result.marked_intervals == (1,)
    assert result.count == 1


def test_dorfler_meets_the_bulk_criterion():
    eta = np.array([0.4, 0.3, 0.2, 0.1])
    for fraction in (0.25, 0.5, 0.75, 1.0):
        result = marking.mark(
            eta, "dorfler", max_new_layers=4, dorfler_fraction=fraction
        )
        assert result.achieved_share >= fraction - 1.0e-12
        # Smallest such set: dropping the least important mark must undershoot.
        if result.count > 1:
            reduced = np.sum(eta[list(result.marked_intervals)]) - np.min(
                eta[list(result.marked_intervals)]
            )
            assert reduced / np.sum(eta) < fraction


def test_dorfler_never_proposes_more_intervals_than_exist():
    """At a bulk fraction of one the cumulative sum can fall an ulp short."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        eta = rng.random(rng.integers(2, 12)) * 10.0 ** rng.integers(-6, 3)
        result = marking.mark(
            eta, "dorfler", max_new_layers=eta.size, dorfler_fraction=1.0
        )
        assert result.proposed_count <= eta.size
        assert result.marked_intervals == tuple(range(eta.size))
        assert not result.budget_capped


def test_dorfler_respects_the_layer_budget():
    eta = np.array([0.25, 0.25, 0.25, 0.25])
    result = marking.mark(eta, "dorfler", max_new_layers=2, dorfler_fraction=0.99)
    assert result.count == 2
    assert result.budget_capped


def test_dorfler_with_one_mark_matches_conservative():
    eta = np.array([0.05, 0.80, 0.10, 0.05])
    conservative = marking.mark(eta, "conservative", max_new_layers=4)
    dorfler = marking.mark(eta, "dorfler", max_new_layers=4, dorfler_fraction=0.5)
    assert dorfler.count == 1
    assert dorfler.marked_intervals == conservative.marked_intervals


def test_unknown_marking_rule_is_rejected():
    for rule in ("greedy", "minimum", "random"):
        with pytest.raises(ValueError):
            marking.mark(np.array([1.0, 2.0]), rule, max_new_layers=1)
    with pytest.raises(ValueError):
        lgoa.load_preset("peaks", marking="random")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_presets_load():
    assert set(lgoa.available_presets()) >= {"peaks", "swiss_roll"}
    for name in ("peaks", "swiss_roll"):
        config = lgoa.load_preset(name)
        assert config.problem.name == name
        assert config.adaptive.initial_depth <= config.adaptive.max_depth


def test_overrides_reach_the_right_section():
    config = lgoa.load_preset(
        "peaks", marking="dorfler", stationarity_tolerance=1.0e-4, hidden_dim=7
    )
    assert config.adaptive.marking == "dorfler"
    assert config.solver.stationarity_tolerance == 1.0e-4
    assert config.problem.hidden_dim == 7


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="Unknown configuration parameter"):
        lgoa.load_preset("peaks", not_a_parameter=1)


def test_invalid_depth_budget_is_rejected():
    with pytest.raises(ValueError):
        lgoa.load_preset("peaks", initial_depth=20, max_depth=5)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_hierarchy_reaches_the_layer_budget(tiny_run):
    config, _, history = tiny_run
    assert history.final_phase.depth == config.adaptive.max_depth
    assert history.inserted_layers == config.adaptive.layer_budget
    assert len(history.phases) == len(history.events) + 1


def test_grids_are_nested_and_increasing(tiny_run):
    _, _, history = tiny_run
    for earlier, later in zip(history.phases[:-1], history.phases[1:], strict=True):
        assert later.depth > earlier.depth
        assert set(np.round(earlier.grid, 12)) <= set(np.round(later.grid, 12))


def test_indicators_are_finite_and_nonnegative(tiny_run):
    _, _, history = tiny_run
    for phase in history.phases:
        assert phase.eta.size == phase.depth
        assert np.all(np.isfinite(phase.eta))
        assert np.all(phase.eta >= 0.0)


def test_control_pullback_defaults_to_the_paper_quadrature():
    assert lgoa.load_preset("peaks").adaptive.control_pullback_quadrature == "gauss"
    with pytest.raises(ValueError, match="control_pullback_quadrature"):
        lgoa.load_preset("peaks", control_pullback_quadrature="trapezoid")


def test_gauss_and_endpoint_linear_pullbacks_differ_but_both_converge(tiny_run):
    """Eq. (89) integrates the nonlinear pullback; the legacy path interpolates it."""
    from lgoa.indicators import compute_local_indicators

    _, data, history = tiny_run
    phase = history.final_phase
    shared = dict(
        params=phase.params,
        time_grid=phase.result.time_grid,
        X=data.X_indicator,
        y=data.y_indicator,
        solution_fn=phase.result.control.solution_fn,
        reg_lambda=history.config.solver.reg_lambda,
    )
    gauss = compute_local_indicators(**shared, control_pullback_quadrature="gauss")
    linear = compute_local_indicators(
        **shared, control_pullback_quadrature="endpoint_linear"
    )

    assert gauss.direct_quadrature_converged
    for result in (gauss, linear):
        assert np.all(np.isfinite(result.total))
        assert np.all(result.total >= 0.0)
    # The two agree on the mass and stiffness pairings, which are closed form.
    np.testing.assert_allclose(
        gauss.control_mass_signed, linear.control_mass_signed, rtol=1e-12
    )
    np.testing.assert_allclose(
        gauss.control_stiffness_signed, linear.control_stiffness_signed, rtol=1e-12
    )


def test_gauss_pullback_is_insensitive_to_the_starting_order(tiny_run):
    """Order doubling must land on the same value from a different start."""
    from lgoa.indicators import compute_local_indicators

    _, data, history = tiny_run
    phase = history.final_phase
    shared = dict(
        params=phase.params,
        time_grid=phase.result.time_grid,
        X=data.X_indicator,
        y=data.y_indicator,
        solution_fn=phase.result.control.solution_fn,
        reg_lambda=history.config.solver.reg_lambda,
    )
    coarse = compute_local_indicators(**shared, direct_quadrature_initial_order=4)
    fine = compute_local_indicators(**shared, direct_quadrature_initial_order=32)
    np.testing.assert_allclose(coarse.total, fine.total, rtol=1e-8, atol=1e-12)


def test_indicator_components_sum_to_the_total(tiny_run):
    _, _, history = tiny_run
    for phase in history.phases:
        eta = phase.indicators
        assert np.allclose(eta.total, eta.state + eta.adjoint + eta.control)
        assert np.isclose(
            eta.global_practical_indicator,
            eta.global_state_indicator
            + eta.global_adjoint_indicator
            + eta.global_control_indicator,
        )


def test_stiffness_pairing_cancels_analytically(tiny_run):
    """The derivative part of the control residual must vanish to roundoff."""
    _, _, history = tiny_run
    for phase in history.phases:
        scale = max(phase.indicators.global_practical_indicator, 1.0)
        assert phase.indicators.max_abs_control_stiffness_term < 1.0e-10 * scale


def test_every_indicator_field_is_consumed():
    """Guard against the payload growing fields nothing reads again."""
    import dataclasses
    from pathlib import Path

    package = Path(lgoa.__file__).parent
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in package.rglob("*.py")
        if path.name != "indicators.py"
    }
    unused = [
        field.name
        for field in dataclasses.fields(lgoa.LocalIndicators)
        if not any(
            f".{field.name}" in text or f'"{field.name}"' in text
            for text in sources.values()
        )
    ]
    assert not unused, f"LocalIndicators fields nothing consumes: {unused}"


def test_phase_and_event_tables_are_well_formed(tiny_run):
    _, _, history = tiny_run
    table = history.phase_table
    assert len(table) == len(history.phases)
    assert {"depth", "objective", "validation_loss", "stationarity_norm"} <= set(
        table.columns
    )
    assert len(history.event_table) == len(history.events)


def test_marking_rules_share_a_calibration(tiny_config):
    data = lgoa.load_dataset(tiny_config.problem)
    calibration = None
    for rule in ("conservative", "dorfler"):
        config = tiny_config.with_overrides(marking=rule)
        history = lgoa.run(config, data, calibration, progress=False)
        calibration = calibration or history.calibration
        assert history.final_phase.depth == tiny_config.adaptive.max_depth, rule
        # The paired comparison must start from identical endpoint maps.
        assert history.calibration is calibration


def test_figures_render(tiny_run, tmp_path):
    _, data, history = tiny_run
    written = plots.save_all(history, data, tmp_path, formats=("png",))
    assert len(written) == 7
    assert all(path.stat().st_size > 0 for path in written)


def test_csv_export(tiny_run, tmp_path):
    _, _, history = tiny_run
    written = history.to_csv(tmp_path)
    assert {path.name.split("_")[-1] for path in written} >= {"phases.csv"}
