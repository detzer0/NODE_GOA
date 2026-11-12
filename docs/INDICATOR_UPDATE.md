# NODE_GOA: corrected direct residual indicator (version 1.0.1)

## Basis and scope

This patch is based on the uploaded package sources, with `config(2).py`
restored to `src/lgoa/config.py` and `__init__(1).py` restored to
`src/lgoa/__init__.py`.

The reference implementations are cell 12 (zero-based) of:

- `Resnet_Swiss_Roll_Doerfler_(ablation,_quadrature).ipynb`
- `Resnet_Peaks_Doerfler_(exact,_ablation) (1).ipynb`

Notebook hashes and the complete regression comparison are in
`docs/notebook_agreement.json`. The original notebook files are not modified
or bundled. Their training/ablation cells were not executed.

## Mathematical correction

The old package used `R_state * omega_adjoint` and
`R_adjoint * omega_state`, including sampled supremum and norm-weighted jump
contributions. Those are removed from the indicator calculation.

For interval `k` (zero-based in Python), the corrected code uses `states[k]`
and `adjoints[k+1]` as the constant DG(0) representatives, interpolates the
control linearly at quadrature nodes, and computes

```text
rho_adjoint[k] = -h[k] * integral_0^1 <D_x F(X[k], theta(s))^* P[k+1],
                                      s * (X[k+1] - X[k])> ds
rho_state[k]   =  h[k] * integral_0^1 <F(X[k], theta(s)),
                                      (1-s) * (P[k+1] - P[k])> ds
eta[k] = abs(rho_adjoint[k]) + abs(rho_control[k]) + abs(rho_state[k])
global_indicator = 0.5 * sum(eta)
```

Inner products include all samples and state coordinates. The discrete
adjoint already carries the mean-loss normalization; no second division by
the number of samples is introduced. The one-sided reconstruction-error
traces used by the jump pairings vanish, so those jump terms are exactly zero.

**The control residual is unchanged**, including the quadratic reconstruction,
endpoint-linear pullback and analytic mass/stiffness/dynamics pairings.
No change to quadrature of the nonlinear control pullback is made.

## Quadrature and compatibility

Defaults match both notebooks:

```yaml
adaptive:
  direct_quadrature_initial_order: 8
  direct_quadrature_max_order: 128
  direct_quadrature_rtol: 1.0e-7
  direct_quadrature_atol: 1.0e-9
```

These keys are optional because their defaults are declared in `AdaptiveConfig`.
No existing YAML file has to change. They also work as flat `load_preset` or
`with_overrides` arguments and as CLI `--set` arguments.

Every state and adjoint integral must satisfy
`abs(high-low) <= atol + rtol*abs(high)`. The quadrature order is shared across
all intervals, as in the notebooks. On reaching the cap, the final values are
retained **silently**, and `direct_quadrature_converged` records whether the
comparison passed. This intentionally retains the notebooks' fallback policy.
The error estimate is the maximum difference of consecutive rules, not a
certified integration bound.

`indicator_sup_samples` in old YAML files, and `supremum_samples` in direct
function calls, remain accepted but no longer influence the estimator.
Existing `LocalIndicators` component/aggregate fields remain valid;
`global_direct_indicator` aliases `global_practical_indicator`.

Both adaptive-loop evaluation sites now forward the quadrature settings:
post-training and immediately after insertion. Phase/event tables and final
indicator CSVs include the applicable quadrature diagnostics and signed terms.
The optimizer, forward/adjoint solver, datasets, marking and refinement
algorithms are unchanged, as are the plotting functions.

## Files to replace

| Repository path | Purpose |
| --- | --- |
| `src/lgoa/indicators.py` | Direct signed residuals and notebook-matched quadrature |
| `src/lgoa/config.py` | Quadrature defaults/validation; old YAML compatibility |
| `src/lgoa/adaptive.py` | Both indicator call sites use the new settings |
| `src/lgoa/history.py` | Signed residual and convergence diagnostics in tables/CSVs |
| `src/lgoa/verification.py` | Independent scalar-integral checks for all three components |
| `src/lgoa/__init__.py` | Version `1.0.1` |
| `pyproject.toml` | Version `1.0.1` |
| `CITATION.cff` | Version `1.0.1` |
| `README.md` | Corrected indicator description and usage |

New files: `tests/test_indicator_regression.py`, six small regression fixtures
under `tests/fixtures/`, and this report plus `docs/notebook_agreement.json`.
Existing presets, notebooks and their generation script were not supplied as
package files and are not overwritten. Their calls to the package continue
to work through the retained API. Their prose is not covered by this patch.

## Validation performed

- The original numerical function definitions were loaded from both supplied
  notebooks (cells 8, 10 and 12 only). No notebook optimization cells ran.
- Six cases were compared with each notebook: a single interval, a uniform
  grid, a nonuniform grid, constant controls, strongly varying controls, and a
  zero vector field. All 24 compared component/diagnostic fields agreed with
  **zero observed absolute difference** in these 12 comparisons.
- The control component was also compared with the original uploaded package
  and was bit-for-bit unchanged in all six cases.
- All **32 focused regression tests** and **4 short adaptive-hierarchy tests**
  passed. The tests were executed in two groups.
- Short hierarchy tests covered Swiss roll and Peaks, each with conservative
  and Dörfler marking, from depth 2 to depth 4. These are integration tests,
  not reproductions of the full training experiments.
- All seven existing plotting functions rendered successfully; CSV exports
  and settings forwarding at both indicator sites were checked.
- All eight `lgoa.self_check()` checks passed in both float64 and float32.

Environment: Python 3.13.5, JAX 0.9.0.1, CPU. Exact dependency versions are
recorded in the JSON report. Full paper ablations, GPU execution, and any
pre-existing repository test files not uploaded here were not run.

## Apply locally and upload

The ZIP is an **overlay patch**, not a complete repository. Merge its contents
into the existing project root (the directory containing `pyproject.toml`),
allowing these files to be replaced. Do not delete the original project,
`src/lgoa/configs/`, other tests, or notebooks. Keep a backup of uncommitted
local edits before replacing files.

In the VS Code terminal, from that project root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_indicator_regression.py -q
python -c "import lgoa; print(lgoa.__version__); print(lgoa.__file__)"
```

The version should be `1.0.1`, and the printed file should be inside the local
project's `src/lgoa/` directory. Restart a notebook kernel that already imported
the previous package before rerunning its cells.

Review the changes before uploading:

```powershell
git status
git diff --stat
git add src/lgoa tests docs README.md pyproject.toml CITATION.cff
git diff --cached --stat
git commit -m "Replace old indicator with direct reconstructed residuals"
git push
```

An existing Colab session that already imported `lgoa` will not fetch this
update through a `try: import lgoa` preamble. After the GitHub push, reinstall
in that existing environment:

```python
%pip install -q --upgrade --no-deps "git+https://github.com/detzer0/NODE_GOA.git@main"
```

Then restart the runtime before importing `lgoa` again. A fresh environment
still needs the usual installation with dependencies (without `--no-deps`).

**Regenerate runs and figures made with the old indicator.** Changing the
indicator can change the marked intervals and the entire subsequent
optimization/refinement path; old results are not corrected by replacing
files or relabeling figures.
