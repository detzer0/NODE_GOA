# Layerwise goal-oriented adaptivity for neural ODEs

Reference implementation of the adaptive solver introduced in

> M. Hintermüller, M. Hinze and D. Korolev
> **Layerwise goal-oriented adaptivity for neural ODEs: an optimal control perspective.**


A neural ODE is treated as an optimal control problem in time. The depth of the
network is a temporal discretization, so *where to add a layer* becomes a mesh
refinement question — answered here by a computable goal-oriented error
indicator rather than by a uniform deepening of the network.

Two notebooks run the two examples of the paper end to end and produce the
figures:

| Notebook | Problem | Section |
|---|---|---|
| [`notebooks/01_swiss_roll.ipynb`](notebooks/01_swiss_roll.ipynb) | two concentric spirals, binary | 8.1 |
| [`notebooks/02_peaks.ipynb`](notebooks/02_peaks.ipynb) | Peaks function, five classes | 8.2 |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/detzer0/NODE_GOA/blob/main/notebooks/01_swiss_roll.ipynb)
Swiss roll &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/detzer0/NODE_GOA/blob/main/notebooks/02_peaks.ipynb)
Peaks

---

## Quick start

```bash
git clone https://github.com/detzer0/NODE_GOA.git
cd NODE_GOA
pip install -e .
```

```python
import lgoa
from lgoa import plots

cfg = lgoa.load_preset("swiss_roll", marking="dorfler", dorfler_fraction=0.5)
data = lgoa.load_dataset(cfg.problem)
history = lgoa.run(cfg, data)

print(history.summary())
plots.learning_curves(history)
plots.refinement_events(history)
```

From the shell:

```bash
lgoa-run peaks --marking dorfler --dorfler-fraction 0.5 --output runs/peaks
```

In Colab nothing needs cloning; the first notebook cell installs the package
straight from GitHub.

## What the method does

On a temporal partition $\mathcal{K}$ the control
$\theta_\tau$ is continuous and piecewise linear, the state is advanced by
explicit Euler with midpoint controls, and the discrete adjoint supplies the
exact derivative of the reduced objective. Gradients are taken in the discrete
$H^1$ metric: with $B^\tau = M^\tau + A^\tau$ the metric gradient solves
$B^\tau G_\tau = R_\tau$, and

$$ s_\tau(\Theta_\tau) \;=\; \lVert g_\tau \rVert_{H^1} \;=\; \bigl[\operatorname{trace}(G_\tau^\top B^\tau G_\tau)\bigr]^{1/2} $$

is the stationarity measure that drives both the inner solver and the outer
loop. The time matrices are tridiagonal, so each gradient solve is $O(K)$ per
control component.

**Algorithm 1 — fixed depth.** Full-batch $H^1$-Adam provides a robust warm
start. Once $s_\tau \le \kappa_s \varepsilon_{\mathrm{stat}}$, the solver
switches to BFGS in the coordinates $\hat\Theta_\tau = (L^\tau)^\top
\Theta_\tau$ given by the Cholesky factor $B^\tau = L^\tau (L^\tau)^\top$. That
change of variables is an isometry, so the Euclidean gradient norm in those
coordinates *is* the $H^1$ stationarity measure and a stock SciPy
implementation respects the $H^1$ geometry.

**Algorithm 2 — adaptive depth.** From the accepted solution the local
indicators

$$ \eta_k = \left|\widetilde\rho_{\mathrm{adj}}^k\right|
           + \left|\widetilde\rho_{\mathrm{grad}}^k\right|
           + \left|\widetilde\rho_{\mathrm{state}}^k\right| $$

are evaluated, a marking rule selects $\mathcal{M}_K$, each marked interval is
bisected at its midpoint, and the optimized control — together with the Adam
moments — is carried to the refined grid by midpoint prolongation. 

Two marking rules are shipped: `conservative` (largest indicator, one layer per
round) and `dorfler` (smallest set meeting a bulk fraction $\vartheta$, several
layers per round). With $\lvert\mathcal{M}_K\rvert = 1$ the Dörfler path reduces
to the conservative one bit for bit, which the test suite checks.

## Repository layout

```
src/lgoa/
├── backend.py        64-bit precision switch; must run before any array is made
├── config.py         frozen dataclasses, YAML presets, flat override routing
├── configs/*.yaml    the hyperparameter bulk for each example
├── datasets.py       Swiss roll and Peaks, with the stratified split and scaler
├── fem.py            CG(1) mass/stiffness/Riesz bands, Cholesky factor
├── neural_ode.py     forward pass, discrete adjoint, exact control derivative
├── optimizers.py     Adam in the H^1 metric
├── initialization.py one-time endpoint-map calibration, then frozen
├── indicators.py     eta_k, its three components and the global aggregates
├── marking.py        conservative and Dörfler marking
├── refinement.py     bisection and midpoint prolongation of control + moments
├── solver.py         Algorithm 1
├── adaptive.py       Algorithm 2
├── history.py        phase and event records, tables, CSV export
├── plots.py          the figures of Section 8
├── verification.py   implementation self-checks
└── cli.py            lgoa-run
notebooks/            the two Colab notebooks (generated, see scripts/)
tests/                pytest suite, CPU-only, about a minute
```

The map from paper to code:

| Paper | Code |
|---|---|
| eq. (61) discrete system | `neural_ode.objective_and_gradients` |
| eq. (63) stationarity measure | `metrics["stationarity_norm"]` |
| eqs. (68)–(70) whitening | `fem.cholesky_factor`, `solver._run_bfgs_polish` |
| Algorithm 1 | `solver.solve_fixed_depth` |
| eq. (90) indicator $\eta_k$ | `indicators.compute_local_indicators` |
| eq. (91) conservative marking | `marking.mark(..., "conservative")` |
| eqs. (92)–(93) Dörfler marking | `marking.mark(..., "dorfler")` |
| eq. (94) refinement | `refinement.refine` |
| Algorithm 2 | `adaptive.run` |

## Configuration

Each example has one YAML preset holding the published hyperparameters:
[`src/lgoa/configs/swiss_roll.yaml`](src/lgoa/configs/swiss_roll.yaml) and
[`src/lgoa/configs/peaks.yaml`](src/lgoa/configs/peaks.yaml). A notebook
overrides only what it studies; overrides are flat and get routed to whichever
section declares the field:

```python
cfg = lgoa.load_preset(
    "peaks",
    marking="dorfler",           # -> AdaptiveConfig
    dorfler_fraction=0.5,        # -> AdaptiveConfig
    stationarity_tolerance=1e-3, # -> SolverConfig
    reg_lambda=1e-2,             # -> SolverConfig
    hidden_dim=6,                # -> ProblemConfig
)
print(cfg.summary())
```

An unknown name raises rather than being silently ignored, and every section
validates itself on construction. `lgoa.load_yaml(path)` reads a config of your
own; `lgoa.config_path(name)` shows where a preset lives.

Published settings for reference: Swiss roll uses $d = 4$, $T = 2.5$, $K: 2 \to
12$, $\varepsilon_{\mathrm{stat}} = 10^{-3}$, $\kappa_s = 5$; Peaks uses
$d = 5$, $T = 5$, $K: 2 \to 15$, $\varepsilon_{\mathrm{stat}} = 5 \times
10^{-3}$, $\kappa_s = 1.5$. Both notebooks default to a reduced profile that
finishes in a couple of minutes on a Colab CPU; set `QUICK = False` for the
paper setup.

## Figures

Every function in `lgoa.plots` takes a `History` and returns a
`matplotlib.figure.Figure`, so the same call works in a notebook, a script and
a test.

| Function | Shows | Paper |
|---|---|---|
| `dataset` | the raw data by class | Fig. 2(A), 2(C) |
| `decision_boundary` | predicted class regions | Fig. 2(B), 2(D) |
| `learning_curves` | stationarity, objective, losses, accuracy vs $K$ | Fig. 3 |
| `trajectories` | state, control and adjoint in time | Fig. 4 |
| `refinement_events` | $\eta_k$ before / after insertion / re-optimized | Figs. 5, 7, 9 |
| `indicator_components` | the three contributions to $\eta_K$ | Fig. 6(A) |
| `grid_history` | every temporal grid of the hierarchy | Figs. 8, 10, 11 |

`plots.save_all(history, data, directory)` writes all of them as PDF and PNG;
`history.to_csv(directory)` writes the phase table, the event table and the
final grid and indicators.

## Scope

This repository provides **the solver and the two example setups for a single
seed**, with the two marking rules of the paper's algorithms. It is not the
harness for the published ablation, which averages ten model initializations per
configuration and also runs the minimum-indicator and random-insertion baselines
of Table 2 — that is a cluster job, not a notebook. What a single seed does show
is the mechanism: how the indicator responds to insertion and re-optimization,
how Dörfler marking inserts several layers per round, and how the resulting
grids differ. Section 8 of either notebook compares conservative against Dörfler
marking at several bulk fractions from one shared calibration, which keeps the
comparison paired.

### Indicator evaluation

The state and adjoint contributions use the **direct reconstructed residual
pairings**, matching the supplied Swiss-roll and Peaks ablation notebooks.
On $I_k$, write $h_k=t_k-t_{k-1}$ and
$\theta_\tau(s)=(1-s)\theta_\tau^{k-1}+s\theta_\tau^k$. The signed terms are

$$
\widetilde\rho_{\mathrm{adj}}^k
= -h_k\int_0^1
\left\langle D_1N_F(x_\tau^{k-1},\theta_\tau(s))^*p_\tau^k,
 s(x_\tau^k-x_\tau^{k-1})\right\rangle\,ds,
$$

$$
\widetilde\rho_{\mathrm{state}}^k
= h_k\int_0^1
\left\langle N_F(x_\tau^{k-1},\theta_\tau(s)),
 (1-s)(p_\tau^k-p_\tau^{k-1})\right\rangle\,ds.
$$

The reconstructed one-sided jump pairings are zero. Absolute values are taken
**after** the signed integrals have been evaluated; no residual-norm or sampled
supremum products enter the estimator. The global value is
$\eta_K=\tfrac12\sum_k\eta_k$.

The state and adjoint integrals use Gauss--Legendre orders $8,16,32,64,128$.
Order doubling stops when **every local signed state and adjoint integral**
agrees with its previous-order value within
$\mathrm{atol}+\mathrm{rtol}\,|\rho_{\mathrm{new}}|$, with
$\mathrm{atol}=10^{-9}$ and $\mathrm{rtol}=10^{-7}$.
As in the reference notebooks, the last estimate is retained silently when the
maximum order is reached without convergence. This is explicitly recorded as
`direct_quadrature_converged=False`; the consecutive-order difference is an
empirical estimate, not a certified quadrature error bound.

The settings can be overridden through `load_preset`, `with_overrides`, YAML,
or the CLI's `--set` option:

```python
cfg = lgoa.load_preset(
    "swiss_roll",
    direct_quadrature_initial_order=8,
    direct_quadrature_max_order=128,
    direct_quadrature_rtol=1e-7,
    direct_quadrature_atol=1e-9,
)
```

Old YAML files remain compatible. `indicator_sup_samples` (and the function
argument `supremum_samples`) is accepted but no longer affects the indicator.

The **control contribution** follows eq. (89). Its mass pairing is integrated in
closed form through $\alpha_k,\beta_k$ and the local mass matrix, and its
derivative pairing cancels analytically because the quadratic reconstruction
matches both endpoints. The remaining nonlinear pullback term,

$$
\rho_{\mathrm{dyn}}^k = h_k\int_0^1
\left\langle D_2N_F(x_\tau^{k-1},\theta_\tau(s))^*p_\tau^k,\;
A_k\,(s^2-s)\right\rangle ds ,
$$

uses the same Gauss--Legendre order-doubling rule as the state and adjoint
residuals, which is what the paper specifies. Setting

```python
cfg = lgoa.load_preset("swiss_roll", control_pullback_quadrature="endpoint_linear")
```

reconstructs that pullback linearly from its two endpoint values instead,
reproducing the earlier ablation notebooks exactly; that path is what
`tests/test_indicator_regression.py` pins against the stored fixtures. The two
differ: on a converged Swiss-roll hierarchy the pullback term moved by up to a
factor of two on individual intervals and by about one per cent of $\eta_k$,
and it changed sign on intervals where the term nearly cancels.

`LocalIndicators` exposes the signed components and the quadrature order,
consecutive-order difference, and convergence flag. These diagnostics also
appear in phase tables and CSV exports, including the immediate
post-insertion evaluation. The existing `global_practical_indicator` field is
retained for compatibility; `global_direct_indicator` is an equivalent alias.

**Results made with the old norm-bound indicator should be regenerated.**
Correcting the estimator can change the marked intervals and therefore the
subsequent adaptive hierarchy.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`lgoa.self_check()` runs eight numerical checks: the original six checks
(tridiagonal solve, control and endpoint derivatives, frozen endpoints,
polynomial control pairing, and single-interval refinement), plus independent
scalar-quadrature checks of the direct state/adjoint and retained control
residuals.

The added `tests/test_indicator_regression.py` compares the corrected package
against reference values generated from **both** supplied notebooks. It also
checks cancellation, signs, quadrature fallback, backward-compatible YAML,
configuration forwarding at both estimator call sites, short adaptive runs for
both problems and marking rules, CSV exports, and plotting compatibility.
Fixture provenance and the numerical comparison are recorded in
`docs/notebook_agreement.json`.

The notebooks are generated so the two stay in step:

```bash
python scripts/build_notebooks.py
```

## Requirements

Python ≥ 3.10 with JAX, NumPy, SciPy, pandas, Matplotlib, scikit-learn and
PyYAML; all are preinstalled on Colab. Double precision is enabled through
`jax_enable_x64` before any array is created — `lgoa.load_preset` does this, so
call it before building data or parameters. The solver is CPU-friendly: the
temporal operators are tridiagonal and the networks are small.

## Citation

```bibtex
@article{HintermuellerKorolev2026,
  author  = {Hintermüller, Michael and Hinze, Michael and Korolev, Denis},
  title   = {Layerwise goal-oriented adaptivity for neural {ODE}s:
             an optimal control perspective},
  journal = {Journal of Numerical Mathematics},
  year    = {2026},
  doi     = {10.1515/jnma-2026-0008}
}
```

## License

MIT — see [LICENSE](LICENSE).
