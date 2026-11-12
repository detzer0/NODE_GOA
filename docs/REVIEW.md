# Review of the indicator correction (1.0.1 → 1.0.2)

This records what was checked in the 1.0.1 overlay patch, what was changed on
top of it, and what remains open. `docs/INDICATOR_UPDATE.md` is the original
1.0.1 report and is kept unedited.

## 1. The correction is right

The 1.0.1 patch replaces the previous `R_state * omega_adjoint` /
`R_adjoint * omega_state` products with the direct signed residual pairings.
That is the correct reading of the paper. Eq. (90) defines

    eta_k = |rho_adj| + |rho_grad| + |rho_state|,

a sum of absolute values of *integrated signed pairings*. The products of a
residual bound with a reconstruction norm belong to the bound (86), not to the
computable indicator. The previous behaviour came from the older ablation
notebooks that were used as the transcription source when the package was first
assembled; it was wrong, and this report does not dispute that.

Checked term by term against the proof:

| Paper | Implementation | Verdict |
|---|---|---|
| eq. (88) `rho_adj = -h_k ∫ (D₁N_F(x^{k-1},θ(s))* p^k, s Δx^k) ds` | `rho_adjoint_volume` | signs and indices match |
| eq. (88) `rho_state = h_k ∫ (N_F(x^{k-1},θ(s)), (1-s) Δp^k) ds` | `rho_state_volume` | matches |
| zero-based `x^{k-1} → states[k]`, `p^k → adjoints[k+1]` | same | matches |
| GL on [0,1] from order 8, doubled to 128 | `8 → 128` | matches |
| tolerances `atol 1e-9`, `rtol 1e-7` | same defaults | matches |
| `eta_K = ½ Σ eta_k` | `0.5 * sum(eta)` | matches |

Reproduction of the supplied fixtures: maximum relative difference in
`eta_k` of **6.5e-16**, with identical marked intervals at every bulk fraction
tested. The claim of notebook agreement holds.

## 2. Two defects found and fixed

**`marking.mark` could propose more intervals than exist.** The Dörfler count
is `searchsorted(cumsum(weights[order]), target) + 1`. The running sum is taken
in descending-weight order while `total_weight` is summed in interval order, so
at a bulk fraction at or near one the cumulative sum can fall one ulp short of
the target; `searchsorted` then returns an index past the end and the count
becomes `K + 1`. The count is now capped at `K`, since marking every interval
always satisfies the criterion. Guarded by
`test_dorfler_never_proposes_more_intervals_than_exist` over 200 random
indicator vectors.

This was latent before the patch; the new regression suite exposed it.

**A regression assertion demanded bit-exact float equality.**
`test_original_notebook_indicator_regression` compared two `MarkingResult`
dataclasses with `==` after having just accepted the underlying indicators at
`rtol=2e-12`. `==` on a frozen dataclass compares every float field exactly, so
derived sums differing in the last ulp failed the test. It now compares the
marked set, the budget decision and the proposed count exactly, and the derived
weights with a tolerance.

## 3. The control pullback now follows eq. (89)

The 1.0.1 patch deliberately left the control residual alone, so the nonlinear
pullback was still reconstructed linearly from its two endpoint values. The
paper states that this term is evaluated by Gauss–Legendre quadrature, like the
state and adjoint residuals. Writing out

    θ*[θ](s) - θ(s) = A_k (s² - s),

the term is

    rho_dyn^k = h_k ∫₀¹ <D₂N_F(x^{k-1}, θ(s))* p^k, A_k (s² - s)> ds,

which is now integrated on the same order-doubling rule and included in its
convergence test. The mass pairing stays closed form through `alpha_k, beta_k`
and the local mass matrix, and the derivative pairing still cancels
analytically — both as the paper prescribes.

The difference is not cosmetic. On a converged Swiss-roll hierarchy the
pullback term moved by up to a factor of two on individual intervals and by
about one per cent of `eta_k`, and it **changed sign** on intervals where the
term nearly cancels. On the stored `nonuniform` fixture, `eta_k` moves by up to
a factor of three.

`control_pullback_quadrature="endpoint_linear"` restores the previous path and
still reproduces the fixtures exactly; that is what the regression suite pins.
The default is `"gauss"`.

Verification: the Gauss path is checked against an independent
`scipy.integrate.quad` integration of the same scalar integrand, to **1.6e-18**
relative. `self_check()` now reports nine checks and runs the control
comparison in both modes.

## 4. Status

- 64 tests pass; `ruff` clean; both notebooks execute end to end.
- `self_check()` passes in float64 and float32.

## 5. Open question for the authors

The paper says the control pullback is integrated by Gauss–Legendre quadrature.
The ablation notebooks reconstruct it linearly. The published figures were
produced by the notebooks. So either the paper's sentence describes something
the runs did not do, or the notebooks drifted from the intended method. The
package now defaults to the paper, but **the published numbers correspond to
`control_pullback_quadrature="endpoint_linear"`**. Decide which is normative
before regenerating anything for the record.

Independently of that choice: results produced with the pre-1.0.1 norm-bound
indicator should be regenerated, since the marked intervals and hence the whole
subsequent hierarchy can differ.
