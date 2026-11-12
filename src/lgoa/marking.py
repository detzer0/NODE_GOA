r"""Marking rules: which intervals receive a new temporal layer.

Given the local indicators :math:`\{\eta_k\}_{k=1}^{K}`, a rule returns the
marked set :math:`\mathcal{M}_K \subset \{1, \dots, K\}`.

``conservative``
    :math:`\mathcal{M}_K = \{k^\star\}` with
    :math:`k^\star = \arg\max_k \eta_k`; one layer per round (eq. (91)).
``dorfler``
    the smallest set with
    :math:`\sum_{k \in \mathcal{M}_K} \eta_k \ge \vartheta \sum_{k} \eta_k`
    (eq. (92)); several layers per round.

The achieved bulk fraction :math:`\beta^\vartheta_K` of eq. (93) is reported for
both rules, and the marked set is truncated when it would exceed the remaining
layer budget :math:`K_{\max} - K`.  With :math:`\lvert \mathcal{M}_K \rvert = 1`
the Dörfler path reduces exactly to the conservative one, which
:mod:`lgoa.verification` checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MarkingResult", "mark", "describe"]


@dataclass(frozen=True)
class MarkingResult:
    """The marked set together with its indicator bookkeeping."""

    rule: str
    marked_intervals: tuple[int, ...]
    proposed_count: int
    budget_capped: bool
    target_share: float | None
    achieved_share: float
    selected_weight: float
    total_weight: float

    @property
    def count(self) -> int:
        return len(self.marked_intervals)


def _nonnegative_weights(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def mark(
    indicator_values,
    rule: str,
    *,
    max_new_layers: int,
    dorfler_fraction: float = 0.5,
) -> MarkingResult:
    """Apply a marking rule to one vector of local indicators."""
    values = np.asarray(indicator_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Local indicator values must be a nonempty vector.")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite local indicator encountered.")
    if max_new_layers < 1:
        raise ValueError("max_new_layers must be positive.")

    weights = _nonnegative_weights(values)
    total_weight = float(np.sum(weights))
    target_share: float | None = None

    if rule == "conservative":
        marked = np.asarray([int(np.argmax(values))], dtype=int)
        proposed_count = 1
    elif rule == "dorfler":
        target_share = float(dorfler_fraction)
        # Stable sorting keeps ties deterministic by retaining interval order.
        order = np.argsort(-weights, kind="mergesort")
        if total_weight > 0.0:
            target = target_share * total_weight
            proposed_count = int(
                np.searchsorted(np.cumsum(weights[order]), target, side="left") + 1
            )
            # The running sum is taken in descending order while total_weight
            # is summed in interval order, so for a bulk fraction at or near
            # one the cumulative sum can fall an ulp short of the target and
            # searchsorted then indexes past the end. Marking every interval
            # always satisfies the criterion, so cap the count at K.
            proposed_count = min(proposed_count, int(values.size))
        else:
            # Keep the rule defined if every indicator vanishes.
            proposed_count = 1
        selected_count = min(proposed_count, int(max_new_layers))
        marked = np.sort(order[:selected_count]).astype(int)
    else:
        raise ValueError(f"Unknown marking rule: {rule!r}")

    selected_weight = float(np.sum(weights[marked]))
    achieved_share = selected_weight / total_weight if total_weight > 0.0 else 0.0

    return MarkingResult(
        rule=rule,
        marked_intervals=tuple(int(index) for index in marked),
        proposed_count=int(proposed_count),
        budget_capped=bool(proposed_count > int(max_new_layers)),
        target_share=target_share,
        achieved_share=float(achieved_share),
        selected_weight=selected_weight,
        total_weight=total_weight,
    )


def describe(rule: str, dorfler_fraction: float | None = None) -> str:
    """Human-readable rule name for figure titles and log lines."""
    if rule == "conservative":
        return "conservative marking (maximum-indicator insertion)"
    if rule == "dorfler":
        if dorfler_fraction is None:
            return "Dörfler marking"
        return f"Dörfler marking ({dorfler_fraction:.0%} bulk)"
    raise ValueError(f"Unknown marking rule: {rule!r}")
