"""Small pytree utilities shared by the optimizers and the adaptive loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["zeros_like", "copy_tree", "scale_tree", "count_parameters"]


def zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def copy_tree(tree):
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def scale_tree(tree, scale):
    return jax.tree_util.tree_map(lambda value: scale * value, tree)


def count_parameters(tree) -> int:
    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(tree))
    )
