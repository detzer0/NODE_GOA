"""Numerical backend configuration.

JAX must be told about 64-bit precision *before* any array is created, so
:func:`configure` is called by :func:`lgoa.config.load_preset` and by the
notebooks as the very first step.  Every other module asks :func:`dtype` for
the current floating-point type instead of caching a module-level constant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["configure", "dtype", "x64_enabled", "backend_report"]


def configure(use_x64: bool = True) -> None:
    """Enable or disable double precision for the whole session."""
    jax.config.update("jax_enable_x64", bool(use_x64))


def x64_enabled() -> bool:
    return bool(jax.config.jax_enable_x64)


def dtype():
    """Floating-point type matching the current precision setting."""
    return jnp.float64 if x64_enabled() else jnp.float32


def backend_report() -> str:
    """One-line description of the active JAX backend, for notebook output."""
    return (
        f"JAX {jax.__version__} | backend: {jax.default_backend()} "
        f"| dtype: {dtype().__name__}"
    )
