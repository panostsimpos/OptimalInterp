from __future__ import annotations
import jax.numpy as jnp
import jax
from jaxtyping import Float, Array
from functools import partial

def scan(prev_vals, _):
    (pt, Tn_minus, Tn) = prev_vals
    Tn_plus = 2 * pt * Tn - Tn_minus
    new_vals = (pt, Tn, Tn_plus)
    return new_vals, Tn_plus


@partial(jax.jit, static_argnums=[1,])
def slice(pt, max_order):
    if max_order == 0:
        return jnp.ones(())
    initial = jnp.pad(pt.reshape(1,), (1, 0), constant_values=1.)
    prev_vals = (pt, jnp.ones(()), pt)
    _, evals = jax.lax.scan(scan, prev_vals, length=max_order-1)
    all_evals = jnp.concat((initial, evals))
    return all_evals


def diff_scan(prev_vals, _):
    (pt, Tn_minus, Tn_minus_diff, Tn, Tn_diff) = prev_vals
    Tn_plus = 2 * pt * Tn - Tn_minus
    Tn_plus_diff = 2 * (Tn + pt * Tn_diff) - Tn_minus_diff
    new_vals = (pt, Tn, Tn_diff, Tn_plus, Tn_plus_diff)
    return new_vals, (Tn_plus, Tn_plus_diff)


@partial(jax.jit, static_argnums=[1,])
def diff_slice(pt, max_order):
    if max_order == 0:
        return (jnp.ones(()), jnp.zeros(()))
    initial_eval = jnp.pad(pt.reshape(1,), (1, 0), constant_values=1.)
    initial_diff = jnp.zeros(2).at[1].set(1.)
    prev_vals = (pt, jnp.ones(()), jnp.zeros(()), pt, jnp.ones(()))
    _, (evals, diffs) = jax.lax.scan(diff_scan, prev_vals, length=max_order-1)
    all_evals = jnp.concat((initial_eval, evals))
    all_diffs = jnp.concat((initial_diff, diffs))
    return all_evals, all_diffs


def diff2_scan(prev_vals, _):
    (pt, Tn_minus, Tn_minus_diff, Tn_minus_diff2, Tn, Tn_diff, Tn_diff2) = prev_vals
    Tn_plus = 2 * pt * Tn - Tn_minus
    Tn_plus_diff = 2 * (Tn + pt * Tn_diff) - Tn_minus_diff
    Tn_plus_diff2 = 2 * (2 * Tn_diff + pt * Tn_diff2) - Tn_minus_diff2
    new_vals = (
        pt, Tn, Tn_diff, Tn_diff2, Tn_plus, Tn_plus_diff, Tn_plus_diff2
    )
    return new_vals, (Tn_plus, Tn_plus_diff, Tn_plus_diff2)


@partial(jax.jit, static_argnums=[1,])
def diff2_slice(pt: Float[Array, ""], max_order: int):
    if max_order == 0:
        return (jnp.ones(()), jnp.zeros(()), jnp.zeros(()))
    initial_eval = jnp.pad(pt.reshape(1,), (1, 0), constant_values=1.)
    initial_diff1 = jnp.zeros(2).at[1].set(1.)
    initial_diff2 = jnp.zeros(2)
    zero, one = jnp.zeros(()), jnp.ones(())
    # pt, Tn_minus, Tn_minus_diff, Tn_minus_diff2, Tn, Tn_diff, Tn_diff2
    prev_vals = (
        pt, one, zero, zero, pt, one, zero
    )
    _, scan_result = jax.lax.scan(
        diff2_scan, prev_vals, length=max_order-1
    )
    evals, diff1s, diff2s = scan_result
    all_evals = jnp.concat((initial_eval, evals))
    all_diff1 = jnp.concat((initial_diff1, diff1s))
    all_diff2 = jnp.concat((initial_diff2, diff2s))
    return all_evals, all_diff1, all_diff2


__eval = jax.vmap(slice, in_axes=(0, None))
__diff1 = jax.vmap(diff_slice, in_axes=(0, None))
__diff2 = jax.vmap(diff2_slice, in_axes=(0, None))

def eval(pts: Float[Array, " N"], max_order: int):
    r"""
    Evaluate Chebyshev polynomials on a set of points
    --------------------------------------------------------------
    Inputs:
    -------
    pts: (N,) array
        Points to evaluate polynomial on
    max_order: int
        Order of polynomial to evaluate up to
    Returns:
    --------
    evaluations: (N, max_order + 1) array
        The basis evaluations from degree 0 through max_order
    """
    return __eval(pts, max_order)

def diff1(pts, max_order):
    r"""
    Evaluate Chebyshev polynomials and derivatives on a set of points
    --------------------------------------------------------------
    Inputs:
    -------
    pts: (N,) array
        Points to evaluate polynomial on
    max_order: int
        Order of polynomial to evaluate up to
    Returns:
    --------
    evaluations: (N, max_order + 1) array
        The basis evaluations from degree 0 through max_order
    diff1s: (N, max_order + 1) array
        The basis derivatives from degree 0 through max_order
    """
    return __diff1(pts, max_order)

def diff2(pts, max_order):
    r"""
    Evaluate Chebyshev polynomials and two derivatives on a set of points
    --------------------------------------------------------------
    Inputs:
    -------
    pts: (N,) array
        Points to evaluate polynomial on
    max_order: int
        Order of polynomial to evaluate up to
    Returns:
    --------
    evaluations: (N, max_order + 1) array
        The basis evaluations from degree 0 through max_order
    diff1s: (N, max_order + 1) array
        The basis derivatives from degree 0 through max_order
    diff2s: (N, max_order + 1) array
        The basis first 2 derivatives from degree 0 through max_order
    """
    return __diff2(pts, max_order)