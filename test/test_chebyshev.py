import jax
import jax.numpy as jnp
import pytest
from optimalinterp import chebyshev
jax.config.update("jax_enable_x64", True)

FD_DELTA = 1e-7

def test_chebyshev_eval():
    coeffs = [
        jnp.array([1,]),
        jnp.array([1, 0]),
        jnp.array([2, 0, -1]),
        jnp.array([4, 0, -3, 0]),
        jnp.array([8, 0, -8, 0, 1]),
        jnp.array([16, 0, -20, 0, 5, 0]),
        jnp.array([32, 0, -48, 0, 18, 0, -1]),
        jnp.array([64, 0, -112, 0, 56, 0, -7, 0]),
        jnp.array([128, 0, -256, 0, 160, 0, -32, 0, 1]),
        jnp.array([256, 0, -576, 0, 432, 0, -120, 0, 9, 0]),
        jnp.array([512, 0, -1280, 0, 1120, 0, -400, 0, 50, 0, -1]),
    ]
    diff1_coeffs = [jnp.polyder(c,1) for c in coeffs]
    diff2_coeffs = [jnp.polyder(c,2) for c in coeffs]
    pts = jnp.linspace(-1, 1, 1001)
    true_evals = jnp.column_stack([jnp.polyval(c, pts) for c in coeffs])
    true_diff1 = jnp.column_stack([jnp.polyval(c, pts) for c in diff1_coeffs])
    true_diff2 = jnp.column_stack([jnp.polyval(c, pts) for c in diff2_coeffs])
    pts_evals = chebyshev.eval(pts, len(coeffs)-1)
    assert pts_evals == pytest.approx(true_evals, rel=1e-12, abs=1e-12)
    pts_evals, pts_diff1 = chebyshev.diff1(pts, len(coeffs)-1)
    assert pts_evals == pytest.approx(true_evals, rel=1e-12, abs=1e-12)
    assert pts_diff1 == pytest.approx(true_diff1, rel=1e-12, abs=1e-12)
    pts_evals, pts_diff1, pts_diff2 = chebyshev.diff2(pts, len(coeffs)-1)
    assert pts_evals == pytest.approx(true_evals, rel=1e-12, abs=1e-12)
    assert pts_diff1 == pytest.approx(true_diff1, rel=1e-12, abs=1e-12)
    # Ease tolerance due to number of flops needed for diff2 TTRR eval
    assert pts_diff2 == pytest.approx(true_diff2, rel=1e-11, abs=1e-11)