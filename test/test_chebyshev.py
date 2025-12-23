import jax
import jax.numpy as jnp
import pytest
from optimalinterp import chebyshev
import optimalinterp as oi
jax.config.update("jax_enable_x64", True)

FD_DELTA = 1e-8

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

def test_chebyshev_linear_basis():
    max_order, N_pts = 5, 101
    pts_m1p1 = jnp.linspace(-1, 1, N_pts)
    evals_m1p1, diff1_m1p1, diff2_m1p1 = chebyshev.diff2(pts_m1p1, max_order)
    diff1_m1p1 *= 2
    diff2_m1p1 *= 4
    basis = oi.basis.LinearBasis(max_order, 'chebyshev')
    pts_01 = jnp.linspace(0, 1, N_pts)
    evals_01, diff1_01, diff2_01 = basis.evaluate_basis_diff2(pts_01)
    assert evals_m1p1 == pytest.approx(evals_01, rel=1e-12, abs=1e-12)
    assert diff1_m1p1 == pytest.approx(diff1_01, rel=1e-12, abs=1e-12)
    assert diff2_m1p1 == pytest.approx(diff2_01, rel=1e-11, abs=1e-11)
    evals_fd_01, diff1_fd_01 = basis.evaluate_basis_diff(pts_01 + FD_DELTA)
    diff1_fd = (evals_fd_01 - evals_01)/FD_DELTA
    diff2_fd = (diff1_fd_01 - diff1_01)/FD_DELTA
    assert diff1_01 == pytest.approx(diff1_fd, rel=200*FD_DELTA, abs=jnp.sqrt(FD_DELTA))
    assert diff2_01 == pytest.approx(diff2_fd, rel=200*FD_DELTA, abs=jnp.sqrt(FD_DELTA))