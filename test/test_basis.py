import jax
import jax.numpy as jnp
import pytest
import optimalinterp as oi
import numpy as np

jax.config.update("jax_enable_x64", True)
FD_DELTA = 1e-6


# Check evaluation of sinc spline
def test_sinc_eval():
    sinc = oi.basis.SincSpline()
    pos_integers = jnp.arange(6)
    int_evals = sinc.evaluate(pos_integers)
    assert int_evals[0].item() == pytest.approx(1.0, rel=1e-15)
    assert np.array(int_evals[1:]) == pytest.approx(0.0, 1e-15)
    HALF_VAL = 0.6366197723675814
    half_evals = sinc.evaluate(jnp.array([-0.5, 0.5]))
    assert np.array(half_evals) == pytest.approx(HALF_VAL, rel=1e-14)


# Check all derivatives of sinc spline
def test_sinc_diff():
    sinc = oi.basis.SincSpline()
    pts = jnp.linspace(-2, 2)
    pts_fd_pos = pts + FD_DELTA
    pts_fd_neg = pts - FD_DELTA
    evals = sinc.evaluate(pts)
    evals_fd_pos = sinc.evaluate(pts_fd_pos)
    evals_fd_neg = sinc.evaluate(pts_fd_neg)
    diffs_fd = (evals_fd_pos - evals_fd_neg) / (2 * FD_DELTA)
    evals_diff, diffs = sinc.evaluate_diff(pts)
    assert np.all(np.array(evals) == np.array(evals_diff))
    assert np.array(diffs) == pytest.approx(np.array(diffs_fd), rel=10 * FD_DELTA)
    evals_diff2, diffs_diff2, diff2 = sinc.evaluate_diff2(pts)
    assert np.all(np.array(evals) == np.array(evals_diff2))
    assert np.all(np.array(diffs) == np.array(diffs_diff2))
    _, diffs_fd_pos = sinc.evaluate_diff(pts_fd_pos)
    _, diffs_fd_neg = sinc.evaluate_diff(pts_fd_neg)
    diff2_fd = (diffs_fd_pos - diffs_fd_neg) / (2 * FD_DELTA)
    assert np.array(diff2) == pytest.approx(np.array(diff2_fd), rel=10 * FD_DELTA)


# TODO: Test hat spline


def test_spline_basis_init():
    N_knots = 11
    ref = oi.SplineBasis(N_knots, oi.basis.SincSpline())
    assert ref == oi.SplineBasis(N_knots, "sinc")
    assert ref != oi.SplineBasis(N_knots + 1, "sinc")
    assert ref != 32  # Arbitrary object that isn't a SplineBasis
    assert ref != oi.SplineBasis(N_knots, "hat")


def test_spline_basis_evals():
    N_knots = 4  # knots: {0, 1/3, 2/3, 1}
    basis = oi.SplineBasis(N_knots, "sinc")
    pts = jnp.linspace(-1 / 3, 1 / 3)
    eval0_m1_1 = basis.evaluate_basis(pts + 0.0)[:, 0]
    eval1_m1_1 = basis.evaluate_basis(pts + 1 / 3)[:, 1]
    eval2_m1_1 = basis.evaluate_basis(pts + 2 / 3)[:, 2]
    eval3_m1_1 = basis.evaluate_basis(pts + 1.0)[:, 3]
    assert jnp.linalg.norm(eval0_m1_1 - eval1_m1_1).item() < 1e-14
    assert jnp.linalg.norm(eval0_m1_1 - eval2_m1_1).item() < 1e-14
    assert jnp.linalg.norm(eval0_m1_1 - eval3_m1_1).item() < 1e-14


def test_spline_basis_diff():
    N_knots = 12
    basis = oi.SplineBasis(N_knots, "sinc")
    pts = jnp.linspace(0, 1, num=23)
    pts_pos_fd, pts_neg_fd = pts + FD_DELTA, pts - FD_DELTA
    evals = basis.evaluate_basis(pts)
    evals_pos_fd = basis.evaluate_basis(pts_pos_fd)
    evals_neg_fd = basis.evaluate_basis(pts_neg_fd)
    diffs_fd = (evals_pos_fd - evals_neg_fd) / (2 * FD_DELTA)
    evals_diff, diffs = basis.evaluate_basis_diff(pts)
    assert np.array(evals_diff) == pytest.approx(np.array(evals), rel=1e-15)
    assert np.array(diffs) == pytest.approx(np.array(diffs_fd), rel=10 * FD_DELTA)
    _, diffs_pos_fd = basis.evaluate_basis_diff(pts_pos_fd)
    _, diffs_neg_fd = basis.evaluate_basis_diff(pts_neg_fd)
    diff2_fd = (diffs_pos_fd - diffs_neg_fd) / (2 * FD_DELTA)
    evals_diff2, diffs_diff2, diff2 = basis.evaluate_basis_diff2(pts)
    assert np.array(evals_diff2) == pytest.approx(np.array(evals), rel=1e-15)
    assert np.array(diffs_diff2) == pytest.approx(np.array(diffs), rel=1e-15)
    assert np.array(diff2) == pytest.approx(np.array(diff2_fd), rel=10 * FD_DELTA)


###### ALSO USING NUMPY HERE, CHANGE######
def test_gaussian_phi_1d_shape():
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    N_terms = 5
    psi_t = np.linspace(-1, 1, N_terms)
    Phi, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
    assert Phi.shape == (N_terms, N_terms)
    assert Phi_prime.shape == (N_terms, N_terms)
    assert Phi_double_prime.shape == (N_terms, N_terms)


def test_gaussian_phi_1d_endpoints():
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    N_terms = 3
    psi_t = np.linspace(-1, 1, N_terms)
    beta_s = np.arange(N_terms)
    test_vals_0 = -beta_s * psi_t[0]
    test_vals_1 = -beta_s * psi_t[-1]
    # Standard gaussian at t=0
    char_function_gauss_0 = np.exp(-0.5 * test_vals_0**2)
    char_function_gauss_1 = np.exp(
        1j * mu * test_vals_1 - 0.5 * sigma**2 * test_vals_1**2
    )
    Phi, _, _ = phi.evaluate(psi_t)
    assert Phi[0, :] == pytest.approx(char_function_gauss_0, rel=1e-15)
    assert Phi[-1, :] == pytest.approx(char_function_gauss_1, rel=1e-15)


def test_gaussian_phi_1d_derivatives():
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    N_terms = 3
    psi_t = np.linspace(-1, 1, N_terms)
    beta_s = np.arange(N_terms)
    _, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
    # Test derivs at t=0
    pts = -beta_s * psi_t[0]
    pts_pos_fd, pts_neg_fd = pts + FD_DELTA, pts - FD_DELTA
    char_funct_0 = lambda t: np.exp(-0.5 * t**2)
    diffs_0 = (char_funct_0(pts_pos_fd) - char_funct_0(pts_neg_fd)) / (2 * FD_DELTA)

    assert np.array(Phi_prime[0, :]) == pytest.approx(
        np.array(diffs_0), rel=10 * FD_DELTA
    )
    # Second deriv at t=0
    # ...
    # Test derivs at t=1
    # Test derivs at t=0
    pts = -beta_s * psi_t[-1]
    pts_pos_fd, pts_neg_fd = pts + FD_DELTA, pts - FD_DELTA
    char_funct_1 = lambda t: np.exp(1j * mu * t - 0.5 * sigma**2 * t**2)
    diffs_1 = (char_funct_1(pts_pos_fd) - char_funct_1(pts_neg_fd)) / (2 * FD_DELTA)

    assert np.array(Phi_prime[-1, :]) == pytest.approx(
        np.array(diffs_1), rel=10 * FD_DELTA
    )
    # FINISH THIS TEST
