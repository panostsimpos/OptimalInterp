import numpy as np
import optimalinterp as oi
import pytest

###### ALSO USING NUMPY HERE, CHANGE######
FD_DELTA = 1e-7


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
    # Standard Gaussian at t=0
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
    # beta_s = np.arange(N_terms)
    _, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
    # First derivs at t=0
    # pts = -beta_s * psi_t[0]
    pts_pos_fd, pts_neg_fd = psi_t + FD_DELTA, psi_t - FD_DELTA
    # def char_funct_0(t): return np.exp(-0.5 * t**2)
    # deriv_0 = (char_funct_0(pts_pos_fd) -
    #            char_funct_0(pts_neg_fd)) / (2 * FD_DELTA)
    phi_pos_fd, phi_prime_pos_fd, _ = phi.evaluate(pts_pos_fd)
    phi_neg_fd, phi_prime_neg_fd, _ = phi.evaluate(pts_neg_fd)
    deriv = (phi_pos_fd - phi_neg_fd) / (2*FD_DELTA)
    second_deriv = (phi_prime_pos_fd - phi_prime_neg_fd) / (2*FD_DELTA)
    print("Finite difference approx")
    print(second_deriv)
    print("Evaluated phi_prime")
    print(Phi_prime)
    assert np.array(Phi_prime) == pytest.approx(
        np.array(deriv), rel=10 * FD_DELTA
    )
    assert np.array(Phi_double_prime) == pytest.approx(
        np.array(second_deriv), rel=10 * np.sqrt(FD_DELTA)
    )
