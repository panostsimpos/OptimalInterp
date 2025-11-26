import numpy as np
import optimalinterp as oi
import pytest

###### ALSO USING NUMPY HERE, CHANGE######
FD_DELTA = 1e-6


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


def test_gaussian_phi_1d_eval():
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    N_terms = 5
    psi_t = np.linspace(-1, 1, N_terms)
    char_function_gauss = np.zeros((N_terms, N_terms), dtype=np.complex128)
    for beta in range(N_terms):
        for alpha in range(N_terms):
            t = -beta * psi_t[alpha]
            ratio = alpha / (N_terms-1)
            mean_term = 1j * mu * ratio * t
            noise_term = 0j + 0.5 * t * t * ((1-ratio)**2 + (ratio*sigma)**2)
            log_phi = mean_term - noise_term
            char_function_gauss[alpha, beta] = np.exp(log_phi)
    Phi, _, _ = phi.evaluate(psi_t)
    assert Phi == pytest.approx(char_function_gauss, rel=1e-15)


def test_gaussian_phi_1d_derivatives():
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    N_terms = 5
    psi_t = np.linspace(-1, 1, N_terms)
    _, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
    pts_pos_fd, pts_neg_fd = psi_t + FD_DELTA, psi_t - FD_DELTA
    phi_pos_fd, phi_prime_pos_fd, _ = phi.evaluate(pts_pos_fd)
    phi_neg_fd, phi_prime_neg_fd, _ = phi.evaluate(pts_neg_fd)
    beta_s = np.arange(N_terms)
    deriv = (phi_pos_fd - phi_neg_fd) / (2 * FD_DELTA)
    second_deriv = (phi_prime_pos_fd - phi_prime_neg_fd) / (2 * FD_DELTA)
    assert np.array(Phi_prime * (-beta_s[None, :])) == pytest.approx(
        np.array(deriv), rel=10 * FD_DELTA
    )
    assert np.array(Phi_double_prime * (-beta_s[None, :])) == pytest.approx(
        np.array(second_deriv), rel=10 * FD_DELTA
    )
