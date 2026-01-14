import jax
import jax.numpy as jnp
import numpy as np
import optimalinterp as oi
import pytest
jax.config.update("jax_enable_x64", True)

###### ALSO USING NUMPY HERE, CHANGE######
FD_DELTA = 1e-6


def test_phi_1d_shape():
    mu = 1.0
    sigma = 2.0
    N_alpha_terms, N_beta_terms = 5, 7
    phi_g = oi.GaussianPhi1D(N_beta_terms, mu, sigma)
    phi_wrapped = oi.WrappedGaussianBridgePhi1D(N_alpha_terms, N_beta_terms, 10, mu, sigma)
    for phi in [phi_g, phi_wrapped]:
        psi_t = jnp.linspace(-1, 1, N_alpha_terms)
        Phi, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
        assert Phi.shape == (N_alpha_terms, N_beta_terms)
        assert Phi_prime.shape == (N_alpha_terms, N_beta_terms)
        assert Phi_double_prime.shape == (N_alpha_terms, N_beta_terms)


def test_gaussian_phi_1d_eval():
    mu = 1.0
    sigma = 2.0
    N_alpha, N_beta = 5, 7
    phi = oi.GaussianPhi1D(N_beta, mu, sigma)
    psi_t = jnp.linspace(-1, 1, N_alpha)
    char_function_gauss = jnp.zeros((N_alpha, N_beta), dtype=jnp.complex128)
    for beta in range(N_beta):
        for alpha in range(N_alpha):
            t = -beta * psi_t[alpha]
            ratio = alpha / (N_alpha - 1)
            mean_term = 1j * mu * ratio * t
            noise_term = 0j + 0.5 * t * t * ((1 - ratio) ** 2 + (ratio * sigma) ** 2)
            log_phi = mean_term - noise_term
            char_function_gauss = char_function_gauss.at[alpha, beta].set(jnp.exp(log_phi))
    Phi, _, _ = phi.evaluate(psi_t)
    assert Phi == pytest.approx(char_function_gauss, rel=1e-15)


def test_phi_1d_derivatives():
    mu = 1.0
    sigma = 2.0
    N_alpha, N_beta = 5, 7
    phi_g = oi.GaussianPhi1D(N_beta, mu, sigma)
    phi_wrapped = oi.WrappedGaussianBridgePhi1D(N_alpha, N_beta, 10, mu, sigma)
    psi_t = jnp.linspace(-1, 1, N_alpha)
    for phi in [phi_g, phi_wrapped]:
        _, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
        pts_pos_fd, pts_neg_fd = psi_t + FD_DELTA, psi_t - FD_DELTA
        phi_pos_fd, phi_prime_pos_fd, _ = phi.evaluate(pts_pos_fd)
        phi_neg_fd, phi_prime_neg_fd, _ = phi.evaluate(pts_neg_fd)
        beta_s = jnp.arange(N_beta)
        deriv = (phi_pos_fd - phi_neg_fd) / (2 * FD_DELTA)
        second_deriv = (phi_prime_pos_fd - phi_prime_neg_fd) / (2 * FD_DELTA)
        assert jnp.array(Phi_prime * (-beta_s[None, :])) == pytest.approx(
            jnp.array(deriv), rel=10 * FD_DELTA
        )
        assert jnp.array(Phi_double_prime * (-beta_s[None, :])) == pytest.approx(
            jnp.array(second_deriv), rel=10 * FD_DELTA
        )
