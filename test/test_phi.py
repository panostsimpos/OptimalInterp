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
    phi_g = oi.GaussianConvolutionPhi1D(N_beta_terms, mu, sigma)
    phi_wrapped = oi.WrappedGaussianConvolutionPhi1D(N_alpha_terms, N_beta_terms, 10, mu, sigma)
    for phi in [phi_g, phi_wrapped]:
        psi_t = jnp.linspace(-1, 1, N_alpha_terms)
        Phi, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
        assert Phi.shape == (N_alpha_terms, N_beta_terms)
        assert Phi_prime.shape == (N_alpha_terms, N_beta_terms)
        assert Phi_double_prime.shape == (N_alpha_terms, N_beta_terms)


def gaussian_characteristic_function(input, mu, sigma):
    output = jnp.zeros_like(input, dtype=jnp.complex128)
    N_alpha, N_beta = input.shape
    for alpha in range(N_alpha):
        for beta in range(N_beta):
            t = input[alpha, beta]# -beta * psi_t[alpha]
            ratio = alpha / (N_alpha - 1)
            mean_term = 1j * mu * ratio * t
            noise_term = 0j + 0.5 * t * t * ((1 - ratio) ** 2 + (ratio * sigma) ** 2)
            log_phi = mean_term - noise_term
            output = output.at[alpha, beta].set(jnp.exp(log_phi))
    return output


def test_gaussian_phi_1d_eval():
    mu = 1.0
    sigma = 2.0
    N_alpha, N_beta = 5, 7
    phi = oi.GaussianConvolutionPhi1D(N_beta, mu, sigma)
    psi_t = jnp.linspace(-1, 1, N_alpha)
    beta = jnp.arange(N_beta)
    input_vals = -beta.reshape(1, -1) * psi_t.reshape(-1, 1)
    char_function_gauss = gaussian_characteristic_function(input_vals, mu, sigma)
    Phi, _, _ = phi.evaluate(psi_t)
    assert Phi == pytest.approx(char_function_gauss, rel=1e-15)


def test_wrapped_gaussian_phi_1d_eval():
    mu = 1.0
    sigma = 2.0
    N_alpha, N_beta, wrap_max_idx = 5, 7, 10
    phi = oi.WrappedGaussianConvolutionPhi1D(N_alpha, N_beta, wrap_max_idx, mu, sigma)
    psi_t = jnp.linspace(-1, 1, N_alpha)
    Phi, _, _ = phi.evaluate(psi_t)
    k_vec = jnp.arange(-wrap_max_idx, wrap_max_idx + 1)
    k_mat = k_vec.reshape(1,-1).repeat(N_alpha,axis=0)
    # (alpha, k)
    char_function_gauss_k = gaussian_characteristic_function(k_mat, mu, sigma)
    # (k, alpha, beta)
    beta_psi = -jnp.arange(N_beta).reshape(1,1,-1) * psi_t.reshape(1,-1,1)
    difference = beta_psi - k_vec.reshape(-1, 1, 1)
    exp = jnp.exp(1j * jnp.pi * difference)
    sinc = jnp.sinc(difference)
    ref_Phi = jnp.einsum("ak,kab,kab->ab", char_function_gauss_k, exp, sinc)
    assert Phi == pytest.approx(ref_Phi, rel=1e-15)


def test_phi_1d_derivatives():
    mu = 1.0
    sigma = 2.0
    N_alpha, N_beta = 5, 7
    phi_g = oi.GaussianConvolutionPhi1D(N_beta, mu, sigma)
    phi_wrapped = oi.WrappedGaussianConvolutionPhi1D(N_alpha, N_beta, 10, mu, sigma)
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
