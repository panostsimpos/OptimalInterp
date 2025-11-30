import optimalinterp as oi
import pytest
import jax.numpy as jnp


def test_D_K_C_tensor_shape():
    N_terms = 5
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    psi_t = jnp.linspace(-1, 1, N_terms)
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = oi.calculate_D(Phi, Phi_prime)
    assert D_tens.shape == (N_terms, N_terms)
    K = oi.calculate_K(Phi)
    assert K.shape == (N_terms,)
    C_tens = oi.calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens, K)
    assert C_tens.shape == (N_terms, N_terms, N_terms)
