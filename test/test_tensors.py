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
    D_tens = oi.ode_residual.calculate_D(Phi, Phi_prime)
    assert D_tens.shape == (N_terms, N_terms)
    K = oi.ode_residual.calculate_K(Phi)
    assert K.shape == (N_terms,)
    C_tens = oi.ode_residual.calculate_C(
        Phi, Phi_prime, Phi_prime_prime, D_tens, K)
    assert C_tens.shape == (N_terms, N_terms, N_terms)


def test_K_kernel():
    N_terms = 5
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    psi_t = jnp.linspace(-2, 2, N_terms) + 1j * jnp.linspace(-2, 2, N_terms)
    Phi, _, _ = phi.evaluate(psi_t)
    K_t = oi.ode_residual.calculate_K(Phi)
    # ----------------------
    # Check that F^{-1}[L_t] * F^{-1}[K] = 1
    L_t = jnp.ones((N_terms), dtype=complex)
    for beta in range(N_terms):
        for alpha in range(N_terms):
            L_t = L_t.at[beta].multiply(Phi[alpha, beta])
    test_val = jnp.fft.ifft(L_t) * jnp.fft.ifft(K_t)
    ones = jnp.ones((N_terms), dtype=complex)
    assert test_val == pytest.approx(ones, rel=1e-15)
    # ----------------------
    # Check real inputs -> real outputs
    zeroes = jnp.zeros((N_terms,), dtype=complex)
    K_real = oi.ode_residual.calculate_K(Phi.real)
    assert K_real.imag == pytest.approx(zeroes, abs=1e-15)
    # ----------------------
    # Check convolution identity:
    # L_t \ast K_t = F[F^{-1}[L_t]] \ast F[1/F^{-1}[L_t]] = F[1] = pulse-in-freq-space
    conv_out = oi.convolution.circ_convolution(
        L_t, K_t, domain_type=oi.convolution.CONVOLUTION_DOMAIN.FREQ
    )
    long_ones = jnp.ones_like(conv_out)
    print("*************CONV OUT:***************")
    print(jnp.round(conv_out, decimals=5))
    assert conv_out == pytest.approx(jnp.fft.fft(long_ones), rel=1e-15)


def test_D_tensor():
    N_terms = 5
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    psi_t = jnp.linspace(-2, 2, N_terms)
    Phi, Phi_prime, _ = phi.evaluate(psi_t)
    D_tens = oi.ode_residual.calculate_D(Phi, Phi_prime)
    L_t = jnp.ones((N_terms), dtype=complex)
    for beta in range(N_terms):
        for alpha in range(N_terms):
            L_t = L_t.at[beta].multiply(Phi[alpha, beta])
    D_tens_manual = jnp.ones((N_terms, N_terms), dtype=complex)
    for beta in range(N_terms):
        for alpha in range(N_terms):
            D_tens_manual = D_tens_manual.at[alpha, beta].set(
                -1j * Phi_prime[alpha, beta] / Phi[alpha, beta] * L_t[beta]
            )
    assert D_tens == pytest.approx(D_tens_manual, rel=1e-15)


def test_C_tensor():
    N_terms = 5
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianPhi1D(mu, sigma)
    psi_t = jnp.linspace(-2, 2, N_terms)
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = oi.ode_residual.calculate_D(Phi, Phi_prime)
    K_tens = oi.ode_residual.calculate_K(Phi)
    C_tens = oi.ode_residual.calculate_C(
        Phi, Phi_prime, Phi_prime_prime, D_tens, K_tens)
    pass
