import optimalinterp as oi
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def test_D_K_C_tensor_shape():
    N_alpha, max_beta_idx = 5, 7
    N_beta = 2 * max_beta_idx + 1
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianConvolutionPhi1D(max_beta_idx, mu, sigma)
    psi_t = jnp.linspace(-1, 1, N_alpha)
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = oi.ode_residual.calculate_D(Phi, Phi_prime)
    assert D_tens.shape == (N_alpha, N_beta)
    K = oi.ode_residual.calculate_K(Phi)
    assert K.shape == (N_beta,)
    C_tens = oi.ode_residual.calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens, K)
    assert C_tens.shape == (N_alpha, N_beta, N_alpha)


def simple_dft(v, direction: str):
    inv_f_v = np.zeros_like(v)
    if direction.startswith("i"):
        sign = 1
    elif direction.startswith("f"):
        sign = -1
    else:
        raise ValueError()

    for x_idx in range(len(v)):
        for beta in range(len(v)):
            inv_f_v[x_idx] += v[beta] * np.exp(1j * sign * 2 * np.pi * beta / len(v))
    return jnp.array(inv_f_v)


# def test_K_kernel():
#     N_terms = 5
#     N_beta = 2 * N_terms + 1
#     mu = 1.0
#     sigma = 2.0
#     phi = oi.GaussianConvolutionPhi1D(N_terms, mu, sigma)
#     psi_t = jnp.linspace(-2, 2, N_terms) + 1j * jnp.linspace(-2, 2, N_terms)
#     Phi, _, _ = phi.evaluate(psi_t)
#     K_t = oi.ode_residual.calculate_K(Phi)
#     # ----------------------
#     # Check that F^{-1}[L_t] * F^{-1}[K] = 2pi
#     L_t = jnp.ones((N_beta), dtype=complex)
#     for beta_idx in range(N_beta):
#         for alpha in range(N_terms):
#             L_t = L_t.at[beta_idx].multiply(Phi[alpha, beta_idx])
#     test_val = simple_dft(L_t, "i") * simple_dft(K_t, "i")
#     ref_val = jnp.ones((N_beta), dtype=complex)
#     assert test_val == pytest.approx(ref_val, rel=1e-15)
#     # ----------------------
#     # Check real inputs -> real outputs
#     zeroes = jnp.zeros((N_beta,), dtype=complex)
#     K_real = oi.ode_residual.calculate_K(Phi.real)
#     assert K_real.imag == pytest.approx(zeroes, abs=1e-15)
#     # ----------------------
#     # Check convolution identity:
#     # L_t \ast K_t = F[F^{-1}[L_t]] \ast F[1/F^{-1}[L_t]] = F[1] = pulse-in-freq-space
#     # conv_out = oi.convolution.circ_convolution(
#     #     L_t, K_t, domain_type=oi.convolution.CONVOLUTION_DOMAIN.FREQ
#     # )
#     # long_ones = jnp.ones_like(conv_out) / 2 * jnp.pi
#     # print("*************CONV OUT:***************")
#     # print(jnp.round(conv_out, decimals=5))
#     # print(jnp.round(long_ones, decimals=5))
#     # assert conv_out == pytest.approx(simple_dft(long_ones, 'f'), rel=1e-15)


def test_D_tensor():
    N_alpha, max_beta_idx = 5, 7
    N_beta = 2 * max_beta_idx + 1
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianConvolutionPhi1D(max_beta_idx, mu, sigma)
    psi_t = jnp.linspace(-2, 2, N_alpha)
    Phi, Phi_prime, _ = phi.evaluate(psi_t)
    D_tens = oi.ode_residual.calculate_D(Phi, Phi_prime)
    L_t = np.ones((N_beta), dtype=np.complex128)
    for beta_idx in range(N_beta):
        for alpha in range(N_alpha):
            L_t[beta_idx] *= Phi[alpha, beta_idx]
    D_tens_manual = np.ones((N_alpha, N_beta), dtype=np.complex128)
    for beta_idx in range(N_beta):
        for alpha in range(N_alpha):
            D_tens_manual[alpha, beta_idx] = (
                Phi_prime[alpha, beta_idx] / Phi[alpha, beta_idx] * L_t[beta_idx]
            )
    assert D_tens == pytest.approx(jnp.array(D_tens_manual), rel=1e-15)


def test_C_tensor():
    """
    Verify the C tensor assembly against the mathematical definition (for d=1):

    C_{α,β,γ}(t) = β * { δ_{αγ} * ∂²Φ_α(-β·ψ_α) * ∏_{δ≠α} Φ_δ(-β·ψ_δ)
                       + (δ_{αγ} - 1) * ∂Φ_α(-β·ψ_α) * ∂Φ_γ(-β·ψ_γ) * ∏_{δ≠α,γ} Φ_δ(-β·ψ_δ)
                       - (D_{•,α} * K * D_{•,γ})(β) }

    Note: In d=1, the sum over j collapses to a single term with β_j = β.
    Note: β ranges from -max_beta_idx to +max_beta_idx (centered at 0).
    """
    N_alpha, max_beta_idx = 5, 7
    N_beta = 2 * max_beta_idx + 1
    mu = 1.0
    sigma = 2.0
    phi = oi.GaussianConvolutionPhi1D(max_beta_idx, mu, sigma)
    psi_t = jnp.linspace(-2, 2, N_alpha)
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = oi.ode_residual.calculate_D(Phi, Phi_prime)
    K_tens = oi.ode_residual.calculate_K(Phi)
    C_tens = oi.ode_residual.calculate_C(
        Phi, Phi_prime, Phi_prime_prime, D_tens, K_tens
    )

    # Build C tensor manually following the mathematical definition
    # β ranges from -max_beta_idx to +max_beta_idx
    beta_v = np.arange(-max_beta_idx, max_beta_idx + 1)
    C_manual = np.zeros((N_alpha, N_beta, N_alpha), dtype=np.complex128)

    # Pre-compute L_t = ∏_δ Φ_δ (full product over all α indices) for each β
    L_t = np.array(Phi.prod(axis=0))  # shape: (N_beta,)

    # Pre-compute the convolution term for all (alpha, gamma) pairs
    # (D_{•,α} * K * D_{•,γ})(β) via triple circular convolution
    conv_term = np.zeros((N_alpha, N_beta, N_alpha), dtype=np.complex128)
    for alpha in range(N_alpha):
        for gamma in range(N_alpha):
            D_alpha = np.array(D_tens[alpha, :])
            D_gamma = np.array(D_tens[gamma, :])
            K_arr = np.array(K_tens)
            conv_result = oi.convolution.triple_circ_convolve_freq(
                jnp.array(D_alpha), jnp.array(K_arr), jnp.array(D_gamma)
            )
            conv_term[alpha, :, gamma] = np.array(conv_result)

    for alpha in range(N_alpha):
        for beta_idx in range(N_beta):
            for gamma in range(N_alpha):
                if alpha == gamma:
                    # Term 1: δ_{αγ} * ∂²Φ_α * ∏_{δ≠α} Φ_δ
                    # = (Φ''_α / Φ_α) * L_t
                    term1 = (
                        Phi_prime_prime[alpha, beta_idx] / Phi[alpha, beta_idx]
                    ) * L_t[beta_idx]
                    term2 = 0.0
                else:
                    # Term 1 is zero when α ≠ γ
                    term1 = 0.0
                    # Term 2: (δ_{αγ} - 1) * ∂Φ_α * ∂Φ_γ * ∏_{δ≠α,γ} Φ_δ
                    # = -1 * (Φ'_α / Φ_α) * (Φ'_γ / Φ_γ) * L_t
                    term2 = (
                        -1.0
                        * (Phi_prime[alpha, beta_idx] / Phi[alpha, beta_idx])
                        * (Phi_prime[gamma, beta_idx] / Phi[gamma, beta_idx])
                        * L_t[beta_idx]
                    )

                # Term 3: -(D_{•,α} * K * D_{•,γ})(β)
                term3 = -conv_term[alpha, beta_idx, gamma]

                # Combine and multiply by β (actual value, not index)
                C_manual[alpha, beta_idx, gamma] = beta_v[beta_idx] * (
                    term1 + term2 + term3
                )

    assert C_tens == pytest.approx(jnp.array(C_manual), rel=1e-12)
