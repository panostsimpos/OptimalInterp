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
    # beta_s = np.arange(N_terms + 1)
    # test_vals_0 = -beta_s * psi_t[0]
    # test_vals_1 = -beta_s * psi_t[-1]
    # Standard Gaussian at t=0
    # char_function_gauss_0 = np.exp(-0.5 * test_vals_0**2)
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


# def test_old_gaussian_phi_1d_derivatives():
#     mu = 1.0
#     sigma = 2.0
#     phi = oi.GaussianPhi1D(mu, sigma)
#     N_terms = 3
#     psi_t = np.linspace(-1, 1, N_terms)
#     beta_s = np.arange(N_terms)
#     _, Phi_prime, Phi_double_prime = phi.evaluate(psi_t)
#     # First derivs at t=0
#     pts = -beta_s * psi_t[0]
#     pts_pos_fd, pts_neg_fd = pts + FD_DELTA, pts - FD_DELTA

#     def char_funct_0(t):
#         return np.exp(-0.5 * t**2)

#     deriv_0 = (char_funct_0(pts_pos_fd) - char_funct_0(pts_neg_fd)) / (2 * FD_DELTA)
#     second_deriv_0 = (
#         char_funct_0(pts_pos_fd) - 2 * char_funct_0(pts) + char_funct_0(pts_neg_fd)
#     ) / (FD_DELTA**2)
#     assert np.array(Phi_prime[0, :]) == pytest.approx(
#         np.array(deriv_0), rel=10 * FD_DELTA
#     )
#     assert np.array(Phi_double_prime[0, :]) == pytest.approx(
#         np.array(second_deriv_0), rel=10 * FD_DELTA
#     )
#     # Test derivs at t=1
#     pts = -beta_s * psi_t[-1]
#     pts_pos_fd, pts_neg_fd = pts + FD_DELTA, pts - FD_DELTA

#     def char_funct_1(t):
#         return np.exp(1j * mu * t - 0.5 * sigma**2 * t**2)

#     deriv_1 = (char_funct_1(pts_pos_fd) - char_funct_1(pts_neg_fd)) / (2 * FD_DELTA)
#     assert np.array(Phi_prime[-1, :]) == pytest.approx(
#         np.array(deriv_1), rel=10 * FD_DELTA
#     )
#     second_deriv_1 = (
#         char_funct_1(pts_pos_fd) - 2 * char_funct_1(pts) + char_funct_1(pts_neg_fd)
#     ) / (FD_DELTA**2)
#     assert np.array(Phi_double_prime[-1, :]) == pytest.approx(
#         np.array(second_deriv_1), rel=10 * FD_DELTA
#     )


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
    # np.set_printoptions(precision=3, linewidth=200)
    # print("Finite difference approx")
    # print(second_deriv)
    # print("\nEvaluated phi''")
    # print(-Phi_double_prime * ((-beta_s[None, :])))
    # print("Shape")
    # print(Phi_double_prime.shape)

    assert np.array(Phi_prime * (-beta_s[None, :])) == pytest.approx(
        np.array(deriv), rel=10 * FD_DELTA
    )
    assert np.array(Phi_double_prime * (-beta_s[None, :])) == pytest.approx(
        np.array(second_deriv), rel=10 * FD_DELTA
    )
