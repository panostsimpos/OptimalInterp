from math import gamma
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple
from .moment_generator import PsiT, PhiTens, MomentGeneratingPhi
import numpy as np  # Provisionally, until Panos becomes comfortable with jax

PsiODET = Float[Array, "alpha+alpha"]  # concat: [Ψ(t); \dot{Ψ}(t)]
Fourier1Tens = Float[Array, "alpha"]
Fourier2Tens = Float[Array, "alpha beta"]
Fourier3Tens = Float[Array, "alpha beta gamma"]

__all__ = [
    "OptimalInterpBVP_RHS",
    "OptimalInterpBVP_mass_matrix",
    "OptimalInterpPDE_residual",
]


def calculate_K(Phi: PhiTens) -> Fourier1Tens:
    """
    Calculate convolution kernel K_t = IFFT(1/FFT(L_t)), where L_t = Prod_alpha Phi_alpha(-beta psi_t[alpha])
    --------------------------------------------------------------
    Inputs:
        Phi: (N_terms, N_terms) array
            Moment generating function evaluations
    Returns:
        K_t: (N_terms,) array
            Convolution kernel
    --------------------------------------------------------------
    """
    L_t = Phi.prod(axis=0)  # (N_terms,) array
    L_t_hat = jnp.fft.ifft(L_t)
    K_t = jnp.fft.fft(jnp.reciprocal(L_t_hat))
    return K_t


def calculate_D(Phi: PhiTens, Phi_prime: PhiTens) -> Fourier2Tens:
    """
    Calculate D_{alpha,beta} = -i * beta * Phi'_alpha(-beta psi_t[alpha]) Prod_{gamma != alpha} Phi_gamma(-beta psi_t[gamma])
    -----------------------------------------------
    Warning:
    --------
    Indexing is flipped compared to paper notation!
    ------------------------------------------------
    Args:
        Phi: (N_terms, N_terms) array
        Phi_prime: (N_terms, N_terms) array
    Returns:
        D: (N_terms, N_terms) array
    """
    return -1j * Phi_prime / Phi * Phi.prod(axis=0)[None, :]


def calculate_C(
    Phi: PhiTens,
    Phi_prime: PhiTens,
    Phi_prime_prime: PhiTens,
    D_tens: Fourier2Tens,
    K: Fourier1Tens,
) -> Fourier3Tens:
    """
    Calculate C_{alpha,beta,gamma} tensor needed for PDE residual
    --------------------------------------------------------------
    Inputs:
    -------
    Phi: (N_terms, N_terms) array
        Moment generating function evaluations
    Phi_prime: (N_terms, N_terms) array
        First derivatives of moment generating function evaluations
    Phi_prime_prime: (N_terms, N_terms) array
        Second derivatives of moment generating function evaluations
    D_tens: (N_terms, N_terms) array
        D_{alpha,beta} tensor
    K: (N_terms,) array
        Convolution kernel
    Returns:
    --------
    C: (N_terms, N_terms, N_terms) array
        C_{alpha,beta,gamma} tensor
    --------------------------------------------------------------
    """

    eye = jnp.eye(D_tens.shape[0])
    ones = jnp.ones(D_tens.shape)

    C = (
        -eye[:, None, :]
        * Phi_prime_prime[:, :, None]
        / Phi[:, :, None]
        * Phi.prod(axis=0)[None, :, None]
    )
    # Use that 1/(-1j) = j and j*j = -1
    C = C - (ones - eye[:, None, :]) * D_tens[:, :, None] * D_tens[None, :, :]

    # ------------------------------
    # TODO: Fix convolution using circ_convolution from aux_tools.py!!
    # ------------------------------
    # Add convolutional term
    def convolve_term(D_alpha, D_gamma):
        # Out is length 2*N_terms-1
        temp = jnp.convolve(K, D_gamma, mode="full")
        # Out is length N_terms
        return jnp.convolve(D_alpha, temp, mode="valid")

    batch_convolve = jax.vmap(
        jax.vmap(convolve_term, in_axes=(0, None), out_axes=0),
        in_axes=(None, 0),
        out_axes=2,  # Get out shape alpha,beta,gamma
    )
    C = C - batch_convolve(D_tens, D_tens)

    # DO NOT FORGET -i*beta factor!
    beta_s = jnp.arange(Phi.shape[0])
    return -1j * beta_s[None, :, None] * C


def OptimalInterpPDEResidualPt(
    phi: MomentGeneratingPhi, psi_t: PsiT, psi_diff_t: PsiT, psi_diff2_t: PsiT
):
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = calculate_D(Phi, Phi_prime)
    C_tens = calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens)
    # TODO
    pass


def OptimalInterpPDE_residual(phi: MomentGeneratingPhi, *args):
    # TODO: Figure out what arguments should go here...
    pass


def OptimalInterpBVP_RHS(concat_psi_psi_dt: PsiODET) -> PsiODET:
    N_terms = concat_psi_psi_dt.shape[0] // 2
    psi_t, psi_dot_t = concat_psi_psi_dt[:N_terms], concat_psi_psi_dt[N_terms:]
    pass


def OptimalInterpBVP_mass_matrix(concat_psi_psi_dt: PsiODET) -> PsiODET:
    N_terms = concat_psi_psi_dt.shape[0] // 2
    psi_t, psi_dot_t = concat_psi_psi_dt[:N_terms], concat_psi_psi_dt[N_terms:]
    pass
