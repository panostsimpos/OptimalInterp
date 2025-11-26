import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple
from .moment_generator import PhiTens, MomentGeneratingPhi
import numpy as np  # Provisionally, until Panos becomes comfortable with jax

PsiT = Float[Array, "alpha"]
PsiODET = Float[Array, "alpha+alpha"]  # concat: [Ψ(t); \dot{Ψ}(t)]
Fourier1Tens = Float[Array, "alpha"]
Fourier2Tens = Float[Array, "alpha beta"]
Fourier3Tens = Float[Array, "alpha beta gamma"]

__all__ = [
    "OptimalInterpBVP_RHS",
    "OptimalInterpBVP_mass_matrix",
    "OptimalInterpPDE_residual",
]


def calculate_K(Phi: PhiTens, Phi_prime: PhiTens) -> Fourier1Tens:
    pass


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
    return -1j * Phi_prime / Phi * np.prod(Phi, axis=0)[None, :]


def calculate_C(
    Phi: PhiTens, Phi_prime: PhiTens, Phi_prime_prime: PhiTens, D_tens: Fourier2Tens
) -> Fourier3Tens:
    eye = np.eye(D_tens.shape[0])
    ones = np.ones(D_tens.shape)

    # DO NOT FORGET -i*beta factor!
    C = (
        -eye[:, None, :]
        * Phi_prime_prime[:, :, None]
        / Phi[:, :, None]
        * np.prod(Phi, axis=0)[None, :, None]
    )
    # Use that 1/(-1j) = j and j*j = -1
    C = C - (ones - eye[:, None, :]) * D_tens[:, :, None] * D_tens[None, :, :]


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
