import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple
from .moment_generator import PhiTens, MomentGeneratingPhi

PsiT = Float[Array, "alpha"]
PsiODET = Float[Array, "alpha+alpha"]  # concat: [Ψ(t); \dot{Ψ}(t)]
Fourier1Tens = Float[Array, "alpha"]
Fourier2Tens = Float[Array, "alpha beta"]
Fourier3Tens = Float[Array, "alpha beta gamma"]

__all__ = [
    'OptimalInterpBVP_RHS',
    'OptimalInterpBVP_mass_matrix',
    'OptimalInterpPDE_residual'
]


def calculate_K(Phi: PhiTens, Phi_prime: PhiTens) -> Fourier1Tens:
    pass


def calculate_D(Phi: PhiTens, Phi_prime: PhiTens) -> Fourier2Tens:
    pass


def calculate_C(Phi: PhiTens, Phi_prime: PhiTens, Phi_prime_prime: PhiTens, D_tens: Fourier2Tens) -> Fourier3Tens:
    pass


def OptimalInterpPDEResidualPt(phi: MomentGeneratingPhi, psi_t: PsiT, psi_diff_t: PsiT, psi_diff2_t: PsiT):
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
