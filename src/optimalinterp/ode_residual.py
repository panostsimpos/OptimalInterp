import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from .moment_generator import PsiT, PhiTens, MomentGeneratingPhi
from .convolution import triple_circ_convolve_freq

PsiODET = Float[Array, "alpha+alpha"]  # concat: [Ψ(t); \dot{Ψ}(t)]
Fourier1Tens = Float[Array, "alpha"]
Fourier2Tens = Float[Array, "alpha beta"]
Fourier3Tens = Float[Array, "alpha beta gamma"]

__all__ = [
    "OptimalInterpBVP_RHS",
    "OptimalInterpBVP_mass_matrix",
    "OptimalInterpPDE_residual",
]


def convolve_tensors(
    Tens1: Fourier2Tens,
    Kernel: Fourier1Tens,
    Tens2: Fourier2Tens,
) -> Fourier2Tens:
    """
    Compute the convolution of the D tensor and the K kernel.
    Circular convolution must be used here.
    --------------------------------------------------------------
    Inputs:
    -------
    Tens1: (N_terms, N_terms) array
        First input tensor
    Kernel: (N_terms,) array
        Convolution kernel
    Tens2: (N_terms, N_terms) array
        Second input tensor
    Returns:
    --------
    None: (N_terms, N_terms, N_terms) array
        Convolution output
    --------------------------------------------------------------
    """
    out_map = jax.vmap(
        jax.vmap(triple_circ_convolve_freq, in_axes=(1, None, None), out_axes=0),
        in_axes=(None, None, 1),
        out_axes=2,  # Get out shape alpha, gamma
    )
    return out_map(Tens1, Kernel, Tens2)
    # TODO: Make sure shapes are correct!!


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
    K_tens: Fourier1Tens,
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
    phi_prod = Phi.prod(axis=0)
    C = (
        -eye[:, None, :]
        * (Phi_prime_prime[:, :, None] / Phi[:, :, None])
        * phi_prod[None, :, None]
    )

    ratio = Phi_prime / Phi
    C = C + (ones - eye[:, None, :]) * (
        ratio[:, :, None] * ratio.T[None, :, :] * phi_prod[None, :, None]
    )

    # Add convolutional term
    C = C - convolve_tensors(D_tens, K_tens, D_tens)

    # DO NOT FORGET -i*beta factor!
    beta_s = jnp.arange(Phi.shape[0])
    return -1j * beta_s[None, :, None] * C


def OptimalInterpPDEResidualPt(
    phi: MomentGeneratingPhi, psi_t: PsiT, psi_diff_t: PsiT, psi_diff2_t: PsiT
):
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = calculate_D(Phi, Phi_prime)
    K_tens = calculate_K(Phi)
    C_tens = calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens, K_tens)
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
