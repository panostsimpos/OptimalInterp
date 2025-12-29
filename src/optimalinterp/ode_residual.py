import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from .moment_generator import PsiT, PhiTens, MomentGeneratingPhi
from .basis import SplineBasis, AbstractLinearBasis
from .convolution import triple_circ_convolve_freq

PsiODET = Float[Array, "alpha+alpha"]  # concat: [Ψ(t); \dot{Ψ}(t)]
Fourier1Tens = Float[Array, "alpha"]
Fourier2Tens = Float[Array, "alpha beta"]
Fourier3Tens = Float[Array, "alpha beta gamma"]

__all__ = [
    "OptimalInterpBVP_ODE_RHS",
    "OptimalInterpBVP_DAE_RHS",
    "OptimalInterpBVP_DAE_LHS",
    "create_spline_residual",
    "create_collocated_basis_residual",
    "pad_coeffs"
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
        jax.vmap(triple_circ_convolve_freq,
                 in_axes=(1, None, None), out_axes=0),
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
        Phi: (alpha, beta) array
        Phi_prime: (alpha, beta) array
    Returns:
        D: (alpha, beta) array
    """
    return -1j * Phi_prime / Phi * Phi.prod(axis=0)[None, :]


def calculate_C(
    Phi: PhiTens,
    Phi_prime: PhiTens,
    Phi_prime_prime: PhiTens,
    D_tens: Fourier2Tens,
    K_tens: Fourier1Tens,
) -> Fourier3Tens:
    r"""
    Calculate C_{alpha,beta,gamma} tensor needed for PDE residual
    --------------------------------------------------------------
    Inputs:
    -------
    Phi: (N_terms, N_terms) array
        Moment generating function evaluations with signature
        Phi[alpha, beta] = \Phi_alpha(-beta psi_t[alpha])
    Phi_prime: (N_terms, N_terms) array
        First derivatives of moment generating function evaluations and signature as above.
    Phi_prime_prime: (N_terms, N_terms) array
        Second derivatives of moment generating function evaluations and signature as above.
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
    N_terms = D_tens.shape[0]
    phi_prod = Phi.prod(axis=0)

    ratio_1 = Phi_prime / Phi
    eltype = Phi.dtype
    C = jnp.einsum(
        'ab,gb,b->abg', ratio_1, ratio_1, phi_prod, preferred_element_type=eltype
    )

    # Terms where alpha == gamma
    idx = jnp.arange(N_terms)
    ratio_2 = Phi_prime_prime / Phi
    C = C.at[idx, :, idx].set(-ratio_2 * phi_prod[None, :])

    # Add convolutional term
    C = C - convolve_tensors(D_tens, K_tens, D_tens)

    # DO NOT FORGET -i*beta factor!
    beta_s = jnp.arange(Phi.shape[0], dtype=eltype)
    return -1j * beta_s[None, :, None] * C


def OptimalInterpPDEResidualPt(
    psi_t: PsiT, psi_dot_t: PsiT, psi_diff2_t: PsiT, phi: MomentGeneratingPhi
):
    r"""
    Evaluate the residual of the optimal interpolant at a given point in time.

    :param psi_t: Evaluation of interpolant basis at time t
    :type psi_t: PsiT
    :param psi_dot_t: First derivative of interpolant basis at time t
    :type psi_dot_t: PsiT
    :param psi_diff2_t: Second derivative of interpolant basis at time t
    :type psi_diff2_t: PsiT
    :param phi: Moment generating function of path
    :type phi: MomentGeneratingPhi
    """
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    eltype = Phi.dtype
    D_tens = calculate_D(Phi, Phi_prime)
    K_tens = calculate_K(Phi)
    C_tens = calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens, K_tens)
    rhs = jnp.einsum('abg,a,g->b', C_tens, psi_dot_t, psi_dot_t, preferred_element_type=eltype)
    lhs = jnp.einsum('ab,a->b', D_tens, psi_diff2_t, preferred_element_type=eltype)
    return rhs - lhs


__OptimalInterpPDEResidual_vmap = jax.vmap(
    OptimalInterpPDEResidualPt, in_axes=(0, 0, 0, None))


def create_spline_residual(psi: SplineBasis, phi: MomentGeneratingPhi):
    """
    Create a residual function for a given spline basis

    :param psi: Basis of spline functions
    :type psi: SplineBasis
    :param phi: Moment generating function
    :type phi: MomentGeneratingPhi
    """
    evals, basis_diff1, basis_diff2 = psi.collocated_basis_transform()

    def spline_residual(coeffs_psi, _):
        # Assume a basis with T+2 knots (0,...,T+1), where t_0 = 0 and t_{T+1} = 1.
        # Each col of coeffs_psi represents the coefficients for a given output.
        # Each row of coeffs_psi represents a given spline function, i.e., col k represents spline centered at k/(T+1).
        # dt_scale is the derivative operator on the coeffs_psi for this basis evaluated at the knots.
        # dt_shift is the shift of the derivative operator to ensure boundary conditions are satisfied.
        coeffs_psi_full = jnp.pad(
            coeffs_psi, ((1, 1), (1, 1)), mode='constant'
        ).at[[0, -1], [0, -1]].set(1.)
        psi = coeffs_psi_full  # Note that the splines are just the coeffs at collocation points
        psi_dot = basis_diff1 @ coeffs_psi_full
        psi_dot_dot = basis_diff2 @ coeffs_psi_full
        residual = __OptimalInterpPDEResidual_vmap(
            psi, psi_dot, psi_dot_dot, phi
        )
        return jnp.abs(residual)
    return jax.jit(spline_residual)


def pad_coeffs(coeffs: Float[Array, "p-2 alpha"], psi: AbstractLinearBasis, fixed_orders: tuple[int, int]):
    """
    Pad the coefficients for a linear basis to ensure function satisfies boundary conditions

    :param coeffs: Non-fixed coefficients
    :type coeffs: Float[Array, "p-2 alpha"]
    :param psi: Basis for coefficients
    :type psi: AbstractLinearBasis
    :param fixed_orders: Indices of coefficients are constrained
    :type fixed_orders: tuple[int, int]
    """
    N_terms = coeffs.shape[1]
    basis_eval = psi.evaluate_basis(jnp.zeros(2).at[1].set(1.))
    order0, order1 = fixed_orders
    bc_scale = basis_eval[[0, 0, -1, -1], [order0, order1, order0, order1]].reshape(2,2)
    inv_bc_scale = jnp.linalg.inv(bc_scale)
    bc_shift = jnp.zeros((2, N_terms)).at[[0, 1], [0, -1]].set(1.)
    first_coeffs = inv_bc_scale @ (
        bc_shift - (basis_eval[:, 2:] @ coeffs)
    )
    return jnp.concat((first_coeffs, coeffs))


def create_collocated_basis_residual(
        t_grid: Float[Array, " T"],
        N_terms: int, psi: AbstractLinearBasis,
        phi: MomentGeneratingPhi,
        fixed_orders: tuple[int, int],
        wts: float | Float[Array, " T"] = 1.
    ):
    """
    Create a residual using collocation. Assume that the first two elements of the linear basis are fixed to ensure boundary conditions.

    :param t_grid: Grid of points to collocate
    :type t_grid: Float[Array, "T"]
    :param N_terms: Number of expansion terms
    :type N_terms: int
    :param psi: Basis for expansion terms
    :type psi: AbstractLinearBasis
    :param phi: Moment generation function
    :type phi: MomentGeneratingPhi
    :param fixed_orders: Indices of coefficients constrained by boundary conditions
    :type fixed_orders: tuple[int, int]
    :param wts: Optional weights for collocations
    :type wts: float | Float[Array, " T"]
    """
    t_grid = jnp.sort(t_grid)
    assert t_grid[0] == 0. and t_grid[-1] == 1.  # Ensure grid is valid
    # Evaluate basis
    basis_eval, basis_diff1, basis_diff2 = psi.evaluate_basis_diff2(t_grid)
    # Get the zero and first order basis at times t=0, t=1
    order0, order1 = fixed_orders
    assert order0 != order1
    keep_orders = jnp.delete(jnp.arange(psi.N_shap+1), jnp.array([order0, order1]), assume_unique_indices=True)
    bc_scale = basis_eval[[0, 0, -1, -1], [order0, order1, order0, order1]].reshape(2,2)
    # Set the boundary conditions
    inv_bc_scale = jnp.linalg.inv(bc_scale)
    bc_shift = jnp.zeros((2, N_terms)).at[[0, 1], [0, -1]].set(1.)
    wts = jnp.reshape(wts, (-1,1))

    def basis_residual(coeffs_psi: Float[Array, "P alpha"], _):
        # Get first two basis elements using boundary conditions
        first_coeffs = inv_bc_scale @ (
            bc_shift - (basis_eval[[0, -1], keep_orders] @ coeffs_psi)
        )
        # Evaluate the basis and derivatives on full coefficient set
        full_coeffs_psi = jnp.concat((first_coeffs, coeffs_psi))
        psi = basis_eval @ full_coeffs_psi
        psi_diff1 = basis_diff1 @ full_coeffs_psi
        psi_diff2 = basis_diff2 @ full_coeffs_psi
        # Evaluate the residual
        residual = __OptimalInterpPDEResidual_vmap(
            psi, psi_diff1, psi_diff2, phi
        )
        return wts * jnp.abs(residual)

    return jax.jit(basis_residual)


def OptimalInterpBVP_ODE_RHS(
    concat_psi: PsiODET, phi: MomentGeneratingPhi, _: Float
) -> PsiODET:
    """
    Righthand side for the ODE for the optimal interpolant (i.e., no mass matrix)

    :param concat_psi: A concatenation of (psi, psi_dot)
    :type concat_psi: Float[Array, "2*N_terms"]
    :param phi: Moment generating function
    :type phi: MomentGeneratingPhi
    :return: The time derivative of (psi, psi_dot)
    :rtype: Float[Array, "2*N_terms"]
    """
    N_terms = concat_psi.shape[0] // 2
    psi_t, psi_dot_t = concat_psi[:N_terms], concat_psi[N_terms:]
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = calculate_D(Phi, Phi_prime)
    K_tens = calculate_K(Phi)
    C_tens = calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens, K_tens)
    rhs_d_psi = psi_dot_t
    d_psi_dot_contract = jnp.einsum("abg,a,g->b", C_tens, psi_dot_t, psi_dot_t)
    # Note that D is (alpha, beta), so we transpose it before solving
    D_solve = D_tens.T  # + D_infl * jnp.eye(N_terms)
    rhs_d_psi_dot = jnp.linalg.lstsq(D_solve, d_psi_dot_contract)[0]
    return jnp.concat((rhs_d_psi, rhs_d_psi_dot))


def OptimalInterpBVP_DAE_RHS(concat_psi: PsiODET, phi: MomentGeneratingPhi) -> PsiODET:
    r"""
    Righthand side for the DAE formulation of the optimal interpolant

    :param concat_psi: A concatenation of (psi, psi_dot)
    :type concat_psi: Float[Array, "2*N_terms"]
    :param phi: Moment generating function
    :type phi: MomentGeneratingPhi
    :return: The righthand side of $M(y) @ \dot{y} = f(y)$
    :rtype: Float[Array, "2*N_terms"]
    """
    N_terms = concat_psi.shape[0] // 2
    psi_t, psi_dot_t = concat_psi[:N_terms], concat_psi[N_terms:]
    Phi, Phi_prime, Phi_prime_prime = phi.evaluate(psi_t)
    D_tens = calculate_D(Phi, Phi_prime)
    K_tens = calculate_K(Phi)
    C_tens = calculate_C(Phi, Phi_prime, Phi_prime_prime, D_tens, K_tens)
    rhs_d_psi = psi_dot_t
    rhs_d_psi_dot = jnp.einsum("abg,a,g->b", C_tens, psi_dot_t, psi_dot_t)
    return jnp.concat((rhs_d_psi, rhs_d_psi_dot))


def OptimalInterpBVP_DAE_LHS(
    concat_d_psi: PsiODET, concat_psi: PsiODET, phi: MomentGeneratingPhi
) -> PsiODET:
    r"""
    Lefthand side for the DAE formulation of the optimal interpolant

    :param concat_d_psi: A concatenation of d_t (psi, psi_dot)
    :type concat_psi: Float[Array, "2*N_terms"]
    :param concat_psi: A concatenation of (psi, psi_dot)
    :type concat_psi: Float[Array, "2*N_terms"]
    :param phi: Moment generating function
    :type phi: MomentGeneratingPhi
    :return: The matrix $M$ satisfying $M(y) @ \dot{y} = f(y)$
    :rtype: Float[Array, "2*N_terms"]
    """
    N_terms = concat_psi.shape[0] // 2
    psi_t = concat_psi[:N_terms]
    d_psi_t, d_psi_dot_t = concat_d_psi[:N_terms], concat_d_psi[N_terms:]
    Phi, Phi_prime, _ = phi.evaluate(psi_t)
    D_tens = calculate_D(Phi, Phi_prime)
    lhs_d_psi = d_psi_t
    lhs_d_psi_dot = jnp.einsum("ab,a->b", D_tens, d_psi_dot_t)
    return jnp.concat((lhs_d_psi, lhs_d_psi_dot))
