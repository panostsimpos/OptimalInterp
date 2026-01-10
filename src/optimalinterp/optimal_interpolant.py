from jaxtyping import Float, Array
from .stochastic_basis import StochasticBasis
from . import basis, shooting
import equinox as eqx
from typing import Optional
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

__all__ = [
    "modal_interpolant_factory",
    "ModalInterpolant",
    "compute_optimal_psi_basis",
    "compute_optimal_psi_shooting"
]


def psi_interp_pt(t_eval: Float[Array, " T"], t: Float[Array, " T_new"], psi_alpha: Float[Array, " T"]):
    "Interpolate for one given psi"
    return jnp.interp(t_eval, t, psi_alpha)


"Interpolate for many psis"
psi_interp_grid = jax.vmap(psi_interp_pt, in_axes=(None, None, 1), out_axes=1)


class ModalInterpolant(eqx.Module, ABC):
    Z: StochasticBasis

    @abstractmethod
    def has_solution(self) -> bool:
        pass

    def default_tgrid(self) -> Array:
        return jnp.linspace(0., 1., 101)

    @abstractmethod
    def sample_solution_and_deriv(self, key: jax.Array, N_samples: int, t_eval: Float[Array, " M"] | None) -> tuple[Float[Array, "M N_samples"], Float[Array, "M N_samples"]]:
        pass

    def sample_solution(self, key: jax.Array, N_samples: int, t_eval: Float[Array, " M"] | None) -> Float[Array, "M N_samples"]:
        return self.sample_solution_and_deriv(key, N_samples, t_eval)[0]

    def sample_modes(self, key: Array, N_samples: int) -> Float[Array, " N_terms"]:
        return self.Z.sample(key, N_samples)

    def __call__(self, key: jax.Array, N_samples: int, t_eval: Optional[Float[Array, " M"]] = None) -> Float[Array, "M N_samples"]:
        r"""
        Sample the optimal interpolant at given time points.

        Args:
            key: JAX PRNG key.
            N_samples: Number of samples to generate.
            t_eval: Time points of shape (M,)

        Returns:
            Interpolated samples X_t of shape (N_samples, M) at times t_eval.
        """
        if not self.has_solution():
            raise ValueError(
                "Interpolant is not computed, cannot evaluate."
            )
        return self.sample_solution(key, N_samples, t_eval)


def modal_interpolant_factory(which_kind: str, *args, **kwargs) -> ModalInterpolant:
    constructors = {
        "time interpolated": TimeInterpolatedModalInterpolant,
        "basis parameterized": BasisParameterizedModalInterpolant
    }
    constructor = constructors.get(which_kind.lower(), None)
    if constructor is None:
        raise ValueError(f"Invalid interpolant type {which_kind}")
    return constructor(*args, **kwargs)


class TimeInterpolatedModalInterpolant(ModalInterpolant):
    r"""
    Optimal interpolant for shooting BVP solution

    Attributes:
        t: Time points of shape (T,)
        Z: Random variable functions Z_α(omega) of shape (N,)
        psi: Coefficient trajectories ψ_α(t) of shape (T, N)
        psi_dot: Velocity trajectories dψ_α/dt of shape (T, N)
    """
    t: Float[Array, " T"]
    psi: Optional[Float[Array, "T N"]] = None
    psi_dot: Optional[Float[Array, "T N"]] = None

    def has_solution(self):
        return self.psi is not None and self.psi_dot is not None

    def default_tgrid(self):
        return self.t

    def sample_solution_and_deriv(self, key, N_samples, t_eval):
        assert self.psi is not None and self.psi_dot is not None
        # Interpolate psi at t_eval (N_times, N_terms)
        if t_eval is None:
            psi_eval, psi_dot_eval = self.psi, self.psi_dot
        else:
            psi_eval = psi_interp_grid(t_eval, self.t, self.psi)
            psi_dot_eval = psi_interp_grid(t_eval, self.t, self.psi_dot)

        # Sample Z
        Z_samples = self.sample_modes(key, N_samples) # Shape (N_samples, N_terms)

        # Compute X_t = Σ_{α=1}^N Z_α ψ_α(t)
        X_t = Z_samples @ psi_eval.T  # Shape (N_samples, N_times)
        X_t_dot = Z_samples @ psi_dot_eval.T  # Shape (N_samples, N_times)
        return X_t, X_t_dot


def compute_optimal_psi_shooting(
    stochastic_basis: StochasticBasis,
    allow_failure: bool = False,
    **solver_kwargs,
) -> ModalInterpolant:
    r"""
    Compute the optimal coefficient trajectories ψ(t) for the given optimal interpolant
    by solving the boundary value problem using the shooting method.

    Args:
        stochastic_basis: Modal basis for random variables
        allow_failure: If true, use output of shooting regardless of failure
        **solver_kwargs: see `shooting.solve`

    Returns:
        An instance of OptimalInterpBVPSolution containing the computed trajectories and metadata.
    """

    # Obtain Phi and solve using shooting method
    Phi = stochastic_basis.build_moment_generating_phi()
    N_terms = stochastic_basis.N_modes

    bvp_soln = shooting.solve(Phi, N_terms, **solver_kwargs)

    # Unpack solution and update interpolant
    def on_success(t, psi, psi_dot):
        return TimeInterpolatedModalInterpolant(stochastic_basis, t, psi, psi_dot)

    def on_failure(t, psi, psi_dot):
        if allow_failure:
            jax.debug.print(
                "Warning: BVP solver did not succeed. Returning failed output..."
            )
            return on_success(t, psi, psi_dot)
        else:
            jax.debug.print(
                "Warning: BVP solver did not succeed. Returning interpolant without psi and psi_dot.\n"
                "Use allow_failure=True if you want the output to be returned"
            )
            return TimeInterpolatedModalInterpolant(stochastic_basis, t)

    return jax.lax.cond(
        bvp_soln.optimization_success and bvp_soln.solver_success,
        on_success,
        on_failure,
        bvp_soln.t, bvp_soln.psi, bvp_soln.psi_dot,
    )


class BasisParameterizedModalInterpolant(ModalInterpolant):
    r"""
    Optimal interpolant for basis BVP solution

    Attributes:
        Z: Random variable functions Z_α(omega) of shape (N,)
        t_grid: time grid used for learning the coefficients
        psi: Linear basis parameterizing modal functions ψ_α(t) of shape (N_shap,)
        psi_coeff: Coefficients for the basis functions of psi of shape (N_shap, N)
    """
    psi: basis.AbstractLinearBasis
    psi_coeff: Optional[Float[Array, "N_shap N"]] = None

    def has_solution(self):
        return self.psi_coeff is None

    def sample_solution_and_deriv(self, key, N_samples, t_eval):
        assert self.psi_coeff is not None
        if t_eval is None:
            t_eval = jnp.linspace(0, 1, 1001)
        # Interpolate psi at t_eval
        psi_basis_eval, psi_diffs_eval = self.psi.evaluate_basis_diff(t_eval)
        psi_evals = psi_basis_eval @ self.psi_coeff  # (N_times, N_basis)
        psi_diffs = psi_diffs_eval @ self.psi_coeff  # (N_times, N_basis)

        # Sample Z
        Z_samples = self.sample_modes(key, N_samples) # Shape (N_samples, N_basis)

        # Compute X_t = Σ_{α=1}^N Z_α ψ_α(t)
        X_t = Z_samples @ psi_evals.T  # Shape (N_samples, M)
        X_t_dot = Z_samples @ psi_diffs.T  # Shape (N_samples, M)

        return X_t, X_t_dot


def compute_optimal_psi_basis(
    stochastic_basis: StochasticBasis,
    allow_failure: bool = False,
    **solver_kwargs,
) -> ModalInterpolant:
    r"""
    Compute the optimal coefficient trajectories ψ(t) for the given optimal interpolant
    by solving the boundary value problem using a basis for ѱ

    Args:
        stochastic_basis: Modal basis for random variables
        **solver_kwargs: see `basis.solve`

    Returns:
        An instance of OptimalInterpBVPSolution containing the computed trajectories and metadata.
    """

    Phi = stochastic_basis.build_moment_generating_phi()
    N_terms = stochastic_basis.N_basis
    # Set up the points and weights
    bvp_soln = basis.solve(Phi, N_terms, **solver_kwargs)

    # Unpack solution and update interpolant
    def on_success(stochastic_basis, psi_basis, coeffs_psi):
        return BasisParameterizedModalInterpolant(stochastic_basis, psi_basis, coeffs_psi)

    def on_failure(stochastic_basis, psi_basis, coeffs_psi):
        if allow_failure:
            jax.debug.print(
                "Warning: BVP solver did not succeed. Returning failed output..."
            )
            return on_success(stochastic_basis, psi_basis, coeffs_psi)
        else:
            jax.debug.print(
                "Warning: BVP solver did not succeed. Returning interpolant without psi and psi_dot.\n"
                "Use allow_failure=True if you want the output to be returned"
            )
            return BasisParameterizedModalInterpolant(stochastic_basis, psi_basis)

    return jax.lax.cond(
        bvp_soln.success,
        on_success,
        on_failure,
        stochastic_basis, bvp_soln.psi_basis, bvp_soln.coeffs_psi
    )
