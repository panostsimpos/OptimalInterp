from typing import NamedTuple
from jaxtyping import Float, Array
from optimalinterp.stochastic_basis import StochasticBasis
import equinox as eqx
from typing import Optional
import jax.numpy as jnp
import jax
import optimalinterp.shooting as shooting


class OptimalInterpolant(eqx.Module):
    """Optimal interpolaent container"

    Attributes:
        t: Time points of shape (T,)
        psi: Coefficient trajectories ψ_α(t) of shape (T, N)
        psi_dot: Velocity trajectories dψ_α/dt of shape (T, N)
        Z: Random variable functions Z_α(omega) of shape (N,)
    """

    t: Float[Array, " T"]
    Z: StochasticBasis
    psi: Optional[Float[Array, "T N"]] = None
    psi_dot: Optional[Float[Array, "T N"]] = None
    has_solution: bool = False

    def with_psi(self, psi):
        """
        Return a new OptimalInterpolant with updated psi.
        """
        return eqx.tree_at(lambda m: m.psi, self, psi, is_leaf=lambda x: x is None)

    def with_psi_dot(self, psi_dot):
        """
        Return a new OptimalInterpolant with updated psi_dot.
        """
        return eqx.tree_at(
            lambda m: m.psi_dot, self, psi_dot, is_leaf=lambda x: x is None
        )

    def with_has_solution(self, has_solution: bool):
        return eqx.tree_at(lambda m: m.has_solution, self, has_solution)

    def with_t(self, t: Float[Array, " T"]):
        return eqx.tree_at(lambda m: m.t, self, t)

    def __call__(
        self, key: jax.Array, t_eval: Float[Array, "M"], N_samples: int
    ) -> Float[Array, "M N_samples"]:
        r"""
        Evaluate the optimal interpolant at given time points.

        Args:
            t_eval: Time points of shape (M,)
            N_samples: Number of samples to generate.
            key: JAX PRNG key.

        Returns:
            Interpolated values X_t of shape (N_samples, M) at times t_eval.
        """
        if self.psi is None:
            raise ValueError(
                "Interpolant coefficients have not been computed yet. Cannot evaluate."
            )

        # Interpolate psi at t_eval
        psi_eval = jnp.array(
            [
                jnp.interp(t_eval, self.t, self.psi[:, n])
                for n in range(self.psi.shape[1])
            ]
        )  # Shape (N_basis, M) #TODO: Make this JIT compatible

        # psi_eval = jax.vmap(
        #     lambda col: jnp.interp(t_eval, self.t, col), in_axes=1, out_axes=0
        # )(self.psi)

        # Sample Z
        Z_samples = self.Z.sample(
            N_samples=N_samples, key=key
        )  # Shape (N_samples, N_basis)

        # Compute X_t = Σ_{α=1}^N Z_α ψ_α(t)
        X_t = Z_samples @ psi_eval  # Shape (N_samples, M)
        X_t = X_t  # Shape (N_samples, M)

        return X_t


def compute_optimal_psi(
    interpolant: OptimalInterpolant,
    allow_failure: bool = False,
    **solver_kwargs,
) -> OptimalInterpolant:
    r"""
    Compute the optimal coefficient trajectories ψ(t) for the given optimal interpolant
    by solving the boundary value problem using the shooting method.

    Args:
        interpolant: An instance of OptimalInterpolant with defined Z and t.
        allow_failure: If true, use output of shooting regardless of failure
        **solver_kwargs: Optional keyword arguments for the ODE solver. Supported options:
            - D_infl: Inflation parameter (default: 0.0)
            - t_span: Interval containing time lower and upper bound (default: (0.0, 1.0))
            - n_time_points: Number of time discretization points (default: 150)
            - rtol: Relative tolerance (default: 1e-8)
            - atol: Absolute tolerance (default: 1e-8)
            - max_solver_steps: Maximum solver iterations (default: 5000)
            - verbose: Print solver progress (default: True)
            - plot_solution: Generate solution plots (default: True)
            - return_real_part: Return only real part of solution (default: True)

    Returns:
        An instance of OptimalInterpBVPSolution containing the computed trajectories and metadata.
    """
    # Default parameters
    defaults = {
        "D_infl": 0.0,
        "t_span": (0.0, 1.0),
        "n_time_points": 150,
        "rtol": 1e-8,
        "atol": 1e-8,
        "max_solver_steps": 5000,
        "N_optimizer_steps": 1000,
        "verbose": True,
        "plot_solution": True,
        "return_real_part": True,
    }

    # Merge user kwargs with defaults (user values override defaults)
    merged_kwargs = {**defaults, **solver_kwargs}

    # Obtain Phi and solve using shooting method
    stochastic_basis = interpolant.Z
    Phi = stochastic_basis.build_moment_generating_phi()
    N_terms = stochastic_basis.N_basis

    bvp_soln = shooting.solve(Phi=Phi, N_terms=N_terms, **merged_kwargs)

    # Unpack solution and update interpolant

    def update_interpolant(
        interpolant: OptimalInterpolant, t, psi, psi_dot, optimization_success, solver_success
    ) -> OptimalInterpolant:
        def on_success(interpolant: OptimalInterpolant, t, psi, psi_dot):
            interpolant = interpolant.with_psi(psi)
            interpolant = interpolant.with_psi_dot(psi_dot)
            interpolant = interpolant.with_has_solution(True)
            interpolant = interpolant.with_t(t)
            return interpolant

        def on_failure(interpolant: OptimalInterpolant, t, psi, psi_dot):
            if allow_failure:
                jax.debug.print(
                    "Warning: BVP solver did not succeed. Returning failed output..."
                )
                return on_success(interpolant, t, psi, psi_dot)
            else:
                jax.debug.print(
                    "Warning: BVP solver did not succeed. Returning interpolant without updated psi and psi_dot.\n" \
                    "Use allow_failure=True if you want the output to be returned"
                )
                interpolant = interpolant.with_psi(jnp.zeros_like(psi))
                interpolant = interpolant.with_psi_dot(jnp.zeros_like(psi_dot))
                interpolant = interpolant.with_has_solution(False)
                interpolant = interpolant.with_t(t)
                return interpolant

        return jax.lax.cond(
            optimization_success and solver_success,
            on_success,
            on_failure,
            *(interpolant, t, psi, psi_dot),
        )

    return update_interpolant(
        interpolant,
        bvp_soln.t,
        bvp_soln.psi,
        bvp_soln.psi_dot,
        bvp_soln.optimization_success,
        bvp_soln.solver_success,
    )
