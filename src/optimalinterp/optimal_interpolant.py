from typing import NamedTuple
from jaxtyping import Float, Array
from optimalinterp.stochastic_basis import StochasticBasis
import equinox as eqx
from typing import Optional
import jax.numpy as jnp
import jax


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
        )  # Shape (N_basis, M)

        # Sample Z
        Z_sampled = self.Z.sample(
            N_samples=N_samples, key=key
        )  # Shape (N_samples, N_basis)

        # Compute X_t = Σ_{α=1}^N Z_α ψ_α(t)
        X_t = Z_sampled @ psi_eval  # Shape (N_samples, M)
        X_t = X_t  # Shape (N_samples, M)

        return X_t
