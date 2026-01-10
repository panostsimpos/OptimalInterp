import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from abc import ABC, abstractmethod
import optimalinterp as oi
import equinox as eqx
from typing import Optional

class StochasticBasis(eqx.Module, ABC):

    N_modes: eqx.AbstractVar[int]

    @abstractmethod
    def sample(
        self, key: jax.Array, N_samples: int
    ) -> Float[Array, "N_samples N_basis"]:
        r"""
        Sample from the random basis Z = (Z_1, Z_2, ..., Z_N) used to form the optimal interpolant via
        X_t = Σ_{α=1}^N Z_α ψ_α(t).

        Args:
            key: JAX PRNG key.
            N_samples: Number of samples to generate.

        Returns:
            Samples of shape (N_samples, N_basis).
        """
        pass

    @abstractmethod
    def moment_generating_phi_builder(self, N_outputs: int) -> oi.moment_generator.MomentGeneratingPhi:
        pass

    def build_moment_generating_phi(self, N_outputs: Optional[int] = 0) -> oi.moment_generator.MomentGeneratingPhi:
        r"""
        Build the MomentGeneratingPhi object corresponding to this stochastic basis.

        Args:
            N_outputs: Number of outputs that the phi function should have

        Returns:
            An instance of MomentGeneratingPhi.
        """
        N_outputs_true = jax.lax.cond(N_outputs == 0, lambda: self.N_modes, lambda: N_outputs)
        return self.moment_generating_phi_builder(N_outputs_true)


class GaussianConvolutionBasis(StochasticBasis):
    r"""
    Stochastic basis where
    Z_0 ~ N(0, 1),
    Z_N ~ N(mean, std_dev^2),
    Z_alpha = (1 - alpha/N) * N(0, 1) + (alpha/N) * N(mean,std_dev^2)
    for alpha = 1, ..., N-1.
    """

    N_modes: int = eqx.field()
    mean: float
    std_dev: float

    def sample(self, key, N_samples):
        alpha_ratios = jnp.linspace(0, 1, self.N_modes)
        mean_Z = alpha_ratios * self.mean
        Cov_Z = jnp.diag(
            (1 - alpha_ratios) ** 2 + (alpha_ratios**2) * self.std_dev**2
        )  # Shape (N_basis, N_basis)

        base_randomness = jax.random.normal(
            key, shape=(N_samples, self.N_modes))
        samples = mean_Z + base_randomness @ jnp.sqrt(
            Cov_Z
        )  # Because diagonal, Cholesky is just sqrt of diag
        return samples

    def moment_generating_phi_builder(self, N_outputs):
        return oi.moment_generator.GaussianPhi1D(
            N_outputs, mu=self.mean, sigma=self.std_dev
        )
