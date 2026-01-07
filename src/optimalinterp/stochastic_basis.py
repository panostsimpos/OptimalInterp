import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from abc import ABC, abstractmethod
import optimalinterp as oi
import equinox as eqx


class StochasticBasis(eqx.Module, ABC):

    N_basis: eqx.AbstractVar[int]

    @abstractmethod
    def sample(
        self, key: jax.Array, N_samples: int = 1
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
    def build_moment_generating_phi(self) -> oi.MomentGeneratingPhi:
        r"""
        Build the MomentGeneratingPhi object corresponding to this stochastic basis.

        Args:
            N_terms: Number of basis functions / terms in the interpolant.

        Returns:
            An instance of MomentGeneratingPhi.
        """
        pass


class GaussianBasis(StochasticBasis):

    N_basis: int = eqx.field(static=True)
    mean: float
    std_dev: float
    bridge_type: str = eqx.field(static=True, default="gaussian_convolution")

    def sample(
        self, key: jax.Array, N_samples: int = 1
    ) -> Float[Array, "N_samples N_basis"]:
        r"""
        Sample from the random variables
        Z_0 ~ N(0, 1),
        Z_N ~ N(mean, std_dev^2),
        Z_alpha = (1 - alpha/N) * N(0, 1) + (alpha/N) * N(mean,std_dev^2)
        for alpha = 1, ..., N-1.

        Args:
            key: JAX PRNG key.
            N_samples: Number of samples to generate.

        Returns:
            Samples of shape (N_samples, N_basis).
        """
        if self.bridge_type == "gaussian_convolution":
            alpha_ratios = jnp.linspace(0, 1, self.N_basis)
            mean_Z = alpha_ratios * self.mean
            Cov_Z = jnp.diag(
                (1 - alpha_ratios) ** 2 + (alpha_ratios**2) * self.std_dev**2
            )  # Shape (N_basis, N_basis)

            base_randomness = jax.random.normal(key, shape=(N_samples, self.N_basis))
            samples = mean_Z + base_randomness @ jnp.sqrt(
                Cov_Z
            )  # Because diagonal, Cholesky is just sqrt of diag
            return samples
        else:
            raise NotImplementedError(
                f"Bridge type {self.bridge_type} not implemented."
            )

    def build_moment_generating_phi(self) -> oi.MomentGeneratingPhi:
        return oi.moment_generator.GaussianPhi1D(mu=self.mean, sigma=self.std_dev)
