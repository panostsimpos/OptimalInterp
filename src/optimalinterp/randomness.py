import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from abc import ABC, abstractmethod


class RandomBasis(ABC):
    @abstractmethod
    def sample(
        self, N_terms: int, key: jax.Array = None, n_samples: int = 1
    ) -> Float[Array, "n_samples n_basis"]:
        r"""
        Sample from the random basis Z = (Z_1, Z_2, ..., Z_N) used to form the optimal interpolant via
        X_t = Σ_{α=1}^N Z_α ψ_α(t).

        Args:
            key: JAX PRNG key.
            n_samples: Number of samples to generate.

        Returns:
            Samples of shape (n_samples, n_basis).
        """
        pass


class GaussianRandomBasis(RandomBasis):
    def __init__(self, N_terms, target_mean: float, target_standard_deviation: float):
        self.N_terms = N_terms
        self.mu = target_mean
        self.sigma = target_standard_deviation

    def sample(
        self, key: jax.Array = None, n_samples: int = 1
    ) -> Float[Array, "n_samples n_basis"]:
        r"""
        Sample from the random variables
        Z_0 ~ N(0, 1),
        Z_N ~ N(mu, sigma^2),
        Z_alpha = (1 - alpha/N) * N(0, 1) + (alpha/N) * N(mu,sigma^2)
        for alpha = 1, ..., N-1.

        Args:
            key: JAX PRNG key.
            n_samples: Number of samples to generate.

        Returns:
            Samples of shape (n_samples, n_basis).
        """
        alpha_ratios = jnp.linspace(0, 1, self.N_terms)
        mean_Z = alpha_ratios * self.mu
        Cov_Z = jnp.diag(
            (1 - alpha_ratios) ** 2 + (alpha_ratios**2) * self.sigma**2
        )  # Shape (N_terms, N_terms)

        base_randomness = jax.random.normal(key, shape=(n_samples, self.N_terms))
        samples = mean_Z + base_randomness @ jnp.sqrt(
            Cov_Z
        )  # Because diagonal, Cholesky is just sqrt of diag
        return samples
