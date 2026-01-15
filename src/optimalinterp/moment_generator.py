from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from typing import Tuple
import jax

PsiT = Float[Array, "alpha"]
PhiTens = Complex[Array, "alpha beta"]
__all__ = ["MomentGeneratingPhi", "GaussianPhi1D"]


class MomentGeneratingPhi(ABC):
    _N_outputs: int

    def __init__(self, N_outputs: int):
        self._N_outputs = N_outputs

    @property
    def N_outputs(self):
        return self._N_outputs

    @abstractmethod
    def evaluate(self, psi_t: PsiT) -> Tuple[PhiTens, PhiTens, PhiTens]:
        """
        Returns Phi, Phi', and Phi''.
        Each returned tensor P has indexing P[alpha, beta] = P_alpha(-beta psi_t[alpha])
        """
        pass


class GaussianPhi1D(MomentGeneratingPhi):
    def __init__(self, N_outputs: int, mu: float, sigma: float):
        r"""
        Build MomentGeneratingPhi for 1D Gaussian example where we take
        Z_0 ~ N(0, 1),
        Z_N ~ N(mu, sigma^2),
        Z_alpha = (1 - alpha/N) * N(0, 1) + (alpha/N) * N(mu,sigma^2)
        for alpha = 1, ..., N-1.

        Args:
            N_outputs: Number of beta terms to take
            mu: Mean of Gaussian at final time.
            sigma: Standard deviation of Gaussian at final time.
        Outputs:
            (Phi, Phi', Phi''): Each of shape (N_alpha, N_beta) with signature
                Phi[alpha, beta] = \Phi_alpha(-beta psi_t[alpha])
                and similarly for Phi' and Phi''.
        """
        super().__init__(N_outputs)
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, psi_t):
        # --------------------------------------------
        # Convention: suffix _1 to indicate length N_terms
        # Convention: suffix _2 to indicate length N_outputs
        # --------------------------------------------
        mu = self.mu
        sigma = self.sigma
        N_outputs = self.N_outputs
        N_alpha = psi_t.shape[0]
        alpha_1 = jnp.linspace(0, 1, N_alpha)
        beta_2 = jnp.arange(N_outputs)
        phi_input_12 = -psi_t[:, jnp.newaxis] * beta_2[jnp.newaxis, :]
        mean_term_1 = 1j * alpha_1 * mu
        var_term_1 = jnp.square(1 - alpha_1) + jnp.square(sigma * alpha_1)
        phi_eval_12 = jnp.exp(
            mean_term_1[:,jnp.newaxis] * phi_input_12 - 0.5 *jnp.square(phi_input_12) * var_term_1[:,jnp.newaxis]
        )
        phi_dot_prefix = mean_term_1[:,jnp.newaxis] - var_term_1[:,jnp.newaxis] * phi_input_12
        phi_dot_12 = phi_dot_prefix * phi_eval_12
        phi_ddot_12 = -var_term_1[:,jnp.newaxis] * phi_eval_12 + phi_dot_prefix * phi_dot_12
        return (phi_eval_12, phi_dot_12, phi_ddot_12)

    def sample(self, key: Array, num_samples: int) -> Float[Array, " num_samples"]:
        r"""
        Sample from the Gaussian distribution defined by this moment generating function.

        Args:
            key: JAX random key.
            num_samples: Number of samples to draw.
        Returns:
            samples: Array of shape (num_samples,) containing the drawn samples.
        """
        return jax.random.normal(key, shape=(num_samples,)) * self.sigma + self.mu
