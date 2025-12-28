from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from typing import Tuple

PsiT = Float[Array, "alpha"]
PhiTens = Complex[Array, "alpha beta"]
__all__ = ["MomentGeneratingPhi", "GaussianPhi1D"]


class MomentGeneratingPhi(ABC):
    @abstractmethod
    def evaluate(self, psi_t: PsiT) -> Tuple[PhiTens, PhiTens, PhiTens]:
        """
        Returns Phi, Phi', and Phi''.
        Each returned tensor P has indexing P[alpha, beta] = P_alpha(-beta psi_t[alpha])
        """
        pass


class GaussianPhi1D(MomentGeneratingPhi):
    def __init__(self, mu: float, sigma: float):
        """
        Build MomentGeneratingPhi for 1D Gaussian example whwere we take
        Z_0 ~ N(0, 1),
        Z_N ~ N(mu, sigma^2),
        Z_alpha = (1 - alpha/N) * N(0, 1) + (alpha/N) * N(mu,sigma^2)
        for alpha = 1, ..., N-1.

        Args:
            mu: Mean of Gaussian at final time.
            sigma: Standard deviation of Gaussian at final time.
        Outputs:
            (Phi, Phi', Phi''): Each of shape (N_terms, N_terms) with signature
                Phi[alpha, beta] = \Phi_alpha(-beta psi_t[alpha])
                and similarly for Phi' and Phi''.
        """
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, psi_t):
        # --------------------------------------------
        # Convention: suffix _s to indicate arrays
        # --------------------------------------------
        mu = self.mu
        sigma = self.sigma
        N_terms = psi_t.shape[0]
        N = N_terms - 1  # Want to index from 0 to N

        alpha_s = jnp.arange(N + 1)
        beta_s = jnp.arange(N + 1)  # Now we do as many DOFs as modes
        eff_mu_s = mu / N * alpha_s
        eff_sigma_s = (1.0 - alpha_s / N) ** 2 + alpha_s**2 / (N**2) * sigma**2
        arg_s = -beta_s[None, :] * psi_t[:, None]
        # Compute Phi, Phi', Phi'' using broadcasting
        Phi_s = jnp.exp(
            1j * eff_mu_s[:, None] * arg_s - 1 / 2 * arg_s**2 * eff_sigma_s[:, None]
        )
        Phi_prime_s = Phi_s * (1j * eff_mu_s[:, None] - eff_sigma_s[:, None] * arg_s)
        Phi_double_prime_s = Phi_s * (
            (1j * eff_mu_s[:, None] - eff_sigma_s[:, None] * arg_s) ** 2
            - eff_sigma_s[:, None]
        )
        return (Phi_s, Phi_prime_s, Phi_double_prime_s)
