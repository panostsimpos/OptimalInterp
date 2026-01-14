from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from typing import Tuple
import jax
from . import util

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


class WrappedGaussianBridgePhi1D(MomentGeneratingPhi):
    def __init__(self, N_inputs: int, N_outputs: int, max_period_idx: int, mu: float, sigma: float):
        r"""
        Build MomentGeneratingPhi for 1D Wrapped Gaussian example where we take
        Z_0 ~ N(0, 1),
        Z_N ~ N(mu, sigma^2),
        Z_alpha = (1 - alpha/N) * N(0, 1) + (alpha/N) * N(mu,sigma^2)
        for alpha = 1, ..., N-1.

        Args:
            N_inputs: Number of alpha terms to take
            N_outputs: Number of beta terms to take
            max_period_idx: maximum of period terms to take in wrapping
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
        self.N_inputs = N_inputs
        self.max_period_idx = max_period_idx
        frac_1 = jnp.arange(self.N_inputs).reshape(-1,1) / (self.N_inputs - 1)
        k_2 = jnp.arange(-self.max_period_idx, self.max_period_idx + 1).reshape(1,-1)

        mu_term = 1j * k_2 * mu * frac_1
        sig_term_1 = ((1 - frac_1) * k_2)**2
        sig_term_2 = (frac_1 * k_2 * sigma)**2
        self.k = k_2
        # (alpha, k)
        self.wrap_coeffs = jnp.exp(mu_term - 0.5 * (sig_term_1 + sig_term_2))

    def evaluate(self, psi_t):
        # --------------------------------------------
        # Convention: suffix _1 to indicate length N_terms
        # Convention: suffix _2 to indicate length N_outputs
        # --------------------------------------------
        # psi_t is size (N_alpha,)
        # wrapped gaussian axes: (k, alpha, beta)
        phi_input = -psi_t.reshape(1, -1, 1) * jnp.arange(self.N_outputs).reshape(1, 1, -1)
        dist = (phi_input - self.k.reshape(-1, 1, 1))
        omega = (1j * jnp.pi) # rotation
        exp = jnp.exp(omega * dist)
        exp_diff, exp_diff2 = omega * exp, omega * omega * exp
        # Sinc has pi already inside of it
        sinc, sinc_diff, sinc_diff2 = util.sinc_diff2(dist)
        summand_evals = exp * sinc
        summand_diff1 = exp * sinc_diff + exp_diff * sinc
        summand_diff2 = exp * sinc_diff2 + 2 * exp_diff * sinc_diff + exp_diff2 * sinc
        phi_evals = jnp.einsum('ak,kab->ab', self.wrap_coeffs, summand_evals)
        phi_diff1 = jnp.einsum('ak,kab->ab', self.wrap_coeffs, summand_diff1)
        phi_diff2 = jnp.einsum('ak,kab->ab', self.wrap_coeffs, summand_diff2)
        return phi_evals, phi_diff1, phi_diff2
