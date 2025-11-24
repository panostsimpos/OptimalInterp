from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple
__all__ = ['MomentGeneratingPhi']

PhiTens = Float[Array, 'alpha beta']


class MomentGeneratingPhi(ABC):
    @abstractmethod
    def evaluate(self, psi_t: jnp.ndarray) -> Tuple[PhiTens, PhiTens, PhiTens]:
        """
        Returns Phi, Phi', and Phi''.
        Each returned tensor P has indexing P[alpha, beta] = P_alpha(-beta psi_t[alpha])
        """
        pass


class GaussianPhi1D(MomentGeneratingPhi):
    def evaluate(self, psi_t):
        # fill in here
        pass
