from typing import NamedTuple
from jaxtyping import Float, Array
from optimalinterp.stochastic_basis import StochasticBasis


class OptimalInterpolant(NamedTuple):
    """Optimal interpolaent container"

    Attributes:
        t: Time points of shape (T,)
        psi: Coefficient trajectories ψ_α(t) of shape (T, N)
        psi_dot: Velocity trajectories dψ_α/dt of shape (T, N)
        Z: Random variable functions Z_α(omega) of shape (N,)
    """

    t: Float[Array, " T"]
    Z: StochasticBasis
    psi: Float[Array, "T N"] = None
    psi_dot: Float[Array, "T N"] = None
