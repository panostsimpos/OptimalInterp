from typing import NamedTuple
from jaxtyping import Float, Array


class OptimalInterpolant(NamedTuple):
    """Optimal interpolaent container"

    Attributes:
        t: Time points of shape (T,)
        psi: Coefficient trajectories ψ_α(t) of shape (T, N)
        psi_dot: Velocity trajectories dψ_α/dt of shape (T, N)
        Z: Random variable functions Z_α(omega) of shape (N,)
    """

    t: Float[Array, " T"]
    psi: Float[Array, "T N"]
    psi_dot: Float[Array, "T N"]
    Z: callable  # TODO: I want Z to be a class containing a method called sample that returns Float[Array, "N"] values for the Z_alpha variables to subsequently form the optimal interpolant X_t = Z @ psi(t)
