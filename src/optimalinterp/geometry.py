import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def project_on_torus(
    x: Float[Array, "N_samples"], period: Float
) -> Float[Array, "N_samples"]:
    """
    Truncate samples x onto a torus with given period.

    :param x: Input samples.
    :param period: Period of the torus.
    :return: Truncated samples in [-period/2, period/2).
    """
    return jnp.mod(x + period / 2, period) - period / 2
