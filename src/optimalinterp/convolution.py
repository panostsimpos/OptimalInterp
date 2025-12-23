import jax.numpy as jnp
import chex
from enum import IntEnum
import jax
from functools import partial


class CONVOLUTION_DOMAIN(IntEnum):
    TIME = 1
    FREQ = 2


@jax.jit
def circ_convolve_time(x, h):
    chex.assert_equal_shape((x, h))
    X = jnp.fft.fft(x)
    H = jnp.fft.fft(h)
    return jnp.fft.ifft(X * H)


@jax.jit
def circ_convolve_freq(X, H):
    chex.assert_equal_shape((X, H))
    x = jnp.fft.ifft(X)
    h = jnp.fft.ifft(H)
    return jnp.fft.fft(x * h)


@partial(jax.jit, static_argnames=["domain_type"])
def circ_convolution(arr_1, arr_2, domain_type: CONVOLUTION_DOMAIN):
    """
    Circular convolution of two arrays using FFTs.
    Note:
    ------
    Because jax.numpy.fft jas no normalization but jax.numpy.ifft normalized by 1/N,
    where N = len(arr_1) == len(arr_2) so making a distinction between time and frequency domain matters.

    Inputs:
    -------
    arr_1: (N,) array
        First input array
    arr_2: (N,) array
        Second input array
    domain_type: CONVOLUTION_DOMAIN
        Domain type of the input arrays (TIME or FREQ)
    Returns:
    --------
    conv_out: (N,) array
        Circular convolution output
    """
    tmp = (domain_type - CONVOLUTION_DOMAIN.FREQ) * (
        domain_type - CONVOLUTION_DOMAIN.TIME
    )
    chex.assert_equal(tmp, 0)
    return jax.lax.cond(
        domain_type == CONVOLUTION_DOMAIN.TIME,
        circ_convolve_time,
        circ_convolve_freq,
        arr_1,
        arr_2,
    )


@jax.jit
def triple_circ_convolve_time(x, h, g):
    chex.assert_equal_shape((x, h))
    chex.assert_equal_shape((h, g))
    X = jnp.fft.fft(x)
    H = jnp.fft.fft(h)
    G = jnp.fft.fft(g)
    return jnp.fft.ifft(X * H * G)


@jax.jit
def triple_circ_convolve_freq(X, H, G):
    chex.assert_equal_shape((X, H))
    chex.assert_equal_shape((H, G))
    x = jnp.fft.ifft(X)
    h = jnp.fft.ifft(H)
    g = jnp.fft.ifft(G)
    return jnp.fft.fft(x * h * g)
