import jax.numpy as jnp
import chex
from enum import Enum
import jax


class CONVOLUTION_DOMAIN(Enum):
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


@chex.chexify
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
    domain_type: str
        "time" if inputs are in time domain, "freq" if inputs are in frequency domain
    Returns:
    --------
    conv_out: (N,) array
        Circular convolution output
    """
    tmp = (domain_type.value - CONVOLUTION_DOMAIN.FREQ.value) * \
          (domain_type.value - CONVOLUTION_DOMAIN.TIME.value)
    chex.assert_equal(tmp, 0)
    return jax.lax.cond(
        domain_type == CONVOLUTION_DOMAIN.TIME,
        circ_convolve_time,
        circ_convolve_freq,
        arr_1, arr_2
    )


@jax.jit
def convolve_term(D_alpha, D_gamma, K):
    # Out is length 2*N_terms-1
    temp = jnp.convolve(K, D_gamma, mode="full")
    # Out is length N_terms
    return jnp.convolve(D_alpha, temp, mode="valid")


batch_convolve = jax.vmap(
    jax.vmap(convolve_term, in_axes=(0, None, None), out_axes=0),
    in_axes=(None, 0, None),
    out_axes=2,  # Get out shape alpha,beta,gamma
)
