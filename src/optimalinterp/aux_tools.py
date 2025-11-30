import jax.numpy as jnp


def circ_convolution(arr_1, arr_2, domain_type):
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

    def circ_convolve_time(x, h):
        x = jnp.asarray(x)
        h = jnp.asarray(h)
        assert x.shape == h.shape
        X = jnp.fft.fft(x)
        H = jnp.fft.fft(h)
        return jnp.fft.ifft(X * H)

    def circ_convolve_freq(X, H):
        X = jnp.asarray(X)
        H = jnp.asarray(H)
        assert X.shape == H.shape
        x = jnp.fft.ifft(X)
        h = jnp.fft.ifft(H)
        return jnp.fft.fft(x * h)

    if domain_type == "time":
        return circ_convolve_time(arr_1, arr_2)
    elif domain_type == "freq":
        return circ_convolve_freq(arr_1, arr_2)
    else:
        raise ValueError("domain_type must be 'time' or 'freq'")
