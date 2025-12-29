import jax.numpy as jnp

__all__ = ["clenshaw_curtis"]


def clenshaw_curtis(N: int):
    """
    Get the N-point Clenshaw-Curtis rule. Implementation based on Fabien Le Floc'h
    https://chasethedevil.github.io/post/clenshaw_fft_implementation/
    """
    x = jnp.cos(jnp.arange(N) * jnp.pi / (N - 1))
    c = jnp.concat((jnp.array([2.]), 2 / (1 - jnp.arange(2, N)[::2]**2)))
    c = jnp.concat((c, c[1:((N // 2) + (1 - N % 2))][::-1]))
    w = jnp.real(jnp.fft.ifft(c))
    w = w.at[0].set(w[0] / 2)
    w = jnp.concat((w, w[0:1]))
    return x, w
