import jax.numpy as jnp
from jaxtyping import Float, Array

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

def fourier_coeffs_to_evals(coeffs: Float[Array, " N"]):
    r"""
    Given a vector of Fourier coefficients $C_\gamma$, calculate the vector
    $$ v_j = \sum_{\gamma} C_\gamma \exp( i \gamma x_j ) $$
    where $x_j = 2\pi (j-1) / N$
    """
    # Forward makes sure normalization is correct
    return jnp.fft.ifft(coeffs, norm='forward')

def evals_to_fourier_coeffs(eval_pts: Float[Array, " N"]):
    r"""
    Given a vector of function evaluations $v_j = f(x_j)$, calculate the Fourier coefficients s.t.
    $$ v_j = \sum_{\gamma} C_\gamma \exp( i \gamma x_j ) $$
    where $ x_j = 2\pi (j-1) / N $
    """
    return jnp.fft.fft(eval_pts, norm='forward')