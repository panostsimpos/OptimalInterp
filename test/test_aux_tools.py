import pytest
import jax
import jax.numpy as jnp
import optimalinterp as oi

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def trig_poly(x, sin_coeff, cos_coeff, constant_coeff):
    sin_terms = jnp.dot(
        sin_coeff, jnp.sin(2 * jnp.pi * jnp.arange(1, 1 + len(sin_coeff)) * x)
    )
    cos_terms = jnp.dot(
        cos_coeff, jnp.cos(2 * jnp.pi * jnp.arange(1, 1 + len(cos_coeff)) * x)
    )
    return sin_terms + cos_terms + constant_coeff


def trig_poly_euler(x, all_coeff):
    assert len(all_coeff) % 2 == 1
    max_idx = len(all_coeff) // 2
    return jnp.dot(
        all_coeff, jnp.exp(2j * jnp.pi * x * jnp.arange(-max_idx, max_idx + 1))
    )
    # sin_terms = jnp.dot(sin_coeff, jnp.sin(2 * jnp.pi * jnp.arange(1,1+len(sin_coeff)) * x))
    # cos_terms = jnp.dot(cos_coeff, jnp.cos(2 * jnp.pi * jnp.arange(1,1+len(cos_coeff)) * x))
    # return sin_terms + cos_terms + constant_coeff


trig_poly_v = jax.vmap(trig_poly, in_axes=(0, None, None, None))
trig_poly_euler_v = jax.vmap(trig_poly_euler, in_axes=(0, None))


def trig_poly_coeffs_to_euler(sin_coeff, cos_coeff, constant_coeff):
    # 2 sin(2pi k x) = -i exp(2pi i k x) + i exp(2pi i (-k) x)
    # 2 cos(2pi k x) =    exp(2pi i k x) +   exp(2pi i (-k) x)
    assert len(sin_coeff) == len(cos_coeff) and jnp.isscalar(constant_coeff)
    sin_coeff_eu = jnp.concat((1j * sin_coeff[::-1], jnp.array([0.0]), -1j * sin_coeff))
    cos_coeff_eu = jnp.concat((cos_coeff[::-1], jnp.array([0.0]), cos_coeff))
    all_coeffs_eu = 0.5 * (sin_coeff_eu + cos_coeff_eu)
    all_coeffs_eu = all_coeffs_eu.at[len(sin_coeff)].set(constant_coeff)
    return all_coeffs_eu


def test_fourier_coeffs_util():
    # Test function: 1.3 + 2 sin(2 pi x) - 2 cos(4 pi x) + 6 * sin(6 pi x)
    constant_coeff = 1.3
    sin_coeff = jnp.array([2, 0, 6])
    cos_coeff = jnp.array([0, -2, 0])
    all_coeffs = trig_poly_coeffs_to_euler(sin_coeff, cos_coeff, constant_coeff)
    N = len(all_coeffs)

    evals_guess = oi.util.fourier_coeffs_to_evals(all_coeffs)
    all_coeffs_guess = oi.util.evals_to_fourier_coeffs(evals_guess)
    assert len(evals_guess) == N + 1
    assert len(all_coeffs_guess) == N
    xgrid = jnp.arange(len(evals_guess), dtype=jnp.complex_) / len(evals_guess)
    evals_true = trig_poly_v(xgrid, sin_coeff, cos_coeff, constant_coeff)
    assert evals_guess == pytest.approx(evals_true)
    assert all_coeffs_guess == pytest.approx(all_coeffs)


def test_fourier_coeffs_util_difficult():
    # Test function: 1.3 + 2 sin(2 pi x) - 2 cos(4 pi x) + 6 * sin(6 pi x)
    # constant_coeff = 1.3
    # sin_coeff = jnp.randn([ 2,  0,  6])
    # cos_coeff = jnp.array([ 0, -2,  0])
    # all_coeffs = trig_poly_coeffs_to_euler(sin_coeff, cos_coeff, constant_coeff)
    max_idx = 7
    N = 2 * max_idx + 1
    key, subkey = jax.random.split(jax.random.key(0))
    all_coeffs = jax.random.normal(key, (N,)) + 1j * jax.random.normal(subkey, (N,))

    evals_guess = oi.util.fourier_coeffs_to_evals(all_coeffs)
    all_coeffs_guess = oi.util.evals_to_fourier_coeffs(evals_guess)
    assert len(evals_guess) == N + 1
    assert len(all_coeffs_guess) == N
    xgrid = jnp.arange(len(evals_guess), dtype=jnp.complex_) / len(evals_guess)
    evals_true = trig_poly_euler_v(xgrid, all_coeffs)
    assert evals_guess == pytest.approx(evals_true)
    assert all_coeffs_guess == pytest.approx(all_coeffs)
