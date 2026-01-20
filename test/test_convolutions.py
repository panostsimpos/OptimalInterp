# %%
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
import optimalinterp.convolution as conv
import optimalinterp.ode_residual as ode_res


# %%
class TestCircConvolveTime:
    """Tests for conv.circ_convolve_time (time-domain circular convolution)."""

    def test_impulse_identity(self):
        """Convolving with impulse [1,0,0,...] returns the original signal."""
        N = 8
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        impulse = jnp.zeros(N).at[0].set(1.0)
        result = conv.circ_convolve_time(x, impulse)
        assert result == pytest.approx(x, rel=1e-12)

    def test_commutativity(self):
        """Circular convolution is commutative: x * h = h * x."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        result_xh = conv.circ_convolve_time(x, h)
        result_hx = conv.circ_convolve_time(h, x)
        assert result_xh == pytest.approx(result_hx, rel=1e-12)

    def test_shift_property(self):
        """Convolving with shifted impulse shifts the signal circularly."""
        N = 8
        x = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Impulse at index 2 should shift by 2
        shift_impulse = jnp.zeros(N).at[2].set(1.0)
        result = conv.circ_convolve_time(x, shift_impulse)
        expected = jnp.zeros(N).at[2].set(1.0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_known_convolution(self):
        """Test against manually computed circular convolution."""
        # For x = [1, 2, 3, 4] and h = [1, 0, 0, 0], result = x
        # For x = [1, 2, 3, 4] and h = [0, 1, 0, 0], result = [4, 1, 2, 3] (circular shift right by 1)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.0, 1.0, 0.0, 0.0])
        result = conv.circ_convolve_time(x, h)
        expected = jnp.array([4.0, 1.0, 2.0, 3.0])
        assert result == pytest.approx(expected, rel=1e-12)

    def test_convolution_with_constant(self):
        """Convolving with constant array gives constant * sum(x)."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        c = 0.5
        h = jnp.ones(4) * c
        result = conv.circ_convolve_time(x, h)
        # Each output element = c * sum(x)
        expected = jnp.ones(4) * (c * jnp.sum(x))
        assert result == pytest.approx(expected, rel=1e-12)

    def test_complex_arrays(self):
        """Circular convolution works with complex arrays."""
        x = jnp.array([1.0 + 1j, 2.0 - 1j, 3.0 + 0j, 4.0 - 2j])
        h = jnp.array([0.5 + 0.5j, -0.5 + 0j, 0.25 - 0.25j, 0.0 + 0.1j])
        result = conv.circ_convolve_time(x, h)
        # Verify commutativity with complex arrays
        result_reverse = conv.circ_convolve_time(h, x)
        assert result == pytest.approx(result_reverse, rel=1e-12)

    def test_linearity_in_first_argument(self):
        """Convolution is linear: (a*x1 + b*x2) * h = a*(x1*h) + b*(x2*h)."""
        x1 = jnp.array([1.0, 2.0, 3.0, 4.0])
        x2 = jnp.array([0.5, -0.5, 1.0, -1.0])
        h = jnp.array([1.0, 0.5, 0.25, 0.125])
        a, b = 2.0, -3.0
        result_combined = conv.circ_convolve_time(a * x1 + b * x2, h)
        result_separate = a * conv.circ_convolve_time(
            x1, h
        ) + b * conv.circ_convolve_time(x2, h)
        assert result_combined == pytest.approx(result_separate, rel=1e-12)

    def test_convolution_is_multiplication_in_freq(self):
        """Verify that FFT of convolution equals product of FFTs."""
        x = jnp.array([1.0, 2.0 + 7j, -3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1 + 3j])
        y = conv.circ_convolve_time(x, h)
        Y = jnp.fft.fft(y)
        X = jnp.fft.fft(x)
        H = jnp.fft.fft(h)
        assert Y == pytest.approx(X * H, rel=1e-12)

    def convolutuon_with_non_trivial_function(self):
        "Compute convolution with exp(i * 2 pi k x) and verify result."
        N = 4
        k = 3
        xgrid = jnp.arange(N)
        freq_func = jnp.exp(1j * 2 * jnp.pi * k * xgrid)
        # Kernel: delta function at index k
        x = jnp.zeros([1.0, -3.0 + 1j, 5.0, 2.0])
        kernel = jnp.ifft(freq_func)
        result_conv = conv.circ_convolve_time(x, kernel)
        result_freq = jnp.fft.fft(x) * freq_func
        result_conv_freq = jnp.fft.fft(result_conv)
        assert result_conv_freq == pytest.approx(result_freq, rel=1e-12)


class TestCircConvolveFreq:
    """Tests for conv.circ_convolve_freq (frequency-domain circular convolution)."""

    def test_known_result(self):
        """
        Test against manually computed result.
        circ_convolve_freq(X, H) = fft(ifft(X) * ifft(H))
        """
        N = 4
        X = jnp.array([4.0, 0.0, 0.0, 0.0])  # fft of [1, 1, 1, 1]
        H = jnp.array([4.0, 0.0, 0.0, 0.0])  # fft of [1, 1, 1, 1]
        # ifft(X) = [1, 1, 1, 1] (jax ifft normalizes, but fft([4,0,0,0]) = [1,1,1,1]*4)
        # ifft(H) = [1, 1, 1, 1]
        # ifft(X) * ifft(H) = [1, 1, 1, 1]
        # fft(above) = [4, 0, 0, 0]
        result = conv.circ_convolve_freq(X, H)
        expected = jnp.array([4.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j])
        assert result == pytest.approx(expected, rel=1e-12)

    def test_different_inputs(self):
        """Test with different frequency-domain inputs."""
        # x = [1, 0, 0, 0] in time domain, X = [1, 1, 1, 1] in freq
        # h = [0, 1, 0, 0] in time domain, H = [1, -i, -1, i] in freq
        x = jnp.array([1.0, 3.0 + 5j, -2.0, 4.0])
        h = jnp.array([0.0, -1.0, 1 + 3j, 5.0])
        X = jnp.fft.fft(x)
        H = jnp.fft.fft(h)
        result = conv.circ_convolve_freq(X, H)
        # In time: x * h (pointwise) = [0, 0, 0, 0]
        # So fft([0,0,0,0]) = [0,0,0,0]
        expected = jnp.fft.fft(x * h)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_commutativity(self):
        """Frequency-domain circular convolution is commutative."""
        X = jnp.fft.fft(jnp.array([1.0, 2.0, 3.0, 4.0]))
        H = jnp.fft.fft(jnp.array([0.5, -0.5, 0.25, 0.1]))
        result_XH = conv.circ_convolve_freq(X, H)
        result_HX = conv.circ_convolve_freq(H, X)
        assert result_XH == pytest.approx(result_HX, rel=1e-12)

    def test_complex_arrays(self):
        """Frequency-domain convolution works with complex arrays."""
        X = jnp.array([1.0 + 1j, 2.0 - 1j, 3.0 + 0j, 4.0 - 2j])
        H = jnp.array([0.5 + 0.5j, -0.5 + 0j, 0.25 - 0.25j, 0.0 + 0.1j])
        result = conv.circ_convolve_freq(X, H)
        result_reverse = conv.circ_convolve_freq(H, X)
        assert result == pytest.approx(result_reverse, rel=1e-12)

    def test_linearity(self):
        """Frequency-domain convolution is linear."""
        X1 = jnp.fft.fft(jnp.array([1.0, 2.0, 3.0, 4.0]))
        X2 = jnp.fft.fft(jnp.array([0.5, -0.5, 1.0, -1.0]))
        H = jnp.fft.fft(jnp.array([1.0, 0.5, 0.25, 0.125]))
        a, b = 2.0, -3.0
        result_combined = conv.circ_convolve_freq(a * X1 + b * X2, H)
        result_separate = a * conv.circ_convolve_freq(
            X1, H
        ) + b * conv.circ_convolve_freq(X2, H)
        assert result_combined == pytest.approx(result_separate, rel=1e-12)


class TestTimFreqDuality:
    """Tests verifying the relationship between time and frequency domain convolutions."""

    def test_time_to_freq_relationship(self):
        """
        Time-domain convolution in time corresponds to multiplication in frequency.
        y = circ_convolve_time(x, h) should satisfy: FFT(y) = FFT(x) * FFT(h)
        """
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1, -0.1, 0.2, -0.2, 0.3])
        y = conv.circ_convolve_time(x, h)
        Y = jnp.fft.fft(y)
        X = jnp.fft.fft(x)
        H = jnp.fft.fft(h)
        assert Y == pytest.approx(X * H, rel=1e-12)

    def test_freq_to_time_relationship(self):
        """
        Frequency-domain convolution corresponds to multiplication in time domain.
        Y = circ_convolve_freq(X, H) should satisfy: IFFT(Y) = IFFT(X) * IFFT(H)
        """
        X = jnp.array([1.0 + 0j, 2.0 - 1j, 3.0 + 2j, 4.0 - 0.5j])
        H = jnp.array([0.5 + 0.5j, -0.5 + 1j, 0.25 - 0.25j, 0.1 + 0.2j])
        Y = conv.circ_convolve_freq(X, H)
        y = jnp.fft.ifft(Y)
        x = jnp.fft.ifft(X)
        h = jnp.fft.ifft(H)
        assert y == pytest.approx(x * h, rel=1e-12)

    def test_roundtrip_consistency(self):
        """
        Starting from time domain, going to freq, convolving, and back should be consistent.
        """
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        # Time domain convolution
        y_time = conv.circ_convolve_time(x, h)
        # Equivalent via frequency domain multiplication
        X = jnp.fft.fft(x)
        H = jnp.fft.fft(h)
        Y_freq = X * H
        y_from_freq = jnp.fft.ifft(Y_freq)
        assert y_time == pytest.approx(y_from_freq, rel=1e-12)
