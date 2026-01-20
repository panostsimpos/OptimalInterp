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


class TestTripleCircConvolveTime:
    """Tests for conv.triple_circ_convolve_time (triple circular convolution in time domain)."""

    def test_impulse_identity(self):
        """Convolving with two impulses [1,0,0,...] returns the original signal."""
        N = 8
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        impulse = jnp.zeros(N).at[0].set(1.0)
        result = conv.triple_circ_convolve_time(x, impulse, impulse)
        assert result == pytest.approx(x, rel=1e-12)

    def test_reduces_to_double_convolution(self):
        """Triple convolution with impulse as third arg reduces to double convolution."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        impulse = jnp.zeros(4).at[0].set(1.0)
        result_triple = conv.triple_circ_convolve_time(x, h, impulse)
        result_double = conv.circ_convolve_time(x, h)
        assert result_triple == pytest.approx(result_double, rel=1e-12)

    def test_commutativity_all_permutations(self):
        """Triple circular convolution is commutative across all argument permutations."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        g = jnp.array([0.1, 0.2, -0.1, 0.3])
        # All 6 permutations should give same result
        result_xhg = conv.triple_circ_convolve_time(x, h, g)
        result_xgh = conv.triple_circ_convolve_time(x, g, h)
        result_hxg = conv.triple_circ_convolve_time(h, x, g)
        result_hgx = conv.triple_circ_convolve_time(h, g, x)
        result_gxh = conv.triple_circ_convolve_time(g, x, h)
        result_ghx = conv.triple_circ_convolve_time(g, h, x)
        assert result_xhg == pytest.approx(result_xgh, rel=1e-12)
        assert result_xhg == pytest.approx(result_hxg, rel=1e-12)
        assert result_xhg == pytest.approx(result_hgx, rel=1e-12)
        assert result_xhg == pytest.approx(result_gxh, rel=1e-12)
        assert result_xhg == pytest.approx(result_ghx, rel=1e-12)

    def test_associativity_with_double(self):
        """Triple conv equals sequential double convolutions: (x*h)*g = x*h*g."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        g = jnp.array([0.1, 0.2, -0.1, 0.3])
        result_triple = conv.triple_circ_convolve_time(x, h, g)
        result_sequential = conv.circ_convolve_time(conv.circ_convolve_time(x, h), g)
        assert result_triple == pytest.approx(result_sequential, rel=1e-12)

    def test_frequency_domain_equivalence(self):
        """FFT of triple time conv equals product of FFTs."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        g = jnp.array([0.1, 0.2, -0.1, 0.3])
        result = conv.triple_circ_convolve_time(x, h, g)
        result_fft = jnp.fft.fft(result)
        expected_fft = jnp.fft.fft(x) * jnp.fft.fft(h) * jnp.fft.fft(g)
        assert result_fft == pytest.approx(expected_fft, rel=1e-12)

    def test_complex_arrays(self):
        """Triple convolution works with complex arrays."""
        x = jnp.array([1.0 + 1j, 2.0 - 1j, 3.0 + 0j, 4.0 - 2j])
        h = jnp.array([0.5 + 0.5j, -0.5 + 0j, 0.25 - 0.25j, 0.0 + 0.1j])
        g = jnp.array([0.1 - 0.1j, 0.2 + 0.2j, -0.1 + 0j, 0.3 - 0.3j])
        result = conv.triple_circ_convolve_time(x, h, g)
        # Verify via FFT relationship
        expected_fft = jnp.fft.fft(x) * jnp.fft.fft(h) * jnp.fft.fft(g)
        result_fft = jnp.fft.fft(result)
        assert result_fft == pytest.approx(expected_fft, rel=1e-12)

    def test_linearity(self):
        """Triple convolution is linear in each argument."""
        x1 = jnp.array([1.0, 2.0, 3.0, 4.0])
        x2 = jnp.array([0.5, -0.5, 1.0, -1.0])
        h = jnp.array([0.5, -0.5, 0.25, 0.1])
        g = jnp.array([0.1, 0.2, -0.1, 0.3])
        a, b = 2.0, -3.0
        result_combined = conv.triple_circ_convolve_time(a * x1 + b * x2, h, g)
        result_separate = a * conv.triple_circ_convolve_time(
            x1, h, g
        ) + b * conv.triple_circ_convolve_time(x2, h, g)
        assert result_combined == pytest.approx(result_separate, rel=1e-12)


class TestTripleCircConvolveFreq:
    """Tests for conv.triple_circ_convolve_freq (triple circular convolution in frequency domain)."""

    def test_commutativity_all_permutations(self):
        """Triple frequency-domain convolution is commutative across all permutations."""
        X = jnp.fft.fft(jnp.array([1.0, 2.0, 3.0, 4.0]))
        H = jnp.fft.fft(jnp.array([0.5, -0.5, 0.25, 0.1]))
        G = jnp.fft.fft(jnp.array([0.1, 0.2, -0.1, 0.3]))
        result_XHG = conv.triple_circ_convolve_freq(X, H, G)
        result_XGH = conv.triple_circ_convolve_freq(X, G, H)
        result_HXG = conv.triple_circ_convolve_freq(H, X, G)
        result_HGX = conv.triple_circ_convolve_freq(H, G, X)
        result_GXH = conv.triple_circ_convolve_freq(G, X, H)
        result_GHX = conv.triple_circ_convolve_freq(G, H, X)
        assert result_XHG == pytest.approx(result_XGH, rel=1e-12)
        assert result_XHG == pytest.approx(result_HXG, rel=1e-12)
        assert result_XHG == pytest.approx(result_HGX, rel=1e-12)
        assert result_XHG == pytest.approx(result_GXH, rel=1e-12)
        assert result_XHG == pytest.approx(result_GHX, rel=1e-12)

    def test_time_domain_equivalence(self):
        """Frequency triple conv corresponds to pointwise time-domain triple product."""
        X = jnp.array([1.0 + 0j, 2.0 - 1j, 3.0 + 2j, 4.0 - 0.5j])
        H = jnp.array([0.5 + 0.5j, -0.5 + 1j, 0.25 - 0.25j, 0.1 + 0.2j])
        G = jnp.array([0.1 - 0.1j, 0.2 + 0.3j, -0.1 - 0.1j, 0.3 + 0j])
        result = conv.triple_circ_convolve_freq(X, H, G)
        # With norm='forward': ifft has no normalization, fft has 1/N
        x = jnp.fft.ifft(X, norm="forward")
        h = jnp.fft.ifft(H, norm="forward")
        g = jnp.fft.ifft(G, norm="forward")
        expected = jnp.fft.fft(x * h * g, norm="forward")
        assert result == pytest.approx(expected, rel=1e-12)

    def test_complex_arrays(self):
        """Frequency triple convolution works with complex arrays."""
        X = jnp.array([1.0 + 1j, 2.0 - 1j, 3.0 + 0j, 4.0 - 2j])
        H = jnp.array([0.5 + 0.5j, -0.5 + 0j, 0.25 - 0.25j, 0.0 + 0.1j])
        G = jnp.array([0.1 - 0.2j, 0.2 + 0.1j, -0.1 + 0.3j, 0.3 - 0.1j])
        # Verify commutativity as a sanity check
        result_XHG = conv.triple_circ_convolve_freq(X, H, G)
        result_GHX = conv.triple_circ_convolve_freq(G, H, X)
        assert result_XHG == pytest.approx(result_GHX, rel=1e-12)

    def test_linearity(self):
        """Frequency triple convolution is linear in each argument."""
        X1 = jnp.fft.fft(jnp.array([1.0, 2.0, 3.0, 4.0]))
        X2 = jnp.fft.fft(jnp.array([0.5, -0.5, 1.0, -1.0]))
        H = jnp.fft.fft(jnp.array([0.5, -0.5, 0.25, 0.1]))
        G = jnp.fft.fft(jnp.array([0.1, 0.2, -0.1, 0.3]))
        a, b = 2.0, -3.0
        result_combined = conv.triple_circ_convolve_freq(a * X1 + b * X2, H, G)
        result_separate = a * conv.triple_circ_convolve_freq(
            X1, H, G
        ) + b * conv.triple_circ_convolve_freq(X2, H, G)
        assert result_combined == pytest.approx(result_separate, rel=1e-12)


class TestConvolveTensors:
    """Tests for ode_res.convolve_tensors (vmapped triple convolution for tensor operations)."""

    def test_output_shape(self):
        """Output shape should be (N_alpha, N_beta, N_alpha)."""
        N_alpha, N_beta = 3, 5
        Tens1 = jnp.ones((N_alpha, N_beta), dtype=complex)
        Kernel = jnp.ones(N_beta, dtype=complex)
        Tens2 = jnp.ones((N_alpha, N_beta), dtype=complex)
        result = ode_res.convolve_tensors(Tens1, Kernel, Tens2)
        assert result.shape == (N_alpha, N_beta, N_alpha)

    def test_symmetry_in_tensor_swap(self):
        """Swapping Tens1 and Tens2 should transpose the alpha dimensions."""
        N_alpha, N_beta = 3, 4
        Tens1 = jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta)
        Tens2 = (
            jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta) + 1j
        )
        Kernel = jnp.ones(N_beta, dtype=complex) * 0.5
        result_12 = ode_res.convolve_tensors(Tens1, Kernel, Tens2)
        result_21 = ode_res.convolve_tensors(Tens2, Kernel, Tens1)
        # Swapping Tens1 and Tens2 should swap the first and third axes
        assert result_12 == pytest.approx(
            jnp.transpose(result_21, (2, 1, 0)), rel=1e-12
        )

    def test_linearity_in_first_tensor(self):
        """convolve_tensors is linear in Tens1."""
        N_alpha, N_beta = 2, 4
        Tens1_a = jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta)
        Tens1_b = (
            jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta) * 0.5j
        )
        Tens2 = jnp.ones((N_alpha, N_beta), dtype=complex) + 0.1j
        Kernel = jnp.array([1.0, -0.5, 0.25, 0.1], dtype=complex)
        a, b = 2.0 + 0.5j, -1.0 + 0.3j
        result_combined = ode_res.convolve_tensors(
            a * Tens1_a + b * Tens1_b, Kernel, Tens2
        )
        result_separate = a * ode_res.convolve_tensors(
            Tens1_a, Kernel, Tens2
        ) + b * ode_res.convolve_tensors(Tens1_b, Kernel, Tens2)
        assert result_combined == pytest.approx(result_separate, rel=1e-12)

    def test_linearity_in_kernel(self):
        """convolve_tensors is linear in Kernel."""
        N_alpha, N_beta = 2, 4
        Tens1 = jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta)
        Tens2 = jnp.ones((N_alpha, N_beta), dtype=complex) + 0.1j
        Kernel_a = jnp.array([1.0, -0.5, 0.25, 0.1], dtype=complex)
        Kernel_b = jnp.array([0.5, 0.5, -0.5, 0.2], dtype=complex)
        a, b = 2.0, -1.5
        result_combined = ode_res.convolve_tensors(
            Tens1, a * Kernel_a + b * Kernel_b, Tens2
        )
        result_separate = a * ode_res.convolve_tensors(
            Tens1, Kernel_a, Tens2
        ) + b * ode_res.convolve_tensors(Tens1, Kernel_b, Tens2)
        assert result_combined == pytest.approx(result_separate, rel=1e-12)

    def test_single_element_tensors(self):
        """Test with minimal 1x1 tensors to verify basic operation."""
        Tens1 = jnp.array([[2.0 + 1j]])
        Tens2 = jnp.array([[3.0 - 1j]])
        Kernel = jnp.array([0.5 + 0.5j])
        result = ode_res.convolve_tensors(Tens1, Kernel, Tens2)
        # For 1x1: result[0,0,0] = triple_circ_convolve_freq(Tens1[0,:], Kernel, Tens2[0,:])
        expected_val = conv.triple_circ_convolve_freq(Tens1[0, :], Kernel, Tens2[0, :])
        assert result[0, 0, 0] == pytest.approx(expected_val[0], rel=1e-12)

    def test_consistency_with_triple_convolve_freq(self):
        """Each slice should match direct call to triple_circ_convolve_freq."""
        N_alpha, N_beta = 2, 4
        Tens1 = jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta)
        Tens2 = (
            jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta) * 0.5
            + 0.1j
        )
        Kernel = jnp.array([1.0, -0.5, 0.25, 0.1], dtype=complex)
        result = ode_res.convolve_tensors(Tens1, Kernel, Tens2)
        # Check individual elements by direct computation
        for alpha in range(N_alpha):
            for gamma in range(N_alpha):
                expected = conv.triple_circ_convolve_freq(
                    Tens1[alpha, :], Kernel, Tens2[gamma, :]
                )
                assert result[alpha, :, gamma] == pytest.approx(expected, rel=1e-12)

    def test_zero_kernel(self):
        """Zero kernel should give zero output."""
        N_alpha, N_beta = 2, 4
        Tens1 = jnp.arange(N_alpha * N_beta, dtype=complex).reshape(N_alpha, N_beta)
        Tens2 = jnp.ones((N_alpha, N_beta), dtype=complex)
        Kernel = jnp.zeros(N_beta, dtype=complex)
        result = ode_res.convolve_tensors(Tens1, Kernel, Tens2)
        assert result == pytest.approx(
            jnp.zeros((N_alpha, N_beta, N_alpha), dtype=complex), abs=1e-14
        )
