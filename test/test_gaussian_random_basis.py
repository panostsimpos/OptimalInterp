# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optimalinterp.stochastic_basis import GaussianBasis

# %%

if __name__ == "__main__":

    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)

    # Set up parameters
    N_terms = 10
    target_mean = 20.0  # Large mu to see shift
    target_std = 2.0  # Large sigma to see dilation
    n_samples = 1000

    # Initialize random basis
    random_basis = GaussianBasis(N_terms, target_mean, target_std)

    # Generate samples
    key = jax.random.PRNGKey(42)
    samples = random_basis.sample(
        key=key, n_samples=n_samples
    )  # Shape: (n_samples, N_terms)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot histogram for each Z_alpha with different color
    colors = plt.cm.viridis(jnp.linspace(0, 1, N_terms))
    alpha_values = jnp.linspace(0, 1, N_terms)

    for alpha_idx in range(N_terms):
        # Plot histogram
        ax.hist(
            samples[:, alpha_idx],
            bins=100,  # Increased from 50 for smoother appearance
            alpha=0.4,  # Reduced alpha to make KDE curves more visible
            color=colors[alpha_idx],
            label=f"Z_{alpha_idx} (α/N={alpha_values[alpha_idx]:.2f})",
            density=True,
        )

        # Add KDE curve
        kde = gaussian_kde(samples[:, alpha_idx])
        x_range = jnp.linspace(
            samples[:, alpha_idx].min(), samples[:, alpha_idx].max(), 200
        )
        kde_values = kde(x_range)
        ax.plot(x_range, kde_values, color=colors[alpha_idx], linewidth=2, alpha=0.8)

    # Theoretical parameters for verification
    ax.axvline(0, color="blue", linestyle="--", linewidth=2, label=f"Z_0 mean (0)")
    ax.axvline(
        target_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Z_N mean ({target_mean})",
    )

    ax.set_xlabel("Sample Value", fontsize=12)
    ax.set_xlim(-5, 30)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Gaussian Random Basis Samples\n(μ={target_mean}, σ={target_std}, N={N_terms})",
        fontsize=14,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("gaussian_random_basis_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print statistics for verification
    print(f"Expected Z_0: mean=0, std=1")
    print(
        f"Actual Z_0:   mean={jnp.mean(samples[:, 0]):.4f}, std={jnp.std(samples[:, 0]):.4f}\n"
    )

    print(f"Expected Z_N: mean={target_mean}, std={target_std}")
    print(
        f"Actual Z_N:   mean={jnp.mean(samples[:, -1]):.4f}, std={jnp.std(samples[:, -1]):.4f}\n"
    )

    # Check intermediate term (e.g., Z_5 if N=10)
    mid_idx = N_terms // 2
    alpha_mid = alpha_values[mid_idx]
    expected_mean_mid = alpha_mid * target_mean
    expected_std_mid = jnp.sqrt((1 - alpha_mid) ** 2 + alpha_mid**2 * target_std**2)
    print(
        f"Expected Z_{mid_idx}: mean={expected_mean_mid:.4f}, std={expected_std_mid:.4f}"
    )
    print(
        f"Actual Z_{mid_idx}:   mean={jnp.mean(samples[:, mid_idx]):.4f}, std={jnp.std(samples[:, mid_idx]):.4f}"
    )

# %%
