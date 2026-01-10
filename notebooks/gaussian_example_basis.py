# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import optimalinterp as oi
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

# %%
max_order, N_time, N_terms, num_steps = 24, 1025, 3, 1000
stochastic_basis = oi.GaussianConvolutionBasis(
    N_terms,
    mean=2.0,
    std_dev=4.0,
)
t_points, t_weights = oi.util.clenshaw_curtis(N_time)
psi = oi.basis.LinearBasis(max_order, 'chebyshev')
optimal_interpolant = oi.optimal_interpolant.compute_optimal_psi_basis(
    stochastic_basis, allow_failure=True, max_optimizer_steps=num_steps,
    psi_basis = psi, t_points=t_points, t_weights=t_weights,
    atol=1e-12, rtol=1e-12
)

# %%
n_paths = 1000
t_eval = jnp.linspace(0, 1, 1001)
key = jax.random.PRNGKey(0)
sample_paths = optimal_interpolant(
    N_samples=n_paths, key=key, t_eval=t_eval
)  # Shape: (n_paths, n_times)

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n_paths):
    ax.plot(t_eval, sample_paths[i, :], alpha=0.5, linewidth=1)

ax.set_title("Optimal Interpolant Sample Paths")
ax.set_xlabel("Time t")
ax.set_ylabel("x")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Finally, we can compute and visualize the mean velocity field induced by the optimal interpolant.
# We use the `interpolate_velocity_field` utility function for this purpose.

# %%
key = jax.random.PRNGKey(42)
x_grid = jnp.linspace(-10, 10, 1000)
oi.visualization.visualize_interpolant_flow(
    optimal_interpolant,
    key,
    x_grid,
    bin_width=0.2,
    kernel_type="gaussian",
    N_velocity_samples=1000,
    N_flow_samples=50,
)

# %%
