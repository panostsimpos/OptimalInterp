# %% [markdown]
# ## Introduction
# In this notebook we show how to use the optimal interpolant contruction to build a straight-line flow the connects univariate Gaussians.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimalinterp as oi
from scipy.stats import gaussian_kde
from optimalinterp.stochastic_basis import GaussianBasis
from optimalinterp.optimal_interpolant import OptimalInterpolant

# %% [markdown]
# ## 1. Problem Setup
# We start by building the problem. The core object here is the "StpchasticBasis" which defined the family $(Z_\alpha)_{\alpha=1}^N$ of random variables that interpolates between the source and target measures.
# Namely, by construction we have
# $$ Z_0 \overset{d}{=} X_0 \quad Z_N \overset{d}{=} X_1 $$
# where $X_0 \sim \mu$ and $X_1 \sim \nu$ are the source and target measures respectively.

# In this example we work in $d=1$ dimensions and take the source to be a standard normal and the target to be a Gaussian with some mean and covariance.

# %%
# Generate samples to plot the source and target distributions
n_samples = 1000

key = jax.random.PRNGKey(0)
stochastic_basis = GaussianBasis(
    mean=10.0,
    std_dev=2.0,
    N_basis=5,
    bridge_type="gaussian_convolution",
)
samples = stochastic_basis.sample(N_samples=1000, key=key)
source_samples = samples[:, 0]
target_samples = samples[:, -1]

# Plot samples with KDE on the same plot

fig, ax = plt.subplots(figsize=(10, 6))

# Source distribution
ax.hist(
    source_samples,
    bins=50,
    density=True,
    alpha=0.5,
    color="blue",
    edgecolor="black",
    label="Source Samples",
)
kde_source = gaussian_kde(source_samples)
x_range = jnp.linspace(-10, 15, 200)
ax.plot(x_range, kde_source(x_range), color="blue", linewidth=2, label="Source KDE")

# Target distribution
ax.hist(
    target_samples,
    bins=50,
    density=True,
    alpha=0.5,
    color="red",
    edgecolor="black",
    label="Target Samples",
)
kde_target = gaussian_kde(target_samples)
ax.plot(x_range, kde_target(x_range), color="red", linewidth=2, label="Target KDE")

ax.set_title("Source and Target Distribution Samples")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_xlim(-10, 15)
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Now we proceed by building an optimal interpolant object.
# Note that upon instantiation, one *will not* be able to call the interpolant, as the
# coefficients $t \mapsto \psi(t)$ have not yet been computed.

# However, before we build an optimal interpolant we build a *standard stochastic interpolant*
# to give ourselves a point of comparison.
# %%

standard_interpolant = OptimalInterpolant(
    t=jnp.linspace(0, 1, 100),  # 100 time points
    Z=stochastic_basis,
).with_psi(
    psi=jnp.linspace(0, 1, 100)[:, None] * jnp.ones((1, stochastic_basis.N_basis))
)

# Plot sample paths of the standard interpolant
n_paths = 20
t_eval = jnp.linspace(0, 1, 100)
key, subkey = jax.random.split(key)
sample_paths = standard_interpolant(
    N_samples=n_paths, key=subkey, t_eval=t_eval
)  # Shape: (n_paths, n_times)

fig, ax = plt.subplots(figsize=(10, 6))
t_vals = standard_interpolant.t

for i in range(n_paths):
    ax.plot(t_vals, sample_paths[i, :], alpha=0.5, linewidth=1)

ax.set_title("Standard Interpolant Sample Paths")
ax.set_xlabel("Time t")
ax.set_ylabel("x")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

optimal_interpolant = OptimalInterpolant(
    t=jnp.linspace(0, 1, 100),  # 100 time points
    Z=stochastic_basis,
)
