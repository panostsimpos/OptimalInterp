# %% [markdown]
# ## Introduction
# In this notebook we show how to use the optimal interpolant contruction to build a straight-line flow the connects univariate Gaussians.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimalinterp as oi

# %% [markdown]
# ## 1. Problem Setup
# We start by building the problem. Let's fist define the source and targer distributions.

source = oi.GaussianPhi1D(mu=0.0, sigma=1.0)
target = oi.GaussianPhi1D(mu=4.0, sigma=3.0)

# Generate samples
n_samples = 1000
source_samples = source.sample(jax.random.PRNGKey(0), n_samples)
target_samples = target.sample(jax.random.PRNGKey(1), n_samples)

# Plot samples
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(source_samples, bins=50, density=True, alpha=0.7, edgecolor="black")
axes[0].set_title("Source Distribution Samples")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Density")
axes[0].set_xlim(-10, 10)

axes[1].hist(target_samples, bins=50, density=True, alpha=0.7, edgecolor="black")
axes[1].set_title("Target Distribution Samples")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Density")
axes[1].set_xlim(-10, 10)


plt.tight_layout()
plt.show()

# %%
