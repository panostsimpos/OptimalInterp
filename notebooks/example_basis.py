# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: optimalinterp
#     language: python
#     name: python3
# ---

# %%
from jaxtyping import Float, Array
import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
import time
import os
import optimalinterp as oi
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

# %%
max_order, N_grid, N_terms = 24, 1025, 5
mu, sigma = 2., 4.
psi = oi.basis.LinearBasis(max_order, 'chebyshev')
phi = oi.moment_generator.GaussianPhi1D(N_terms, mu, sigma)
pts, wts = oi.util.clenshaw_curtis(N_grid)
t_grid = jnp.sort(pts)
t_wts = jnp.sqrt(wts)
t_grid = (t_grid - t_grid[0])/(t_grid[-1] - t_grid[0])
fixed_orders = (0,1) # Fix the constant and linear function coefficients

# %%
residual = oi.basis.create_collocated_basis_residual(t_grid, N_terms, psi, phi, fixed_orders, t_wts, jnp.real)
solver = optx.LevenbergMarquardt(
    rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})
)
y0 = jnp.zeros((psi.N_shap - 2, N_terms))

# %%
residual(y0, None)

# %%
max_steps = 200
sol = optx.least_squares(residual, solver, y0, throw=False, max_steps=max_steps)

# %%
if not os.path.isdir('serial'):
    os.mkdir('serial')
jnp.save(f"serial/coeffs_{time.time()}.npy", sol.value)

# %%
coeffs_psi = oi.basis.pad_coeffs(sol.value, psi, fixed_orders)
eval_t_grid = jnp.arange(0,1001)/1000
evals_basis_test, diff1_basis_test = psi.evaluate_basis_diff(eval_t_grid)
evals_sol = evals_basis_test @ coeffs_psi
diff1_sol = diff1_basis_test @ coeffs_psi
labs = [str(j) for j in range(N_terms)]
plt.plot(eval_t_grid, evals_sol, label=labs, linewidth=3)
plt.legend()
plt.show()

# %%
def gaussian_velocity(x: Float, psi: Float[Array, " N"], psi_dot: Float[Array, " N"], mu_Z: Float[Array, " N"], Sigma_Z: Float[Array, "N N"]):
    N = psi.shape[0]
    assert psi_dot.shape[0] == N and mu_Z.shape[0] == N and Sigma_Z.shape == (N, N)
    sig_psi = Sigma_Z @ psi
    dot_prod = jnp.dot(psi, sig_psi)
    bias = (x - jnp.dot(mu_Z, psi))
    shift = (sig_psi * bias) / dot_prod
    return psi_dot @ (mu_Z + shift)

# %%
mu_Z = jnp.linspace(0, phi.mu, N_terms)
Sigma_Z = jnp.diag((1 - mu_Z)**2 + (mu_Z**2)*phi.sigma**2)
velocity_vmap = jax.vmap(
    jax.vmap(
        lambda x, psi, psi_dot: gaussian_velocity(x, psi, psi_dot, mu_Z, Sigma_Z),
        in_axes=(0, None, None),
    ), in_axes=(None, 0, 0)
)

# %%
x_min, x_max, N_X = -10, 10, 2001
viz_x_grid = jnp.linspace(x_min, x_max, N_X)
velocity_eval = velocity_vmap(viz_x_grid, evals_sol, diff1_sol)
hm = plt.imshow(velocity_eval[100:].T, extent=(eval_t_grid[100].item(), 1, x_min, x_max), aspect=1/(x_max - x_min))
plt.plot(eval_t_grid, jnp.linspace(0, phi.mu, len(eval_t_grid)), color='k', linewidth=3, label=r"$\mu$")
plt.xlabel('t')
plt.ylabel('x')
plt.colorbar(hm, label='v(t,x)')
plt.legend()
plt.show()

# %%
