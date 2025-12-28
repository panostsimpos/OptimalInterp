# %%
# %env JAX_DISABLE_JIT 1

# %%
from jaxtyping import Float, Array
import equinox as eqx
import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
import diffrax
import optimalinterp as oi
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

# %%


def rhs(t, y, args):
    N_psi = len(y) // 2
    psi_real, psi_imag = y[:N_psi], y[N_psi:]
    psi = psi_real + (1j * psi_imag)
    _, phi, D_infl = args
    d_psi = oi.ode_residual.OptimalInterpBVP_ODE_RHS(psi, phi, D_infl)
    dy = jnp.concat((jnp.real(d_psi), jnp.imag(d_psi)))
    return dy


def lhs(dy, t, y, args, control):
    _, phi = args
    return oi.ode_residual.OptimalInterpBVP_DAE_LHS(dy, y, phi)


# %%
N_terms = 3
psi_0 = jnp.zeros(N_terms, dtype=jnp.complex128)
psi_0 = psi_0.at[0].set(1.0 + 0.0j)
psi_dot_0 = jnp.zeros(N_terms, dtype=jnp.complex128)
psi_dot_0 = psi_dot_0.at[jnp.array([0, -1])].set(jnp.array([-3, 2]))
psi_concat_0 = jnp.concat((psi_0, psi_dot_0))
y0 = jnp.concat((jnp.real(psi_concat_0), jnp.imag(psi_concat_0)))
phi = oi.GaussianPhi1D(mu=2.0, sigma=4.0)
D_infl = 1e-4
args = (lhs, phi, D_infl)

# %%
N_t = 4096
term = diffrax.ODETerm(rhs)
solver = diffrax.Kvaerno5()
saveat_t = jnp.linspace(0, 1, 1001)
saveat = diffrax.SaveAt(ts=saveat_t)
stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-8)

# %%
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=saveat_t[0],
    t1=saveat_t[-1],
    dt0=1e-3,
    y0=y0,
    args=args,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=N_t,
    throw=False,
    progress_meter=diffrax.TqdmProgressMeter(),
)

# %%
assert sol.ys is not None
psi_concat_t = sol.ys[:, : 2 * N_terms] + 1j * sol.ys[:, 2 * N_terms :]
psi_t, psi_dot_t = psi_concat_t[:, :N_terms], psi_concat_t[:, N_terms:]
plt.plot(
    saveat_t,
    sol.ys[:, :N_terms],
    lw=3,
    label=["$\\psi_{}$".format(j) for j in range(N_terms)],
)
# plt.plot(saveat_t, psi_dot_t)
plt.xscale("log")
plt.title("Arbitrary solution for {} $\\psi_\\alpha$ terms".format(N_terms))
plt.xlabel("$t$")
plt.legend()
plt.show()

# %%


def solve(psi_dot_0, *args, **solver_kwargs):
    psi_0, solver, term, solver_args = args
    N_terms = len(psi_0)
    y0 = jnp.concat(
        (psi_0, psi_dot_0, jnp.zeros(N_terms))
    )  # TODO: Do we not need 2N zeros in the velocity state?
    # y0 = jnp.concat((psi_0, psi_dot_0, jnp.zeros(N_terms), jnp.zeros(N_terms)))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        y0=y0,
        throw=False,
        args=solver_args,
        **solver_kwargs,
    )
    return jax.lax.cond(
        sol.result._value == 0, lambda: sol.ys, lambda: jnp.inf * sol.ys
    )


# %%
def residual(psi_dot_0, psi_1, *args, **solver_kwargs):
    N_terms = len(psi_1)
    pred_y1_concat = solve(psi_dot_0, *args, **solver_kwargs)[-1]
    real_res = psi_1 - pred_y1_concat[:N_terms]
    im_res = pred_y1_concat[2 * N_terms : 3 * N_terms]
    return jnp.concat((real_res, im_res))


# %%
N_terms = 5
z = jnp.zeros(N_terms)
psi_0 = z.at[0].set(1.0)
psi_1 = z.at[-1].set(1.0)
# .at[jnp.array([0,-1])].set(jnp.array([-1,1]))
initial_psi_dot_0 = jnp.concat((z, z))
solver_kwargs = {
    "adjoint": diffrax.DirectAdjoint(),
    "max_steps": 5000,
    "stepsize_controller": diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-8),
    "dt0": 1e-3,
}
solve_args = (psi_0, diffrax.Kvaerno5(), term, args)

# %%


@jax.jit
def residual_fcn(psi_dot_0, _):
    return residual(psi_dot_0, psi_1, *solve_args, **solver_kwargs)


# %%
residual(initial_psi_dot_0, psi_1, *solve_args, **solver_kwargs)

# %%
solver = optx.BestSoFarLeastSquares(
    optx.LevenbergMarquardt(
        rtol=1e-8,
        atol=1e-8,
        verbose=frozenset({"step", "accepted", "loss", "step_size"}),
    )
)

# %%
sol = optx.least_squares(
    residual_fcn,
    solver,
    initial_psi_dot_0,
    # max_steps=17,
    throw=False,
)

# %%
sol.result, solver.norm(residual_fcn(sol.value, None)).item()

# %%
ode_sol = solve(sol.value, *solve_args, saveat=saveat, **solver_kwargs)
y1_concat = ode_sol[-1]
psi_1_concat = y1_concat[: 2 * N_terms] + 1j * y1_concat[2 * N_terms :]
psi_1 = psi_1_concat[:N_terms]
psi_1

# %%
plt.plot(
    saveat_t,
    ode_sol[:, :N_terms],
    lw=3,
    label=["$\\psi_{}$".format(j) for j in range(N_terms)],
)
# plt.plot(saveat_t, psi_dot_t)
# plt.xscale('log')
plt.title("Approx optimal soln for {} $\\psi_\\alpha$ terms".format(N_terms))
plt.xlabel("$t$")
plt.legend()
plt.show()

# %%
# =============================================================================
# Generic binning-based velocity field computation
# =============================================================================


@jax.jit
def compute_velocity_at_t_binned(
    psi: Float[Array, "N"],
    psi_dot: Float[Array, "N"],
    Z_samples: Float[Array, "n_samples N"],
    x_grid: Float[Array, "n_grid"],
    bin_width: Float,
) -> Float[Array, "n_grid"]:
    """
    Compute v(x,t) = E[dot{X}_t | X_t = x] using binning at a single time t.

    Given samples Z^{(i)}_alpha, we compute:
        X_t^{(i)} = sum_alpha psi_alpha * Z^{(i)}_alpha
        dot{X}_t^{(i)} = sum_alpha dot{psi}_alpha * Z^{(i)}_alpha
    Then estimate v(x,t) by averaging dot{X}_t over samples where X_t falls in a bin around x.

    :param psi: Coefficients psi_alpha(t) at time t.
    :param psi_dot: Time derivatives dot{psi}_alpha(t) at time t.
    :param Z_samples: Samples of Z_alpha, shape (n_samples, N_terms).
    :param x_grid: Grid of x values where to evaluate v(x,t).
    :param bin_width: Width of bins for conditioning.
    :return: Velocity field v(x,t) evaluated on x_grid.
    """
    # Compute X_t and dot{X}_t for all samples
    X_samples = Z_samples @ psi  # (n_samples,)
    Xdot_samples = Z_samples @ psi_dot  # (n_samples,)

    # For each x in x_grid, find samples in bin and average Xdot
    def velocity_at_x(x):
        # Soft binning using a smooth kernel (Gaussian-like)
        # weights = exp(-0.5 * ((X_samples - x) / bin_width)^2)
        distances = (X_samples - x) / bin_width
        weights = jnp.exp(-0.5 * distances**2)
        weighted_sum = jnp.sum(weights * Xdot_samples)
        weight_total = jnp.sum(weights)
        # Avoid division by zero
        return jnp.where(weight_total > 1e-10, weighted_sum / weight_total, 0.0)

    return jax.vmap(velocity_at_x)(x_grid)


@jax.jit
def extract_psi_and_psi_dot(
    y_t: Float[Array, "4*N"], N_terms: int
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """
    Extract psi and psi_dot from concatenated ODE solution at time t.

    The ODE solution is stored as: [Re(psi), Re(psi_dot), Im(psi), Im(psi_dot)]
    or similar concatenation. Adjust based on actual storage format.

    :param y_t: Concatenated ODE solution at time t.
    :param N_terms: Number of psi_alpha coefficients.
    :return: (psi, psi_dot) as real arrays (taking real part).
    """
    # Based on example: y = [Re(psi_concat), Im(psi_concat)] where psi_concat = [psi, psi_dot]
    psi_concat = y_t[: 2 * N_terms] + 1j * y_t[2 * N_terms : 4 * N_terms]
    psi = jnp.real(psi_concat[:N_terms])
    psi_dot = jnp.real(psi_concat[N_terms : 2 * N_terms])
    return psi, psi_dot


def compute_velocity_field_binned(
    ode_sol: Float[Array, "n_times 4*N"],
    Z_samples: Float[Array, "n_samples N"],
    x_grid: Float[Array, "n_grid"],
    N_terms: int,
    bin_width: Float = 0.5,
) -> Float[Array, "n_times n_grid"]:
    """
    Compute velocity field v(x,t) over all times and x positions using binning.

    :param ode_sol: ODE solution array, shape (n_times, 4*N_terms).
    :param Z_samples: Samples of Z_alpha, shape (n_samples, N_terms).
    :param x_grid: Grid of x values, shape (n_grid,).
    :param N_terms: Number of psi_alpha coefficients.
    :param bin_width: Width for soft binning kernel.
    :return: Velocity field, shape (n_times, n_grid).
    """

    def velocity_at_time(y_t):
        psi, psi_dot = extract_psi_and_psi_dot(y_t, N_terms)
        return compute_velocity_at_t_binned(psi, psi_dot, Z_samples, x_grid, bin_width)

    return jax.vmap(velocity_at_time)(ode_sol)


# %%
# =============================================================================
# Example: Instantiate Z_alpha samples for Gaussian case
# =============================================================================
# For the Gaussian interpolation problem:
# Z_0 ~ N(0, 1), Z_N ~ N(mu, sigma^2)
# Z_alpha = (1 - alpha/N) * Z_0 + (alpha/N) * Z_N  (in distribution, independent)
# Each Z_alpha ~ N(mu_alpha, sigma_alpha^2) with:
#   mu_alpha = (alpha/N) * mu
#   sigma_alpha^2 = (1 - alpha/N)^2 + (alpha/N)^2 * sigma^2

key = jax.random.PRNGKey(42)
n_samples = 10000

# Compute per-alpha means and standard deviations
alpha_ratios = jnp.arange(N_terms) / (N_terms - 1)
mu_Z = alpha_ratios * phi.mu
sigma_Z = jnp.sqrt((1 - alpha_ratios) ** 2 + alpha_ratios**2 * phi.sigma**2)

# Sample Z_alpha independently for each alpha
Z_samples = jax.random.normal(key, shape=(n_samples, N_terms)) * sigma_Z + mu_Z

# %%
# Define x grid and bin width
x_grid = jnp.linspace(-5, 5, 100)
bin_width = 0.3

# %%
# Compute velocity field using binning
velocity_field = compute_velocity_field_binned(
    ode_sol, Z_samples, x_grid, N_terms, bin_width
)
velocity_field.shape

# %%
# Visualize velocity field
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(
    velocity_field.T,
    aspect="auto",
    origin="lower",
    extent=(saveat_t[0], saveat_t[-1], x_grid[0], x_grid[-1]),
    cmap="RdBu_r",
)
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_title("Velocity field $v(x,t) = E[\\dot{X}_t | X_t = x]$ (binning)")
plt.colorbar(im, ax=ax, label="$v(x,t)$")
plt.tight_layout()
plt.show()

# %%
# Plot velocity at a few time slices
fig, ax = plt.subplots(figsize=(6, 4))
time_indices = [0, len(saveat_t) // 4, len(saveat_t) // 2, 3 * len(saveat_t) // 4, -1]
for idx in time_indices:
    ax.plot(x_grid, velocity_field[idx], label=f"$t = {saveat_t[idx]:.2f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$v(x,t)$")
ax.set_title("Velocity field at different times")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
