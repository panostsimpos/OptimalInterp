import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
from jaxtyping import Float, Array
import jax
import optimalinterp as oi
import diffrax

# %%
# =============================================================================
# Generic binning-based velocity field computation
# =============================================================================


def truncate_on_torus(
    x: Float[Array, "n_samples"], period: Float
) -> Float[Array, "n_samples"]:
    """
    Truncate samples x onto a torus with given period.

    :param x: Input samples.
    :param period: Period of the torus.
    :return: Truncated samples in [-period/2, period/2).
    """
    return jnp.mod(x + period / 2, period) - period / 2


def compute_velocity_at_t_binned(
    psi: Float[Array, "N"],
    psi_dot: Float[Array, "N"],
    Z_samples: Float[Array, "n_samples N"],
    x_grid: Float[Array, "n_grid"],
    bin_width: Float,
    kernel_type: str = "gaussian",
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
        if kernel_type == "gaussian":

            def kernel(x):
                return jnp.exp(-0.5 * (x) ** 2)

        elif kernel_type == "square":

            def kernel(x):
                return jnp.where(jnp.abs(x) <= 0.5, 1.0, 0.0)

        else:
            raise ValueError("Unknown kernel type")
        distances = (X_samples - x) / bin_width
        weights = kernel(distances)
        weighted_sum = jnp.sum(weights * Xdot_samples)
        weight_total = jnp.sum(weights)
        # Avoid division by zero
        return jnp.where(weight_total > 1e-10, weighted_sum / weight_total, 0.0)

    return jax.vmap(velocity_at_x)(x_grid)


def extract_psi_and_psi_dot(
    y_t: Float[Array, "4*N"], N_terms: int
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """
    Extract psi and psi_dot from concatenated ODE solution at time t.

    The ODE solution is stored as: [Re(psi), Re(psi_dot), Im(psi), Im(psi_dot)].

    :param y_t: Concatenated ODE solution at time t.
    :param N_terms: Number of psi_alpha coefficients.
    :return: (psi, psi_dot) as real arrays (taking real part).
    """
    psi = y_t[:N_terms]
    psi_dot = y_t[N_terms : 2 * N_terms]
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
    velocity_list = []
    for y_t in ode_sol:
        psi, psi_dot = extract_psi_and_psi_dot(y_t, N_terms)
        v_t = compute_velocity_at_t_binned(psi, psi_dot, Z_samples, x_grid, bin_width)
        velocity_list.append(v_t)

    return jnp.stack(velocity_list)


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

# Truncate on the torus
Z_samples = truncate_on_torus(Z_samples, 2 * jnp.pi)

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
    extent=(time_grid[0], time_grid[-1], x_grid[0], x_grid[-1]),
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
time_indices = [
    0,
    len(time_grid) // 4,
    len(time_grid) // 2,
    3 * len(time_grid) // 4,
    -1,
]
for idx in time_indices:
    ax.plot(x_grid, velocity_field[idx], label=f"$t = {time_grid[idx]:.2f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$v(x,t)$")
ax.set_title("Velocity field at different times")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Simulate particle trajectories under velocity field
# =============================================================================


def interpolate_velocity_field(
    t: float,
    x: float,
    time_grid: Float[Array, "n_times"],
    x_grid: Float[Array, "n_grid"],
    velocity_field: Float[Array, "n_times n_grid"],
) -> float:
    """
    Interpolate velocity field v(t, x) at arbitrary (t, x).

    Uses bilinear interpolation in time and space.

    :param t: Time at which to evaluate velocity.
    :param x: Position at which to evaluate velocity.
    :param time_grid: Time grid points.
    :param x_grid: Spatial grid points.
    :param velocity_field: Precomputed velocity field values.
    :return: Interpolated velocity v(t, x).
    """
    # Find time indices for interpolation
    t_idx = jnp.searchsorted(time_grid, t)
    t_idx = jnp.clip(t_idx, 1, len(time_grid) - 1)
    t_idx_low = t_idx - 1
    t_idx_high = t_idx

    # Time interpolation weight
    t_low = time_grid[t_idx_low]
    t_high = time_grid[t_idx_high]
    t_weight = (t - t_low) / (t_high - t_low + 1e-10)

    # Find spatial indices for interpolation
    x_idx = jnp.searchsorted(x_grid, x)
    x_idx = jnp.clip(x_idx, 1, len(x_grid) - 1)
    x_idx_low = x_idx - 1
    x_idx_high = x_idx

    # Spatial interpolation weight
    x_low = x_grid[x_idx_low]
    x_high = x_grid[x_idx_high]
    x_weight = (x - x_low) / (x_high - x_low + 1e-10)

    # Bilinear interpolation
    v_ll = velocity_field[t_idx_low, x_idx_low]
    v_lh = velocity_field[t_idx_low, x_idx_high]
    v_hl = velocity_field[t_idx_high, x_idx_low]
    v_hh = velocity_field[t_idx_high, x_idx_high]

    v_low = (1 - x_weight) * v_ll + x_weight * v_lh
    v_high = (1 - x_weight) * v_hl + x_weight * v_hh

    v = (1 - t_weight) * v_low + t_weight * v_high

    return v


def particle_ode_rhs(t, X, args):
    """
    RHS for particle trajectory ODE: dX/dt = v(t, X).

    :param t: Current time.
    :param X: Current position (scalar for 1D).
    :param args: Tuple (time_grid, x_grid, velocity_field).
    :return: Time derivative dX/dt.
    """
    time_grid, x_grid, velocity_field = args
    return interpolate_velocity_field(t, X, time_grid, x_grid, velocity_field)


def simulate_particle_trajectory(
    X0: float,
    time_grid: Float[Array, "n_times"],
    x_grid: Float[Array, "n_grid"],
    velocity_field: Float[Array, "n_times n_grid"],
) -> Float[Array, "n_times"]:
    """
    Simulate a single particle trajectory under velocity field.

    :param X0: Initial position of particle.
    :param time_grid: Time points at which to save trajectory.
    :param x_grid: Spatial grid for velocity field.
    :param velocity_field: Precomputed velocity field.
    :return: Particle trajectory X(t) at times time_grid.
    """
    term = diffrax.ODETerm(particle_ode_rhs)
    solver = diffrax.Tsit5()  # Explicit RK method for non-stiff ODEs
    saveat = diffrax.SaveAt(ts=time_grid)
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    args = (time_grid, x_grid, velocity_field)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=time_grid[0],
        t1=time_grid[-1],
        dt0=(time_grid[-1] - time_grid[0]) / 100,
        y0=X0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
    )

    return sol.ys


# %%
# Sample initial positions from standard normal and simulate trajectories
N_flow_samples = 50
key_flow = jax.random.PRNGKey(123)
X0_samples = jax.random.normal(key_flow, shape=(N_flow_samples,))
X0_samples = truncate_on_torus(X0_samples, 2 * jnp.pi)

# Simulate all trajectories
trajectories = []
for i in range(N_flow_samples):
    traj = simulate_particle_trajectory(
        X0_samples[i], time_grid, x_grid, velocity_field
    )
    trajectories.append(traj)

trajectories = jnp.stack(trajectories)  # Shape: (N_flow_samples, n_times)

# %%
# Plot particle trajectories
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each trajectory
for i in range(N_flow_samples):
    ax.plot(time_grid, trajectories[i], alpha=0.5, lw=1)

ax.set_xlabel("$t$")
ax.set_ylabel("$X_t$")
ax.set_title(f"Particle trajectories under flow (N={N_flow_samples})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Alternative visualization: trajectories in (t, x) space overlaid on velocity field
fig, ax = plt.subplots(figsize=(10, 6))

# Background: velocity field
im = ax.imshow(
    velocity_field.T,
    aspect="auto",
    origin="lower",
    extent=(time_grid[0], time_grid[-1], x_grid[0], x_grid[-1]),
    cmap="RdBu_r",
    alpha=0.6,
)

# Overlay particle trajectories
for i in range(N_flow_samples):
    ax.plot(time_grid, trajectories[i], "k-", alpha=0.4, lw=0.8)

ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_title(f"Particle trajectories on velocity field (N={N_flow_samples})")
plt.colorbar(im, ax=ax, label="$v(x,t)$")
plt.tight_layout()
plt.show()

# %%
