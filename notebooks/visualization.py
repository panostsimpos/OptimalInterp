import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
from jaxtyping import Float, Array
import jax
import diffrax
from optimalinterp.optimal_interpolant import OptimalInterpolant

# %%
# =============================================================================
# Generic binning-based velocity field computation
# =============================================================================


# def truncate_on_torus(
#     x: Float[Array, "N_samples"], period: Float
# ) -> Float[Array, "N_samples"]:
#     """
#     Truncate samples x onto a torus with given period.

#     :param x: Input samples.
#     :param period: Period of the torus.
#     :return: Truncated samples in [-period/2, period/2).
#     """
#     return jnp.mod(x + period / 2, period) - period / 2


def compute_velocity(
    interpolant: OptimalInterpolant,
    key: jax.Array,
    x_span: tuple[float, float],
    bin_width: Float,
    kernel_type: str = "gaussian",
    N_samples: int = 1000,
) -> Float[Array, "x_grid t_grid"]:
    """
    Compute v(x,t) = E[dot{X}_t | X_t = x] using binning at a single time t.
    Given an interopolant X_t = Σ_{α=1}^N Z_α ψ_α(t) with time derivative

    Then estimate v(x,t) by averaging dot{X}_t over samples where X_t falls in a bin around x.

    :param interpolant: An instance of OptimalInterpolant.
    :param key: JAX PRNG key.
    :param x_span: Tuple (x_min, x_max) defining grid range.
    :param bin_width: Width of bins for conditioning.
    :param N_samples: Number of samples of Z_α to use.

    Returns:
    :return: Velocity values at grid points, shape (x_grid, t_grid).
    """
    # Compute X_t and dot{X}_t for all samples
    Z_samples = interpolant.Z.sample(
        N_samples=N_samples, key=key
    )  # (N_samples, N_basis)
    psi, psi_dot = interpolant.psi, interpolant.psi_dot  # (T, N_basis)
    X_samples = Z_samples @ psi.T  # (N_samples, T)
    Xdot_samples = Z_samples @ psi_dot.T  # (N_samples, T)

    # For each x in x_grid, find samples in bin and average Xdot
    def velocity_at_x_and_t(x, X_samples_X_dot_conact_at_t):
        X_samples_at_t, Xdot_samples_at_t = X_samples_X_dot_conact_at_t
        if kernel_type == "gaussian":

            def kernel(x):
                return jnp.exp(-0.5 * (x) ** 2)

        elif kernel_type == "square":

            def kernel(x):
                return jnp.where(jnp.abs(x) <= 0.5, 1.0, 0.0)

        else:
            raise ValueError("Unknown kernel type")

        distances = (X_samples_at_t - x) / bin_width
        weights = kernel(distances)
        weighted_sum = jnp.sum(weights * Xdot_samples_at_t)
        weight_total = jnp.sum(weights)
        # Avoid division by zero
        return jnp.where(weight_total > 1e-10, weighted_sum / weight_total, 0.0)

    def velocity_at_x(x):
        return jax.vmap(
            lambda X_samples_at_t, Xdot_samples_at_t: velocity_at_x_and_t(
                x, (X_samples_at_t, Xdot_samples_at_t)
            ),
            in_axes=(0, 0),
        )(X_samples, Xdot_samples)

    x_grid = jnp.linspace(x_span[0], x_span[1], 1 / bin_width)
    return jax.vmap(velocity_at_x)(x_grid)


def plot_velocity_field(
    velocity_field: Float[Array, "x_grid t_grid"],
    x_grid: Float[Array, "x_grid"],
    t_grid: Float[Array, "t_grid"],
):
    """
    Plot the velocity field v(x,t).

    :param velocity_field: Velocity values at grid points.
    :param x_grid: Grid points in x.
    :param t_grid: Grid points in t.

    :return: None.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        velocity_field.T,
        aspect="auto",
        origin="lower",
        extent=(t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]),
        cmap="RdBu_r",
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("Velocity field $v(x,t) = E[\\dot{X}_t | X_t = x]$")
    plt.colorbar(im, ax=ax, label="$v(x,t)$")
    plt.tight_layout()
    plt.show()


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


def plot_flow_field(
    interpolant: OptimalInterpolant,
    velocity_field: Float[Array, "x_grid t_grid"],
    x_grid: Float[Array, "x_grid"],
    t_grid: Float[Array, "t_grid"],
    key: jax.Array,
    N_flow_samples: int = 20,
):
    """
    Plot flow field with particle trajectories.

    :param interpolant: An instance of OptimalInterpolant.
    :param velocity_field: Precomputed velocity field.
    :param x_grid: Spatial grid for velocity field.
    :param t_grid: Time grid for velocity field.
    :param key: JAX PRNG key.
    :param N_flow_samples: Number of particle trajectories to simulate and plot.

    :return: None.
    """

    # Sample initial positions
    X0_samples = interpolant.Z.sample(N_samples=N_flow_samples, key=key)[:, 0]

    # Simulate all trajectories
    trajectories = []
    for i in range(N_flow_samples):
        traj = simulate_particle_trajectory(
            X0_samples[i], t_grid, x_grid, velocity_field
        )
        trajectories.append(traj)

    trajectories = jnp.stack(trajectories)  # Shape: (N_flow_samples, n_times)

    # Plot particle trajectories
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each trajectory
    for i in range(N_flow_samples):
        ax.plot(t_grid, trajectories[i], alpha=0.5, lw=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$X_t$")
    ax.set_title(f"Particle trajectories under flow (N={N_flow_samples})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================
    # Alternative visualization: trajectories in (t, x) space overlaid on velocity field
    # ==========================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Background: velocity field
    im = ax.imshow(
        velocity_field.T,
        aspect="auto",
        origin="lower",
        extent=(t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]),
        cmap="RdBu_r",
        alpha=0.6,
    )

    # Overlay particle trajectories
    for i in range(N_flow_samples):
        ax.plot(t_grid, trajectories[i], "k-", alpha=0.4, lw=0.8)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title(f"Particle trajectories on velocity field (N={N_flow_samples})")
    plt.colorbar(im, ax=ax, label="$v(x,t)$")
    plt.tight_layout()
    plt.show()


def visualize_interpolant_flow(
    interpolant: OptimalInterpolant,
    key: jax.Array,
    x_span: tuple[float, float],
    bin_width: Float,
    kernel_type: str = "gaussian",
    N_velocity_samples: int = 1000,
    N_flow_samples: int = 20,
):
    # Compute velocity field
    velocity_field = compute_velocity(
        interpolant,
        key,
        x_span,
        bin_width,
        kernel_type,
        N_samples=N_velocity_samples,
    )

    x_grid = jnp.linspace(
        x_span[0], x_span[1], int((x_span[1] - x_span[0]) / bin_width)
    )
    t_grid = interpolant.t

    # Plot velocity field
    plot_velocity_field(velocity_field, x_grid, t_grid)

    # Plot flow field with particle trajectories
    plot_flow_field(
        interpolant,
        velocity_field,
        x_grid,
        t_grid,
        key,
        N_flow_samples=N_flow_samples,
    )
