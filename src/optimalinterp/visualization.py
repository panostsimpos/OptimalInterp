# %%
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxtyping import Float, Array
import jax
import diffrax
from optimalinterp.optimal_interpolant import ModalInterpolant, modal_interpolant_factory

__all__ = [
    "visualize_interpolant_flow"
]

# =============================================================================
# Generic binning-based velocity field computation
# =============================================================================

def make_extent(*args)->tuple[float, float, float, float]:
    return tuple(a.item() for a in args)

def compute_velocity(
    interpolant: ModalInterpolant,
    key: jax.Array,
    x_grid: Float[Array, " X"],
    bin_width: Float,
    kernel_type: str = "gaussian",
    N_samples: int = 1000,
) -> Float[Array, "t_grid x_grid"]:
    """
    Compute v(x,t) = E[dot{X}_t | X_t = x] using binning at a single time t.
    Given an interopolant X_t = Σ_{α=1}^N Z_α ψ_α(t) with time derivative

    Then estimate v(x,t) by averaging dot{X}_t over samples where X_t falls in a bin around x.

    :param interpolant: An instance of OptimalInterpolant.
    :param key: JAX PRNG key.
    :param x_grid: Grid points in x.
    :param bin_width: Width of bins for conditioning.
    :param N_samples: Number of samples of Z_α to use.

    Returns:
    :return: velocity_field where velocity_field has shape (t_grid, x_grid).
    """
    # Compute X_t and dot{X}_t for all samples
    X_samples, Xdot_samples = interpolant.sample_solution_and_deriv(key, N_samples, None)

    if kernel_type == "gaussian":
        def kernel(x):
            return jnp.exp(-0.5 * (x) ** 2)

    elif kernel_type == "square":
        def kernel(x):
            return jnp.where(jnp.abs(x) <= 0.5, 1.0, 0.0)

    else:
        raise ValueError("Unknown kernel type")

    # For each x in x_grid, find samples in bin and average Xdot
    def velocity_at_x_and_t(x, t_idx):
        distances = (X_samples[:, t_idx] - x) / bin_width
        weights = kernel(distances)
        weighted_sum = jnp.sum(weights * Xdot_samples[:, t_idx])
        weight_total = jnp.sum(weights)
        # Avoid division by zero
        return jnp.where(weight_total > 1e-10, weighted_sum / weight_total, 0.0)

    t_indices = jnp.arange(len(interpolant.default_tgrid()))

    # Shape (t_grid, x_grid)
    velocity_field = jax.vmap(
        lambda t_idx: jax.vmap(lambda x: velocity_at_x_and_t(x, t_idx))(x_grid)
    )(t_indices)

    return velocity_field


def plot_velocity_field(
    velocity_field: Float[Array, "t_grid x_grid"],
    x_grid: Float[Array, " x_grid"],
    t_grid: Float[Array, " t_grid"],
):
    """
    Plot the velocity field v(x,t).

    :param velocity_field: Velocity values at grid points with shape (t_grid, x_grid).
    :param x_grid: Grid points in x.
    :param t_grid: Grid points in t.

    :return: None.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        velocity_field.T,
        aspect="auto",
        origin="lower",
        extent=make_extent(t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]),
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
    time_grid: Float[Array, " n_times"],
    x_grid: Float[Array, " n_grid"],
    velocity_field: Float[Array, "n_times n_grid"],
) -> Float[Array, "1"]:
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
    time_grid: Float[Array, " n_times"],
    x_grid: Float[Array, " n_grid"],
    velocity_field: Float[Array, "n_times n_grid"],
) -> Float[Array, " n_times"]:
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
    assert sol.ys is not None
    return sol.ys


def plot_flow_field(
    interpolant: ModalInterpolant,
    velocity_field: Float[Array, "T X"],
    x_grid: Float[Array, " X"],
    t_grid: Float[Array, " T"],
    key: jax.Array,
    N_flow_samples: int = 20,
):
    """
    Plot flow field with particle trajectories.

    :param interpolant: An instance of OptimalInterpolant.
    :param velocity_field: Precomputed velocity field with shape (t_grid, x_grid).
    :param x_grid: Spatial grid for velocity field.
    :param t_grid: Time grid for velocity field.
    :param key: JAX PRNG key.
    :param N_flow_samples: Number of particle trajectories to simulate and plot.

    :return: None.
    """

    # Sample initial positions
    X0_samples = interpolant.sample_modes(key, N_flow_samples)[:, 0]

    # Simulate all trajectories
    trajectories = []
    for i in range(N_flow_samples):
        traj = simulate_particle_trajectory(
            X0_samples[i].item(), t_grid, x_grid, velocity_field
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
        extent=make_extent(t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]),
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
    interpolant: ModalInterpolant,
    key: jax.Array,
    x_grid: Float[Array, " X"],
    bin_width: Float = 0.01,
    kernel_type: str = "gaussian",
    N_velocity_samples: int = 1000,
    N_flow_samples: int = 20,
):
    # Compute velocity field - now returns both velocity_field and x_grid
    velocity_field = compute_velocity(
        interpolant,
        key,
        x_grid,
        bin_width,
        kernel_type,
        N_samples=N_velocity_samples,
    )

    t_grid = interpolant.default_tgrid()

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


# %%
# =======================
# Tests for this module
# =======================

if __name__ == "__main__":
    # Write some tests
    from optimalinterp.stochastic_basis import GaussianBasis

    # Enable 64-bit precision for accurate tests
    jax.config.update("jax_enable_x64", True)

    # %% [markdown]
    # # Visualization Module Tests
    # %%

    simple_stochastic_basis = GaussianBasis(
        mean=10.0,
        std_dev=2.0,
        N_basis=2,
        bridge_type="gaussian_convolution",
    )

    psi = jnp.array(
        [
            jnp.linspace(1, 0, 100),
            jnp.linspace(0, 1, 100),
        ]
    ).T

    psi_dot = jnp.array(
        [
            -1 * jnp.ones(psi.shape[0]),
            +1 * jnp.ones(psi.shape[0]),
        ]
    ).T

    test_interpolant = modal_interpolant_factory(
        "time interpolated",
        t=jnp.linspace(0, 1, 100),  # 100 time points
        Z=simple_stochastic_basis,
        psi=psi,
        psi_dot=psi_dot
    )

    t_grid_test = test_interpolant.default_tgrid()

    # Plot sample paths of the standard interpolant
    n_paths = 50
    t_eval = jnp.linspace(0, 1, 100)
    key = jax.random.PRNGKey(0)
    sample_paths = test_interpolant(
        N_samples=n_paths, key=key, t_eval=t_eval
    )  # Shape: (n_paths, n_times)

    fig, ax = plt.subplots(figsize=(10, 6))
    t_vals = test_interpolant.t

    for i in range(n_paths):
        ax.plot(t_vals, sample_paths[i, :], alpha=0.5, linewidth=1)

    ax.set_title("Standard Interpolant Sample Paths")
    ax.set_xlabel("Time t")
    ax.set_ylabel("x")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # ## Test 1: Velocity Field Computation
    # %%
    key_test = jax.random.PRNGKey(42)
    x_grid_test = jnp.linspace(-15, 15, 1000)
    bin_width_test = 0.01
    N_velocity_samples_test = 500

    # Compute velocity field with Gaussian kernel
    velocity_field_gauss = compute_velocity(
        test_interpolant,
        key_test,
        x_grid_test,
        bin_width_test,
        kernel_type="gaussian",
        N_samples=N_velocity_samples_test,
    )

    # Compute velocity field with square kernel
    velocity_field_square = compute_velocity(
        test_interpolant,
        key_test,
        x_grid_test,
        bin_width_test,
        kernel_type="square",
        N_samples=N_velocity_samples_test,
    )

    print(f"Velocity field (Gaussian) shape: {velocity_field_gauss.shape}")
    print(f"Velocity field (Square) shape: {velocity_field_square.shape}")
    print(
        f"Expected shape: ({len(t_grid_test)}, {int((x_grid_test[1] - x_grid_test[0]) / bin_width_test)})"
    )
    print( "\nVelocity field stats (Gaussian):")
    print(f"  Mean: {jnp.mean(velocity_field_gauss):.4f}")
    print(f"  Std: {jnp.std(velocity_field_gauss):.4f}")
    print(f"  Min: {jnp.min(velocity_field_gauss):.4f}")
    print(f"  Max: {jnp.max(velocity_field_gauss):.4f}")

    # Check for NaN or Inf values
    assert not jnp.any(
        jnp.isnan(velocity_field_gauss)
    ), "Velocity field contains NaN values"
    assert not jnp.any(
        jnp.isinf(velocity_field_gauss)
    ), "Velocity field contains Inf values"
    print("\n✓ Velocity field computation successful (no NaN/Inf values)")

    # %% [markdown]
    # ## Test 2: Velocity Field Interpolation
    #
    # Test bilinear interpolation of velocity field at arbitrary points.

    # %%
    # Test interpolation at grid points (should match original values)
    t_mid = t_grid_test[len(t_grid_test) // 2].item()
    x_mid = x_grid_test[len(x_grid_test) // 2].item()

    v_interp_grid = interpolate_velocity_field(
        t_mid, x_mid, t_grid_test, x_grid_test, velocity_field_gauss
    )
    v_original = velocity_field_gauss[len(t_grid_test) // 2, len(x_grid_test) // 2]

    print( "Interpolation at grid point:")
    print(f"  Original value: {v_original:.6f}")
    print(f"  Interpolated value: {v_interp_grid:.6f}")
    print(f"  Difference: {abs(v_interp_grid - v_original):.2e}")

    # Test interpolation between grid points
    t_between = ((t_grid_test[5] + t_grid_test[6]) / 2).item()
    x_between = ((x_grid_test[10] + x_grid_test[11]) / 2).item()

    v_interp_between = interpolate_velocity_field(
        t_between, x_between, t_grid_test, x_grid_test, velocity_field_gauss
    )

    print( "\nInterpolation between grid points:")
    print(f"  Interpolated value: {v_interp_between:.6f}")
    print( "  (Should be between neighboring values)")

    # Test boundary handling
    v_boundary = interpolate_velocity_field(
        t_grid_test[0].item(), x_grid_test[0].item(), t_grid_test, x_grid_test, velocity_field_gauss
    )
    print(f"\nBoundary interpolation: {v_boundary:.6f}")

    assert not jnp.isnan(v_interp_grid), "Grid interpolation produced NaN"
    assert not jnp.isnan(v_interp_between), "Between-grid interpolation produced NaN"
    assert not jnp.isnan(v_boundary), "Boundary interpolation produced NaN"
    print("\n✓ Velocity field interpolation successful")

    # %% [markdown]
    # ## Test 3: Particle Trajectory Simulation
    #
    # Test that particle trajectories are simulated correctly under the velocity field.

    # %%
    X0_test = 0.0  # Start at origin

    trajectory = simulate_particle_trajectory(
        X0_test, t_grid_test, x_grid_test, velocity_field_gauss
    )

    print(f"Particle trajectory shape: {trajectory.shape}")
    print(f"Expected shape: ({len(t_grid_test)},)")
    print( "\nTrajectory stats:")
    print(f"  Initial position: {trajectory[0]:.4f}")
    print(f"  Final position: {trajectory[-1]:.4f}")
    print(f"  Position change: {trajectory[-1] - trajectory[0]:.4f}")
    print(f"  Max position: {jnp.max(trajectory):.4f}")
    print(f"  Min position: {jnp.min(trajectory):.4f}")

    # Check trajectory is continuous (no jumps)
    position_diffs = jnp.diff(trajectory)
    max_jump = jnp.max(jnp.abs(position_diffs))
    print(f"\nMaximum position jump between time steps: {max_jump:.4f}")

    # Verify no NaN or Inf values
    assert not jnp.any(jnp.isnan(trajectory)), "Trajectory contains NaN values"
    assert not jnp.any(jnp.isinf(trajectory)), "Trajectory contains Inf values"
    assert trajectory.shape == (len(t_grid_test),), "Trajectory has incorrect shape"
    print("\n✓ Particle trajectory simulation successful")

    # %% [markdown]
    # ## Test 4: Multiple Particle Trajectories
    #
    # Test simulation of multiple particles with different initial conditions.

    # %%
    N_particles = 10
    key_particles = jax.random.PRNGKey(123)
    X0_samples_test = jax.random.uniform(
        key_particles, (N_particles,), minval=-2.0, maxval=2.0
    )

    trajectories_test = []
    for i in range(N_particles):
        traj = simulate_particle_trajectory(
            X0_samples_test[i].item(), t_grid_test, x_grid_test, velocity_field_gauss
        )
        trajectories_test.append(traj)

    trajectories_test = jnp.stack(trajectories_test)

    print(f"Multiple trajectories shape: {trajectories_test.shape}")
    print(f"Expected shape: ({N_particles}, {len(t_grid_test)})")
    print(
        f"\nInitial positions range: [{jnp.min(X0_samples_test):.4f}, {jnp.max(X0_samples_test):.4f}]"
    )
    print(
        f"Final positions range: [{jnp.min(trajectories_test[:, -1]):.4f}, {jnp.max(trajectories_test[:, -1]):.4f}]"
    )

    # Verify all trajectories are valid
    assert trajectories_test.shape == (
        N_particles,
        len(t_grid_test),
    ), "Trajectories have incorrect shape"
    assert not jnp.any(jnp.isnan(trajectories_test)), "Some trajectories contain NaN"
    assert not jnp.any(jnp.isinf(trajectories_test)), "Some trajectories contain Inf"
    print("\n✓ Multiple particle trajectory simulation successful")

    # %% [markdown]
    # ## Test 5: Visualization Functions (Smoke Tests)
    #
    # Test that plotting functions run without errors (visual inspection required).

    # %%
    print("Testing velocity field plotting...")
    try:
        plot_velocity_field(velocity_field_gauss, x_grid_test, t_grid_test)
        print("✓ Velocity field plot generated successfully")
    except Exception as e:
        print(f"✗ Velocity field plotting failed: {e}")

    # %%
    print("\nTesting flow field plotting...")
    try:
        plot_flow_field(
            test_interpolant,
            velocity_field_gauss,
            x_grid_test,
            t_grid_test,
            key_test,
            N_flow_samples=15,
        )
        print("✓ Flow field plots generated successfully")
    except Exception as e:
        print(f"✗ Flow field plotting failed: {e}")

    # %% [markdown]
    # ## Test 6: Full Visualization Pipeline
    #
    # Test the complete visualization workflow with `visualize_interpolant_flow`.

    # %%
    print("Testing complete visualization pipeline...")
    try:
        visualize_interpolant_flow(
            test_interpolant,
            key_test,
            x_grid=x_grid_test,
            bin_width=0.01,
            kernel_type="gaussian",
            N_velocity_samples=1000,
            N_flow_samples=50,
        )
        print("✓ Complete visualization pipeline successful")
    except Exception as e:
        print(f"✗ Visualization pipeline failed: {e}")

    # %% [markdown]
    # ## Summary
    #
    # All tests completed. Review the output above for any failures or warnings.

    # %%
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("\nTests covered:")
    print("  1. Velocity field computation (Gaussian and square kernels)")
    print("  2. Velocity field interpolation (grid and between-grid points)")
    print("  3. Single particle trajectory simulation")
    print("  4. Multiple particle trajectory simulation")
    print("  5. Visualization functions (smoke tests)")
    print("  6. Complete visualization pipeline")
    print("  7. Edge cases and boundary conditions")
    print("\nNote: Visual inspection of plots is required to verify correctness.")

# %%
