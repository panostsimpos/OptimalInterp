from jaxtyping import Float, Array, PyTree, Real
import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
import diffrax
import optimalinterp as oi
import jax
from typing import NamedTuple

__all__ = ["OptimalInterpBVPShootingSolution", "solve"]


class OptimalInterpBVPShootingSolution(NamedTuple):
    """Solution container for optimal interpolation BVP.

    Attributes:
        t: Time points of shape (T,)
        psi: Real coefficient trajectories ψ_α(t) of shape (T, N)
        psi_dot: Real velocity trajectories dψ_α/dt of shape (T, N)
        initial_velocity: Solved initial velocity ψ̇(0) of shape (N,)
        residual_norm: Final BVP residual ||ψ(1) - ψ_target||
        success: Whether shooting method converged
    """

    t: Float[Array, " T"]
    psi: Float[Array, "T N"]
    psi_dot: Float[Array, "T N"]
    initial_velocity: Float[Array, " N"]
    residual_norm: float
    optimization_success: bool
    solver_success: bool


def rhs(t: Real, y: PyTree[Float[Array, " 4*N"]], args: PyTree):
    r"""
    Simulate RHS of $\dot{y} = M(y)^{-1}f(y), where
    $$y = (re(\psi), re(\dot{\psi}), im(\psi), im(\dot{\psi}))$$
    """
    N_psi = len(y) // 2  # len(y) = 4 * N_terms
    psi_concat_real, psi_concat_imag = y[:N_psi], y[N_psi:]
    concat_psi = psi_concat_real + (1j * psi_concat_imag)
    phi, D_infl = args
    d_psi_concat = oi.ode_residual.OptimalInterpBVP_ODE_RHS(concat_psi, phi, D_infl)
    dy = jnp.concat((jnp.real(d_psi_concat), jnp.imag(d_psi_concat)))
    return dy


def shoot_once(
    psi_dot_0: Float[Array, " N"], *args, **solver_kwargs
) -> tuple[Float[Array, " 4*N"], diffrax.RESULTS]:
    r"Given $\dot{\psi}(0)$, return $\psi(1)$ satisfying ODE."
    psi_0, solver, term, solver_args = args
    N_terms = len(psi_0)
    y0 = jnp.concat((psi_0, psi_dot_0, jnp.zeros(2 * N_terms)))
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
    sol_ys = sol.ys
    sol_result = sol.result
    assert sol_ys is not None
    return jax.lax.cond(
        sol_result == diffrax.RESULTS.successful,
        lambda: (sol_ys, sol_result),
        lambda: (jnp.inf * sol_ys, sol_result),
    )


def residual(
    psi_dot_0: Float[Array, " N"], psi_1: Float[Array, " N"], *args, **solver_kwargs
):
    N_terms = len(psi_1)
    # Recall that for an instantiation sol of diffrax.Solution the values sol.ys are of shape (time, y_dim)
    trajectory, _ = shoot_once(psi_dot_0, *args, **solver_kwargs)
    pred_y1_concat = trajectory[-1]
    # Recall that y1_concat = [psi_real, psi_dot_real, psi_imag, psi_dot_imag]
    real_res = psi_1 - pred_y1_concat[:N_terms]
    im_res = pred_y1_concat[2 * N_terms : 3 * N_terms]
    return jnp.concat((real_res, im_res))


def solve(
    Phi: oi.MomentGeneratingPhi,
    N_terms: int,
    D_infl: float = 0.0,
    t_span: tuple[float, float] = (0.0, 1.0),
    n_time_points: int = 150,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_solver_steps: int = 5000,
    max_optimizer_steps: int = 1000,
    verbose: bool = True,
    plot_solution: bool = True,
    only_return_real_part: bool = True,
    solver: diffrax.AbstractSolver = diffrax.Kvaerno5(),
) -> OptimalInterpBVPShootingSolution:

    term = diffrax.ODETerm(rhs)  # Create Diffrax term

    # Boundary condition initialization
    z = jnp.zeros(N_terms)
    psi_0 = z.at[0].set(1.0)
    psi_1 = z.at[-1].set(1.0)

    # ODE solve and optimization parameters
    solver_args = (Phi, D_infl)
    args = (psi_0, solver, term, solver_args)

    initial_psi_dot_0 = z
    solver_kwargs = {
        "adjoint": diffrax.DirectAdjoint(),
        "max_steps": max_solver_steps,
        "stepsize_controller": diffrax.PIDController(rtol=rtol, atol=atol, dtmin=1e-8),
        "dt0": 1e-3,
    }

    # Test residual evaluation on initial guess to make sure it returns
    residual(initial_psi_dot_0, psi_1, *args, **solver_kwargs)

    # Create jit'ted residual for optimization
    @jax.jit
    def residual_fcn(psi_dot_0, _):
        return residual(psi_dot_0, psi_1, *args, **solver_kwargs)

    # Choose optimizer as Levenberg--Marquardt
    optimizer = optx.BestSoFarLeastSquares(
        optx.LevenbergMarquardt(
            rtol=1e-8,
            atol=1e-8,
            verbose=verbose,
        )
    )

    # Perform optimization
    opt_sol = optx.least_squares(
        residual_fcn,
        optimizer,
        initial_psi_dot_0,
        max_steps=max_optimizer_steps,
        throw=False,
    )

    opt_obj_value = optimizer.norm(residual_fcn(opt_sol.value, None)).item()

    if verbose:
        # Print optimization result and final residual
        jax.debug.print("{},\n{}".format(opt_sol.result, opt_obj_value))

    # Get trajectory for the optimized value of $\dot{\psi}(0)$
    saveat_t = jnp.linspace(t_span[0], t_span[1], n_time_points)
    saveat = diffrax.SaveAt(ts=saveat_t)
    ode_sol, ode_result = shoot_once(
        opt_sol.value, *args, saveat=saveat, **solver_kwargs
    )
    if only_return_real_part:
        psi_t = ode_sol[:, :N_terms]
        psi_dot_t = ode_sol[:, N_terms : 2 * N_terms]
    else:
        concat_psi_t = (
            ode_sol[:, : 2 * N_terms] + 1j * ode_sol[:, 2 * N_terms : 4 * N_terms]
        )
        psi_t = concat_psi_t[:, :N_terms]
        psi_dot_t = concat_psi_t[:, N_terms:]

    if plot_solution:
        plt.plot(
            saveat_t,
            ode_sol[:, :N_terms],
            lw=3,
            label=["$\\psi_{}$".format(j) for j in range(N_terms)],
        )
        plt.title("Approx optimal soln for {} $\\psi_\\alpha$ terms".format(N_terms))
        plt.xlabel("$t$")
        plt.legend()
        plt.show()
    optimization_success = (opt_sol.result == optx.RESULTS.successful).item()
    solver_success = (ode_result == diffrax.RESULTS.successful).item()
    return OptimalInterpBVPShootingSolution(
        t=saveat_t,
        psi=psi_t,
        psi_dot=psi_dot_t,
        initial_velocity=opt_sol.value,
        residual_norm=opt_obj_value,
        optimization_success=optimization_success,
        solver_success=solver_success,
    )


# %%
# Test the BVP solver
if __name__ == "__main__":
    # Define test problem parameters
    N_terms = 5
    Phi = oi.GaussianConvolutionPhi1D(N_terms, mu=2.0, sigma=4.0)

    # Solve BVP with shooting method
    print("Solving BVP with shooting method...")
    solution = solve(
        Phi=Phi,
        N_terms=N_terms,
        D_infl=0.0,
        t_span=(0.0, 1.0),
        n_time_points=150,
        rtol=1e-8,
        atol=1e-8,
        max_solver_steps=5000,
        verbose=True,
        plot_solution=True,
        only_return_real_part=True,
    )
    # %%

    # Validate solution
    print("\n" + "=" * 60)
    print("Solution Validation:")
    print("=" * 60)

    # Check boundary conditions
    psi_0_expected = jnp.zeros(N_terms).at[0].set(1.0)
    psi_1_expected = jnp.zeros(N_terms).at[-1].set(1.0)

    bc_error_0 = jnp.linalg.norm(solution.psi[0] - psi_0_expected)
    bc_error_1 = jnp.linalg.norm(solution.psi[-1] - psi_1_expected)

    print(f"Boundary condition at t=0: ||ψ(0) - ψ_target|| = {bc_error_0:.2e}")
    print(f"Boundary condition at t=1: ||ψ(1) - ψ_target|| = {bc_error_1:.2e}")
    print(f"Final residual norm: {solution.residual_norm:.2e}")
    print(f"Optimization converged: {solution.optimization_success}")
    print(f"ODE solver succeeded: {solution.solver_success}")

    # # Check solution quality
    # tol = 1e-6
    # assert bc_error_0 < tol, f"Initial BC violated: {bc_error_0}"
    # assert bc_error_1 < tol, f"Final BC violated: {bc_error_1}"
    # assert solution.optimization_success, "Optimization failed to converge"
    # assert solution.solver_success, "ODE solver failed"

    # print("\n✓ All tests passed!")

    # Plot velocity field
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(solution.t, solution.psi, lw=2)
    plt.xlabel("$t$")
    plt.ylabel("$\\psi_\\alpha(t)$")
    plt.title("Coefficient trajectories")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(solution.t, solution.psi_dot, lw=2)
    plt.xlabel("$t$")
    plt.ylabel("$\\dot{\\psi}_\\alpha(t)$")
    plt.title("Velocity trajectories")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
