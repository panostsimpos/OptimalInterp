# %%
"""Solver for optimal interpolation BVP."""

# %%
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
from jaxtyping import Float, Array
from typing import NamedTuple
import optimalinterp as oi
import matplotlib.pyplot as plt

# %%

jax.config.update("jax_enable_x64", True)


class OptimalInterpSolution(NamedTuple):
    """Solution container for optimal interpolation BVP."""

    t: Float[Array, " T"]  # Time points
    psi: Float[Array, "T N"]  # Coefficients ψ_α(t)
    psi_dot: Float[Array, "T N"]  # Velocities dψ_α/dt
    initial_velocity: Float[Array, " N"]  # Solved ψ̇(0)
    residual_norm: float  # Final BVP residual
    success: bool  # Whether shooting converged


def solve_optimal_interp_bvp(
    psi_0: Float[Array, " N"],
    psi_1: Float[Array, " N"],
    phi: oi.MomentGeneratingPhi,
    D_infl: float = 1e-4,
    t_span: tuple[float, float] = (0.0, 1.0),
    n_save: int = 1001,
    initial_psi_dot_0: Float[Array, " N"] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 5000,
    shooting_max_steps: int | None = None,
    verbose: bool = False,
) -> OptimalInterpSolution:
    """
    Solve optimal interpolation BVP using shooting method.

    Parameters
    ----------
    psi_0 : Initial coefficients ψ(t_0)
    psi_1 : Final coefficients ψ(t_1)
    phi : Moment generating function (e.g., GaussianPhi1D)
    D_infl : Diffusion inflation parameter
    t_span : (t_0, t_1) time interval
    n_save : Number of time points to save
    initial_psi_dot_0 : Initial guess for ψ̇(0) (default: zeros)
    rtol, atol : Relative/absolute tolerances for ODE solver
    max_steps : Max steps for ODE integration
    shooting_max_steps : Max steps for shooting solver (default: no limit)
    verbose : Print shooting progress

    Returns
    -------
    OptimalInterpSolution with trajectory and diagnostics
    """
    N_terms = len(psi_0)
    assert len(psi_1) == N_terms, "psi_0 and psi_1 must have same length"

    if initial_psi_dot_0 is None:
        initial_psi_dot_0 = jnp.zeros(N_terms, dtype=jnp.complex128)

    # Setup ODE system
    def rhs(t, y, args):
        N = len(y) // 2
        psi = y[:N] + 1j * y[N:]
        _, phi_arg, D = args
        d_psi = oi.ode_residual.OptimalInterpBVP_ODE_RHS(psi, phi_arg, D)
        return jnp.concat((jnp.real(d_psi), jnp.imag(d_psi)))

    def lhs(dy, t, y, args, control):
        _, phi_arg, _ = args
        return oi.ode_residual.OptimalInterpBVP_DAE_LHS(dy, y, phi_arg)

    args = (lhs, phi, D_infl)
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmin=1e-10)

    # Solve function for shooting
    def solve(psi_dot_0, *solve_args, **solver_kwargs):
        psi_0_arg, solver_arg, term_arg, args_arg = solve_args
        y0 = jnp.concat(
            (
                jnp.real(psi_0_arg),
                jnp.imag(psi_0_arg),
                jnp.real(psi_dot_0),
                jnp.imag(psi_dot_0),
            )
        )
        sol = diffrax.diffeqsolve(
            term_arg,
            solver_arg,
            t0=t_span[0],
            t1=t_span[1],
            y0=y0,
            args=args_arg,
            throw=False,
            **solver_kwargs,
        )
        sol_ys = sol.ys
        assert sol_ys is not None
        # Return inf if solver failed
        return jax.lax.cond(
            sol.result._value == 0, lambda: sol_ys, lambda: jnp.inf * sol_ys
        )

    # Residual function
    def residual(psi_dot_0, psi_1_arg, *solve_args, **solver_kwargs):
        pred_y1 = solve(psi_dot_0, *solve_args, **solver_kwargs)[-1]
        # y = [Re(ψ), Im(ψ), Re(ψ̇), Im(ψ̇)]
        pred_psi_1 = pred_y1[:N_terms] + 1j * pred_y1[N_terms : 2 * N_terms]
        residual_vec = psi_1_arg - pred_psi_1
        return jnp.concat((jnp.real(residual_vec), jnp.imag(residual_vec)))

    solve_args = (psi_0, solver, term, args)
    solver_kwargs = {
        "adjoint": diffrax.DirectAdjoint(),
        "max_steps": max_steps,
        "stepsize_controller": stepsize_controller,
        "dt0": (t_span[1] - t_span[0]) / 1000,
    }

    # Shooting solver
    @jax.jit
    def residual_fcn(psi_dot_0, _):
        return residual(psi_dot_0, psi_1, *solve_args, **solver_kwargs)

    verbose_set = frozenset({"step", "loss"}) if verbose else frozenset()
    shooting_solver = optx.BestSoFarLeastSquares(
        optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=verbose_set)
    )

    shooting_sol = optx.least_squares(
        residual_fcn,
        shooting_solver,
        initial_psi_dot_0,
        max_steps=shooting_max_steps,
        throw=False,
    )

    # Get final trajectory with dense output
    saveat_t = jnp.linspace(t_span[0], t_span[1], n_save)
    saveat = diffrax.SaveAt(ts=saveat_t)

    final_ys = solve(shooting_sol.value, *solve_args, saveat=saveat, **solver_kwargs)

    # Extract psi and psi_dot
    psi_t = final_ys[:, :N_terms] + 1j * final_ys[:, N_terms : 2 * N_terms]
    psi_dot_t = final_ys[:, 2 * N_terms : 3 * N_terms] + 1j * final_ys[:, 3 * N_terms :]

    residual_norm = shooting_solver.norm(residual_fcn(shooting_sol.value, None)).item()
    success = shooting_sol.result == optx.RESULTS.successful

    return OptimalInterpSolution(
        t=saveat_t,
        psi=psi_t,
        psi_dot=psi_dot_t,
        initial_velocity=shooting_sol.value,
        residual_norm=residual_norm,
        success=success,
    )


# %%
# Define problem
N_terms = 5
psi_0 = jnp.zeros(N_terms).at[0].set(1.0)
psi_1 = jnp.zeros(N_terms).at[-1].set(1.0)
phi = oi.GaussianPhi1D(mu=2.0, sigma=4.0)

# Solve
solution = solve_optimal_interp_bvp(
    psi_0=psi_0,
    psi_1=psi_1,
    phi=phi,
    D_infl=1e-4,
    verbose=True,
)

# Check results
print(f"Success: {solution.success}")
print(f"Residual norm: {solution.residual_norm:.2e}")
print(f"Final psi: {solution.psi[-1]}")

# Plot trajectory
plt.figure(figsize=(8, 4))
plt.plot(solution.t, jnp.real(solution.psi), lw=2)
plt.xlabel("t")
plt.ylabel("Re(ψ)")
plt.title(f"Optimal interpolation ({N_terms} terms)")
plt.legend([f"ψ_{i}" for i in range(N_terms)])
plt.show()

# %%
