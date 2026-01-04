# %%
from jaxtyping import Float, Array
import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
import diffrax
import optimalinterp as oi
import jax
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)


class OptimalInterpSolution(NamedTuple):
    """Solution container for optimal interpolation BVP."""

    t: Float[Array, " T"]  # Time points
    psi: Float[Array, "T N"]  # Coefficients ψ_α(t)
    psi_dot: Float[Array, "T N"]  # Velocities dψ_α/dt
    initial_velocity: Float[Array, " N"]  # Solved ψ̇(0)
    residual_norm: float  # Final BVP residual
    success: bool  # Whether shooting converged


def rhs(t, y, args):
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


def solve(psi_dot_0, *args, **solver_kwargs):
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
    assert sol_ys is not None
    return jax.lax.cond(
        sol.result._value == 0, lambda: sol_ys, lambda: jnp.inf * sol_ys
    )


def residual(psi_dot_0, psi_1, *args, **solver_kwargs):
    N_terms = len(psi_1)
    # Recall that for an instation sol of diffrax.Solution the values sol.ys are of shape (time, y_dim)
    pred_y1_concat = solve(psi_dot_0, *args, **solver_kwargs)[-1]
    # Recall that y1_concat = [psi_real, psi_dot_real, psi_imag, psi_dot_imag]
    real_res = psi_1 - pred_y1_concat[:N_terms]
    im_res = pred_y1_concat[2 * N_terms : 3 * N_terms]
    return jnp.concat((real_res, im_res))


def solve_for_psi(
    t_span: tuple[float, float] = (0.0, 1.0),
    n_time_points: int = 150,
    Phi: oi.MomentGeneratingPhi = None,
) -> OptimalInterpSolution:

    # ODE initialization
    N_terms, D_infl = 5, 0.0  # D_infl doesn't do anything right now.
    term = diffrax.ODETerm(rhs)  # Create Diffrax term

    # Boundary condition initialization
    z = jnp.zeros(N_terms)
    psi_0 = z.at[0].set(1.0)
    psi_1 = z.at[-1].set(1.0)

    # ODE solve and optimization parameters
    solver = diffrax.Kvaerno5()  # ODE time discretization
    solver_args = (phi, D_infl)
    args = (psi_0, solver, term, solver_args)

    initial_psi_dot_0 = z
    solver_kwargs = {
        "adjoint": diffrax.DirectAdjoint(),
        "max_steps": 5000,
        "stepsize_controller": diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-8),
        "dt0": 1e-3,
    }

    # Test residual evaluation on initial guess to make sure it returns
    residual(initial_psi_dot_0, psi_1, *args, **solver_kwargs)

    # Create jit'ted residual for optimization
    @jax.jit
    def residual_fcn(psi_dot_0, _):
        return residual(psi_dot_0, psi_1, *args, **solver_kwargs)

    # Choose optimizer as Levenberg--Marquardt
    solver = optx.BestSoFarLeastSquares(
        optx.LevenbergMarquardt(
            rtol=1e-8,
            atol=1e-8,
            verbose=frozenset({"step", "accepted", "loss", "step_size"}),
        )
    )

    # Perform optimization
    N_optimizer_step = 1000
    sol = optx.least_squares(
        residual_fcn,
        solver,
        initial_psi_dot_0,
        max_steps=N_optimizer_step,
        throw=False,
    )

    # %%
    # Print optimization result and final residual
    print(
        "{},\n{}".format(sol.result, solver.norm(residual_fcn(sol.value, None)).item())
    )

    # %%
    # Get trajectory for the optimized value of $\dot{\psi}(0)$
    saveat_t = jnp.linspace(t_span[0], t_span[1], n_time_points)
    saveat = diffrax.SaveAt(ts=saveat_t)
    ode_sol = solve(sol.value, *args, saveat=saveat, **solver_kwargs)
    y1_concat = ode_sol[-1]
    psi_1_concat = y1_concat[: 2 * N_terms] + 1j * y1_concat[2 * N_terms :]
    psi_1 = psi_1_concat[:N_terms]

    # %%
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
