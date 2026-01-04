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
# %env JAX_DISABLE_JIT 1

# %%
from jaxtyping import Float, Array
import matplotlib.pyplot as plt
import optimistix as optx
import jax.numpy as jnp
import diffrax
import optimalinterp as oi
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)  # TODO Should we comment out for performance?


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


# %%
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


# %%
def residual(psi_dot_0, psi_1, *args, **solver_kwargs):
    N_terms = len(psi_1)
    # Recall that for an instation sol of diffrax.Solution the values sol.ys are of shape (time, y_dim)
    pred_y1_concat = solve(psi_dot_0, *args, **solver_kwargs)[-1]
    # Recall that y1_concat = [psi_real, psi_dot_real, psi_imag, psi_dot_imag]
    real_res = psi_1 - pred_y1_concat[:N_terms]
    im_res = pred_y1_concat[2 * N_terms : 3 * N_terms]
    return jnp.concat((real_res, im_res))


# %%
# Target initialization
target_mu, target_sigma = 2.0, 4.0
phi = oi.GaussianPhi1D(mu=target_mu, sigma=target_sigma)

# ODE initialization
N_terms, D_infl = 5, 0.0  # D_infl doesn't do anything right now.
term = diffrax.ODETerm(rhs)  # Create Diffrax term

# Boundary condition initialization
z = jnp.zeros(N_terms)
psi_0 = z.at[0].set(1.0)
psi_1 = z.at[-1].set(1.0)

# %%
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

# %%
# Test residual evaluation on initial guess to make sure it returns
residual(initial_psi_dot_0, psi_1, *args, **solver_kwargs)


# %%
# Create jit'ted residual for optimization
@jax.jit
def residual_fcn(psi_dot_0, _):
    return residual(psi_dot_0, psi_1, *args, **solver_kwargs)


# %%
# Choose optimizer as Levenberg--Marquardt
solver = optx.BestSoFarLeastSquares(
    optx.LevenbergMarquardt(
        rtol=1e-8,
        atol=1e-8,
        verbose=frozenset({"step", "accepted", "loss", "step_size"}),
    )
)

# %%
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
print("{},\n{}".format(sol.result, solver.norm(residual_fcn(sol.value, None)).item()))

# %%
# Get trajectory for the optimized value of $\dot{\psi}(0)$
N_t = 150
saveat_t = jnp.linspace(0, 1, N_t)
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


# %%
def eval_velocity(
    x: Float,
    psi: Float[Array, " N"],
    psi_dot: Float[Array, " N"],
    mu_Z: Float[Array, " N"],
    Sigma_Z: Float[Array, "N N"],
) -> Float:
    r"""
    Evaluate the conditional velocity $v(x,t) = E[\dot X_t | X_t = x].
    Use the formula
        v(x,t) = \sum_\alpha \dot \psi_\alpha(t) \E[Z_\alpha | X_t = x]
    Now recalling that
        X_t = \sum_\alpha \psi_\alpha(t) Z_\alpha
    and putting all the Z_\alpha in a joint-Gaussian space such that Z_alpha is indep. of Z_\gamma for \alpha \neq \gamma
    we can compute
        \E[Z_alpha | X_t = x] = ( \sigma_\alpha \psi_\alpha ) * ( x - \sum_\beta \psi_\beta \mu_\beta ) / ( \sum_\beta \psi_\beta^2 \sigma_\beta )
    where Z_\alpha \sim N(\mu_\alpha, \sigma_\alpha^2).

    :param x: Position variable in \Rd.
    :type x: Float
    :param psi: Coefficients psi_alpha(t) in vector form, for fixed time t.
    :type psi: Float[Array, "N"]
    :param psi_dot: Time derivatives of coefficients psi_alpha(t) in vector form, for fixed time t.
    :type psi_dot: Float[Array, "N"]
    :param mu_Z: Mean vector of the Gaussian variables Z_alpha.
    :type mu_Z: Float[Array, "N"]
    :param Sigma_Z: Covariance matrix of the Gaussian variables Z_alpha.
    :type Sigma_Z: Float[Array, "N N"]
    :return: conditional velocity v(x,t) = E[\dot X_t | X_t = x].
    :rtype: Float
    """

    N = psi.shape[0]
    assert psi_dot.shape[0] == N and mu_Z.shape[0] == N and Sigma_Z.shape == (N, N)
    assert jnp.allclose(Sigma_Z, jnp.diag(jnp.diag(Sigma_Z))), "Non-diagonal Sigma_Z!"
    var_X_t = jnp.dot(psi, Sigma_Z @ psi)  # Denote Z = (Z_alpha)_alpha
    mu_X_t = jnp.dot(mu_Z, psi)
    expect_Z_given_X_t = (
        mu_Z + (Sigma_Z @ psi * (x - mu_X_t)) / var_X_t
    )  # works because Sigma_Z is diagonal

    out = psi_dot @ expect_Z_given_X_t
    return out


def eval_vel_fcn(
    x, y_t: Float[Array, " 4*N"], mu_Z: Float[Array, " N"], Sigma_Z: Float[Array, "N N"]
) -> Float:
    r"""
    Wrapper to evaluate velocity from concatenated ODE solution y_t.

    :param x: Position variable in \Rd.
    :param y_t: Concatenated ODE solution vector at time t.
    :type y_t: Float[Array, "4*N"]
    :param mu_Z: Mean vector of the Gaussian variables Z_alpha.
    :type mu_Z: Float[Array, "N"]
    :param Sigma_Z: Covariance matrix of the Gaussian variables Z_alpha.
    :type Sigma_Z: Float[Array, "N N"]
    :return: conditional velocity $v(x,t) = E[\dot{X_t} | X_t = x]$.
    :rtype: Float
    """
    N = len(mu_Z)
    psi_concat = y_t[: 2 * N] + 1j * y_t[2 * N :]
    psi, psi_dot = psi_concat[:N], psi_concat[N:]
    return eval_velocity(x, psi, psi_dot, mu_Z, Sigma_Z)


# %%
mu_Z = jnp.linspace(0, phi.mu, N_terms)
Sigma_Z = jnp.diag((1 - mu_Z) ** 2 + (mu_Z**2) * phi.sigma**2)
print("Mean: {}\nCovariance:\n{}".format(mu_Z, Sigma_Z))

# %%
# Make matrix-valued function (x_i, psi(t_j)) -> v(x_i, t_j)
velocity_vmap = jax.vmap(
    jax.vmap(
        lambda x, y_t: eval_vel_fcn(x, y_t, mu_Z, Sigma_Z),
        in_axes=(0, None),
    ),
    in_axes=(None, 0),
)
velocity_eval = velocity_vmap(jnp.linspace(-5, 5), ode_sol)

# %%
# ----------------------------------------------------------------------------------------------
# TODO: this is WRONG! When solves a dynamical system, one may not just integrate the vlocity.
# Consider dynamics v(x,t) = x and the ODE d/dt x(t) = v(x(t),t) with x(0) = x0.
# The solution is x(t) = x(0) exp(t), but integrating the velocity gives x(t) = x(0) + x(0) t.
# ----------------------------------------------------------------------------------------------


## Simple quadrature of marginal velocity
# plt.plot(jnp.cumsum(jnp.real(velocity_eval)[:, velocity_eval.shape[1] // 2]) / N_t)

## Plot entire velocity field
# fig, ax = plt.subplots(figsize=(3, 3))
# c = ax.imshow(jnp.real(velocity_eval).T, aspect=0.1, extent=(0, 1, -5, 5))
# fig.colorbar(c)
# ax.set_xlabel("t")
# ax.set_ylabel("x")
# plt.show()

# %%
