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
psi_0 = psi_0.at[0].set(1. + 0.0j)
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
sol = diffrax.diffeqsolve(term, solver, t0=saveat_t[0], t1=saveat_t[-1], dt0=1e-3,
                          y0=y0, args=args, saveat=saveat, stepsize_controller=stepsize_controller,
                          max_steps=N_t, throw=False, progress_meter=diffrax.TqdmProgressMeter())

# %%
assert sol.ys is not None
psi_concat_t = sol.ys[:, :2*N_terms] + 1j * sol.ys[:, 2*N_terms:]
psi_t, psi_dot_t = psi_concat_t[:, :N_terms], psi_concat_t[:, N_terms:]
plt.plot(saveat_t, sol.ys[:, :N_terms], lw=3, label=[
         "$\\psi_{}$".format(j) for j in range(N_terms)])
# plt.plot(saveat_t, psi_dot_t)
plt.xscale('log')
plt.title('Arbitrary solution for {} $\\psi_\\alpha$ terms'.format(N_terms))
plt.xlabel('$t$')
plt.legend()
plt.show()

# %%


def solve(psi_dot_0, *args, **solver_kwargs):
    psi_0, solver, term, solver_args = args
    N_terms = len(psi_0)
    y0 = jnp.concat((psi_0, psi_dot_0, jnp.zeros(N_terms)))
    sol = diffrax.diffeqsolve(
        term, solver, t0=0., t1=1.,
        y0=y0, throw=False, args=solver_args, **solver_kwargs)
    return jax.lax.cond(sol.result._value == 0, lambda: sol.ys, lambda: jnp.inf * sol.ys)


# %%
def residual(psi_dot_0, psi_1, *args, **solver_kwargs):
    N_terms = len(psi_1)
    pred_y1_concat = solve(psi_dot_0, *args, **solver_kwargs)[-1]
    real_res = psi_1 - pred_y1_concat[:N_terms]
    im_res = pred_y1_concat[2*N_terms:3*N_terms]
    return jnp.concat((real_res, im_res))


# %%
N_terms = 5
z = jnp.zeros(N_terms)
psi_0 = z.at[0].set(1.)
psi_1 = z.at[-1].set(1.)
# .at[jnp.array([0,-1])].set(jnp.array([-1,1]))
initial_psi_dot_0 = jnp.concat((z, z))
solver_kwargs = {
    'adjoint': diffrax.DirectAdjoint(),
    'max_steps': 5000,
    'stepsize_controller': diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-8),
    'dt0': 1e-3
}
solve_args = (psi_0, diffrax.Kvaerno5(), term, args)

# %%


@jax.jit
def residual_fcn(psi_dot_0, _):
    return residual(psi_dot_0, psi_1, *solve_args, **solver_kwargs)


# %%
residual(initial_psi_dot_0, psi_1, *solve_args, **solver_kwargs)

# %%
solver = optx.BestSoFarLeastSquares(optx.LevenbergMarquardt(
    rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})
))

# %%
sol = optx.least_squares(
    residual_fcn,
    solver,
    initial_psi_dot_0,
    # max_steps=17,
    throw=False
)

# %%
sol.result, solver.norm(residual_fcn(sol.value, None)).item()

# %%
ode_sol = solve(sol.value, *solve_args, saveat=saveat, **solver_kwargs)
y1_concat = ode_sol[-1]
psi_1_concat = y1_concat[:2*N_terms] + 1j*y1_concat[2*N_terms:]
psi_1 = psi_1_concat[:N_terms]
psi_1

# %%
plt.plot(saveat_t, ode_sol[:, :N_terms], lw=3, label=[
         "$\\psi_{}$".format(j) for j in range(N_terms)])
# plt.plot(saveat_t, psi_dot_t)
# plt.xscale('log')
plt.title('Approx optimal soln for {} $\\psi_\\alpha$ terms'.format(N_terms))
plt.xlabel('$t$')
plt.legend()
plt.show()

# %%


def eval_velocity(x: Float, psi: Float[Array, "N"], psi_dot: Float[Array, "N"], mu_Z: Float[Array, "N"], Sigma_Z: Float[Array, "N N"]):
    N = psi.shape[0]
    assert psi_dot.shape[0] == N and mu_Z.shape[0] == N and Sigma_Z.shape == (
        N, N)
    sig_psi = Sigma_Z @ psi
    dot_prod = jnp.dot(psi, sig_psi)
    bias = (x - jnp.dot(mu_Z, psi))
    shift = (sig_psi * bias) / dot_prod
    return psi_dot @ (mu_Z + shift)


def eval_vel_fcn(x, y_t: Float[Array, "4*N"], mu_Z: Float[Array, "N"], Sigma_Z: Float[Array, "N N"]):
    N = len(mu_Z)
    psi_concat = y_t[:2*N] + 1j * y_t[2*N:]
    psi, psi_dot = psi_concat[:N], psi_concat[N:]
    return eval_velocity(x, psi, psi_dot, mu_Z, Sigma_Z)


# %%
mu_Z = jnp.linspace(0, phi.mu, N_terms)
Sigma_Z = jnp.diag((1 - mu_Z)**2 + (mu_Z**2)*phi.sigma**2)
mu_Z, Sigma_Z

# %%
eval_vel_fcn(1., ode_sol[-1], mu_Z, Sigma_Z)

# %%
velocity_vmap = jax.vmap(
    jax.vmap(
        lambda x, y_t: eval_vel_fcn(x, y_t, mu_Z, Sigma_Z),
        in_axes=(0, None),
    ), in_axes=(None, 0)
)

# %%
velocity_eval = velocity_vmap(jnp.linspace(-5, 5), ode_sol)

# %%
velocity_eval.shape

# %%
plt.plot(jnp.cumsum(jnp.real(velocity_eval)[:, velocity_eval.shape[1]//2])/N_t)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(jnp.real(velocity_eval).T, aspect=0.1, extent=(0, 1, -5, 5))
ax.set_xlabel("t")
ax.set_ylabel("x")
plt.show()
