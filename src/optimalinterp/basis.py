from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple, Dict, Type, Callable
from types import ModuleType
import warnings
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import optimistix as optx
from .moment_generator import MomentGeneratingPhi
from .ode_residual import OptimalInterpPDEResidual
from . import chebyshev

__all__ = ["SplineBasis", "LinearBasis"]

EvalPointsT = Float[Array, "time"]
BasisEvalT = Float[Array, "time alpha"]
BasisTensT = Float[Array, "*size"]


class AbstractLinearBasis(ABC):
    @property
    @abstractmethod
    def N_shap(self) -> int:
        r"""
        Number of shape functions in the basis
        """
        pass

    @abstractmethod
    def evaluate_basis(self, points: EvalPointsT) -> BasisEvalT:
        r"""
        Evaluate a linear basis

        :param points: Vector of points to evaluate on
        :type points: Float[Array, "T"]
        :return: Evaluation of collection of basis functions
        :rtype: Float[Array, "T N_shap"]
        """
        pass

    @abstractmethod
    def evaluate_basis_diff(self, points: EvalPointsT) -> Tuple[BasisEvalT, BasisEvalT]:
        r"""
        Evaluate a linear basis and one derivative

        :param points: Vector of points to evaluate on
        :type points: Float[Array, "T"]
        :return: Evaluation of collection of basis functions and their derivative
        :rtype: Tuple[Float[Array, "T N_shap"],Float[Array, "T N_shap"]]
        """
        pass

    @abstractmethod
    def evaluate_basis_diff2(self, points: EvalPointsT) -> Tuple[BasisEvalT, BasisEvalT, BasisEvalT]:
        r"""
        Evaluate a linear basis and two derivatives

        :param points: Vector of points to evaluate on
        :type points: Float[Array, "T"]
        :return: Evaluation of collection of basis functions and two derivatives
        :rtype: Tuple[Float[Array, "T N_shap"],Float[Array, "T N_shap"],Float[Array, "T N_shap"]]
        """
        pass


class Spline(ABC):
    r"""
    Spline $f$ must satisfy $f(j) = \delta_{0j}$ for any integer $j \in \mathbb{Z}$
    """
    @abstractmethod
    def evaluate(self, points: BasisTensT) -> BasisTensT:
        pass

    @abstractmethod
    def evaluate_diff(self, points: BasisTensT) -> Tuple[BasisTensT, BasisTensT]:
        pass

    @abstractmethod
    def evaluate_diff2(self, points: BasisTensT) -> Tuple[BasisTensT, BasisTensT, BasisTensT]:
        pass


@jax.jit
def _eval_sinc_spline_diff2(points: BasisTensT):
    eval = jnp.sinc(points)
    cosx = jnp.cos(jnp.pi * points)
    diff = (cosx - eval) / points
    diff = jnp.where(points == 0., 0., diff)
    diff2 = -(2 * diff / points + jnp.pi*jnp.pi*eval)
    diff2 = jnp.where(points == 0., -jnp.pi * jnp.pi / 3, diff2)
    return eval, diff, diff2


class SincSpline(Spline):
    r"Sinc spline function"
    def __eq__(self, other):
        return isinstance(other, SincSpline)

    def evaluate(self, points):
        return jnp.sinc(points)

    def evaluate_diff(self, points):
        eval = self.evaluate(points)
        cosx = jnp.cos(jnp.pi * points)
        diff = (cosx - eval) / points
        diff = jnp.where(points == 0., 0., diff)
        return eval, diff

    def evaluate_diff2(self, points):
        return _eval_sinc_spline_diff2(points)


class HatSpline(Spline):
    r"Hat Spline function"
    def __eq__(self, other):
        return isinstance(other, HatSpline)

    def evaluate(self, points):
        return jnp.where(jnp.abs(points) < 1, 1 - jnp.abs(points), 0.)

    def evaluate_diff(self, points):
        diff = jnp.where(jnp.abs(points + 0.5) < 0.5, 1, 0)
        diff = (jnp.abs(points + 0.5) < 0.5).astype(float) - \
               (jnp.abs(points - 0.5) < 0.5).astype(float)
        return self.evaluate(points), diff

    def evaluate_diff2(self, points):
        return (*self.evaluate_diff(points), jnp.zeros_like(points))


SPLINES: Dict[str, Type] = {
    "sinc": SincSpline,
    "hat": HatSpline
}


@jax.jit
def _global_to_local(N_knots: int, knots: Float[Array, " knots"], points: Float[Array, " time"]):
    return points[:, jnp.newaxis]*(N_knots-1) - knots


class SplineBasis(AbstractLinearBasis):
    """
    Linear basis of splines

    :var N_knots: Number of knots in the basis
    :vartype N_knots: int
    :var knots: Array of knots for the basis
    :vartype knots: Float[Array, "N_knots"]
    :var spline: Spline used
    :vartype spline: Spline
    """
    N_knots: int
    knots: jnp.ndarray
    spline: Spline

    def __init__(self, N_knots, spline: Spline | str):
        r"""
        Use a master spline (e.g., sinc, piecewise polynomial, etc.) that satisfies $f(j) = \delta_{0j}$
        N_knots includes endpoints: must be >= 2
        evaluate(x) -> f(x)
        evaluate_diff(x) -> (f(x), f'(x))
        evaluate_diff2(x) -> (f(x), f'(x), f''(x))
        """
        if N_knots < 2:
            raise ValueError(f"Expected N_knots >= 2. Got N_knots={N_knots}")
        self.N_knots = N_knots
        if isinstance(spline, Spline):
            self.spline = spline
        else:
            self.spline = SPLINES[spline]()
        self.knots = jnp.arange(self.N_knots)

    def __eq__(self, other):
        return isinstance(other, SplineBasis) and (self.N_knots == other.N_knots) and (self.spline == other.spline)

    @property
    def N_shap(self):
        return self.N_knots

    def evaluate_basis(self, points):
        "returns eval.shape = (N_points, N_knots)"
        return self.spline.evaluate(_global_to_local(self.N_knots, self.knots, points))

    def evaluate_basis_diff(self, points):
        local_points = _global_to_local(self.N_knots, self.knots, points)
        evals, diff = self.spline.evaluate_diff(local_points)
        return evals, diff*(self.N_knots - 1)

    def evaluate_basis_diff2(self, points):
        scale = self.N_knots - 1
        local_points = _global_to_local(self.N_knots, self.knots, points)
        evals, diff, diff2 = self.spline.evaluate_diff2(local_points)
        return evals, diff*scale, diff2*(scale**2)

    def collocated_basis_transform(self):
        evals, diff, diff2 = self.evaluate_basis_diff2(
            self.knots/(self.N_knots - 1))
        return evals, diff, diff2


BASES: Dict[str, ModuleType] = {'chebyshev': chebyshev}
BASES_INTERVAL: Dict[str, tuple[Float, Float]] = {'chebyshev': (-1, 1)}

BasisEvalFcn = Callable[[Float[Array, " N"], int], BasisEvalT]
BasisDiff1Fcn = Callable[[Float[Array, " N"], int],
                         Tuple[BasisEvalT, BasisEvalT]]
BasisDiff2Fcn = Callable[[Float[Array, " N"], int],
                         Tuple[BasisEvalT, BasisEvalT, BasisEvalT]]


class LinearBasis(AbstractLinearBasis):
    r"""
    Simple and general implementation of a linear basis
    """
    max_order: int
    eval: BasisEvalFcn
    diff1: BasisDiff1Fcn
    diff2: BasisDiff2Fcn
    lo: float
    hi: float

    def __init__(
            self, max_order: int, eval_or_module: ModuleType | BasisEvalFcn | str,
            diff1: BasisDiff1Fcn | None = None, diff2: BasisDiff2Fcn | None = None,
            original_interval: tuple[float, float] = (0., 1.)
    ):
        """
        Use a general linear basis for approximation. ASSUMES THAT ANY INPUT IS IN (0,1), I.E., OPTIMAL INTERPOLANT SETUP

        :param max_order: Order of the basis
        :type max_order: int
        :param eval_or_module: Evaluation function or name of module/class implementing eval/diff1/diff2
        :type eval_or_module: ModuleType | BasisEvalFcn | str
        :param diff1: Eval/derivative function
        :type diff1: BasisDiff1Fcn | None
        :param diff2: Eval/derivative/2 derivative
        :type diff2: BasisDiff2Fcn | None
        :param original_interval: intervale that the basis works on (e.g., (-1,1) for Chebyshev)
        :type original_interval: tuple[float, float]
        """
        if max_order < 1:
            raise ValueError(
                f'Expected max_order >= 1. Got max_order = {max_order}')
        self.max_order = max_order
        if isinstance(eval_or_module, ModuleType | str):
            if isinstance(eval_or_module, str):
                mod = BASES[eval_or_module]
            else:
                mod = eval_or_module
            basis_spec = mod.__spec__
            assert basis_spec is not None
            basis_name = basis_spec.name.split('.')[-1]
            self.lo, self.hi = BASES_INTERVAL.get(
                basis_name, original_interval
            )
            self.eval = mod.eval
            self.diff1 = mod.diff1
            self.diff2 = mod.diff2
        else:
            assert diff1 is not None and diff2 is not None
            self.eval = eval_or_module
            self.diff1 = diff1
            self.diff2 = diff2

    def __eq__(self, other):
        return isinstance(other, LinearBasis) and \
            (self.max_order == other.max_order) and \
            (self.eval == other.eval) and \
            (self.diff1 == other.diff1) and \
            (self.diff2 == other.diff2)

    @property
    def N_shap(self):
        return self.max_order + 1

    def evaluate_basis(self, points):
        return self.eval(points*(self.hi - self.lo) + self.lo, self.max_order)

    def evaluate_basis_diff(self, points):
        pts01 = points*(self.hi - self.lo) + self.lo
        eval, diff = self.diff1(pts01, self.max_order)
        diff = diff * (self.hi - self.lo)
        return eval, diff

    def evaluate_basis_diff2(self, points):
        pts01 = points*(self.hi - self.lo) + self.lo
        eval, diff1, diff2 = self.diff2(pts01, self.max_order)
        diff1 = diff1 * (self.hi - self.lo)
        diff2 = diff2 * ((self.hi - self.lo)**2)
        return eval, diff1, diff2


def create_spline_residual(psi: SplineBasis, phi: MomentGeneratingPhi):
    """
    Create a residual function for a given spline basis

    :param psi: Basis of spline functions
    :type psi: SplineBasis
    :param phi: Moment generating function
    :type phi: MomentGeneratingPhi
    """
    _, basis_diff1, basis_diff2 = psi.collocated_basis_transform()

    def spline_residual(coeffs_psi, _):
        # Assume a basis with T+2 knots (0,...,T+1), where t_0 = 0 and t_{T+1} = 1.
        # Each col of coeffs_psi represents the coefficients for a given output.
        # Each row of coeffs_psi represents a given spline function, i.e., col k represents spline centered at k/(T+1).
        # dt_scale is the derivative operator on the coeffs_psi for this basis evaluated at the knots.
        # dt_shift is the shift of the derivative operator to ensure boundary conditions are satisfied.
        coeffs_psi_full = jnp.pad(
            coeffs_psi, ((1, 1), (1, 1)), mode='constant'
        ).at[[0, -1], [0, -1]].set(1.)
        psi = coeffs_psi_full  # Note that the splines are just the coeffs at collocation points
        psi_dot = basis_diff1 @ coeffs_psi_full
        psi_dot_dot = basis_diff2 @ coeffs_psi_full
        residual = OptimalInterpPDEResidual(
            psi, psi_dot, psi_dot_dot, phi
        )
        return jnp.abs(residual)
    return jax.jit(spline_residual)


def pad_coeffs(coeffs: Float[Array, "p-2 alpha"], psi: AbstractLinearBasis, fixed_orders: tuple[int, int]):
    """
    Pad the coefficients for a linear basis to ensure function satisfies boundary conditions

    :param coeffs: Non-fixed coefficients
    :type coeffs: Float[Array, "p-2 alpha"]
    :param psi: Basis for coefficients
    :type psi: AbstractLinearBasis
    :param fixed_orders: Indices of coefficients are constrained
    :type fixed_orders: tuple[int, int]
    """
    N_terms = coeffs.shape[1]
    basis_eval = psi.evaluate_basis(jnp.zeros(2).at[1].set(1.))
    order0, order1 = fixed_orders
    bc_scale = basis_eval[[0, 0, -1, -1], [order0, order1, order0, order1]].reshape(2,2)
    inv_bc_scale = jnp.linalg.inv(bc_scale)
    bc_shift = jnp.zeros((2, N_terms)).at[[0, 1], [0, -1]].set(1.)
    first_coeffs = inv_bc_scale @ (
        bc_shift - (basis_eval[:, 2:] @ coeffs)
    )
    return jnp.concat((first_coeffs, coeffs))


def create_collocated_basis_residual(
        t_grid: Float[Array, " T"],
        N_terms: int, psi: AbstractLinearBasis,
        phi: MomentGeneratingPhi,
        fixed_orders: tuple[int, int],
        wts: float | Float[Array, " T"],
        residual_real_fcn: Callable[[Array], Array],
    ):
    """
    Create a residual using collocation. Assume that the first two elements of the linear basis are fixed to ensure boundary conditions.

    :param t_grid: Grid of points to collocate
    :type t_grid: Float[Array, "T"]
    :param N_terms: Number of expansion terms
    :type N_terms: int
    :param psi: Basis for expansion terms
    :type psi: AbstractLinearBasis
    :param phi: Moment generation function
    :type phi: MomentGeneratingPhi
    :param fixed_orders: Indices of coefficients constrained by boundary conditions
    :type fixed_orders: tuple[int, int]
    :param wts: Optional weights for collocations
    :type wts: float | Float[Array, " T"]
    :param residual_real_fcn: Function mapping the complex residual to the real plane
    :type residual_real_fcn: Callable[[Array], Array]
    """
    t_grid = jnp.sort(t_grid)
    assert t_grid[0] == 0. and t_grid[-1] == 1.  # Ensure grid is valid
    # Evaluate basis
    basis_eval, basis_diff1, basis_diff2 = psi.evaluate_basis_diff2(t_grid)
    # Get the zero and first order basis at times t=0, t=1
    order0, order1 = fixed_orders
    assert order0 < order1
    bc_scale = basis_eval[[0, 0, -1, -1], [order0, order1, order0, order1]].reshape(2,2)
    # Set the boundary conditions
    inv_bc_scale = jnp.linalg.inv(bc_scale)
    bc_shift = jnp.zeros((2, N_terms)).at[[0, 1], [0, -1]].set(1.)
    wts = jnp.reshape(wts, (-1,1))

    def basis_residual(coeffs_psi: Float[Array, "P alpha"], _):
        # Get first two basis elements using boundary conditions
        bdry_basis = basis_eval[jnp.array([0,-1])]
        bdry_transform = jnp.concat(
            (bdry_basis[:,:order0], bdry_basis[:,order0+1:order1], bdry_basis[:,order1+1:]),
            axis=1
        )
        first_coeffs = inv_bc_scale @ (
            bc_shift - (bdry_transform @ coeffs_psi)
        )
        # Evaluate the basis and derivatives on full coefficient set
        full_coeffs_psi = jnp.concat((first_coeffs, coeffs_psi))
        psi = basis_eval @ full_coeffs_psi
        psi_diff1 = basis_diff1 @ full_coeffs_psi
        psi_diff2 = basis_diff2 @ full_coeffs_psi
        # Evaluate the residual
        residual = OptimalInterpPDEResidual(
            psi, psi_diff1, psi_diff2, phi
        )
        return wts * residual_real_fcn(residual)

    return jax.jit(basis_residual)

def are_uniform_points(t: Float[Array, " T"]) -> bool:
    return (jnp.square((t - jnp.linspace(0, 1, t.shape[0]))).sum() < 1e-10).item()

class OptimalInterpBVPBasisSolution(NamedTuple):
    """Solution container for optimal interpolation BVP.

    Attributes:
        psi_basis: Basis for the modal functions ψ of shape (N_shap,)
        coeffs_psi: Coefficients for the basis of shape (N_shap, N)
        residual_norm: Final BVP residual ∑w_k R_{t_k}[ψ]^2, where R_{t_k} is residual functional at time t_k
        success: Whether basis optimization converged
    """
    psi_basis: AbstractLinearBasis
    coeffs_psi: Float[Array, "N_shap N"]
    residual_norm: float
    success: bool

def process_grid(t_points: Float[Array, " T"] | int, t_weights: Optional[Float[Array, " T"]]):
    if t_weights is None:
        t_weights = jnp.ones(())
    else:
        # Take sqrt since Levenberg--Marquadt will square each term in residual
        t_weights = jnp.sqrt(t_weights)

    if isinstance(t_points, int):
        t_points = jnp.linspace(0., 1., t_points)
    else:
        grid_perm = jnp.argsort(t_points)
        t_points = t_points[grid_perm]

        t_weights = t_weights[grid_perm]
        t_points = (t_points - t_points[0])/(t_points[-1] - t_points[0])
    return t_points,t_weights

def solve(
    Phi: MomentGeneratingPhi,
    N_terms: int,
    psi_basis_order: int = 20,
    psi_basis: AbstractLinearBasis | str = 'chebyshev',
    t_points: Float[Array, " T"] | int = 150,
    t_weights: Optional[Float[Array, " T"]] = None,
    fixed_orders: tuple[int, int] = (0,1),
    verbose: bool = True,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    max_optimizer_steps: int = 1000,
    residual_real_fcn: Callable[[Array], Array] = jnp.real,
    optimizer: Optional[optx.AbstractLeastSquaresSolver] = None
):

    t_points, t_weights = process_grid(t_points, t_weights)
    print(
          t_points.shape, t_weights.shape, t_points.min(), t_points.max(), jnp.square(t_weights).sum()
    )

    if isinstance(psi_basis, str):
        psi_basis = LinearBasis(psi_basis_order, psi_basis)
    elif isinstance(psi_basis, SplineBasis):
        if not are_uniform_points(t_points):
            raise ValueError("Expected uniform t_points for SplineBasis")
        warnings.warn("SplineBasis not tested yet")

    # Obtain Phi and solve using shooting method
    if isinstance(psi_basis, SplineBasis):
        residual_fcn = create_spline_residual(psi_basis, Phi)
    else:
        residual_fcn = create_collocated_basis_residual(
            t_points, N_terms, psi_basis, Phi, fixed_orders, t_weights, residual_real_fcn
        )

    verbose_set = frozenset({"step", "accepted", "loss", "step_size"}) if verbose else frozenset()
    if optimizer is None:
        optimizer = optx.LevenbergMarquardt(
            rtol=rtol, atol=atol, verbose=verbose_set
        )
    y0 = jnp.zeros((psi_basis.N_shap - 2, N_terms))
    residual_fcn(y0, None)
    opt_sol = optx.least_squares(residual_fcn, optimizer, y0, throw=False, max_steps=max_optimizer_steps)
    coeffs_psi = pad_coeffs(opt_sol.value, psi_basis, fixed_orders)
    success = (opt_sol.result == optx.RESULTS.successful).item()
    residual_norm = optimizer.norm(residual_fcn(opt_sol.value, None)).item()
    bvp_soln = OptimalInterpBVPBasisSolution(psi_basis, coeffs_psi, residual_norm, success)
    return bvp_soln

