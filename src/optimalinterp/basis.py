from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Dict, Type, Callable
from types import ModuleType
from . import chebyshev

__all__ = ["SplineBasis", "LinearBasis"]

EvalPointsT = Float[Array, "time"]
BasisEvalT = Float[Array, "time alpha"]
BasisTensT = Float[Array, "*size"]


class AbstractLinearBasis(ABC):
    @property
    @abstractmethod
    def N_shap(self) -> int:
        pass

    @abstractmethod
    def evaluate_basis(self, points: EvalPointsT) -> BasisEvalT:
        pass

    @abstractmethod
    def evaluate_basis_diff(self, points: EvalPointsT) -> Tuple[BasisEvalT, BasisEvalT]:
        pass

    @abstractmethod
    def evaluate_basis_diff2(self, points: EvalPointsT) -> Tuple[BasisEvalT, BasisEvalT, BasisEvalT]:
        pass


class Spline(ABC):
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
    N_knots: int
    knots: jnp.ndarray
    spline: Spline

    def __init__(self, N_knots, spline: Spline | str):
        r"""
        Use a master spline (e.g., sinc, piecewise polynomial, etc) that satisfies $f(j) = \delta_{0j}$
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
        # Return [A]_{ij} = b'_j(t_i)
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
    max_order: int
    eval: BasisEvalFcn
    diff1: BasisDiff1Fcn
    diff2: BasisDiff2Fcn
    lo: float = 0.
    hi: float = 1.

    def __init__(
            self, max_order: int, eval_or_module: ModuleType | BasisEvalFcn | str,
            diff1: BasisDiff1Fcn | None = None, diff2: BasisDiff2Fcn | None = None,
            original_interval: tuple = (0., 1.)
    ):
        r"""
        Use a general linear basis for approximation. ASSUMES THAT ANY INPUT IS IN (0,1), I.E., OPTIMAL INTERPOLANT SETUP
        """
        if max_order < 1:
            raise ValueError(
                f'Expected max_order >= 1. Got max_order = {max_order}')
        self.max_order = max_order
        if isinstance(eval_or_module, ModuleType | str):
            mod = BASES[eval_or_module] if isinstance(
                eval_or_module, str) else eval_or_module
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
        "returns eval.shape = (N_points, N_shap)"
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
