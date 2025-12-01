from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Dict, Type

__all__ = ["SplineBasis"]

EvalPointsT = Float[Array, "time"]
BasisEvalT = Float[Array, "time alpha"]
BasisTensT = Float[Array, "*size"]


class LinearBasis(ABC):
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


class SplineBasis(LinearBasis):
    N_knots: int
    knots: jnp.ndarray
    spline: Spline

    def __init__(self, N_knots, spline: Spline | str):
        r"""
        SplineBasis(N_knots, eval, eval_diff, eval_diff2)
        Use a master spline (e.g., sinc, piecewise polynomial, etc) that satisfies $f(j) = \delta_{0j}$
        N_knots includes endpoints: must be >= 2
        evaluate(x) -> f(x)
        evaluate_diff(x) -> (f(x), f'(x))
        evaluate_diff2(x) -> (f(x), f'(x), f''(x))
        """
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
