from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Type

__all__ = ['SplineBasis']


class LinearBasis(ABC):
    @property
    @abstractmethod
    def N_shap(self) -> int:
        pass

    @abstractmethod
    def evaluate_basis(self, points: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def evaluate_basis_diff(self, points: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def evaluate_basis_diff2(self, points: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        pass


class Spline(ABC):
    @abstractmethod
    def evaluate(self, points: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def evaluate_diff(self, points: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def evaluate_diff2(self, points: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        pass


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
        eval = self.evaluate(points)
        cosx = jnp.cos(jnp.pi * points)
        diff = (cosx - eval) / points
        diff = jnp.where(points == 0., 0., diff)
        diff2 = -(2 * diff / points + jnp.pi*jnp.pi*eval)
        diff2 = jnp.where(points == 0., -jnp.pi * jnp.pi / 3, diff2)
        return eval, diff, diff2


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
    'sinc': SincSpline,
    'hat': HatSpline
}


class SplineBasis(LinearBasis):
    N_knots: int
    knots: jax.Array
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

    def global_to_local(self, points: jax.Array) -> jax.Array:
        return points[:, jnp.newaxis]*(self.N_knots-1) - self.knots

    def evaluate_basis(self, points):
        "returns eval.shape = (N_points, N_knots)"
        return self.spline.evaluate(self.global_to_local(points))

    def evaluate_basis_diff(self, points):
        local_points = self.global_to_local(points)
        evals, diff = self.spline.evaluate_diff(local_points)
        return evals, diff*(self.N_knots - 1)

    def evaluate_basis_diff2(self, points):
        scale = self.N_knots - 1
        local_points = self.global_to_local(points)
        evals, diff, diff2 = self.spline.evaluate_diff2(local_points)
        return evals, diff*scale, diff2*(scale**2)
