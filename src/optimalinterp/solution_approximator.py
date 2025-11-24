import jax
import jax.numpy as jnp
import chex
from abc import ABC, abstractmethod
from jaxtyping import Float, Array, PyTree
from .basis import LinearBasis, EvalPointsT
from typing import Tuple

SolutionT = Float[Array, 'time alpha']
CoeffsT = Float[Array, 'shap alpha']
SolutionDiffsT = Tuple[SolutionT, SolutionT, SolutionT]


class SolutionApproximator(ABC):
    @abstractmethod
    def evaluate_all(self, params: PyTree) -> SolutionDiffsT:
        pass


@jax.jit
def _linear_solution_evaluator(basis_eval: SolutionT, basis_diff: SolutionT, basis_diff2: SolutionT, params: CoeffsT):
    diff0 = basis_eval @ params
    diff1 = basis_diff @ params
    diff2 = basis_diff2 @ params
    return diff0, diff1, diff2


class LinearSolutionApproximator(SolutionApproximator):
    basis: LinearBasis
    N_terms: int
    basis_eval: jnp.ndarray
    basis_diff: jnp.ndarray
    basis_diff2: jnp.ndarray
    eval_pts: jnp.ndarray

    def __init__(self, basis: LinearBasis, evaluation_points: EvalPointsT, N_terms: int):
        self.basis, self.eval_pts, self.N_terms = basis, evaluation_points, N_terms
        basis_evals = basis.evaluate_basis_diff2(evaluation_points)
        self.basis_eval, self.basis_diff, self.basis_diff2 = basis_evals

    @chex.chexify
    def evaluate_all(self, params: CoeffsT):
        chex.assert_shape(params, (self.basis.N_shap, self.N_terms))
        return _linear_solution_evaluator(self.basis_eval, self.basis_diff, self.basis_diff2)
