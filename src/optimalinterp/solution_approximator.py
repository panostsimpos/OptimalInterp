# import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod
from .basis import LinearBasis
from typing import Tuple


class SolutionApproximator(ABC):
    @abstractmethod
    def evaluate_all(self, params) -> Tuple[jax.Array, jax.Array, jax.Array]:
        pass


class LinearSolutionApproximator(SolutionApproximator):
    basis: LinearBasis
    N_terms: int
    basis_eval: jax.Array
    basis_diff: jax.Array
    basis_diff2: jax.Array
    eval_pts: jax.Array

    def __init__(self, basis: LinearBasis, evaluation_points: jax.Array, N_terms: int):
        self.basis, self.eval_pts, self.N_terms = basis, evaluation_points, N_terms
        basis_evals = basis.evaluate_basis_diff2(evaluation_points)
        self.basis_eval, self.basis_diff, self.basis_diff2 = basis_evals

    def evaluate_all(self, params: jax.Array):
        assert params.shape == (self.basis.N_shap, self.N_terms)
        diff0 = self.basis_eval @ params
        diff1 = self.basis_diff @ params
        diff2 = self.basis_diff2 @ params
        return diff0, diff1, diff2
