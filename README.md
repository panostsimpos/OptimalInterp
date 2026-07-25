# Optimal Interpolants

This repository was built in collaboration with Daniel Sharp, a fellow PhD student at the Uncertainty Quantification Group at MIT.

We created this repository to explore certain ideas that emerged from my [workshop paper](https://arxiv.org/pdf/2510.11657) and led us to write [this paper together.](https://arxiv.org/abs/2604.15439)
None of the content of the repository ended up in the paper. 
However, going through this gave us essential intuition.

A lot of the experiments failed and this is precisely what led us to conjecture and prove the impossibility theorems in [our paper](https://arxiv.org/abs/2604.15439).
Nonetheless, some of the mathematical ideas and computational abstractions developed here could be quite useful in understanding the **Optimal Design** question for stochastic interpolants, discussed below. 
Due to these considerations we decided to make our code public.

This README contains the following:

1. Installation instructions \& quickstart.
2. Mathematical framework.
3. Description of experiments and algorithmic abstractions.

Installation instructions
---
This project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository first
````bash
 git clone https://github.com/panostsimpos/OptimalInterp.git
 cd OptimalInterp
````
Install Python and create the project environment
````bash
uv python install 3.12
uv sync --locked --python 3.12
````
Verify the installation by 
````bash
uv run pytest
````
All tests should pass.

The project directory structure should roughly be:

```text
OptimalInterp/
├── README.md
├── DERIVATIONS.md
├── LICENSE
├── pyproject.toml
├── .python-version
├── uv.lock
├── src/
│   └── optimalinterp/
└── test/
└── notebooks/
```

Tests are in `test/`, implementations are in `src/` and experiment notebooks are in `notebooks/`.

Quickstart
---
A quick introduction that uses the same features of the repository can be found in 
[`notebooks/gaussian_example_shooting.ipynb`](notebooks/gaussian_example_shooting.ipynb).
The notebook attempts to compute the optimal (straight-line) modal interpolant that connects two univariate Gaussian distributions by using a Gaussian stochastic basis.


Mathematical framework
---
The [Stochastic Interpolants paper](https://arxiv.org/pdf/2303.08797) proposes an elegant framework for generative modelling. 
<!-- Namely, generative models are viewed as probability measure trajectories $`t \mapsto P_t`$ and each $`P_t \in \mathcal{P}(\mathbb{R}^d)`$ that are induced by  -->
Given a data (target) distribution $`P_1`$ and a source (noise) distribution $`P_0`$ draw $`(X_0, X_1) \sim P_0 \otimes P_1`$
choose a nice function $`I_t: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d`$
and set

```math
    X_t = I_t(X_0, X_1) .
```

Then, compute the conditional expectation

```math
    v_t(X_t) = \mathbb{E} \left[ \dot X_t \mid X_t \right]
```

solve the ODE

```math
\begin{cases}
    \dot \phi_t(x) = v_t(\phi_t(x)) \\
    \phi_0(x) = x
\end{cases}
```

and sample (generate) from $`P_1`$ by drawing $`X_0 \sim P_0`$ and setting

```math
    X_1 = \phi_1(X_0) .
```

This powerful framework encompasses many known generative models, e.g., diffusion models, but it also describes, at least in principle, *many new* generative models. Indeed, the known generative models correspond to "relatively straightforward" choices of $`I_t`$ e.g.

```math
    X_t = \alpha_t X_0 + \beta_t X_1
```

for scalar functions $`t \mapsto \alpha_t`$ and $`t \mapsto \beta_t`$.

So we wondered: are there *better* generative models, described by exotic interpolant functions $`I_t`$? What does better even mean, in this context?

**Optimal Design Question**
1. What makes some stochastic interpolants better than others? Find an objective $`\cal{O}`$ that assigns a cost $`\mathcal{O}(I)`$ to a stochastic interpolant $`I`$.
2. Solve the optimization problem $`\textup{min}_{I} \mathcal{O}(I)`$ .

Key Mathematical Insight
---

A perspective I took in this [workshop paper](https://arxiv.org/pdf/2510.11657) was to *assert* that a better interpolant is an interpolant that leads to an easily integrable $`v_t`$.
By elementary considerations, the easiest field $`v`$ to integrate is a field which is constant along the flow:

```math
    \frac{d}{dt} v_t(\phi_t(x)) = 0 \iff \partial_t v_t(x) + (v_t \cdot \nabla) v_t(x) \equiv 0
```

and the latter equation needs to hold for all $`x \in \mathbb{R}^d`$, assuming surjectivity of the flow.
This PDE on the right is sometimes also referred to as the (inviscid) Burgers' equation.
We can now pose the following problem:

**Problem**: Given source and target measures $`P_0, P_1`$ find an interpolant $`I_t`$ such that with $`X_t = I_t(X_0, X_1)`$ and $`(X_0, X_1) \sim P_0 \otimes P_1`$ the conditional velocity $`v_t`$ defined above satisfies the Burgers' equation

```math
    \partial_t v_t + (v_t \cdot \nabla) v_t(x) \equiv 0 .
```



Now the main result of the [workshop paper](https://arxiv.org/pdf/2510.11657) is that we can transform the above PDE to a nicer PDE. To wit, with

```math
    a_t(x) = \mathbb{E} \left[ \ddot X_t \mid X_t = x \right] 
    \quad \textup{and} \quad
    \Pi_t(x) = \textup{Cov} \left( \dot X_t \mid X_t = x  \right)
```

the Burgers' equation is equivalent to the PDE

```math
    \nabla \cdot \left( \rho_t \, \Pi_t \right) = \rho_t \, a_t ,
```

and $`\rho_t`$ is the density of $`X_t`$.

Key computational question
---
We want to solve the **Problem** described in the above questions.
Solving PDEs numerically is computationally challenging enough and yet our problem is even more involved. Indeed, we are not searching for a function that satisfies a certain differential equation; we are searching for interpolants $`I_t`$ whose path statistics $`\rho_t`$, $`a_t`$ and $`\Pi_t`$, functions over space-time, satisfy a differential equation. To make progress, we assert an ansatz. Suppose we can write

```math
    X_t = \sum_\alpha \psi_{\alpha}(t) Z_\alpha
```

for fixed, $`\mathbb{R}^d`$ valued random variables $`\{ Z_{\alpha} \}_\alpha `$, multi-indices $`\alpha \in \mathcal{A}`$ and real-valued functions $`\psi_\alpha : [0,1] \to \mathbb{R}`$.
We call this ansatz a **modal interpolant**.

A key mathematical result shown in [DERIVATIONS.md](DERIVATIONS.md) shows that under this ansatz we have

```math
    \nabla \cdot \left( \rho_t \, \Pi_t \right) = \rho_t \, a_t , \iff \mathbf{D} \, \ddot \psi(t) = \dot \psi(t)^\top \, \mathbf{C} \, \dot \psi(t) 
```

and $`\mathbf{D} \, , \, \mathbf{C}`$ are tensors of order $`3`$ and $`4`$, respectively, defined in terms of the characteristic functions

```math
    \Phi_\alpha(\xi) = \mathbb{E} \left[ e^{i \xi Z_\alpha} \right] .
```

In index notation we can re-write the above equation as 

```math
    \sum_{\alpha \in \mathbb{Z}^d} D^k_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{(\alpha, \gamma) \in \mathbb{Z}^d \times \mathbb{Z}^d} \dot \psi_\alpha(t) \, C^k_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, ,
    % \quad \textup{for all} \quad k \in \{1, \ldots, d\} \quad \textup{and} \quad \beta \in \mathbb{R}^d
```

for all $`k \in \{1, \ldots, d`\}$ and $`\beta \in \mathbb{Z}^d`$.
Notice that once we truncate the $`\mathbb{Z}^d`$ sums over some finite set of indices $`\{-K, \ldots, K \}^d`$ for $`K \in \mathbb{N}`$ we obtain a system of ODEs.

In short, we have converted our original PDE to a system of ODEs for the basis coefficients $`\{ \psi_\alpha \}_\alpha`$. The code in this repository is dedicated to solving this ODE system, for specific choices of $`\{Z_\alpha \}_\alpha`$ and end-point measures $`P_0, P_1`$.


Code
---

We now take a closer look at the code. We discuss both the experiment in `notebooks/` as well as the various abstractions used in `src/optimalinterp/`.
A first time user is advised to look at `notebooks/gaussian_example_shooting.ipynb` since it has the most detailed comments.

All notebooks are run in $`d=1`$ dimensions. Most use Gaussian end-point measures $`P_0`$ and $`P_1`$ although that should be transparent in each notebook.

Notebooks
---
1. `notebooks/gaussian_example_shooting.ipynb` solves the ODE via a shooting method for a stochastic basis $`Z_\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)`$ consisting of Gaussian random variables.
2. `notebooks/wrapped_gaussian_example_shooting.ipynb` does as in (1) above but now the stochastic basis $`\{Z_\alpha \}_\alpha`$ consists of Gaussian distributions on the torus; these can be defined in terms of quotient maps $`\mathbb{R}^d \to \mathbb{R}^d / \mathbb{Z}^d`$.
3. `notebooks/wrapped_gaussian_example_basis.ipynb` treats the target ODE as a $`d=1`$ PDE and uses a collocation method to solve it. The stochastic basis used is the one discussed above in (2).

Abstractions
---
The main abstractions the user should be aware of are the following:

---

The class
````python
class StochasticBasis(eqx.Module, ABC):
````
encapsulates a choice of stochastic basis $`\{ Z_\alpha \}_\alpha`$.
Examples include the class
````python
class GaussianConvolutionBasis(StochasticBasis):
````
used in experiments (1) above and the class
````python
class WrappedGaussianConvolutionBasis(StochasticBasis):
````
used in experiments (2) and (3).
Note that in the current implementation, a stochastic basis contains random variables drawn according to the end-point measures, meaning that $`Z_0 \sim P_0, Z_{N-1} \sim P_1`$ and the index runs $`\alpha \in \{0, 1, \ldots, N-1\}`$.
As a result, a stochastic basis contains a `.sample()` method that can return samples from the source and target.

---

The class
````python
class MomentGeneratingPhi(ABC):
````
provides the core functions of the `StochasticBasis` class above: it encapsulates the characteristic functions

```math
    \Phi_\alpha(\xi) = \mathbb{E} \left[ e^{i \xi Z_\alpha} \right] 
```

for each $`\alpha \in \mathcal{A}`$, which mathematically is the way in which the chosen  stochastic basis $`\{ Z_\alpha \}_\alpha`$ enters the ODE.

---

The class
````python
class ModalInterpolant(eqx.Module, ABC):
````
encapsulates the ansatz 

```math
 X_t = \sum_{\alpha} \psi_\alpha(t) Z_\alpha
```

A modal interpolant can be instantiated with user specified $`\{ \psi_\alpha \}_\alpha`$ and $`\{Z_\alpha \}_\alpha`$ or for a specified $`\{Z_\alpha \}_\alpha`$ stored in `stochastic_basis` the user can (try to) solve the above ODE by e.g. running
````python
optimal_interpolant.compute_optimal_psi_shooting(stochastic_basis)
````
which returns another modal interpolant.
