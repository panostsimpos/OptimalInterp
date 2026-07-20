# Optimal Interpolants

This repository was built in collaboration with Daniel Sharp, a fellow PhD student at the Uncertainty Quantification Group at MIT. Development happened in a private repository thus git history does not reflect contribution.

We created this repository to explore certain ideas that came out my [workshop paper](https://arxiv.org/pdf/2510.11657) and led us to write [this paper](https://arxiv.org/abs/2504.14425) together.
None of the content of the repository ended up in the paper. 
However, going through this gave us essential intuition.

The nature of this repository is exploratory. 
We study variations of [Stochastic Interpolants](https://arxiv.org/pdf/2303.08797) using tools from Numerical and Stochastic Analysis.

A lot of the experiments failed and this is precisely what led us to conjecture and prove the impossibility theorems in [our paper](https://arxiv.org/abs/2504.14425).
Nonetheless, some of the mathematical ideas and computational abstractions could be use to explore adjacent questions of optimality on the design of stochastic interpolants, e.g. the open questions in our work. 
Due to these considerations we decided to make our code public.

For an overview of the problem and the math involved please keep on reading.

This README contains the following:

1. Installation instructions.
2. Usage and content description.
3. Mathematical framework exposition.

I would like to note that this codebase was built from scratch by the two of us, without the use of any LLM or agentic development tools.


Installation instructions
---
This project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository first
````bash
 git clone TODO
 cd OptimalInterp
````
Install Python and create the project environment
````bash
uv python install 3.12
uv sync --locked --python 3.12
````
Verufy the installation by 
````bash
uv run pytest
````
All tests should pass.


Usage and content description
---
The project directory structure is roughly as follows

```text
OptimalInterp/
├── README.md
├── pyproject.toml
├── python-version
├── uv.lock
├── src/
│   └── optimalinterp/
└── tests/
    └── 
└── notebooks/
    └── 

```
Tests are in `test/`, implementations are in `src/` and experiment notebooks are in `notebooks/`.

Mathematical framework
---
The [Stochastic Interpolants paper](https://arxiv.org/pdf/2303.08797) proposes an elegant framework for generative modelling. 
<!-- Namely, generative models are viewed as probability measure trajectories $t \mapsto P_t$ and each $P_t \in \mathcal{P}(\R^d)$ that are induced by  -->
Given a data (target) distribution $P_1$ and a source (noise) distribution $P_0$ draw $(X_0, X_1) \in P_0 \otimes P_1$ 
choose a nice interpolant functions $I_t: \R^d \times \R^d \to \R^d$
and set
$$
    X_t = I_t(X_0, X_1) .
$$
Then, compute the conditional expectation
$$
    v_t(X_t) = \mathbb{E} \left[ \dot X_t \mid X_t \right]
$$
solve the ODE
$$
\begin{cases}
    \dot \phi_t(x_t) = v_t(x_t) \\
    x_0 = x
\end{cases}
$$
and sample (generate) from $P_1$ by drawing $X_0 \sim P_0$ and setting
$$
    X_1 = \phi_1(X_0) .
$$
<!-- $$
    P_t = \operatorname{Law} X_t
$$ -->

This powerful framework encompasses many known generative models, e.g., diffusion models, continuous normalizing flows, variational auto-encoders etc but it also describes, at least in principle, *many new* generative models. Indeed, the known generative models correspond to "relatively straightforward" choices of $I_t$ e.g. 
$$
    X_t = \alpha_t X_0 + \beta_t X_1
$$
for scalar functions $t \mapsto \alpha_t$ and $t \mapsto \beta_t$. 

So we wondered: are there *better* generative models, described by exotic interpolant functions $I_t$? What does better even mean, in this context?

**Questions**
1. What makes some stochastic interpolants better than others? Find an objective $\cal{O}$ the assigns a cost $\mathcal{O}(I)$ to a stochastic interpolant $I$.
2. Solve the optimization problem
$
    \min_{I} \mathcal{O}(I)
$
and $\mathcal{O}$ is the answer to question $1$.

Key Mathematical Result
---

A perspective I took in this [workshop paper](https://arxiv.org/pdf/2510.11657) was to *assert* that a better interpolant is an interpolant that leads to an easily integrable $v_t$. 
By elementary considerations, the easiest field $v$ to integrate is a field which is constant along the flow:
$$
    \frac{d}{dt} v_t(x_t) = 0 \iff \partial_t v_t(x) + (v_t \cdot \nabla) v_t(x) \equiv 0
$$
and the latter equation needs to hold for all $x \in \R^d$, assuming surjectivity of the flow---it is also sometimes referred to as the (inviscid) Burger's equation.
This allows us to pose the following problem

**Problem**: Given source and target measures $P_0, P_1$ find an interpolant $I_t$ such that with $X_t = I_t(X_0, X_1)$ and $(X_0, X_1) \sim P_0 \otimes P_1$ and $v_t$ as above we satisfy the Burger's equation
$$
    \partial_t v_t + (v_t \cdot \nabla) v_t(x) \equiv 0 .
$$


Now the main result in the workshop paper is that we can transform the above PDE to one with more favorable properties. To wit, with
$$
    a_t(x) = \mathbb{E} \left[ \ddot X_t \mid X_t = x \right] 
    \quad \textup{and} \quad
    \Pi_t(x) = \operatorname{Cov} \left( \dot X_t \mid X_t = x  \right)
$$
the Burger's equation is equivalent to the PDE
$$
    \nabla \cdot \left( \rho_t \, \Pi_t \right) = \rho_t \, a_t ,
$$
and $\rho_t$ is the density of $X_t$.

Key computational question
---
We want to solve the **Problem** described in the above questions.
Solving PDEs numerically is computationally challenging enough and yet our problem is even more involved. Indeed, we are not searching for a function that satisfies a certain differential equation; we are searching for interpolants $I_t$ whose path statistics $\rho_t$, $a_t$ and $\Pi_t$, functions over space-time, satisfy a differential equation. To make progress, we assert an ansatz. Suppose we can write
$$
    X_t = \sum_\alpha \psi_{\alpha}(t) Z_\alpha
$$
for fixed, $\R^d$ valued random variables $\{ Z_{\alpha} \}_\alpha$, multi-indices $\alpha \in \mathcal{A}$ and real-valued functions $\psi_\alpha : [0,1] \to \R$.

A key mathematical result shown in DERIVATIONS.md shows that under this ansatz we have
$$
    \nabla \cdot \left( \rho_t \, \Pi_t \right) = \rho_t \, a_t , \iff \mathbf{D} \, \ddot \psi(t) = \dot \psi(t)^\top \, \mathbf{C} \, \dot \psi(t) 
$$
and $\mathbf{D} \, , \, \mathbf{C}$ are tensors of order $3$ and $4$, respectively, defined in terms of the moment generating functions
$$
    \Phi_\alpha(\xi) = \mathbb{E} \left[ e^{i \xi Z_\alpha} \right] .
$$
In index notation we can re-write the above equation as 
$$
    \sum_{\alpha \in \Z^d} D^k_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{(\alpha, \gamma) \in \Z^d \times \Z^d} \dot \psi_\alpha(t) \, C^k_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, ,
    % \quad \textup{for all} \quad k \in \{1, \ldots, d\} \quad \textup{and} \quad \beta \in \R^d
$$
for all $k \in \{1, \ldots, d\}$ and $\beta \in \Z^d$.
Notice that once we truncate the $\Z^d$ sums over some finite set of indices $-K, -K + 1, \ldots, K-1, K$ for $K \in \N$ we obtain a system of ODEs.

In short, we have converted our original PDE to a system of ODEs for the basis coefficients $\{ \psi_\alpha \}_\alpha$. The code in this repository is dedicated to solving this ODE system, for specific choices of $\{Z_\alpha \}_\alpha$ and end-point measures $P_0, P_1$.


Code
---

We note take a closer look at the code. We discuss both the experiment in `notebooks/` as well as the various abstractions used in `src/optimalinterp/`.
A first time user is advised to look at `notebooks/gaussian_example_shooting.py` since it has the most detailed comments and functions as a mini tutorial.

All notebooks are run in $d=1$ dimensions. Most use Gaussian end-point measures $P_0$ and $P_1$ although that should be transparent in each notebook.

Notebooks
---
1. `notebooks/gaussian_example_shooting.py` solves the ODE via a shooting method for a stochastic basis $Z_\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$ consisting of Gaussians.
2. `notebooks/wrapped_gaussian_example_shooting.py` does as in (1) above but now the stochastic basis $\{Z_\alpha \}_\alpha$ consists of Gaussian distributions on the torus which are defined in terms of quotient maps $\R^d \to \R^d / \Z^d$.
3. `notebooks/gaussian_example_basis.py` uses the same stochastic basis as (1) and treats the resulting ODE as a $d=1$ PDE and uses a collocation method.
4. `notebooks/gaussian_example_basis.py` uses a collocation method on the stochastic basis discussed above in (3).

Abstractions
---
The main abstractions the user should be aware of are the following:
1. The class
````python
class StochasticBasis(eqx.Module, ABC):
````
encapsulates a choice of stochastic basis $\{ Z_\alpha \}_\alpha$.
Examples include the class
````python
class GaussianConvolutionBasis(StochasticBasis):
````
used in experiments (1) and (3) above and the class
````python
class WrappedGaussianConvolutionBasis(StochasticBasis):
````
used in experiments (2) and (4).
Note that in the current implementation, a stochastic basis contains random variables drawn according to the end-point measures, meaning that $Z_0 \sim P_0, Z_N \sim P_N$ and the index runs $\alpha \in \{0, 1, \ldots, N-1, N\}$.
As a result, a stochastic basis contains a `.sample()` method that can return samples from the source and target.

2. The class
````python
class MomentGeneratingPhi(ABC):
````
provides the core functions of the `StochasticBasis` class above: it encapsulates the moment generating functions
$$
    \Phi_\alpha(\xi) = \mathbb{E} \left[ e^{i \xi Z_\alpha} \right] 
$$
for each $\alpha \in \mathcal{A}$, which mathematically is the way in which the chosen  stochastic basis $\{ Z_\alpha \}_\alpha$ enters the ODE at hand.
3. The class
````python
class ModalInterpolant(eqx.Module, ABC):
````
encapsulates the ansatz 
$$
 X_t = \sum_{\alpha} \psi_\alpha(t) Z_\alpha
$$
A modal interpolant can be instantiated with user specified $\{ \psi_\alpha \}_\alpha$ and $\{Z_\alpha \}_\alpha$; or for a specified \{Z_\alpha \}_\alpha$ called `stochastic_basis` the user can run
````python
optimal_interpolant.compute_optimal_psi_shooting(stochastic_basis)
````
which returns another modal interpolant.



