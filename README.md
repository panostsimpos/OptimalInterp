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
The project contains three main directories

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
1. What makes some stochastic interpolants better than others?
2. Given a notion of "nice" interpolant, which interpolant is the "nicest"? In other, solve the optimization problem
$
    \min_{I} \mathcal{O}(I)
$
and $\mathcal{O}$ is the answer to question $1$.

Key Idea
---

A perspective I took in this [workshop paper](https://arxiv.org/pdf/2510.11657) was to *assert* that a better interpolant is an interpolant that leads to an easily integrable $v_t$. 
By elementary considerations, the easiest field to integrate is a field which is constant along the flow:
$$
    \frac{d}{dt} v_t(x_t) = 0 \iff \partial_t v_t(x) + (v_t \cdot \nabla) v_t(x) \equiv 0
$$
and the latter equation needs to hold for all $x \in \R^d$, assuming surjectivity of the flow---it is also sometimes referred to as the (inviscid) Burger's equation.
This allows us to pose the following problem


Now the main result in the workkshop paper is that this 
