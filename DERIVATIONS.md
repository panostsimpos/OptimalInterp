## General Treatment of the PDE

In this section we outline a general strategy to tackle the PDE using the expansion as well as elementary tools from Fourier analysis.

### Preliminaries

First, we set the stage for our analysis.

**Definition.**

  The *torus of side-length $2 \pi$ in $d$ dimensions* is the quotient manifold

$$
    \mathbb{T}^d = \mathbb{R}^d / ( 2 \pi \mathbb{Z}^d ) \, .
$$

Now following the exposition of Grafakos (2008, Chapter 3), albeit with a different normalization on $\mathbb{T}^d$, we can define:

**Definition.**

  The *Fourier transform* of a complex valued function $f \in L^1(\mathbb{T}^d)$ is given by

$$
    \mathcal{F}[f](\xi) \coloneq \widehat{f}(\xi) \coloneq \frac{1}{(2 \pi)^d} \int_{\mathbb{T}^d} f(x) \, e^{-i \, \xi \cdot x} \, \mathrm{d} x \, , \quad \xi \in \mathbb{Z}^d \, .
$$

**Definition.**

  Given a summable sequence $s \in \ell^1(\mathbb{Z}^d)$, i.e. $s: \mathbb{Z}^d \to \mathbb{C}$ and

$$
    \sum_{\xi \in \mathbb{Z}^d} |s(\xi)| < \infty \, ,
$$

  then the *Fourier series* associated to $s$ is given by

$$
    \widecheck{s}(x) \coloneq \sum_{\xi \in \mathbb{Z}^d} s(\xi) \, e^{i \, \xi \cdot x} \, , \quad x \in \mathbb{T}^d \, .
$$

Now a classical theorem states that the Fourier series is, in fact, *inverse* to the Fourier transform, i.e.:

**Theorem (Grafakos, 2008, Proposition 3.2.5.).**

  For any $f \in L^1(\mathbb{T}^d)$ one has $\widehat f \in \ell^1(\mathbb{Z}^d)$ and moreover

$$
    f(x) = \frac{1}{(2 \pi)^d} \widecheck{\widehat f}(x)
$$

  In particular, we have a pair of bijective operators that are inverse of each other:

$$
\begin{aligned}
  \mathcal{F}: L^1(\mathbb{T}^d) \to \ell^1(\mathbb{Z}^d) \qquad & & \mathcal{F}: f \mapsto \widehat f \, , \\
  \mathcal{F}^{-1}: \ell^1(\mathbb{Z}^d) \to L^1(\mathbb{T}^d) \qquad & & \mathcal{F}^{-1}: s \mapsto \widecheck s \, .
\end{aligned}
$$

**Remark.**

  We note that although the above theorem gives both injectivity and surjectivity of $\mathcal{F}$ one typically proves the above by *using* injectivity which is established separately see, e.g., Grafakos (2008, Theorem 3.2.4).

### Fundemental domains and periodization

Working on the torus $\mathbb{T}^d$ as defined comes with a few caveats.
One important issue is that *non $2 \pi$-periodic functions are not well-defined on $\mathbb{T}^d$*.
For example, consider the multiplication map

$$
    m_c: x \mapsto c \, x
$$

  is not well defined on $\mathbb{T}^d$ for $c \notin \mathbb{Z}$.
  Indeed, take any $x = 2 \pi n$ for $n \in Z^d$ and note that $x \sim 0$ in $\mathbb{T}^d$ but

$$
    m_c(x) = 2 \pi c \, n \not\sim 0 \, .
$$

To alleviate this issue we work with a *fundamental domain* of $\mathbb{T}^d$:

**Definition.**

  A *fundamental domain* of $\mathbb{T}^d$ is any topological space $D \subseteq \mathbb{R}^d$ such that the map

$$
    \gamma: D \to \mathbb{T}^d \, , \quad x \mapsto [x] \, ,
$$

  is a homeomorphism. In particular, the cube with side-length $2 \pi$ endowed with periodic boundary conditions is a fundamental domain, i.e.

$$
    D \coloneq [0, 2 \pi]^d / \sim \, ,
$$

  where $\sim$ identifies $x$ and $y$ if $x - y \in (2 \pi \mathbb{Z})^d$.

For the remainder of this work, $D$ will be defined by the equation above.
Moreover, all analysis will be performed on $D$, i.e. the source and target measures will be supported on $D$ and so will our interpolating stochastic processes.

**Assumption.**

  Assume that source measure $P_0$ and target measure $P_1$ are supported on the fundamental domain $D$.
  Moreover, assume that the stochastic process $X_\bullet$ takes values in $D$ almost surely.

Now since the ultimate goal of this work is sampling, one might wonder how restrictive this assumptions are. In practice any density $p \in L^1(\mathbb{R}^d)$ can be put on $\mathbb{T}^d$ and then projected on $D$ by periodization:

**Definition (Periodization).**

  Given a density $p \in L^1(\mathbb{R}^d)$ we define its *periodization* on the fundamental domain $D$ as

$$
    (\pi_\sharp \, p)(x) = \sum_{n \in \mathbb{Z}^d} p(x + 2 \pi n) \, , \quad x \in D \, .
$$

The following lemma justifies the notation

**Lemma.**

  Consider the natural projection map

$$
    \pi: \mathbb{R}^d \to D \, , \quad x \mapsto [x] \, .
$$

  For any $P \in \mathcal{P}(\mathbb{R}^d)$ with density $p \in L^1(\mathbb{R}^d)$ we can sample $X \sim P$ and set

$$
    Y = \pi(X) \, .
$$

  Then, $Y$ has density $\pi_\sharp \, p$ given by the equation above.

**Proof.**

  Take some measurable set $A \subseteq D$ and compute

$$
\begin{aligned}
    \mathbb{P}(Y \in A) &= \mathbb{P}(X \in \pi^{-1}(A)) \\
    &= \int_{\pi^{-1}(A)} p(x) \, \mathrm{d} x \\
    &= \sum_{n \in \mathbb{Z}^d} \int_{A + 2 \pi n} p(x) \, \mathrm{d} x \\
    &= \int_A \sum_{n \in \mathbb{Z}^d} p(x + 2 \pi n) \, \mathrm{d} x \, ,
\end{aligned}
$$

  as required, where we have used the integrability of $p$ and Fubini's theorem to exchange the sum and the integral.

**Remark.**

  Note that the proof above remains valid if we replace $D$ with $\mathbb{T}^d$.

**Corollary.**

  Viewing $D$ as a subset of $\mathbb{R}^d$ we have that any $P \in \mathcal{P}(D)$ satisfying $P(D) \geq 1 - \epsilon$ for some $0 < \epsilon < 1$ one has the estimate

$$
    \operatorname{TV} \left( \pi_\sharp \, P , P \right) \leq \epsilon \, .
$$

**Proof.**

  Take any measurable set $A \subseteq D$ and compute

$$
\begin{aligned}
    |\pi_\sharp \, P(A) - P(A)| &= \left| \sum_{n \in \mathbb{Z}^d} P(A + 2 \pi n) - P(A) \right| \\
    &= \left| \sum_{n \neq 0} P(A + 2 \pi n) \right| \\
    &\leq \sum_{n \neq 0} P(D + 2 \pi n) \\
    &= P(\mathbb{R}^d \setminus D) \\
    &\leq \epsilon \, ,
\end{aligned}
$$

  as required.

In other words, we do not expect periodization, equivalently projection on $D$, to significantly alter measures concentrated on $D$. Moreover, any measure with sufficiently light tails can be rescaled so as to concentrate on $D$ thus we conclude that working on $D$ is not a significant restriction from a practical standpoint.

### Treatment of the PDE

First, we make the following assumption on the representation:

**Assumption.**

  Assume that the stochastic process $X_\bullet$ is 

$$
    X_t = \sum_{\alpha} \psi_\alpha(t) \, Z_\alpha \, ,
$$

  where $\psi_\alpha: [0,1] \to \mathbb{R}$ are the *modal functions* and the random variables $\{ Z_\alpha \}_\alpha$ with $Z_\alpha \in L^2(\Omega, \mathcal{F}, \mathbb{P})$ are the *stochastic basis* of $X_\bullet$.
  Moreover, assume that

$$
  Z^\alpha \perp\!\!\!\perp \, Z^\beta \, , \quad \forall \, \alpha \neq \beta \, ,
$$

  and that the sum in the displayed equation above converges in $L^2(\Omega, \mathcal{F}, \mathbb{P}; \mathbb{R}^d)$.

**Example.**

  A canonical example of such a representation is given by the Karhunen-LoÃ¨ve Expansion (KLE) of an $\mathbb{R}$-valued, centered Gaussian Process $X_\bullet \sim \mathcal{GP}( \mathbf{0}, C)$ that is square integrable. Indeed, taking the $\{ \psi_\alpha \}_\alpha$ to be the eigenfunctions of the covariance operator $C$ one obtains an $L^2$ convergent sum of the form above with $Z_\alpha$ satisfying

$$
    Z_\alpha = \int_0^1 \psi_\alpha(t) \, X_t \, \mathrm{d} t \, ,
$$

  see Sullivan (2015, Theorem 11.4) for details. Now since a Gaussian Process induces a Gaussian measure on path space (Bogachev, 1998, Proposition 2.3.9.) and a Gaussian measure yields a univariate Gaussian when passed through a linear functional (Bogachev, 1998, Definition 2.2.1.ii.) it follows that the $Z_\alpha$ are Gaussian random variables.
  Moreover, a central feature of the KLE is that the $Z_\alpha$ are uncorrelated (Sullivan, 2015, Theorem 11.4) thus being Gaussian it follows that they are independent.

**Definition.**

    For each $\alpha \in \mathbb{Z}$ and $\xi \in \mathbb{R}$ write

$$
    \Phi_\alpha(\xi) = \mathbb{E}[e^{i \xi Z_\alpha}]
$$

  for the characteristic function of the random variables $Z_\alpha$ and also write $\Phi'_\alpha$ and $\Phi''_\alpha$ for its first and second derivatives, respectively.

Write $\psi(t) = (\psi_\alpha(t))_\alpha$ for the vector of modal functions.

**Theorem.**

  Assume $X_\bullet$ is a stochastic process satisfying the two assumptions above.
  Then, the PDE is equivalent to the infinite system of equations

$$
    \mathbf D\, \ddot \psi(t) = \dot \psi(t)^\top \, \mathbf C \, \dot \psi(t) \, .
$$

  or, more explicitly, for each $\beta \in \mathbb{Z}$

$$
    \sum_\alpha \mathbf D_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \mathbf C_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, ,
$$

  where for each $k \in \{1, \ldots, d\}$

$$
    D^k_{\beta \alpha}(t) = \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\gamma \neq \alpha} \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, ,
$$

  and

$$
\begin{aligned}
    C_{\alpha \beta \gamma}^{k}(t)
    = \sum_{j=1}^d \beta_j \, \Big\{ & \delta_{\alpha \gamma} \, \partial_j \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (\delta_{\alpha \gamma} - 1) \, \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \partial_k \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &- \Big( D_{\bullet \alpha}^j(t) \ast K_\bullet(t) \ast D_{\bullet \gamma}^k(t) \Big)(\beta) \Big\} \, .
\end{aligned}
$$

  and the functions $t \mapsto K_{\beta}(t)$ are given by the the formula

$$
    K_{\beta}(t) = \mathcal{F} \left[ \frac{1}{ \sum_\gamma \prod_\alpha \Phi_\alpha\left( -\gamma \, \psi_\alpha(t) \right) \, e^{i \, \gamma \, x} } \right](\beta)
$$

**Proof.**

  By using the definitions for $v_t, a_t,$ and $\Pi_t$ as well as the **modal interpolant** ansatz we obtain

$$
    \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \dot \psi_\gamma(t) \, \nabla \cdot \Big( \rho_t(x) \, G_{\alpha \gamma}(x,t) \Big) = \sum_{\alpha} \ddot \psi_\alpha(t) \, \rho_t(x) \, H_\alpha(x,t)
$$

  for all $x$, where

$$
\begin{aligned}
    \mathbf{G}_{\alpha \gamma}(x,t) \coloneq \operatorname{Cov}(Z_\alpha, Z_\gamma \mid X_t = x) \;\; \text{and} \;\; \mathbf{H}_\alpha(x,t) \coloneq \mathbb{E}[Z_\alpha \mid X_t = x] \, .
\end{aligned}
$$

  and we note that $\mathbf{G}_{\alpha \gamma}(x,t) \in \mathbb{R}^{d \times d}$ and $\mathbf{H}_\alpha(x,t) \in \mathbb{R}^d$.
  Now one can use a Fourier expansion to write

$$
\begin{aligned}
    \rho_t(x) \, \mathbf{G}_{\alpha \gamma}(x,t) &= \sum_{\beta \in \mathbb{Z}^d}  \tilde{\tilde{\mathbf C}}_{\alpha \beta \gamma}(t) \, e^{i \, \beta \cdot x} \, , \\
    \rho_t(x) \, \mathbf{H}_\alpha(x,t) &= \sum_{\beta \in \mathbb{Z}^d} \tilde{\mathbf{D}}_{\beta \alpha}(t) \, e^{i \, \beta \cdot x} \, .
\end{aligned}
$$

  therefore

$$
\begin{aligned}
    & \tilde{\tilde{\mathbf C}}_{\alpha \beta \gamma} \in \mathbb{C}^{d \times d} \quad \text{and} \quad \tilde{\mathbf D}_{\beta \alpha} \in \mathbb{C}^d \, .
\end{aligned}
$$

  Plugging these expressions back into the displayed PDE above and equating the Fourier coefficients on both sides we obtain the system

$$
    \sum_\alpha \tilde{\mathbf D}_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \tilde{\mathbf C}_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, ,
$$

  for each $\beta \in \mathbb{Z}^d$ where, using the Einstein summation convention, we define

$$
\begin{aligned}
    \tilde{C}_{\alpha \beta \gamma}^{k}(t) &\coloneq \mathcal{F} \big[ \, \partial_j \left( \rho_t \, {G}_{\alpha \gamma}^{j k} \right) \, \big](\beta) \\
    &= i \, \beta_j \, \mathcal{F} \big[ \, {G}_{\alpha \gamma}^{j k}(x,t) \, \big](\beta) \\
    &= i \, \beta_j \, \tilde{\tilde{ C}}_{\alpha \beta \gamma}^{j k}(t) \, ,
\end{aligned}
$$

  where the Fourier operator acts componentwise and we have used a renormalization of Grafakos (2008, Proposition 3.1.2.).
  Succinctly, we have $\tilde{\mathbf{C}}_{\alpha \beta \gamma} \in \mathbb{C}^{d}$ defined by

$$
    \tilde{\mathbf C}_{\alpha \beta \gamma}(t) = i \, \beta \cdot \tilde{\tilde{\mathbf C}}_{\alpha \beta \gamma}(t) \, .
$$

  Finally, setting

$$
\begin{aligned}
    & {\mathbf C}_{\alpha \beta \gamma}(t) \coloneq i \, (2 \pi)^d \,  {\tilde{\mathbf C}}_{\alpha \beta \gamma}(t) \\
    & {\mathbf D}_{\beta \alpha}(t)        \coloneq i \, (2 \pi)^d \, \tilde{\mathbf D}_{\beta \alpha}(t) \, ,
\end{aligned}
$$

  we see that the intermediate system above is equivalent to a system that writes

$$
    \sum_\alpha \mathbf D_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \mathbf C_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, .
$$

  To finish the proof, we need to show that the coefficients $\mathbf C_{\alpha \beta \gamma}(t)$ and $\mathbf D_{\beta \alpha}(t)$ are as claimed in the statement of the theorem.
  Thus, for $j \in \{1, \ldots, d\}$ we compute

$$
\begin{aligned}
    D^j_{\beta \alpha}(t) &= i \, (2 \pi)^d\frac{1}{(2 \pi)^d} \int_{D} \rho_t(x) \, \mathbb{E}[Z^j_\alpha \mid x] \, e^{-i \, \beta \, x} \, \mathrm{d} x \\
    &= i \, \mathbb{E} \Big[ \mathbb{E} \big[ Z_\alpha^j | X_t \big] \, e^{-i \, \beta \cdot X_t} \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \cdot X_t} \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \cdot \left( \sum_\gamma \psi_\gamma(t) \, Z_\gamma \right) } \Big] \\
    &= i \, \mathbb{E} \Big[ \prod_\gamma Z_\alpha^j \, e^{-i \, \beta \, \psi_\gamma(t) \, Z_\gamma} \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \, \psi_\alpha(t) \, Z_\alpha} \Big] \, \prod_{\gamma \neq \alpha} \mathbb{E} \Big[ e^{-i \, \beta \, \psi_\gamma(t) \, Z_\gamma} \Big] \\
    &= \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\gamma \neq \alpha} \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, .
\end{aligned}
$$

  where in the second-to-last line we have used independence and in the last line we have used the definition of the characteristic function.

  Similarly, we can compute:

$$
\begin{aligned}
    C_{\alpha \beta \gamma}^{j k}(t) &= \beta \, i^2 \, (2 \pi)^d \, \frac{1}{(2 \pi)^d} \int_{\mathbb{R}^d} \rho_t(x) \, \operatorname{Cov}(Z_\alpha^j , Z_\gamma^k \mid x) \, e^{-i \, \beta \, x} \, \mathrm{d} x \\
    &= - \beta \, \mathbb{E} \Big[ \operatorname{Cov}(Z_\alpha^j, Z_\gamma^k \mid X_t) \, e^{-i \, \beta \, X_t} \Big] \\
    &= - \beta \, \mathbb{E} \Big\{ \Big[ \mathbb{E} \big[ Z_\alpha^j Z_\gamma^k | X_t \big] - \mathbb{E}\big[Z_\alpha^j | X_t \big] \, \mathbb{E}\big[Z_\gamma^k | X_t \big] \Big] e^{-i \, \beta \, X_t} \Big\} \\
    &= - \beta \, \mathbb{E} \Big[ Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \, X_t} \Big] + \beta \, \mathbb{E} \Big[ \mathbb{E}[Z_\alpha^j | X_t] \, \mathbb{E}[Z_\gamma^k | X_t] e^{-i \, \beta \, X_t}  \Big]  \\
\end{aligned}
$$

  Now using the independence assumption we can compute the first term as

$$
\begin{aligned}
    &\mathbb{E} \Big[ Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \, X_t} \Big] = \\
    &= \Big[ Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \, \sum_\delta \psi_\delta(t) Z_\delta} \Big] \\
    &= \mathbb{E} \Big[ \prod_\delta \, Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \, \psi_\delta(t) \, Z_\delta} \Big] \\
    &= \delta_{\alpha \gamma} \, \mathbb{E} \Big[ Z_\alpha^j \, Z_\alpha^k \, e^{-i \, \beta \, \psi_\alpha(t) \, Z_\alpha} \Big] \, \prod_{\delta \neq \alpha} \mathbb{E} \Big[ e^{-i \, \beta \, \psi_\delta(t) \, Z_\delta} \Big] \\
    &+ (1 - \delta_{\alpha \gamma}) \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \, \psi_\alpha(t) \, Z_\alpha} \Big] \, \mathbb{E} \Big[ Z_\gamma^k \, e^{-i \, \beta \, \psi_\gamma(t) \, Z_\gamma} \Big] \, \prod_{\delta \neq \alpha, \gamma} \mathbb{E} \Big[ e^{-i \, \beta \, \psi_\delta(t) \, Z_\delta} \Big] \\
    &= - \delta_{\alpha \gamma} \, \partial_j \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (1 - \delta_{\alpha \gamma}) \, \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \partial_k \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \, .
\end{aligned}
$$

  Finally, we can use the convolution theorem, i.e. the discrete Fourier convolution lemma below, to obtain

$$
\begin{aligned}
    \mathbb{E} \Big[ \mathbb{E}[Z_\alpha^j | X_t] \, \mathbb{E}[Z_\gamma^k | X_t] e^{-i \, \beta \, X_t}  \Big] &= (2 \pi)^d \, \mathcal{F}\Big[ \rho_t \, H_\alpha^j \, H_\gamma^k \Big](\beta) \\
    &= (2 \pi)^d \, \mathcal{F}\left[ \rho_t \, H_\alpha^j \, \frac{1}{\rho_t} \, \rho_t \, H_\gamma^k \right] \\
    &= (2 \pi)^d \, \mathcal{F}[\rho_t \, H_\alpha^j] \ast \mathcal{F}\left[ \frac{1}{\rho_t} \right] \ast \mathcal{F}[\rho_t \, H_\gamma^k] \, ,
\end{aligned}
$$

  and note that we have computed

$$
    \mathbf D_{\xi \alpha}(t) = i \, (2 \pi)^d \, \mathcal{F}[\rho_t \, \mathbf H_\alpha](\beta)  \, ,
$$

  for any $\alpha \in \mathbb{N}$ and $\beta \in \mathbb{Z}^d$.
  Thus, setting $\tilde K_\beta(t) = \mathcal{F}\left[ \frac{1}{\rho_t} \right](\beta)$ we have

$$
    \mathbb{E} \Big[ \mathbb{E}[Z_\alpha^j | X_t] \, \mathbb{E}[Z_\gamma^k | X_t] e^{-i \, \beta \, X_t}  \Big] = -\frac{1}{(2 \pi)^d} \, \Big( D^j_{\bullet \alpha}(t) \ast \tilde K_\bullet(t) \ast D^k_{\bullet \gamma}(t) \Big)(\beta) \, .
$$

  Finally, the kernel $K$ can be computed as follows: first, use the Fourier representation theorem above to write

$$
    \rho_t(x) = \sum_{\beta} \mathcal{F}[\rho_t](\beta) \, e^{i \, \beta \, x} \, .
$$

  Now, compute

$$
\begin{aligned}
      \mathcal{F}[\rho_t](\beta) &= \frac{1}{(2 \pi)^d} \int_{\mathbb{T}^d} e^{-i \, \beta \, x} \, \rho_t(x) \, \mathrm{d} x \\
      &= \frac{1}{(2 \pi)^d} \mathbb{E} \left[ e^{-i \, \beta \, X_t} \right] \\
      &= \frac{1}{(2 \pi)^d} \prod_\alpha \Phi_\alpha\left( -\beta \, \psi_\alpha(t) \right) \, ,
\end{aligned}
$$

  where the last line follows by writing $X_t = \sum_\alpha \psi_\alpha(t) \, Z_\alpha$ and using independence.
  Thus, we have

$$
\begin{aligned}
    \tilde K_\beta(t) &= \mathcal{F} \left[ \frac{1}{\sum_\beta \frac{1}{(2 \pi)^d} \prod_\alpha \Phi_\alpha\left( -\beta \, \psi_\alpha(t) \right) \, e^{i \, \beta \, x}} \right](\beta) \\
    &= (2 \pi)^d \, \mathcal{F} \left[ \frac{1}{ \sum_\beta \prod_\alpha \Phi_\alpha\left( -\beta \, \psi_\alpha(t) \right) \, e^{i \, \beta \, x} } \right](\beta) \\
\end{aligned}
$$

  Finally, set

$$
    K_{\beta}(t) \coloneq \frac{1}{(2 \pi)^d} \, \tilde K_{\beta}(t) \, ,
$$

  and putting it all together we get the final expression

$$
\begin{aligned}
    C_{\alpha \beta \gamma}^k(t)
    = \beta_j \, \Big\{ & \delta_{\alpha \gamma} \, \partial_j \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (\delta_{\alpha \gamma} - 1) \, \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \partial_k \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &- \Big( D^j_{\bullet \alpha}(t) \ast K_\bullet(t) \ast D^k_{\bullet \gamma}(t) \Big)(\beta) \Big\}
\end{aligned}
$$

  or in vector form

$$
\begin{aligned}
    \mathbf{C}_{\alpha \beta \gamma}(t)
    = \beta \cdot \Big\{ & \delta_{\alpha \gamma} \, \left( \boldsymbol{\nabla}^2 \Phi_\alpha \right)\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (\delta_{\alpha \gamma} - 1) \, \left( \boldsymbol{\nabla} \Phi_\alpha \right)\left( -\beta \psi_\alpha(t) \right) \otimes \left( \boldsymbol{\nabla} \Phi_\gamma \right)\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &- \Big( D^j_{\bullet \alpha}(t) \ast K_\bullet(t) \ast D^k_{\bullet \gamma}(t) \Big)(\beta) \Big\} \, .
\end{aligned}
$$

**Lemma.**

  For functions $f, g: \mathbb{T}^d \to \mathbb{R}$ we have

$$
    \mathcal{F}[f \, g](\beta) = \sum_{\alpha} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\beta - \alpha) \, .
$$

**Proof.**

  We compute

$$
    \mathcal{F}[f \, g](\beta) = \frac{1}{(2 \pi)^d} \int_{\mathbb{T}^d} e^{-i \, \beta \, x} \, f(x) \, g(x) \, \mathrm{d} x \, .
$$

  and by expanding $f$ and $g$ in Fourier series we obtain

$$
\begin{aligned}
    \mathcal{F}[f \, g](\beta) &= \frac{1}{(2\pi)^d} \int_{\mathbb{T}^d} e^{-i \, \beta \, x} \, \left( \sum_{\alpha} \mathcal{F}[f](\alpha) \, e^{i \, \alpha \, x} \right) \, \left( \sum_{\gamma} \mathcal{F}[g](\gamma) \, e^{i \, \gamma \, x} \right) \, \mathrm{d} x \\
    &= \sum_{\alpha, \gamma} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\gamma) \,\frac{1}{(2\pi)^d} \int_{\mathbb{T}^d} e^{i \, (\alpha + \gamma - \beta) \, x} \, \mathrm{d} x \\
    &= \sum_{\alpha, \gamma} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\gamma) \, \mathbf{1}_{\alpha + \gamma = \beta} \\
    &= \sum_{\alpha} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\beta - \alpha) \, .
\end{aligned}
$$