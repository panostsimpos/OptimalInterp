### PDE as an ODE system

**Definition.**
  Let the torus $`D = \mathbb{R}^d / \left(2\pi \mathbb{Z}^d\right)`$.
  The *Fourier transform* of a complex valued function $`f \in L^2(D)`$ is given by

  ```math
    \mathcal{F}[f](\xi) \coloneq \widehat{f}(\xi) \coloneq \frac{1}{(2 \pi)^d} \int_{D} f(x) \, e^{-i \, \xi \cdot x} \, dx \, , \quad \xi \in \mathbb{Z}^d \, .
  ```

  Furthermore, the *Fourier series* associated to $`f`$ is

  ```math
    \mathcal{S}[f](x) \coloneq \sum_{\xi \in \mathbb{Z}^d} \widehat{f}(\xi) \, e^{i \, \xi \cdot x} \, , \quad x \in D \, .
  ```

  and we have the identity

  ```math
    f = \mathcal{S}[f] \quad \textup{in} \quad L^2(D)
  ```


**Definition.**

  A stochastic process $`X_\bullet`$ is a **modal interpolant** if it can be written in the form

```math
    X_t = \sum_{\alpha} \psi_\alpha(t) \, Z_\alpha \, ,
```

  where $`\psi_\alpha: [0,1] \to \mathbb{R}`$ are the **modal functions** and the random variables $`\{ Z_\alpha \}_\alpha`$ with $`Z_\alpha \in \mathbb{R}^d`$ are the **stochastic basis** of $`X_\bullet`$.
  Moreover, assume that

```math
  Z^\alpha \perp \, Z^\beta \, , \quad \forall \, \alpha \neq \beta \, ,
```

  and that the sum in the displayed equation above converges in $`L^2(\Omega, \mathcal{F}, \mathbb{P}; \mathbb{R}^d)`$.

**Example.**

  A canonical example is the Karhunen–Loève Expansion (KLE) of an $`\mathbb{R}`$-valued, centered Gaussian Process $`X_\bullet \sim \mathcal{GP}( \mathbf{0}, C)`$ that is square integrable. Indeed, taking the $`\{ \psi_\alpha \}_\alpha`$ to be the eigenfunctions of the covariance operator $`C`$ one obtains an $`L^2`$ convergent sum of the form above with $`Z_\alpha`$ satisfying

```math
    Z_\alpha = \int_0^1 \psi_\alpha(t) \, X_t \, \mathrm{d} t \, ,
```

  see Sullivan (2015, Theorem 11.4) for details. Now since a Gaussian Process induces a Gaussian measure on path space (Bogachev, 1998, Proposition 2.3.9.) and a Gaussian measure yields a univariate Gaussian when passed through a linear functional (Bogachev, 1998, Definition 2.2.1.ii.) it follows that the $`Z_\alpha`$ are Gaussian random variables.
  Moreover, a central feature of the KLE is that the $`Z_\alpha`$ are uncorrelated (Sullivan, 2015, Theorem 11.4) thus being Gaussian it follows that they are independent.

**Definition.**

  For each $`\alpha \in \mathbb{Z}`$ and $`\xi \in \mathbb{R}`$ write

```math
    \Phi_\alpha(\xi) = \mathbb{E}[e^{i \xi \cdot Z_\alpha}]
```

  for the characteristic function of the random variables $`Z_\alpha`$ and also write $`\Phi'_\alpha`$ and $`\Phi''_\alpha`$ for its first and second derivatives, respectively.

Write $`\psi(t) = (\psi_\alpha(t))_\alpha`$ for the vector of modal functions.

**Theorem (Informal).**

  Fix $`D = \mathbb{R}^d / \left(2\pi \mathbb{Z}^d\right)`$.
  Assume $`X_\bullet`$ is a modal interpolant defined on $`D`$, consisting of a nice enough stochastic basis $`\{Z_\alpha\}_\alpha`$ and nice enough modal functions $`\{\psi_\alpha \}_\alpha`$.
  Then, the PDE 

  ```math
    \nabla \cdot \left( \rho_t \, \Pi_t \right) = \rho_t \, a_t 
  ```

  is equivalent to the infinite system of equations

```math
    \mathbf D\, \ddot \psi(t) = \dot \psi(t)^\top \, \mathbf C \, \dot \psi(t) \, .
```

  or, more explicitly, for each $`\beta \in \mathbb{Z}^d`$ and each $`k \in \{1, \ldots, d\}`$ we get

```math
    \sum_\alpha \mathbf D_{\beta \alpha}^k(t) \, \ddot \psi_\alpha(t) = \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \mathbf C_{\alpha \beta \gamma}^k(t) \, \dot \psi_\gamma(t) \, ,
```

  where

```math
    D^k_{\beta \alpha}(t) = \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\gamma \neq \alpha} \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, ,
```

  and

```math
\begin{aligned}
    C_{\alpha \beta \gamma}^{k}(t)
    = \sum_{j=1}^d \beta_j \, \Big\{ & \delta_{\alpha \gamma} \, \partial_j \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (\delta_{\alpha \gamma} - 1) \, \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \partial_k \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &- \Big( D_{\bullet \alpha}^j(t) \ast K_\bullet(t) \ast D_{\bullet \gamma}^k(t) \Big)(\beta) \Big\} \, .
\end{aligned}
```

  and the functions $`t \mapsto K_{\beta}(t)`$ are given by the formula

```math
    K_{\beta}(t) = \mathcal{F} \left[ \frac{1}{ \sum_\gamma \prod_\alpha \Phi_\alpha\left( -\gamma \, \psi_\alpha(t) \right) \, e^{i \, \gamma \cdot x} } \right](\beta)
```

**Proof.**

  By using the definitions for $`v_t, a_t,`$ and $`\Pi_t`$ as well as the **modal interpolant** ansatz we obtain

```math
    \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \dot \psi_\gamma(t) \, \nabla \cdot \Big( \rho_t(x) \, G_{\alpha \gamma}(x,t) \Big) = \sum_{\alpha} \ddot \psi_\alpha(t) \, \rho_t(x) \, H_\alpha(x,t)
```

  for all $`x`$, where

```math
\begin{aligned}
    \mathbf{G}_{\alpha \gamma}(x,t) \coloneq \textup{Cov}(Z_\alpha, Z_\gamma \mid X_t = x) \;\; \text{and} \;\; \mathbf{H}_\alpha(x,t) \coloneq \mathbb{E}[Z_\alpha \mid X_t = x] \, .
\end{aligned}
```

  and we note that $`\mathbf{G}_{\alpha \gamma}(x,t) \in \mathbb{R}^{d \times d}`$ and $`\mathbf{H}_\alpha(x,t) \in \mathbb{R}^d`$.
  Now one can use a Fourier expansion to write

```math
\begin{aligned}
    \rho_t(x) \, \mathbf{G}_{\alpha \gamma}(x,t) &= \sum_{\beta \in \mathbb{Z}^d}  \tilde{\tilde{\mathbf C}}_{\alpha \beta \gamma}(t) \, e^{i \, \beta \cdot x} \, , \\
    \rho_t(x) \, \mathbf{H}_\alpha(x,t) &= \sum_{\beta \in \mathbb{Z}^d} \tilde{\mathbf{D}}_{\beta \alpha}(t) \, e^{i \, \beta \cdot x} \, .
\end{aligned}
```

  therefore

```math
\begin{aligned}
    & \tilde{\tilde{\mathbf C}}_{\alpha \beta \gamma} \in \mathbb{C}^{d \times d} \quad \text{and} \quad \tilde{\mathbf D}_{\beta \alpha} \in \mathbb{C}^d \, .
\end{aligned}
```

  Plugging these expressions back into the displayed PDE above and equating the Fourier coefficients on both sides we obtain the system

```math
    \sum_\alpha \tilde{\mathbf D}_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \tilde{\mathbf C}_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, ,
```

  for each $`\beta \in \mathbb{Z}^d`$ where, using the Einstein summation convention, we define

```math
\begin{aligned}
    \tilde{C}_{\alpha \beta \gamma}^{k}(t) &\coloneq \mathcal{F} \big[ \, \partial_j \left( \rho_t \, {G}_{\alpha \gamma}^{j k} \right) \, \big](\beta) \\
    &= i \, \beta_j \, \mathcal{F} \big[ \, \rho_t \, {G}_{\alpha \gamma}^{j k}(x,t) \, \big](\beta) \\
    &= i \, \beta_j \, \tilde{\tilde{ C}}_{\alpha \beta \gamma}^{j k}(t) \, ,
\end{aligned}
```

  where the Fourier operator acts componentwise and we have used a renormalization of Grafakos (2008, Proposition 3.1.2.).
  Succinctly, we have $`\tilde{\mathbf{C}}_{\alpha \beta \gamma} \in \mathbb{C}^{d}`$ defined by

```math
    \tilde{\mathbf C}_{\alpha \beta \gamma}(t) = i \, \beta \cdot \tilde{\tilde{\mathbf C}}_{\alpha \beta \gamma}(t) \, .
```

  Finally, setting

```math
\begin{aligned}
    & {\mathbf C}_{\alpha \beta \gamma}(t) \coloneq i \, (2 \pi)^d \,  {\tilde{\mathbf C}}_{\alpha \beta \gamma}(t) \\
    & {\mathbf D}_{\beta \alpha}(t)        \coloneq i \, (2 \pi)^d \, \tilde{\mathbf D}_{\beta \alpha}(t) \, ,
\end{aligned}
```

  we see that the intermediate system above is equivalent to a system that writes

```math
    \sum_\alpha \mathbf D_{\beta \alpha}(t) \, \ddot \psi_\alpha(t) = \sum_{\alpha, \gamma} \dot \psi_\alpha(t) \, \mathbf C_{\alpha \beta \gamma}(t) \, \dot \psi_\gamma(t) \, .
```

  To finish the proof, we need to show that the coefficients $`\mathbf C_{\alpha \beta \gamma}(t)`$ and $`\mathbf D_{\beta \alpha}(t)`$ are as claimed in the statement of the theorem.
  Thus, for $`j \in \{1, \ldots, d\}`$ we compute

```math
\begin{aligned}
    D^j_{\beta \alpha}(t) &= i \, (2 \pi)^d\frac{1}{(2 \pi)^d} \int_{D} \rho_t(x) \, \mathbb{E}[Z^j_\alpha \mid x] \, e^{-i \, \beta \cdot x} \, \mathrm{d} x \\
    &= i \, \mathbb{E} \Big[ \mathbb{E} \big[ Z_\alpha^j | X_t \big] \, e^{-i \, \beta \cdot X_t} \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \cdot X_t} \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \cdot \left( \sum_\gamma \psi_\gamma(t) \, Z_\gamma \right) } \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, \prod_\gamma e^{-i \, \beta \cdot \psi_\gamma(t) \, Z_\gamma} \Big] \\
    &= i \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \cdot \psi_\alpha(t) \, Z_\alpha} \Big] \, \prod_{\gamma \neq \alpha} \mathbb{E} \Big[ e^{-i \, \beta \cdot \psi_\gamma(t) \, Z_\gamma} \Big] \\
    &= \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\gamma \neq \alpha} \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, .
\end{aligned}
```

  where in the second-to-last line we have used independence and in the last line we have used the definition of the characteristic function.

  Similarly, we can compute:

```math
\begin{aligned}
    C_{\alpha \beta \gamma}^{j k}(t) &= \sum_j \beta_j \, i^2 \, (2 \pi)^d \, \frac{1}{(2 \pi)^d} \int_{D} \rho_t(x) \, \textup{Cov}(Z_\alpha^j , Z_\gamma^k \mid x) \, e^{-i \, \beta \cdot x} \, \mathrm{d} x \\
    &= - \sum_j \beta_j \, \mathbb{E} \Big[ \textup{Cov}(Z_\alpha^j, Z_\gamma^k \mid X_t) \, e^{-i \, \beta \cdot X_t} \Big] \\
    &= - \sum_j \beta_j \, \mathbb{E} \Big\{ \Big[ \mathbb{E} \big[ Z_\alpha^j Z_\gamma^k | X_t \big] - \mathbb{E}\big[Z_\alpha^j | X_t \big] \, \mathbb{E}\big[Z_\gamma^k | X_t \big] \Big] e^{-i \, \beta \cdot X_t} \Big\} \\
    &= - \sum_j \beta_j \, \mathbb{E} \Big[ Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \cdot X_t} \Big] + \beta_j \, \mathbb{E} \Big[ \mathbb{E}[Z_\alpha^j | X_t] \, \mathbb{E}[Z_\gamma^k | X_t] e^{-i \, \beta \cdot X_t}  \Big]  \\
\end{aligned}
```

  Now using the independence assumption we can compute the first term as

```math
\begin{aligned}
    \mathbb{E} \Big[ Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \cdot X_t} \Big] &= \mathbb{E} \Big[ Z_\alpha^j \, Z_\gamma^k \, e^{-i \, \beta \cdot \sum_\delta \psi_\delta(t) Z_\delta} \Big] \\
    &= \mathbb{E} \Big[ Z_\alpha^j \, Z_\gamma^k \,  \prod_\delta e^{-i \, \beta \cdot \psi_\delta(t) \, Z_\delta} \Big] \\
    &= \delta_{\alpha \gamma} \, \mathbb{E} \Big[ Z_\alpha^j \, Z_\alpha^k \, e^{-i \, \beta \cdot \psi_\alpha(t) \, Z_\alpha} \Big] \, \prod_{\delta \neq \alpha} \mathbb{E} \Big[ e^{-i \, \beta \cdot \psi_\delta(t) \, Z_\delta} \Big] \\
    &+ (1 - \delta_{\alpha \gamma}) \, \mathbb{E} \Big[ Z_\alpha^j \, e^{-i \, \beta \cdot \psi_\alpha(t) \, Z_\alpha} \Big] \, \mathbb{E} \Big[ Z_\gamma^k \, e^{-i \, \beta \cdot \psi_\gamma(t) \, Z_\gamma} \Big] \, \prod_{\delta \neq \alpha, \gamma} \mathbb{E} \Big[ e^{-i \, \beta \cdot \psi_\delta(t) \, Z_\delta} \Big] \\
    &= - \delta_{\alpha \gamma} \, \partial_j \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (\delta_{\alpha \gamma}-1) \, \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \partial_k \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \, .
\end{aligned}
```

  Finally, we can use the convolution theorem, i.e. the discrete Fourier convolution lemma below, to obtain

```math
\begin{aligned}
    \mathbb{E} \Big[ \mathbb{E}[Z_\alpha^j | X_t] \, \mathbb{E}[Z_\gamma^k | X_t] e^{-i \, \beta \, X_t}  \Big] &= (2 \pi)^d \, \mathcal{F}\Big[ \rho_t \, H_\alpha^j \, H_\gamma^k \Big](\beta) \\
    &= (2 \pi)^d \, \mathcal{F}\left[ \rho_t \, H_\alpha^j \, \frac{1}{\rho_t} \, \rho_t \, H_\gamma^k \right] \\
    &= (2 \pi)^d \, \mathcal{F}[\rho_t \, H_\alpha^j] \ast \mathcal{F}\left[ \frac{1}{\rho_t} \right] \ast \mathcal{F}[\rho_t \, H_\gamma^k] \, ,
\end{aligned}
```

  and note that we have computed

```math
    \mathbf D_{\beta \alpha}(t) = i \, (2 \pi)^d \, \mathcal{F}[\rho_t \, \mathbf H_\alpha](\beta)  \, ,
```

  for any $`\alpha \in \mathbb{N}`$ and $`\beta \in \mathbb{Z}^d`$.
  Thus, setting $`\tilde K_\beta(t) = \mathcal{F}\left[ \frac{1}{\rho_t} \right](\beta)`$ we have

```math
    \mathbb{E} \Big[ \mathbb{E}[Z_\alpha^j | X_t] \, \mathbb{E}[Z_\gamma^k | X_t] e^{-i \, \beta \, X_t}  \Big] = -\frac{1}{(2 \pi)^d} \, \Big( D^j_{\bullet \alpha}(t) \ast \tilde K_\bullet(t) \ast D^k_{\bullet \gamma}(t) \Big)(\beta) \, .
```

  Finally, the kernel $`K`$ can be computed as follows: first, use the Fourier representation theorem above to write

```math
    \rho_t(x) = \sum_{\beta} \mathcal{F}[\rho_t](\beta) \, e^{i \, \beta \cdot x} \, .
```

  Now, compute

```math
\begin{aligned}
      \mathcal{F}[\rho_t](\beta) &= \frac{1}{(2 \pi)^d} \int_{D} e^{-i \, \beta \cdot x} \, \rho_t(x) \, \mathrm{d} x \\
      &= \frac{1}{(2 \pi)^d} \mathbb{E} \left[ e^{-i \, \beta \cdot X_t} \right] \\
      &= \frac{1}{(2 \pi)^d} \prod_\alpha \Phi_\alpha\left( -\beta \, \psi_\alpha(t) \right) \, ,
\end{aligned}
```

  where the last line follows by writing $`X_t = \sum_\alpha \psi_\alpha(t) \, Z_\alpha`$ and using independence.
  Thus, we have

```math
\begin{aligned}
    \tilde K_\beta(t) &= \mathcal{F} \left[ \frac{1}{\sum_\ell \frac{1}{(2 \pi)^d} \prod_\alpha \Phi_\alpha\left( -\ell \, \psi_\alpha(t) \right) \, e^{i \, \ell \cdot x}} \right](\beta) \\
    &= (2 \pi)^d \, \mathcal{F} \left[ \frac{1}{ \sum_\ell \prod_\alpha \Phi_\alpha\left( -\ell \, \psi_\alpha(t) \right) \, e^{i \, \ell \cdot x} } \right](\beta) \\
\end{aligned}
```

  Finally, set

```math
    K_{\beta}(t) \coloneq \frac{1}{(2 \pi)^d} \, \tilde K_{\beta}(t) \, ,
```

  and putting it all together we get the final expression

```math
\begin{aligned}
    C_{\alpha \beta \gamma}^{k}(t)
    = \sum_j \beta_j \, \Big\{ & \delta_{\alpha \gamma} \, \partial_j \partial_k \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \prod_{\delta \neq \alpha} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &+ (\delta_{\alpha \gamma} - 1) \, \partial_j \Phi_\alpha\left( -\beta \psi_\alpha(t) \right) \, \partial_k \Phi_\gamma\left( -\beta \psi_\gamma(t) \right) \, \prod_{\delta \neq \alpha, \gamma} \Phi_\delta\left( -\beta \psi_\delta(t) \right) \\
    &- \Big( D^j_{\bullet \alpha}(t) \ast K_\bullet(t) \ast D^k_{\bullet \gamma}(t) \Big)(\beta) \Big\} .
\end{aligned}
```

**Lemma.**
  Fix $`D = \mathbb{R}^d / \left(2\pi \mathbb{Z}^d\right)`$.
  For functions $`f, g: D \to \mathbb{R}`$ with absolutely convergent Fourier series we have we have

```math
    \mathcal{F}[f \, g](\beta) = \sum_{\alpha} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\beta - \alpha) \, .
```

**Proof.**

  We compute

```math
    \mathcal{F}[f \, g](\beta) = \frac{1}{(2 \pi)^d} \int_{D} e^{-i \, \beta \cdot x} \, f(x) \, g(x) \, \mathrm{d} x \, .
```

  and by expanding $`f`$ and $`g`$ in Fourier series we obtain

```math
\begin{aligned}
    \mathcal{F}[f \, g](\beta) &= \frac{1}{(2\pi)^d} \int_{D} e^{-i \, \beta \cdot x} \, \left( \sum_{\alpha} \mathcal{F}[f](\alpha) \, e^{i \, \alpha \cdot x} \right) \, \left( \sum_{\gamma} \mathcal{F}[g](\gamma) \, e^{i \, \gamma \cdot x} \right) \, \mathrm{d} x \\
    &= \sum_{\alpha, \gamma} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\gamma) \,\frac{1}{(2\pi)^d} \int_{D} e^{i \, (\alpha + \gamma - \beta) \cdot x} \, \mathrm{d} x \\
    &= \sum_{\alpha, \gamma} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\gamma) \, \mathbf{1}_{\alpha + \gamma = \beta} \\
    &= \sum_{\alpha} \mathcal{F}[f](\alpha) \, \mathcal{F}[g](\beta - \alpha) \, .
\end{aligned}
```
