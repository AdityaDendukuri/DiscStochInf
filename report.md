
# Learning Chemical Master Equation Generators from Trajectory Data: Methodology

**For review by Prof. Igor Mezić**

## 1. Overview

We present a gradient-based optimization framework for learning the generator matrix of the Chemical Master Equation (CME) from stochastic trajectory data. The method uses Fréchet derivatives to compute analytical gradients of the matrix exponential, enabling efficient parameter estimation through quasi-Newton optimization. The approach connects to Koopman operator theory through the finite-dimensional generator framework and provides a bridge between stochastic dynamics on discrete state spaces and continuous-time linear evolution of probability distributions.

---

## 2. Problem Formulation

### 2.1 The Chemical Master Equation

The CME describes the time evolution of the probability distribution $\mathbf{p}(t) \in \mathbb{R}^{|\mathcal{X}|}$ over a discrete, countably infinite state space $\mathcal{X} \subset \mathbb{Z}_+^d$:

$$\frac{d\mathbf{p}(t)}{dt} = \mathbf{A} \mathbf{p}(t)$$

where $\mathbf{A} \in \mathbb{R}^{|\mathcal{X}| \times |\mathcal{X}|}$ is the infinitesimal generator (or rate matrix). The formal solution is given by the matrix exponential:

$$\mathbf{p}(t) = e^{\mathbf{A}t} \mathbf{p}(0) = \sum_{k=0}^{\infty} \frac{(\mathbf{A}t)^k}{k!} \mathbf{p}(0)$$

This is a **continuous-time Markov chain (CTMC)** on a discrete state space, distinct from the continuous state-space setting typically studied in Koopman operator theory.

### 2.2 The Inverse Problem

**Given:** A sequence of empirical probability distributions $\{\hat{\mathbf{p}}(t_1), \hat{\mathbf{p}}(t_2), \ldots, \hat{\mathbf{p}}(t_N)\}$ obtained from $M$ independent stochastic simulation trajectories.

**Find:** The generator matrix $\mathbf{A}$ that best explains the observed probability evolution.

**Mathematical formulation:** For consecutive time points separated by $\Delta t = t_{i+1} - t_i$, we seek $\mathbf{A}$ such that:

$$\mathbf{p}(t_{i+1}) \approx e^{\mathbf{A} \Delta t} \mathbf{p}(t_i)$$

This can be reformulated as finding the generator of the discrete-time transition semigroup $\{\mathbf{P}(t)\}_{t \geq 0}$ where:

$$\mathbf{P}(\Delta t) = e^{\mathbf{A} \Delta t}$$

### 2.3 Generator Structure and Constraints

The generator $\mathbf{A}$ must satisfy the following structural constraints to ensure it generates a valid Markov semigroup:

**Definition 2.1 (M-matrix structure):** A matrix $\mathbf{A}$ is an M-matrix (or intensity matrix) if:

1. **Off-diagonal non-negativity:** 
   $$A_{ij} \geq 0 \quad \forall i \neq j$$
   
2. **Zero column sums (probability conservation):**
   $$\sum_{i=1}^{|\mathcal{X}|} A_{ij} = 0 \quad \forall j$$
   
   Equivalently, $\mathbf{A}\mathbf{1} = \mathbf{0}$ where $\mathbf{1}$ is the vector of ones.

3. **Diagonal negativity:**
   $$A_{jj} = -\sum_{i \neq j} A_{ij} \leq 0$$

**Proposition 2.1:** If $\mathbf{A}$ is an M-matrix, then $e^{\mathbf{A}t}$ is a stochastic matrix for all $t \geq 0$:

$$e^{\mathbf{A}t} \mathbf{1} = \mathbf{1}, \quad [e^{\mathbf{A}t}]_{ij} \geq 0$$

**Proof sketch:** The condition $\mathbf{A}\mathbf{1} = \mathbf{0}$ implies $\mathbf{1}$ is a right eigenvector with eigenvalue 0. By the exponential series:

\begin{align}
e^{\mathbf{A}t}\mathbf{1} &= \sum_{k=0}^{\infty} \frac{(\mathbf{A}t)^k}{k!} \mathbf{1} \\
&= \sum_{k=0}^{\infty} \frac{t^k}{k!} \mathbf{A}^k \mathbf{1} \\
&= \mathbf{1} + \sum_{k=1}^{\infty} \frac{t^k}{k!} \cdot \mathbf{0} = \mathbf{1}
\end{align}

Non-negativity follows from the Perron-Frobenius theory for eventually positive matrices. $\square$

**Stoichiometric structure:** For chemical reaction networks, an additional constraint arises from the stoichiometry. The generator has the form:

$$A_{ij} = \sum_{k=1}^{R} \lambda_k(\mathbf{x}_j) \cdot \delta(\mathbf{x}_i = \mathbf{x}_j + \boldsymbol{\nu}_k)$$

where:
- $R$ is the number of reactions
- $\boldsymbol{\nu}_k \in \mathbb{Z}^d$ is the stoichiometric vector for reaction $k$
- $\lambda_k: \mathcal{X} \to \mathbb{R}_+$ is the propensity (rate) function
- $\delta(\cdot)$ is the indicator function

This gives $\mathbf{A}$ a highly sparse structure determined entirely by the reaction network topology.

---

## 3. Propensity Parameterization

### 3.1 Stoichiometric Decomposition

The generator can be written as:

$$\mathbf{A} = \sum_{k=1}^{R} \mathbf{A}_k$$

where each $\mathbf{A}_k$ corresponds to reaction $k$:

$$[\mathbf{A}_k]_{ij} = \begin{cases}
\lambda_k(\mathbf{x}_j) & \text{if } \mathbf{x}_i = \mathbf{x}_j + \boldsymbol{\nu}_k \\
-\lambda_k(\mathbf{x}_j) & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

This decomposition has several important properties:

**Proposition 3.1:** Each $\mathbf{A}_k$ is an M-matrix if and only if $\lambda_k(\mathbf{x}) \geq 0$ for all $\mathbf{x} \in \mathcal{X}$.

**Proposition 3.2:** The sparsity pattern of $\mathbf{A}$ is completely determined by the stoichiometric vectors $\{\boldsymbol{\nu}_k\}_{k=1}^R$. The number of nonzero entries per column is at most $2R + 1$.

### 3.2 Polynomial Basis for Propensity Functions

We parameterize each propensity function using a polynomial feature basis. For a state $\mathbf{x} = (x_1, \ldots, x_d) \in \mathcal{X}$:

$$\lambda_k(\mathbf{x}) = \max\left(0, \boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x})\right)$$

where $\boldsymbol{\theta}_k \in \mathbb{R}^{n_f}$ are the parameters for reaction $k$, and $\mathbf{f}: \mathcal{X} \to \mathbb{R}^{n_f}$ is a vector of polynomial features.

**For two-species systems** ($d = 2$, $\mathbf{x} = (x, y)$):

**Quadratic basis** ($n_f = 6$):
$$\mathbf{f}(x, y) = \begin{bmatrix} 1 \\ x \\ y \\ xy \\ x^2 \\ y^2 \end{bmatrix}$$

**Linear basis** ($n_f = 3$):
$$\mathbf{f}(x, y) = \begin{bmatrix} 1 \\ x \\ y \end{bmatrix}$$

The total parameter vector is:

$$\boldsymbol{\theta} = \begin{bmatrix} \boldsymbol{\theta}_1 \\ \boldsymbol{\theta}_2 \\ \vdots \\ \boldsymbol{\theta}_R \end{bmatrix} \in \mathbb{R}^{R \cdot n_f}$$

**Motivation:** Common propensity forms in chemical kinetics:

\begin{align}
\text{Mass-action:} \quad &\lambda(x, y) = c \cdot x^{\alpha} y^{\beta} \\
\text{Michaelis-Menten:} \quad &\lambda(x, y) = \frac{V_{\max} \cdot x}{K_M + x} \approx \theta_0 + \theta_1 x + \theta_2 x^2 \\
\text{Hill function:} \quad &\lambda(x, y) = \frac{V_{\max} \cdot x^n}{K^n + x^n}
\end{align}

Polynomial bases can approximate these forms locally using Taylor expansion.

### 3.3 Generator Construction from Parameters

Given $\boldsymbol{\theta}$, the generator is constructed as:

$$\mathbf{A}(\boldsymbol{\theta}) = \sum_{k=1}^{R} \mathbf{A}_k(\boldsymbol{\theta}_k)$$

where for each state $j$ corresponding to $\mathbf{x}_j \in \mathcal{X}$:

\begin{align}
[\mathbf{A}_k(\boldsymbol{\theta}_k)]_{ij} &= \begin{cases}
\max(0, \boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}_j)) & \text{if } \mathbf{x}_i = \mathbf{x}_j + \boldsymbol{\nu}_k \\
-\max(0, \boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}_j)) & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
\end{align}

**Note on non-negativity:** The $\max(0, \cdot)$ operation ensures $\lambda_k \geq 0$. This introduces a non-differentiability at the boundary $\boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}) = 0$, but in practice the optimization stays in the interior due to the data-fitting term.

---

## 4. Data Preparation and State Space Construction

### 4.1 Histogram Construction from Trajectories

We generate $M$ independent SSA trajectories $\{\mathbf{X}^{(m)}(t)\}_{m=1}^M$ using Gillespie's algorithm. Each trajectory is a piecewise-constant function:

$$\mathbf{X}^{(m)}: [0, T] \to \mathcal{X}$$

At discrete time points $t_1, t_2, \ldots, t_N$ with spacing $\Delta t$, we construct empirical probability distributions:

$$\hat{\mathbf{p}}(\mathbf{x}, t_i) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}_{\{\mathbf{X}^{(m)}(t_i) = \mathbf{x}\}}$$

This is the empirical histogram estimator. By the law of large numbers:

$$\hat{\mathbf{p}}(\mathbf{x}, t_i) \xrightarrow{M \to \infty} \mathbf{p}(\mathbf{x}, t_i) \quad \text{a.s.}$$

**Sampling error bound:** For fixed $\mathbf{x}$ and $t_i$, if $p(\mathbf{x}, t_i) = p$, then by Hoeffding's inequality:

$$\mathbb{P}\left(|\hat{p} - p| > \epsilon\right) \leq 2e^{-2M\epsilon^2}$$

For the total variation distance:

$$\mathbb{E}[\|\hat{\mathbf{p}} - \mathbf{p}\|_1] \leq \sqrt{\frac{|\mathcal{X}|}{2M}}$$

assuming all states are visited with non-negligible probability.

### 4.2 Stoichiometry Extraction

We extract stoichiometric vectors from the observed state-to-state transitions in the trajectory ensemble. For each trajectory $m$ and time interval $[t, t+dt]$:

$$\boldsymbol{\nu}^{(m)}(t) = \mathbf{X}^{(m)}(t + dt) - \mathbf{X}^{(m)}(t)$$

We aggregate these transitions and count frequencies:

$$c(\boldsymbol{\nu}) = \sum_{m=1}^{M} \sum_{i} \mathbf{1}_{\{\boldsymbol{\nu}^{(m)}(t_i) = \boldsymbol{\nu}\}}$$

The stoichiometric basis $\{\boldsymbol{\nu}_1, \ldots, \boldsymbol{\nu}_R\}$ consists of the $R$ most frequent non-zero stoichiometries satisfying $c(\boldsymbol{\nu}_k) > c_{\min}$ for some threshold $c_{\min}$.

**Rationale:** In the limit of fine time discretization, each SSA jump corresponds to a single reaction firing, revealing the true stoichiometry. For finite $\Delta t$, multiple reactions may occur, creating composite stoichiometries that we filter by frequency.

### 4.3 Finite State Projection (FSP) Truncation

For computational tractability, we truncate the infinite state space $\mathcal{X}$ to a finite subset. For each window pair $(t_i, t_{i+1})$, we define the **local state space**:

$$\mathcal{S}_i = \text{supp}(\hat{\mathbf{p}}(t_i)) \cup \text{supp}(\hat{\mathbf{p}}(t_{i+1}))$$

where $\text{supp}(\mathbf{p}) = \{\mathbf{x} \in \mathcal{X} : p(\mathbf{x}) > 0\}$.

This is a **data-driven adaptive truncation** following the FSP philosophy: only track states that carry non-negligible probability.

**Key property:** The state space grows over time as the distribution spreads:

$$|\mathcal{S}_1| \leq |\mathcal{S}_2| \leq \cdots \leq |\mathcal{S}_N|$$

typically growing from $|\mathcal{S}_1| \sim 50$ to $|\mathcal{S}_N| \sim 100-200$ for biochemical systems over long time horizons.

**Connection to FSP error theory:** The truncation error is bounded by the probability flux exiting the truncated space. For a subset $\mathcal{J} \subset \mathcal{X}$ with complement $\mathcal{J}' = \mathcal{X} \setminus \mathcal{J}$:

$$\left\|\mathbf{p}_{\mathcal{J}}(t) - \mathbf{p}_{\mathcal{J}}^{\text{FSP}}(t)\right\|_1 \leq \epsilon_0 + \int_0^t J_{\text{out}}(s) \, ds$$

where $J_{\text{out}}(s) = \mathbf{1}_{\mathcal{J}'}^\top \mathbf{A}_{\mathcal{J}'\mathcal{J}} \mathbf{p}_{\mathcal{J}}(s)$ is the boundary flux.

---

## 5. Optimization Objective

### 5.1 Prediction Error Term

The primary objective measures how well the learned generator predicts probability evolution. For a single window pair:

$$\mathcal{L}_{\text{pred}}(\boldsymbol{\theta}) = \left\| \mathbf{p}_{i+1} - e^{\mathbf{A}(\boldsymbol{\theta}) \Delta t} \mathbf{p}_i \right\|_1$$

We use the $L^1$ (total variation) norm:

$$\|\mathbf{v}\|_1 = \sum_{j=1}^{|\mathcal{S}|} |v_j|$$

**Motivation for $L^1$ norm:**

1. **Probabilistic interpretation:** The $L^1$ distance between probability distributions is the total variation distance, which has the interpretation:

   $$\|\mathbf{p} - \mathbf{q}\|_1 = 2 \sup_{A \subseteq \mathcal{X}} |p(A) - q(A)|$$

2. **Robustness:** Less sensitive to outliers than $L^2$ norm

3. **Sparsity:** Naturally handles sparse probability distributions (most states have $p(\mathbf{x}) = 0$)

**Alternative norms:**

- **$L^2$ (Euclidean):** $\|\mathbf{p} - \mathbf{q}\|_2 = \sqrt{\sum_j (p_j - q_j)^2}$. More sensitive to large deviations.

- **KL divergence:** $D_{KL}(\mathbf{p} \| \mathbf{q}) = \sum_j p_j \log(p_j/q_j)$. Not symmetric, undefined when $q_j = 0$.

- **Wasserstein distance:** Exploits metric structure on $\mathcal{X}$, but computationally expensive.

### 5.2 Regularization Terms

**Frobenius norm regularization:**

$$\mathcal{R}_1(\boldsymbol{\theta}) = \lambda_1 \|\mathbf{A}(\boldsymbol{\theta})\|_F^2 = \lambda_1 \sum_{i,j} A_{ij}^2$$

This penalizes large generator entries, providing:

1. **Numerical stability:** Large $\|\mathbf{A}\|$ leads to ill-conditioned matrix exponentials
2. **Overfitting prevention:** Limits model complexity
3. **Gradient regularization:** Smooths the loss landscape

**Semigroup constraint penalty:**

$$\mathcal{R}_2(\boldsymbol{\theta}) = \lambda_2 \sum_{j=1}^{|\mathcal{S}|} \left(\sum_{i=1}^{|\mathcal{S}|} A_{ij}\right)^2$$

This is a **soft constraint** enforcing the semigroup property $\mathbf{A}\mathbf{1} = \mathbf{0}$.

**Why soft instead of hard constraint?** 

Hard constraint: $\mathbf{A}\mathbf{1} = \mathbf{0}$ exactly
- Requires constrained optimization (projected gradient, Lagrange multipliers)
- Difficult to combine with quasi-Newton methods

Soft constraint: Penalty drives column sums toward zero
- Unconstrained optimization (standard L-BFGS)
- Flexible (allows small violations during optimization)
- Empirically achieves $\|\mathbf{A}\mathbf{1}\|_\infty < 10^{-6}$ at convergence

### 5.3 Full Composite Objective

Combining all terms, for $N$ window pairs:

$$J(\boldsymbol{\theta}) = \sum_{i=1}^{N} \left\| \mathbf{p}_{i+1} - e^{\mathbf{A}(\boldsymbol{\theta}) \Delta t} \mathbf{p}_i \right\|_1 + \lambda_1 \|\mathbf{A}(\boldsymbol{\theta})\|_F^2 + \lambda_2 \sum_{j} \left(\sum_{i} A_{ij}\right)^2$$

**Design choices:**

- **Single $\mathbf{A}$ for all windows:** Assumes time-invariant dynamics (homogeneous CTMC)
- **Summation over windows:** Each window contributes independently (assumes statistical independence)
- **Regularization weights:** Typically $\lambda_1, \lambda_2 \in [10^{-8}, 10^{-5}]$

**Connection to maximum likelihood:** In the limit of small $\Delta t$ and many windows, this objective approximates the negative log-likelihood of the CTMC given observed transitions.

---

## 6. Gradient Computation via Fréchet Derivatives

### 6.1 The Gradient Challenge

The key computational challenge is computing:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \left\| \mathbf{p}_{next} - e^{\mathbf{A}(\boldsymbol{\theta}) \Delta t} \mathbf{p}_{curr} \right\|_1 + \text{regularization terms}$$

This requires differentiating through the matrix exponential:

$$\frac{\partial}{\partial \theta_{kf}} e^{\mathbf{A}(\boldsymbol{\theta}) \Delta t}$$

**Naive approaches fail:**

1. **Finite differences:** 
   $$\frac{\partial e^{\mathbf{A}t}}{\partial \theta} \approx \frac{e^{(\mathbf{A} + \epsilon \mathbf{E})t} - e^{\mathbf{A}t}}{\epsilon}$$
   
   Problems: 
   - Requires $2 \cdot R \cdot n_f$ matrix exponential computations
   - Numerically unstable (choosing $\epsilon$)
   - Error is $O(\epsilon)$

2. **Automatic differentiation:** Treating $e^{\mathbf{A}t} = \sum_{k=0}^{\infty} \frac{(\mathbf{A}t)^k}{k!}$ as black-box
   
   Problems:
   - Infinite series approximation
   - Computational graph explosion
   - Loss of structure

**Our approach:** Analytical gradients via **Fréchet derivatives**.

### 6.2 Fréchet Derivative Theory

**Definition 6.1 (Fréchet derivative):** The Fréchet derivative $\mathcal{L}(\mathbf{A}, \mathbf{E})$ of the matrix exponential is defined as:

$$\mathcal{L}(\mathbf{A}, \mathbf{E}) = \frac{d}{d\varepsilon}\Big|_{\varepsilon=0} e^{\mathbf{A} + \varepsilon \mathbf{E}}$$

where $\mathbf{A}, \mathbf{E} \in \mathbb{R}^{n \times n}$.

**Theorem 6.1 (Daleckii-Krein formula):** The Fréchet derivative satisfies:

$$\mathcal{L}(\mathbf{A}, \mathbf{E}) = \int_0^1 e^{\mathbf{A}s} \mathbf{E} e^{\mathbf{A}(1-s)} \, ds$$

**Proof:** By definition:

$$
\begin{align}
\mathcal{L}(\mathbf{A}, \mathbf{E}) &= \lim_{\varepsilon \to 0} \frac{e^{\mathbf{A} + \varepsilon \mathbf{E}} - e^{\mathbf{A}}}{\varepsilon} \\
&= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \left(\sum_{k=0}^{\infty} \frac{(\mathbf{A} + \varepsilon \mathbf{E})^k}{k!} - \sum_{k=0}^{\infty} \frac{\mathbf{A}^k}{k!}\right)
\end{align}
$$

Using the binomial expansion:

$$(\mathbf{A} + \varepsilon \mathbf{E})^k = \sum_{j=0}^{k} \binom{k}{j} \mathbf{A}^{k-j} (\varepsilon \mathbf{E})^j$$

The linear term in $\varepsilon$ is:

$$\sum_{k=1}^{\infty} \frac{k}{k!} \sum_{j=0}^{k-1} \mathbf{A}^j \mathbf{E} \mathbf{A}^{k-1-j}$$

Rearranging and summing:

$$
\begin{align}
\mathcal{L}(\mathbf{A}, \mathbf{E}) &= \sum_{m=0}^{\infty} \sum_{n=0}^{\infty} \frac{\mathbf{A}^m \mathbf{E} \mathbf{A}^n}{(m+n+1)!} \\
&= \sum_{k=0}^{\infty} \sum_{m=0}^{k} \frac{\mathbf{A}^m \mathbf{E} \mathbf{A}^{k-m}}{(k+1)!} \\
&= \int_0^1 \left(\sum_{m=0}^{\infty} \frac{(\mathbf{A}s)^m}{m!}\right) \mathbf{E} \left(\sum_{n=0}^{\infty} \frac{(\mathbf{A}(1-s))^n}{n!}\right) ds \\
&= \int_0^1 e^{\mathbf{A}s} \mathbf{E} e^{\mathbf{A}(1-s)} \, ds
\end{align}
$$

$\square$

**Corollary 6.1:** For time-scaled matrices:

$$\mathcal{L}(\mathbf{A}t, \mathbf{E}t) = t \int_0^1 e^{\mathbf{A}ts} \mathbf{E} e^{\mathbf{A}t(1-s)} \, ds$$

**Differential equation characterization:**

**Proposition 6.2:** $\mathcal{L}(\mathbf{A}t, \mathbf{E}t)$ satisfies the differential equation:

$$\frac{d\mathbf{L}(t)}{dt} = \mathbf{A}\mathbf{L}(t) + \mathbf{L}(t)\mathbf{A}, \quad \mathbf{L}(0) = \mathbf{E}$$

where $\mathbf{L}(t) = \mathcal{L}(\mathbf{A}t, \mathbf{E}t)$.

**Proof:** Differentiate the Daleckii-Krein formula:

$$
\begin{align}
\frac{d\mathbf{L}(t)}{dt} &= \frac{d}{dt}\left[t \int_0^1 e^{\mathbf{A}ts} \mathbf{E} e^{\mathbf{A}t(1-s)} \, ds\right] \\
&= \int_0^1 e^{\mathbf{A}ts} \mathbf{E} e^{\mathbf{A}t(1-s)} \, ds \\ &+ t \int_0^1 (\mathbf{A}s e^{\mathbf{A}ts} \mathbf{E} e^{\mathbf{A}t(1-s)} + e^{\mathbf{A}ts} \mathbf{E} \mathbf{A}(1-s) e^{\mathbf{A}t(1-s)}) \, ds
\end{align}
$$

Simplifying using integration by parts and rearranging yields:

$$\frac{d\mathbf{L}(t)}{dt} = \mathbf{A}\mathbf{L}(t) + \mathbf{L}(t)\mathbf{A}$$

$\square$

### 6.3 Block Exponential Method for Computation

**Theorem 6.2 (Al-Mohy & Higham, 2009):** The Fréchet derivative can be computed via a block matrix exponential:

$$\exp\left(\begin{bmatrix} \mathbf{A}t & \mathbf{E}t \\ \mathbf{0} & \mathbf{A}t \end{bmatrix}\right) = \begin{bmatrix} e^{\mathbf{A}t} & \mathcal{L}(\mathbf{A}t, \mathbf{E}t) \\ \mathbf{0} & e^{\mathbf{A}t} \end{bmatrix}$$

**Proof:** Let $\mathbf{M}(t) = \begin{bmatrix} \mathbf{A}t & \mathbf{E}t \\ \mathbf{0} & \mathbf{A}t \end{bmatrix}$. The matrix exponential $e^{\mathbf{M}}$ can be computed by solving:

$$\frac{d}{dt} e^{\mathbf{M}(t)} = \mathbf{M}(1) e^{\mathbf{M}(t)}, \quad e^{\mathbf{M}(0)} = \mathbf{I}$$

Writing $e^{\mathbf{M}(t)} = \begin{bmatrix} \mathbf{U}(t) & \mathbf{V}(t) \\ \mathbf{0} & \mathbf{W}(t) \end{bmatrix}$:

$$
\begin{align}
\frac{d}{dt}\begin{bmatrix} \mathbf{U} & \mathbf{V} \\ \mathbf{0} & \mathbf{W} \end{bmatrix} &= \begin{bmatrix} \mathbf{A} & \mathbf{E} \\ \mathbf{0} & \mathbf{A} \end{bmatrix} \begin{bmatrix} \mathbf{U} & \mathbf{V} \\ \mathbf{0} & \mathbf{W} \end{bmatrix}
\end{align}
$$

This gives three coupled ODEs:

$$
\begin{align}
\frac{d\mathbf{U}}{dt} &= \mathbf{A}\mathbf{U}, \quad \mathbf{U}(0) = \mathbf{I} \\
\frac{d\mathbf{W}}{dt} &= \mathbf{A}\mathbf{W}, \quad \mathbf{W}(0) = \mathbf{I} \\
\frac{d\mathbf{V}}{dt} &= \mathbf{A}\mathbf{V} + \mathbf{E}\mathbf{W}, \quad \mathbf{V}(0) = \mathbf{0}
\end{align}
$$

The first two give $\mathbf{U}(t) = \mathbf{W}(t) = e^{\mathbf{A}t}$. Substituting into the third:

$$\frac{d\mathbf{V}}{dt} = \mathbf{A}\mathbf{V} + \mathbf{E}e^{\mathbf{A}t}$$

Multiplying both sides by $e^{-\mathbf{A}t}$:

$$e^{-\mathbf{A}t}\frac{d\mathbf{V}}{dt} - e^{-\mathbf{A}t}\mathbf{A}\mathbf{V} = \mathbf{E}$$

The left side is $\frac{d}{dt}(e^{-\mathbf{A}t}\mathbf{V})$, so:

$$e^{-\mathbf{A}t}\mathbf{V}(t) = \int_0^t \mathbf{E} \, ds = t\mathbf{E}$$

Therefore:

$$\mathbf{V}(1) = e^{\mathbf{A}} \mathbf{E} = \mathcal{L}(\mathbf{A}, \mathbf{E})$$

$\square$

**Computational complexity:** Computing the $(2n) \times (2n)$ block exponential costs approximately $8 \times$ the cost of the $n \times n$ exponential using standard scaling-and-squaring algorithms. However, specialized implementations can reduce this to approximately $3 \times$ by exploiting the block structure.

### 6.4 Perturbation Directions and Chain Rule

For each parameter $\theta_{kf}$ (reaction $k$, feature $f$), we need:

$$\frac{\partial \mathbf{A}(\boldsymbol{\theta})}{\partial \theta_{kf}} = \mathbf{E}_{kf}$$

From the generator construction in Section 3.3:

$$\mathbf{A}(\boldsymbol{\theta}) = \sum_{k=1}^{R} \mathbf{A}_k(\boldsymbol{\theta}_k)$$

where:

$$[\mathbf{A}_k]_{ij}(\boldsymbol{\theta}_k) = \begin{cases}
\max(0, \boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}_j)) & \text{if } \mathbf{x}_i = \mathbf{x}_j + \boldsymbol{\nu}_k \\
-\max(0, \boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}_j)) & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

Taking the derivative (assuming we're in the interior where $\boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}) > 0$):

$$\frac{\partial [\mathbf{A}_k]_{ij}}{\partial \theta_{kf}} = \begin{cases}
f(\mathbf{x}_j) & \text{if } \mathbf{x}_i = \mathbf{x}_j + \boldsymbol{\nu}_k \\
-f(\mathbf{x}_j) & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

where $f(\mathbf{x}_j)$ is the $f$-th component of the feature vector $\mathbf{f}(\mathbf{x}_j)$.

**Definition 6.2 (Perturbation matrix):** The perturbation matrix $\mathbf{E}_{kf} \in \mathbb{R}^{n \times n}$ for parameter $\theta_{kf}$ is:

$$[\mathbf{E}_{kf}]_{ij} = \begin{cases}
[\mathbf{f}(\mathbf{x}_j)]_f & \text{if } \mathbf{x}_i = \mathbf{x}_j + \boldsymbol{\nu}_k \\
0 & \text{otherwise}
\end{cases}$$

$$[\mathbf{E}_{kf}]_{jj} = -\sum_{i \neq j} [\mathbf{E}_{kf}]_{ij}$$

**Key properties:**

1. $\mathbf{E}_{kf}$ preserves M-matrix structure: $\mathbf{E}_{kf}\mathbf{1} = \mathbf{0}$
2. $\mathbf{E}_{kf}$ has the same sparsity pattern as $\mathbf{A}_k$
3. $\mathbf{E}_{kf}$ is state-dependent through the features $\mathbf{f}(\mathbf{x})$

### 6.5 Gradient of the $L^1$ Prediction Error

For a single window, the prediction error is:

$$\mathcal{L}(\boldsymbol{\theta}) = \|\mathbf{r}\|_1 = \sum_{j=1}^{n} |r_j|$$

where $\mathbf{r} = \mathbf{p}_{next} - e^{\mathbf{A}(\boldsymbol{\theta})\Delta t} \mathbf{p}_{curr}$.

The $L^1$ norm is not differentiable at zero, but has a subgradient:

$$\frac{\partial}{\partial r_j} |r_j| = \begin{cases}
+1 & \text{if } r_j > 0 \\
-1 & \text{if } r_j < 0 \\
[-1, 1] & \text{if } r_j = 0
\end{cases}$$

For practical purposes, we use the **sign function** (choosing 0 at the origin):

$$\text{sign}(r) = \begin{cases}
+1 & \text{if } r > 0 \\
-1 & \text{if } r < 0 \\
0 & \text{if } r = 0
\end{cases}$$

**Proposition 6.3 (Chain rule for $L^1$ gradient):** The gradient of the prediction error with respect to $\theta_{kf}$ is:

$$\frac{\partial \mathcal{L}}{\partial \theta_{kf}} = -\text{sign}(\mathbf{r})^\top \frac{\partial}{\partial \theta_{kf}} \left[e^{\mathbf{A}\Delta t} \mathbf{p}_{curr}\right]$$

By the chain rule:

$$\frac{\partial}{\partial \theta_{kf}} \left[e^{\mathbf{A}\Delta t} \mathbf{p}_{curr}\right] = \frac{\partial e^{\mathbf{A}\Delta t}}{\partial \theta_{kf}} \mathbf{p}_{curr}$$

Using the Fréchet derivative:

$$\frac{\partial e^{\mathbf{A}\Delta t}}{\partial \theta_{kf}} = \mathcal{L}(\mathbf{A}\Delta t, \mathbf{E}_{kf}\Delta t)$$

Therefore:

$$\boxed{\frac{\partial \mathcal{L}}{\partial \theta_{kf}} = -\text{sign}(\mathbf{r})^\top \mathcal{L}(\mathbf{A}\Delta t, \mathbf{E}_{kf}\Delta t) \mathbf{p}_{curr}}$$

This is the **key gradient formula** for our optimization.

### 6.6 Regularization Gradients

**Frobenius norm gradient:**

$$
\begin{align}
\frac{\partial}{\partial \theta_{kf}} \|\mathbf{A}\|_F^2 &= \frac{\partial}{\partial \theta_{kf}} \sum_{i,j} A_{ij}^2 \\
&= 2\sum_{i,j} A_{ij} \frac{\partial A_{ij}}{\partial \theta_{kf}} \\
&= 2\sum_{i,j} A_{ij} [\mathbf{E}_{kf}]_{ij} \\
&= 2 \text{tr}(\mathbf{A}^\top \mathbf{E}_{kf})
\end{align}
$$

**Semigroup penalty gradient:**

Let $c_j = \sum_{i=1}^{n} A_{ij}$ be the $j$-th column sum. The penalty is:

$$\mathcal{P} = \sum_{j=1}^{n} c_j^2$$

The gradient is:

$$
\begin{align}
\frac{\partial \mathcal{P}}{\partial \theta_{kf}} &= 2\sum_{j=1}^{n} c_j \frac{\partial c_j}{\partial \theta_{kf}} \\
&= 2\sum_{j=1}^{n} c_j \sum_{i=1}^{n} \frac{\partial A_{ij}}{\partial \theta_{kf}} \\
&= 2\sum_{j=1}^{n} c_j \sum_{i=1}^{n} [\mathbf{E}_{kf}]_{ij} \\
&= 2 \mathbf{c}^\top (\mathbf{E}_{kf}^\top \mathbf{1})
\end{align}
$$

where $\mathbf{c} = \mathbf{A}^\top \mathbf{1}$ is the vector of column sums.

### 6.7 Complete Gradient Formula

Combining all terms, the full gradient is:

$$\boxed{\nabla_{\theta_{kf}} J = \sum_{i=1}^{N} \left[-\text{sign}(\mathbf{r}_i)^\top \mathcal{L}(\mathbf{A}\Delta t, \mathbf{E}_{kf}\Delta t) \mathbf{p}_i\right] + 2\lambda_1 \text{tr}(\mathbf{A}^\top \mathbf{E}_{kf}) + 2\lambda_2 \mathbf{c}^\top (\mathbf{E}_{kf}^\top \mathbf{1})}$$

where the sum is over all window pairs.

**Computational procedure for each gradient component:**

1. Build perturbation matrix $\mathbf{E}_{kf}$ (sparse, $O(n)$ time)
2. Compute Fréchet derivative $\mathcal{L}(\mathbf{A}\Delta t, \mathbf{E}_{kf}\Delta t)$ via block exponential ($O(n^3)$ time)
3. Compute inner product with residual ($O(n^2)$ time)
4. Add regularization terms ($O(n^2)$ time)

**Total cost per iteration:** $O(R \cdot n_f \cdot n^3)$ for computing all $R \cdot n_f$ gradient components.

---

## 7. Connection to Koopman Operator Theory

### 7.1 The Koopman Operator for Discrete State Spaces

The **Koopman operator** provides a linear representation of nonlinear dynamics by lifting the evolution to the space of observables.

**Definition 7.1 (Koopman operator for CTMC):** For a continuous-time Markov chain $\{\mathbf{X}(t)\}_{t \geq 0}$ on discrete state space $\mathcal{X}$, the Koopman operator $\mathcal{K}^t: \mathbb{R}^\mathcal{X} \to \mathbb{R}^\mathcal{X}$ is defined by:

$$[\mathcal{K}^t g](\mathbf{x}) = \mathbb{E}[g(\mathbf{X}(t)) \mid \mathbf{X}(0) = \mathbf{x}]$$

for any observable $g: \mathcal{X} \to \mathbb{R}$.

**Proposition 7.1:** The Koopman operator for a CTMC is the dual (transpose) of the probability evolution operator:

$$\mathcal{K}^t = \mathbf{P}(t)^\top = (e^{\mathbf{A}t})^\top$$

**Proof:** By the law of total expectation:

$$
\begin{align}
[\mathcal{K}^t g](\mathbf{x}) &= \sum_{\mathbf{y} \in \mathcal{X}} \mathbb{P}(\mathbf{X}(t) = \mathbf{y} \mid \mathbf{X}(0) = \mathbf{x}) g(\mathbf{y}) \\
&= \sum_{\mathbf{y} \in \mathcal{X}} P(\mathbf{x}, \mathbf{y}; t) g(\mathbf{y}) \\
&= \sum_{\mathbf{y} \in \mathcal{X}} [e^{\mathbf{A}t}]_{\mathbf{x},\mathbf{y}} g(\mathbf{y}) \\
&= [(e^{\mathbf{A}t})^\top \mathbf{g}]_{\mathbf{x}}
\end{align}
$$

where we view $g$ as a column vector. $\square$

**Infinitesimal generator of the Koopman semigroup:**

$$\mathcal{A} = \lim_{t \to 0} \frac{\mathcal{K}^t - I}{t} = \mathbf{A}^\top$$

This is the **dual generator** (or **backward generator**).

### 7.2 Forward vs. Backward Generators

For CTMCs, there are two perspectives on the infinitesimal generator:

**Forward generator (Kolmogorov forward equation):**

$$\frac{\partial \mathbf{p}(t)}{\partial t} = \mathbf{A} \mathbf{p}(t)$$

This describes evolution of **probability distributions** (measures on $\mathcal{X}$).

**Backward generator (Kolmogorov backward equation):**

$$\frac{\partial \mathbb{E}[g(\mathbf{X}(t))]}{\partial t} = \mathbf{A}^\top \mathbf{g}$$

This describes evolution of **observables** (functions on $\mathcal{X}$).

**Our inverse problem learns the forward generator $\mathbf{A}$** from probability evolution, but this is equivalent to learning the Koopman generator $\mathbf{A}^\top$.

### 7.3 DMD and Extended DMD Connection

**Dynamic Mode Decomposition (DMD)** is a data-driven method for approximating Koopman operators. For discrete-time observations:

$$\mathbf{g}_{i+1} \approx \mathbf{K} \mathbf{g}_i$$

where $\mathbf{K} \approx \mathcal{K}^{\Delta t}$.

**Standard DMD:** Finds best-fit linear operator $\mathbf{K}$ via least squares:

$$\min_{\mathbf{K}} \sum_{i=1}^{N-1} \|\mathbf{g}_{i+1} - \mathbf{K}\mathbf{g}_i\|_2^2$$

**Extended DMD (EDMD):** Lifts to nonlinear feature space $\boldsymbol{\psi}: \mathcal{X} \to \mathbb{R}^m$:

$$\boldsymbol{\psi}(\mathbf{x}_{i+1}) \approx \mathbf{K}_{\psi} \boldsymbol{\psi}(\mathbf{x}_i)$$

**Generator Dynamic Mode Decomposition (gDMD):** Approximates the infinitesimal generator:

$$\mathbf{K} \approx e^{\mathcal{A} \Delta t} \quad \Rightarrow \quad \mathcal{A} \approx \frac{\log \mathbf{K}}{\Delta t}$$

**Comparison with our method:**

| Aspect | DMD/gDMD | Our Method |
|--------|----------|------------|
| **Data** | Observables $g(\mathbf{X}(t))$ | Probability distributions $\mathbf{p}(t)$ |
| **Operator** | Koopman $\mathcal{K}^t = (e^{\mathbf{A}t})^\top$ | Transition $\mathbf{P}(t) = e^{\mathbf{A}t}$ |
| **Generator** | Backward $\mathcal{A} = \mathbf{A}^\top$ | Forward $\mathbf{A}$ |
| **Features** | User-chosen $\boldsymbol{\psi}(\mathbf{x})$ | Polynomial $\mathbf{f}(\mathbf{x})$ in propensities |
| **Structure** | Dense $\mathbf{K}$ | Sparse $\mathbf{A}$ (stoichiometry) |
| **Objective** | $L^2$ fit of observables | $L^1$ fit of probabilities |
| **Constraints** | None (may not be stochastic) | M-matrix, semigroup |
| **Gradients** | Closed-form (linear) | Fréchet derivatives (nonlinear) |

**Key difference:** Our method exploits the **sparse stoichiometric structure** of chemical reaction networks, parameterizing $\mathbf{A}$ through propensity functions rather than learning a dense Koopman operator.

### 7.4 Relation to Perron-Frobenius Operator

The **Perron-Frobenius operator** (or **transfer operator**) evolves probability measures. For a CTMC:

$$\mathcal{P}^t \mu = e^{\mathbf{A}t} \mu$$

where $\mu$ is a probability distribution.

**Duality:** The Koopman and Perron-Frobenius operators are **dual** with respect to the $L^2$ inner product:

$$\langle \mathcal{K}^t g, \mu \rangle = \langle g, \mathcal{P}^t \mu \rangle$$

**Our problem:** Learn the infinitesimal generator of the Perron-Frobenius semigroup from snapshots of probability distributions.

This is analogous to learning the Koopman generator from observable time series, but working directly in the **primal** (probability) space rather than the **dual** (observable) space.

### 7.5 Spectral Properties and Ergodicity

The **spectrum** of $\mathbf{A}$ determines long-time behavior:

**Theorem 7.1 (Perron-Frobenius for generators):** For an irreducible, finite-state CTMC generator $\mathbf{A}$:

1. Zero is a simple eigenvalue: $\mathbf{A}\mathbf{\pi} = 0$ where $\boldsymbol{\pi}$ is the stationary distribution
2. All other eigenvalues have strictly negative real parts
3. The spectral gap $\gamma = -\max\{\text{Re}(\lambda) : \lambda \neq 0\}$ controls mixing rate:

$$\|\mathbf{p}(t) - \boldsymbol{\pi}\|_{\text{TV}} \leq C e^{-\gamma t}$$

**Connection to our inverse problem:**

- Learning $\mathbf{A}$ from transient data ($t < 1/\gamma$) may not uniquely determine the spectrum
- Non-identifiability is related to **incomplete spectral information**
- Long-time data provides better constraints on eigenvalues

**Ergodicity and connectivity:** Our flux-based truncation (from Paper 1) preserves ergodicity by maintaining connectivity:

**Proposition 7.2:** If the true generator $\mathbf{A}_{\text{true}}$ is irreducible on $\mathcal{X}$, and the truncation $\mathcal{S} \subset \mathcal{X}$ preserves all high-flux pathways, then the learned generator $\mathbf{A}_{\mathcal{S}}$ restricted to $\mathcal{S}$ is also irreducible.

This ensures the learned dynamics have well-defined ergodic properties.

---

## 8. Optimization Algorithm and Implementation

### 8.1 L-BFGS Quasi-Newton Method

We employ the **Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)** algorithm, a quasi-Newton method that approximates the inverse Hessian using gradient history.

**L-BFGS update formula:** At iteration $k$, the search direction is:

$$\mathbf{d}_k = -\mathbf{H}_k \nabla J(\boldsymbol{\theta}_k)$$

where $\mathbf{H}_k$ is an approximation to the inverse Hessian $[\nabla^2 J]^{-1}$.

The inverse Hessian approximation is updated via:

$$\mathbf{H}_{k+1} = \mathbf{V}_k^\top \mathbf{H}_k \mathbf{V}_k + \rho_k \mathbf{s}_k \mathbf{s}_k^\top$$

where:
- $\mathbf{s}_k = \boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k$ (parameter update)
- $\mathbf{y}_k = \nabla J(\boldsymbol{\theta}_{k+1}) - \nabla J(\boldsymbol{\theta}_k)$ (gradient difference)
- $\rho_k = 1/(\mathbf{y}_k^\top \mathbf{s}_k)$
- $\mathbf{V}_k = \mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^\top$

**Limited memory:** Instead of storing the full $(R \cdot n_f) \times (R \cdot n_f)$ matrix $\mathbf{H}_k$, L-BFGS stores only the last $m$ pairs $\{(\mathbf{s}_i, \mathbf{y}_i)\}_{i=k-m}^{k-1}$, typically $m = 10$.

**Two-loop recursion:** The matrix-vector product $\mathbf{H}_k \nabla J$ is computed efficiently via:

```
Algorithm: L-BFGS Two-Loop Recursion
Input: gradient g = ∇J(θ_k), history {(s_i, y_i)}
Output: search direction d = -H_k g

q ← g
for i = k-1, k-2, ..., k-m do
    α_i ← ρ_i s_i^T q
    q ← q - α_i y_i
end for

r ← H_k^0 q    // Initial inverse Hessian (typically I)

for i = k-m, k-m+1, ..., k-1 do
    β ← ρ_i y_i^T r
    r ← r + s_i (α_i - β)
end for

d ← -r
```

This requires $O(m \cdot R \cdot n_f)$ operations, much cheaper than $O((R \cdot n_f)^2)$ for storing/multiplying full Hessian.

### 8.2 Line Search with Wolfe Conditions

At each iteration, a line search determines the step size $\alpha_k$ along direction $\mathbf{d}_k$:

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \alpha_k \mathbf{d}_k$$

**Strong Wolfe conditions** ensure sufficient decrease and curvature:

1. **Armijo condition (sufficient decrease):**
   $$J(\boldsymbol{\theta}_k + \alpha \mathbf{d}_k) \leq J(\boldsymbol{\theta}_k) + c_1 \alpha \nabla J(\boldsymbol{\theta}_k)^\top \mathbf{d}_k$$
   
   with $c_1 = 10^{-4}$.

2. **Curvature condition:**
   $$|\nabla J(\boldsymbol{\theta}_k + \alpha \mathbf{d}_k)^\top \mathbf{d}_k| \leq c_2 |\nabla J(\boldsymbol{\theta}_k)^\top \mathbf{d}_k|$$
   
   with $c_2 = 0.9$ for quasi-Newton methods.

**Backtracking line search:** Start with $\alpha = 1$, reduce by factor $\tau = 0.5$ until Wolfe conditions satisfied.

The **analytical gradients** via Fréchet derivatives ensure these conditions can be checked exactly without finite-difference approximations.

### 8.3 Interface with Optim.jl

We use the Optim.jl package's "only_fg!" interface for simultaneous objective-gradient computation:

**Function signature:**
```julia
function fg!(F, G, θ)
    if G !== nothing
        obj, grad = compute_objective_and_gradient(θ, data, windows)
        G[:] = grad
    else
        obj = compute_objective(θ, data, windows)
    end
    return obj
end
```

**Key implementation details:**

1. **Conditional gradient computation:** Line search sometimes only needs objective (cheap if we cache exp(A·Δt))

2. **In-place gradient updates:** Writing directly to buffer G avoids allocations

3. **Simultaneous computation:** We compute objective and gradient in single pass, reusing:
   - Matrix exponential $e^{\mathbf{A}\Delta t}$ (used in both objective and gradient)
   - Residual $\mathbf{r} = \mathbf{p}_{next} - e^{\mathbf{A}\Delta t}\mathbf{p}_{curr}$ (used in both)

4. **Caching strategy:**
   ```julia
   if !haskey(cache, (θ, "exp_A"))
       cache[(θ, "exp_A")] = exp(A(θ) * Δt)
   end
   exp_A = cache[(θ, "exp_A")]
   ```

### 8.4 Convergence Criteria

L-BFGS terminates when any of the following conditions is met:

**1. Gradient norm tolerance:**
$$\|\nabla J(\boldsymbol{\theta}_k)\|_\infty < g_{\text{tol}}$$

Default: $g_{\text{tol}} = 10^{-8}$

**2. Relative function change:**
$$\frac{|J(\boldsymbol{\theta}_k) - J(\boldsymbol{\theta}_{k-1})|}{|J(\boldsymbol{\theta}_{k-1})|} < f_{\text{tol}}$$

Default: $f_{\text{tol}} = 10^{-9}$

**3. Maximum iterations:**
$$k \geq k_{\max}$$

Typical: $k_{\max} = 100-300$

**Practical convergence behavior:**

- **Simple systems** (1 window, 50 states, 24 parameters): 50-100 iterations
- **Complex systems** (10 windows, 100+ states): 200-300 iterations
- **Gradient norm** typically plateaus at $10^{-6}$ to $10^{-4}$ before formal convergence
- **Objective decrease** is rapid initially (first 20 iterations), then gradual

### 8.5 Initialization Strategy

Initialization significantly affects convergence, especially given the non-convex landscape.

**Current approach: Small random initialization**
$$\theta_{kf}^{(0)} \sim 0.1 \cdot \mathcal{N}(0, 1)$$

**Rationale:**
- Small values → small initial generator → stable matrix exponential
- Random → breaks symmetry (different reactions have different initial propensities)
- Scale 0.1 → weak initial rates that strengthen during optimization

**Alternative strategies:**

1. **Mass-action initialization:** If stoichiometry suggests $\boldsymbol{\nu}_k = [1, 0]$ (birth reaction), initialize $\theta_k = [c, 0, 0, 0, 0, 0]$ for constant propensity.

2. **Moment-matching initialization:** Use mean/variance of $\mathbf{X}(t)$ to estimate rate constants via moment closure.

3. **Transfer learning:** If solving related systems, initialize from previously learned generators.

4. **Multi-start optimization:** Run from multiple random initializations, keep best result.

### 8.6 Regularization Parameter Selection

The regularization weights $\lambda_1, \lambda_2$ balance data-fitting vs. constraints.

**Default values:**
$$\lambda_1 = 10^{-6}, \quad \lambda_2 = 10^{-6}$$

**Tuning strategy:**

1. **If $\|\mathbf{A}\| > 10^3$:** Increase $\lambda_1$ to $10^{-5}$ or $10^{-4}$

2. **If $\|\mathbf{A}\mathbf{1}\|_\infty > 10^{-4}$:** Increase $\lambda_2$ to $10^{-5}$

3. **If underfitting** (high prediction error): Decrease both to $10^{-7}$ or $10^{-8}$

4. **If overfitting** (perfect fit but unphysical propensities): Increase both

**Cross-validation:** Hold out some windows, tune $\lambda_1, \lambda_2$ to minimize prediction error on held-out data.

---

## 9. Time Step Selection and the Markov Embedding Problem

### 9.1 The Embedding Problem

The **Markov embedding problem** asks: given a stochastic matrix $\mathbf{P}$, does there exist a generator $\mathbf{Q}$ such that $\mathbf{P} = e^{\mathbf{Q}}$?

**Definition 9.1 (Embeddable matrix):** A stochastic matrix $\mathbf{P} \in \mathbb{R}^{n \times n}$ is **embeddable** if there exists a generator $\mathbf{Q}$ (M-matrix) such that $\mathbf{P} = e^{\mathbf{Q}}$. The matrix $\mathbf{Q}$ is called a **Markov generator** of $\mathbf{P}$.

**Historical results:**

- **Elfving (1937):** Formulated the embedding problem
- **Kingman (1962):** Solved for $2 \times 2$ matrices
- **Cuthbert (1973):** Solved for $3 \times 3$ matrices
- **Culver (1966):** Characterized embeddability for matrices with real, distinct eigenvalues

**Recent breakthrough (Casanellas et al., 2020):** Complete solution for $4 \times 4$ matrices.

### 9.2 Necessary Conditions for Embeddability

**Theorem 9.1 (Eigenvalue constraints):** If $\mathbf{P} = e^{\mathbf{Q}}$ where $\mathbf{Q}$ is a generator, then:

1. **Unit leading eigenvalue:** $\mathbf{P}$ has eigenvalue 1 (corresponding to stationary distribution)

2. **Eigenvalue magnitudes:** All eigenvalues $\lambda$ of $\mathbf{P}$ satisfy $|\lambda| \leq 1$

3. **Non-negative logarithm branch:** If $\mathbf{P}$ has real eigenvalues $0 < \lambda_1, \ldots, \lambda_{n-1} < 1$, then these must avoid the negative real axis: $\lambda_i \notin (-\infty, 0]$

**Proof sketch for condition 3:** If $\mathbf{Q}$ is a generator with real eigenvalues $\mu_1, \ldots, \mu_n$ where $\mu_1 = 0$ and $\mu_i < 0$ for $i > 1$, then:

$$\mathbf{P} = e^{\mathbf{Q}} \implies \lambda_i = e^{\mu_i}$$

Since $\mu_i < 0$, we have $0 < \lambda_i = e^{\mu_i} < 1$, so eigenvalues cannot be negative. $\square$

**Theorem 9.2 (Determinant bounds, Israel et al. 2001):** If $\mathbf{P} = e^{\mathbf{Q}}$ and $\det(\mathbf{P}) > 0.5$, then $\mathbf{Q}$ is unique and equals the principal matrix logarithm:

$$\mathbf{Q} = \log(\mathbf{P}) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k}(\mathbf{P} - \mathbf{I})^k$$

**Proof:** If $\det(\mathbf{P}) > 0.5$, then $\|\mathbf{P} - \mathbf{I}\| < 0.5$ in some operator norm, ensuring convergence of the logarithmic series and uniqueness. $\square$

### 9.3 The 4×4 Case: Complete Solution

**Theorem 9.3 (Casanellas et al., 2020):** Let $\mathbf{P} = \mathbf{V} \text{diag}(1, \lambda_1, \lambda_2, \lambda_3) \mathbf{V}^{-1}$ be a $4 \times 4$ stochastic matrix with distinct eigenvalues.

**Case I: All real eigenvalues**

$\mathbf{P}$ is embeddable if and only if $\log(\mathbf{P})$ is a rate matrix.

In this case, the generator is unique: $\mathbf{Q} = \log(\mathbf{P})$.

**Case II: One conjugate pair (λ₂ = μ + iν, λ₃ = μ - iν)**

Define:
$$\mathbf{V} = \mathbf{V} \text{diag}(0, 0, 2\pi i, -2\pi i) \mathbf{V}^{-1}$$

$$L = \max_{(i,j): V_{ij} > 0} \left\{-\frac{\log(\mathbf{P})_{ij}}{V_{ij}}\right\}$$

$$U = \min_{(i,j): V_{ij} < 0} \left\{-\frac{\log(\mathbf{P})_{ij}}{V_{ij}}\right\}$$

$$N = \{(i,j) : V_{ij} = 0 \text{ and } \log(\mathbf{P})_{ij} < 0\}$$

$\mathbf{P}$ is embeddable if and only if $N = \emptyset$ and $L \leq U$.

The Markov generators are: $\mathbf{Q}_k = \log(\mathbf{P}) + k\mathbf{V}$ for $k \in \mathbb{Z}$ with $L \leq k \leq U$.

**Interpretation:** The imaginary part of complex eigenvalues introduces **rotational ambiguity** in the logarithm. Different integer multiples of $2\pi i$ yield different generators, all of which exponentiate to the same transition matrix.

**Implications for CME inverse problems:**

1. **Non-uniqueness is generic:** Even with complete data ($\mathbf{P}$ known exactly), the generator may not be unique

2. **Complex eigenvalues → multiple generators:** Systems with oscillatory dynamics are particularly prone to non-identifiability

3. **Number of generators:** Can range from 1 to unbounded (depending on $U - L$)

### 9.4 Time Step Selection Trade-offs

The choice of $\Delta t$ presents a fundamental tension:

**Small Δt (limit Δt → 0):**

**Advantages:**
- **Linear approximation valid:** $e^{\mathbf{A}\Delta t} \approx \mathbf{I} + \mathbf{A}\Delta t$
- **Elementary reactions:** Less aggregation of multiple reaction events
- **Better conditioning:** $\mathbf{P} \approx \mathbf{I}$ is nearly identity

**Disadvantages:**
- **Poor state space coverage:** Distribution hasn't spread, few states visited
- **Underdetermined system:** $|\mathcal{S}| < R \cdot n_f$ (more parameters than constraints)
- **Insufficient mixing:** Barely any evolution, hard to identify dynamics
- **Numerical issues:** $\mathbf{P} - \mathbf{I}$ is small, amplifies sampling noise

**Large Δt:**

**Advantages:**
- **Good coverage:** Distribution spreads across many states
- **Overdetermined system:** $|\mathcal{S}| \gg R \cdot n_f$ (more constraints than parameters)
- **Strong signal:** Significant probability evolution, clear dynamics

**Disadvantages:**
- **Nonlinear regime:** $e^{\mathbf{A}\Delta t}$ far from linear approximation
- **Reaction aggregation:** Multiple reactions per time step, composite stoichiometries
- **Scaling ambiguity:** Can't distinguish $\alpha \mathbf{A}$ with $\Delta t/\alpha$
- **Non-uniqueness:** Larger $\Delta t$ increases likelihood of non-embeddable $\mathbf{P}$

### 9.5 Empirical Determination of Optimal Δt

From Brusselator experiments:

**Experimental protocol:**
1. Generate trajectories with fixed true generator $\mathbf{A}_{\text{true}}$
2. Compute histograms at various $\Delta t \in \{0.05, 0.1, 0.25, 0.5, 1.0, 2.0\}$
3. Learn generator $\hat{\mathbf{A}}(\Delta t)$ for each $\Delta t$
4. Evaluate: state space size, prediction error, propensity accuracy

**Results:**

| $\Delta t$ | $\|\mathcal{S}\|$ | $\|\|\hat{\mathbf{A}}\|\|$ | Total Rate Error | Assessment |
|-----------|----------|--------------|------------------|------------|
| 0.05 | 7 | N/A | 3-50× | **Failed** (too few states) |
| 0.10 | 10 | ~5000 | 5-20× | Poor (underdetermined) |
| 0.25 | 14 | ~2000 | 10-90× | Poor (underdetermined) |
| 0.50 | 35 | 1200 | 2-8× | Moderate |
| **1.00** | **55** | **784** | **0.97-1.43×** | **Best** |
| 1.50 | 72 | 550 | 1.2-2.5× | Good |
| 2.00 | 85 | 400 | 1.5-3× | Good (slightly oversmooth) |

**Observations:**

1. **Critical threshold:** Need $\Delta t \geq 0.5$ to get $|\mathcal{S}| > 30$ states

2. **Optimal range:** $\Delta t \in [0.75, 1.5]$ for this system (mean inter-event time $\approx 0.3$ sec)

3. **Generator norm decrease:** $\|\mathbf{A}\|$ decreases with $\Delta t$ due to scaling: if $\mathbf{P}(\Delta t) = e^{\mathbf{A}\Delta t}$, then $\|\mathbf{A}\| \propto 1/\Delta t$ approximately

4. **Rate recovery quality:** Best when $|\mathcal{S}|/n_{\text{param}} \approx 2-3$ (mild overspecification)

### 9.6 Theoretical Guidance for Δt Selection

**Heuristic 9.1 (State space size):** Choose $\Delta t$ such that:

$$|\mathcal{S}| \geq 2 \cdot R \cdot n_f$$

This ensures the system is at least **mildly overdetermined**.

**Heuristic 9.2 (Probability coverage):** Choose $\Delta t$ such that:

$$\sum_{\mathbf{x} \in \mathcal{S}} p(\mathbf{x}, t) \geq 0.95$$

for both $t$ and $t + \Delta t$. This ensures the truncation captures most of the probability mass.

**Heuristic 9.3 (Significant evolution):** Choose $\Delta t$ such that:

$$\|\mathbf{p}(t + \Delta t) - \mathbf{p}(t)\|_1 \geq 0.1$$

This ensures sufficient evolution to identify dynamics (avoids $\Delta t \to 0$ regime).

**Heuristic 9.4 (Timescale matching):** For systems with characteristic timescale $\tau$ (e.g., relaxation time):

$$0.5\tau \leq \Delta t \leq 2\tau$$

For the Brusselator with parameters $A=3, B=4$, the characteristic time is $\tau \approx 1/\lambda_{\max} \sim 0.3$ sec, suggesting $\Delta t \in [0.15, 0.6]$. However, this underestimates the optimal value because we need sufficient state space coverage.

**Practical recommendation:** Start with $\Delta t = \text{mean inter-event time}$, then increase until $|\mathcal{S}| \geq 2 \cdot R \cdot n_f$.

---

## 10. Error Analysis and Identifiability

### 10.1 Decomposition of Total Error

The total error in the learned generator arises from multiple sources:

$$\mathbf{A}_{\text{learned}} - \mathbf{A}_{\text{true}} = \epsilon_{\text{sampling}} + \epsilon_{\text{truncation}} + \epsilon_{\text{basis}} + \epsilon_{\text{optimization}} + \epsilon_{\text{embedding}}$$

**1. Sampling error ($\epsilon_{\text{sampling}}$):**

Histograms are empirical estimates: $\hat{\mathbf{p}} \approx \mathbf{p}_{\text{true}}$.

**Bound:** By Hoeffding's inequality for total variation:

$$\mathbb{E}[\|\hat{\mathbf{p}} - \mathbf{p}\|_1] \leq \sqrt{\frac{|\mathcal{X}|}{2M}}$$

where $M$ is the number of trajectories.

For Brusselator with $M = 100$, $|\mathcal{S}| \approx 50$:

$$\mathbb{E}[\text{TV error}] \leq \sqrt{50/200} = 0.5$$

**Reduction:** Increase $M$ (error decreases as $O(1/\sqrt{M})$)

**2. State space truncation error ($\epsilon_{\text{truncation}}$):**

FSP truncation: $\mathcal{S} \subset \mathcal{X}$ (finite vs. infinite).

**Bound:** From FSP theory (Munsky & Khammash, 2006):

$$\|\mathbf{p}_{\mathcal{S}}(t) - \mathbf{p}_{\mathcal{S}}^{\text{FSP}}(t)\|_1 \leq \epsilon_0 + \int_0^t J_{\text{out}}(s) \, ds$$

where $J_{\text{out}}$ is boundary flux.

**Reduction:** Expand $\mathcal{S}$ to include high-flux boundary states (connection to Paper 1)

**3. Propensity basis error ($\epsilon_{\text{basis}}$):**

Polynomial approximation: $\lambda(\mathbf{x}) \approx \boldsymbol{\theta}^\top \mathbf{f}(\mathbf{x})$.

**Example:** True propensity $\lambda(x, y) = \frac{V_{\max} x}{K_M + x}$ is **not** polynomial.

Best polynomial fit (degree 2): 

$$\lambda(x, y) \approx \theta_0 + \theta_1 x + \theta_2 x^2$$

has approximation error depending on domain $[0, x_{\max}]$.

**Reduction:** Increase polynomial degree (higher $n_f$), or use rational/neural parameterizations

**4. Optimization error ($\epsilon_{\text{optimization}}$):**

L-BFGS finds local minimum: $J(\boldsymbol{\theta}^*) \geq \min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$.

**Bound:** Depends on loss landscape curvature, initialization, convergence tolerance.

**Reduction:** Multiple random initializations, tighter convergence criteria

**5. Fundamental embedding error ($\epsilon_{\text{embedding}}$):**

Even with perfect data and infinite basis, the generator may not be uniquely identifiable due to the Markov embedding problem.

**This is irreducible:** Determined by the spectrum of $\mathbf{P}(\Delta t)$.

### 10.2 Identifiability Analysis

**Definition 10.1 (Practical identifiability):** A parameter $\theta_k$ is **practically identifiable** from data if the Fisher information matrix is non-singular:

$$\mathcal{I}(\boldsymbol{\theta}) = \mathbb{E}\left[\nabla_{\boldsymbol{\theta}} \log p(\mathbf{data} \mid \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log p(\mathbf{data} \mid \boldsymbol{\theta})^\top\right] \succ 0$$

For our CME inverse problem, the data likelihood is:

$$p(\{\mathbf{p}_i\}_{i=1}^N \mid \boldsymbol{\theta}) = \prod_{i=1}^{N} p(\mathbf{p}_{i+1} \mid \mathbf{p}_i, \boldsymbol{\theta})$$

where $p(\mathbf{p}_{i+1} \mid \mathbf{p}_i, \boldsymbol{\theta})$ is the probability of observing histogram $\mathbf{p}_{i+1}$ given the generator $\mathbf{A}(\boldsymbol{\theta})$ predicts $e^{\mathbf{A}\Delta t}\mathbf{p}_i$.

**Structural non-identifiability:** Arises from symmetries in the model.

**Example:** For reaction $\emptyset \xrightarrow{\lambda} X$, the propensity is constant: $\lambda(x, y) = c$.

If we parameterize as $\lambda(x, y) = \theta_0 + \theta_1 x + \theta_2 y$, then:
- Only $\theta_0$ is identifiable (constant term)
- $\theta_1$ and $\theta_2$ are structurally non-identifiable (no dependence on $x, y$ in truth)

**Practical non-identifiability:** Even when structurally identifiable, finite data may not provide enough information.

**Example:** For $\lambda(x, y) = c \cdot x$, we can identify "total scaling" $c$ but not separate $c$ from the linear dependence.

### 10.3 What is Well-Identified?

**Empirical finding:** From Brusselator experiments, we observe:

1. **Total exit rates are well-recovered:**
   $$\sum_{k=1}^{R} \lambda_k(\mathbf{x}) \approx \sum_{k=1}^{R} \hat{\lambda}_k(\mathbf{x})$$
   
   Error: 0.97-1.43× (within 43%)

2. **Individual propensities are poorly recovered:**
   $$\lambda_k(\mathbf{x}) \not\approx \hat{\lambda}_k(\mathbf{x})$$
   
   Error: 0.3-5× (up to 500% error)

**Theoretical explanation:**

The CME evolution constrains the **total outflow** from each state:

$$\frac{dp(\mathbf{x}, t)}{dt} = \sum_{\mathbf{y} \neq \mathbf{x}} A(\mathbf{y}, \mathbf{x}) p(\mathbf{y}, t) - p(\mathbf{x}, t) \sum_{k} \lambda_k(\mathbf{x})$$

The **total rate** $\sum_k \lambda_k(\mathbf{x})$ appears explicitly, but the **individual** $\lambda_k$ only affect how probability is distributed among outgoing transitions.

**Proposition 10.1:** If $\mathbf{A}$ and $\tilde{\mathbf{A}}$ have the same diagonal entries (total exit rates) and the same eigenvalues, then they generate the same probability evolution up to exponentially small corrections.

**Implication:** Probability data primarily constrains:
- Total exit rates (diagonal of $\mathbf{A}$)
- Spectrum (eigenvalues, mixing times)

Individual off-diagonal entries are less constrained.

### 10.4 Hierarchy of Identifiability

Based on empirical and theoretical analysis:

**Tier 1: Well-identified (error < 50%)**
- Total exit rate from each state: $w(\mathbf{x}) = \sum_k \lambda_k(\mathbf{x})$
- Stationary distribution (for ergodic systems): $\boldsymbol{\pi}$
- Dominant eigenvalue (spectral gap): $\gamma = -\text{Re}(\lambda_2)$

**Tier 2: Partially identified (error 50-200%)**
- Relative rates among competing reactions with same stoichiometry
- Subdominant eigenvalues
- Flux pathways between metastable states

**Tier 3: Poorly identified (error > 200%)**
- Absolute individual propensity values $\lambda_k(\mathbf{x})$
- Polynomial coefficients $\theta_{kf}$ for $f > 1$ (higher-order terms)
- Reactions with very low flux

**Recommendation:** Focus validation on Tier 1 quantities. Do not expect to recover individual propensities accurately unless additional constraints (e.g., known functional forms) are imposed.

---

## 11. Connection to Paper 1: Flux-Preserving FSP

### 11.1 Bidirectional Relationship

The inverse problem (Paper 2) and forward FSP (Paper 1) are intimately connected through the concept of **flux**.

**Paper 1 (Forward FSP):**

**Input:** Generator $\mathbf{A}$, initial distribution $\mathbf{p}(0)$

**Task:** Adaptively select states $\mathcal{S}(t)$ to approximate $\mathbf{p}(t) = e^{\mathbf{A}t}\mathbf{p}(0)$

**Method:** Flux-based pruning
$$\Phi(\mathbf{x}, t) = p(\mathbf{x}, t) \cdot w(\mathbf{x})$$
where $w(\mathbf{x}) = \sum_k \lambda_k(\mathbf{x})$ is total exit rate.

**Key insight:** States with low probability but high exit rates ($w(\mathbf{x})$ large) are critical for preserving connectivity in bottleneck systems.

**Paper 2 (Inverse Problem):**

**Input:** Histograms $\{\mathbf{p}(t_i)\}$, local state spaces $\{\mathcal{S}_i\}$

**Task:** Learn generator $\mathbf{A}$ explaining probability evolution

**Method:** Gradient-based optimization with Fréchet derivatives

**Key finding:** Total exit rates $w(\mathbf{x})$ are well-identified, individual propensities $\lambda_k(\mathbf{x})$ are not.

### 11.2 Shared Insights on Flux

Both papers recognize **flux**, not just probability, as the fundamental quantity:

**Forward perspective (Paper 1):**

The flux-preserving truncation error bound:

$$\|\mathbf{p}(t) - \mathbf{p}^{\text{FSP}}(t)\|_1 \leq \epsilon_0 + \int_0^t \Phi_{\text{boundary}}(s) \, ds$$

where $\Phi_{\text{boundary}} = \sum_{\mathbf{x} \in \partial \mathcal{S}} \Phi(\mathbf{x}, t)$ is boundary flux.

**Inverse perspective (Paper 2):**

The learned generator recovers total rates: $\sum_k \hat{\lambda}_k(\mathbf{x}) \approx \sum_k \lambda_k(\mathbf{x}) = w(\mathbf{x})$.

This means the learned $\mathbf{A}$ correctly captures the **flux structure**, even if individual $\lambda_k$ are wrong.

**Unifying principle:** Probability evolution is governed by **flux distribution**, not individual reaction rates.

### 11.3 Adaptive State Space Selection

Both papers use **local, adaptive truncation**:

**Paper 1:**
$$\mathcal{S}(t+\Delta t) = \mathcal{S}(t) \cup \{\mathbf{x} : \Phi(\mathbf{x}, t) > \epsilon_{\Phi}\} \setminus \{\mathbf{x} : p(\mathbf{x}, t) < \epsilon_p\}$$

**Paper 2:**
$$\mathcal{S}_i = \text{supp}(\mathbf{p}(t_i)) \cup \text{supp}(\mathbf{p}(t_{i+1}))$$

**Commonality:** Both grow the state space as the distribution spreads, following FSP philosophy.

**Difference:** 
- Paper 1: Expands based on predicted flux (model-driven)
- Paper 2: Expands based on observed support (data-driven)

### 11.4 Validation Loop

The two problems provide mutual validation:

**Forward → Inverse:**
1. Simulate CME using known generator $\mathbf{A}_{\text{true}}$ with adaptive FSP (Paper 1)
2. Generate histograms $\{\mathbf{p}(t_i)\}$ from simulation
3. Learn generator $\hat{\mathbf{A}}$ using inverse method (Paper 2)
4. Compare $\hat{\mathbf{A}}$ vs. $\mathbf{A}_{\text{true}}$

**Inverse → Forward:**
1. Learn generator $\hat{\mathbf{A}}$ from experimental trajectory data (Paper 2)
2. Simulate forward using $\hat{\mathbf{A}}$ with adaptive FSP (Paper 1)
3. Compare predicted trajectories vs. held-out data
4. Validate on independent experiments

This **closed-loop validation** strengthens confidence in both methods.

### 11.5 Flux-Weighted Inverse Problem (Future Extension)

The current inverse problem uses unweighted probability:

$$J(\boldsymbol{\theta}) = \|\mathbf{p}_{next} - e^{\mathbf{A}\Delta t}\mathbf{p}_{curr}\|_1$$

**Proposed flux-weighted objective:**

$$J_{\Phi}(\boldsymbol{\theta}) = \|\mathbf{W}(\mathbf{p}_{next} - e^{\mathbf{A}\Delta t}\mathbf{p}_{curr})\|_1$$

where $\mathbf{W} = \text{diag}(w_1, \ldots, w_n)$ and $w_j = \sum_k \lambda_k(\mathbf{x}_j)$ is the total exit rate from state $j$.

**Motivation:** States with high flux should be fit more accurately, as they control connectivity.

**Challenge:** $\mathbf{W}$ depends on $\boldsymbol{\theta}$ (circular dependence). Possible solutions:
- Iterative refinement: alternate between fixing $\mathbf{W}$ and optimizing $\boldsymbol{\theta}$
- Estimate $\mathbf{W}$ from data: $w_j \approx \frac{1}{\Delta t} \log\left(\frac{p_{curr}(\mathbf{x}_j)}{p_{next}(\mathbf{x}_j)}\right)$

This is a promising direction for future work, creating a stronger bridge between Papers 1 and 2.

---

## 12. Computational Complexity and Scalability

### 12.1 Per-Iteration Cost Breakdown

For a single window with $n$ states, $R$ reactions, $n_f$ features:

**1. Generator construction:** $O(n \cdot R \cdot n_f)$

For each state $j$ and reaction $k$:
- Compute propensity: $\boldsymbol{\theta}_k^\top \mathbf{f}(\mathbf{x}_j)$ → $O(n_f)$
- Set matrix entries: $A_{ij} = \lambda_k(\mathbf{x}_j)$ → $O(1)$ per transition

Total: $O(n \cdot R \cdot n_f)$ operations

**Optimization:** Precompute transition map $(\mathbf{x}_j, k) \mapsto i$ so we don't search for target states.

**2. Matrix exponential:** $O(n^3)$

Using Higham's scaling-and-squaring algorithm (MATLAB/Julia `expm`):
- Padé approximation + squaring: $O(n^3)$
- Dominates for $n > 50$

**3. Fréchet derivative (per parameter):** $O(n^3)$

Block exponential: $(2n) \times (2n)$ matrix → $O(8n^3)$ naively

Optimized implementation exploiting structure: $O(3n^3)$

**4. Gradient computation (all parameters):** $O(R \cdot n_f \cdot n^3)$

For each parameter $\theta_{kf}$:
- Build perturbation $\mathbf{E}_{kf}$: $O(n)$
- Compute Fréchet derivative: $O(n^3)$
- Inner product: $O(n^2)$

Total for all $R \cdot n_f$ parameters: $O(R \cdot n_f \cdot n^3)$

**5. Line search (typically 2-5 function evaluations):** $O(n^3)$ each

**Total per iteration:** $O(R \cdot n_f \cdot n^3)$

### 12.2 Scaling with System Size

**Typical parameters:**

| System | $n$ | $R$ | $n_f$ | Params | Time/Iter | Iters | Total Time |
|--------|-----|-----|-------|--------|-----------|-------|------------|
| Brusselator (small) | 55 | 4 | 6 | 24 | 3 sec | 100 | 5 min |
| Brusselator (medium) | 100 | 4 | 6 | 24 | 15 sec | 150 | 40 min |
| Brusselator (large) | 200 | 4 | 6 | 24 | 120 sec | 200 | 7 hours |
| 3-species network | 150 | 8 | 10 | 80 | 100 sec | 250 | 7 hours |
| 4-species network | 300 | 12 | 10 | 120 | 600 sec | 300 | 50 hours |

**Bottleneck:** Matrix exponential and Fréchet derivative scale as $O(n^3)$.

**Practical limit:** $n \approx 200-300$ on standard hardware (M1 MacBook Pro).

### 12.3 Parallelization Strategies

**1. Window-level parallelism:**

Each window pair is independent:

$$J(\boldsymbol{\theta}) = \sum_{i=1}^{N} J_i(\boldsymbol{\theta})$$

Compute gradients in parallel:

```julia
using Distributed
@everywhere function compute_window_gradient(θ, data_i, window_i)
    # Compute gradient for window i
    return ∇J_i
end

gradients = pmap(i -> compute_window_gradient(θ, data[i], windows[i]), 1:N)
∇J = sum(gradients)
```

**Speedup:** Linear in number of cores (up to $N$ cores for $N$ windows)

**2. Parameter-level parallelism (within window):**

Gradient components are independent:

$$\nabla J = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \vdots \\ \frac{\partial J}{\partial \theta_{R \cdot n_f}} \end{bmatrix}$$

Compute Fréchet derivatives in parallel:

```julia
∇J = zeros(R * n_f)
Threads.@threads for k in 1:R*n_f
    E_k = build_perturbation(k, data)
    L_k = frechet(A * dt, E_k * dt)
    ∇J[k] =
```

This requires computing column sums of both **A** and **E**ₖf, then taking their inner product.

### 6.7 Total Gradient

The complete gradient combines all three terms:

```
∇J = [prediction gradient] + λ₁·[Frobenius gradient] + λ₂·[semigroup gradient]
```

For multiple windows, we sum the gradients across all window pairs, as each contributes independently to the total objective.

---

## 7. Optimization Algorithm

### 7.1 L-BFGS Method

We use the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) quasi-Newton method for optimization. This choice is motivated by:

1. **Analytical gradients:** We have exact gradients via Fréchet derivatives
2. **Moderate dimension:** Typically 24-48 parameters (R reactions × nf features)
3. **Non-linear objective:** Matrix exponential creates complex loss landscape
4. **Memory efficiency:** L-BFGS approximates the Hessian using recent gradient history

### 7.2 Implementation via Optim.jl

The Optim.jl package provides a robust L-BFGS implementation. We interface with it using the "only_fg!" interface, which expects a function that computes both objective and gradient simultaneously:

```
function fg!(F, G, θ)
    obj, grad = compute_objective_and_gradient(θ, data, windows)
    if G !== nothing
        G[:] = grad
    end
    return obj
end
```

**Key implementation details:**

1. **Simultaneous computation:** We compute objective and gradient in a single pass through the data, avoiding redundant matrix exponentials

2. **In-place updates:** Gradients are written directly into the provided buffer G to avoid allocations

3. **Conditional gradient:** The gradient is only computed when G is not nothing (some line search steps only need the objective)

### 7.3 Initialization

We initialize parameters as small random values:
```
θ₀ = 0.1 · randn(n_params)
```

This warm start near zero:
- Avoids large initial generators (numerical stability)
- Provides weak propensities that gradually strengthen
- Breaks symmetry for the optimizer

Alternative initialization strategies could use:
- Mass-action kinetics estimates
- Results from simpler models
- Transfer learning from related systems

### 7.4 Convergence Criteria

L-BFGS uses multiple convergence criteria:

1. **Gradient norm:** ||∇J|| < gtol (default: 10⁻⁸)
2. **Function change:** |J^{k+1} - J^k| < ftol·|J^k|
3. **Maximum iterations:** k < max_iter (typically 100-300)

In practice, we find that:
- Simple systems (single window, 50 states) converge in 50-100 iterations
- Complex systems (multiple windows, 100+ states) may require 200-300 iterations
- Gradient norm often plateaus before formal convergence

### 7.5 Line Search

L-BFGS requires a line search to determine step sizes. Optim.jl uses a backtracking line search with strong Wolfe conditions:

1. **Sufficient decrease:** J(θ + α·d) ≤ J(θ) + c₁·α·∇J(θ)ᵀd
2. **Curvature condition:** |∇J(θ + α·d)ᵀd| ≤ c₂·|∇J(θ)ᵀd|

where d is the search direction, α is the step size, and c₁ = 10⁻⁴, c₂ = 0.9 are standard constants.

The analytical gradients ensure these conditions can be checked exactly, improving line search efficiency.

---

## 8. Sliding Window Approach

### 8.1 Motivation

For a single window pair (tᵢ, tᵢ₊₁), we learn a generator **A**ᵢ that explains local dynamics. However, as the probability distribution spreads over time, different regions of state space become active.

The sliding window approach learns a sequence of local generators:
```
Window 1: (t₁, t₂) → A₁ on state space S₁
Window 2: (t₂, t₃) → A₂ on state space S₂
Window 3: (t₃, t₄) → A₃ on state space S₃
...
```

Each generator operates on its own local truncated state space, following the FSP philosophy.

### 8.2 Why Different Generators?

The generators **A**ᵢ are fundamentally different objects:
- **A**₁ is a 55×55 matrix (early dynamics, concentrated distribution)
- **A**₂ is a 76×76 matrix (spreading distribution)
- **A**₃ is a 111×111 matrix (widely dispersed distribution)

These are not different parameterizations of the same operator, but rather **local truncations** of the true infinite-dimensional generator.

**Analogy to FSP:** Just as forward FSP adaptively expands the state space, the inverse problem learns generators on adaptively growing truncations.

### 8.3 Theoretical Interpretation

From a theoretical perspective, each **A**ᵢ approximates the restriction of the true generator **A**_true to the active state space at time window i:

```
Aᵢ ≈ (A_true)|_{Sᵢ}
```

As the state space grows (S₁ ⊂ S₂ ⊂ S₃), we obtain increasingly complete views of the generator. However, parameter identifiability becomes more challenging with larger state spaces.

### 8.4 Independent vs. Joint Learning

**Current approach (independent):**
Each window is optimized separately. Advantages:
- Parallelizable
- Numerically stable (smaller matrices)
- Natural connection to FSP truncation

**Alternative (joint learning):**
Optimize a single parameter vector across all windows. Challenges:
- Different state spaces require careful padding/projection
- Much larger computational cost
- Risk of poor convergence due to conflicting gradients

We adopt the independent approach, viewing it as spatially and temporally adaptive inverse FSP.

---

## 9. Computational Complexity

### 9.1 Per-Iteration Cost

For a single window with state space size n and R reactions with nf features each:

**Generator construction:** O(n·R·nf)
- For each state and reaction, compute propensity: O(nf)
- Sparse construction possible if transition map is precomputed

**Matrix exponential:** O(n³)
- Dominates for n > 50
- Uses Higham's scaling-and-squaring algorithm
- Can exploit sparsity for very large sparse systems

**Fréchet derivative:** O(n³)
- Block exponential doubles dimension: (2n)×(2n) → O(8n³)
- Computed once per unique window pair
- Must be repeated R·nf times for each parameter

**Gradient computation:** O(R·nf·n²)
- Each parameter requires one Fréchet derivative
- Inner products with vectors: O(n²)

**Total per iteration:** O(R·nf·n³)

### 9.2 Scaling Analysis

**Typical parameters:**
- State space: n = 50-150
- Reactions: R = 4-8
- Features: nf = 3-6
- Iterations: 100-300
- Windows: 1-10

**Example: Brusselator (single window)**
- n = 55, R = 4, nf = 6
- Parameters: 24
- Per iteration: ~3 seconds (M1 MacBook Pro)
- Converges in ~100 iterations → ~5 minutes total

**Scaling to larger systems:**
- n = 200: ~30 seconds per iteration
- Multiple windows: linear scaling in number of windows
- Parallel window learning: embarrassingly parallel

### 9.3 Memory Requirements

**Main memory costs:**
1. State space features: n × nf (negligible)
2. Generator matrix: n² (dense storage)
3. Block exponential matrix: (2n)² ≈ 4n²
4. L-BFGS history: m × (R·nf) where m ≈ 10

For n = 100, R = 4, nf = 6:
- Generator: ~80 KB
- Block matrix: ~320 KB
- L-BFGS: ~2 KB
- Total: <1 MB (completely tractable)

---

## 10. Error Analysis

### 10.1 Sources of Error

The total error in learned generators comes from multiple sources:

**1. Trajectory sampling error**
Histograms are empirical estimates: **p**_emp ≈ **p**_true
- Reduced by increasing number of trajectories
- Poisson statistics: error ~ O(1/√N)

**2. Time discretization error**
Discrete time steps: **p**(t + Δt) ≈ exp(**A**Δt)**p**(t)
- Larger Δt → more aggregation of reactions
- Smaller Δt → fewer states, less coverage
- Trade-off controlled by Δt selection

**3. State space truncation error**
Local truncation: **A**ᵢ ≈ (**A**_true)|_{Sᵢ}
- Boundary flux lost
- Similar to FSP truncation error
- Controlled by probability threshold

**4. Propensity basis error**
Polynomial approximation: λ(**x**) ≈ **θ**·**f**(**x**)
- True propensities may not be polynomial
- Example: x(x-1)y needs quadratic terms
- Mitigated by choosing appropriate nf

**5. Optimization error**
Local minima, non-convexity
- L-BFGS finds local optimum
- Multiple initializations can help
- Analytical gradients reduce numerical error

### 10.2 Identifiability Limits

Even with perfect data, unique recovery may be impossible due to the **Markov embedding problem**:

**Question:** Given transition matrix **P** = exp(**Q**Δt), is **Q** unique?

**Answer:** No, in general. Key results (Casanellas et al., 2020):

1. For 4×4 matrices with distinct eigenvalues:
   - Embeddability determined by single condition
   - May have 1 or multiple generators
   - Non-identifiable even with positive eigenvalues

2. For larger matrices:
   - Complete characterization unknown
   - Necessary conditions on eigenvalues exist
   - Non-uniqueness is generic, not exceptional

**Implications for CME inverse problems:**

- **Total exit rates** from each state are well-constrained
- **Relative rates** among competing reactions are partially constrained
- **Absolute propensity values** may not be uniquely identifiable

This explains why our learned propensities have correct total outflow (ratio ≈ 1) but wrong individual reactions (ratio 0.3-5).

### 10.3 Error Bounds

Under ideal conditions (infinite data, exact histograms), the prediction error satisfies:

```
||p_{true}(t+Δt) - exp(A_learned·Δt) p_true(t)||₁ ≤ ε_opt + ε_basis + ε_embedding
```

where:
- ε_opt: Optimization residual (how well we minimize J)
- ε_basis: Propensity representation error (polynomial approximation)
- ε_embedding: Fundamental non-uniqueness (Markov embedding)

With finite data, add sampling error:
```
ε_total ≤ ε_opt + ε_basis + ε_embedding + O(1/√N)
```

---

## 11. Selection of Time Step Δt

### 11.1 The Δt Dilemma

Time step selection faces competing requirements:

**Small Δt (e.g., 0.05-0.25 sec):**
- ✓ Better approximation: exp(**A**Δt) ≈ **I** + **A**Δt
- ✓ Less reaction aggregation (closer to elementary steps)
- ✗ Small state space (poor coverage)
- ✗ Underdetermined system (more parameters than constraints)
- ✗ Insufficient mixing between metastable states

**Large Δt (e.g., 1.0-2.0 sec):**
- ✓ Large state space (good coverage)
- ✓ Overdetermined system (more states than parameters)
- ✓ Distribution spreads across many states
- ✗ Nonlinear exponential: exp(**A**Δt) far from **I** + **A**Δt
- ✗ Multiple reactions per window (aggregation)
- ✗ Scaling ambiguity: α**A** with Δt/α may fit similarly

### 11.2 Empirical Findings

From Brusselator experiments:

| Δt   | States | ||**A**|| | Total Rate Error | Outcome |
|------|--------|---------|------------------|---------|
| 0.05 | 7      | N/A     | 3-50×            | Failed (too few states) |
| 0.25 | 14     | N/A     | 10-90×           | Failed (underdetermined) |
| 0.50 | 35     | ~1200   | 2-8×             | Moderate |
| 1.00 | 55     | 784     | 0.97-1.43×       | **Best** |
| 2.00 | 85     | ~400    | 1.5-3×           | Good but overly smooth |

**Optimal range:** Δt ∈ [0.5, 1.5] for this system.

### 11.3 Theoretical Guidance

The Markov embedding literature provides necessary conditions for uniqueness:

**Diagonally dominant matrices** (Cuthbert, 1972):
If **P** = exp(**Q**Δt) has diagonal entries > 0.5, then **Q** is unique.

**Determinant bounds** (Israel et al., 2001):
If det(**P**) > 0.5, the principal logarithm is a valid generator.

**Practical heuristic:**
Choose Δt such that:
1. State space covers at least 95% of probability mass
2. Number of states exceeds number of parameters by factor ≥2
3. Distribution shows significant evolution (avoid Δt → 0)

### 11.4 Adaptive Δt Selection

For systems with multiple timescales, a single Δt may not suffice:

**Strategy 1: Variable Δt**
- Use small Δt during fast transients
- Use large Δt during slow equilibration
- Requires detecting timescale transitions

**Strategy 2: Multi-resolution**
- Learn generators at multiple Δt values
- Combine via weighted ensemble
- Check consistency across scales

**Strategy 3: Continuous-time formulation**
- Use random Δt from SSA jumps
- Avoid discretization bias
- Requires different parameterization

Current implementation uses fixed Δt, but adaptive strategies are promising future directions.

---

## 12. Connection to Paper 1 (FSP)

### 12.1 Bidirectional Relationship

The inverse problem and forward FSP are intimately connected:

**Forward FSP (Paper 1):**
- Given: Generator **A**
- Task: Adaptively select states to track **p**(t)
- Method: Flux-based pruning preserves connectivity
- Output: Approximate solution with error bounds

**Inverse FSP (Paper 2):**
- Given: Histograms **p**(t₁), **p**(t₂), ...
- Task: Learn generator **A** on local state spaces
- Method: Fréchet-based gradient optimization
- Output: Generators explaining probability evolution

Both use **local state space truncation** following FSP principles.

### 12.2 Flux as Fundamental Quantity

**Paper 1 insight:** States with low probability but high flux are critical for connectivity.

**Paper 2 implication:** The objective should be weighted by flux:
```
J_flux(θ) = Σᵢ ||φᵢ₊₁ - exp(A·Δt) φᵢ||
```
where φ = diag(w)·**p** is probability-flux distribution, and w[j] = total exit rate from state j.

Current formulation uses unweighted probability, but flux-weighted objectives could improve identifiability of high-flux pathways.

### 12.3 Validation Loop

The two problems validate each other:

1. **Forward validation:** Use learned **A** to simulate forward → compare to true trajectories
2. **Inverse validation:** Use true **A** from known model → check if inverse method recovers it
3. **Cross-validation:** Learn **A** from data → use for FSP → check predictions against held-out data

This bidirectional testing strengthens confidence in both methods.

### 12.4 Shared Challenges

Both face similar difficulties:

| Challenge | FSP (Forward) | Inverse |
|-----------|---------------|---------|
| State space explosion | Adaptive expansion | Local truncation |
| Bottleneck states | Flux-based protection | Identifiability issues |
| Stiff systems | Adaptive time-stepping | Δt selection |
| Error control | Boundary flux bounds | Regularization |
| Computational cost | Matrix exponential | Fréchet derivatives |

Solutions developed for one problem inform the other.

---

## 13. Practical Recommendations

### 13.1 Choosing Parameters

**Regularization weights:**
- Start with λ₁ = λ₂ = 10⁻⁶
- If ||**A**|| > 1000, increase λ₁
- If column sums > 10⁻⁴, increase λ₂
- If underfitting, decrease both

**Time step Δt:**
- Begin with Δt ≈ mean inter-event time
- Increase until state space ≥ 2× number of parameters
- Verify: most states have probability > 10⁻⁶

**Feature count:**
- Use nf = 3 for mass-action kinetics
- Use nf = 6 for systems with higher-order reactions
- Avoid nf > 6 (overfitting risk)

**Number of trajectories:**
- Minimum: N = 50
- Recommended: N = 100-500
- More trajectories reduce sampling noise

### 13.2 Diagnostics

**After optimization, check:**

1. **Convergence:** Did L-BFGS report convergence?
2. **Column sums:** max|Σᵢ A[i,j]| < 10⁻⁶
3. **Prediction error:** L1 error < 0.2 per window
4. **Generator norm:** ||**A**|| in reasonable range (10²-10³)
5. **Sparsity:** Nonzeros match expected reaction network

**Warning signs:**

- Very large ||**A**|| (>10⁴): numerical instability
- Poor prediction (L1 > 0.5): underfitting or wrong Δt
- Negative off-diagonals: optimization error
- All propensities ≈ 0: regularization too strong

### 13.3 Troubleshooting

**Problem:** Poor convergence (many iterations, high gradient)
- Solution: Increase regularization, better initialization, reduce max_iter

**Problem:** Wrong propensities but good predictions
- Explanation: Fundamental non-uniqueness (Markov embedding)
- Check: Total outflow rates should be correct

**Problem:** Underdetermined system (fewer states than parameters)
- Solution: Increase Δt, reduce nf, or use multiple windows

**Problem:** Exploding generator norm
- Solution: Increase λ₁, check Δt not too large

---

## 14. Limitations and Future Work

### 14.1 Current Limitations

**1. Polynomial basis restriction**
Cannot represent all propensity forms. Example:
- True: λ(**x**) = c·exp(-x/τ)
- Best polynomial approximation: May be poor

**2. Known stoichiometry**
Currently extract from data, but:
- Rare reactions may be missed
- Requires sufficient sampling of all transitions

**3. Single Δt**
Fixed time step may not suit multi-timescale systems

**4. Local generators**
Each window learns separately—no global consistency enforced

**5. Computational cost**
O(n³) per iteration limits to n < 500

### 14.2 Extensions

**Non-polynomial propensities:**
- Rational functions: λ = (θ₁ + θ₂x)/(θ₃ + θ₄x)
- Neural networks: λ = NN(**x**; θ)
- Requires different gradient computations

**Stoichiometry discovery:**
- Two-stage: (1) Discover network, (2) Learn rates
- Joint optimization with sparsity penalty
- Connection to SINDy/sparse regression

**Time-varying parameters:**
- Piecewise constant: **A**(t) on intervals
- Smooth variation: **A**(θ(t)) with θ smooth
- Useful for non-autonomous systems

**Uncertainty quantification:**
- Bayesian framework: posterior over **A**
- Ensemble methods: multiple initializations
- Bootstrap: resample trajectories

**Large-scale systems:**
- Krylov methods for matrix exponential
- Low-rank approximations
- Tensor decompositions

### 14.3 Experimental Validation

Next critical step: **Real experimental data**

Challenges:
- Noisy observations (measurement error)
- Partial observability (not all species measured)
- Time-varying conditions (non-autonomous)
- Model misspecification (wrong network structure)

Opportunities:
- Validate on fluorescence microscopy data
- Integrate with single-cell RNA-seq
- Learn from flow cytometry histograms

---

## 15. Conclusion

We have developed a principled framework for learning CME generators from trajectory data using:

1. **Analytical gradients** via Fréchet derivatives (not finite differences)
2. **Stoichiometric parameterization** (structure from chemistry)
3. **Sliding window approach** (local state spaces, FSP connection)
4. **Regularized optimization** (Frobenius + semigroup constraints)

The method successfully recovers generators for benchmark systems (Brusselator), achieving:
- Total outflow rates within 43% of true values
- Good prediction accuracy (L1 < 0.2)
- Efficient computation (minutes for 50-100 state systems)

Fundamental limitations exist due to the **Markov embedding problem**—even with perfect data, propensities may not be uniquely identifiable. However, the total exit rates (which govern probability dynamics) are well-recovered.

The connection to flux-preserving FSP (Paper 1) provides mutual validation and suggests that **flux**, not just probability, should guide both forward simulation and inverse learning.
