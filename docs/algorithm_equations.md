# Optimization Algorithms - Mathematical Formulations

## 1. OR-Tools CP-SAT (Constraint Programming)

**Decision Variables:**
$$x_i \in \{0, 1, 2\} \quad \forall i \in T$$

where $T$ is the set of trainsets, and:
- $x_i = 0$: trainset $i$ assigned to service
- $x_i = 1$: trainset $i$ assigned to standby
- $x_i = 2$: trainset $i$ assigned to maintenance

**Constraints:**
$$\sum_{i \in T} \mathbb{1}_{x_i = 0} = N_{\text{service}}$$

$$\sum_{i \in T} \mathbb{1}_{x_i = 1} \geq N_{\text{standby}}^{\text{min}}$$

$$\sum_{i \in T} \mathbb{1}_{x_i = 0} \leq C_{\text{service}}^{\text{max}}$$

**Objective Function:**
$$\max \quad Z = w_r \sum_{i \in T} r_i \cdot \mathbb{1}_{x_i = 0} + w_b \cdot B - w_v \cdot V$$

where:
- $r_i$: readiness score of trainset $i$
- $B$: balance score
- $V$: total violations
- $w_r, w_b, w_v$: weights

---

## 2. Mixed Integer Programming (MIP)

**Decision Variables:**
$$y_{i,s} \in \{0, 1\} \quad \forall i \in T, s \in S$$

where $S = \{0, 1, 2\}$ (service, standby, maintenance)

**Constraints:**
$$\sum_{s \in S} y_{i,s} = 1 \quad \forall i \in T$$

$$\sum_{i \in T} y_{i,0} = N_{\text{service}}$$

$$\sum_{i \in T} y_{i,1} \geq N_{\text{standby}}^{\text{min}}$$

**Objective Function:**
$$\max \quad Z = \sum_{i \in T} \sum_{s \in S} c_{i,s} \cdot y_{i,s}$$

where $c_{i,s}$ is the cost coefficient for assigning trainset $i$ to state $s$.

---

## 3. Genetic Algorithm (GA)

**Chromosome Representation:**
$$\mathbf{x} = [x_1, x_2, \ldots, x_n] \quad x_i \in \{0, 1, 2\}$$

**Fitness Function:**
$$f(\mathbf{x}) = w_r \sum_{i: x_i=0} r_i + w_b \cdot \frac{1}{1 + \sigma_m^2} - w_v \cdot P(\mathbf{x})$$

where:
- $\sigma_m^2$: mileage variance
- $P(\mathbf{x})$: penalty for constraint violations

**Selection (Tournament):**
$$P(\text{select } \mathbf{x}_i) = \frac{\mathbb{1}_{f(\mathbf{x}_i) = \max_{j \in K} f(\mathbf{x}_j)}}{1}$$

where $K$ is a random tournament subset of size $k$.

**Two-Point Crossover:**
$$\mathbf{c}_1 = [\mathbf{p}_1[1:p], \mathbf{p}_2[p:q], \mathbf{p}_1[q:n]]$$
$$\mathbf{c}_2 = [\mathbf{p}_2[1:p], \mathbf{p}_1[p:q], \mathbf{p}_2[q:n]]$$

where $p, q$ are random crossover points, $\mathbf{p}_1, \mathbf{p}_2$ are parents.

**Mutation:**
$$x_i' = \begin{cases} 
\text{random}(\{0,1,2\}) & \text{with probability } p_m \\
x_i & \text{otherwise}
\end{cases}$$

---

## 4. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Sampling Distribution:**
$$\mathbf{x}_k \sim \mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$$

where:
- $\mathbf{m}$: mean vector
- $\sigma$: step size
- $\mathbf{C}$: covariance matrix

**Mean Update:**
$$\mathbf{m}^{(g+1)} = \mathbf{m}^{(g)} + \sigma^{(g)} \mathbf{y}_w$$

where $\mathbf{y}_w = \sum_{i=1}^{\mu} w_i \mathbf{y}_{i:\lambda}$ is weighted recombination.

**Step Size Update:**
$$\sigma^{(g+1)} = \sigma^{(g)} \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_\sigma^{(g+1)}\|}{\mathbb{E}\|\mathcal{N}(0,I)\|} - 1\right)\right)$$

**Covariance Matrix Update:**
$$\mathbf{C}^{(g+1)} = (1-c_1-c_\mu)\mathbf{C}^{(g)} + c_1\mathbf{p}_c\mathbf{p}_c^T + c_\mu\sum_{i=1}^{\mu}w_i\mathbf{y}_{i:\lambda}\mathbf{y}_{i:\lambda}^T$$

---

## 5. Particle Swarm Optimization (PSO)

**Position and Velocity:**
$$\mathbf{x}_i(t) \in \mathbb{R}^n, \quad \mathbf{v}_i(t) \in \mathbb{R}^n$$

**Velocity Update:**
$$\mathbf{v}_i(t+1) = w\mathbf{v}_i(t) + c_1r_1(\mathbf{p}_i - \mathbf{x}_i(t)) + c_2r_2(\mathbf{g} - \mathbf{x}_i(t))$$

where:
- $w$: inertia weight
- $c_1, c_2$: cognitive and social coefficients
- $r_1, r_2 \sim U(0,1)$: random numbers
- $\mathbf{p}_i$: personal best position
- $\mathbf{g}$: global best position

**Position Update:**
$$\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + \mathbf{v}_i(t+1)$$

**Discrete Conversion:**
$$x_i = \begin{cases}
0 & \text{if } \text{sigmoid}(\tilde{x}_i) > \text{threshold}_0 \\
1 & \text{if } \text{sigmoid}(\tilde{x}_i) > \text{threshold}_1 \\
2 & \text{otherwise}
\end{cases}$$

---

## 6. Simulated Annealing (SA)

**Energy Function:**
$$E(\mathbf{x}) = -f(\mathbf{x})$$

where $f(\mathbf{x})$ is the fitness function to maximize.

**Acceptance Probability:**
$$P(\text{accept}) = \begin{cases}
1 & \text{if } \Delta E \leq 0 \\
\exp\left(-\frac{\Delta E}{T}\right) & \text{if } \Delta E > 0
\end{cases}$$

where:
- $\Delta E = E(\mathbf{x}_{\text{new}}) - E(\mathbf{x}_{\text{current}})$
- $T$: current temperature

**Cooling Schedule (Geometric):**
$$T_{k+1} = \alpha \cdot T_k \quad \text{where } 0 < \alpha < 1$$

**Perturbation Operator:**
$$\mathbf{x}' = \text{swap}(\mathbf{x}, i, j) \quad \text{where } i, j \sim U\{1, n\}$$

---

## 7. Multi-Objective Optimization (NSGA-II)

**Objective Vector:**
$$\mathbf{F}(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_k(\mathbf{x})]^T$$

where:
- $f_1$: maximize readiness
- $f_2$: minimize mileage variance
- $f_3$: maximize branding
- $f_4$: minimize violations

**Pareto Dominance:**
$$\mathbf{x}_1 \prec \mathbf{x}_2 \iff \begin{cases}
\forall i: f_i(\mathbf{x}_1) \geq f_i(\mathbf{x}_2) \\
\exists j: f_j(\mathbf{x}_1) > f_j(\mathbf{x}_2)
\end{cases}$$

**Crowding Distance:**
$$d_i = \sum_{m=1}^{k} \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{\max} - f_m^{\min}}$$

**Selection Operator:**
$$\mathbf{x}_1 \succ \mathbf{x}_2 \iff \begin{cases}
\text{rank}(\mathbf{x}_1) < \text{rank}(\mathbf{x}_2) & \text{or} \\
\text{rank}(\mathbf{x}_1) = \text{rank}(\mathbf{x}_2) \land d_1 > d_2
\end{cases}$$

---

## 8. Ensemble Optimization

**Weighted Combination:**
$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in \{\mathbf{x}_1, \ldots, \mathbf{x}_m\}} f(\mathbf{x})$$

where $\mathbf{x}_j$ is the solution from algorithm $j$.

**Consensus Voting:**
$$x_i^* = \text{mode}\{x_i^{(1)}, x_i^{(2)}, \ldots, x_i^{(m)}\}$$

**Weighted Average (for continuous):**
$$\tilde{x}_i = \sum_{j=1}^{m} w_j \cdot x_i^{(j)} \quad \text{where } \sum_{j=1}^{m} w_j = 1$$

---

## Common Constraint Penalty Function

$$P(\mathbf{x}) = \alpha_1 \max(0, N_{\text{service}} - n_s)^2 + \alpha_2 \max(0, N_{\text{standby}}^{\min} - n_{sb})^2 + \alpha_3 \sum_{i \in T} v_i$$

where:
- $n_s = \sum_i \mathbb{1}_{x_i = 0}$: actual service count
- $n_{sb} = \sum_i \mathbb{1}_{x_i = 1}$: actual standby count
- $v_i$: trainset-specific violations (e.g., maintenance requirements)
- $\alpha_1, \alpha_2, \alpha_3$: penalty coefficients

---

## Balance Score (Mileage Variance Minimization)

$$B = 1 - \frac{\sigma_m}{\sigma_m^{\max}}$$

where:
$$\sigma_m^2 = \frac{1}{|S|} \sum_{i \in S} (m_i - \bar{m})^2$$

- $S = \{i : x_i = 0\}$: trainsets in service
- $m_i$: mileage of trainset $i$
- $\bar{m} = \frac{1}{|S|}\sum_{i \in S} m_i$: mean mileage