# Bandit Neighbor Sampling — Experimental Plan

## Overview

This document outlines a progressive experimental plan for evaluating multi-armed bandit algorithms for optimal neighbor sampling in decentralized learning. The setting is as follows: each node selects $K$ neighbors per round using a bandit algorithm, observes per-neighbor rewards (semi-bandit feedback), and adapts its sampling strategy over time. The environment is non-stationary stochastic — rewards drift smoothly as model weights evolve during training, with high variance early and stabilization late.

---

## Phase 1: Establish Baselines

**Goal:** Bracket the performance space before introducing any bandit logic.

### Algorithms
- **Uniform sampling** — each node samples $K$ neighbors uniformly at random each round. This is the null hypothesis and primary baseline.
- **Static oracle** — sample the $K$ neighbors with highest average reward computed in hindsight. This is the performance ceiling.

### Hypothesis
If the gap between uniform and oracle is small, neighbor selection has little impact and the bandit framing may not be worthwhile. A large gap confirms there is real value to capture.

### What to Log
- Final model loss / accuracy
- Convergence curve (loss vs. round)
- Per-neighbor reward signal $r_i(t)$ over time — check that it is informative and not pure noise

---

## Phase 2: Stationary Bandits

**Goal:** Verify that bandit sampling helps at all, ignoring non-stationarity.

### Algorithms
- **CUCB** — Combinatorial UCB with semi-bandit feedback. Simplest combinatorial bandit, strong theoretical guarantees under stochastic rewards.
- **CTS** — Combinatorial Thompson Sampling with semi-bandit feedback. Empirically often faster to converge than CUCB.

### Update Rule (Semi-Bandit)
For each $i \in S_t$, after observing reward $r_i(t)$:

$$\hat{\mu}_i \leftarrow \frac{T_i \cdot \hat{\mu}_i + r_i(t)}{T_i + 1}, \qquad T_i \leftarrow T_i + 1$$

### Hypothesis
CTS converges faster than CUCB empirically. Both outperform uniform sampling once the bandit has identified good neighbors.

### What to Log
- Convergence speed vs. uniform baseline
- Neighbor selection over time — do good neighbors get selected more frequently?
- Regret of the bandit: cumulative reward vs. static oracle

---

## Phase 3: Non-Stationary Adaptations

**Goal:** Account for the fact that neighbor quality drifts as model weights evolve during training.

### Algorithms

#### 3a. Discounted CTS (Fixed $\gamma$)
Maintain exponentially discounted sufficient statistics for each arm's Beta posterior:

$$\alpha_i(t) = \gamma \cdot \alpha_i(t-1) + r_i(t) \cdot \mathbf{1}[i \in S_t]$$
$$\beta_i(t) = \gamma \cdot \beta_i(t-1) + (1 - r_i(t)) \cdot \mathbf{1}[i \in S_t]$$

Sweep $\gamma \in \{0.95, 0.99, 0.999\}$ to understand sensitivity to drift rate.

#### 3b. Discounted CTS (Annealed $\gamma(t)$)
Couple the discount factor to the learning rate schedule, exploiting the insight that drift is high early and low late:

$$\gamma(t) = 1 - c \cdot \eta(t)$$

As the learning rate $\eta(t)$ decays and models stabilize, $\gamma(t)$ automatically increases — more forgetting early, more accumulation late. This adds no new hyperparameters if tied directly to the existing learning rate schedule.

Alternatively, use cosine annealing:

$$\gamma(t) = \gamma_{\min} + \frac{1}{2}(\gamma_{\max} - \gamma_{\min})\left(1 + \cos\left(\pi \cdot \frac{T - t}{T}\right)\right)$$

### Hypothesis
Fixed $\gamma$ improves over stationary CTS. Annealed $\gamma(t)$ further improves over fixed $\gamma$, particularly in early training where drift is fastest.

### What to Log
- Sensitivity plot: final performance vs. fixed $\gamma$ value
- Comparison of annealed vs. best fixed $\gamma$
- $\gamma(t)$ schedule overlaid with reward variance over time — verify they align

---

## Phase 4: Robustness Check

**Goal:** Validate assumptions about the environment by comparing against adversarial-regime algorithms.

### Algorithms
- **Exp3.M** — adversarial combinatorial bandit for top-$K$ selection. No stochastic assumptions. Expected to underperform Discounted CTS if the environment is well-behaved.
- **Tsallis-INF** (optional) — Best-of-Both-Worlds algorithm. Achieves $O(\log T)$ regret in stochastic environments and $O(\sqrt{T})$ in adversarial environments without knowing which regime applies.

### Hypothesis
Discounted CTS with annealed $\gamma$ clearly outperforms Exp3.M, confirming the environment is exploitably stochastic. If Exp3.M is competitive, the environment may be more adversarial than assumed and the Byzantine assumption should be reconsidered.

### What to Log
- Head-to-head comparison: Discounted CTS vs. Exp3.M vs. Tsallis-INF
- Regret curves for all algorithms

---

## Metrics

Track the following across all phases:

| Metric | Description |
|---|---|
| **Convergence speed** | Rounds to reach a target validation loss |
| **Final model quality** | Loss / accuracy at end of training |
| **Bandit regret** | Cumulative reward vs. static oracle |
| **Neighbor stability** | Do selected neighbors stabilize late in training? |
| **Reward signal variance** | $\text{Var}(r_i(t))$ over time — validates early/late drift insight |

---

## Practical Guidelines

- **Start small.** Prototype on a network of 5–10 nodes with a simple task. Iterate fast, visualize neighbor selection directly.
- **Fix everything else.** When comparing algorithms, keep $K$, learning rate, architecture, and data fixed. Only vary the sampling strategy.
- **Run multiple seeds.** Average over at least 5 random seeds. Bandit algorithms have high variance early in training.
- **Log the reward signal.** Plot $r_i(t)$ over time for each neighbor before running any bandit. If rewards are pure noise, no bandit will help.
- **Reward signal design.** The reward $r_i(t)$ for sampling neighbor $i$ should be locally computable and correlate with learning progress. Natural candidates: gradient cosine similarity, inverse model distance $\|\theta_t - \theta_i\|^{-1}$, or validation loss reduction after aggregation.

---

## Narrative Arc

Each phase adds one idea with a clear hypothesis:

> **Uniform → CUCB/CTS:** does bandit sampling help at all?  
> **CTS → Discounted CTS:** does handling drift further improve performance?  
> **Fixed $\gamma$ → Annealed $\gamma(t)$:** does coupling to training dynamics help?  
> **Discounted CTS vs. Exp3.M:** is the environment stochastic enough to exploit?

This structure supports a clean project report where each experimental phase motivates the next.
