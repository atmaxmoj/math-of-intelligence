# Semiring: The Right Continuization

## Why Semiring, Not Just Fuzzy

Fuzzy logic ([0,1] with min/max or product) is ONE special case. Semiring unifies:
- Boolean logic
- Fuzzy logic (multiple variants)
- Probabilistic logic
- Tropical algebra (optimization)
- Gradient computation
- ...any algebraic structure with "and" and "or"

**If we're going to continuize, do it at the right level of generality.**

---

## Semiring Basics

A **commutative semiring** is (S, ⊕, ⊗, 0, 1) where:
- (S, ⊕, 0) is a commutative monoid (identity 0)
- (S, ⊗, 1) is a commutative monoid (identity 1)
- ⊗ distributes over ⊕
- 0 ⊗ a = 0 (annihilation)

Intuition:
- ⊕ = "or" / "choice" / "addition"
- ⊗ = "and" / "sequence" / "multiplication"

---

## Examples

| Semiring | S | ⊕ | ⊗ | 0 | 1 | Use |
|----------|---|---|---|---|---|-----|
| Boolean | {⊥,⊤} | ∨ | ∧ | ⊥ | ⊤ | Classical logic |
| Probability | ℝ≥0 | + | × | 0 | 1 | Probabilistic inference |
| Fuzzy (Gödel) | [0,1] | max | min | 0 | 1 | Fuzzy logic |
| Fuzzy (Product) | [0,1] | max | × | 0 | 1 | Fuzzy with product |
| Łukasiewicz | [0,1] | min(1,a+b) | max(0,a+b-1) | 0 | 1 | MV-algebras |
| Tropical (min-plus) | ℝ∪{∞} | min | + | ∞ | 0 | Shortest paths |
| Tropical (max-plus) | ℝ∪{-∞} | max | + | -∞ | 0 | Longest paths |
| Viterbi | [0,1] | max | × | 0 | 1 | Most probable explanation |
| Log | ℝ∪{±∞} | log(eᵃ+eᵇ) | + | -∞ | 0 | Numerical stability |
| Gradient | (v,∂v) pairs | (v₁+v₂, g₁+g₂) | (v₁v₂, v₁g₂+v₂g₁) | (0,0) | (1,0) | Automatic differentiation |

---

## Why Semiring for Learning

### 1. Gradient Semiring (真实存在！)

来自 **DeepProbLog** (NeurIPS 2018)，基于 Eisner 2002。

The **gradient semiring** embeds differentiation into logic:

```
Element: (p, ∂p/∂θ) = (value, partial derivative w.r.t. parameter)

⊗: (p₁,g₁) ⊗ (p₂,g₂) = (p₁×p₂, p₁×g₂ + p₂×g₁)   # product rule
⊕: (p₁,g₁) ⊕ (p₂,g₂) = (p₁+p₂, g₁+g₂)           # sum rule

Identity: 1 = (1, 0)
Zero:     0 = (0, 0)
```

This is **forward-mode automatic differentiation** algebraized.

**Key insight:** inference and gradient computation happen in parallel.
No separate forward/backward pass — it's all one algebraic computation.

```python
# 实际代码存在: github.com/ML-KULeuven/deepproblog
class GradientSemiring:
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

    def __mul__(self, other):  # ⊗
        return GradientSemiring(
            self.value * other.value,
            self.value * other.gradient + other.value * self.gradient  # product rule
        )

    def __add__(self, other):  # ⊕
        return GradientSemiring(
            self.value + other.value,
            self.gradient + other.gradient  # sum rule
        )
```

### 相关概念：Dual Numbers

Gradient semiring 是 **dual numbers** 的特例：

```
Dual number: a + bε，其中 ε² = 0

(a + bε) × (c + dε) = ac + (ad + bc)ε
                       ↑      ↑
                     value   derivative
```

Dual numbers 形成一个 semiring，是自动微分的经典数学基础。

更一般的推广是 **Nagata numbers**（1950s），见 arXiv:2212.11088。

### 2. Tropical Semiring for Optimization

Tropical algebra turns logical inference into optimization:

```
(ℝ∪{∞}, min, +)

"and" = + (add costs)
"or" = min (choose minimum)
```

Finding a satisfying assignment = finding shortest path = optimization.

### 3. Probabilistic Semiring for Learning

```
(ℝ≥0, +, ×)

"and" = × (probability of conjunction)
"or" = + (probability of disjunction, mutually exclusive)
```

Logical inference becomes probabilistic inference.

---

## Semiring-Valued Coalgebra

Combine coalgebra (the right structure) with semiring (the right continuization):

```
Crisp coalgebra:      X → F(X)           (set-valued)
Semiring coalgebra:   X → S^F(X)         (semiring-valued)
```

For Kripke-like structures:
```
Crisp:     R ⊆ W × W              (relation)
Semiring:  R: W × W → S           (S-valued relation)
```

### The Full Picture

```
Semiring-valued coalgebra (continuous, differentiable)
                ↓
Parameters: S-valued transitions
                ↓
Optimization (gradient semiring enables backprop)
                ↓
Converge to {0,1} ⊆ S
                ↓
Crisp coalgebra = Program
```

---

## Neural Semirings: Learning the Operations

Even more general: don't choose a semiring, **learn it**.

```python
class NeuralSemiring:
    def __init__(self):
        self.plus = MLP(...)   # learn ⊕
        self.times = MLP(...)  # learn ⊗

    def forward(self, a, b, op='plus'):
        if op == 'plus':
            return self.plus(torch.cat([a, b]))
        else:
            return self.times(torch.cat([a, b]))
```

The network learns what "and" and "or" should mean for the task.

**This is meta-learning at the algebraic level.**

---

## Connection to Coalgebra

| Level | Structure | Continuization |
|-------|-----------|----------------|
| Object | Coalgebra (X → F(X)) | Semiring-valued |
| Learning | Fold/Unfold | Gradient semiring |
| Meta | Learn the coalgebra | Learn the semiring |

Both dimensions generalize:
- Coalgebra generalizes Kripke/automata/etc.
- Semiring generalizes fuzzy/probabilistic/tropical/etc.

**The most general framework: Semiring-valued Coalgebra with learnable operations.**

---

## Key Papers

### Gradient Semiring & Automatic Differentiation

1. **"DeepProbLog: Neural Probabilistic Logic Programming"** (NeurIPS 2018)
   - [arXiv:1805.10872](https://arxiv.org/abs/1805.10872)
   - [GitHub](https://github.com/ML-KULeuven/deepproblog)
   - **Introduces gradient semiring for differentiable logic programming**
   - Key: inference in gradient semiring = forward-mode AD

2. **"Forward- or Reverse-Mode Automatic Differentiation: What's the Difference?"**
   - [arXiv:2212.11088](https://arxiv.org/abs/2212.11088)
   - Algebraic view of AD using semirings and Nagata numbers
   - Unifies forward/reverse mode

3. **Eisner (2002)** - Original gradient semiring for parsing
   - Referenced by DeepProbLog as the source

### Semiring Logic Programming

4. **"Semirings for Probabilistic and Neuro-Symbolic Logic Programming"**
   - [arXiv:2402.13782](https://arxiv.org/pdf/2402.13782)
   - Unifies probabilistic logic programming under semiring

5. **"Neural Semirings"**
   - [CEUR](https://ceur-ws.org/Vol-2986/paper7.pdf)
   - Learn the semiring operations themselves

### Classic

6. **"Algebraic Path Problems"** (Tarjan, others)
   - Semiring framework for graph algorithms
   - Shortest path, transitive closure, etc.

---

## Research Questions

1. **Semiring-valued coalgebraic modal logic?**
   - What's the right generalization of Moss's ∇?

2. **Convergence conditions**
   - When does semiring-valued coalgebra converge to crisp?
   - Different for different semirings?

3. **Which semiring for which task?**
   - Tropical for planning/optimization?
   - Probabilistic for uncertainty?
   - Gradient for learning?

4. **Compositional semirings**
   - Can we compose semirings? (product, coproduct)
   - Learn which composition to use?

5. **Neural semiring + coalgebra**
   - Full generality: learn both the structure (coalgebra) and the algebra (semiring)

---

## The Hierarchy

```
Most General
    │
    ▼
┌─────────────────────────────────────────┐
│  Neural Semiring-valued Coalgebra       │
│  (learn both structure and operations)  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Semiring-valued Coalgebra              │
│  (fixed semiring, learn structure)      │
└─────────────────────────────────────────┘
    │
    ├──► Fuzzy Coalgebra ([0,1], max, ×)
    ├──► Probabilistic Coalgebra (ℝ≥0, +, ×)
    ├──► Tropical Coalgebra (ℝ∪{∞}, min, +)
    │
    ▼
┌─────────────────────────────────────────┐
│  Crisp Coalgebra                        │
│  (Boolean semiring, the target)         │
└─────────────────────────────────────────┘
    │
    ├──► Kripke frames
    ├──► Automata (DFA, NFA, ...)
    ├──► Transition systems
    │
    ▼
Most Specific
```

---

## Connection to Other Ideas

- **coalgebra.md**: Semiring is the right way to "fuzzify" coalgebra
- **meta-learning.md**: Neural semiring = meta-learning at algebraic level
- **neural-symbolic.md**: Gradient semiring bridges logic and backprop
- **scaffold.md**: Framework should support multiple semirings
