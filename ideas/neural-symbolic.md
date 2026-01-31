# Neural-Symbolic Integration via Modal Logic

## The Fundamental Problem: Logic Has No Learning

Since Frege (1879), logic has been about **defining** structures, not **learning** them:
- Axioms are written by humans
- Inference rules are defined by humans
- Kripke frames are constructed by humans

**2000+ years of logic, zero notion of learning.**

## The Solution: Fuzzy Kripke as Differentiable Relaxation

```
Fuzzy Kripke (R, V ∈ [0,1])
        ↓ gradient descent = LEARNING
        ↓ values converge to 0/1
Classical Kripke (discrete)
        ↓
Executable Program (state machine)
```

**Key insight:** Kripke structure = program semantics (PDL perspective)

So learning a Kripke structure = learning a program, but at the semantic level.

## The Bridge Problem (Reframed)

| Has | Logic | NN/Optimization |
|-----|-------|-----------------|
| Learning | ✗ | ✓ |
| Verification | ✓ | ✗ |
| Precision | ✓ | ✗ |
| Semantics | ✗ | ✓ |

**Fuzzy Kripke bridges this:** Learn via gradients → converge to discrete logic → get executable program.

## Why Fuzzy Logic Bridges

Classical logic: discrete {0,1}, non-differentiable → can't do gradient descent

Fuzzy logic: continuous [0,1], differentiable → gradient descent works

```
Classical Logic ←── too hard, discrete
       ↓
Fuzzy relaxation
       ↓
Differentiable Logic ←→ Optimization / NN
```

## Fuzzy Kripke Semantics

### Option 1: Fuzzy accessibility
R: W×W → [0,1] (degree of accessibility)

### Option 2: Fuzzy valuation
V: W×P → [0,1] (degree of truth)

### Option 3: Both fuzzy
Full continuous relaxation.

### Semantics
```
v(□φ, w) = inf { R(w,u) → v(φ,u) : u ∈ W }
v(◇φ, w) = sup { R(w,u) ⊗ v(φ,u) : u ∈ W }
```

Where → is fuzzy implication, ⊗ is t-norm.

## Learning Kripke Structures

Key idea: parameterize Kripke model, train via gradient descent

```python
class FuzzyKripkeModel:
    R = Parameter(...)  # accessibility, learnable
    V = Parameter(...)  # valuation, learnable

loss = 1 - model.eval(formula)  # want formula true
loss += sparsity_term           # push toward 0/1
optimizer.minimize(loss)
```

Result: learn classical Kripke model that satisfies constraints!

## Existing Work

### Modal Logical Neural Networks (MLNNs)
- Sulc, 2024
- Exactly this idea: differentiable Kripke + learnable R
- [arXiv](https://arxiv.org/abs/2512.03491) | [GitHub](https://github.com/sulcantonin/MLNN_public)

### Logic Tensor Networks (LTN)
- Propositions as tensors
- Connectives as differentiable ops
- Knowledge base → loss function

## Potential Novel Direction: Differentiable Graded Modal Logic

**Gap in existing work:**
- MLNNs: differentiable Kripke, but only □/◇
- Graded modal logic: ◇≥n, ◇≤n, M, P≥r — but discrete, not differentiable

**Idea:** Soft approximations of graded quantifiers

```
◇≥n φ  →  σ(Σᵥ R(w,v)·v(φ,v) - n)
M φ    →  σ(Σ R·v(φ) - Σ R·v(¬φ))
```

This enables **learning Kripke structures that satisfy graded modal constraints**.

For agents: "learn a model where at least 3 escape routes exist" — more useful than just "some escape route".

See `scaffold.md` for details.

---

## Open Questions

- Meta-theory for fuzzy modal logic is weak (no Sahlqvist equivalent)
- Which t-norm/implication is best for learning?
- How to ensure learned model is "close enough" to classical?
- Approximation bounds for soft graded modalities?
