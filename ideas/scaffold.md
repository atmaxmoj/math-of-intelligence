# Lean Scaffolding for Custom Modal Logics

## Goal

Build a Lean framework for rapidly defining and verifying custom modal logics.

## Core Features

### 1. Sahlqvist Automation
- Input: axioms for a new □
- Check: is it a Sahlqvist formula?
- If yes: auto-generate frame correspondence + completeness proof
- If no: allow building, mark "completeness unverified"

### 2. Generalized Quantifiers

Beyond □ (∀ over worlds) and ◇ (∃ over worlds):

| Quantifier | Meaning |
|------------|---------|
| ◇≥n, ◇≤n, ◇=n | At least/at most/exactly n worlds |
| M | Majority (most worlds) |
| P≥r | Probability at least r |
| Custom | User-defined |

Why: "at least 3 backup plans" ≠ "some backup plan"

### 3. Two-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Strict Layer (Classical)               │
│  - Full meta-theory                     │
│  - Sahlqvist, completeness, decidability│
│  - For: definition, verification        │
└────────────────────┬────────────────────┘
                     │ fuzzy relaxation
                     ↓
┌─────────────────────────────────────────┐
│  Soft Layer (Differentiable)            │
│  - [0,1] truth values                   │
│  - Learnable accessibility relation     │
│  - For: NN integration, training        │
└─────────────────────────────────────────┘
```

## Key Insight: Fuzzy Kripke as Differentiable Relaxation

Fuzzy Kripke semantics allows:
- R: W×W → [0,1] (fuzzy accessibility)
- V: W×P → [0,1] (fuzzy valuation)
- All parameters differentiable

Training can push values toward 0/1, effectively learning classical Kripke models via gradient descent.

## Potential Novel Contribution

**Differentiable Graded Modal Logic**

### Literature survey (Jan 2025)

| Work | Learns R? | Graded quantifiers? | Differentiable? |
|------|-----------|---------------------|-----------------|
| MLNNs (Sulc 2024) | ✓ | ✗ (only □/◇) | ✓ |
| GNN ↔ Graded Modal (Barceló et al.) | ✗ (graph is input) | ✓ | ✓ |
| Graded Accessibility Kripke (1997) | ✗ | ✓ (fuzzy R) | N/A |
| Graded Concurrent PDL (2025) | ✗ | ✓ | ✗ |

**Key distinction:**
- GNN + Graded Modal work: studies EXPRESSIVENESS (what can GNN express?)
- Graph structure is INPUT, not learned
- MLNNs: LEARNS R, but only □/◇

**The gap:** Learning Kripke structures with graded quantifiers via differentiable relaxation.

**Our combination:** Fuzzy relaxation of graded/generalized quantifiers, enabling gradient-based learning of Kripke structures that satisfy counting constraints.

### How to soften graded modalities

| Quantifier | Discrete | Soft (differentiable) |
|------------|----------|----------------------|
| ◇≥n φ | \|{v: Rwv ∧ φ(v)}\| ≥ n | σ(Σᵥ R(w,v)·v(φ,v) - n) |
| ◇≤n φ | \|{v: Rwv ∧ φ(v)}\| ≤ n | σ(n - Σᵥ R(w,v)·v(φ,v)) |
| M φ | majority | σ(Σᵥ R·v(φ) - Σᵥ R·v(¬φ)) |
| P≥r φ | probability ≥ r | σ(normalized_sum - r) |

Where σ is sigmoid or other smooth threshold, R(w,v) ∈ [0,1], v(φ,v) ∈ [0,1].

### Why this matters

- **Expressiveness:** "at least 3 safe options" is more useful than "some safe option"
- **Learnability:** gradient descent can find Kripke structures satisfying graded constraints
- **Novel:** this specific combination doesn't seem to exist in literature

### Research questions

1. What are the right soft approximations for each quantifier?
2. Does training converge to crisp (0/1) structures?
3. What meta-theoretic properties transfer from strict to soft layer?
4. Can we prove approximation bounds?

---

## The Core Insight: Logic Can Finally Learn

**The historical gap:**
- Logic (since Frege): precise, verifiable, but NO notion of learning
- ML: learning, but black-box, no logical structure

**The solution:**
```
Fuzzy Kripke (continuous, differentiable)
        ↓ gradient descent = LEARNING
        ↓ converge to 0/1
Classical Kripke (discrete logical structure)
        ↓ extract / compile
Executable Program (state machine / code)
```

**What this achieves:**
1. Logical structures can be LEARNED (not just defined by humans)
2. The learned structure IS a program (Kripke = transition system = state machine)
3. Satisfies logical constraints (graded modalities as loss functions)
4. Interpretable (you can inspect R and V)

**Why this wasn't done before:**
- Logic is discrete → can't do gradient descent
- Fuzzy logic existed but wasn't connected to NN/learning
- Logic community and ML community don't talk

**This bridges:**
- Fuzzy logic (continuization)
- Kripke semantics (logical structure)
- Gradient descent (learning)
= **Learnable Logic**

---

## Open Questions

- How to preserve meta-theoretic properties in soft layer?
- What's the right fuzzy semantics (which t-norm, which implication)?
- Interaction between different generalized quantifiers in soft setting?
- How to compile learned Kripke structure to efficient code?
- Convergence guarantees: will it always reach 0/1?
