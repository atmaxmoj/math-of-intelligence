# Research Directions & Inspirations

This folder contains research ideas, directions, and references that emerged from studying modal logic.

## Index

| File | Topic |
|------|-------|
| [coalgebra.md](coalgebra.md) | **Core theory**: Coalgebra as the right abstraction |
| [semiring.md](semiring.md) | **Core theory**: Semiring as the right continuization |
| [meta-learning.md](meta-learning.md) | **Deep insight**: Learning is coalgebraic, so meta-learning is too |
| [universal_learner.md](universal_learner.md) | **Deep insight**: Fixed point bypasses uncomputability of K |
| [scaffold.md](scaffold.md) | Lean scaffolding for custom modal logics |
| [neural-symbolic.md](neural-symbolic.md) | Differentiable logic, fuzzy Kripke, NN integration |
| [agents.md](agents.md) | Modal logic for AI agents, LLM integration |
| [quantifiers.md](quantifiers.md) | Beyond ∀/∃: generalized quantifiers in modal logic |
| [references.md](references.md) | Key papers and resources |

## Core Vision

**Logic can finally learn.**

The fundamental problem: Logic (since Frege, 2000+ years) has no notion of learning. Structures are defined, not learned.

The solution:
```
Semiring-valued Coalgebra (continuous) → gradient descent → Crisp Coalgebra (discrete) → Program
```

Two orthogonal generalizations:
- **Coalgebra** (not just Kripke): unifies automata, transition systems, Markov chains, ...
- **Semiring** (not just fuzzy): unifies probabilistic, tropical, gradient, ...

The hierarchy:
```
Neural Semiring-valued Coalgebra  ← learn both structure AND operations
            ↓
Semiring-valued Coalgebra         ← learn structure, fixed semiring
            ↓
Crisp Coalgebra = Program         ← discrete, executable
```

And since learning itself is coalgebraic (fold/unfold), meta-learning is too. It's coalgebras all the way up.

## Status

These are research directions, not implementations. The actual study of modal logic meta-theory happens in the main project; these files capture where it might lead.
