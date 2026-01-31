# Beyond ∀/∃: Generalized Quantifiers in Modal Logic

## The Critique

Kripke inherited Frege's ∀/∃ fixation:
- □ = ∀ over accessible worlds
- ◇ = ∃ over accessible worlds

But natural language has many more:

```
most, few, many, several
exactly one, at least n, at most n
more than half, almost all
only, except
```

For agents: "at least 3 escape routes" ≠ "some escape route"

## Generalized Quantifier Theory

Mostowski 1957, Lindström 1966.

Quantifiers as predicates on sets:
```
∀x.P(x)        true iff |P| = |Domain|
∃x.P(x)        true iff |P| > 0
(most x)P(x)   true iff |P| > |¬P|
(≥n x)P(x)    true iff |P| ≥ n
(few x)P(x)    true iff |P|/|Domain| < threshold
```

## Graded Modal Logic

Counting over accessible worlds:

```
◇≥n φ    "at least n accessible worlds satisfy φ"
◇≤n φ    "at most n accessible worlds satisfy φ"
◇=n φ    "exactly n accessible worlds satisfy φ"
```

Semantics:
```
M,w ⊨ ◇≥n φ  iff  |{v : Rwv ∧ M,v ⊨ φ}| ≥ n
```

## Majority Modal Logic

```
Mφ   "most accessible worlds satisfy φ"
```

Interesting because:
- Cannot be defined with ∀/∃ (Lindström)
- Very common in everyday reasoning
- Breaks compactness

## Probabilistic Modal Logic

```
P≥r(φ)   "probability of φ is at least r"
```

Requires probability distribution over worlds, not just accessibility.

## Trade-offs

| | ∀/∃ only | Generalized |
|---|----------|-------------|
| Compactness | ✓ | Often ✗ |
| Löwenheim-Skolem | ✓ | Often ✗ |
| Decidability | Case-by-case | Usually worse |
| Expressiveness | Limited | Rich |

Lindström's theorem: first-order logic is unique in having both compactness and L-S. Adding quantifiers breaks something.

## For the Scaffold

Support defining custom quantifiers:
```lean
-- User defines "most" quantifier
def most_accessible (R : W → W → Prop) (φ : W → Prop) (w : W) : Prop :=
  |{v | R w v ∧ φ v}| > |{v | R w v ∧ ¬φ v}|
```

Generate what meta-theory we can, flag what's unknown.

## Description Logic Connection

DL already has number restrictions:
```
(≥ 3 hasChild.Doctor)   "at least 3 children are doctors"
(≤ 1 hasSpouse)         "at most 1 spouse"
```

OWL (Web Ontology Language) uses this. Industrial-strength.
