# Key References

## Neural-Symbolic & Differentiable Logic

### Modal Logical Neural Networks (MLNNs)
- **Author:** Antonin Sulc (Lawrence Berkeley National Lab)
- **Date:** December 2024
- **Key contribution:** Differentiable Kripke semantics with learnable accessibility relation
- **Links:** [arXiv](https://arxiv.org/abs/2512.03491) | [GitHub](https://github.com/sulcantonin/MLNN_public)
- **Relevance:** Exactly our "fuzzy Kripke → learn classical Kripke" idea

### Logic Tensor Networks (LTN)
- **Key idea:** Propositions as tensors, connectives as differentiable ops
- **Use:** Knowledge base → loss function → gradient descent

### Connectionist Modal Logic
- **Link:** [Paper](https://www.staff.city.ac.uk/~aag/papers/cml.pdf)
- **Note:** Earlier work, not as unified as MLNNs

### Kripke Semantics for Fuzzy Logics
- **Link:** [arXiv:1607.04090](https://arxiv.org/abs/1607.04090)
- **Content:** Conditions for Kripke frames to satisfy fuzzy logic axioms

### Fuzzy Modal Logics
- **Link:** [Springer](https://link.springer.com/article/10.1007/s10958-005-0281-1)
- **Content:** Formal definition of fuzzy Kripke models

---

## Foundational Logic

### Curry-Howard Correspondence
- **Key insight:** Proofs = Programs, Propositions = Types
- **People:** Curry (1930s), Howard (1969)

### Martin-Löf Type Theory
- **Creator:** Per Martin-Löf (1970s)
- **Key insight:** Unified logic + set theory + computation
- **Descendants:** Coq, Agda, Lean

### Hoare Logic
- **Creator:** C.A.R. Hoare (1969)
- **Paper:** "An Axiomatic Basis for Computer Programming"
- **Descendants:** Abstract interpretation, separation logic, static analysis

---

## Modal Logic Classics

### Propositional Dynamic Logic (PDL)
- **Creator:** Vaughan Pratt (1976)
- **Key idea:** Programs inside modal operators
- **Complexity:** EXPTIME-complete, decidable

### Sahlqvist Theorem
- **Creator:** Henrik Sahlqvist (1975)
- **Key insight:** Automatic correspondence + canonicity for large class of formulas

### Van Benthem Characterization
- **Key insight:** Modal logic = bisimulation-invariant fragment of FOL

---

## Generalized Quantifiers

### Mostowski (1957), Lindström (1966)
- **Key insight:** Quantifiers as predicates on sets
- **Lindström's theorem:** FOL is unique with compactness + L-S

### Graded Modal Logic
- **Key idea:** ◇≥n φ means "at least n worlds satisfy φ"

---

## Chomsky Hierarchy & NN

### Neural Networks and the Chomsky Hierarchy
- **Source:** DeepMind, 2022
- **Key finding:** Transformer/LSTM fail beyond regular; need stack/tape to climb hierarchy
- **Relevance:** Agent frameworks are reinventing automata theory

---

## Combinatory Categorial Grammar (CCG)

### Mark Steedman
- **Key idea:** Words have types, combine via combinators
- **Connection:** Parsing = type checking = proof search (Curry-Howard)

---

## Process Algebra

### CCS (Calculus of Communicating Systems)
- **Creator:** Robin Milner
- **Key concept:** Bisimulation as process equivalence

### CSP (Communicating Sequential Processes)
- **Creator:** C.A.R. Hoare

### π-calculus
- **Creators:** Milner, Parrow, Walker
- **Key idea:** Mobile processes, channel passing

---

## Graded Modal Logic + GNN (Related but Different)

### GNN ↔ Graded Modal Logic Expressiveness
- **Barceló et al. (2020)**: GNNs can express exactly graded modal logic
- [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/641f14d06ce46cce5e7cef78e22af0f8-Paper-Conference.pdf)
- **Note:** This is about EXPRESSIVENESS, not learning. Graph is input, not learned.

### Graded Modal Logic and Counting Bisimulation
- [arXiv:1910.00039](https://arxiv.org/abs/1910.00039) (Otto 2019)
- Theory: characterization of graded modal logic via counting bisimulation

### Graded Modal Logic and Message Passing Automata
- [arXiv:2401.06519](https://arxiv.org/abs/2401.06519) (Ahvonen et al. 2024)
- Connection to Weisfeiler-Leman algorithm

### K# Logic (Counting Modalities in Linear Inequalities)
- [Lecture notes](https://perso.ens-lyon.fr/francois.schwarzentruber/teaching/m2-gml/main.pdf) (Schwarzentruber)
- K# captures AC-GNNs, more expressive than FO

### Kripke Frame with Graded Accessibility
- [Studia Logica 1997](https://link.springer.com/article/10.1023/A:1004956418185)
- Fuzzy accessibility r(x,y) ∈ [0,1], but no learning

### Graded Concurrent PDL
- [arXiv:2501.09285](https://arxiv.org/pdf/2501.09285) (Jan 2025)
- Graded + dynamic logic, but no learning

---

## Semiring & Algebraic Logic

### Semirings for Probabilistic and Neuro-Symbolic Logic Programming
- **Date:** 2024
- **Link:** [arXiv:2402.13782](https://arxiv.org/pdf/2402.13782)
- **Key insight:** Unifies probabilistic logic programming under semiring framework

### Neural Semirings
- **Link:** [CEUR](https://ceur-ws.org/Vol-2986/paper7.pdf)
- **Key insight:** Learn the semiring operations (⊕, ⊗) via neural networks

### DeepProbLog (NeurIPS 2018)
- **Link:** [arXiv:1805.10872](https://arxiv.org/abs/1805.10872) | [GitHub](https://github.com/ML-KULeuven/deepproblog)
- **Key insight:** Gradient semiring enables differentiable logic programming
- **Gradient semiring:** element = (value, ∂value/∂θ), operations follow product/sum rule
- 推理时梯度自动算出，不需要分开的 forward/backward pass

### Automatic Differentiation via Semirings
- **Paper:** "Forward- or Reverse-Mode AD: What's the Difference?"
- **Link:** [arXiv:2212.11088](https://arxiv.org/abs/2212.11088)
- **Key insight:** AD = semiring + dual numbers / Nagata numbers
- Unifies forward and reverse mode algebraically

### Deep Differentiable Logic Gate Networks
- **Date:** NeurIPS 2022
- **Link:** [PDF](https://papers.neurips.cc/paper_files/paper/2022/file/0d3496dd0cec77a999c98d35003203ca-Paper-Conference.pdf)
- **Key insight:** Relaxing logic gates via real-valued logic

---

## Category Theory & Deep Learning (Fold/Unfold)

### Categorical Deep Learning is an Algebraic Theory of All Architectures
- **Authors:** Gavranović, Lessard, et al.
- **Date:** ICML 2024
- **Links:** [arXiv:2402.15332](https://arxiv.org/abs/2402.15332)
- **Key insight:** Neural network architectures = algebras & coalgebras
- **Relevance:** RNN unrolling = fold, RNN generation = unfold, encoder-decoder = hylomorphism

### Fundamental Components of Deep Learning (PhD Thesis)
- **Author:** Bruno Gavranović
- **Link:** [PDF](https://www.brunogavranovic.com/assets/FundamentalComponentsOfDeepLearning.pdf)
- **Key insight:** Backpropagation as coalgebra, parametric optics for neural networks

### Folding over Neural Networks
- **Date:** 2022
- **Links:** [arXiv:2207.01090](https://arxiv.org/abs/2207.01090) | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-16912-0_5)
- **Key insight:** Forward propagation = fold, backward propagation = unfold

### Towards a Categorical Foundation of Deep Learning: A Survey
- **Date:** October 2024
- **Link:** [arXiv:2410.05353](https://arxiv.org/abs/2410.05353)
- **Content:** Comprehensive survey of category theory in deep learning

### Backprop as Functor
- **Link:** [arXiv:1711.10455](https://arxiv.org/abs/1711.10455)
- **Key insight:** Backpropagation preserves compositional structure

### Categories for Machine Learning (Course)
- **Link:** [cats.for.ai](https://cats.for.ai/)
- **Content:** Lecture series on categorical ML

### Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire
- **Authors:** Meijer, Fokkinga, Paterson (1991)
- **Key insight:** Original paper on catamorphism, anamorphism, hylomorphism

---

## To Read

- [ ] MLNNs paper in detail
- [ ] Barceló et al. GNN expressiveness paper
- [ ] Fuzzy modal logic meta-theory
- [ ] Graded modal logic decidability results
- [ ] Description Logic complexity landscape
- [ ] Categorical Deep Learning paper
- [ ] Gavranović thesis (backprop as coalgebra)
- [ ] Semirings for Neuro-Symbolic Logic Programming
- [ ] Neural Semirings paper
