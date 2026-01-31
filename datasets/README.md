# Benchmark Datasets

Datasets for testing **Semiring-valued Coalgebra Learning**.

## Priority: Automata Benchmarks (直接是 Coalgebra!)

### automata.cs.ru.nl ⭐⭐⭐ 首选
- **Source:** [Automata Wiki](https://automata.cs.ru.nl/) | [Downloads](https://automata.cs.ru.nl/Downloads)
- **Paper:** Neider, Smetsers, Vaandrager, Kuppens (2019)
- **License:** MIT
- **Contains:**
  - DFA (deterministic finite automata)
  - Moore machines
  - Mealy machines
  - Interface automata
  - Register automata
- **Categories:** Industry, Protocol, Security, Random, Toy
- **Real-world models:** 银行卡协议、嵌入式系统、电路
- **Format:** .dot files
- **Why perfect:** Automata = Coalgebra，可直接对比 L* 算法

### Tomita Grammars
- **What:** 7 个经典正则语言
- **Task:** 从正负样本学习 DFA
- **Why:** 最简单的起步点，所有 automata learning 论文都测这个

---

## Downloaded

### FRoG - Fuzzy Reasoning of Generalized Quantifiers
- **Source:** [GitHub](https://github.com/Nativeatom/FRoG) | [HuggingFace](https://huggingface.co/datasets/GAIR/FRoG)
- **Paper:** EMNLP 2024
- **Task:** Reasoning with generalized quantifiers ("most", "few", "at least")
- **Why relevant:** Directly tests graded quantifiers. LLMs perform poorly.
- **Key insight:** Inverse scaling effect - bigger models not necessarily better
- **Load data:**
  ```python
  from datasets import load_dataset
  frog = load_dataset("GAIR/FRoG", "mask_quant", "easy")  # or "hard"
  ```

### CLUTRR - Compositional Language Understanding with Text-based Relational Reasoning
- **Source:** [GitHub](https://github.com/facebookresearch/clutrr)
- **Paper:** EMNLP 2019
- **Task:** Infer family relationships from text (transitive reasoning)
- **Why relevant:** Tests learning relational composition. GNNs struggle with length generalization.

### StepGame - Multi-hop Spatial Reasoning
- **Source:** [GitHub](https://github.com/ZhengxiangShi/StepGame)
- **Paper:** AAAI 2022
- **Task:** Spatial reasoning from text descriptions
- **Why relevant:** Multi-hop reasoning. Neural-symbolic methods show 40-50% improvement.

### MLNN_public - Modal Logical Neural Networks
- **Source:** [GitHub](https://github.com/sulcantonin/MLNN_public)
- **Paper:** arXiv 2024
- **Contains:** Code + experiments on Diplomacy trust, CaSiNo negotiation
- **Why relevant:** Baseline for differentiable Kripke learning (but only □/◇)

### ARC-AGI - Abstraction and Reasoning Corpus
- **Source:** [GitHub](https://github.com/fchollet/ARC-AGI)
- **Paper:** "On the Measure of Intelligence" (Chollet 2019)
- **Task:** Few-shot abstract reasoning (从 2-5 个示例推断规则)
- **Format:** JSON，input/output grids
- **Size:** 800 tasks (400 training + 400 evaluation)
- **Difficulty:** 极难。LLM ~20%，人类 ~85%
- **Why relevant:** 需要真正的 program induction
- **Challenge:** ARC Prize ($1M+) 目前最高分 ~55%

---

## Not Downloaded (Available Online)

### bAbI
- **Source:** [Facebook Research](https://research.facebook.com/downloads/babi/)
- **Task:** 20 reasoning tasks
- **Get:** `wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz`

### CaSiNo (Negotiation)
- **Source:** [HuggingFace](https://huggingface.co/datasets/kchawla123/casino)
- **Task:** Negotiation deception detection
- **Get:** `datasets.load_dataset("kchawla123/casino")`

## Experiment Plan

### Phase 0: Tomita Grammars（最简单）
- 7 个正则语言
- 从 input-output traces 学习 DFA
- 用 gradient semiring + coalgebra
- 验证：学到的 DFA 是否正确？

### Phase 1: automata.cs.ru.nl
- 下载真实协议模型
- 生成 traces 作为训练数据
- 学习 semiring-valued coalgebra
- 收敛到 crisp → 对比 ground truth
- Baseline: L* (query-based), RNN

### Phase 2: FRoG (Graded Quantifiers)
- 测试 soft graded modalities (◇≥n, M, ...)
- 学习满足 graded modal constraints 的 Kripke 结构

### Phase 3: Full Evaluation
- Compare:
  - L* (query-based, discrete)
  - RNN → WFA extraction
  - MLNNs (only □/◇)
  - **Ours: Semiring Coalgebra (gradient-based, general)**
- Metrics:
  - Accuracy (learned structure correctness)
  - Sample efficiency
  - Generalization to unseen inputs
