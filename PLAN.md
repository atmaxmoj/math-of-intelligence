# Modal Logic Meta-Theory Study Plan

目标：系统学习模态逻辑元理论，边学边写LaTeX书，达到与BRV (Blackburn, de Rijke, Venema) 睥睨的水平。

---

## 目录结构

```
modal-logic/
├── 01-basics/          # 语法、Kripke语义基础
├── 02-model-theory/    # Bisimulation、bounded morphism
├── 03-completeness/    # Canonical model、完备性证明
├── 04-correspondence/  # Frame definability、Sahlqvist定理
├── 05-decidability/    # Finite model property、可判定性
├── 06-combining/       # Fusion、product、多模态逻辑
├── 07-handbook/        # Handbook of Modal Logic章节
└── book/               # 你的LaTeX书稿 (待创建)
```

---

## 学习计划

### Phase 1: 基础 (Week 1-2)
- [ ] 读 `01-basics/zalta-basic-concepts.pdf` - 入门概念
- [ ] 读 `01-basics/van-benthem-open-minds.pdf` - 前5章
- [ ] 读 `01-basics/cerami-modal1-intro.pdf` - 历史背景
- [ ] 写书：Chapter 1 - 语法与语义

### Phase 2: 模型论 (Week 3-4)
- [ ] 读 `02-model-theory/goranko-otto-model-theory.pdf` - 核心章节
- [ ] 读 `02-model-theory/blackburn-vanbenthem-semantic.pdf`
- [ ] 读 `02-model-theory/bezhanishvili-bisimulation.pdf`
- [ ] 读 `02-model-theory/stirling-bisimulation-logic.pdf`
- [ ] 写书：Chapter 2 - Bisimulation与模型构造

### Phase 3: 完备性 (Week 5-6)
- [ ] 读 `03-completeness/kuznetsov-logic2.pdf` - canonical model
- [ ] 读 `03-completeness/szczuka-applied-logic.pdf`
- [ ] 读 `03-completeness/freiburg-modal-logics.pdf` - 最完整
- [ ] 读 `01-basics/cerami-modal3-soundness.pdf`
- [ ] 写书：Chapter 3 - Soundness与Completeness

### Phase 4: 对应理论 (Week 7-8)
- [ ] 读 `04-correspondence/sahlqvist-ceur.pdf`
- [ ] 读 `04-correspondence/sahlqvist-sopml.pdf`
- [ ] 读 `04-correspondence/sahlqvist-sabotage.pdf`
- [ ] 写书：Chapter 4 - Frame Definability与Sahlqvist定理

### Phase 5: 可判定性 (Week 9-10)
- [ ] 读 `05-decidability/cerami-modal7-fmp.pdf`
- [ ] 读 `05-decidability/cerami-modal9-decidability.pdf`
- [ ] 读 `05-decidability/urquhart-decidability-fmp.pdf`
- [ ] 写书：Chapter 5 - Finite Model Property与Decidability

### Phase 6: 组合逻辑 (Week 11-12)
- [ ] 读 `06-combining/fusions-revisited.pdf`
- [ ] 读 `06-combining/products-modal-logics.pdf`
- [ ] 读 Stanford Encyclopedia: Combining Logics
- [ ] 写书：Chapter 6 - 多模态逻辑的组合

### Phase 7: 深化与完善 (Week 13+)
- [ ] 选读 `07-handbook/` 中的相关章节
- [ ] 整合、修订书稿
- [ ] 考虑Lean形式化

---

## 核心概念清单

学完后你应该能够清晰解释：

### 语法与语义
- [ ] Modal language L(◇,□)
- [ ] Kripke frame (W, R)
- [ ] Kripke model (W, R, V)
- [ ] Satisfaction relation ⊨
- [ ] Validity in a frame vs. validity in a model

### 模型论
- [ ] Bisimulation Z ⊆ W × W'
- [ ] Bounded morphism (p-morphism)
- [ ] Generated submodel
- [ ] Disjoint union
- [ ] Tree unravelling
- [ ] Hennessy-Milner theorem

### 完备性
- [ ] Normal modal logic
- [ ] Maximal consistent set (MCS)
- [ ] Canonical frame F_L
- [ ] Canonical model M_L
- [ ] Truth lemma
- [ ] Completeness via canonical models

### 对应理论
- [ ] Standard translation ST_x
- [ ] Frame correspondence
- [ ] First-order definability
- [ ] Sahlqvist formulas
- [ ] Canonicity

### 可判定性
- [ ] Finite model property (FMP)
- [ ] Filtration
- [ ] Decidability via FMP
- [ ] Complexity classes (PSPACE, EXPTIME)

### 组合
- [ ] Fusion L₁ ⊗ L₂
- [ ] Product L₁ × L₂
- [ ] Transfer theorems

---

## 写书建议

1. **每读一个PDF，立即写笔记** - 用自己的话复述证明
2. **证明要自己重做** - 不要只是抄，要理解每一步
3. **画图** - Kripke frame的图示很重要
4. **找反例** - 对每个定理想：如果条件不满足会怎样？
5. **用Lean验证** - 关键定理可以尝试形式化

---

## 参考书目（非免费，图书馆可能有）

- Blackburn, de Rijke, Venema. *Modal Logic*. Cambridge, 2001. (BRV)
- Chagrov, Zakharyaschev. *Modal Logic*. Oxford, 1997.
- Hughes, Cresswell. *A New Introduction to Modal Logic*. Routledge, 1996.
- Chellas. *Modal Logic: An Introduction*. Cambridge, 1980.

---

Created: 2026-01-28
