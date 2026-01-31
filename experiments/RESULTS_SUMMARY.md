# Distance Matching 实验结果汇总

## 核心假说

**Distance Matching 原则**: 好的参数化 = 参数距离 ≈ 函数距离

如果 θ₁ 和 θ₂ 在参数空间中接近，则 f(θ₁) 和 f(θ₂) 在函数空间中也应该接近（反之亦然）。

## 理论框架

### Loss Function 诱导 Behavioral Metric

关键洞察：**Loss function 定义了 behavioral metric，不是随意选择。**

```
Loss Function (MSE: ||f(X) - Y||²)
      ↓ 诱导
输出空间 R^n 上的 L2 metric
      ↓ 诱导
函数空间上的 metric: d(f,g) = ||f(X) - g(X)||₂
      ↓ 就是
Behavioral metric
```

| Loss Function | 诱导的 Metric |
|---------------|---------------|
| MSE | L2 |
| MAE | L1 |
| Cross-entropy | KL divergence |
| Wasserstein | Wasserstein |

### κ(J) 是"好学"的精确度量

设 J = ∂f/∂θ 是 Jacobian，条件数 κ(J) = σ_max / σ_min。

**定理**:
- κ(J) 小 → Hessian 条件数 κ(H) = κ(J)² 小 → 梯度下降收敛快
- κ(J) 大 → 收敛慢或不稳定

**更精确的结论**: κ(J) 决定的不是确定的收敛速度，而是收敛速度的**分布**：
- κ(J) 小 → 分布窄（所有初始化都收敛好）
- κ(J) 大 → 分布宽（取决于运气/初始化）

### 范畴论连接

```
Loss Function
    ↓ 定义
Lawvere Metric Space (输出空间)
    ↓ 诱导
Behavioral Metric (函数空间)
    ↓
Distance Matching = γ: Θ → F 是 bi-Lipschitz 映射
    ↓
κ(J) = Lipschitz 常数的比值
```

## 度量指标

- **Correlation**: 参数距离和函数距离的 Pearson 相关系数
- **CV**: 比值 d_func/d_param 的变异系数

评价标准:
- ★★★: corr > 0.9 且 cv < 0.2 (优秀匹配)
- ★★: corr > 0.7 且 cv < 0.4 (良好)
- ★: corr > 0.5 (一般)
- ✗: 其他 (差)

## 关键实验结果

### 1. 冗余是第一破坏因素

| 模型 | Corr | CV | 评价 |
|------|------|-----|------|
| Linear (无冗余) | 0.956 | 0.07 | ★★★ |
| Linear a-b (2x 冗余) | 0.476 | 0.37 | ✗ |
| Linear a-b+c-d (4x 冗余) | 0.261 | 0.25 | ✗ |
| Matrix A@B (矩阵分解) | 0.260 | 0.32 | ✗ |

**结论**: 冗余参数化严重破坏 distance matching。越冗余越差。

### 2. 深度是第二破坏因素 (即使无非线性!)

| 模型 | Corr | CV | 评价 |
|------|------|-----|------|
| 1-layer linear | 0.973 | 0.065 | ★★★ |
| 2-layer linear (W₂@W₁@x) | 0.263 | 0.285 | ✗ |
| 3-layer linear | 0.312 | 0.343 | ✗ |
| 4-layer linear | 0.276 | 0.405 | ✗ |

**关键发现**: 深度线性网络，函数空间完全相同（都是线性函数），但 distance matching 崩溃！

**原因**: W₂ @ W₁ = W₂' @ W₁' 有无穷多解，引入隐式冗余。

### 3. 稀疏性帮助 distance matching

| 模型 | Corr | CV | 评价 |
|------|------|-----|------|
| Dense linear (20 params) | 0.914 | 0.068 | ★★★ |
| Sparse linear (5 params) | 0.979 | 0.064 | ★★★ |
| Soft-sparse (sigmoid mask) | 0.476 | 0.177 | ✗ |

**结论**: 固定稀疏模式 → 更少参数 → 更好匹配。软稀疏引入冗余。

### 4. 矩阵分解总是有害的

| 模型 | Corr | CV | 评价 |
|------|------|-----|------|
| Direct linear | 0.959 | 0.066 | ★★★ |
| Low-rank (k=2) | 0.313 | 0.277 | ✗ |
| Low-rank (k=5) | 0.288 | 0.254 | ✗ |
| Low-rank (k=10) | 0.221 | 0.229 | ✗ |
| SVD param | 0.315 | 0.368 | ✗ |

**结论**: 任何矩阵分解都引入冗余，破坏 distance matching。

### 5. 非线性的影响

| 模型 | Corr | CV | 评价 |
|------|------|-----|------|
| 2-layer Linear (identity) | 0.134 | 0.254 | ✗ |
| 2-layer ReLU | 0.196 | 0.262 | ✗ |
| 2-layer Tanh | 0.185 | 0.225 | ✗ |
| 2-layer Sigmoid | 0.170 | 0.585 | ✗ |

**重要**: 即使是 identity 激活（纯线性），2层结构也表现很差！

**结论**: 非线性本身不是主因，**深度带来的冗余**才是。

### 6. Attention 机制

| 模型 | Corr | CV | 评价 |
|------|------|-----|------|
| Linear-pool | 0.923 | 0.143 | ★★★ |
| Self-Attention (QKV) | 0.314 | 0.380 | ✗ |
| Bilinear (无 softmax) | 0.372 | 0.408 | ✗ |

**结论**: QKV 三组矩阵引入冗余。Softmax 进一步复杂化。

### 7. 输入维度的影响

**Linear 模型**:
- d=5: 0.984
- d=10: 0.946
- d=20: 0.925
- d=50: 0.804

**MLP 模型**:
- d=5: 0.345
- d=10: 0.217
- d=20: 0.106
- d=50: -0.008

**结论**:
- Linear 在所有维度都表现良好
- MLP 随维度增加而恶化

### 8. 初始化尺度无影响

Linear 和 MLP 在不同尺度 (0.1 到 2.0) 下表现一致。

**结论**: Distance matching 是结构性质，与初始化无关。

## 核心洞察

### Distance Matching 的破坏因素 (按重要性排序)

1. **冗余参数化** - 最直接的破坏因素
   - a-b 表示
   - 矩阵分解 A@B

2. **深度** - 即使是线性网络
   - W₂ @ W₁ 有无穷多分解
   - 每增加一层，冗余指数增长

3. **复杂非线性** - 加剧深度问题
   - Softmax, Attention
   - 但不是根本原因

### 为什么这重要?

Distance matching 与优化效率直接相关：
- 好的匹配 → 梯度方向正确 → 收敛快
- 差的匹配 → 梯度方向误导 → 收敛慢

### 这解释了什么?

1. **为什么 Linear Regression 是最优的**: 无冗余、无深度、距离完美匹配

2. **为什么深度网络难训练**: 深度 → 冗余 → distance mismatch

3. **为什么 ResNet 有效**: Skip connection 让网络更接近 identity (减少有效深度?)

4. **为什么 CNN 对图像有效**: 参数共享减少冗余 (但我们的实验结果不明显)

## 开放问题

1. **ResNet 的真正机制**: Skip connection 如何改善 distance matching?

2. **Attention 的成功**: 如果 distance matching 差，为什么 Transformer 这么成功?
   - 可能: 表达能力的提升超过了优化的困难
   - 可能: 其他因素补偿 (如 LayerNorm, 残差)

3. **如何设计好的架构**: 在保持表达能力的同时，最小化冗余?

4. **理论下界**: 对于特定函数类，distance matching 的最优上界是什么?

## 经典任务-架构匹配实验

### 完美匹配 (κ ≈ 1)

| 任务 | 正确架构 | κ(J) | Loss | CV |
|------|---------|------|------|-----|
| 置换不变 sum(x) | DeepSets | **1.0** | 0.0000 | 0.87 |
| 乘法 x₁·x₂ | 乘法门 | **1.0** | 0.0000 | **0.00** |
| 计数 count(x>0) | Counter | **1.0** | 0.1010 | 0.05 |
| 稀疏 x₀+x₁ | Sparse | **1.2** | 0.0000 | 0.78 |

乘法门是完美例子：κ=1.0, loss=0, CV=0（所有seed都完美收敛）

### 不匹配架构

| 任务 | 错误架构 | κ(J) | Loss | 问题 |
|------|---------|------|------|------|
| 线性 | MLP | ∞ | 2.54 | 冗余 |
| 置换不变 | MLP | 336 | 2.02 | 冗余 |
| 乘法 | Linear | 1.1 | 1.25 | 表达力不够 |

### 两种失败模式

1. **表达力不够**: κ 小但 loss 高（无法表示目标）
2. **冗余**: κ 大，训练不稳定（参数过多）

**最佳架构** = 表达力刚好 + 无冗余 = κ(J) 最小化

## Seed 诱导的分布实验 (30 seeds)

| 架构 | κ(J) | Mean Loss | Std Loss | CV |
|------|------|-----------|----------|-----|
| Linear 直接 | 1.5 | 0.008 | 0.000 | **0.00** |
| MLP 2层 | ∞ | 0.59 | 0.57 | **0.96** |
| ResNet 2层 | 15667 | 0.11 | 0.03 | 0.26 |

**结论**: κ(J) 决定收敛结果的方差，不只是均值。

## 相关工作

### 已有研究

| 概念 | 来源 | 我们的术语 |
|------|------|-----------|
| Dynamical isometry | Saxe et al., 2013 | κ(J) ≈ 1 |
| NTK eigenvalue / trainability | Jacot et al., 2018 | λ_min 决定收敛 |
| Natural gradient | Amari, 1998 | 用 F⁻¹ 校正距离失配 |
| Coalgebraic behavioral metric | Baldan et al., 2018 | Kantorovich lifting |

### 我们可能的贡献

1. **Loss → Behavioral Metric**: 损失函数诱导 Lawvere metric（不是独立选择）
2. **分布视角**: κ(J) 决定 seed 诱导的收敛分布，不只是均值
3. **任务-架构匹配实验**: 系统验证匹配 = κ(J) 最小化
4. **范畴论框架**: bi-Lipschitz 映射视角

### 参考文献

- Saxe et al. "Exact solutions to the nonlinear dynamics of learning in deep linear networks" (2013)
- Jacot et al. "Neural Tangent Kernel" (2018)
- Amari "Natural Gradient Works Efficiently in Learning" (1998)
- Baldan et al. "Coalgebraic Behavioral Metrics" (LMCS 2018)
- "Using Enriched Category Theory to Construct the Nearest Neighbour Classification Algorithm" (2023)

## Coalgebra 参数化实验

### 核心发现：One-step Kantorovich 匹配参数距离

对于 Soft DFA (functor H = 2 × (-)^Σ)，测试不同 behavioral metric 与参数距离的匹配：

| Behavioral Metric | Param Metric | Correlation | CV |
|-------------------|--------------|-------------|-----|
| Structure (trivial) | Euclidean | **1.000** | 0.000 |
| One-step | Fisher-Rao | **0.802** | 0.101 |
| One-step | Euclidean | 0.761 | 0.106 |
| Fixed-point (iterated) | Euclidean | 0.435 | 0.352 |
| Language (global) | Euclidean | 0.279 | 0.490 |

### 关键洞察

1. **One-step metric 匹配好，global metric 匹配差**
   - 一步 Kantorovich lifting: Corr = 0.80
   - 完整语言距离: Corr = 0.28

2. **深度类比**

   | Automata | Neural Networks |
   |----------|-----------------|
   | 词长 \|w\| | 网络深度 |
   | One-step metric | 单层比较 |
   | Language metric | 端到端比较 |
   | 长词破坏匹配 | 深度破坏匹配 |

3. **自然涌现原则**
   - Functor H 定义参数空间结构
   - One-step Kantorovich lifting 定义 behavioral metric
   - 两者自然匹配！
   - 结论：局部目标容易优化，全局目标困难

### 这解释了

- RNNs 难以处理长序列（全局目标，匹配差）
- Attention 帮助（减少"有效深度"）
- Curriculum learning 有效（从短序列开始）

## RL 中的 Bisimulation Metrics (相关工作)

### Ferns, Panangaden, Precup (2004-2011)

- 为 MDP 开发 **bisimulation metrics** (bisimulation 的量化松弛)
- 关键定理：bisimulation metric 给出 **最优值函数的紧界**:
  ```
  |V*(s) - V*(t)| ≤ (1/(1-γ)) · d_bisim(s, t)
  ```
- 使用 Kantorovich functional 递归计算

### Deep Bisimulation for Control (Zhang et al., 2020)

- 训练 encoder 使得 **latent distance = bisimulation distance**
- 理论结果：最优值函数关于 bisimulation metric 是 **Lipschitz**
- 提供表示学习的次优性界

### 与我们的框架的联系

```
DBC: 学习参数使得 latent distance = behavioral distance
我们: 检验参数距离是否 ≈ behavioral distance

两者都是同一原则的表现：
好的表示保持行为结构
```

## MasterCard Loss Function 对比实验

### 实验设计

只改变 loss function，其他完全相同（模型、数据、优化器、epochs）：
- One-step: 每一步的 CE 平均
- Final-only: 只看最后一步的 CE
- Last-5: 最后 5 步的 CE
- Weighted: 加权 CE（后面步骤权重更大）

### 学习效果

| Loss Function | Symbol Acc | Seq Acc |
|---------------|------------|---------|
| One-step | 99.8% | **94.0%** |
| Final-only | 86.2% | **10.3%** |
| Last-5 | 86.7% | **19.7%** |
| Weighted | 86.5% | **15.3%** |

**核心发现**：只改变 loss，sequence accuracy 从 94% 降到 10%！

### Jacobian 有效秩分析

| Loss Type | J rows | Eff Rank | Actual Acc |
|-----------|--------|----------|------------|
| onestep | 30 | **16.48** | 94.0% |
| final | 1 | **1.00** | 10.3% |
| last5 | 5 | **4.30** | 19.7% |
| weighted | 30 | 14.41 | 15.3% |

Correlation(effective_rank, accuracy) = 0.67

### 理论基础：Rank 和信息量

**核心连接：Jacobian → NTK → 收敛**

1. **NTK = J^T @ J**
   - Jacobian 的秩 = NTK 的秩
   - NTK 的特征值分布决定收敛速度
   - [Jacot et al., 2018](https://arxiv.org/abs/1806.07572)

2. **收敛条件**
   - λ_min(NTK) > 0 是全局收敛的充分条件
   - 如果 NTK 是 rank-deficient，某些方向 **完全无法学习**

3. **Gradient Diversity** [Yin et al., 2018](https://arxiv.org/abs/1706.05699)
   - 梯度多样性 ≈ Jacobian 的有效秩
   - 高多样性 → 收敛快，泛化好
   - 低多样性 → 性能下降

**应用到实验**：

| Loss | J rows | Eff Rank | NTK 性质 | 理论预测 |
|------|--------|----------|----------|----------|
| One-step | 30 | 16 | ~16 个非零特征值 | 能学 |
| Final-only | 1 | 1 | Rank-1 | 只能学 1 个方向 |

**关键洞察**：
- Final-only 不是"难学"，是 **大部分参数根本无法学**
- Jacobian rank = 梯度能传播的独立方向数 = 可学习的信息量
- Rank 低 → 信息瓶颈 → 学不到

### 与 Coalgebra 实验的联系

| 实验 | One-step | Global |
|------|----------|--------|
| Metric matching (DFA) | Corr = 0.80 | Corr = 0.28 |
| MasterCard acc | 94% | 10% |
| Jacobian eff rank | 16 | 1 |

**初步统一框架**：

```
Loss L 定义 behavioral metric d_L
    ↓
d_L 的"分辨率" = J 的行数 = 观测维度
    ↓
rank(J) = d_L 能区分的参数空间维度
    ↓
rank(J) 高 ⟺ 局部 metric matching 好 ⟺ 好学
```

核心：Loss function 决定"观测分辨率"
- One-step = 高分辨率 → rank 高 → 好学
- Final-only = 低分辨率 → rank 低 → 难学

**统一框架：学习 = (H, S, Θ, γ)**

```
H = Functor，定义 coalgebra 类型
S = Semiring，定义度量值域
Θ = 参数空间，索引 coalgebra 族 {C_θ}
γ = 优化算子，作用在 Θ 上
```

**数学结构**：

```
d_H^S: Coalg(H) × Coalg(H) → S     -- S-bisimilarity

固定目标 C_target，得到：

L: Θ → S
θ ↦ d_H^S(C_θ, C_target)

(H, S) 诱导 Θ 上的几何结构
γ 是这个几何上的下降算子
```

**核心命题**：

学得好坏 = γ 在 (H, S)-诱导几何上的收敛性质

**对应关系**：

| 数学概念 | 具体实例 |
|----------|----------|
| H | Mealy functor, DFA functor, etc. |
| S | R⁺, [0,∞], tropical semiring |
| Θ | 参数化的 coalgebra 族 |
| d_H^S | S-bisimilarity (Kantorovich lifting) |
| (H,S)-几何 | Θ 上的诱导度量/流形结构 |
| γ | 梯度下降、自然梯度等 |

**待探索**：
- (H, S) 如何决定 Θ 上的几何性质（曲率、条件数）？
- 给定 H，什么 S 使得几何最"好"（γ 收敛最快）？
- Fisher-Rao 是否是某种 (H, S) 的自然诱导？
- 能否从 (H, S) 推导 γ 收敛速度的理论界？

## 学习的数学定义（草稿）

### Behavior 的定义

```
设 H: C → C 是 functor
设 (Z, ω) 是 final H-coalgebra

对于 H-coalgebra (X, α: X → HX)
存在唯一态射 !: X → Z

behavior(x) = !(x) ∈ Z
```

例子：
- Mealy (H = O × (-)^I): Z ≅ causal functions I* → O*
- DFA (H = 2 × (-)^Σ): Z ≅ P(Σ*)

### 学习任务 vs 学习

**学习任务 (Learning Task)**：
```
T = (I, O, C_target)

I, O = 输入输出类型
C_target : Coalg(H) -- 存在但未知
```

**观测 D**：
```
D = behavior(C_target) 的部分信息

可能的形式化：
- D ∈ Z_n，其中 Z_n = Z 截断到深度 n
- D : I^≤n → O^≤n（有限 traces）
- D 是一个可查询的 oracle / 程序
```

**学习 (Learning)**：
```
L = (H, S, Θ, γ)

H = Functor（结构假设）
S = Semiring（度量方式）
Θ = 参数空间，索引 {C_θ}
γ = 优化算子

输入：D（观测）
输出：θ* = γ(D)
```

**关键区分**：
```
已知/选定：H, S, Θ, γ, D
未知：C_target, θ*

学习 = 从 D 推断 θ*
C_target 是存在量化的，不是输入
```

**学习表现（不是一维的，是元组）**：
```
Performance(L, T) = (ε_opt, ε_gen, ε_exp, ...)

ε_opt = 优化表现：γ 在 D 上的收敛性质
ε_gen = 泛化表现：D 上的解在 D 外的表现
ε_exp = 表达能力：inf_θ d_H^S(C_θ, C_target)
```

这些维度之间有关系/trade-off：
- 表达能力 ↑ → 优化难度 ↑（通常）
- 优化太好 → 泛化可能 ↓（过拟合）

"学得好" = 在这个元组空间里的某种最优（Pareto？）

### 待形式化

1. D 作为"部分 behavior"的精确定义
2. 从 D 到 θ* 的推断过程的范畴论描述
3. 泛化 = D 外的 behavior 预测准确度
4. C_min = argmin_{θ} d(C_θ, C_target) 的存在性和性质

## 下一步

1. 对更多实际架构（Transformer、GNN）测量 κ(J)
2. 研究如何从任务推导最优架构
3. 探索泛化性与 κ(J) 的关系
4. 深入研究 Lawvere metric space 与 optimization 的连接
5. 探索其他 functor（Markov chains, Kripke frames）的参数化
6. **理论**：为什么 effective rank 决定学习效果？与 NTK、Fisher information 的关系？
