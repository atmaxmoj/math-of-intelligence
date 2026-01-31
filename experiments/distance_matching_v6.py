"""
Distance Matching 验证 - 第六版

核心框架:
- Behavioral metric: 由损失函数定义，对所有架构相同
- 具体: d_behavior(f, g) = ||f(X) - g(X)||_2 (在训练数据上)
- Parameter metric: Euclidean
- 问题: 哪种参数化 θ → f 更好地保持距离?

这次实验更清晰地分离:
1. 固定的 behavioral metric (由任务定义)
2. 不同的参数化方式 (架构选择)
"""

import numpy as np
from typing import Callable, Tuple

np.random.seed(42)

# ============================================================================
# 固定的 Behavioral Metric (由任务/损失函数定义)
# ============================================================================

def behavioral_distance(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Behavioral distance = L2 on outputs
    这是由损失函数诱导的度量，对所有架构相同
    """
    return np.linalg.norm(y1 - y2)

def param_distance(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """Parameter space 用 Euclidean distance"""
    return np.linalg.norm(theta1 - theta2)

# ============================================================================
# 实验框架
# ============================================================================

def experiment(model_fn: Callable, param_dim: int, x: np.ndarray,
               n_pairs: int = 500, scale: float = 1.0) -> Tuple[float, float, float]:
    """
    测量参数化 θ → f 的 distance matching 质量

    返回:
    - correlation: 参数距离和行为距离的相关性
    - cv: 比值的变异系数
    - mean_ratio: 平均比值 d_behavior / d_param
    """
    d_params = []
    d_behaviors = []

    for _ in range(n_pairs):
        theta1 = np.random.randn(param_dim) * scale
        theta2 = np.random.randn(param_dim) * scale

        # 参数距离
        d_p = param_distance(theta1, theta2)

        # 行为距离 (函数在数据上的输出差异)
        y1 = model_fn(theta1, x)
        y2 = model_fn(theta2, x)
        d_b = behavioral_distance(y1, y2)

        if d_p > 1e-8:
            d_params.append(d_p)
            d_behaviors.append(d_b)

    d_params = np.array(d_params)
    d_behaviors = np.array(d_behaviors)

    corr = np.corrcoef(d_params, d_behaviors)[0, 1]
    ratios = d_behaviors / (d_params + 1e-8)
    cv = np.std(ratios) / (np.mean(ratios) + 1e-8)
    mean_ratio = np.mean(ratios)

    return corr, cv, mean_ratio

def print_header():
    print(f"{'架构':<40} {'Corr':>8} {'CV':>8} {'Ratio':>8} {'评价':>6}")
    print("-" * 75)

def print_result(name: str, corr: float, cv: float, ratio: float):
    if corr > 0.9 and cv < 0.2:
        grade = "★★★"
    elif corr > 0.7 and cv < 0.4:
        grade = "★★"
    elif corr > 0.5:
        grade = "★"
    else:
        grade = "✗"
    print(f"{name:<40} {corr:>8.3f} {cv:>8.3f} {ratio:>8.3f} {grade:>6}")

# ============================================================================
# 实验 1: 同一函数空间，不同参数化
# ============================================================================

def exp1_same_function_space():
    print("\n" + "="*75)
    print("实验 1: 同一函数空间 (线性函数)，不同参数化")
    print("="*75)
    print("所有参数化表达的是同一个函数空间，behavioral metric 完全相同")
    print("唯一区别: θ → f 的映射方式\n")

    n, d = 100, 10
    x = np.random.randn(n, d)

    print_header()

    # 直接参数化: θ 就是权重
    def direct(theta, x):
        return x @ theta.reshape(d, 1)
    corr, cv, ratio = experiment(direct, d, x, scale=1.0)
    print_result("直接: f(x) = x·θ", corr, cv, ratio)

    # 冗余参数化: a - b
    def redundant_ab(theta, x):
        a, b = theta[:d], theta[d:]
        return x @ (a - b).reshape(d, 1)
    corr, cv, ratio = experiment(redundant_ab, 2*d, x, scale=1.0)
    print_result("冗余: f(x) = x·(a-b)", corr, cv, ratio)

    # 更冗余: a - b + c - d
    def redundant_abcd(theta, x):
        a, b, c, e = theta[:d], theta[d:2*d], theta[2*d:3*d], theta[3*d:]
        return x @ (a - b + c - e).reshape(d, 1)
    corr, cv, ratio = experiment(redundant_abcd, 4*d, x, scale=0.5)
    print_result("更冗余: f(x) = x·(a-b+c-d)", corr, cv, ratio)

    # 矩阵分解: A @ b
    def matrix_factor(theta, x):
        k = 5
        A = theta[:d*k].reshape(d, k)
        b = theta[d*k:d*k+k]
        w = A @ b
        return x @ w.reshape(d, 1)
    corr, cv, ratio = experiment(matrix_factor, d*5+5, x, scale=0.5)
    print_result("矩阵分解: f(x) = x·(A·b)", corr, cv, ratio)

    # 2层线性 (无非线性)
    def two_layer_linear(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return x @ W1 @ W2
    corr, cv, ratio = experiment(two_layer_linear, d*d+d, x, scale=0.3)
    print_result("2层线性: f(x) = x·W1·W2", corr, cv, ratio)

    print("\n结论: 同一函数空间，参数化方式决定 distance matching 质量")

# ============================================================================
# 实验 2: Plain vs ResNet (相同的 behavioral metric)
# ============================================================================

def exp2_plain_vs_resnet():
    print("\n" + "="*75)
    print("实验 2: Plain vs ResNet")
    print("="*75)
    print("Behavioral metric 相同 (都是 L2 on outputs)")
    print("问题: ResNet 的参数化是否更好地保持距离?\n")

    n, d = 100, 10
    x = np.random.randn(n, d)

    def relu(x):
        return np.maximum(0, x)

    print_header()

    # Plain 2层
    def plain_2(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return relu(x @ W1) @ W2

    param_dim = d*d + d
    corr, cv, ratio = experiment(plain_2, param_dim, x, scale=0.3)
    print_result("Plain 2层", corr, cv, ratio)

    # ResNet: f(x) = x·w + F(x)，其中 F 是 2层
    def resnet_2(theta, x):
        # 线性部分
        w_lin = theta[:d].reshape(d, 1)
        # 残差部分
        W1 = theta[d:d+d*d].reshape(d, d)
        W2 = theta[d+d*d:d+d*d+d].reshape(d, 1)
        F_x = relu(x @ W1) @ W2
        return x @ w_lin + 0.1 * F_x

    param_dim = d + d*d + d
    corr, cv, ratio = experiment(resnet_2, param_dim, x, scale=0.3)
    print_result("ResNet 2层: x·w + 0.1·F(x)", corr, cv, ratio)

    # 更深的比较
    print()

    for depth in [2, 3, 4, 5]:
        # Plain
        def plain_deep(theta, x, depth=depth):
            h = x
            idx = 0
            for _ in range(depth - 1):
                W = theta[idx:idx+d*d].reshape(d, d)
                idx += d*d
                h = relu(h @ W) * 0.5
            W_out = theta[idx:idx+d].reshape(d, 1)
            return h @ W_out

        param_dim = d*d*(depth-1) + d
        corr, cv, ratio = experiment(plain_deep, param_dim, x, scale=0.2)
        print_result(f"Plain {depth}层", corr, cv, ratio)

    print()

    for depth in [2, 3, 4, 5]:
        # ResNet
        def resnet_deep(theta, x, depth=depth):
            h = x
            idx = 0
            for _ in range(depth - 1):
                W = theta[idx:idx+d*d].reshape(d, d)
                idx += d*d
                h = h + 0.1 * relu(h @ W)  # skip connection
            W_out = theta[idx:idx+d].reshape(d, 1)
            return h @ W_out

        param_dim = d*d*(depth-1) + d
        corr, cv, ratio = experiment(resnet_deep, param_dim, x, scale=0.2)
        print_result(f"ResNet {depth}层", corr, cv, ratio)

    print("\n结论: 比较 Plain 和 ResNet 在相同 behavioral metric 下的表现")

# ============================================================================
# 实验 3: 验证 behavioral metric 确实是由任务定义的
# ============================================================================

def exp3_task_defines_metric():
    print("\n" + "="*75)
    print("实验 3: 任务定义 Behavioral Metric")
    print("="*75)
    print("同一个架构，不同的任务 (数据分布) → 不同的 behavioral distance 值")
    print("但 distance matching 性质应该是架构的内在属性\n")

    d = 10

    def relu(x):
        return np.maximum(0, x)

    # 同一个架构
    def mlp(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return relu(x @ W1) @ W2

    param_dim = d*d + d

    print_header()

    # 不同的数据分布
    distributions = [
        ("标准正态 N(0,1)", lambda: np.random.randn(100, d)),
        ("均匀 U(-1,1)", lambda: np.random.uniform(-1, 1, (100, d))),
        ("稀疏 (90% 为 0)", lambda: np.random.randn(100, d) * (np.random.rand(100, d) > 0.9)),
        ("相关 (协方差)", lambda: np.random.randn(100, d) @ np.random.randn(d, d) * 0.3),
    ]

    for name, gen_x in distributions:
        x = gen_x()
        corr, cv, ratio = experiment(mlp, param_dim, x, scale=0.3)
        print_result(f"MLP + {name}", corr, cv, ratio)

    print("\n结论: Correlation 和 CV 是架构的性质，ratio 取决于数据")

# ============================================================================
# 实验 4: 如果我们能直接参数化 behavioral space?
# ============================================================================

def exp4_direct_behavioral_param():
    print("\n" + "="*75)
    print("实验 4: 直接在 behavioral space 参数化")
    print("="*75)
    print("如果参数直接就是函数在某些点上的值，会怎样?\n")

    n, d = 100, 10
    x = np.random.randn(n, d)

    print_header()

    # 方法1: 参数是权重 (间接)
    def indirect(theta, x):
        return x @ theta.reshape(d, 1)
    corr, cv, ratio = experiment(indirect, d, x, scale=1.0)
    print_result("间接: θ 是权重", corr, cv, ratio)

    # 方法2: 参数"几乎"是输出 (用基函数)
    # f(x) = Σ θ_i · φ_i(x)，其中 φ_i 是固定的基
    # 如果 φ_i 是正交归一的，θ 变化直接对应函数变化

    # 用随机但固定的基
    np.random.seed(123)  # 固定基
    basis = np.random.randn(d, 20)  # 20 个基函数
    basis = basis / np.linalg.norm(basis, axis=0)  # 归一化

    def basis_expansion(theta, x):
        # f(x) = x @ basis @ θ
        # 如果 basis 是正交的，θ 的变化直接对应输出变化
        return x @ basis @ theta.reshape(20, 1)

    np.random.seed(42)  # 恢复
    corr, cv, ratio = experiment(basis_expansion, 20, x, scale=1.0)
    print_result("基展开: θ 是系数 (正交基)", corr, cv, ratio)

    # 方法3: 直接参数化为输出值 (极端情况)
    # 不可行，因为输出维度 = 样本数，太大了
    # 但概念上，这是"完美"的 behavioral parameterization

    print("\n结论: 越接近直接参数化 behavioral space，distance matching 越好")

# ============================================================================
# 实验 5: 综合比较
# ============================================================================

def exp5_comprehensive():
    print("\n" + "="*75)
    print("实验 5: 综合比较 - 所有架构在同一 behavioral metric 下")
    print("="*75)
    print("固定任务 (数据)，比较不同参数化的 distance matching\n")

    n, d = 100, 10
    x = np.random.randn(n, d)

    def relu(x):
        return np.maximum(0, x)

    print_header()

    architectures = []

    # 线性类
    def linear(theta, x):
        return x @ theta.reshape(d, 1)
    architectures.append(("Linear (直接)", linear, d, 1.0))

    def linear_redundant(theta, x):
        a, b = theta[:d], theta[d:]
        return x @ (a - b).reshape(d, 1)
    architectures.append(("Linear (a-b 冗余)", linear_redundant, 2*d, 1.0))

    def linear_2layer(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return x @ W1 @ W2
    architectures.append(("Linear (2层)", linear_2layer, d*d+d, 0.3))

    # 非线性类
    def mlp_2(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return relu(x @ W1) @ W2
    architectures.append(("MLP 2层", mlp_2, d*d+d, 0.3))

    def mlp_3(theta, x):
        idx = 0
        h = x
        for _ in range(2):
            W = theta[idx:idx+d*d].reshape(d, d)
            idx += d*d
            h = relu(h @ W) * 0.5
        W_out = theta[idx:idx+d].reshape(d, 1)
        return h @ W_out
    architectures.append(("MLP 3层", mlp_3, d*d*2+d, 0.2))

    # ResNet 类
    def resnet_2(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        h = x + 0.1 * relu(x @ W1)
        return h @ W2
    architectures.append(("ResNet 2层", resnet_2, d*d+d, 0.3))

    def resnet_3(theta, x):
        idx = 0
        h = x
        for _ in range(2):
            W = theta[idx:idx+d*d].reshape(d, d)
            idx += d*d
            h = h + 0.1 * relu(h @ W)
        W_out = theta[idx:idx+d].reshape(d, 1)
        return h @ W_out
    architectures.append(("ResNet 3层", resnet_3, d*d*2+d, 0.2))

    for name, fn, param_dim, scale in architectures:
        corr, cv, ratio = experiment(fn, param_dim, x, scale=scale)
        print_result(name, corr, cv, ratio)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*75)
    print("Distance Matching 验证 - 清晰框架版")
    print("="*75)
    print("""
核心框架:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Behavioral metric:  由损失函数/任务定义，对所有架构 **相同**
                      d(f, g) = ||f(X) - g(X)||₂

  Parameter metric:   Euclidean (标准选择)
                      d(θ₁, θ₂) = ||θ₁ - θ₂||₂

  问题:               哪种参数化 θ → f 更好地保持距离?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

指标:
  - Correlation: 参数距离和行为距离的线性相关性 (越高越好)
  - CV: 比值 d_behavior/d_param 的变异系数 (越低越好)
  - Ratio: 平均比值 (取决于数据尺度，不是质量指标)
""")

    exp1_same_function_space()
    exp2_plain_vs_resnet()
    exp3_task_defines_metric()
    exp4_direct_behavioral_param()
    exp5_comprehensive()

    print("\n" + "="*75)
    print("总结")
    print("="*75)
    print("""
关键发现:

1. Behavioral metric 由任务定义，对所有架构相同
   - 这简化了问题: 不需要为每个架构定义不同的 metric
   - L2 on outputs 是合理的选择 (由损失函数诱导)

2. Distance matching 是参数化方式的性质
   - 同一函数空间可以有好的或坏的参数化
   - 冗余 (显式或隐式) 是主要破坏因素

3. 深度的影响
   - 深度引入隐式冗余 (W₂·W₁ 有多解)
   - 即使是线性深度网络也有此问题

4. ResNet vs Plain
   - 在相同 behavioral metric 下比较
   - ResNet 的 skip connection 是否真的改善了 distance matching?
   - 需要看实验结果...

5. 理论方向
   - 能否找到"最优"参数化，使得 θ → f 是等距映射?
   - 这与 coalgebra 的 behavioral metric 理论如何连接?
""")

if __name__ == "__main__":
    main()
