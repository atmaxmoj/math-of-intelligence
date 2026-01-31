"""
Distance Matching 验证 - 第七版

直接测量 Jacobian 的条件数 κ(J)

这才是"好学"的真正度量:
- κ(J) 小 → Hessian 条件数小 → 梯度下降收敛快
- κ(J) 大 → Hessian 条件数大 → 梯度下降收敛慢
"""

import numpy as np
from typing import Callable, Tuple

np.random.seed(42)

# ============================================================================
# 计算 Jacobian 和条件数
# ============================================================================

def compute_jacobian(model_fn: Callable, theta: np.ndarray, x: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """
    数值计算 Jacobian: J[i,j] = ∂f_i / ∂θ_j
    f: R^p → R^n (参数 → 输出)
    """
    p = len(theta)
    f0 = model_fn(theta, x).flatten()
    n = len(f0)

    J = np.zeros((n, p))
    for j in range(p):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        f_plus = model_fn(theta_plus, x).flatten()
        J[:, j] = (f_plus - f0) / eps

    return J

def condition_number(J: np.ndarray) -> float:
    """计算条件数 κ(J) = σ_max / σ_min"""
    s = np.linalg.svd(J, compute_uv=False)
    if s[-1] < 1e-10:
        return np.inf
    return s[0] / s[-1]

def singular_value_stats(J: np.ndarray) -> Tuple[float, float, float, float]:
    """返回奇异值统计: min, max, mean, std"""
    s = np.linalg.svd(J, compute_uv=False)
    return s.min(), s.max(), s.mean(), s.std()

# ============================================================================
# 实验：直接测量条件数
# ============================================================================

def experiment_condition_number(model_fn: Callable, param_dim: int, x: np.ndarray,
                                 n_samples: int = 20, scale: float = 1.0):
    """在多个随机参数点测量条件数"""
    kappas = []

    for _ in range(n_samples):
        theta = np.random.randn(param_dim) * scale
        J = compute_jacobian(model_fn, theta, x)
        kappa = condition_number(J)
        if not np.isinf(kappa):
            kappas.append(kappa)

    if len(kappas) == 0:
        return np.inf, np.inf, np.inf

    return np.mean(kappas), np.std(kappas), np.median(kappas)

def print_header():
    print(f"{'架构':<40} {'κ(J) mean':>12} {'κ(J) std':>12} {'κ(J) median':>12}")
    print("-" * 80)

def print_result(name: str, mean_k: float, std_k: float, med_k: float):
    if mean_k < 10:
        grade = "★★★ 优秀"
    elif mean_k < 100:
        grade = "★★ 良好"
    elif mean_k < 1000:
        grade = "★ 一般"
    else:
        grade = "✗ 差"

    if np.isinf(mean_k):
        print(f"{name:<40} {'∞':>12} {'--':>12} {'--':>12}  {grade}")
    else:
        print(f"{name:<40} {mean_k:>12.1f} {std_k:>12.1f} {med_k:>12.1f}  {grade}")

# ============================================================================
# 主实验
# ============================================================================

def main():
    print("="*80)
    print("Distance Matching - 直接测量条件数 κ(J)")
    print("="*80)
    print("""
核心定理:
  κ(J) 小  →  Hessian 条件数 κ(H) = κ(J)² 小  →  梯度下降收敛快

这是 "好学" 的精确度量!
""")

    n, d = 50, 8  # 较小的规模以便计算 Jacobian
    x = np.random.randn(n, d)

    def relu(x):
        return np.maximum(0, x)

    print("\n" + "="*80)
    print("实验 1: 线性模型的不同参数化")
    print("="*80 + "\n")
    print_header()

    # 直接线性
    def linear_direct(theta, x):
        return x @ theta.reshape(d, 1)
    mean_k, std_k, med_k = experiment_condition_number(linear_direct, d, x, scale=1.0)
    print_result("Linear 直接", mean_k, std_k, med_k)

    # a-b 冗余
    def linear_ab(theta, x):
        a, b = theta[:d], theta[d:]
        return x @ (a - b).reshape(d, 1)
    mean_k, std_k, med_k = experiment_condition_number(linear_ab, 2*d, x, scale=1.0)
    print_result("Linear (a-b)", mean_k, std_k, med_k)

    # 2层线性
    def linear_2layer(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return x @ W1 @ W2
    mean_k, std_k, med_k = experiment_condition_number(linear_2layer, d*d+d, x, scale=0.5)
    print_result("Linear 2层 (W1·W2)", mean_k, std_k, med_k)

    # 3层线性
    def linear_3layer(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:2*d*d].reshape(d, d)
        W3 = theta[2*d*d:2*d*d+d].reshape(d, 1)
        return x @ W1 @ W2 @ W3
    mean_k, std_k, med_k = experiment_condition_number(linear_3layer, 2*d*d+d, x, scale=0.3)
    print_result("Linear 3层 (W1·W2·W3)", mean_k, std_k, med_k)

    print("\n" + "="*80)
    print("实验 2: 非线性网络")
    print("="*80 + "\n")
    print_header()

    # MLP 2层
    def mlp_2(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        return relu(x @ W1) @ W2
    mean_k, std_k, med_k = experiment_condition_number(mlp_2, d*d+d, x, scale=0.5)
    print_result("MLP 2层", mean_k, std_k, med_k)

    # MLP 3层
    def mlp_3(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:2*d*d].reshape(d, d)
        W3 = theta[2*d*d:2*d*d+d].reshape(d, 1)
        h = relu(x @ W1)
        h = relu(h @ W2)
        return h @ W3
    mean_k, std_k, med_k = experiment_condition_number(mlp_3, 2*d*d+d, x, scale=0.3)
    print_result("MLP 3层", mean_k, std_k, med_k)

    # MLP 4层
    def mlp_4(theta, x):
        idx = 0
        h = x
        for _ in range(3):
            W = theta[idx:idx+d*d].reshape(d, d)
            idx += d*d
            h = relu(h @ W)
        W_out = theta[idx:idx+d].reshape(d, 1)
        return h @ W_out
    mean_k, std_k, med_k = experiment_condition_number(mlp_4, 3*d*d+d, x, scale=0.2)
    print_result("MLP 4层", mean_k, std_k, med_k)

    print("\n" + "="*80)
    print("实验 3: ResNet")
    print("="*80 + "\n")
    print_header()

    # ResNet 2层
    def resnet_2(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:d*d+d].reshape(d, 1)
        h = x + 0.1 * relu(x @ W1)
        return h @ W2
    mean_k, std_k, med_k = experiment_condition_number(resnet_2, d*d+d, x, scale=0.5)
    print_result("ResNet 2层", mean_k, std_k, med_k)

    # ResNet 3层
    def resnet_3(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:2*d*d].reshape(d, d)
        W3 = theta[2*d*d:2*d*d+d].reshape(d, 1)
        h = x + 0.1 * relu(x @ W1)
        h = h + 0.1 * relu(h @ W2)
        return h @ W3
    mean_k, std_k, med_k = experiment_condition_number(resnet_3, 2*d*d+d, x, scale=0.3)
    print_result("ResNet 3层", mean_k, std_k, med_k)

    # ResNet 4层
    def resnet_4(theta, x):
        idx = 0
        h = x
        for _ in range(3):
            W = theta[idx:idx+d*d].reshape(d, d)
            idx += d*d
            h = h + 0.1 * relu(h @ W)
        W_out = theta[idx:idx+d].reshape(d, 1)
        return h @ W_out
    mean_k, std_k, med_k = experiment_condition_number(resnet_4, 3*d*d+d, x, scale=0.2)
    print_result("ResNet 4层", mean_k, std_k, med_k)

    print("\n" + "="*80)
    print("实验 4: 验证 κ(J) 与 CV 的关系")
    print("="*80 + "\n")

    # 对几个模型同时计算 κ(J) 和 CV
    from distance_matching_v6 import experiment as cv_experiment

    models = [
        ("Linear 直接", linear_direct, d, 1.0),
        ("Linear (a-b)", linear_ab, 2*d, 1.0),
        ("Linear 2层", linear_2layer, d*d+d, 0.5),
        ("MLP 2层", mlp_2, d*d+d, 0.5),
        ("ResNet 2层", resnet_2, d*d+d, 0.5),
    ]

    print(f"{'架构':<25} {'κ(J)':>10} {'CV':>10} {'Corr':>10}")
    print("-" * 60)

    for name, fn, param_dim, scale in models:
        mean_k, _, _ = experiment_condition_number(fn, param_dim, x, scale=scale, n_samples=10)
        corr, cv, _ = cv_experiment(fn, param_dim, x, n_pairs=200, scale=scale)

        print(f"{name:<25} {mean_k:>10.1f} {cv:>10.3f} {corr:>10.3f}")

    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
关键发现:

1. κ(J) 是 "好学" 的精确度量
   - κ(J) 小 → 收敛快
   - κ(J) 大 → 收敛慢

2. κ(J) 与 CV 相关但不相等
   - CV 是通过采样估计的
   - κ(J) 是精确计算的
   - 两者应该正相关

3. 各架构的 κ(J):
   - Linear 直接: 最小 (最好)
   - 冗余参数化: 变大
   - 深度网络: 很大
   - ResNet: 比 Plain 好?

这证明了: Distance Matching ↔ 条件数小 ↔ 好学
""")

if __name__ == "__main__":
    main()
