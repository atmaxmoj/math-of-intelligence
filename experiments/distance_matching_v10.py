"""
Distance Matching 验证 - 第十版

经典的"任务-架构匹配"例子:
1. 线性函数 → Linear
2. 平移不变 → CNN vs FC
3. 置换不变 (集合函数) → DeepSets vs MLP
4. 计数任务 → RNN vs FC
5. 对称函数 → 对称网络 vs 一般网络
6. 低秩函数 → 低秩网络 vs 全秩
7. 稀疏函数 → 稀疏网络 vs 稠密
8. 加法 → 正确结构 vs 错误结构
"""

import numpy as np
from typing import Callable, List

np.random.seed(42)

# ============================================================================
# 工具函数
# ============================================================================

def compute_loss(model_fn, theta, x, y):
    pred = model_fn(theta, x)
    return np.mean((pred - y) ** 2)

def compute_gradient(model_fn, theta, x, y, eps=1e-5):
    grad = np.zeros_like(theta)
    loss0 = compute_loss(model_fn, theta, x, y)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (compute_loss(model_fn, theta_plus, x, y) - loss0) / eps
    return grad

def train_multiple_seeds(model_fn, param_dim, x, y, n_seeds=20, n_epochs=50,
                         lr=0.01, init_scale=0.5):
    final_losses = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        theta = np.random.randn(param_dim) * init_scale
        for _ in range(n_epochs):
            grad = compute_gradient(model_fn, theta, x, y)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10:
                grad = grad * 10 / grad_norm
            theta = theta - lr * grad
        final_losses.append(compute_loss(model_fn, theta, x, y))
    return np.array(final_losses)

def compute_kappa(model_fn, param_dim, x, init_scale=0.5, n_samples=5):
    kappas = []
    for seed in range(n_samples):
        np.random.seed(seed + 1000)
        theta = np.random.randn(param_dim) * init_scale
        f0 = model_fn(theta, x).flatten()
        J = np.zeros((len(f0), param_dim))
        eps = 1e-5
        for j in range(param_dim):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            J[:, j] = (model_fn(theta_plus, x).flatten() - f0) / eps
        s = np.linalg.svd(J, compute_uv=False)
        if s[-1] > 1e-10:
            kappas.append(s[0] / s[-1])
    return np.mean(kappas) if kappas else np.inf

def print_result(task, arch, kappa, losses):
    mean_l = losses.mean()
    std_l = losses.std()
    cv = std_l / mean_l if mean_l > 1e-10 else 0
    kappa_str = f"{kappa:.1f}" if kappa < 1e6 else "∞"

    if cv < 0.1 and mean_l < 0.1:
        grade = "★★★"
    elif cv < 0.3 and mean_l < 0.5:
        grade = "★★"
    elif mean_l < 1.0:
        grade = "★"
    else:
        grade = "✗"

    print(f"  {arch:<30} κ={kappa_str:<12} loss={mean_l:.4f}±{std_l:.4f}  CV={cv:.2f}  {grade}")

# ============================================================================
# 任务 1: 线性函数
# ============================================================================

def task1_linear():
    print("\n" + "="*80)
    print("任务 1: 线性函数  y = Wx")
    print("="*80)
    print("正确架构: Linear,  错误架构: 冗余/深度")

    n, d = 50, 6
    np.random.seed(0)
    x = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = (x @ w_true).reshape(-1, 1)

    # Linear 直接
    def linear(theta, x):
        return x @ theta.reshape(d, 1)
    kappa = compute_kappa(linear, d, x)
    losses = train_multiple_seeds(linear, d, x, y, lr=0.1)
    print_result("线性函数", "Linear (直接)", kappa, losses)

    # Linear 冗余
    def linear_ab(theta, x):
        return x @ (theta[:d] - theta[d:]).reshape(d, 1)
    kappa = compute_kappa(linear_ab, 2*d, x)
    losses = train_multiple_seeds(linear_ab, 2*d, x, y, lr=0.05)
    print_result("线性函数", "Linear (a-b 冗余)", kappa, losses)

    # 2层线性
    def linear_2layer(theta, x):
        W1 = theta[:d*d].reshape(d, d)
        W2 = theta[d*d:].reshape(d, 1)
        return x @ W1 @ W2
    kappa = compute_kappa(linear_2layer, d*d+d, x, init_scale=0.3)
    losses = train_multiple_seeds(linear_2layer, d*d+d, x, y, lr=0.01, init_scale=0.3)
    print_result("线性函数", "Linear (2层)", kappa, losses)

    # MLP
    def mlp(theta, x):
        W1 = theta[:d*8].reshape(d, 8)
        W2 = theta[d*8:].reshape(8, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, d*8+8, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, d*8+8, x, y, lr=0.01, init_scale=0.3)
    print_result("线性函数", "MLP (过度复杂)", kappa, losses)

# ============================================================================
# 任务 2: 平移不变函数 (1D 卷积)
# ============================================================================

def task2_translation_invariant():
    print("\n" + "="*80)
    print("任务 2: 平移不变函数  y = conv(x, kernel)")
    print("="*80)
    print("正确架构: CNN,  错误架构: FC")

    n, seq_len, k = 50, 16, 3
    np.random.seed(0)
    x = np.random.randn(n, seq_len)
    kernel_true = np.array([1, -2, 1])  # 二阶差分

    # 生成目标: 卷积结果的和
    y = np.zeros((n, 1))
    for i in range(seq_len - k + 1):
        y[:, 0] += np.sum(x[:, i:i+k] * kernel_true, axis=1)

    # CNN (正确)
    def cnn(theta, x):
        kernel = theta[:k]
        out = np.zeros((n, 1))
        for i in range(seq_len - k + 1):
            out[:, 0] += np.sum(x[:, i:i+k] * kernel, axis=1)
        return out
    kappa = compute_kappa(cnn, k, x)
    losses = train_multiple_seeds(cnn, k, x, y, lr=0.01)
    print_result("平移不变", "CNN (kernel)", kappa, losses)

    # FC 全连接 (错误 - 过参数)
    def fc(theta, x):
        return x @ theta.reshape(seq_len, 1)
    kappa = compute_kappa(fc, seq_len, x)
    losses = train_multiple_seeds(fc, seq_len, x, y, lr=0.01)
    print_result("平移不变", "FC (全连接)", kappa, losses)

    # Toeplitz (等价 CNN)
    def toeplitz(theta, x):
        kernel = theta[:k]
        W = np.zeros((seq_len, 1))
        for i in range(seq_len - k + 1):
            W[i:i+k, 0] += kernel
        return x @ W
    kappa = compute_kappa(toeplitz, k, x)
    losses = train_multiple_seeds(toeplitz, k, x, y, lr=0.01)
    print_result("平移不变", "Toeplitz (结构化)", kappa, losses)

# ============================================================================
# 任务 3: 置换不变函数 (集合函数)
# ============================================================================

def task3_permutation_invariant():
    print("\n" + "="*80)
    print("任务 3: 置换不变函数  y = sum(x)")
    print("="*80)
    print("正确架构: DeepSets (sum pooling),  错误架构: FC")

    n, d = 50, 8
    np.random.seed(0)
    x = np.random.randn(n, d)
    y = x.sum(axis=1, keepdims=True)  # 目标: 求和

    # DeepSets: y = sum(phi(x_i))
    # 最简单情况: phi = identity, 所以 y = sum(x)
    def deepsets(theta, x):
        # theta 是每个元素的权重，但因为置换不变，应该都相等
        # 用单一权重
        return x.sum(axis=1, keepdims=True) * theta[0]
    kappa = compute_kappa(deepsets, 1, x)
    losses = train_multiple_seeds(deepsets, 1, x, y, lr=0.1)
    print_result("置换不变", "DeepSets (sum)", kappa, losses)

    # 对称权重: 所有位置用同一个权重
    def symmetric(theta, x):
        return (x * theta[0]).sum(axis=1, keepdims=True)
    kappa = compute_kappa(symmetric, 1, x)
    losses = train_multiple_seeds(symmetric, 1, x, y, lr=0.1)
    print_result("置换不变", "对称权重 (1 param)", kappa, losses)

    # FC: 每个位置不同权重 (错误 - 没有利用对称性)
    def fc(theta, x):
        return x @ theta.reshape(d, 1)
    kappa = compute_kappa(fc, d, x)
    losses = train_multiple_seeds(fc, d, x, y, lr=0.05)
    print_result("置换不变", "FC (d params)", kappa, losses)

    # MLP (更错误)
    def mlp(theta, x):
        W1 = theta[:d*8].reshape(d, 8)
        W2 = theta[d*8:].reshape(8, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, d*8+8, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, d*8+8, x, y, lr=0.01, init_scale=0.3)
    print_result("置换不变", "MLP (过度复杂)", kappa, losses)

# ============================================================================
# 任务 4: 计数任务
# ============================================================================

def task4_counting():
    print("\n" + "="*80)
    print("任务 4: 计数任务  y = count(x > 0)")
    print("="*80)
    print("正确架构: 累加结构,  错误架构: FC")

    n, d = 50, 10
    np.random.seed(0)
    x = np.random.randn(n, d)
    y = (x > 0).sum(axis=1, keepdims=True).astype(float)  # 数正数个数

    # 正确结构: sum(sigmoid(x * w))
    def counter(theta, x):
        w = theta[0]
        return (1 / (1 + np.exp(-x * w * 10))).sum(axis=1, keepdims=True)
    kappa = compute_kappa(counter, 1, x)
    losses = train_multiple_seeds(counter, 1, x, y, lr=0.1)
    print_result("计数", "Counter (sum sigmoid)", kappa, losses)

    # FC
    def fc(theta, x):
        return x @ theta.reshape(d, 1)
    kappa = compute_kappa(fc, d, x)
    losses = train_multiple_seeds(fc, d, x, y, lr=0.01)
    print_result("计数", "FC", kappa, losses)

    # MLP
    def mlp(theta, x):
        W1 = theta[:d*8].reshape(d, 8)
        W2 = theta[d*8:].reshape(8, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, d*8+8, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, d*8+8, x, y, lr=0.01, init_scale=0.3)
    print_result("计数", "MLP", kappa, losses)

# ============================================================================
# 任务 5: 二次函数 y = x^T A x
# ============================================================================

def task5_quadratic():
    print("\n" + "="*80)
    print("任务 5: 二次函数  y = x^T A x")
    print("="*80)
    print("正确架构: 二次层,  错误架构: 线性/MLP")

    n, d = 50, 5
    np.random.seed(0)
    x = np.random.randn(n, d)
    A_true = np.random.randn(d, d)
    A_true = (A_true + A_true.T) / 2  # 对称
    y = np.array([xi @ A_true @ xi for xi in x]).reshape(-1, 1)

    # 二次层 (正确)
    def quadratic(theta, x):
        A = theta.reshape(d, d)
        return np.array([xi @ A @ xi for xi in x]).reshape(-1, 1)
    kappa = compute_kappa(quadratic, d*d, x, init_scale=0.3)
    losses = train_multiple_seeds(quadratic, d*d, x, y, lr=0.001, init_scale=0.3)
    print_result("二次函数", "Quadratic (x^T A x)", kappa, losses)

    # 对称二次 (更少参数)
    def quadratic_sym(theta, x):
        # 只用上三角，共 d(d+1)/2 个参数
        A = np.zeros((d, d))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                A[i, j] = theta[idx]
                A[j, i] = theta[idx]
                idx += 1
        return np.array([xi @ A @ xi for xi in x]).reshape(-1, 1)
    n_params = d * (d + 1) // 2
    kappa = compute_kappa(quadratic_sym, n_params, x, init_scale=0.3)
    losses = train_multiple_seeds(quadratic_sym, n_params, x, y, lr=0.001, init_scale=0.3)
    print_result("二次函数", "Quadratic 对称 (更少参数)", kappa, losses)

    # Linear (错误)
    def linear(theta, x):
        return x @ theta.reshape(d, 1)
    kappa = compute_kappa(linear, d, x)
    losses = train_multiple_seeds(linear, d, x, y, lr=0.01)
    print_result("二次函数", "Linear (表达力不够)", kappa, losses)

    # MLP
    def mlp(theta, x):
        W1 = theta[:d*16].reshape(d, 16)
        W2 = theta[d*16:].reshape(16, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, d*16+16, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, d*16+16, x, y, lr=0.005, init_scale=0.3)
    print_result("二次函数", "MLP", kappa, losses)

# ============================================================================
# 任务 6: 稀疏函数 (只依赖少数输入)
# ============================================================================

def task6_sparse():
    print("\n" + "="*80)
    print("任务 6: 稀疏函数  y = x[0] + x[1] (只用前2个)")
    print("="*80)
    print("正确架构: 稀疏,  错误架构: 稠密")

    n, d = 50, 20
    np.random.seed(0)
    x = np.random.randn(n, d)
    y = (x[:, 0] + x[:, 1]).reshape(-1, 1)  # 只依赖前两个

    # 稀疏 (正确)
    def sparse(theta, x):
        return (x[:, 0] * theta[0] + x[:, 1] * theta[1]).reshape(-1, 1)
    kappa = compute_kappa(sparse, 2, x)
    losses = train_multiple_seeds(sparse, 2, x, y, lr=0.1)
    print_result("稀疏函数", "Sparse (2 params)", kappa, losses)

    # 稠密 Linear
    def dense(theta, x):
        return x @ theta.reshape(d, 1)
    kappa = compute_kappa(dense, d, x)
    losses = train_multiple_seeds(dense, d, x, y, lr=0.05)
    print_result("稀疏函数", "Dense Linear (20 params)", kappa, losses)

    # MLP
    def mlp(theta, x):
        W1 = theta[:d*8].reshape(d, 8)
        W2 = theta[d*8:].reshape(8, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, d*8+8, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, d*8+8, x, y, lr=0.01, init_scale=0.3)
    print_result("稀疏函数", "MLP", kappa, losses)

# ============================================================================
# 任务 7: 乘法 y = x1 * x2
# ============================================================================

def task7_multiplication():
    print("\n" + "="*80)
    print("任务 7: 乘法  y = x[0] * x[1]")
    print("="*80)
    print("正确架构: 乘法门,  错误架构: 加法网络")

    n = 50
    np.random.seed(0)
    x = np.random.randn(n, 2)
    y = (x[:, 0] * x[:, 1]).reshape(-1, 1)

    # 乘法门 (正确)
    def mult_gate(theta, x):
        return (x[:, 0] * x[:, 1] * theta[0]).reshape(-1, 1)
    kappa = compute_kappa(mult_gate, 1, x)
    losses = train_multiple_seeds(mult_gate, 1, x, y, lr=0.1)
    print_result("乘法", "乘法门 (exact)", kappa, losses)

    # Linear (错误 - 无法表示乘法)
    def linear(theta, x):
        return x @ theta.reshape(2, 1)
    kappa = compute_kappa(linear, 2, x)
    losses = train_multiple_seeds(linear, 2, x, y, lr=0.05)
    print_result("乘法", "Linear (无法表示)", kappa, losses)

    # MLP (可以近似)
    def mlp(theta, x):
        W1 = theta[:2*16].reshape(2, 16)
        W2 = theta[2*16:].reshape(16, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, 2*16+16, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, 2*16+16, x, y, lr=0.01, init_scale=0.3)
    print_result("乘法", "MLP (可近似)", kappa, losses)

# ============================================================================
# 任务 8: 最大值 y = max(x)
# ============================================================================

def task8_max():
    print("\n" + "="*80)
    print("任务 8: 最大值  y = max(x)")
    print("="*80)
    print("正确架构: MaxPool,  错误架构: Linear")

    n, d = 50, 5
    np.random.seed(0)
    x = np.random.randn(n, d)
    y = x.max(axis=1, keepdims=True)

    # Soft max (可微近似)
    def soft_max(theta, x):
        temp = theta[0] if abs(theta[0]) > 0.1 else 0.1
        weights = np.exp(x * temp)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return (x * weights).sum(axis=1, keepdims=True)
    kappa = compute_kappa(soft_max, 1, x)
    losses = train_multiple_seeds(soft_max, 1, x, y, lr=0.5)
    print_result("最大值", "SoftMax (温度参数)", kappa, losses)

    # Linear (错误)
    def linear(theta, x):
        return x @ theta.reshape(d, 1)
    kappa = compute_kappa(linear, d, x)
    losses = train_multiple_seeds(linear, d, x, y, lr=0.05)
    print_result("最大值", "Linear", kappa, losses)

    # MLP
    def mlp(theta, x):
        W1 = theta[:d*16].reshape(d, 16)
        W2 = theta[d*16:].reshape(16, 1)
        return np.maximum(0, x @ W1) @ W2
    kappa = compute_kappa(mlp, d*16+16, x, init_scale=0.3)
    losses = train_multiple_seeds(mlp, d*16+16, x, y, lr=0.01, init_scale=0.3)
    print_result("最大值", "MLP", kappa, losses)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*80)
    print("Distance Matching - 经典任务-架构匹配")
    print("="*80)
    print("""
验证: 当架构匹配任务结构时，κ(J) 更小，学习更稳定

每个任务测试:
- 正确架构 (利用任务结构)
- 错误架构 (通用但不匹配)

指标:
- κ(J): 条件数 (越小越好)
- loss: 最终损失 (mean ± std over 20 seeds)
- CV: 标准差/均值 (越小越稳定)
""")

    task1_linear()
    task2_translation_invariant()
    task3_permutation_invariant()
    task4_counting()
    task5_quadratic()
    task6_sparse()
    task7_multiplication()
    task8_max()

    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
核心发现:

1. 架构匹配任务 → κ(J) 小 → 学得快且稳定
   - 线性任务 + Linear: κ ≈ 1-2
   - 平移不变 + CNN: κ 小
   - 置换不变 + DeepSets: κ = 1

2. 架构不匹配 → κ(J) 大 → 学得慢或不稳定
   - 线性任务 + MLP: κ 大
   - 置换不变 + FC: κ 大 (有冗余)

3. 表达力不够 vs 冗余
   - Linear 学乘法: loss 高 (表达力不够)
   - MLP 学线性: CV 高 (冗余导致不稳定)

4. κ(J) 同时反映:
   - 表达力匹配 (能否表示目标)
   - 参数效率 (是否有冗余)

结论: "正确的架构" = 架构结构匹配任务结构 = κ(J) 小 = 好学
""")

if __name__ == "__main__":
    main()
