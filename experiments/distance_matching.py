"""
验证 Distance Matching 假说

对比几种架构：
1. 线性回归 (应该完美匹配)
2. MLP (可能不太匹配)
3. CNN (应该较好匹配)
4. 带 Skip 的网络 (应该较好匹配)

方法：采样参数对，计算参数距离和函数距离，看是否成比例
"""

import numpy as np
from typing import Callable, Tuple

np.random.seed(42)

# ============================================================================
# 几种架构
# ============================================================================

def linear_model(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """线性模型: f(x) = theta @ x"""
    return x @ theta

def mlp_model(theta: np.ndarray, x: np.ndarray, hidden_dim: int = 32) -> np.ndarray:
    """两层 MLP: f(x) = W2 @ relu(W1 @ x)"""
    input_dim = x.shape[1]
    output_dim = 1

    # 解包参数
    w1_size = input_dim * hidden_dim
    w2_size = hidden_dim * output_dim

    W1 = theta[:w1_size].reshape(input_dim, hidden_dim)
    W2 = theta[w1_size:w1_size + w2_size].reshape(hidden_dim, output_dim)

    # 前向传播
    h = np.maximum(0, x @ W1)  # ReLU
    return h @ W2

def cnn_1d_model(theta: np.ndarray, x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """1D 卷积: 用 kernel 卷积输入"""
    # theta 就是 kernel
    kernel = theta[:kernel_size]

    # 简单卷积 (valid mode)
    n_samples, seq_len = x.shape
    out_len = seq_len - kernel_size + 1
    output = np.zeros((n_samples, out_len))

    for i in range(out_len):
        output[:, i] = np.sum(x[:, i:i+kernel_size] * kernel, axis=1)

    return output.mean(axis=1, keepdims=True)  # 全局池化

def resnet_block(theta: np.ndarray, x: np.ndarray, hidden_dim: int = 32) -> np.ndarray:
    """残差块: f(x) = x + F(x)，其中 F 是小 MLP"""
    input_dim = x.shape[1]

    # 解包参数 (两层小网络)
    w1_size = input_dim * hidden_dim
    w2_size = hidden_dim * input_dim

    W1 = theta[:w1_size].reshape(input_dim, hidden_dim)
    W2 = theta[w1_size:w1_size + w2_size].reshape(hidden_dim, input_dim)

    # 残差: x + F(x)
    h = np.maximum(0, x @ W1)
    F_x = h @ W2

    return x + 0.1 * F_x  # 缩放残差，保持在恒等附近

# ============================================================================
# 全连接学习"对称函数"（平均值）—— 模拟冗余情况
# ============================================================================

def fc_symmetric(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    全连接网络学习一个对称函数（均值）
    目标: f(x) = mean(x)，但用全连接参数化
    这会有大量冗余
    """
    input_dim = x.shape[1]
    # theta 是 input_dim 个权重，理想情况都应该是 1/input_dim
    # 但很多不同的 theta 可以给出接近均值的结果
    return (x @ theta).reshape(-1, 1)

# ============================================================================
# 距离计算
# ============================================================================

def param_distance(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """参数空间的欧氏距离"""
    return np.linalg.norm(theta1 - theta2)

def func_distance(f: Callable, theta1: np.ndarray, theta2: np.ndarray,
                  x: np.ndarray) -> float:
    """函数空间的距离: ||f(theta1, x) - f(theta2, x)||"""
    y1 = f(theta1, x)
    y2 = f(theta2, x)
    return np.linalg.norm(y1 - y2)

# ============================================================================
# 实验
# ============================================================================

def experiment(model_fn: Callable, param_dim: int, x: np.ndarray,
               n_pairs: int = 500, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    采样参数对，计算距离
    返回: (参数距离数组, 函数距离数组)
    """
    d_params = []
    d_funcs = []

    for _ in range(n_pairs):
        # 采样两个参数
        theta1 = np.random.randn(param_dim) * scale
        theta2 = np.random.randn(param_dim) * scale

        # 计算距离
        d_p = param_distance(theta1, theta2)
        d_f = func_distance(model_fn, theta1, theta2, x)

        d_params.append(d_p)
        d_funcs.append(d_f)

    return np.array(d_params), np.array(d_funcs)

def compute_correlation(d_params: np.ndarray, d_funcs: np.ndarray) -> float:
    """计算相关系数"""
    return np.corrcoef(d_params, d_funcs)[0, 1]

def compute_condition(d_params: np.ndarray, d_funcs: np.ndarray) -> float:
    """计算 ratio 的变异系数 (越小越匹配)"""
    ratios = d_funcs / (d_params + 1e-8)
    return np.std(ratios) / (np.mean(ratios) + 1e-8)

# ============================================================================
# 主实验
# ============================================================================

def main():
    # 生成测试数据
    n_samples = 100
    input_dim = 20
    x = np.random.randn(n_samples, input_dim)

    # 配置各模型
    models = {
        'Linear': {
            'fn': linear_model,
            'param_dim': input_dim,
            'scale': 1.0
        },
        'MLP': {
            'fn': lambda t, x: mlp_model(t, x, hidden_dim=32),
            'param_dim': input_dim * 32 + 32 * 1,
            'scale': 0.5
        },
        'CNN': {
            'fn': lambda t, x: cnn_1d_model(t, x, kernel_size=5),
            'param_dim': 5,
            'scale': 1.0
        },
        'ResNet': {
            'fn': lambda t, x: resnet_block(t, x, hidden_dim=16),
            'param_dim': input_dim * 16 + 16 * input_dim,
            'scale': 0.3
        }
    }

    # 运行实验
    print("="*60)
    print("Distance Matching Verification Experiment")
    print("="*60)
    print()

    results = {}
    for name, config in models.items():
        print(f"Testing {name}...")
        d_p, d_f = experiment(
            config['fn'],
            config['param_dim'],
            x,
            n_pairs=500,
            scale=config['scale']
        )

        corr = compute_correlation(d_p, d_f)
        cv = compute_condition(d_p, d_f)

        results[name] = {
            'd_params': d_p,
            'd_funcs': d_f,
            'correlation': corr,
            'cv': cv
        }

    # 总结
    print()
    print("="*60)
    print("Results Summary")
    print("="*60)
    print()
    print(f"{'Model':<12} {'Correlation':<15} {'CV (lower=better)':<20} {'Match?':<10}")
    print("-"*60)

    for name, res in results.items():
        corr = res['correlation']
        cv = res['cv']

        if corr > 0.95 and cv < 0.2:
            match = "Excellent"
        elif corr > 0.85 and cv < 0.4:
            match = "Good"
        elif corr > 0.7:
            match = "Moderate"
        else:
            match = "Poor"

        print(f"{name:<12} {corr:<15.4f} {cv:<20.4f} {match:<10}")

    print()
    print("="*60)
    print("Interpretation")
    print("="*60)
    print("""
Correlation: How linear is the relationship?
  - 1.0 = perfect linear relationship (param dist ~ func dist)
  - <0.8 = nonlinear or no clear relationship

CV (Coefficient of Variation of ratio):
  - Low (<0.3) = consistent ratio across all pairs (good match)
  - High (>0.5) = ratio varies a lot (poor match, some directions stretched)

Expected results:
  - Linear: Should be excellent (it's literally linear)
  - CNN: Should be good (few params, little redundancy)
  - ResNet: Should be decent (skip helps linearize)
  - MLP: May be worse (more complex, potential redundancy)
""")

if __name__ == "__main__":
    main()
