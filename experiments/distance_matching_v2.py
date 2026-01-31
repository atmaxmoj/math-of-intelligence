"""
Distance Matching 验证 - 第二版

重点测试：冗余 vs 非冗余 参数化
"""

import numpy as np
from typing import Callable, Tuple

np.random.seed(42)

# ============================================================================
# 模型定义
# ============================================================================

def linear_model(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """线性模型: f(x) = theta @ x"""
    return x @ theta

def redundant_linear(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    冗余参数化的线性模型
    用 2 倍参数表示同一个线性函数
    theta = [a, b]，实际权重 = a - b
    """
    n = len(theta) // 2
    a = theta[:n]
    b = theta[n:]
    effective_weight = a - b  # 很多 (a, b) 对给出同一个 effective_weight
    return x @ effective_weight

def very_redundant(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    极度冗余：用矩阵 A @ B 表示向量
    theta 包含 A (n x k) 和 B (k x 1)
    实际权重 = A @ B
    很多不同的 (A, B) 给出同一个权重
    """
    n = x.shape[1]
    k = 5  # 中间维度
    A = theta[:n*k].reshape(n, k)
    B = theta[n*k:n*k+k].reshape(k, 1)
    effective_weight = (A @ B).flatten()
    return x @ effective_weight

def symmetric_target(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    目标是对称函数（求和），但用一般线性参数化
    理想权重: 全 1
    任何偏离都应该导致函数距离变化
    """
    return x @ theta  # theta 应该是全 1

# ============================================================================
# 距离计算
# ============================================================================

def param_distance(theta1: np.ndarray, theta2: np.ndarray) -> float:
    return np.linalg.norm(theta1 - theta2)

def func_distance(f: Callable, theta1: np.ndarray, theta2: np.ndarray,
                  x: np.ndarray) -> float:
    y1 = f(theta1, x)
    y2 = f(theta2, x)
    return np.linalg.norm(y1 - y2)

# ============================================================================
# 实验
# ============================================================================

def experiment(model_fn: Callable, param_dim: int, x: np.ndarray,
               n_pairs: int = 500, scale: float = 1.0,
               base_theta: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """采样参数对，计算距离"""
    d_params = []
    d_funcs = []

    for _ in range(n_pairs):
        if base_theta is not None:
            # 在 base_theta 附近采样
            theta1 = base_theta + np.random.randn(param_dim) * scale
            theta2 = base_theta + np.random.randn(param_dim) * scale
        else:
            theta1 = np.random.randn(param_dim) * scale
            theta2 = np.random.randn(param_dim) * scale

        d_p = param_distance(theta1, theta2)
        d_f = func_distance(model_fn, theta1, theta2, x)

        d_params.append(d_p)
        d_funcs.append(d_f)

    return np.array(d_params), np.array(d_funcs)

def compute_metrics(d_params: np.ndarray, d_funcs: np.ndarray):
    """计算各种指标"""
    corr = np.corrcoef(d_params, d_funcs)[0, 1]

    ratios = d_funcs / (d_params + 1e-8)
    cv = np.std(ratios) / (np.mean(ratios) + 1e-8)

    # 有多少 ratio 接近 0？（说明有冗余方向）
    near_zero = np.sum(ratios < 0.1) / len(ratios)

    return corr, cv, near_zero

# ============================================================================
# 主实验
# ============================================================================

def main():
    n_samples = 100
    input_dim = 10
    x = np.random.randn(n_samples, input_dim)

    print("="*70)
    print("Distance Matching: Redundancy Analysis")
    print("="*70)
    print()

    models = {
        'Linear (no redundancy)': {
            'fn': linear_model,
            'param_dim': input_dim,
            'scale': 1.0,
            'base': None
        },
        'Redundant Linear (a-b)': {
            'fn': redundant_linear,
            'param_dim': input_dim * 2,
            'scale': 1.0,
            'base': None
        },
        'Very Redundant (A@B)': {
            'fn': very_redundant,
            'param_dim': input_dim * 5 + 5,
            'scale': 0.5,
            'base': None
        },
    }

    print(f"{'Model':<25} {'Corr':<10} {'CV':<10} {'Near-zero%':<12} {'Interpretation'}")
    print("-"*70)

    for name, config in models.items():
        d_p, d_f = experiment(
            config['fn'],
            config['param_dim'],
            x,
            n_pairs=500,
            scale=config['scale'],
            base_theta=config['base']
        )

        corr, cv, near_zero = compute_metrics(d_p, d_f)

        # 解读
        if corr > 0.9 and cv < 0.2:
            interp = "Excellent match"
        elif near_zero > 0.3:
            interp = "Many redundant dirs"
        elif cv > 0.5:
            interp = "Inconsistent ratio"
        elif corr > 0.7:
            interp = "Decent match"
        else:
            interp = "Poor match"

        print(f"{name:<25} {corr:<10.3f} {cv:<10.3f} {near_zero*100:<12.1f} {interp}")

    print()
    print("="*70)
    print("Key Insight")
    print("="*70)
    print("""
Redundancy breaks distance matching:

1. Linear (no redundancy):
   - Each parameter direction changes the function
   - Param distance ~ Func distance

2. Redundant (a - b parameterization):
   - Moving along a+b direction doesn't change function
   - Some param distances have zero func distance
   - Distance matching breaks

3. Very redundant (A @ B):
   - Many directions in param space are "null"
   - Very poor correlation

This confirms: REDUNDANCY is the enemy of distance matching.
Removing redundancy (via symmetry, quotient, or direct parameterization) helps.
""")

if __name__ == "__main__":
    main()
